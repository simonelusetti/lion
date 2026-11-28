"""Whole-word sequential MLM swaps; training labels come only from source gold."""
import json,shutil,time
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from ..diagnostics.common import absolute,read_csv,write_csv,save_json
from ..matched_regime import data as matched
from ..certification import data as source
from ..encoder import load_masked_encoder
from ..runtime import autocast


def root(cfg):return absolute(cfg.paradigm_swaps.results)
def load(p):return torch.load(p,mmap=True,weights_only=False)


def prepare(cfg,out):
    old=absolute(cfg.matched.results)
    for name in ('raw.pt','subset.json','sentences.csv'):shutil.copyfile(old/'prepare'/name,out/name)
    bank,rows=matched.originals(cfg);ref=load(absolute('results/language_checks/prepare/reference.pt'));forms=ref['forms'];offset=0;lookup={}
    for row in rows:
        n=len(row['focals']);lookup[row['source_id']]={k:v for k,v in row.items() if k!='hidden'}|{'words':forms[offset:offset+n]};offset+=n
    torch.save(lookup,out/'originals.pt');torch.save({'counts':ref['counts'],'lookup':ref['lookup'],'mapping':ref['mapping'],'labels':ref['labels']},out/'lexicon.pt')
    # Recover individual mention occurrences in original generation order, including repeats.
    pool=load(old/'new_pools/MENTION.pt');ordered=pool['anchor'].argsort().tolist();groups=[]
    for i in ordered:
        lo,hi=pool['offsets'][i:i+2].tolist();ids=pool['input_ids'][lo:hi];sid=pool['source_id'][i];f=int(pool['focal'][i])
        if groups and groups[-1]['source_id']==sid and groups[-1]['end']==f and torch.equal(groups[-1]['ids'],ids):groups[-1]['end']=f+1
        else:groups.append({'source_id':sid,'start':f,'end':f+1,'ids':ids.clone(),'word_ids':pool['word_ids'][lo:hi].clone()})
    anchors=[]
    for g in groups:
        row=lookup[g['source_id']];start=g['start'];gold=row['gold']['ner'];label=int(gold[start]);end=start+1
        while end<len(gold) and int(gold[end])==label:end+=1
        anchors.append({'source_id':g['source_id'],'start':start,'end':end,'label':label})
    expected=json.loads((old/'new_pools/mention_method.json').read_text())['constructed_mentions']
    if len(anchors)!=expected:raise ValueError(f'MENTION anchor recovery {len(anchors)} != {expected}')
    torch.save(anchors,out/'mention_anchors.pt');save_json(out/'method.json',{'originals':len(rows),'original_labelled_tokens':offset,'mention_anchors_recovered':len(anchors),'MENTION':'Reused unchanged. Its historical source-correct anchor eligibility is inherited by B/C because the exact same mentions are requested; no new scorer calls or source filters. D/E have no such eligibility.','fill':'New minimal left-to-right sequential fill: collapse each replaced word to one [MASK] slot; draw one non-continuation, non-special WordPiece per slot from the complete restricted MLM softmax. Earlier paradigm whole-word vocabulary convention retained, including punctuation. Previous pipeline fill was parallel, not sequential. No top-k/p, temperature 1. Later slots remain masked, earlier fills visible.','labels':'Original source coarse IO entity type or focal gold label; new span BIO boundaries are derived from the unchanged coarse-label convention at evaluation. Only selected replaced-token features receive augmentation loss.','span_anchors':'Replay all recovered MENTION source mention occurrences in generation order. If 1932 source-span tokens require additional mentions, sample with replacement from this same anchor list; each strict rejection advances to the next anchor. Overshoot sampled down uniformly with fixed seed.','token_anchors':'Uniform with replacement over 60321 original word occurrences; D/E share every anchor. PARADIGM_TOKEN rejects identity and draws a fresh uniform anchor; the final accepted anchor set is shared with LWTR, which redraws identity donors at those anchors. LWTR samples original token occurrences of the gold label uniformly, preserving donor frequency.','seeds':'One fixed generation pool reused for every training seed; no scorer/pseudo-label/agreement filter.'})
    return {'anchors':len(anchors)}


def collate(rows):
    n=max(len(r['ids']) for r in rows);ids=torch.zeros(len(rows),n,dtype=torch.long);attention=torch.zeros_like(ids,dtype=torch.bool)
    for i,r in enumerate(rows):ids[i,:len(r['ids'])]=r['ids'];attention[i,:len(r['ids'])]=True
    return ids,attention,rows


def template(row,start,end):
    original=row['ids'].long();wid=row['word_ids'];left=(wid==start).nonzero()[0,0];right=(wid==end-1).nonzero()[-1,0]+1;n=end-start
    return torch.cat((original[:left],torch.full((n,),103),original[right:])),int(left)


@torch.no_grad()
def sequential(cfg,encoder,templates,allowed,rng):
    results=[]
    for ids,attention,rows in DataLoader(templates,batch_size=cfg.paradigm_swaps.generation_batch_size,collate_fn=collate,num_workers=cfg.runtime.workers):
        ids=ids.to(cfg.runtime.device);attention=attention.to(cfg.runtime.device);length=torch.tensor([r['slots'] for r in rows]);starts=torch.tensor([r['mask_start'] for r in rows],device=ids.device)
        for j in range(int(length.max())):
            active=(length>j).nonzero().flatten().to(ids.device);positions=starts[active]+j
            with autocast(cfg):logits=encoder(ids[active],attention[active],positions,mlm=True)['logits'].float()
            logits[:,~allowed]=-torch.inf;draw=torch.multinomial(logits.softmax(-1),1,generator=rng)[:,0];ids[active,positions]=draw
        for i,r in enumerate(rows):results.append(ids[i,int(r['mask_start']):int(r['mask_start'])+r['slots']].cpu())
    return results


def encode_words(tokenizer,row,start,end,replacement):
    words=row['words'][:start]+replacement+row['words'][end:];encoded=tokenizer(words,is_split_into_words=True,add_special_tokens=True,truncation=True,max_length=tokenizer.model_max_length);wid=encoded.word_ids();kept=max(w for w in wid if w is not None)+1
    return {'ids':torch.tensor(encoded['input_ids']),'word_ids':torch.tensor([-1 if w is None else w for w in wid]),'words':words[:kept],'source_id':row['source_id'],'start':start,'end':start+len(replacement),'label':int(row['gold']['ner'][start]),'original_start':start,'original_end':end,'original_words':row['words'],'original_ids':row['ids'].clone()}


@torch.no_grad()
def generate(cfg,out):
    originals=load(root(cfg)/'prepare/originals.pt');lex=load(root(cfg)/'prepare/lexicon.pt');anchors=load(root(cfg)/'prepare/mention_anchors.pt');tokenizer=AutoTokenizer.from_pretrained(cfg.encoder.name,use_fast=True,local_files_only=True);encoder=load_masked_encoder(cfg);vocab=tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))));roundtrip=tokenizer(vocab,add_special_tokens=False)['input_ids'];allowed=torch.tensor([not t.startswith('##') and i not in tokenizer.all_special_ids and roundtrip[i]==[i] for i,t in enumerate(vocab)],device=cfg.runtime.device);N=cfg.paradigm_swaps.count;rng=torch.Generator(device=cfg.runtime.device).manual_seed(cfg.paradigm_swaps.seed);cpu=torch.Generator().manual_seed(cfg.paradigm_swaps.seed);all_pools={};generation=[]
    for arm in list(cfg.paradigm_swaps.arms)[:2]:
        records=[];total=0;attempts=identities=risk_rejected=0;anchor_cursor=0
        while total<N or anchor_cursor<len(anchors):
            batch=[]
            for _ in range(min(cfg.paradigm_swaps.generation_batch_size,len(anchors)-anchor_cursor if anchor_cursor<len(anchors) else N-total)):
                a=anchors[anchor_cursor] if anchor_cursor<len(anchors) else anchors[int(torch.randint(len(anchors),(1,),generator=cpu))];anchor_cursor+=1;row=originals[a['source_id']];ids,start=template(row,a['start'],a['end']);batch.append({'ids':ids,'mask_start':start,'slots':a['end']-a['start'],'anchor':a})
            pending=batch
            while pending:
                fills=sequential(cfg,encoder,pending,allowed,rng);retry=[]
                for t,fill in zip(pending,fills):
                    attempts+=1;a=t['anchor'];row=originals[a['source_id']];replacement=tokenizer.convert_ids_to_tokens(fill.tolist())
                    if replacement==row['words'][a['start']:a['end']]:identities+=1;continue
                    risk=any(w in lex['lookup'] and int(lex['counts'][lex['lookup'][w]].sum())>0 and int(lex['counts'][lex['lookup'][w]].argmax())!=a['label'] for w in replacement)
                    if arm.endswith('STRICT') and risk:risk_rejected+=1;continue
                    r=encode_words(tokenizer,row,a['start'],a['end'],replacement)
                    if r['end']>len(r['words']):continue
                    records.append(r);total+=len(replacement)
                pending=retry
            print(f'{arm}: {total} replaced words, {attempts} draws',flush=True)
        torch.save({'records':records},out/f'{arm}_sentences.pt');all_pools[arm]=records;generation.append({'arm':arm,'constructed_sentences':len(records),'labelled_tokens_before_selection':total,'fill_draws':attempts,'identity_rejections':identities,'lexicon_rejections':risk_rejected,'anchors_drawn':anchor_cursor})
    # Shared occurrence-uniform token anchors; no source prediction is consulted.
    sids=list(originals);lengths=torch.tensor([len(originals[s]['words']) for s in sids]);ends=lengths.cumsum(0);starts=ends-lengths;flat=torch.randint(int(ends[-1]),(N,),generator=cpu);owner=torch.searchsorted(ends,flat,right=True);token_anchors=[{'source_id':sids[o],'start':int(i-starts[o]),'end':int(i-starts[o])+1} for i,o in zip(flat,owner.tolist())];torch.save(token_anchors,out/'token_anchors.pt');templates=[]
    for a in token_anchors:
        row=originals[a['source_id']];ids,start=template(row,a['start'],a['end']);templates.append({'ids':ids,'mask_start':start,'slots':1,'anchor':a})
    pending=list(enumerate(templates));records={};attempts=identities=0
    while pending:
        fills=sequential(cfg,encoder,[t for _,t in pending],allowed,rng);retry=[]
        for (index,t),fill in zip(pending,fills):
            attempts+=1;a=t['anchor'];row=originals[a['source_id']];replacement=tokenizer.convert_ids_to_tokens(fill.tolist())
            if replacement==row['words'][a['start']:a['end']]:
                identities+=1;flat_new=int(torch.randint(int(ends[-1]),(1,),generator=cpu));o=int(torch.searchsorted(ends,torch.tensor(flat_new),right=True));f=flat_new-int(starts[o]);a={'source_id':sids[o],'start':f,'end':f+1};token_anchors[index]=a;fresh=originals[a['source_id']];ids,start=template(fresh,f,f+1);retry.append((index,{'ids':ids,'mask_start':start,'slots':1,'anchor':a}));continue
            records[index]=encode_words(tokenizer,row,a['start'],a['end'],replacement)
        pending=retry
        print(f'PARADIGM_TOKEN: {len(records)}/{N} accepted, {attempts} draws, {len(pending)} fresh anchors pending',flush=True)
    torch.save(token_anchors,out/'token_anchors.pt')
    all_pools['PARADIGM_TOKEN']=[records[i] for i in range(N)];generation.append({'arm':'PARADIGM_TOKEN','constructed_sentences':N,'labelled_tokens_before_selection':N,'fill_draws':attempts,'identity_rejections':identities,'lexicon_rejections':0,'anchors_drawn':attempts})
    donors=defaultdict(list)
    for row in originals.values():
        for w,y in zip(row['words'],row['gold']['ner'].tolist()):donors[y].append(w)
    records=[];attempts=identities=0
    for a in token_anchors:
        row=originals[a['source_id']];options=donors[int(row['gold']['ner'][a['start']])]
        while True:
            attempts+=1;word=options[int(torch.randint(len(options),(1,),generator=cpu))]
            if word!=row['words'][a['start']]:break
            identities+=1
        records.append(encode_words(tokenizer,row,a['start'],a['end'],[word]))
    all_pools['LWTR_TOKEN']=records;generation.append({'arm':'LWTR_TOKEN','constructed_sentences':N,'labelled_tokens_before_selection':N,'fill_draws':attempts,'identity_rejections':identities,'lexicon_rejections':0,'anchors_drawn':N})
    for arm,records in all_pools.items():
        items=[(i,f) for i,r in enumerate(records) for f in range(r['start'],r['end'])];select=torch.randperm(len(items),generator=torch.Generator().manual_seed(cfg.paradigm_swaps.seed))[:N];items=[items[i] for i in select.tolist()];torch.save({'records':records,'items':items},out/f'{arm}.pt')
        # Encode only the focal positions that actually receive a training label.
        requested=defaultdict(list)
        for item,(i,f) in enumerate(items):requested[i].append((item,f))
        feature_rows=[];selected=torch.empty(N,dtype=torch.long);offset=0
        for i,pairs in requested.items():
            r=records[i];n=len(r['words']);feature_rows.append({**r,'focals':torch.arange(n),'gold':{'ner':torch.full((n,),-1)},'sentence':i})
            for item,f in pairs:selected[item]=offset+f
            offset+=n
        features=source.full_features(encoder,{arm:feature_rows},cfg)[arm][selected];labels=torch.tensor([records[i]['label'] for i,f in items]);folder=root(cfg)/'generate'/arm;folder.mkdir(exist_ok=True);torch.save({'features':features,'y':labels},folder/'accepted_features_draw0.pt')
    write_csv(out/'generation_counts.csv',generation);return {'arms':len(all_pools),'count_per_arm':N}
