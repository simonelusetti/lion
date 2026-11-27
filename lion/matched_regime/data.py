"""Reuse original pools; source-correct anchors and word-aligned cached outputs."""
import json,shutil
from collections import defaultdict
import torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from ..certification import data as source,selector
from ..diagnostics.common import absolute,save_json,read_csv
from ..label_change.fewnerd_diagnostic import parquet
from ..independent_judge import representations
from ..partition.train import load_model
from ..guided_refinement import generate


def root(cfg):return absolute(cfg.matched.results)
def load(path):return torch.load(path,mmap=True,weights_only=False)


def originals(cfg):
    bank=source.load(cfg);chosen=set(json.loads((root(cfg)/'prepare'/'subset.json').read_text())['source_ids']);rows=[r for r in bank['sets']['train'] if r['source_id'] in chosen]
    raw=load(root(cfg)/'prepare'/'raw.pt');return bank,source.attach_features(rows,raw['features'][:,0])


def anchors(cfg):
    bank,rows=originals(cfg);indices=load(root(cfg)/'prepare'/'anchor_original_indices.pt');length=torch.tensor([len(r['focals']) for r in rows]);ends=length.cumsum(0);starts=ends-length;owner=torch.searchsorted(ends,indices,right=True);result=[]
    for index,o in zip(indices.tolist(),owner.tolist()):
        f=torch.tensor([index-int(starts[o])]);r=rows[o];result.append({**r,'focals':f,'gold':{'ner':r['gold']['ner'][f]}})
    return bank,result


@torch.no_grad()
def prepare(cfg,output):
    original=absolute(cfg.matched.originals_run)/'prepare'
    for name in ('raw.pt','subset.json','sentences.csv'):shutil.copyfile(original/name,output/name)
    bank,rows=originals(cfg);raw=load(output/'raw.pt');a=load_model(source.paths(cfg)/'scorer'/'ner',cfg).requires_grad_(False)
    prediction=torch.cat([selector.scorer_log_probs(a,x.to(cfg.runtime.device),cfg).argmax(-1).cpu() for x in raw['features'][:,0].split(1024)])
    torch.save(prediction,output/'source_predictions.pt')
    eligible=(prediction==raw['y']).nonzero().flatten();draw=torch.randint(len(eligible),(cfg.guided_training.anchors,),generator=torch.Generator().manual_seed(cfg.matched.source_seed));indices=eligible[draw];torch.save(indices,output/'anchor_original_indices.pt')
    _,sample=anchors(cfg);torch.save(generate.target_metadata(cfg,bank,sample),output/'sources_targets.pt')
    # Majority lookup is case-sensitive, from originals only, ties in saved label order.
    native=pq.read_table(parquet('train'),columns=['tokens']).to_pylist();counts=defaultdict(lambda:torch.zeros(len(bank['labels']['ner']),dtype=torch.long))
    for r in rows:
        tokens=native[int(r['source_id'].split('/')[-1])]['tokens']
        for word,label in zip(tokens,r['gold']['ner'].tolist()):counts[word][label]+=1
    torch.save({'majority':{w:int(c.argmax()) for w,c in counts.items()},'counts':dict(counts),'labels':bank['labels']['ner']},output/'majorities.pt')
    cal=load(absolute(cfg.matched.judge_run)/'prepare'/'calibration.pt');torch.save([native[int(r['source_id'].split('/')[-1])]['tokens'][int(r['focals'][0])] for r in cal],output/'calibration_words.pt')
    save_json(output/'regime.json',{'original_sentences':len(rows),'original_tokens':len(raw['y']),'eligible_source_correct_tokens':len(eligible),'anchors':len(indices),'attempts_per_design':len(indices)*cfg.mechanisms.draws,'majority_pool':'Exactly the 2400 originals, not the 105414-sentence A/B fitting pool. Case-sensitive; smallest saved-label index wins ties.','source_correctness':'Applied before anchor sampling/generation, including ALL. Uniform with replacement among eligible original word occurrences. Shared 40000 sampled anchors across designs, five attempts each.','calibration':'Unmodified independent_judge calibration focal identities, A probabilities and B probabilities reused. Never student training/dev.','dependencies':'Five draws share anchors and many anchors share original source sentences. Report sentence counts; token-count CP mass intervals are nominal for generator populations unless the requisite independence is additionally assumed. No unqualified 95% guarantee is asserted.'})
    return {'originals':len(rows),'eligible_tokens':len(eligible)}


def pack(records,features):
    lengths=torch.tensor([len(r['ids']) for r in records]);offsets=torch.cat((torch.zeros(1,dtype=torch.long),lengths.cumsum(0)))
    return {'features':features,'y':torch.tensor([r['y'] for r in records]),'a_prediction':torch.tensor([r.get('a_prediction',r['y']) for r in records]),'a_confidence':torch.tensor([r.get('confidence',1.) for r in records]),
        'source_gold':torch.tensor([r['gold'] for r in records]),'source_id':[r['source_id'] for r in records],'word':[r['word'] for r in records],'focal':torch.tensor([r['focal'] for r in records]),'draw':torch.tensor([r['draw'] for r in records]),'anchor':torch.tensor([r['anchor'] for r in records]),
        'input_ids':torch.cat([r['ids'].int() for r in records]),'word_ids':torch.cat([r['word_ids'].int() for r in records]),'offsets':offsets,'changed':torch.tensor([r.get('changed',True) for r in records])}


def historical_pools(cfg,output):
    old=absolute(cfg.matched.historical_run);meta=load(old/'prepare'/'sources_targets.pt');bank,rows=originals(cfg);lookup={r['source_id']:r for r in rows};expected={r['arm']:int(r['accepted']) for r in read_csv(old/'generate'/'yield.csv')};audit=[]
    for arm,total in expected.items():
        records=[];features=[]
        for draw in range(5):
            folder=old/'generate'/arm;saved=load(folder/f'accepted_features_draw{draw}.pt');features.append(saved['features']);indices=saved['anchor_id'].tolist()
            raw=load(folder/f'replicate{draw}.pt') if (folder/f'replicate{draw}.pt').exists() else None
            offsets=torch.cat((torch.zeros(1,dtype=torch.long),saved['lengths'].cumsum(0))) if 'input_ids' in saved else None
            for j,index in enumerate(indices):
                if offsets is not None:
                    lo,hi=offsets[j:j+2].tolist();ids=saved['input_ids'][lo:hi];confidence=float(saved['confidence'][j])
                else:
                    lo,hi=raw['offsets'][index:index+2].tolist();ids=raw['state_input_ids'][-1,lo:hi];confidence=float(raw['state_probabilities'][index,-1].max())
                source_id=meta['source_id'][index];records.append({'ids':ids,'word_ids':lookup[source_id]['word_ids'],'y':int(saved['y'][j]),'confidence':confidence,'gold':int(meta['gold'][index]),'source_id':source_id,'word':meta['forms'][index],'focal':int(meta['focal'][index]),'draw':draw,'anchor':index})
        pool=pack(records,torch.cat(features));torch.save(pool,output/f'{arm}.pt');audit.append({'arm':arm,'recorded':total,'recovered':len(records),'matches':total==len(records),'regenerated':False,'reason':'All accepted token IDs were present: full trajectories for refinement, packed accepted IDs for fractions.'})
        print(f'Reused {arm}: {len(records)} accepted inputs',flush=True)
    save_json(output/'reuse.json',audit);return {'arms':len(audit),'mismatches':sum(not r['matches'] for r in audit)}


def collect_new(cfg,design,output):
    _,rows=anchors(cfg);meta=load(root(cfg)/'prepare'/'sources_targets.pt');records=[];features=[];folder=root(cfg)/'new_generate'/design
    for draw in range(cfg.mechanisms.draws):
        saved=load(folder/f'all_features_draw{draw}.pt');features.append(saved['features'])
        raw=load(folder/f'draw{draw}.pt') if design=='local70' else load(folder/f'replicate{draw}.pt')
        final=raw['input_ids'] if design=='local70' else raw['state_input_ids'][-1];confidence=raw['confidence'] if design=='local70' else raw['state_probabilities'][:,-1].max(-1).values
        for i,row in enumerate(rows):
            lo,hi=raw['offsets'][i:i+2].tolist();ids=final[lo:hi];records.append({'ids':ids,'word_ids':row['word_ids'],'y':int(saved['y'][i]),'confidence':float(confidence[i]),'gold':int(meta['gold'][i]),'source_id':meta['source_id'][i],'word':meta['forms'][i],'focal':int(meta['focal'][i]),'draw':draw,'anchor':i,'changed':not torch.equal(ids.long(),row['ids'])})
    pool=pack(records,torch.cat(features));torch.save(pool,output/f'{design}.pt');return {'design':design,'outputs':len(records)}


def judge(cfg,pool,indices,encoder,tokenizer,b):
    tokenizer_A=AutoTokenizer.from_pretrained(cfg.encoder.name,local_files_only=True);records=[]
    for index in indices.tolist():
        lo,hi=pool['offsets'][index:index+2].tolist();ids=pool['input_ids'][lo:hi];wid=pool['word_ids'][lo:hi];n=int(wid.max())+1
        words=[tokenizer_A.convert_tokens_to_string(tokenizer_A.convert_ids_to_tokens(ids[wid==w].tolist())) for w in range(n)];records.append({'words':words,'focals':pool['focal'][index:index+1]})
    if not records:return torch.empty(0,dtype=torch.long)
    features,_=representations.encode(cfg,encoder,tokenizer,records)
    return torch.cat([selector.scorer_log_probs(b,x.to(cfg.runtime.device),cfg).argmax(-1).cpu() for x in features.split(1024)])
