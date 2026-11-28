"""Pool counts and MLM leave-one-WordPiece-out pseudo-log-likelihood."""
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from ..diagnostics.common import absolute,write_csv,save_json
from ..encoder import load_masked_encoder
from ..runtime import autocast
from .generate import root,load,collate


def pools(cfg):
    result={a:load(root(cfg)/'generate'/f'{a}.pt') for a in cfg.paradigm_swaps.arms};old=load(absolute(cfg.matched.results)/'new_pools/MENTION.pt');originals=load(root(cfg)/'prepare/originals.pt');tokenizer=AutoTokenizer.from_pretrained(cfg.encoder.name,local_files_only=True);records=[];items=[]
    for i in range(len(old['y'])):
        lo,hi=old['offsets'][i:i+2].tolist();ids=old['input_ids'][lo:hi].long();wid=old['word_ids'][lo:hi];words=[tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids[wid==j].tolist())) for j in range(int(wid.max())+1)];row=originals[old['source_id'][i]];f=int(old['focal'][i]);words[f]=old['word'][i];records.append({'ids':ids,'word_ids':wid,'words':words,'source_id':old['source_id'][i],'original_ids':row['ids'],'original_words':row['words'],'start':f,'end':f+1,'label':int(old['y'][i])});items.append((i,f))
    result['MENTION']={'records':records,'items':items};return result


@torch.no_grad()
def stats(cfg,out):
    ps=pools(cfg);lex=load(root(cfg)/'prepare/lexicon.pt');tokenizer=AutoTokenizer.from_pretrained(cfg.encoder.name,local_files_only=True);encoder=load_masked_encoder(cfg);sequence={};references={}
    for arm,pool in ps.items():
        for i,f in pool['items']:
            r=pool['records'][i]
            for kind,key in [('augmented','ids'),('original','original_ids')]:
                ids=r[key].long();identity=tuple(ids.tolist());index=sequence.setdefault(identity,len(sequence));references[arm,i,kind]=index
    ids_list=[torch.tensor(k) for k in sequence];sums=torch.zeros(len(sequence),dtype=torch.double);counts=torch.zeros(len(sequence),dtype=torch.long);tasks=[]
    for index,ids in enumerate(ids_list):
        for position in range(len(ids)):
            if int(ids[position]) not in tokenizer.all_special_ids:tasks.append((index,position))
    class Masked(torch.utils.data.Dataset):
        def __len__(self):return len(tasks)
        def __getitem__(self,i):
            index,pos=tasks[i];ids=ids_list[index].clone();gold=int(ids[pos]);ids[pos]=tokenizer.mask_token_id;return {'ids':ids,'position':pos,'gold':gold,'sequence':index}
    print(f'PLL: {len(sequence)} unique sentences, {len(tasks)} masked pieces',flush=True)
    for batch_no,(ids,attention,rows) in enumerate(DataLoader(Masked(),batch_size=cfg.paradigm_swaps.pll_batch_size,collate_fn=collate,num_workers=cfg.runtime.workers)):
        positions=torch.tensor([r['position'] for r in rows],device=cfg.runtime.device);target=torch.tensor([r['gold'] for r in rows]);owner=torch.tensor([r['sequence'] for r in rows])
        with autocast(cfg):logits=encoder(ids.to(cfg.runtime.device),attention.to(cfg.runtime.device),positions,mlm=True)['logits'].float().cpu()
        ll=logits.log_softmax(-1).gather(1,target[:,None])[:,0].double();sums.scatter_add_(0,owner,ll);counts.scatter_add_(0,owner,torch.ones_like(owner))
        if (batch_no+1)%500==0:print(f'PLL batches {batch_no+1}/{(len(tasks)+cfg.paradigm_swaps.pll_batch_size-1)//cfg.paradigm_swaps.pll_batch_size}',flush=True)
    torch.save({'sums':sums,'pieces':counts,'sequences':ids_list},out/'pll.pt');tables=[];labels=[];examples=[];rows=[];rng=torch.Generator().manual_seed(cfg.paradigm_swaps.seed)
    for arm,pool in ps.items():
        words=[pool['records'][i]['words'][f] for i,f in pool['items']];y=torch.tensor([pool['records'][i]['label'] for i,f in pool['items']]);known=torch.tensor([lex['lookup'].get(w,-1) for w in words]);seen=(known>=0)&(lex['counts'][known.clamp_min(0)].sum(-1)>0);risk=seen&(lex['counts'][known.clamp_min(0)].argmax(-1)!=y);new={w for w,s in zip(words,seen.tolist()) if not s};aug=[];orig=[]
        for i,f in pool['items']:
            ai=references[arm,i,'augmented'];oi=references[arm,i,'original'];aug.append(float(sums[ai]/counts[ai]));orig.append(float(sums[oi]/counts[oi]));rows.append({'arm':arm,'record':i,'focal':f,'word':pool['records'][i]['words'][f],'label':lex['labels'][int(pool['records'][i]['label'])],'augmented_mean_piece_PLL':aug[-1],'original_mean_piece_PLL':orig[-1],'status':'computed; MENTION pool reused'})
        a=torch.tensor(aug,dtype=torch.double);o=torch.tensor(orig,dtype=torch.double);tables.append({'arm':arm,'labelled_augmented_tokens':len(words),'new_word_types':len(new),'new_word_occurrences':int((~seen).sum()),'label_risk_occurrences':int(risk.sum()),'label_risk_share':float(risk.double().mean()),'mean_augmented_piece_PLL':float(a.mean()),'mean_original_piece_PLL':float(o.mean()),'paired_PLL_change':float((a-o).mean()),'status':'computed on selected training pool; MENTION inputs reused'})
        for k,name in enumerate(lex['labels']):labels.append({'arm':arm,'label':name,'count':int((y==k).sum()),'share':float((y==k).double().mean()),'status':'computed'})
        for index in torch.randperm(len(pool['items']),generator=rng)[:cfg.paradigm_swaps.example_count].tolist():
            i,f=pool['items'][index];r=pool['records'][i];examples.append({'arm':arm,'source_id':r['source_id'],'focal':f,'assigned_label':lex['labels'][r['label']],'original':' '.join(r['original_words']),'augmented':' '.join(r['words']),'filled_word':r['words'][f],'status':'computed from saved pools; example selection seeded'})
    write_csv(out/'pool_statistics.csv',tables);write_csv(out/'label_distribution.csv',labels);write_csv(out/'PLL_per_item.csv',rows);write_csv(out/'examples.csv',examples);save_json(out/'PLL_method.json',{'definition':'For each non-special input WordPiece, mask that piece only and record log p(original_piece | all_other_pieces), full vocabulary. Sum and divide by ordinary piece count. Pool statistic averages these mean-per-piece PLLs over labelled augmented focal items, paired with their source sentences. Repeated sequences are scored once and retain pool-item multiplicity in averages. This is a descriptive MLM naturalness proxy, not a scorer/filter or label-validity certificate.','unique_sequences':len(sequence),'masked_piece_forwards':len(tasks)});return {'pools':len(ps),'unique_sentences':len(sequence)}
