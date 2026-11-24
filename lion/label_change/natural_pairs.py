"""Exact held-out different-label word pairs; fixed Few-NERD scorer, no MLM/training."""
import sys,time
from collections import defaultdict
import torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from forge import run,start_run
from ..diagnostics.common import ROOT,absolute,save_json
from ..runtime import configure
from ..certification import data
from ..encoder import load_masked_encoder
from ..partition.train import load_model
from ..certification.selector import scorer_log_probs
from .fewnerd_diagnostic import parquet


def pair_counts(word,gold,pred,sentence,classes,types):
    def cross(count):return (count.sum(-1).square()-count.square().sum(-1))//2
    n=torch.bincount(word*classes+gold,minlength=types*classes).reshape(types,classes)
    correct=gold==pred;c=torch.bincount((word*classes+gold)[correct],minlength=types*classes).reshape(types,classes)
    eligible=cross(n);both=cross(c)
    # Remove same-sentence pairs: the two contexts must be from distinct sentences.
    keys,group=torch.unique(torch.stack((word,sentence),1),dim=0,return_inverse=True)
    sn=torch.bincount(group*classes+gold,minlength=len(keys)*classes).reshape(-1,classes)
    sc=torch.bincount((group*classes+gold)[correct],minlength=len(keys)*classes).reshape(-1,classes)
    eligible.scatter_add_(0,keys[:,0],-cross(sn));both.scatter_add_(0,keys[:,0],-cross(sc))
    joint=torch.bincount((word*classes+pred)*classes+gold,minlength=types*classes*classes).reshape(types,classes,classes)
    same_prediction=cross(joint).sum(-1)
    sj=torch.bincount((group*classes+pred)*classes+gold,minlength=len(keys)*classes*classes).reshape(len(keys),classes,classes)
    same_prediction.scatter_add_(0,keys[:,0],-cross(sj).sum(-1));supported=eligible>0
    return {'eligible_unordered_pairs':int(eligible.sum()),'both_correct_pairs':int(both.sum()),'pair_accuracy':float(both.sum()/eligible.sum()) if supported.any() else None,'prediction_differs_rate':float(1-same_prediction.sum()/eligible.sum()) if supported.any() else None,
            'ambiguous_word_types':int(supported.sum()),'word_macro_pair_accuracy':float((both[supported].double()/eligible[supported]).mean()) if supported.any() else None}


@torch.no_grad()
def main(cfg):
    configure(cfg);tracking=start_run(cfg);started=time.monotonic();root=absolute(cfg.natural_pairs.results if 'natural_pairs' in cfg else 'results/natural_pairs');root.mkdir(parents=True,exist_ok=True)
    bank=data.load(cfg);scorer=load_model(data.paths(cfg)/'scorer'/'ner',cfg).requires_grad_(False);raw_train=pq.read_table(parquet('train'),columns=['tokens','ner_tags']).to_pylist();seen=defaultdict(set);training_sentences=set()
    for row in bank['sets']['train']:
        if row['fold']!='fit':continue
        native=raw_train[int(row['source_id'].split('/')[-1])];training_sentences.add(tuple(native['tokens']))
        for word,y in zip(native['tokens'][:len(row['focals'])],row['gold']['ner'].tolist()):seen[word].add(y)
    raw_test=pq.read_table(parquet('test'),columns=['tokens','ner_tags']).to_pylist();rows=[];forms=[];duplicates=overlap=0;test_sentences=set();tokenizer=AutoTokenizer.from_pretrained(cfg.encoder.name,use_fast=True,local_files_only=True)
    for offset in range(0,len(raw_test),256):
        chunk=[]
        for i,r in enumerate(raw_test[offset:offset+256],offset):
            key=tuple(r['tokens'])
            if key in training_sentences:overlap+=1;continue
            if key in test_sentences:duplicates+=1;continue
            test_sentences.add(key);chunk.append((i,r))
        encoded=tokenizer([r['tokens'] for _,r in chunk],is_split_into_words=True,add_special_tokens=True,truncation=True,max_length=cfg.encoder.max_length) if chunk else None
        for j,(i,r) in enumerate(chunk):
            wid=encoded.word_ids(j);kept=list(dict.fromkeys(w for w in wid if w is not None));packed={w:s for s,w in enumerate(kept)}
            rows.append({'ids':torch.tensor(encoded['input_ids'][j]),'word_ids':torch.tensor([packed.get(w,-1) for w in wid]),'focals':torch.arange(len(kept)),'gold':{'ner':torch.tensor([r['ner_tags'][w] for w in kept])},'sentence':i});forms.extend(r['tokens'][w] for w in kept)
    print(f'Encoding {len(rows)} distinct native test sentences, {len(forms)} words',flush=True)
    encoder=load_masked_encoder(cfg);pred=[];gold=[];sentences=[]
    for batch in data.loader(rows,cfg):
        from ..runtime import autocast
        with autocast(cfg):hidden=encoder(batch['ids'].to(cfg.runtime.device),batch['attention'].to(cfg.runtime.device))['features']
        vectors=data.pool_words(hidden,batch['word_ids'].to(cfg.runtime.device),int(batch['focal'].max())+1)[batch['owner'].to(cfg.runtime.device),batch['focal'].to(cfg.runtime.device)]
        pred.append(scorer_log_probs(scorer,vectors,cfg).argmax(-1).cpu());gold.append(batch['gold']['ner']);sentences.append(batch['sentence'])
    prediction=torch.cat(pred);y=torch.cat(gold);sentence=torch.cat(sentences);lookup={};word=torch.tensor([lookup.setdefault(w,len(lookup)) for w in forms]);classes=len(bank['labels']['ner']);observed=torch.tensor([int(label) in seen.get(form,set()) for form,label in zip(forms,y.tolist())])
    results={'scorer':f'Fixed Few-NERD coarse scorer, {sum(r["fold"]=="fit" for r in bank["sets"]["train"])} native train sentences; frozen cased BERT block12 + original MLP','split':'native test only; no confidence threshold or generated examples','test_sentences':len(rows),'test_words':len(forms),'excluded_exact_training_sentence_matches':overlap,'excluded_duplicate_test_sentences':duplicates,'overall_test_token_accuracy':float((prediction==y).double().mean()),'pair_definition':'All unordered exact case-sensitive same-word occurrence pairs with distinct gold coarse labels and distinct test sentences; exact aggregate counts, no pair sampling. Pair-weighted metric; word-macro metric also saved.','groups':{}}
    for name,mask in [('all_labels_including_O',torch.ones(len(y),dtype=torch.bool)),('entity_vs_entity',y!=0),('both_labels_seen_in_scorer_training',observed)]:results['groups'][name]=pair_counts(word[mask],y[mask],prediction[mask],sentence[mask],classes,len(lookup))
    if 'natural_pairs' in cfg and cfg.natural_pairs.get('pilot_reference'):
        old=torch.load(absolute(cfg.natural_pairs.pilot_reference),weights_only=False)
        def row_keys(sent):
            pos=torch.arange(len(sent));start=torch.ones(len(sent),dtype=torch.bool);start[1:]=sent[1:]!=sent[:-1];offset=torch.where(start,pos,0).cummax(0).values
            return sent*(cfg.encoder.max_length+1)+pos-offset
        oldkeys=row_keys(old['sentence']);newkeys=row_keys(sentence);index=torch.searchsorted(oldkeys,newkeys);common=index<len(oldkeys);index=index.clamp_max(len(oldkeys)-1);common&=oldkeys[index]==newkeys
        results['same_cohort_comparison']={}
        for label,mask in [('all_labels_including_O',common),('entity_vs_entity',common&(y!=0))]:
            results['same_cohort_comparison'][label]={'test_words':int(mask.sum()),'pilot':pair_counts(word[mask],y[mask],old['prediction'][index[mask]],sentence[mask],classes,len(lookup)),'new':pair_counts(word[mask],y[mask],prediction[mask],sentence[mask],classes,len(lookup))}
    results['seconds']=time.monotonic()-started;save_json(root/'numbers.json',results);torch.save({'prediction':prediction,'gold':y,'sentence':sentence,'forms':forms},root/'predictions.pt');print(results,flush=True);tracking.finish({'test_words':len(forms)});return 0


if __name__=='__main__':run('lion.label_change',sys.argv[1:],main_module='natural_pairs',config_dir=str(ROOT/'conf'),config_name='label_change_fewnerd_diagnostic')
