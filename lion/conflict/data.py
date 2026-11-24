"""Full native data, singleton entity slots and training-only lexical label counts."""
from collections import Counter
import torch
from transformers import AutoTokenizer
from ..coverage import data as coverage
from ..label_share.measure import collapse
from ..sensitivity_diagnostics.metadata import spans
from ..diagnostics.common import absolute,write_csv,save_json


def root(cfg):return absolute(cfg.conflict.results)


def load(cfg,name):return torch.load(root(cfg)/'prepare'/f'{name}.pt',weights_only=False)


def source(cfg,name):return coverage.source(cfg,name)


def prepare(cfg,output):
    records=[];origins=[]
    for name in cfg.coverage.datasets:
        src=source(cfg,name);bank=coverage.full_bank(cfg,name);task=coverage.task(bank);mapping,labels=collapse(bank['labels'][task]);outside=labels.index('O') if name=='conll2003' else -1
        tokenizer=AutoTokenizer.from_pretrained(src.encoder.name,use_fast=True,local_files_only=True);tokens=tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
        # Whole, non-continuation WordPieces made only of Unicode letters/digits; no punctuation/specials.
        valid=torch.tensor([i for i,t in enumerate(tokens) if t.isalnum() and not t.startswith('##') and i not in tokenizer.all_special_ids])
        forms={split:[r['words'] if 'words' in r else f for r,f in zip(rows,coverage.forms(rows,tokens))] for split,rows in bank['sets'].items() if split in ('train','validation','id')}
        counter=Counter((w,int(mapping[y])) for words,row in zip(forms['train'],bank['sets']['train']) for w,y in zip(words,row['gold'][task]) if y>=0)
        by_word={}
        for (word,label),count in counter.items():by_word.setdefault(word,torch.zeros(len(labels),dtype=torch.long))[label]=count
        counts=torch.stack([by_word.get(tokens[i],torch.zeros(len(labels),dtype=torch.long)) for i in valid.tolist()]);total=counts.sum(-1);majority=counts.argmax(-1)
        # Exact ties have no unique majority label and are excluded from both insertion classes.
        unique=(counts==counts.max(-1,keepdim=True).values).sum(-1)==1
        lexicon={'ids':valid,'counts':counts,'total':total,'majority':majority,'unique_majority':unique,'tokens':tokens}
        sets={};slots={}
        for split in ('train','validation','id'):
            rows=bank['sets'][split];sets[split]=rows;arrays={k:[] for k in ('owner','focal','start','end','y','sentence','group','conflict_prone')};kept_forms=[];entities=kept_entities=words_total=0
            for owner,(row,words) in enumerate(zip(rows,forms[split])):
                native=row['gold'][task];gold=torch.where(native>=0,mapping[native.clamp_min(0)],-1);n=len(gold);words_total+=n
                keep=gold>=0
                if name=='conll2003':
                    tags=[bank['labels'][task][int(y)] for y in native];singleton=torch.zeros(n,dtype=torch.bool)
                    for a,b,_ in spans(tags):
                        if b-a==1:singleton[a]=True
                    entity=gold!=outside;entities+=int(entity.sum());kept_entities+=int((singleton&entity).sum());keep&=(~entity|singleton)
                wid=row['word_ids'];p=torch.arange(len(wid));v=wid>=0
                starts=torch.full((n,),len(wid)).scatter_reduce_(0,wid[v],p[v],'amin');ends=torch.zeros(n,dtype=torch.long).scatter_reduce_(0,wid[v],p[v]+1,'amax')
                index=keep.nonzero().flatten();y=gold[index];member=torch.zeros(len(index),dtype=torch.bool)
                for j,focal in enumerate(index.tolist()):
                    count=by_word.get(words[focal]);member[j]=count is not None and (int((count>0).sum())>1 or int(count[gold[focal]])==0)
                for k,value in [('owner',torch.full_like(index,owner)),('focal',index),('start',starts[index]),('end',ends[index]),('y',y),('sentence',torch.full_like(index,row['sentence'])),('group',(y!=outside).long() if outside>=0 else torch.zeros_like(y)),('conflict_prone',member)]:arrays[k].append(value)
                kept_forms.extend(words[i] for i in index.tolist())
            slots[split]={k:torch.cat(v) for k,v in arrays.items()};slots[split]['forms']=kept_forms
            records.append({'dataset':name,'split':split,'sentences':len(rows),'all_words':words_total,'slots':len(kept_forms),'entity_words':entities,'retained_entity_words':kept_entities,'retained_entity_share':kept_entities/entities if entities else None})
        docs=sorted({r['document'] for r in sets['train']});order=torch.randperm(len(docs),generator=torch.Generator().manual_seed(cfg.conflict.context_seed));held={docs[i] for i in order[:max(1,round(len(docs)*cfg.conflict.internal_validation_fraction))].tolist()}
        slots['train']['holdout']=torch.tensor([sets['train'][i]['document'] in held for i in slots['train']['owner'].tolist()])
        torch.save({'sets':sets,'slots':slots,'labels':labels,'native_labels':bank['labels'][task],'mapping':mapping,'lexicon':lexicon,'mask_id':tokenizer.mask_token_id,'groups':['O','entity'] if outside>=0 else ['all']},output/f'{name}.pt')
        origins.append({'dataset':name,'source_config':str(absolute(cfg.coverage.datasets[name].source_config)),'full_bank':str(cfg.coverage.datasets[name].full_bank),'native_labels':bank['labels'][task],'labels':labels,
                        'context_fit_slots':int((~slots['train']['holdout']).sum()),'internal_checkpoint_slots':int(slots['train']['holdout'].sum()),'majority_tie_words_excluded':int(((total>=cfg.conflict.minimum_occurrences)&~unique).sum()),'layer_heads':str(absolute(cfg.conflict.layer_run)/'train'/name)})
        print(name, {s:len(v['y']) for s,v in slots.items()},flush=True)
    write_csv(output/'slot_counts.csv',records);save_json(output/'provenance.json',origins)
    return {'datasets':len(origins)}
