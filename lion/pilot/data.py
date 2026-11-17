"""Paired genre/document samples, gold-tree slots and content-blind masks."""
import json
from collections import defaultdict
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
from ..diagnostics.common import absolute,dataset_config,write_csv
from ..diagnostics.data import token_path,metadata
from ..partition.data import ud_raw,distances
from ..invariance.data import paths
from ..data import signature
from omegaconf import OmegaConf

FIXED=['full','word','lin1','lin2','tree1','tree2','rand_tree1','full_prev']
SLOTS=['left1','right1','left2','right2','head','dependents','siblings','grandparent']
SOURCES={'ud_upos':'ewt','ud_deprel':'ewt','gum_upos':'gum','ud_discourse':'gum'}


def cache(cfg):
    key=signature(OmegaConf.create({'protocol':'slot-pilot-v1','encoder':cfg.encoder,'amp':cfg.runtime.amp,'tf32':cfg.runtime.tf32,
                                   'sample':[cfg.pilot.train_per_genre,cfg.pilot.validation_per_genre,cfg.pilot.test_per_genre,cfg.pilot.positions_per_sentence,cfg.diagnostics.sample_seed],
                                   'random':[cfg.pilot.random_masks,cfg.pilot.scorer_seed]}))
    return absolute(cfg.pilot.cache)/key


def load(cfg,task):
    bank=torch.load(cache(cfg)/(SOURCES[task]+'.pt'),map_location='cpu',weights_only=False)
    bank['y']=bank['ys'][task];bank['labels']=bank['label_names'][task]
    return bank


def rotation(bank,genre):
    test=bank['genres'].index(genre);val=(test+1)%len(bank['genres']);genres=[g for g in range(len(bank['genres'])) if g not in (test,val)]
    train=torch.isin(bank['z'],torch.tensor(genres))
    return genres,{'train':(train&(bank['native']==0)).nonzero().flatten(),'validation':(bank['z']==val).nonzero().flatten(),
                   'id_validation':(train&(bank['native']==1)).nonzero().flatten(),'id':(train&(bank['native']==2)).nonzero().flatten(),
                   'ood':(bank['z']==test).nonzero().flatten()}


def shape(mask,slots,focal,previous_words=0):
    nonempty=slots.any(-1)
    present=((mask[:,None,:]|~slots[None]).all(-1)&nonempty[None]).float()
    return torch.cat([torch.tensor([[mask.shape[-1]+previous_words,focal+previous_words]],dtype=torch.float32).expand(len(mask),-1),
                      (mask.sum(-1)+previous_words).float()[:,None],present],dim=1)


def masks(row,kind,menus=None):
    width=len(row['words']);focals=row['focals'];result=[];shapes=[]
    for j,f in enumerate(focals):
        if kind=='menu':
            ids=torch.as_tensor(menus[j] if torch.as_tensor(menus).ndim==2 else menus)
            bits=((ids[:,None]>>torch.arange(8))&1).bool()
            keep=(bits[:,:,None]&row['slots'][j][None]).any(1)
        else:
            kept=[]
            for rule in kind:
                if rule in ('full','full_prev'):m=torch.ones(width,dtype=torch.bool)
                elif rule=='word':m=torch.zeros(width,dtype=torch.bool)
                elif rule.startswith('lin'):m=(torch.arange(width)-f).abs()<=int(rule[-1])
                elif rule.startswith('tree'):m=row['tree'][j]<=int(rule[-1])
                else:
                    number=int((row['tree'][j]<=1).sum())-1;generator=torch.Generator().manual_seed(row['global_sentence']*4099+f+1729)
                    score=torch.rand(width,generator=generator);score[f]=2;m=score.argsort().argsort()<number
                m[f]=True;kept.append(m)
            keep=torch.stack(kept)
        keep[:,f]=True;result.append(keep);shapes.append(shape(keep,row['slots'][j],f))
    return torch.stack(result),torch.stack(shapes)


def prepare(cfg,output):
    original=json.loads(absolute('results/invariance/banks.json').read_text());folder=cache(cfg);folder.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(cfg.encoder.name,local_files_only=True,use_fast=True);counts=[]
    for source,tasks,token_task,treebank in [('ewt',['ud_upos','ud_deprel'],'ud_upos','en_ewt'),('gum',['gum_upos','ud_discourse'],'ud_discourse','en_gum')]:
        banks={task:torch.load(original[task],weights_only=False) for task in tasks};reference=banks[tasks[0]]
        tokens=load_from_disk(token_path(dataset_config(cfg,token_task)));raw=ud_raw(treebank);info=metadata(token_task)
        rows=[];ys=defaultdict(list);z=[];native=[];document=[];sentence=[];positions=[];words=[];genre_names=reference['genres'];global_sentence=0
        generator=torch.Generator().manual_seed(cfg.diagnostics.sample_seed)
        for split_index,(split,dataset) in enumerate(tokens.items()):
            candidates=defaultdict(lambda:defaultdict(list));split_start=global_sentence
            for sid,row in enumerate(dataset):
                genre=info[split][sid]['genre'];doc=info[split][sid]['document'];candidates[genre][doc].append(sid)
            chosen=defaultdict(list)
            budget=[cfg.pilot.train_per_genre,cfg.pilot.validation_per_genre,cfg.pilot.test_per_genre][split_index]
            for genre,docs in candidates.items():
                queues=[]
                for doc in sorted(docs):
                    sids=docs[doc];queues.append([sids[i] for i in torch.randperm(len(sids),generator=generator).tolist()])
                remaining=budget
                while remaining and any(queues):
                    for queue in queues:
                        if not queue or not remaining:continue
                        sid=queue.pop();width=len(dataset[sid]['words']);focals=torch.randperm(width,generator=generator)[:min(cfg.pilot.positions_per_sentence,remaining)].sort().values.tolist()
                        chosen[sid]=focals;remaining-=len(focals)
            for sid in sorted(chosen):
                row=dict(dataset[sid]);focals=chosen[sid];original_row=raw[split][sid]
                encoded=tokenizer(original_row['tokens'],is_split_into_words=True,truncation=True,max_length=cfg.encoder.max_length,add_special_tokens=cfg.encoder.special_tokens)
                if encoded['input_ids']!=row['ids']:raise ValueError(f'Tree/token alignment mismatch {source}/{split}/{sid}')
                wid=encoded.word_ids();kept=[w for i,w in enumerate(wid) if w is not None and (i==0 or w!=wid[i-1])];lookup={w:i for i,w in enumerate(kept)}
                parents=list(map(int,original_row['head']));slots=torch.zeros(len(focals),8,len(kept),dtype=torch.bool)
                for j,f in enumerate(focals):
                    for slot,offset in enumerate((-1,1,-2,2)):
                        if 0<=f+offset<len(kept):slots[j,slot,f+offset]=True
                    token=kept[f];parent=parents[token]-1
                    relatives=[[parent],[w for w,p in enumerate(parents) if p==token+1],
                               [w for w,p in enumerate(parents) if p==parent+1 and w!=token] if parent>=0 else [],
                               [parents[parent]-1] if parent>=0 else []]
                    for slot,rel in enumerate(relatives,4):
                        for w in rel:
                            if w in lookup and w!=token:slots[j,slot,lookup[w]]=True
                prefix=[];previous_words=0;previous_total_words=0
                if sid and info[split][sid-1]['document']==info[split][sid]['document']:
                    previous=dataset[sid-1];previous_total_words=len(previous['words']);room=cfg.encoder.max_length-len(row['ids']);starts=previous['positions']
                    fitting=[i for i,start in enumerate(starts) if len(previous['ids'])-start<=room]
                    if fitting:
                        start=fitting[0];prefix=previous['ids'][starts[start]:];previous_words=len(starts)-start
                sample_ids=list(range(len(z),len(z)+len(focals)))
                row.update(focals=focals,slots=slots,tree=distances(parents,[kept[f] for f in focals],kept),global_sentence=split_start+sid,
                           sample_ids=sample_ids,previous_ids=prefix,previous_words=previous_words,previous_total_words=previous_total_words)
                rows.append(row)
                offset=int(reference['offsets'][split_start+sid])
                for task in tasks:ys[task]+=[int(banks[task]['y'][offset+f]) for f in focals]
                z.extend([genre_names.index(info[split][sid]['genre'])]*len(focals));native.extend([split_index]*len(focals));document.extend([info[split][sid]['document']]*len(focals))
                sentence.extend([split_start+sid]*len(focals));positions.extend(focals);words.extend([row['words'][f] for f in focals])
            global_sentence+=len(dataset)
            for genre in genre_names:
                relevant=[r for r in rows if native[r['sample_ids'][0]]==split_index and z[r['sample_ids'][0]]==genre_names.index(genre)]
                counts.append({'source':source,'split':split,'genre':genre,'focals':sum(len(r['focals']) for r in relevant),
                               'previous_available':sum(len(r['focals']) for r in relevant if r['previous_words']>0),
                               'previous_truncated':sum(len(r['focals']) for r in relevant if r['previous_words']<r['previous_total_words'])})
        bank={'rows':rows,'ys':{k:torch.tensor(v) for k,v in ys.items()},'label_names':{k:b['labels'] for k,b in banks.items()},'genres':genre_names,
              'z':torch.tensor(z),'native':torch.tensor(native),'document':document,'sentence':torch.tensor(sentence),'position':torch.tensor(positions),'words':words}
        torch.save(bank,folder/(source+'.pt'))
    write_csv(output/'samples.csv',counts)
    return {'sources':2,'focals':sum(r['focals'] for r in counts)}
