"""Batched full-sentence MLM sampling and shared-window CRT scoring."""
import json,time,math
import torch
from torch.utils.data import DataLoader
from ..runtime import autocast
from ..diagnostics.common import save_json
from .data import paths,Queries,collate,features,model,window,TASKS
from .train import readouts


@torch.no_grad()
def distributions(encoder,queries,special,cfg):
    for batch in DataLoader(Queries(queries),batch_size=cfg.crt.mlm_batch_size,collate_fn=collate,num_workers=cfg.runtime.workers):
        with autocast(cfg):logits=encoder(batch['ids'].to(cfg.runtime.device),batch['attention_mask'].to(cfg.runtime.device),batch['positions'].to(cfg.runtime.device),mlm=True)['logits']
        logits=logits.float();logits[:,special]=-torch.inf;p=logits.softmax(-1).cpu()
        entropy=-torch.special.xlogy(p,p).sum(-1)
        for index,probability,h in zip(batch['index'].tolist(),p,entropy):yield index,probability,h


@torch.no_grad()
def statistics(encoder,core,queries,labels,cfg):
    h=features(encoder,queries,cfg).float();output=torch.empty(len(queries),3)
    for start in range(0,len(h),cfg.runtime.batch_size):
        values=core(h[start:start+cfg.runtime.batch_size].to(cfg.runtime.device));target=labels[start:start+cfg.runtime.batch_size].to(cfg.runtime.device)
        for task,value in enumerate(values):output[start:start+len(value),task]=value.float().log_softmax(-1).gather(1,target[:,task,None]).squeeze(-1).cpu()
    return output


def baseline(encoder,core,rows,cfg):
    queries=[]
    for row in rows:
        ids,focal=window(row['ids'],row['focal'],cfg);queries.append({'ids':ids,'focal':focal})
    return statistics(encoder,core,queries,torch.tensor([r['y'] for r in rows]),cfg)


@torch.no_grad()
def condition(encoder,core,rows,offsets,code,special,cfg,original):
    size=len(rows);k=cfg.crt.draws;pair=len(offsets)==2
    draws=torch.empty(size,k,len(offsets),dtype=torch.long);entropy=torch.zeros(size);first_entropy=torch.zeros(size);second_entropy=torch.zeros(size,k) if pair else None
    generators=[torch.Generator().manual_seed(cfg.crt.sample_seed+row['row']*1009+code) for row in rows]
    first_queries=[]
    for row in rows:
        ids=[101,*row['ids'],102]
        for d in offsets:ids[row['focal']+d+1]=special['mask']
        first_queries.append({'ids':ids,'focal':row['focal']+offsets[0]+1})
    for index,p,h in distributions(encoder,first_queries,special['special_ids'],cfg):
        draws[index,:,0]=torch.multinomial(p,k,replacement=True,generator=generators[index]);first_entropy[index]=h
    if pair:
        second_queries=[];keys=[]
        for index,row in enumerate(rows):
            for token in draws[index,:,0].unique(sorted=True).tolist():
                ids=[101,*row['ids'],102];ids[row['focal']+offsets[0]+1]=token;ids[row['focal']+offsets[1]+1]=special['mask']
                second_queries.append({'ids':ids,'focal':row['focal']+offsets[1]+1});keys.append((index,token))
        for query,p,h in distributions(encoder,second_queries,special['special_ids'],cfg):
            index,token=keys[query];selected=draws[index,:,0]==token
            draws[index,selected,1]=torch.multinomial(p,int(selected.sum()),replacement=True,generator=generators[index]);second_entropy[index,selected]=h
        entropy=first_entropy+second_entropy.mean(-1)
    else:entropy=first_entropy.clone()
    # Score each distinct altered window once; identical originals copy baseline exactly.
    queries=[];labels=[];maps=[];masked=[];masked_labels=[]
    for index,row in enumerate(rows):
        unique,inverse=torch.unique(draws[index],dim=0,sorted=True,return_inverse=True);mapping=[]
        for values in unique.tolist():
            ids=list(row['ids'])
            for d,value in zip(offsets,values):ids[row['focal']+d]=value
            if ids==row['ids']:mapping.append(-1)
            else:
                value,focal=window(ids,row['focal'],cfg);mapping.append(len(queries));queries.append({'ids':value,'focal':focal});labels.append(row['y'])
        maps.append((torch.tensor(mapping),inverse))
        ids=list(row['ids'])
        for d in offsets:ids[row['focal']+d]=special['mask']
        value,focal=window(ids,row['focal'],cfg);masked.append({'ids':value,'focal':focal});masked_labels.append(row['y'])
    values=statistics(encoder,core,queries,torch.tensor(labels),cfg) if queries else torch.empty(0,3)
    resampled=torch.empty(size,k,3)
    for index,(mapping,inverse) in enumerate(maps):
        indices=mapping[inverse];same=indices<0
        resampled[index,same]=original[index];resampled[index,~same]=values[indices[~same]]
    mask=statistics(encoder,core,masked,torch.tensor(masked_labels),cfg)
    return {'row':torch.tensor([r['row'] for r in rows]),'T':original,'resampled':resampled,'masked':mask,'entropy':entropy,'first_entropy':first_entropy,
            'draws':draws.int(),'second_entropy':second_entropy,'offsets':offsets,'original_tokens':torch.tensor([[r['ids'][r['focal']+d] for d in offsets] for r in rows])}


def cases(cfg):return [([int(d)],i,f'offset_{int(d):+d}') for i,d in enumerate(cfg.crt.offsets)]+[([int(d) for d in pair],10+i,'pair_'+'_'.join(f'{int(d):+d}' for d in pair)) for i,pair in enumerate(cfg.crt.pairs)]


def execute(encoder,core,all_rows,indices,original,cfg,special,folder):
    folder.mkdir(parents=True,exist_ok=True);seconds={}
    for offsets,code,name in cases(cfg):
        target=folder/(name+'.pt');started=time.time()
        if cfg.crt.resume and target.exists():
            seconds[name]=0.;continue
        valid=[int(i) for i in indices if all(0<=all_rows[int(i)]['focal']+d<len(all_rows[int(i)]['ids']) for d in offsets)]
        parts=[]
        reuse=folder.parent/'benchmark'/(name+'.pt')
        if folder.name=='all' and reuse.exists():
            saved=torch.load(reuse,weights_only=False);wanted=set(valid);mask=torch.tensor([int(i) in wanted for i in saved['row']],dtype=torch.bool)
            if mask.any():
                parts.append({key:(value[mask] if isinstance(value,torch.Tensor) else value) for key,value in saved.items()})
                kept=set(saved['row'][mask].tolist());valid=[i for i in valid if i not in kept]
        if not valid and not parts:
            torch.save({'row':torch.empty(0,dtype=torch.long),'T':torch.empty(0,3),'resampled':torch.empty(0,cfg.crt.draws,3),'masked':torch.empty(0,3),'entropy':torch.empty(0),'first_entropy':torch.empty(0),'draws':torch.empty(0,cfg.crt.draws,len(offsets),dtype=torch.int32),'second_entropy':None,'offsets':offsets,'original_tokens':torch.empty(0,len(offsets),dtype=torch.long)},target)
            seconds[name]=time.time()-started;continue
        # Similar-length full sentences share MLM batches without padding to corpus maximum.
        valid.sort(key=lambda i:len(all_rows[i]['ids']))
        for start in range(0,len(valid),cfg.crt.row_batch_size):
            chosen=valid[start:start+cfg.crt.row_batch_size]
            parts.append(condition(encoder,core,[all_rows[i] for i in chosen],offsets,code,special,cfg,original[chosen]))
        result={key:(torch.cat([p[key] for p in parts]) if isinstance(parts[0][key],torch.Tensor) else parts[0][key]) for key in parts[0]}
        torch.save(result,target);seconds[name]=time.time()-started
        print(f'{folder.name} {name}: {len(result["row"])} rows, {seconds[name]:.1f}s',flush=True)
    return seconds


def experiment(cfg,output):
    results,cache=paths(cfg);rows=torch.load(cache/'rows.pt',weights_only=False);special=json.loads((cache/'vocabulary.json').read_text());encoder=model(cfg);core=readouts(cfg)
    original=baseline(encoder,core,rows,cfg);torch.save({'T':original,'rows':rows,'tasks':TASKS,'draws':cfg.crt.draws,'encoder':cfg.encoder.name},output/'baseline.pt')
    order=torch.randperm(len(rows),generator=torch.Generator().manual_seed(cfg.crt.row_seed+1));benchmark_count=max(1,math.ceil(len(rows)*cfg.crt.benchmark_fraction));benchmark=order[:benchmark_count]
    print(f'Timing {len(benchmark)}/{len(rows)} sentences ({100*len(benchmark)/len(rows):.2f}%): UPOS singles plus the joint-pair cost, sharing encoder work across three tasks.',flush=True)
    times=execute(encoder,core,rows,benchmark,original,cfg,special,output/'benchmark')
    measurement=results/'benchmark.json'
    if cfg.crt.resume and measurement.exists():timing=json.loads(measurement.read_text())
    else:
        timing={'benchmark_rows':len(benchmark),'available_rows':len(rows),'seconds':times,'seconds_per_sentence':sum(times.values())/len(benchmark)};save_json(measurement,timing)
    elapsed=time.time()-json.loads((results/'timing.json').read_text())['started_unix'];count=len(rows);halvings=0
    while elapsed+timing['seconds_per_sentence']*max(0,count-len(benchmark))>cfg.crt.budget_seconds and count>1:
        count=max(1,count//2);halvings+=1
    selected=order[:count];estimated=elapsed+timing['seconds_per_sentence']*max(0,count-len(benchmark));print(f'Projected total {estimated/3600:.2f}h; selected {count}/{len(rows)} sentences after {halvings} halvings; K={cfg.crt.draws}.',flush=True)
    if estimated>cfg.crt.budget_seconds:raise RuntimeError('Elapsed preparation and benchmark already exceed the three-hour budget.')
    save_json(results/'budget.json',{'selected_rows':count,'available_rows':len(rows),'halvings':halvings,'draws':cfg.crt.draws,'estimated_total_seconds':estimated,'elapsed_at_decision':elapsed})
    torch.save(selected,output/'selected.pt')
    execute(encoder,core,rows,selected,original,cfg,special,output/'all')
    return {'sentences':count,'estimated_total_seconds':estimated}
