"""Paired readout trajectories with separate ID/held-genre checkpoint selection."""
import json
import math
import time
import torch
from torch.nn import functional as F
from tagger.ensemble import IndependentReadouts,clip_heads
from ..runtime import autocast,parallel,device_ids
from ..diagnostics.common import save_json,write_csv
from ..invariance.data import banks,rotations,paths,loader
from ..invariance.train import evaluate
from .loss import penalty


POLICIES=('id','genre')


def batch_context(batch,context,cfg):
    ids=batch['index'];source=context['random_source'][ids].clamp_min(0)
    return [value.to(cfg.runtime.device) for value in (context['residual'][ids],context['residual'][source],context['covered'][ids])]


def partition(bank,genres,validation,test):
    training=torch.isin(bank['sentence_z'],torch.tensor(genres))
    return {'train':(training&(bank['sentence_split']==0)).nonzero().flatten(),
            'id_validation':(training&(bank['sentence_split']==1)).nonzero().flatten(),
            'genre_validation':(bank['sentence_z']==validation).nonzero().flatten(),
            'id':(training&(bank['sentence_split']==2)).nonzero().flatten(),
            'ood':(bank['sentence_z']==test).nonzero().flatten()}


def calibration(cfg,output):
    results,_=paths(cfg);grids={};rows=[]
    for name,bank in banks(cfg):
        genre,genres,validation,test=next(rotations(name,bank,cfg));ids=partition(bank,genres,validation,test)
        context=torch.load(results/'contexts'/name/genre/'context.pt',weights_only=False)
        torch.manual_seed(cfg.diagnostics.seeds[0]);core=IndependentReadouts(cfg.tagger,bank['features'].shape[-1],len(bank['labels']),1).to(cfg.runtime.device)
        model=parallel(core,cfg);optimizer=torch.optim.AdamW(core.parameters(),lr=cfg.train.lr,weight_decay=cfg.train.weight_decay)
        batches=loader(bank,ids['train'],cfg,train_genres=genres,seed=cfg.diagnostics.seeds[0]);iterator=iter(batches)
        values=[]
        for step in range(cfg.zfree.calibration_steps+cfg.zfree.calibration_batches):
            try:batch=next(iterator)
            except StopIteration:iterator=iter(batches);batch=next(iterator)
            x=batch['x'].to(cfg.runtime.device)[:,None];y=batch['y'].to(cfg.runtime.device);z=batch['z'].to(cfg.runtime.device)
            if step<cfg.zfree.calibration_steps:
                model.train();optimizer.zero_grad(set_to_none=True)
                with autocast(cfg):logits=model(x)
                loss=F.cross_entropy(logits[:,0].float(),y);loss.backward()
                if cfg.train.clip is not None:clip_heads(core,cfg.train.clip)
                optimizer.step()
            else:
                model.eval()
                with torch.no_grad(),autocast(cfg):logits=model(x)
                logp=logits.float().log_softmax(-1);ce=F.cross_entropy(logits[:,0].float(),y)
                d,random,covered=batch_context(batch,context,cfg)
                value=penalty(logp.expand(-1,3,-1),y,z,d,random,covered,['Rz','Rg','Rrand'],torch.tensor(genres,device=z.device))
                values.append(torch.cat([ce[None],value]).detach().cpu())
        measurements=torch.stack(values);ratios=(measurements[:,0,None]/measurements[:,1:].clamp_min(1e-12)).median(0).values
        logs=ratios.log10();middle=(logs.min()+logs.max())/2;spacing=(logs.max()-logs.min()).clamp_min(1)
        grid=(10**(middle+torch.tensor([-1.5,-.5,.5,1.5])*spacing)).tolist();grids[name]={'rotation':genre,'lambdas':grid,'median_ce_to_penalty':dict(zip(('Rz','Rg','Rrand'),ratios.tolist()))}
        for batch_number,v in enumerate(measurements):
            for i,rule in enumerate(('Rz','Rg','Rrand')):rows.append({'dataset':name,'rotation':genre,'batch':batch_number,'rule':rule,'ce':float(v[0]),'raw_penalty':float(v[i+1]),**{f'weighted_penalty_over_ce_{j}':float(lam*v[i+1]/v[0]) for j,lam in enumerate(grid)}})
        save_json(output/'grids.json',grids);write_csv(output/'calibration.csv',rows)
        print(f'Calibrated {name}: {grid}',flush=True)
    return {'tasks':len(grids)}


@torch.no_grad()
def predictions(model,bank,ids,heads,cfg):
    result=evaluate(model,bank,ids,range(heads),cfg)
    logp=[]
    for batch in loader(bank,ids,cfg):
        chunks=[]
        for start in range(0,heads,cfg.invariance.head_chunk):
            with autocast(cfg):value=model(batch['x'].to(cfg.runtime.device)[:,None],heads=list(range(start,min(heads,start+cfg.invariance.head_chunk))))
            chunks.append(value.float().log_softmax(-1).cpu())
        logp.append(torch.cat(chunks,dim=1))
    result['log_probability']=torch.cat(logp)
    return result


def fit(bank,context,ids,genres,specs,cfg,seed,output):
    if cfg.invariance.resume and (output/'predictions.pt').exists():return
    output.mkdir(parents=True,exist_ok=True);started=time.time();torch.manual_seed(seed);n=len(specs)
    core=IndependentReadouts(cfg.tagger,bank['features'].shape[-1],len(bank['labels']),n).to(cfg.runtime.device);model=parallel(core,cfg)
    optimizer=torch.optim.AdamW(core.parameters(),lr=cfg.train.lr,weight_decay=cfg.train.weight_decay)
    strengths=torch.tensor([s['lambda'] for s in specs],device=cfg.runtime.device);genre_tensor=torch.tensor(genres,device=cfg.runtime.device)
    batches=loader(bank,ids['train'],cfg,train_genres=genres,seed=seed);iterator=iter(batches)
    minimum=max(cfg.invariance.min_steps,len(batches));active=torch.ones(n,dtype=torch.bool)
    best=torch.full((2,n),-torch.inf);best_step=torch.zeros(2,n,dtype=torch.long);stale=torch.zeros(2,n,dtype=torch.long)
    states=[{k:v.detach().cpu().clone() for k,v in core.state_dict().items()} for _ in POLICIES];history=[]
    sums=torch.zeros(n,2,device=cfg.runtime.device);counts=torch.zeros(n,device=cfg.runtime.device)
    print(f'Fit {output}: {n} heads, {cfg.invariance.max_steps} max steps, two validation policies',flush=True)
    for step in range(1,cfg.invariance.max_steps+1):
        try:batch=next(iterator)
        except StopIteration:iterator=iter(batches);batch=next(iterator)
        model.train();optimizer.zero_grad(set_to_none=True)
        x=batch['x'].to(cfg.runtime.device)[:,None];y=batch['y'].to(cfg.runtime.device);z=batch['z'].to(cfg.runtime.device)
        d,random,covered=batch_context(batch,context,cfg)
        cpu_rng=torch.get_rng_state();gpu_rng={dev:torch.cuda.get_rng_state(dev) for dev in device_ids(cfg)}
        active_ids=active.nonzero().flatten().tolist()
        for start in range(0,len(active_ids),cfg.invariance.head_chunk):
            heads=active_ids[start:start+cfg.invariance.head_chunk];torch.set_rng_state(cpu_rng)
            for dev,state in gpu_rng.items():torch.cuda.set_rng_state(state,dev)
            with autocast(cfg):logits=model(x,heads=heads)
            logits=logits.float();ce=F.cross_entropy(logits.permute(1,2,0),y[None].expand(len(heads),-1),reduction='none').mean(-1)
            value=penalty(logits.log_softmax(-1),y,z,d,random,covered,[specs[h]['rule'] for h in heads],genre_tensor)
            (ce+strengths[heads]*value).sum().backward();sums[heads,0]+=ce.detach();sums[heads,1]+=value.detach();counts[heads]+=1
        if cfg.train.clip is not None:clip_heads(core,cfg.train.clip)
        optimizer.step()
        if step%cfg.invariance.validate_every and step!=cfg.invariance.max_steps:continue
        means=(sums/counts[:,None].clamp_min(1)).cpu()
        for policy_index,policy in enumerate(POLICIES):
            current=evaluate(model,bank,ids[policy+'_validation'],range(n),cfg)['macro_f1'];improved=(current>best[policy_index])&active
            for key,value in core.state_dict().items():states[policy_index][key][improved]=value.detach().cpu()[improved]
            best[policy_index,improved]=current[improved];best_step[policy_index,improved]=step
            stale[policy_index,active]+=1;stale[policy_index,improved]=0
            for h,spec in enumerate(specs):history.append({'step':step,'policy':policy,'head':h,**spec,'validation_macro_f1':float(current[h]),'best_validation_macro_f1':float(best[policy_index,h]),'ce':float(means[h,0]),'penalty':float(means[h,1]),'active':bool(active[h])})
        if step>=minimum:active&=(stale<cfg.invariance.patience).any(0)
        sums.zero_();counts.zero_();print(f'{output.name} step {step}: {int(active.sum())}/{n} active',flush=True)
        if not active.any():break
    selected={}
    for p,policy in enumerate(POLICIES):
        chosen={}
        for rule in ('Rz','Rg','Rrand'):
            candidates=[0]+[h for h,s in enumerate(specs) if s['rule']==rule]
            chosen[rule]=max(candidates,key=lambda h:(float(best[p,h]),-specs[h]['lambda']))
        selected[policy]=chosen
    torch.save({'states':dict(zip(POLICIES,states)),'heads':specs,'validation_macro_f1':best,'best_step':best_step,'selected':selected,
                'steps':step,'dimension':bank['features'].shape[-1],'classes':len(bank['labels'])},output/'model.pt');write_csv(output/'history.csv',history)
    saved={}
    for p,policy in enumerate(POLICIES):
        core.load_state_dict(states[p]);saved[policy]={split:predictions(model,bank,ids[split],n,cfg) for split in ('id','ood','id_validation')}
    torch.save(saved,output/'predictions.pt');save_json(output/'timing.json',{'seconds':time.time()-started,'steps':step})


def experiment(cfg,output):
    results,_=paths(cfg);grids=json.loads((results/'calibration'/'grids.json').read_text())
    for name,bank in banks(cfg):
        specs=[{'rule':'R0','lambda':0.}]+[{'rule':rule,'lambda':lam} for rule in ('Rz','Rg','Rrand') for lam in grids[name]['lambdas']]
        for genre,genres,validation,test in rotations(name,bank,cfg):
            context=torch.load(results/'contexts'/name/genre/'context.pt',weights_only=False)
            ids=partition(bank,genres,validation,test)
            for seed in cfg.diagnostics.seeds:fit(bank,context,ids,genres,specs,cfg,seed,output/name/genre/f'seed{seed}')
    return {'tasks':len(grids)}
