"""Pilot-selected fixed-mode regularization with step budgets and early stopping."""
import json
import time
import torch
from torch.nn import functional as F
from tagger.ensemble import IndependentReadouts,clip_heads
from ..diagnostics.common import save_json,write_csv
from ..runtime import autocast,parallel,device_ids
from .data import banks,rotations,plan,loader,paths
from .train import evaluate
from .stage1_loss import samples,penalty


def specifications(strengths):
    return [{'rule':'R0','lambda':0.}]+[{'rule':rule,'lambda':float(lam)} for rule in ('R1','R2') for lam in strengths[rule] if lam>0]


def fit(bank,rotation,validation,test,genres,specs,cfg,seed,output):
    if cfg.invariance.resume and (output/'predictions.pt').exists() and (output/'model.pt').exists():return
    output.mkdir(parents=True,exist_ok=True);started=time.time();torch.manual_seed(seed)
    n=len(specs)
    core=IndependentReadouts(cfg.tagger,bank['features'].shape[-1],len(bank['labels']),n).to(cfg.runtime.device)
    model=parallel(core,cfg);optimizer=torch.optim.AdamW(core.parameters(),lr=cfg.train.lr,weight_decay=cfg.train.weight_decay)
    strengths=torch.tensor([s['lambda'] for s in specs],device=cfg.runtime.device)
    genre_tensor=torch.tensor(genres,device=cfg.runtime.device)
    batches=loader(bank,rotation['train_sentences'],cfg,train_genres=genres,seed=seed);iterator=iter(batches)
    minimum=max(cfg.invariance.min_steps,len(batches))
    generator=torch.Generator().manual_seed(seed+1729)
    best=torch.full((n,),-torch.inf);best_step=torch.zeros(n,dtype=torch.long);stale=torch.zeros(n,dtype=torch.long)
    active=torch.ones(n,dtype=torch.bool);selected={k:v.detach().cpu().clone() for k,v in core.state_dict().items()};history=[]
    sums=torch.zeros(n,2,device=cfg.runtime.device);counts=torch.zeros(n,device=cfg.runtime.device)
    print(f'Fit {output}: {n} heads, maximum {cfg.invariance.max_steps} steps, stop patience {cfg.invariance.patience}',flush=True)
    for step in range(1,cfg.invariance.max_steps+1):
        try:batch=next(iterator)
        except StopIteration:iterator=iter(batches);batch=next(iterator)
        model.train();optimizer.zero_grad(set_to_none=True)
        x=batch['x'].to(cfg.runtime.device)[:,None];y=batch['y'].to(cfg.runtime.device);z=batch['z'].to(cfg.runtime.device)
        sampled=samples(batch,genres,cfg,generator)
        cpu_rng=torch.get_rng_state();gpu_rng={d:torch.cuda.get_rng_state(d) for d in device_ids(cfg)}
        active_ids=active.nonzero().flatten().tolist()
        for start in range(0,len(active_ids),cfg.invariance.head_chunk):
            ids=active_ids[start:start+cfg.invariance.head_chunk]
            torch.set_rng_state(cpu_rng)
            for d,state in gpu_rng.items():torch.cuda.set_rng_state(state,d)
            with autocast(cfg):logits=model(x,heads=ids)
            logits=logits.float();ce=F.cross_entropy(logits.permute(1,2,0),y[None].expand(len(ids),-1),reduction='none').mean(-1)
            rules=[specs[i]['rule'] for i in ids]
            value=penalty(logits.softmax(-1),z,sampled,rules,genre_tensor,cfg) if any(r!='R0' for r in rules) else torch.zeros_like(ce)
            (ce+strengths[ids]*value).sum().backward()
            sums[ids,0]+=ce.detach();sums[ids,1]+=value.detach();counts[ids]+=1
        if cfg.train.clip is not None:clip_heads(core,cfg.train.clip)
        optimizer.step()
        if step%cfg.invariance.validate_every and step!=cfg.invariance.max_steps:continue
        current=evaluate(model,bank,validation,range(n),cfg)['macro_f1'];improved=(current>best)&active
        for key,value in core.state_dict().items():selected[key][improved]=value.detach().cpu()[improved]
        best[improved]=current[improved];best_step[improved]=step
        stale[active]+=1;stale[improved]=0
        if step>=minimum:active&=stale<cfg.invariance.patience
        means=(sums/counts[:,None].clamp_min(1)).cpu()
        for i,spec in enumerate(specs):history.append({'step':step,'head':i,**spec,'validation_macro_f1':float(current[i]),'best_validation_macro_f1':float(best[i]),'active':bool(active[i]),'ce':float(means[i,0]),'penalty':float(means[i,1])})
        sums.zero_();counts.zero_()
        print(f'{output.name} step {step}: {int(active.sum())}/{n} active; R0 best val F1 {float(best[0]):.4f}',flush=True)
        if not active.any():break
    core.load_state_dict(selected)
    torch.save({'state':selected,'heads':specs,'validation_macro_f1':best,'best_step':best_step,'steps':step,'seconds':time.time()-started,
                'dimension':bank['features'].shape[-1],'classes':len(bank['labels'])},output/'model.pt')
    write_csv(output/'history.csv',history)
    predictions={key:evaluate(model,bank,ids,range(n),cfg) for key,ids in test.items()}
    torch.save(predictions,output/'predictions.pt')
    save_json(output/'timing.json',{'seconds':time.time()-started,'steps':step})


def experiment(cfg,output,pilot=False):
    results,_=paths(cfg);selection={}
    if not pilot:selection=json.loads((results/'pilot'/'selection.json').read_text())
    for name,bank in banks(cfg):
        strengths={rule:list(cfg.invariance.strengths) if pilot else selection[name][rule]['run_lambdas'] for rule in ('R1','R2')}
        specs=specifications(strengths)
        rotation_list=list(rotations(name,bank,cfg))
        if pilot:rotation_list=rotation_list[:1]
        for genre,genres,validation,test in rotation_list:
            # The first rotation is already fitted for every lambda by the pilot.
            if not pilot and genre==selection[name]['rotation']:continue
            rotation=plan(bank,genres,cfg,specifications=specs)
            folder=output/name/genre;folder.mkdir(parents=True,exist_ok=True);torch.save(rotation,folder/'plan.pt')
            validation_ids=(bank['sentence_z']==validation).nonzero().flatten()
            test_ids={'ood':(bank['sentence_z']==test).nonzero().flatten(),
                      'id':(torch.isin(bank['sentence_z'],torch.tensor(genres))&(bank['sentence_split']==2)).nonzero().flatten()}
            for seed in cfg.diagnostics.seeds:fit(bank,rotation,validation_ids,test_ids,genres,specs,cfg,seed,folder/f'seed{seed}')
            if pilot:
                scores=torch.stack([torch.load(folder/f'seed{seed}'/'model.pt',map_location='cpu',weights_only=False)['validation_macro_f1'] for seed in cfg.diagnostics.seeds]).mean(0)
                selected={'rotation':genre,'validation_genre':bank['genres'][validation]}
                grid=[0.,*map(float,cfg.invariance.strengths)]
                for rule in ('R1','R2'):
                    candidates=[0]+[i for i,s in enumerate(specs) if s['rule']==rule]
                    head=max(candidates,key=lambda i:(float(scores[i]),-specs[i]['lambda']))
                    main=specs[head]['lambda'];position=grid.index(main)
                    neighbour=grid[position+1] if position+1<len(grid) else grid[position-1]
                    selected[rule]={'lambda':main,'neighbour':neighbour,'run_lambdas':[main,neighbour],'validation_macro_f1':float(scores[head])}
                selection[name]=selected;save_json(output/'selection.json',selection)
    return {'tasks':len(selection),'pilot':pilot}
