"""Vectorized independent grid cells, with one frozen random-mask scorer."""
import json,time
import torch
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader,Sampler
from tagger.ensemble import IndependentReadouts,clip_heads
from ..runtime import autocast,parallel,device_ids
from ..diagnostics.common import save_json,write_csv
from ..partition.train import metric_vectors
from ..zfree.loss import center,covariance_norm
from ..invariance.data import paths
from .data import load,cache,rotation,SOURCES,FIXED


def genres_for(task,bank,cfg):return list(cfg.invariance.gum_test_genres) if task in ('gum_upos','ud_discourse') else bank['genres']


class Sentences(Dataset):
    def __init__(self,bank,features,ids):
        self.bank,self.features=bank,features;groups=bank['sentence'][ids].unique()
        self.groups=[ids[bank['sentence'][ids]==s] for s in groups];self.genre=torch.tensor([bank['z'][g[0]] for g in self.groups])
    def __len__(self):return len(self.groups)
    def __getitem__(self,i):
        ids=self.groups[i];return {'x':self.features[ids].float(),'y':self.bank['y'][ids],'z':self.bank['z'][ids],'index':ids}


def collate(rows):return {k:torch.cat([r[k] for r in rows]) for k in rows[0]}


class Balanced(Sampler):
    def __init__(self,data,batch,seed):
        self.groups=[(data.genre==g).nonzero().flatten() for g in data.genre.unique()];self.per=max(1,batch//len(self.groups));self.steps=max((len(g)+self.per-1)//self.per for g in self.groups);self.seed=seed;self.pass_index=0
    def __len__(self):return self.steps
    def __iter__(self):
        gen=torch.Generator().manual_seed(self.seed+self.pass_index);self.pass_index+=1;columns=[]
        for g in self.groups:
            chunks=[];n=self.steps*self.per
            while n:
                chosen=g[torch.randperm(len(g),generator=gen)][:n];chunks.append(chosen);n-=len(chosen)
            columns.append(torch.cat(chunks).reshape(self.steps,self.per))
        for step in range(self.steps):yield torch.cat([c[step] for c in columns]).tolist()


def loader(bank,features,ids,cfg,seed=None):
    data=Sentences(bank,features,ids);options={'collate_fn':collate,'num_workers':cfg.runtime.workers,'pin_memory':cfg.runtime.device.startswith('cuda')}
    if seed is not None:return DataLoader(data,batch_sampler=Balanced(data,cfg.train.batch_size,seed),**options)
    return DataLoader(data,batch_size=cfg.runtime.batch_size,**options)


@torch.no_grad()
def predict(model,bank,features,ids,specs,cfg,scorer=False):
    model.eval();logits=[];index=[]
    for batch in loader(bank,features,ids,cfg):
        x=batch['x'].to(cfg.runtime.device);values=[]
        for start in range(0,len(specs),cfg.invariance.head_chunk):
            heads=list(range(start,min(start+cfg.invariance.head_chunk,len(specs))))
            columns=[specs[h]['column'] for h in heads]
            with autocast(cfg):value=model(x[:,[0]] if scorer else x[:,columns],heads=heads)
            values.append(value.float().cpu())
        logits.append(torch.cat(values,1));index.append(batch['index'])
    logits=torch.cat(logits);index=torch.cat(index);order=index.argsort()
    return {'index':index[order],'log_probability':logits[order].log_softmax(-1),'prediction':logits[order].argmax(-1).short(),
            **metric_vectors(logits,bank['y'][index],len(bank['labels']))}


def fit(bank,features,ids,genres,specs,cfg,seed,output,scorer=False,save_splits=('id','ood','id_validation')):
    output.mkdir(parents=True,exist_ok=True)
    if cfg.invariance.resume and (output/'predictions.pt').exists():return torch.load(output/'model.pt',weights_only=False)
    started=time.time();torch.manual_seed(seed);n=len(specs)
    core=IndependentReadouts(cfg.tagger,features.shape[-1],len(bank['labels']),n).to(cfg.runtime.device);model=parallel(core,cfg)
    optimizer=torch.optim.AdamW(core.parameters(),lr=cfg.train.lr,weight_decay=cfg.train.weight_decay)
    strengths=torch.tensor([s['lambda'] for s in specs],device=cfg.runtime.device);best=torch.full((n,),-torch.inf);best_step=torch.zeros(n,dtype=torch.long);stale=torch.zeros(n,dtype=torch.long);active=torch.ones(n,dtype=torch.bool)
    state={k:v.detach().cpu().clone() for k,v in core.state_dict().items()};history=[]
    batches=loader(bank,features,ids['train'],cfg,seed);iterator=iter(batches);minimum=max(cfg.invariance.min_steps,len(batches))
    print(f'Fit {output}: {n} independent heads, {len(ids["train"])} train focals',flush=True)
    for step in range(1,cfg.invariance.max_steps+1):
        try:batch=next(iterator)
        except StopIteration:iterator=iter(batches);batch=next(iterator)
        model.train();optimizer.zero_grad(set_to_none=True);x=batch['x'].to(cfg.runtime.device);y=batch['y'].to(cfg.runtime.device);z=batch['z'].to(cfg.runtime.device)
        if scorer:x=x[torch.arange(len(x),device=x.device),torch.randint(x.shape[1],(len(x),),device=x.device)][:,None]
        cpu_rng=torch.get_rng_state();gpu_rng={d:torch.cuda.get_rng_state(d) for d in device_ids(cfg)};heads=active.nonzero().flatten().tolist()
        for start in range(0,len(heads),cfg.invariance.head_chunk):
            h=heads[start:start+cfg.invariance.head_chunk];torch.set_rng_state(cpu_rng)
            for d,value in gpu_rng.items():torch.cuda.set_rng_state(value,d)
            with autocast(cfg):value=model(x if scorer else x[:,[specs[j]['column'] for j in h]],heads=h)
            value=value.float();ce=F.cross_entropy(value.permute(1,2,0),y[None].expand(len(h),-1),reduction='none').mean(-1)
            nuisance=(z[:,None]==torch.tensor(genres,device=z.device)).float()
            pen=covariance_norm(center(value.log_softmax(-1),y),nuisance)
            (ce+strengths[h]*pen).sum().backward()
        if cfg.train.clip is not None:clip_heads(core,cfg.train.clip)
        optimizer.step()
        if step%cfg.invariance.validate_every and step!=cfg.invariance.max_steps:continue
        current=predict(model,bank,features,ids['validation'],specs,cfg,scorer)['macro_f1'];improved=(current>best)&active
        for k,v in core.state_dict().items():state[k][improved]=v.detach().cpu()[improved]
        best[improved]=current[improved];best_step[improved]=step;stale[active]+=1;stale[improved]=0
        if step>=minimum:active&=stale<cfg.invariance.patience
        for h,spec in enumerate(specs):history.append({'step':step,'head':h,**spec,'validation_macro_f1':float(current[h]),'best':float(best[h]),'active':bool(active[h])})
        print(f'{output.name}: step {step}, {int(active.sum())}/{n} active',flush=True)
        if not active.any():break
    core.load_state_dict(state);saved={'state':state,'heads':specs,'best_step':best_step,'validation_macro_f1':best,'steps':step,'dimension':features.shape[-1],'classes':len(bank['labels'])}
    torch.save(saved,output/'model.pt');write_csv(output/'history.csv',history)
    torch.save({split:predict(model,bank,features,ids[split],specs,cfg,scorer) for split in save_splits},output/'predictions.pt')
    save_json(output/'timing.json',{'seconds':time.time()-started,'steps':step});return saved


def scorer(cfg,output):
    bank=load(cfg,cfg.pilot.task);genre=genres_for(cfg.pilot.task,bank,cfg)[cfg.pilot.rotation];genres,ids=rotation(bank,genre)
    features=torch.load(cache(cfg)/SOURCES[cfg.pilot.task]/'random.pt',mmap=True,weights_only=False)
    return {'steps':fit(bank,features,ids,genres,[{'column':0,'lambda':0.,'rule':'scorer','regularizer':'none'}],cfg,cfg.pilot.scorer_seed,output/cfg.pilot.task/genre,scorer=True,save_splits=('id_validation',))['steps']}


def pilot(cfg,output):
    results,_=paths(cfg);task=cfg.pilot.task;bank=load(cfg,task);genre=genres_for(task,bank,cfg)[0];genres,ids=rotation(bank,genre)
    features=torch.load(cache(cfg)/SOURCES[task]/'fixed.pt',mmap=True,weights_only=False)[:,[0]]
    # A short independent full-view baseline calibrates the penalty scale.
    pilot_cfg=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True))
    pilot_cfg.invariance.max_steps=100;pilot_cfg.invariance.validate_every=100
    baseline=fit(bank,features,ids,genres,[{'column':0,'lambda':0.,'rule':'full','regularizer':'none'}],pilot_cfg,0,output/task/'calibration',save_splits=('id_validation',))
    core=IndependentReadouts(cfg.tagger,768,len(bank['labels']),1).to(cfg.runtime.device);core.load_state_dict(baseline['state']);core.eval();ratios=[]
    with torch.no_grad():
        for number,batch in enumerate(loader(bank,features,ids['train'],cfg,0)):
            x=batch['x'].to(cfg.runtime.device);y=batch['y'].to(cfg.runtime.device);z=batch['z'].to(cfg.runtime.device)
            with autocast(cfg):value=core(x)
            value=value.float();ce=F.cross_entropy(value[:,0],y);pen=covariance_norm(center(value.log_softmax(-1),y),(z[:,None]==torch.tensor(genres,device=z.device)).float())[0]
            ratios.append((ce/pen.clamp_min(1e-12)).cpu())
            if number==11:break
    anchor=float(torch.stack(ratios).median());grid=[0.]+[anchor*10**p for p in (-1.5,-.5,.5,1.5)]
    specs=[{'column':0,'lambda':lam,'rule':'full','regularizer':'none' if lam==0 else 'cond'} for lam in grid]
    scores=[]
    for seed in cfg.diagnostics.seeds:scores.append(fit(bank,features,ids,genres,specs,cfg,seed,output/task/f'seed{seed}',save_splits=('id_validation',))['validation_macro_f1'])
    mean=torch.stack(scores).mean(0);chosen=max(range(len(grid)),key=lambda h:(float(mean[h]),-grid[h]))
    save_json(output/(task+'.json'),{'lambda':grid[chosen],'grid':grid,'anchor':anchor,'validation_macro_f1':mean.tolist(),'rotation':genre})
    return {'lambda':grid[chosen]}


def cell(cfg,output):
    task=cfg.pilot.task;bank=load(cfg,task);genre=genres_for(task,bank,cfg)[cfg.pilot.rotation];genres,ids=rotation(bank,genre);results,_=paths(cfg)
    chosen=json.loads((results/'select'/'selections.json').read_text())[task][genre]['subsets'];lam=json.loads((results/'pilot'/(task+'.json')).read_text())['lambda']
    fixed=torch.load(cache(cfg)/SOURCES[task]/'fixed.pt',mmap=True,weights_only=False);features=[fixed[:,i] for i in range(8)];columns={r:i for i,r in enumerate(FIXED)}
    for rule,subset in chosen.items():columns[rule]=len(features);features.append(torch.load(cache(cfg)/SOURCES[task]/f'menu_{subset:03d}.pt',mmap=True,weights_only=False))
    validity=torch.randperm(256,generator=torch.Generator().manual_seed(cfg.pilot.scorer_seed+20))[:cfg.pilot.validity_subsets].sort().values.tolist()
    specs=[]
    rules=list(cfg.pilot.rules) if cfg.pilot.rule=='all' else [cfg.pilot.rule]
    regularizers=list(cfg.pilot.regularizers) if cfg.pilot.regularizer=='all' else [cfg.pilot.regularizer]
    for rule in rules:
        for reg in regularizers:specs.append({'rule':rule,'regularizer':reg,'lambda':lam if reg=='cond' else 0.,'column':columns[rule]})
    if cfg.pilot.rule=='all' and cfg.pilot.regularizer=='all':
        for subset in validity:
            specs.append({'rule':f'validity_{subset}','regularizer':'none','lambda':0.,'column':len(features)})
            features.append(torch.load(cache(cfg)/SOURCES[task]/f'menu_{subset:03d}.pt',mmap=True,weights_only=False))
    features=torch.stack(features,dim=1)
    folder=output/task/genre/f'seed{cfg.pilot.seed}'
    if cfg.pilot.rule!='all':folder=folder/(cfg.pilot.rule+'_'+cfg.pilot.regularizer)
    saved=fit(bank,features,ids,genres,specs,cfg,cfg.pilot.seed,folder)
    return {'heads':len(specs),'steps':saved['steps']}
