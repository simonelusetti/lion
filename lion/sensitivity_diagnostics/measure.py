"""Retain per-draw KL/flips on the single regenerated fill set."""
import time
import torch
from tagger.ensemble import IndependentReadouts
from ..partition.train import load_model
from ..certification.data import task_names
from ..diagnostics.common import absolute,write_csv,save_json
from ..sensitivity.data import paths,source_config,source_root
from ..sensitivity.measure import probabilities,ARMS


@torch.no_grad()
def per_draw(model,raw,draws,source,gamma=.01):
    kl=[];flip=[];prediction=[]
    for start in range(0,len(raw),512):
        original=raw[start:start+512];sampled=draws[start:start+512];n,r,d=sampled.shape
        p=probabilities(model,original,source,gamma);q=probabilities(model,sampled.flatten(0,1),source,gamma).reshape(n,r,p.shape[1],-1)
        same=(sampled==original[:,None]).all(-1).to(q.device);q[same]=p[:,None].expand_as(q)[same]
        kl.append((p[:,None]*(p[:,None].log()-q.log())).sum(-1).clamp_min(0).cpu());flip.append((q.argmax(-1)!=p.argmax(-1)[:,None]).double().cpu());prediction.append(p.argmax(-1).cpu())
    return torch.cat(kl),torch.cat(flip),torch.cat(prediction)


@torch.no_grad()
def experiment(cfg,output):
    source=source_config(cfg);root=paths(cfg);started=time.monotonic()
    for folder in sorted((root/'generate').iterdir()):
        if not folder.is_dir():continue
        dataset=folder.name;raw=torch.load(folder/'raw.pt',mmap=True,weights_only=False)['features'][:,0]
        cache=[torch.load(folder/(fold+'.pt'),mmap=True,weights_only=False) for fold in ('g1','g2')]
        for kind,seed in [('scorer',-1)]+[('classifiers',s) for s in source.augmentation.seeds]:
            baseline_only=cfg.step1b.get('classifiers')==['baseline','scorer']
            names=['scorer'] if kind=='scorer' else (['baseline'] if baseline_only else ARMS)
            if all((output/f'{dataset}_{name}_{seed}.pt').exists() for name in names):continue
            if kind=='scorer':model=load_model(source_root(cfg)/'scorer'/task_names(source)[0],source)
            else:
                saved=torch.load(absolute(cfg.sensitivity.augmentation_run)/'train'/f'seed{seed}'/'model.pt',weights_only=False,map_location='cpu')
                classes=saved['labels'] if isinstance(saved['labels'],int) else len(saved['labels'])
                model=IndependentReadouts(source.tagger,768,classes,saved.get('heads',3)).to(source.runtime.device);model.load_state_dict(saved['state'])
            model.eval().requires_grad_(False);ks=[];fs=[]
            for values in cache:
                k=[];f=[]
                for mask in ('selector','random'):
                    a,b,p=per_draw(model,raw,values[mask],source,cfg.sensitivity.get('gamma',.01));k.append(a);f.append(b)
                ks.append(torch.stack(k,1));fs.append(torch.stack(f,1))
            k=torch.stack(ks,1);f=torch.stack(fs,1)
            for i,name in enumerate(names):torch.save({'kl':k[...,i].float(),'flip':f[...,i].float(),'prediction':p[:,i],'group':cache[0]['group']},output/f'{dataset}_{name}_{seed}.pt')
            print(f'Regenerated-fill measurement: {dataset}/{kind}/{seed}',flush=True)
    save_json(output/'timing.json',{'measurement_seconds':time.monotonic()-started});return {'measurement_files':len(list(output.glob('*.pt')))}
