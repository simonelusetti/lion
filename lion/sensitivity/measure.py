"""KL and flips of frozen existing classifiers; uncertainty is over word rows."""
import time
import torch
from ..runtime import autocast
from ..partition.train import load_model
from ..diagnostics.common import absolute,write_csv,save_json,read_csv
from tagger.ensemble import IndependentReadouts
from .data import paths,source_root,source_config

ARMS=['baseline','selector_augmented','random_augmented']


def moments(value):
    return float(value.mean()),float(value.std()/len(value)**.5) if len(value)>1 else float('nan')


@torch.no_grad()
def probabilities(model,features,source,gamma):
    with autocast(source):logits=model(features.float().to(source.runtime.device)[:,None])
    return (1-gamma)*logits.double().softmax(-1)+gamma/logits.shape[-1]


@torch.no_grad()
def row_measures(model,raw,draws,source,gamma):
    kl=[];flips=[];batch_size=512
    for start in range(0,len(raw),batch_size):
        original=raw[start:start+batch_size];sampled=draws[start:start+batch_size];n,r,d=sampled.shape
        p=probabilities(model,original,source,gamma);q=probabilities(model,sampled.flatten(0,1),source,gamma).reshape(n,r,p.shape[1],-1)
        same=(sampled==original[:,None]).all(-1).to(q.device)
        q[same]=p[:,None].expand_as(q)[same]
        kl.append((p[:,None]*(p[:,None].log()-q.log())).sum(-1).clamp_min(0).mean(1).cpu())
        flips.append((q.argmax(-1)!=p.argmax(-1)[:,None]).double().mean(1).cpu())
    return torch.cat(kl),torch.cat(flips)


def references(cfg):
    rows=read_csv(absolute(cfg.sensitivity.augmentation_run)/'report'/'enriched_bounds.csv')
    return {(r['selector'],r['group']):float(r['absolute_ce_upper_bound']) for r in rows}


def reference(bound,fold,group,groups):
    def one(f,g):
        if g=='all':
            w=float((groups==0).double().mean());return w*bound[(f,'O')]+(1-w)*bound[(f,'entity')]
        return bound[(f,g)]
    return sum(one(f,group) for f in ('g1','g2'))/2 if fold=='mean' else one(fold,group)


def experiment(cfg,output):
    source=source_config(cfg);gamma=source.certification.gamma;root=paths(cfg);bound=references(cfg);records=[];ratios=[];diagnostics=[];aggregate=[];started=time.monotonic()
    for folder in sorted((root/'generate').iterdir()):
        if not folder.is_dir():continue
        dataset=folder.name;raw=torch.load(folder/'raw.pt',mmap=True,weights_only=False)['features'][:,0]
        cache=[torch.load(folder/(fold+'.pt'),mmap=True,weights_only=False) for fold in ('g1','g2')];groups=cache[0]['group']
        for fold,values in zip(('g1','g2'),cache):
            for label,index in [('O',0),('entity',1),('all',-1)]:
                mask=torch.ones(len(groups),dtype=torch.bool) if index<0 else groups==index
                diagnostics.append({'dataset':dataset,'fold':fold,'group':label,'rows':int(mask.sum()),'mean_dropped_words':float(values['dropped'][mask].double().mean()),
                                    'same_set_share':float(values['coincide'][mask].double().mean()),'all_context_dropped_share':float(((values['dropped']==values['available'])&(values['available']>0))[mask].double().mean()),
                                    'no_drop_share':float((values['dropped'][mask]==0).double().mean())})
        seed_values={}
        configurations=[('scorer',-1)]+[('classifiers',seed) for seed in source.augmentation.seeds]
        for kind,seed in configurations:
            if kind=='scorer':model=load_model(source_root(cfg)/'scorer'/'ner',source);names=['scorer']
            else:
                saved=torch.load(absolute(cfg.sensitivity.augmentation_run)/'train'/f'seed{seed}'/'model.pt',map_location='cpu',weights_only=False)
                model=IndependentReadouts(source.tagger,768,len(saved['labels']),3).to(source.runtime.device);model.load_state_dict(saved['state']);names=ARMS
            model.eval().requires_grad_(False);kls=[];flips=[]
            for values in cache:
                fk=[];ff=[]
                for mask_kind in ('selector','random'):
                    k,f=row_measures(model,raw,values[mask_kind],source,gamma);fk.append(k);ff.append(f)
                kls.append(torch.stack(fk,1));flips.append(torch.stack(ff,1))
            # [word,fold,kind,head], with fold means taken within word before SE.
            kl=torch.stack(kls,1);flip=torch.stack(flips,1)
            for h,name in enumerate(names):
                torch.save({'kl':kl[:,:,:,h].float(),'flip':flip[:,:,:,h].float(),'group':groups,'sentence':cache[0]['sentence'],'focal':cache[0]['focal']},output/f'{dataset}_{name}_seed{seed}_rows.pt')
                seed_values.setdefault(name,[]).append((kl[:,:,:,h],flip[:,:,:,h]))
                for fold_index,fold in enumerate(('g1','g2','mean')):
                    ks=kl[:,:,:,h].mean(1) if fold=='mean' else kl[:,fold_index,:,h]
                    fs=flip[:,:,:,h].mean(1) if fold=='mean' else flip[:,fold_index,:,h]
                    for label,index in [('O',0),('entity',1),('all',-1)]:
                        mask=torch.ones(len(groups),dtype=torch.bool) if index<0 else groups==index
                        a,b=ks[mask,0],ks[mask,1];ratio=float(a.mean()/b.mean()) if b.mean()>0 else float('nan')
                        ratio_se=float(((a-ratio*b)/b.mean()).std()/len(a)**.5) if b.mean()>0 and len(a)>1 else float('nan')
                        ratios.append({'dataset':dataset,'classifier':name,'seed':seed,'fold':fold,'group':label,'ratio_selector_random':ratio,'ratio_row_se':ratio_se})
                        ref=reference(bound,fold,label,groups)
                        for j,mask_kind in enumerate(('selector','random')):
                            km,kse=moments(ks[mask,j]);fm,fse=moments(fs[mask,j])
                            records.append({'dataset':dataset,'status':'in_domain_reference' if dataset=='conll2003' else 'UNCERTIFIED','classifier':name,'seed':seed,'fold':fold,'mask':mask_kind,'group':label,'rows':int(mask.sum()),
                                            'S':km,'S_row_se':kse,'flip':fm,'flip_row_se':fse,'ID_label_reference':ref,'reference_below_one':ref<1})
            print(f'Measured {dataset}/{kind}/seed{seed}',flush=True)
        for name,values in seed_values.items():
            k=torch.stack([v[0] for v in values]);f=torch.stack([v[1] for v in values])
            for fold_index,fold in enumerate(('g1','g2','mean')):
                ks=k.mean(2) if fold=='mean' else k[:,:,fold_index];fs=f.mean(2) if fold=='mean' else f[:,:,fold_index]
                for label,index in [('O',0),('entity',1),('all',-1)]:
                    mask=torch.ones(len(groups),dtype=torch.bool) if index<0 else groups==index
                    ratio=ks[:,mask,0].mean(1)/ks[:,mask,1].mean(1)
                    for j,mask_kind in enumerate(('selector','random')):
                        seed_k=ks[:,mask,j].mean(1);seed_f=fs[:,mask,j].mean(1)
                        _,kse=moments(ks[:,mask,j].mean(0));_,fse=moments(fs[:,mask,j].mean(0))
                        aggregate.append({'dataset':dataset,'status':'in_domain_reference' if dataset=='conll2003' else 'UNCERTIFIED','classifier':name,'seeds':len(values),'fold':fold,'group':label,'mask':mask_kind,'rows':int(mask.sum()),
                                          'S_mean':float(seed_k.mean()),'S_seed_sd':float(seed_k.std()) if len(values)>1 else None,'S_mean_row_se':kse,
                                          'flip_mean':float(seed_f.mean()),'flip_seed_sd':float(seed_f.std()) if len(values)>1 else None,'flip_mean_row_se':fse,
                                          'ratio_mean':float(ratio.mean()),'ratio_seed_sd':float(ratio.std()) if len(values)>1 else None,'ID_label_reference':reference(bound,fold,label,groups)})
        for filename,rows in [('per_classifier_seed',records),('ratios',ratios),('mask_diagnostics',diagnostics),('seed_summary',aggregate)]:write_csv(output/(filename+'.csv'),rows)
    save_json(output/'timing.json',{'measurement_seconds':time.monotonic()-started});return {'measurements':len(records)}
