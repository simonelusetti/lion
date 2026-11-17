"""Document-held-out E3 probes, violation statistics and paired comparisons."""
from collections import defaultdict
import torch
from omegaconf import OmegaConf
from ..diagnostics.common import write_csv
from ..invariance.data import banks,rotations,paths
from ..invariance.analyze import bootstrap,metrics,summary
from .probes import conditional,bootstrap as id_bootstrap
from .train import POLICIES


def dependence(predicted,pool,eligible,inverse,weights):
    labels,genres=eligible.shape;heads=predicted.shape[1];docs=weights.shape[1]
    supported=eligible.sum(-1)>=2
    allowed=supported[pool['y']]&eligible[pool['y'],pool['z']]
    cells=(inverse[allowed]*labels+pool['y'][allowed])*genres+pool['z'][allowed]
    correct=(predicted[allowed]==pool['z'][allowed,None]).float()
    totals=torch.bincount(cells,minlength=docs*labels*genres).reshape(docs,labels,genres).float()
    successes=torch.zeros(docs*labels*genres,heads).index_add_(0,cells,correct).reshape(docs,labels,genres,heads)
    n=torch.einsum('bd,dyg->byg',weights,totals)
    c=torch.einsum('bd,dygh->bygh',weights,successes)
    rates=c/n[:,:,:,None].clamp_min(1)
    label_score=(rates*eligible[None,:,:,None]).sum(2)/eligible.sum(-1)[None,:,None].clamp_min(1)
    value=label_score[:,supported].mean(1)
    valid=((n>0)|~eligible[None]).all((1,2))
    value[~valid]=torch.nan
    return value


def violation(logp,y,d,covered,inverse,weights):
    """Recompute weighted, label-centered covariance using document sufficient statistics."""
    docs=weights.shape[1];labels=logp.shape[-1];heads=logp.shape[1];dimension=d.shape[-1]
    cell=inverse*labels+y
    counts=torch.bincount(cell,minlength=docs*labels).reshape(docs,labels).float()
    sums=torch.zeros(docs*labels,heads,labels).index_add_(0,cell,logp).reshape(docs,labels,heads,labels)
    cover_count=torch.bincount(cell[covered],minlength=docs*labels).reshape(docs,labels).float()
    cover_d=torch.zeros(docs*labels,dimension).index_add_(0,cell[covered],d[covered]).reshape(docs,labels,dimension)
    cover_p=torch.zeros(docs,heads,labels).index_add_(0,inverse[covered],logp[covered])
    product=torch.zeros(docs,heads,labels,dimension)
    for doc in range(docs):
        mask=covered&(inverse==doc)
        product[doc]=torch.einsum('nhl,nd->hld',logp[mask],d[mask])
    n=weights@cover_count.sum(-1);label_counts=weights@counts
    label_means=torch.einsum('bd,dyhl->byhl',weights,sums)/label_counts[:,:,None,None].clamp_min(1)
    ds=torch.einsum('bd,dyk->byk',weights,cover_d)
    cross=torch.einsum('bd,dhkl->bhkl',weights,product)-torch.einsum('byhl,byk->bhlk',label_means,ds)
    rsum=torch.einsum('bd,dhl->bhl',weights,cover_p)-torch.einsum('byhl,by->bhl',label_means,weights@cover_count)
    covariance=cross/n[:,None,None,None].clamp_min(1)-(rsum/n[:,None,None].clamp_min(1))[:,:,:,None]*(ds.sum(1)/n[:,None].clamp_min(1))[:,None,None,:]
    value=covariance.square().sum((-1,-2));value[n<2]=torch.nan
    return value


def ranks(value):
    order=value.argsort(dim=-1,stable=True);ordered=value.gather(-1,order)
    start=torch.ones_like(ordered,dtype=torch.bool);start[...,1:]=ordered[...,1:]!=ordered[...,:-1]
    groups=start.long().cumsum(-1)-1
    rank=torch.arange(value.shape[-1],dtype=value.dtype).expand_as(value)
    sums=torch.zeros_like(value).scatter_add_(-1,groups,rank);counts=torch.zeros_like(value).scatter_add_(-1,groups,torch.ones_like(value))
    sorted_rank=(sums/counts.clamp_min(1)).gather(-1,groups)
    result=torch.empty_like(value).scatter_(-1,order,sorted_rank)
    return result


def spearman(x,y):
    x=ranks(x);y=ranks(y);x=x-x.mean(-1,keepdim=True);y=y-y.mean(-1,keepdim=True)
    return (x*y).sum(-1)/(x.square().sum(-1)*y.square().sum(-1)).sqrt()


def experiment(cfg,output):
    results,_=paths(cfg);summaries=[];models=[];selection_rows=[];probe_skips=[];correlations=[]
    all_observations=defaultdict(list);all_seed_observations=defaultdict(list)
    probe_cfg=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True));probe_cfg.train.epochs=cfg.zfree.probe_epochs
    for name,bank in banks(cfg):
        collected=defaultdict(list);rank_values=defaultdict(list)
        for rotation_index,(genre,genres,validation,test) in enumerate(rotations(name,bank,cfg)):
            folder=results/'train'/name/genre;context=torch.load(results/'contexts'/name/genre/'context.pt',weights_only=False)
            rotation_values=defaultdict(list);rotation_ranks=defaultdict(list)
            for seed in cfg.diagnostics.seeds:
                checkpoint=torch.load(folder/f'seed{seed}'/'model.pt',map_location='cpu',mmap=True,weights_only=False)
                saved=torch.load(folder/f'seed{seed}'/'predictions.pt',map_location='cpu',mmap=True,weights_only=False);specs=checkpoint['heads'];heads=len(specs)
                ids=saved['id']['id']['index'];y=bank['y'][ids]
                inverse,weights=id_bootstrap(bank,ids,cfg.invariance.bootstrap_seed+rotation_index,cfg.invariance.bootstrap)
                pool={'y':y,'z':bank['z'][ids],'sentence':context['sentence'][ids],'document':[bank['document'][i] for i in ids.tolist()]}
                features=torch.cat([saved[policy]['id']['log_probability'] for policy in POLICIES],dim=1)
                val_ids=saved['id']['id_validation']['index']
                val_features=torch.cat([saved[policy]['id_validation']['log_probability'] for policy in POLICIES],dim=1)
                val_pool={'y':bank['y'][val_ids],'z':bank['z'][val_ids],'sentence':context['sentence'][val_ids]}
                probe_output=results/'probes'/name/genre/f'seed{seed}'
                if cfg.invariance.resume and (probe_output/'oof.pt').exists():
                    cached=torch.load(probe_output/'oof.pt',weights_only=False);predicted,eligible,fold_count=cached['predicted'],cached['eligible'],cached['folds']
                else:predicted,eligible,fold_count=conditional(features,val_features,pool,val_pool,bank['labels'],bank['genres'],probe_cfg,seed,cfg.diagnostics.sample_seed+rotation_index,probe_output)
                dep=dependence(predicted,pool,eligible,inverse,weights)
                vio=violation(features,y,context['residual'][ids],context['covered'][ids],inverse,weights)
                probe_skips.append({'dataset':name,'test_genre':genre,'seed':seed,'document_folds':fold_count,'eligible_labels':int((eligible.sum(-1)>=2).sum()),'total_labels':len(bank['labels']),'supported_tokens':int(((eligible.sum(-1)>=2)[y]&eligible[y,pool['z']]).sum()),'id_tokens':len(ids)})
                for p,policy in enumerate(POLICIES):
                    data=saved[policy];chosen=[0]+[checkpoint['selected'][policy][rule] for rule in ('Rz','Rg','Rrand')]
                    ood=data['ood'];oid=ood['index'];oinverse,oweights=bootstrap(bank,oid,cfg.invariance.bootstrap_seed+rotation_index,cfg.invariance.bootstrap)
                    scores=metrics(ood['prediction'][:,chosen].long(),bank['y'][oid],torch.ones(len(oid),dtype=torch.bool),oinverse,oweights,len(bank['labels']),cfg.invariance.head_chunk)
                    id_scores=metrics(data['id']['prediction'][:,chosen].long(),y,torch.ones(len(y),dtype=torch.bool),inverse,weights,len(bank['labels']),cfg.invariance.head_chunk)
                    selected_dep=dep[:,[p*heads+h for h in chosen]];selected_vio=vio[:,[p*heads+h for h in chosen]]
                    for metric,value in [('ood_accuracy',scores['accuracy']),('ood_macro_f1',scores['macro_f1']),('id_accuracy',id_scores['accuracy']),('genre_dependence',selected_dep),('zfree_violation',selected_vio)]:
                        for h,rule in enumerate(('R0','Rz','Rg','Rrand')):rotation_values[(policy,'score',rule,metric)].append(value[:,h])
                    rotation_values[(policy,'primary','Rz_minus_R0','genre_dependence')].append(selected_dep[:,1]-selected_dep[:,0])
                    for label,a,b in [('Rz_minus_R0',1,0),('Rz_minus_Rrand',1,3),('Rz_minus_Rg',1,2)]:rotation_values[(policy,'primary',label,'ood_accuracy')].append(scores['accuracy'][:,a]-scores['accuracy'][:,b])
                    for i,h in enumerate(chosen):
                        selection_rows.append({'dataset':name,'test_genre':genre,'seed':seed,'policy':policy,'rule':('R0','Rz','Rg','Rrand')[i],'lambda':specs[h]['lambda'],'best_step':int(checkpoint['best_step'][p,h]),'validation_macro_f1':float(checkpoint['validation_macro_f1'][p,h])})
                    for h,spec in enumerate(specs):
                        models.append({'dataset':name,'test_genre':genre,'seed':seed,'policy':policy,'head':h,**spec,
                                       'selected':h in chosen,'ood_accuracy':float(ood['accuracy'][h]),'ood_macro_f1':float(ood['macro_f1'][h]),'id_accuracy':float(data['id']['accuracy'][h]),
                                       'genre_dependence':float(dep[0,p*heads+h]),'zfree_violation':float(vio[0,p*heads+h])})
                    rotation_ranks[policy].append(torch.stack([vio[:,p*heads:(p+1)*heads],dep[:,p*heads:(p+1)*heads]],dim=-1))
            for key,values in rotation_values.items():collected[key].append(torch.stack(values))
            for policy,values in rotation_ranks.items():rank_values[policy].append(torch.stack(values))
        for key,values in collected.items():
            policy,kind,comparison,metric=key
            summaries.append({'dataset':name,'policy':policy,'kind':kind,'comparison':comparison,'metric':metric,**summary(torch.stack(values,dim=1))})
        for policy,values in rank_values.items():
            # [rotation,seed,observed+bootstrap,head,(violation,dependence)].
            data=torch.stack(values)
            observations=data.permute(2,0,1,3,4).flatten(1,3)
            valid=torch.isfinite(observations).all((1,2));rho=torch.full((len(valid),),torch.nan)
            rho[valid]=spearman(observations[valid,:,0],observations[valid,:,1])
            seed_data=data[:,:,0].permute(1,0,2,3).flatten(1,2)
            all_observations[policy].append(observations);all_seed_observations[policy].append(seed_data)
            seed_rho=spearman(seed_data[:,:,0],seed_data[:,:,1])
            boot=rho[1:]
            correlations.append({'dataset':name,'policy':policy,'models':int(observations.shape[1]),'spearman':float(rho[0]),'seed_sd':float(seed_rho.std()),'ci_low':float(torch.nanquantile(boot,.025)),'ci_high':float(torch.nanquantile(boot,.975)),'valid_bootstraps':int(torch.isfinite(boot).sum())})
        # Keep compact final tables up to date at task boundaries.
        for filename,rows in [('summary',summaries),('models',models),('selections',selection_rows),('probe_support',probe_skips),('correlations',correlations)]:write_csv(output/(filename+'.csv'),rows)
    for policy,values in all_observations.items():
        observations=torch.cat(values,dim=1);valid=torch.isfinite(observations).all((1,2));rho=torch.full((len(valid),),torch.nan)
        rho[valid]=spearman(observations[valid,:,0],observations[valid,:,1])
        seed_data=torch.cat(all_seed_observations[policy],dim=1);seed_rho=spearman(seed_data[:,:,0],seed_data[:,:,1]);boot=rho[1:]
        correlations.append({'dataset':'all_tasks','policy':policy,'models':int(observations.shape[1]),'spearman':float(rho[0]),'seed_sd':float(seed_rho.std()),'ci_low':float(torch.nanquantile(boot,.025)),'ci_high':float(torch.nanquantile(boot,.975)),'valid_bootstraps':int(torch.isfinite(boot).sum())})
    write_csv(output/'correlations.csv',correlations)
    return {'models':len(models),'summary_rows':len(summaries)}
