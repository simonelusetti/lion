"""Word-row diagnostics with paired mask cohorts and row-level uncertainty."""
import math,time
from collections import Counter
import torch
from ..diagnostics.common import write_csv,save_json,absolute,read_csv
from ..sensitivity.data import paths,source_config


# Preserve g1/g2 averaging: average eligible draws within fold, then available folds within row.
def reduce(value,eligible=None):
    if eligible is None:return value.mean(-1).mean(2)
    count=eligible.sum(-1);mean=(value*eligible[None]).sum(-1)/count[None].clamp_min(1)
    mean=mean.masked_fill(count[None]==0,torch.nan)
    return torch.nanmean(mean,dim=2)


def experiment(cfg,output):
    root=paths(cfg);source=source_config(cfg);minimum=cfg.step1b.minimum_ratio_rows
    summaries=[];per_seed=[];fills=[];distances=[];cells=[];matches=[];concentration=[];words=[];reproduction=[];started=time.monotonic()
    def record(dataset,classifier,analysis,group,bin_name,values,mask,extra=None,error=None):
        common=torch.isfinite(values).all((0,2,3));selected=mask&common;n=int(selected.sum());data=values[:,selected];seeds=data.shape[0]
        for kind_index,kind in enumerate(('selector','random')):
            for metric_index,metric in enumerate(('S','flip')):
                x=data[:,:,kind_index,metric_index];point=x.mean(1) if n else torch.full((seeds,),torch.nan)
                row_se=x.std(1)/n**.5 if n>1 else torch.full((seeds,),torch.nan)
                record={'dataset':dataset,'status':'ID' if dataset=='conll2003' else 'UNCERTIFIED','classifier':classifier,'analysis':analysis,'group':group,'bin':bin_name,'mask':kind,'metric':metric,'rows':n,'small_bin':n<50,
                        'mean':float(point.mean()),'seed_sd':float(point.std()) if seeds>1 else None,'row_se':float(x.mean(0).std()/n**.5) if n>1 else float('nan'),**(extra or {})}
                summaries.append(record)
                for s in range(seeds):per_seed.append({**record,'seed':s if seeds>1 else -1,'mean':float(point[s]),'row_se':float(row_se[s])})
        if error is not None:
            x=error[:,selected];point=x.mean(1) if n else torch.full((seeds,),torch.nan)
            summaries.append({'dataset':dataset,'status':'ID' if dataset=='conll2003' else 'UNCERTIFIED','classifier':classifier,'analysis':analysis,'group':group,'bin':bin_name,'mask':'original','metric':'error','rows':n,'small_bin':n<50,'mean':float(point.mean()),'seed_sd':float(point.std()) if seeds>1 else None,'row_se':float(x.mean(0).std()/n**.5) if n>1 else float('nan')})
    for folder in sorted((root/'generate').iterdir()):
        if not folder.is_dir():continue
        dataset=folder.name;cache=[torch.load(folder/(f+'.pt'),mmap=True,weights_only=False) for f in ('g1','g2')];meta=torch.load(root/'metadata'/(dataset+'.pt'),weights_only=False)
        changed=torch.stack([torch.stack([v[k+'_changed_count'] for k in ('selector','random')],1) for v in cache],1)
        nearest=torch.stack([torch.stack([v[k+'_nearest_changed'] for k in ('selector','random')],1) for v in cache],1)
        counts=torch.where(changed<5,changed,torch.where(changed<10,5,6));distance_bin=torch.bucketize(nearest,torch.tensor([1,2,3,5,10]))
        groups=cache[0]['group'];nrows=len(groups);word_labels=cfg.step1b.get('word_labels',False)
        group_masks={'all':torch.ones(nrows,dtype=torch.bool)} if word_labels else {'O':groups==0,'entity':groups==1,'all':torch.ones(nrows,dtype=torch.bool)}
        dropped=torch.stack([v['dropped'] for v in cache],1)
        for group,mask in group_masks.items():
            n=int(mask.sum())
            for k,kind in enumerate(('selector','random')):
                changed_rows=changed[mask,:,k].double().mean((1,2));null=(changed[mask,:,k]==0).double().mean((1,2));drop_rows=dropped[mask].double().mean(1)
                total_drop=float(dropped[mask].sum())*changed.shape[-1];restored=(float(dropped[mask].sum())*changed.shape[-1]-float(changed[mask,:,k].sum()))/total_drop if total_drop else float('nan')
                fills.append({'dataset':dataset,'group':group,'mask':kind,'rows':n,'mean_dropped':float(drop_rows.mean()),'mean_changed':float(changed_rows.mean()),'changed_row_se':float(changed_rows.std()/n**.5),'restored_fraction':restored,'null_swap_fraction':float(null.mean()),'null_swap_row_se':float(null.std()/n**.5)})
                all_dist=[]
                for values in cache:
                    length=values['word_lengths'];owner=torch.repeat_interleave(torch.arange(nrows),length);position=torch.arange(int(length.sum()))-values['word_offsets'][owner]
                    d=(position-values['focal'][owner]).abs();selected=values[kind+'_drop_words']&mask[owner];all_dist.append(d[selected])
                d=torch.cat(all_dist).double()
                distances.append({'dataset':dataset,'group':group,'mask':kind,'dropped_positions':len(d),'mean':float(d.mean()),'median':float(d.quantile(.5)) if len(d) else float('nan'),'within_two':float((d<=2).double().mean())})
        for classifier in cfg.step1b.get('classifiers',('baseline','selector_augmented','random_augmented','scorer')):
            seed_ids=[-1] if classifier=='scorer' else list(source.augmentation.seeds)
            measured=[torch.load(root/'measure'/f'{dataset}_{classifier}_{s}.pt',weights_only=False) for s in seed_ids]
            kl=torch.stack([v['kl'] for v in measured]).double();flip=torch.stack([v['flip'] for v in measured]).double();error=torch.stack([v['prediction']!=meta['truth'] for v in measured]).double()
            original=torch.stack([reduce(kl),reduce(flip)],-1)
            nonnull=torch.stack([reduce(kl,changed>0),reduce(flip,changed>0)],-1)
            for group,mask in group_masks.items():
                record(dataset,classifier,'headline',group,'all_draws',original,mask)
                excluded={f'excluded_{kind}':int((mask&~(changed[:,:,j]>0).any((1,2))).sum()) for j,kind in enumerate(('selector','random'))}
                record(dataset,classifier,'A_nonnull',group,'non_null',nonnull,mask,extra=excluded)
            if dataset=='conll2003' and classifier=='baseline':
                m=groups==1
                for j,kind in enumerate(('selector','random')):
                    actual=float(original[:,m,j,0].mean());given=.1024 if j==0 else .1283
                    old=read_csv(absolute(cfg.step1b.original_results)/'measure'/'seed_summary.csv')
                    reference=float(next(r['S_mean'] for r in old if (r['dataset'],r['classifier'],r['fold'],r['group'],r['mask'])==('conll2003','baseline','mean','entity',kind)))
                    reproduction.append({'mask':kind,'rows':int(m.sum()),'regenerated_S':actual,'step1_S':reference,'given_rounded_S':given,'difference_from_step1':actual-reference,'difference_from_given':actual-given})
            for axis,names,bin_tensor in [('changed_count',['0','1','2','3','4','5–9','10+'],counts),('nearest_changed',['1','2','3','4–5','6–10','>10'],distance_bin)]:
                for b,label in enumerate(names):
                    eligible=bin_tensor==b
                    if axis=='nearest_changed':eligible&=changed>0
                    value=torch.stack([reduce(kl,eligible),reduce(flip,eligible)],-1)
                    for group,mask in group_masks.items():record(dataset,classifier,'B_'+axis,group,label,value,mask)
            for group,mask in group_masks.items():
                weight_values=[];ratio_values=[];influences=[];total_selector=int(((changed[:,:,0]>0)&mask[:,None,None]).sum())
                for count_bin in range(1,7):
                    for dist_bin in range(6):
                        eligible=(counts==count_bin)&(distance_bin==dist_bin)&(changed>0)
                        value=reduce(kl,eligible);common=mask&torch.isfinite(value).all((0,2));n=int(common.sum())
                        a=value[:,common,0];b=value[:,common,1];mass=int((eligible[:,:,0]&mask[:,None,None]).sum())
                        valid=n>=minimum and bool((b.mean(1)>0).all())
                        r=a.mean(1)/b.mean(1) if valid else torch.full((len(seed_ids),),torch.nan)
                        cells.append({'dataset':dataset,'classifier':classifier,'group':group,'count_bin':count_bin,'distance_bin':dist_bin,'paired_rows':n,'small_bin':n<50,'selector_draws':mass,'ratio_mean':float(r.mean()),'ratio_seed_sd':float(r.std()) if len(seed_ids)>1 else None,'supported':valid})
                        if valid:
                            weight_values.append(mass);ratio_values.append(r)
                            influence=torch.zeros(len(seed_ids),nrows,dtype=torch.float64);influence[:,common]=(a-r[:,None]*b)/(b.mean(1)[:,None]*n);influences.append(influence)
                mass=sum(weight_values)
                if mass:
                    w=torch.tensor(weight_values,dtype=torch.float64)/mass;r=(torch.stack(ratio_values,1)*w[None]).sum(1);inf=(torch.stack(influences,1)*w[None,:,None]).sum(1).mean(0)
                    se=float((inf.square().sum()*nrows/max(1,nrows-1)).sqrt())
                else:r=torch.full((len(seed_ids),),torch.nan);se=float('nan')
                matches.append({'dataset':dataset,'classifier':classifier,'group':group,'matched_ratio_mean':float(r.mean()),'matched_ratio_seed_sd':float(r.std()) if len(seed_ids)>1 else None,'ratio_row_se_fixed_weights':se,'supported_cells':len(weight_values),'selector_nonnull_mass_retained':mass/total_selector if total_selector else float('nan')})
            categories=('unseen_word','seen_other_label','seen_same_label') if word_labels else ('unseen','seen_different_type','seen_same_type','no_gold_entity')
            for index,label in enumerate(categories):
                group='all' if word_labels else 'entity'
                record(dataset,classifier,'C_word_seen' if word_labels else 'C_entity_seen',group,label,original,group_masks[group]&(meta['seen']==index),error=error)
            for group in (('all',) if word_labels else ('O','entity')):
                for index,label in enumerate(('unseen_word','one_type','two_or_more_types')):
                    mask=group_masks[group]&(meta['ambiguity']==index)
                    record(dataset,classifier,'D_ambiguity',group,label,original,mask,extra={'share_of_group':float(mask.sum()/group_masks[group].sum())})
            if word_labels:
                for index,label in enumerate(meta['labels']):
                    record(dataset,classifier,'gold_label','all',label,original,meta['truth']==index,error=error)
            # Each ranking defines the same top rows for both masks, satisfying paired symmetry.
            row_s=reduce(kl);row_f=reduce(flip)
            for group,mask in group_masks.items():
                ids=mask.nonzero().flatten();n=len(ids)
                for rank_kind,rank_name in enumerate(('selector','random')):
                    for fraction in (.01,.05,.10):
                        number=max(1,math.ceil(n*fraction));top=torch.zeros(len(seed_ids),n,dtype=torch.bool)
                        top.scatter_(1,row_s[:,ids,rank_kind].argsort(dim=1,descending=True,stable=True)[:,:number],True)
                        f=row_f[:,ids];den=f.sum(1);share=(f*top[:,:,None]).sum(1)/den
                        influence=(f*top[:,:,None]-share[:,None]*f)/den[:,None]
                        for j,kind in enumerate(('selector','random')):
                            se=float((influence[:,:,j].mean(0).square().sum()*n/max(n-1,1)).sqrt())
                            concentration.append({'dataset':dataset,'classifier':classifier,'group':group,'ranking':rank_name,'top_fraction':fraction,'top_rows':number,'small_bin':number<50,'mask':kind,'flip_share_mean':float(share[:,j].mean()),'flip_share_seed_sd':float(share[:,j].std()) if len(seed_ids)>1 else None,'row_se_fixed_ranking':se})
                    for seed_index,seed in [(-1,'mean')]+list(enumerate(seed_ids)):
                        score=row_s[:,ids,rank_kind].mean(0) if seed_index<0 else row_s[seed_index,ids,rank_kind]
                        top_ids=ids[score.argsort(descending=True,stable=True)[:max(1,math.ceil(n*.01))]]
                        for rank,(form,count) in enumerate(Counter(meta['forms'][i] for i in top_ids.tolist()).most_common(20),1):words.append({'dataset':dataset,'classifier':classifier,'group':group,'ranking':rank_name,'seed':seed,'rank':rank,'form':form,'count':count})
            print(f'Analysed {dataset}/{classifier}',flush=True)
        for name,values in [('summary',summaries),('per_seed',per_seed),('A_fills',fills),('B_distance',distances),('B_cells',cells),('B_matched',matches),('D_concentration',concentration),('D_words',words),('reproduction',reproduction)]:
            if values:write_csv(output/(name+'.csv'),values)
    save_json(output/'timing.json',{'analysis_seconds':time.monotonic()-started});return {'summaries':len(summaries)}
