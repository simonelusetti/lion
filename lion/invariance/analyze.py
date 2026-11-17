"""Paired rule comparisons, fixed lookup routing and document-block intervals."""
from collections import defaultdict

import torch

from ..diagnostics.common import write_csv
from .data import banks,rotations,paths,random_types


def bootstrap(bank,ids,seed,count):
    names=sorted({bank['document'][i] for i in ids.tolist()});lookup={d:i for i,d in enumerate(names)}
    inverse=torch.tensor([lookup[bank['document'][i]] for i in ids.tolist()])
    generator=torch.Generator().manual_seed(seed)
    weights=torch.zeros(count+1,len(names));weights[0]=1
    draws=torch.randint(len(names),(count,len(names)),generator=generator)
    weights[1:].scatter_add_(1,draws,torch.ones_like(draws,dtype=torch.float32))
    return inverse,weights


def metrics(prediction,y,mask,inverse,weights,classes,chunk):
    prediction,y,inverse=prediction[mask],y[mask],inverse[mask]
    output={k:torch.full((weights.shape[0],prediction.shape[1]),torch.nan) for k in ('accuracy','macro_f1')}
    if not len(y):return output
    for start in range(0,prediction.shape[1],chunk):
        p=prediction[:,start:start+chunk].long();n=p.shape[1]
        cells=inverse[:,None]*n*classes*classes+torch.arange(n)[None]*classes*classes+y[:,None]*classes+p
        counts=torch.bincount(cells.flatten(),minlength=weights.shape[1]*n*classes*classes).reshape(weights.shape[1],n,classes,classes).float()
        confusion=torch.einsum('bd,dckl->bckl',weights,counts)
        correct=confusion.diagonal(dim1=-2,dim2=-1);wanted=confusion.sum(-1);predicted=confusion.sum(-2)
        total=wanted.sum(-1)
        output['accuracy'][:,start:start+n]=correct.sum(-1)/total
        f1=(2*correct/(wanted+predicted).clamp_min(1)).mean(-1)
        output['macro_f1'][:,start:start+n]=torch.where(total>0,f1,torch.nan)
    return output


def summary(values):
    # [seed,rotation,observed+bootstrap]; retain a fixed set of supported rotations.
    valid=torch.isfinite(values[:,:,0]).all(0);values=values[:,valid]
    if not valid.any():return {'mean':float('nan'),'seed_sd':float('nan'),'ci_low':float('nan'),'ci_high':float('nan'),'rotations':0,'valid_bootstraps':0}
    seed=values.mean(1);boot=seed[:,1:].mean(0)
    return {'mean':float(seed[:,0].mean()),'seed_sd':float(seed[:,0].std()),
            'ci_low':float(torch.nanquantile(boot,.025)),'ci_high':float(torch.nanquantile(boot,.975)),
            'rotations':int(valid.sum()),'valid_bootstraps':int(torch.isfinite(boot).sum())}


def experiment(cfg,output):
    results,_=paths(cfg)
    curves,selections,raw,primary,rank_rows,shift_rows,group_shift_rows=[],[],[],[],[],[],[]
    model_names=['R0','R1','R2','R3','R4']+[f'R5_{d}' for d in range(cfg.invariance.random_draws)]+['routing','ensemble']+[f'random_routing_{d}' for d in range(cfg.invariance.random_draws)]
    for name,bank in banks(cfg):
        collected=defaultdict(list)
        rotation_list=list(rotations(name,bank,cfg))
        for rot_index,(genre,train_genres,validation,test) in enumerate(rotation_list):
            folder=results/'train'/name/genre;plan=torch.load(folder/'plan.pt',weights_only=False)
            for label,score in enumerate(plan['label_shift']):shift_rows.append({'dataset':name,'test_genre':genre,'label':bank['labels'][label],'training_JS_nats':float(score)})
            for threshold,groups in plan['groups'].items():
                count=torch.zeros(3,len(bank['labels'])).index_put_((groups[bank['type'][plan['training_tokens']]],bank['y'][plan['training_tokens']]),torch.ones(int(plan['training_tokens'].sum())),accumulate=True)
                for group,label in enumerate(('unambiguous','ambiguous','rare_unseen')):
                    score=(count[group]*plan['label_shift']).sum()/count[group].sum()
                    group_shift_rows.append({'dataset':name,'test_genre':genre,'threshold':threshold,'group':label,'training_token_weighted_JS':float(score),'training_tokens':int(count[group].sum())})
            rotation_values=defaultdict(list)
            for seed in cfg.diagnostics.seeds:
                checkpoint=torch.load(folder/f'seed{seed}'/'model.pt',map_location='cpu',mmap=True,weights_only=False)
                predictions=torch.load(folder/f'seed{seed}'/'predictions.pt',map_location='cpu',mmap=True,weights_only=False)
                selected=checkpoint['selected'];specs=checkpoint['heads']
                for key,head in selected.items():
                    selections.append({'dataset':name,'test_genre':genre,'seed':seed,'rule':key[0],'threshold':key[1],'draw':key[2],
                                       'selected_lambda':specs[head]['lambda'],'selected_epoch':int(checkpoint['best_epoch'][head]),'validation_macro_f1':float(checkpoint['validation_macro_f1'][head])})
                ids=predictions['ood']['index'];y=bank['y'][ids];inverse,weights=bootstrap(bank,ids,cfg.invariance.bootstrap_seed+rot_index,cfg.invariance.bootstrap)
                for threshold,groups in plan['groups'].items():
                    indices=[0,selected[('R1',-1.,-1)],selected[('R2',-1.,-1)],selected[('R3',threshold,-1)],selected[('R4',threshold,-1)]]+[selected[('R5',threshold,d)] for d in range(cfg.invariance.random_draws)]
                    for rule,draw in [('R1',-1),('R2',-1),('R3',-1),('R4',-1),*[('R5',d) for d in range(cfg.invariance.random_draws)]]:
                        candidates=[0]+[i for i,h in enumerate(specs) if h['rule']==rule and h['draw']==draw and (h['threshold']==threshold or rule in {'R1','R2'})]
                        chosen=selected[(rule,-1. if rule in {'R1','R2'} else threshold,draw)]
                        for i in candidates:
                            curves.append({'dataset':name,'test_genre':genre,'seed':seed,'threshold':threshold,'rule':rule,'draw':draw,
                                           'lambda':specs[i]['lambda'],'ood_accuracy':float(predictions['ood']['accuracy'][i]),'ood_macro_f1':float(predictions['ood']['macro_f1'][i]),
                                           'id_accuracy':float(predictions['id']['accuracy'][i]),'validation_macro_f1':float(checkpoint['validation_macro_f1'][i]),'selected':i==chosen})
                    val=predictions['validation'];vmap={h:i for i,h in enumerate(val['probability_heads'])}
                    pmap={h:i for i,h in enumerate(predictions['ood']['probability_heads'])}
                    p1=predictions['ood']['probability'][:,pmap[indices[1]]];p2=predictions['ood']['probability'][:,pmap[indices[2]]]
                    v1=val['probability'][:,vmap[indices[1]]].argmax(-1);v2=val['probability'][:,vmap[indices[2]]].argmax(-1)
                    vy=bank['y'][val['index']];vg=groups[bank['type'][val['index']]]
                    choices=[]
                    for group in range(3):
                        mask=vg==group
                        if not mask.any():mask=torch.ones_like(mask)
                        choices.append(bool((v1[mask]==vy[mask]).float().mean()>=(v2[mask]==vy[mask]).float().mean()))
                    anti=torch.tensor(choices)[groups]
                    route=anti[bank['type'][ids]]
                    random_predictions=[];random_assignments=[]
                    target=int(plan['frequency'][anti].sum())
                    for draw in range(cfg.invariance.random_draws):
                        assignment=random_types(plan['frequency'],target,torch.Generator().manual_seed(cfg.diagnostics.sample_seed+100000+draw))
                        random_assignments.append(assignment)
                        pick=assignment[bank['type'][ids]]
                        random_predictions.append(torch.where(pick[:,None],p1,p2).argmax(-1))
                    columns=[predictions['ood']['prediction'][:,i].long() for i in indices]
                    columns+=[torch.where(route[:,None],p1,p2).argmax(-1),((p1+p2)/2).argmax(-1),*random_predictions]
                    matrix=torch.stack(columns,dim=1)
                    token_group=groups[bank['type'][ids]]
                    shift=torch.bucketize(plan['label_shift'][y],plan['shift_cuts'])
                    scores={}
                    for stratum in (-1,0,1,2):
                        smask=torch.ones(len(y),dtype=torch.bool) if stratum==-1 else shift==stratum
                        for group in (-1,0,1,2,3):
                            mask=smask if group==-1 else smask&(token_group!=0 if group==3 else token_group==group)
                            measured=metrics(matrix,y,mask,inverse,weights,len(bank['labels']),cfg.invariance.head_chunk)
                            scores[(stratum,group)]=measured
                            if stratum==-1:
                                for m,label in enumerate(model_names):
                                    for metric,values in measured.items():
                                        rotation_values[(threshold,'score',label,group,metric)].append(values[:,m])
                                    raw.append({'dataset':name,'test_genre':genre,'seed':seed,'threshold':threshold,'model':label,'group':group,'tokens':int(mask.sum()),
                                                'accuracy':float(measured['accuracy'][0,m]),'macro_f1':float(measured['macro_f1'][0,m])})
                                for label,a,b in (('R5_mean',5,5+cfg.invariance.random_draws),('random_routing_mean',7+cfg.invariance.random_draws,len(model_names))):
                                    for metric,values in measured.items():rotation_values[(threshold,'score',label,group,metric)].append(values[:,a:b].mean(-1))
                    all_accuracy=scores[(-1,-1)]['accuracy']
                    rotation_values[(threshold,'primary','R1_minus_R2',-1,'accuracy')].append(all_accuracy[:,1]-all_accuracy[:,2])
                    for stratum in (-1,0,1,2):
                        ua=scores[(stratum,0)]['accuracy'];aa=scores[(stratum,1)]['accuracy'];rr=scores[(stratum,2)]['accuracy'];combined=scores[(stratum,3)]['accuracy']
                        for group_label,measure in (('unambiguous',ua),('ambiguous',aa),('rare_unseen',rr)):
                            rotation_values[(threshold,'mode_delta',group_label,stratum,'accuracy')].append(measure[:,1]-measure[:,2])
                        for group,other in (('ambiguous',aa),('rare_unseen',rr),('combined',combined)):
                            value=(ua[:,1]-ua[:,2])-(other[:,1]-other[:,2])
                            rotation_values[(threshold,'interaction',group,stratum,'accuracy')].append(value)
                    random_mean=all_accuracy[:,5:5+cfg.invariance.random_draws].mean(-1)
                    comparisons={'R3_minus_R4':all_accuracy[:,3]-all_accuracy[:,4],
                                 'R3_minus_mean_R5':all_accuracy[:,3]-random_mean,
                                 'R3_minus_better_fixed':all_accuracy[:,3]-torch.maximum(all_accuracy[:,1],all_accuracy[:,2])}
                    for label,value in comparisons.items():rotation_values[(threshold,'primary',label,-1,'accuracy')].append(value)
                    rank_rows.append({'dataset':name,'test_genre':genre,'seed':seed,'threshold':threshold,'R3_rank_of_6':1+int((all_accuracy[0,5:10]>all_accuracy[0,3]).sum())})
                    id_data=predictions['id'];id_type=bank['type'][id_data['index']];id_y=bank['y'][id_data['index']]
                    ip1=id_data['probability'][:,pmap[indices[1]]];ip2=id_data['probability'][:,pmap[indices[2]]]
                    id_columns=[id_data['prediction'][:,i].long() for i in indices]
                    id_columns += [torch.where(anti[id_type,None],ip1,ip2).argmax(-1),((ip1+ip2)/2).argmax(-1)]
                    id_columns += [torch.where(a[id_type,None],ip1,ip2).argmax(-1) for a in random_assignments]
                    id_accuracy=(torch.stack(id_columns,dim=1)==id_y[:,None]).float().mean(0)
                    for m,label in enumerate(model_names):
                        rotation_values[(threshold,'id',label,-1,'accuracy')].append(torch.full((weights.shape[0],),float(id_accuracy[m])))
            for key,values in rotation_values.items():collected[key].append(torch.stack(values))
        for key,rotations_values in collected.items():
            threshold,kind,label,group,metric=key
            values=torch.stack(rotations_values,dim=1)
            record={'dataset':name,'threshold':threshold,'kind':kind,'comparison':label,'group_or_tertile':group,'metric':metric,**summary(values)}
            primary.append(record)
    write_csv(output/'summary.csv',primary)
    write_csv(output/'lambda_curves.csv',curves)
    write_csv(output/'selections.csv',selections)
    write_csv(output/'per_rotation_scores.csv',raw)
    write_csv(output/'R3_ranks.csv',rank_rows)
    write_csv(output/'label_shift.csv',shift_rows)
    write_csv(output/'group_label_shift.csv',group_shift_rows)
    return {'summary_rows':len(primary)}
