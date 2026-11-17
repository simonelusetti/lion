"""Stage-1 fixed-mode and group interactions, with paired document bootstrap."""
import json
from collections import defaultdict
import torch
from ..diagnostics.common import write_csv
from .data import banks,rotations,paths
from .analyze import bootstrap,metrics,summary


def experiment(cfg,output):
    results,_=paths(cfg);selection=json.loads((results/'pilot'/'selection.json').read_text())
    summaries=[];curves=[];raw=[];shifts=[];group_shifts=[];selections=[]
    for name,bank in banks(cfg):
        collected=defaultdict(list)
        for rotation_index,(genre,genres,validation,test) in enumerate(rotations(name,bank,cfg)):
            folder=results/('pilot' if genre==selection[name]['rotation'] else 'train')/name/genre
            plan=torch.load(folder/'plan.pt',weights_only=False)
            for label,value in enumerate(plan['label_shift']):shifts.append({'dataset':name,'test_genre':genre,'label':bank['labels'][label],'training_JS_nats':float(value)})
            training_y=bank['y'][plan['training_tokens']]
            for threshold,groups in plan['groups'].items():
                train_group=groups[bank['type'][plan['training_tokens']]]
                for group in range(3):
                    mask=train_group==group
                    group_shifts.append({'dataset':name,'test_genre':genre,'threshold':threshold,'group':group,'training_tokens':int(mask.sum()),'training_token_weighted_JS':float(plan['label_shift'][training_y[mask]].mean())})
            rotation_values=defaultdict(list)
            for seed in cfg.diagnostics.seeds:
                checkpoint=torch.load(folder/f'seed{seed}'/'model.pt',map_location='cpu',mmap=True,weights_only=False)
                predictions=torch.load(folder/f'seed{seed}'/'predictions.pt',map_location='cpu',mmap=True,weights_only=False)
                specs=checkpoint['heads'];ood=predictions['ood'];ids=ood['index'];y=bank['y'][ids]
                inverse,weights=bootstrap(bank,ids,cfg.invariance.bootstrap_seed+rotation_index,cfg.invariance.bootstrap)
                def head(rule,strength):
                    return 0 if strength==0 else next(i for i,s in enumerate(specs) if s['rule']==rule and s['lambda']==strength)
                for rule in ('R1','R2'):
                    for i,spec in enumerate(specs):
                        if i and spec['rule']!=rule:continue
                        curves.append({'dataset':name,'test_genre':genre,'seed':seed,'pilot':genre==selection[name]['rotation'],'rule':rule,'lambda':spec['lambda'],
                                       'selected':spec['lambda']==selection[name][rule]['lambda'],'neighbour':spec['lambda']==selection[name][rule]['neighbour'],
                                       'validation_macro_f1':float(checkpoint['validation_macro_f1'][i]),'best_step':int(checkpoint['best_step'][i]),
                                       'ood_accuracy':float(ood['accuracy'][i]),'ood_macro_f1':float(ood['macro_f1'][i]),'id_accuracy':float(predictions['id']['accuracy'][i])})
                for policy,field in (('main','lambda'),('neighbour','neighbour')):
                    heads=[0]+[head(rule,selection[name][rule][field]) for rule in ('R1','R2')]
                    matrix=ood['prediction'][:,heads].long()
                    for rule,h in zip(('R0','R1','R2'),heads):selections.append({'dataset':name,'test_genre':genre,'seed':seed,'policy':policy,'rule':rule,'lambda':specs[h]['lambda'],'best_step':int(checkpoint['best_step'][h]),'validation_macro_f1':float(checkpoint['validation_macro_f1'][h])})
                    shift=torch.bucketize(plan['label_shift'][y],plan['shift_cuts'])
                    for threshold,groups in plan['groups'].items():
                        token_group=groups[bank['type'][ids]]
                        for stratum in (-1,0,1,2):
                            smask=torch.ones(len(y),dtype=torch.bool) if stratum==-1 else shift==stratum
                            measured={}
                            for group in (-1,0,1,2,3):
                                mask=smask if group==-1 else smask&(token_group!=0 if group==3 else token_group==group)
                                score=metrics(matrix,y,mask,inverse,weights,len(bank['labels']),cfg.invariance.head_chunk);measured[group]=score
                                for metric,values in score.items():
                                    for i,rule in enumerate(('R0','R1','R2')):rotation_values[(policy,threshold,'score',rule,stratum,group,metric)].append(values[:,i])
                                raw.append({'dataset':name,'test_genre':genre,'seed':seed,'policy':policy,'threshold':threshold,'tertile':stratum,'group':group,'tokens':int(mask.sum()),'documents':int(inverse[mask].unique().numel()),
                                            **{f'{rule}_{metric}':float(values[0,i]) for metric,values in score.items() for i,rule in enumerate(('R0','R1','R2'))}})
                                delta=score['accuracy'][:,1]-score['accuracy'][:,2]
                                rotation_values[(policy,threshold,'mode_delta','R1_minus_R2',stratum,group,'accuracy')].append(delta)
                            ua=measured[0]['accuracy'];base=ua[:,1]-ua[:,2]
                            for group in (1,2,3):
                                other=measured[group]['accuracy']
                                rotation_values[(policy,threshold,'interaction','unambiguous_minus_other',stratum,group,'accuracy')].append(base-(other[:,1]-other[:,2]))
            for key,values in rotation_values.items():collected[key].append(torch.stack(values))
        for key,values in collected.items():
            policy,threshold,kind,comparison,stratum,group,metric=key
            summaries.append({'dataset':name,'policy':policy,'threshold':threshold,'kind':kind,'comparison':comparison,'tertile':stratum,'group':group,'metric':metric,**summary(torch.stack(values,dim=1))})
    for filename,rows in [('summary',summaries),('lambda_curves',curves),('selections',selections),('per_rotation_scores',raw),('label_shift',shifts),('group_label_shift',group_shifts)]:write_csv(output/(filename+'.csv'),rows)
    return {'summary_rows':len(summaries)}
