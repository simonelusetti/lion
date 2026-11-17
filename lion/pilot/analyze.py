"""All-cell measurements, paired hypotheses, scorer ranking and calibration."""
import json
from collections import defaultdict
import torch
from omegaconf import OmegaConf
from ..diagnostics.common import write_csv
from ..invariance.data import paths
from ..invariance.analyze import metrics,summary
from ..zfree.probes import conditional,bootstrap
from ..zfree.analyze import dependence,spearman
from .data import load,cache,rotation,masks,FIXED
from .train import genres_for
from .select import invariance


def mask_shapes(bank,chosen):
    result=torch.empty(len(bank['z']),11,11)
    for row in bank['rows']:
        _,fixed=masks(row,FIXED);fixed[:,-1,:3]+=torch.tensor([row['previous_words']]*3)
        _,selected=masks(row,'menu',torch.tensor(list(chosen.values())))
        result[row['sample_ids']]=torch.cat([fixed,selected],dim=1)
    return result


def experiment(cfg,output):
    results,_=paths(cfg);choices=json.loads((results/'select'/'selections.json').read_text());cells=[];summaries=[];contrasts=[];rank_rows=[];calibration=[];patterns=[];supports=[]
    probe_cfg=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True));probe_cfg.train.epochs=cfg.zfree.probe_epochs
    for task in cfg.invariance.datasets:
        bank=load(cfg,task);collected=defaultdict(list);deltas=defaultdict(list)
        for rot_index,genre in enumerate(genres_for(task,bank,cfg)):
            genres,splits=rotation(bank,genre);chosen=choices[task][genre]['subsets'];shapes=mask_shapes(bank,chosen)
            mu=shapes[splits['train']].mean(0,keepdim=True);sd=shapes[splits['train']].std(0,unbiased=False,keepdim=True).clamp_min(1e-6);shapes=(shapes-mu)/sd
            rotation_values=defaultdict(list);rotation_deltas=defaultdict(list)
            for seed in cfg.diagnostics.seeds:
                folder=results/'cell'/task/genre/f'seed{seed}';checkpoint=torch.load(folder/'model.pt',weights_only=False);pred=torch.load(folder/'predictions.pt',mmap=True,weights_only=False)
                specs=checkpoint['heads'];main=[h for h,s in enumerate(specs) if not s['rule'].startswith('validity_')];specifications=[specs[h] for h in main]
                iid=pred['id']['index'];vid=pred['id_validation']['index'];oid=pred['ood']['index']
                inverse,weights=bootstrap(bank,iid,cfg.invariance.bootstrap_seed+rot_index,cfg.pilot.bootstrap)
                oinverse,oweights=bootstrap(bank,oid,cfg.invariance.bootstrap_seed+rot_index,cfg.pilot.bootstrap)
                id_metrics=metrics(pred['id']['prediction'][:,main].long(),bank['y'][iid],torch.ones(len(iid),dtype=torch.bool),inverse,weights,len(bank['labels']),cfg.invariance.head_chunk)
                ood_metrics=metrics(pred['ood']['prediction'][:,main].long(),bank['y'][oid],torch.ones(len(oid),dtype=torch.bool),oinverse,oweights,len(bank['labels']),cfg.invariance.head_chunk)
                test=invariance(pred['id']['log_probability'][:,main].exp(),bank['y'][iid],bank['z'][iid],[bank['document'][i] for i in iid.tolist()],weights)
                pool={'y':bank['y'][iid],'z':bank['z'][iid],'sentence':bank['sentence'][iid],'document':[bank['document'][i] for i in iid.tolist()]}
                validation={'y':bank['y'][vid],'z':bank['z'][vid],'sentence':bank['sentence'][vid]}
                measured={}
                for kind,x,v in [('prediction',pred['id']['log_probability'][:,main],pred['id_validation']['log_probability'][:,main]),
                                 ('mask',shapes[iid][:,[s['column'] for s in specifications]],shapes[vid][:,[s['column'] for s in specifications]])]:
                    probe_folder=results/'probes'/task/genre/f'seed{seed}'/kind
                    if cfg.invariance.resume and (probe_folder/'oof.pt').exists():
                        saved=torch.load(probe_folder/'oof.pt',weights_only=False);prediction,eligible,folds=saved['predicted'],saved['eligible'],saved['folds']
                    else:prediction,eligible,folds=conditional(x,v,pool,validation,bank['labels'],bank['genres'],probe_cfg,seed,cfg.diagnostics.sample_seed+rot_index,probe_folder)
                    measured[kind]=dependence(prediction,pool,eligible,inverse,weights)
                    supports.append({'task':task,'rotation':genre,'seed':seed,'kind':kind,'folds':folds,'eligible_labels':int((eligible.sum(-1)>=2).sum()),'total_labels':len(bank['labels']),'tokens':len(iid)})
                lookup={(s['rule'],s['regularizer']):h for h,s in enumerate(specifications)}
                for h,spec in enumerate(specifications):
                    baseline=lookup[('word',spec['regularizer'])]
                    values={'id_accuracy':id_metrics['accuracy'][:,h],'id_macro_f1':id_metrics['macro_f1'][:,h],'ood_accuracy':ood_metrics['accuracy'][:,h],'ood_macro_f1':ood_metrics['macro_f1'][:,h],
                            'gain_id':id_metrics['accuracy'][:,h]-id_metrics['accuracy'][:,baseline],'gain_ood':ood_metrics['accuracy'][:,h]-ood_metrics['accuracy'][:,baseline],
                            'invariance_statistic':test['statistic'][:,h],'invariance_p':test['p_value'][:,h],'rejected':(test['p_value'][:,h]<cfg.pilot.alpha).float(),
                            'genre_dependence':measured['prediction'][:,h],'mask_leakage':measured['mask'][:,h]}
                    values['gain_shrink']=values['gain_id']-values['gain_ood']
                    for metric,value in values.items():rotation_values[(spec['rule'],spec['regularizer'],metric)].append(value)
                    cells.append({'task':task,'rotation':genre,'seed':seed,'rule':spec['rule'],'regularizer':spec['regularizer'],'lambda':spec['lambda'],'best_step':int(checkpoint['best_step'][main[h]]),
                                  'subset':chosen.get(spec['rule'],-1),**{k:float(v[0]) for k,v in values.items()}})
                for reg in cfg.pilot.regularizers:
                    def accuracy(rule):return ood_metrics['accuracy'][:,lookup[(rule,reg)]]
                    for label,a,b in [('tree1_full','tree1','full'),('lin1_full','lin1','full'),('tree1_rand','tree1','rand_tree1'),('lin1_rand','lin1','rand_tree1'),('tree1_lin1','tree1','lin1'),('prev_full','full_prev','full'),('inv_suff','sel_inv_genre','sel_suff'),('free_genre','sel_inv_free','sel_inv_genre')]:rotation_deltas[(label,reg)].append(accuracy(a)-accuracy(b))
                    best_fixed=torch.stack([accuracy(rule) for rule in FIXED if (rule,reg) in lookup]).max(0).values
                    rotation_deltas[('inv_best_fixed',reg)].append(accuracy('sel_inv_genre')-best_fixed)
                    previous=test['p_value'][:,lookup[('full_prev',reg)]];full=test['p_value'][:,lookup[('full',reg)]];tree=test['p_value'][:,lookup[('tree1',reg)]]
                    rotation_deltas[('prev_rejection_minus_full',reg)].append((previous<cfg.pilot.alpha).float()-(full<cfg.pilot.alpha).float())
                    patterns.append({'task':task,'rotation':genre,'seed':seed,'regularizer':reg,'full_pass_tree_gain':bool(full[0]>=cfg.pilot.alpha and accuracy('tree1')[0]>accuracy('full')[0]),'full_reject_tree_pass':bool(full[0]<cfg.pilot.alpha and tree[0]>=cfg.pilot.alpha)})
                if 'cond' in cfg.pilot.regularizers:
                    full_gain=ood_metrics['accuracy'][:,lookup[('full','cond')]]-ood_metrics['accuracy'][:,lookup[('full','none')]]
                    tree_gain=ood_metrics['accuracy'][:,lookup[('tree1','cond')]]-ood_metrics['accuracy'][:,lookup[('tree1','none')]]
                    rotation_deltas[('penalty_full_minus_tree','cond')].append(full_gain-tree_gain)
                validity=[h for h,s in enumerate(specs) if s['rule'].startswith('validity_')];subsets=[int(specs[h]['rule'].split('_')[1]) for h in validity]
                target=bank['y'][vid];dedicated=-pred['id_validation']['log_probability'][:,validity].gather(-1,target[:,None,None].expand(-1,len(validity),1)).squeeze(-1).mean(0)
                frozen=torch.tensor(choices[task][genre]['loss'])[subsets]
                rank_rows.append({'task':task,'rotation':genre,'seed':seed,'subsets':len(subsets),'spearman':float(spearman(frozen[None],dedicated[None])[0]),'poor_threshold':0.5})
                doc_names=sorted(set(pool['document']));doc_index={d:i for i,d in enumerate(doc_names)};token_doc=torch.tensor([doc_index[d] for d in pool['document']]);generator=torch.Generator().manual_seed(cfg.pilot.scorer_seed+rot_index)
                full_heads=[lookup[('full',reg)] for reg in cfg.pilot.regularizers];null=[]
                for draw in range(cfg.pilot.calibration_draws):
                    assignment=torch.empty(len(doc_names),dtype=torch.long);assignment[torch.randperm(len(doc_names),generator=generator)]=torch.arange(len(doc_names))%len(genres)
                    null.append(invariance(pred['id']['log_probability'][:,[main[h] for h in full_heads]].exp(),bank['y'][iid],assignment[token_doc],pool['document'])['p_value'])
                null=torch.stack(null)
                for h,reg in enumerate(cfg.pilot.regularizers):calibration.append({'task':task,'rotation':genre,'seed':seed,'regularizer':reg,'draws':len(null),'rejection_rate':float((null[:,h]<cfg.pilot.alpha).float().mean())})
            for key,values in rotation_values.items():collected[key].append(torch.stack(values))
            for key,values in rotation_deltas.items():deltas[key].append(torch.stack(values))
        for key,values in collected.items():
            rule,reg,metric=key;summaries.append({'task':task,'rule':rule,'regularizer':reg,'metric':metric,**summary(torch.stack(values,dim=1))})
        for key,values in deltas.items():
            label,reg=key;contrasts.append({'task':task,'contrast':label,'regularizer':reg,**summary(torch.stack(values,dim=1))})
        for filename,rows in [('cells',cells),('summary',summaries),('contrasts',contrasts),('scorer_validity',rank_rows),('calibration',calibration),('H2_patterns',patterns),('probe_support',supports)]:write_csv(output/(filename+'.csv'),rows)
    return {'cells':len(cells),'contrasts':len(contrasts)}
