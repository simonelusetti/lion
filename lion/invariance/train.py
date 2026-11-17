"""Genre-balanced all-word fitting of the fixed rules and lambda curves."""
from pathlib import Path

import torch
from torch.nn import functional as F

from tagger.ensemble import IndependentReadouts,clip_heads
from ..diagnostics.common import write_csv
from ..runtime import autocast,parallel,device_ids
from .data import banks,rotations,plan,loader,paths
from .loss import penalty


@torch.no_grad()
def evaluate(model,bank,sentences,heads,cfg,probability=False):
    model.eval();classes=len(bank['labels'])
    head_ids=list(heads);confusion=torch.zeros(len(head_ids),classes,classes,device=cfg.runtime.device)
    indices,predictions,probabilities=[],[],[]
    for batch in loader(bank,sentences,cfg):
        x=batch['x'].to(cfg.runtime.device)[:,None];y=batch['y'].to(cfg.runtime.device)
        values=[]
        for start in range(0,len(head_ids),cfg.invariance.head_chunk):
            selected=head_ids[start:start+cfg.invariance.head_chunk]
            with autocast(cfg):logits=model(x,heads=selected)
            values.append(logits.float().softmax(-1))
        p=torch.cat(values,dim=1);prediction=p.argmax(-1)
        offsets=torch.arange(len(head_ids),device=y.device)[None]*classes*classes
        count=torch.bincount((offsets+y[:,None]*classes+prediction).flatten(),minlength=len(head_ids)*classes*classes).reshape_as(confusion)
        confusion+=count
        indices.append(batch['index']);predictions.append(prediction.cpu().short())
        if probability:
            columns=range(len(head_ids)) if probability is True else [head_ids.index(i) for i in probability]
            probabilities.append(p[:,columns].cpu())
    correct=confusion.diagonal(dim1=-2,dim2=-1);wanted=confusion.sum(-1);predicted=confusion.sum(-2)
    result={'accuracy':(correct.sum(-1)/wanted.sum(-1)).cpu(),
            'macro_f1':(2*correct/(wanted+predicted).clamp_min(1)).mean(-1).cpu()}
    result.update(index=torch.cat(indices),prediction=torch.cat(predictions))
    if probability:result['probability']=torch.cat(probabilities)
    return result


def fit_rotation(bank,training,validation,test,genres,rotation,cfg,seed,output):
    output.mkdir(parents=True,exist_ok=True)
    specifications=rotation['heads'];n=len(specifications);torch.manual_seed(seed)
    core=IndependentReadouts(cfg.tagger,bank['features'].shape[-1],len(bank['labels']),n).to(cfg.runtime.device)
    model=parallel(core,cfg)
    optimizer=torch.optim.AdamW(core.parameters(),lr=cfg.train.lr,weight_decay=cfg.train.weight_decay)
    assignments=rotation['assignments'].to(cfg.runtime.device)
    strengths=torch.tensor([h['lambda'] for h in specifications],device=cfg.runtime.device)
    genres_gpu=torch.tensor(genres,device=cfg.runtime.device)
    training_loader=loader(bank,training,cfg,train_genres=genres,seed=seed)
    best=torch.full((n,),-torch.inf);best_epoch=torch.zeros(n,dtype=torch.long);history=[]
    print(f"Fit {output}: {n} readouts, {len(training)} training sentences, {len(training_loader)} batches/epoch",flush=True)
    for epoch in range(1,cfg.train.epochs+1):
        model.train();sums={key:torch.zeros(n,device=cfg.runtime.device) for key in ('ce','conditional','marginal','conditional_pairs','marginal_pairs','conditional_supported_mass')};steps=0
        for batch in training_loader:
            optimizer.zero_grad(set_to_none=True)
            x=batch['x'].to(cfg.runtime.device)[:,None];y=batch['y'].to(cfg.runtime.device);z=batch['z'].to(cfg.runtime.device)
            modes=assignments[batch['type'].to(cfg.runtime.device)]
            # Compute integer group membership once on CPU, then transfer one packed index tensor.
            parts=[(batch['y']==label).nonzero().flatten() for label in batch['y'].unique()]
            packed=torch.cat(parts).to(cfg.runtime.device)
            label_indices=packed.split([len(part) for part in parts])
            cpu_rng=torch.get_rng_state();gpu_rng={d:torch.cuda.get_rng_state(d) for d in device_ids(cfg)}
            for start in range(0,n,cfg.invariance.head_chunk):
                end=min(start+cfg.invariance.head_chunk,n)
                torch.set_rng_state(cpu_rng)
                for d,state in gpu_rng.items():torch.cuda.set_rng_state(state,d)
                with autocast(cfg):logits=model(x,heads=slice(start,end))
                logits=logits.float()
                ce=F.cross_entropy(logits.permute(1,2,0),y[None].expand(end-start,-1),reduction='none').mean(-1)
                value,diagnostics=penalty(logits.softmax(-1),y,z,modes[:,start:end],genres_gpu,cfg.invariance.conditional_minimum,label_indices)
                loss=(ce+strengths[start:end]*value).sum();loss.backward()
                sums['ce'][start:end]+=ce.detach()
                for key,values in diagnostics.items():sums[key][start:end]+=values
            if cfg.train.clip is not None:clip_heads(core,cfg.train.clip)
            optimizer.step();steps+=1
        validation_scores=evaluate(model,bank,validation,range(n),cfg)
        current=validation_scores['macro_f1'];improved=current>best
        if epoch==1:selected={key:value.detach().cpu().clone() for key,value in core.state_dict().items()}
        for key,value in core.state_dict().items():selected[key][improved]=value.detach().cpu()[improved]
        best[improved]=current[improved];best_epoch[improved]=epoch
        sums={key:values.cpu() for key,values in sums.items()}
        for i,h in enumerate(specifications):history.append({'epoch':epoch,'head':i,**h,'validation_macro_f1':float(current[i]),'validation_accuracy':float(validation_scores['accuracy'][i]),**{key:float(values[i]/steps) for key,values in sums.items()}})
        print(f"{output.name} epoch {epoch}/{cfg.train.epochs}: baseline val F1 {float(current[0]):.4f}",flush=True)
    core.load_state_dict(selected)
    groups={}
    for i,h in enumerate(specifications):
        if h['rule']!='R0':groups.setdefault((h['rule'],h['threshold'],h['draw']),[0]).append(i)
    selected_heads={key:max(indices,key=lambda i:(float(best[i]),-specifications[i]['lambda'])) for key,indices in groups.items()}
    unique=sorted({0,*selected_heads.values()})
    torch.save({'state':selected,'heads':specifications,'best_epoch':best_epoch,'validation_macro_f1':best,
                'selected':selected_heads,'dimension':bank['features'].shape[-1],'classes':len(bank['labels'])},output/'model.pt')
    write_csv(output/'history.csv',history)
    saved={}
    for split,ids in test.items():
        scores=evaluate(model,bank,ids,range(n),cfg,probability=unique)
        saved[split]={'index':scores['index'],'prediction':scores['prediction'],'accuracy':scores['accuracy'],'macro_f1':scores['macro_f1'],
                      'probability_heads':unique,'probability':scores['probability']}
    # Routing needs validation probabilities, not a test-label decision.
    saved['validation']=evaluate(model,bank,validation,unique,cfg,probability=True)
    saved['validation']['probability_heads']=unique
    torch.save(saved,output/'predictions.pt')
    return best_epoch


def experiment(cfg,output):
    plans=[]
    for name,bank in banks(cfg):
        for rotation_name,train_genres,validation,test in rotations(name,bank,cfg):
            rotation=plan(bank,train_genres,cfg)
            folder=output/name/rotation_name;folder.mkdir(parents=True,exist_ok=True)
            torch.save(rotation,folder/'plan.pt')
            masks={'ood':bank['sentence_z']==test,'id':torch.isin(bank['sentence_z'],torch.tensor(train_genres))&(bank['sentence_split']==2)}
            validation_ids=(bank['sentence_z']==validation).nonzero().flatten()
            test_ids={key:mask.nonzero().flatten() for key,mask in masks.items()}
            for threshold,group in rotation['groups'].items():
                mass=torch.zeros(3,dtype=torch.long).index_add_(0,group,rotation['frequency'])
                plans.append({'dataset':name,'test_genre':rotation_name,'validation_genre':bank['genres'][validation],
                              'training_genres':'|'.join(bank['genres'][g] for g in train_genres),'threshold':threshold,
                              'unambiguous_train_tokens':int(mass[0]),'ambiguous_train_tokens':int(mass[1]),'rare_train_tokens':int(mass[2]),
                              'shift_cut_low':float(rotation['shift_cuts'][0]),'shift_cut_high':float(rotation['shift_cuts'][1])})
            write_csv(output/'rotation_plans.csv',plans)
            for seed in cfg.diagnostics.seeds:
                seed_output=folder/f'seed{seed}'
                if cfg.invariance.resume and (seed_output/'model.pt').exists() and (seed_output/'predictions.pt').exists():
                    print(f'Resume: keeping completed fit {seed_output}',flush=True)
                    continue
                fit_rotation(bank,rotation['train_sentences'],validation_ids,test_ids,train_genres,rotation,cfg,seed,seed_output)
    write_csv(output/'rotation_plans.csv',plans)
    return {'rotations':len(plans)//len(cfg.invariance.thresholds)}
