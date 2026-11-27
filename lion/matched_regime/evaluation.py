"""Existing seqeval metrics and exact cached natural-pair construction."""
import math
import torch
from ..diagnostics.common import absolute,read_csv,write_csv
from ..independent_judge.run import evaluation as student_evaluation,evaluation_from_logits
from ..label_change.natural_pairs import pair_counts
from .data import root,load


def paired(output):
    rows=read_csv(output/'seqeval_per_seed.csv');summary=[]
    for target,space,head in sorted({(r['target'],r['space'],r['head']) for r in rows}):
        selected=[r for r in rows if (r['target'],r['space'],r['head'])==(target,space,head)]
        for metric in ('span_f1','precision','recall'):
            score=torch.tensor([float(r[metric]) for r in selected],dtype=torch.double);delta=torch.tensor([float(r[metric])-float(next(b[metric] for b in rows if (b['seed'],b['target'],b['space'],b['head'])==(r['seed'],target,space,'baseline'))) for r in selected],dtype=torch.double)
            summary.append({'target':target,'space':space,'head':head,'metric':metric,'seeds':len(score),'mean':float(score.mean()),'seed_sd':float(score.std()),'paired_mean':float(delta.mean()),'paired_sd':float(delta.std()),'paired_se':float(delta.std()/math.sqrt(len(delta)))})
    write_csv(output/'paired_summary.csv',summary)
    per_label=read_csv(output/'per_label_per_seed.csv');label_rows=[]
    for target,space,head,label in sorted({(r['target'],r['space'],r['head'],r['label']) for r in per_label}):
        selected=[r for r in per_label if (r['target'],r['space'],r['head'],r['label'])==(target,space,head,label)]
        for metric in ('f1-score','precision','recall'):
            pairs=[(r,next((b for b in per_label if (b['seed'],b['target'],b['space'],b['head'],b['label'])==(r['seed'],target,space,'baseline',label)),None)) for r in selected]
            available=[(r,b) for r,b in pairs if b is not None];score=torch.tensor([float(r[metric]) for r in selected],dtype=torch.double)
            delta=torch.tensor([float(r[metric])-float(b[metric]) for r,b in available],dtype=torch.double)
            label_rows.append({'target':target,'space':space,'head':head,'label':label,'metric':metric,'mean':float(score.mean()),'seed_sd':float(score.std()) if len(score)>1 else None,'paired_seeds':len(delta),'paired_mean':float(delta.mean()) if len(delta) else None,'paired_sd':float(delta.std()) if len(delta)>1 else None,'paired_se':float(delta.std()/math.sqrt(len(delta))) if len(delta)>1 else None})
    write_csv(output/'per_label_paired.csv',label_rows)


def new(cfg,output):
    result=student_evaluation(cfg,output);paired(output);return result


def old(cfg,output):
    prior=list(cfg.augmentation.seeds);cfg.augmentation.seeds=[0,1,2]
    result=evaluation_from_logits(cfg,output,absolute(cfg.matched.historical_run)/'test');paired(output);cfg.augmentation.seeds=prior;return result


def rowkeys(sentence,max_length):
    position=torch.arange(len(sentence));begins=torch.ones(len(sentence),dtype=torch.bool);begins[1:]=sentence[1:]!=sentence[:-1];offset=torch.where(begins,position,0).cummax(0).values
    return sentence*(max_length+1)+position-offset


def pairs(cfg,output):
    reference=load(absolute(cfg.natural_pairs.results)/'predictions.pt');cache=load(absolute(cfg.grid_pilot.evaluation_cache))['sets']['id']['data'];keys=rowkeys(cache['sentence'],cfg.encoder.max_length);wanted=rowkeys(reference['sentence'],cfg.encoder.max_length);indices=torch.searchsorted(keys,wanted)
    if not torch.equal(keys[indices],wanted) or not torch.equal(cache['y'][indices],reference['gold']):raise ValueError('Natural-pair occurrences do not match cached native test')
    lookup={};word=torch.tensor([lookup.setdefault(w,len(lookup)) for w in reference['forms']]);gold=reference['gold'];sentence=reference['sentence'];K=int(gold.max())+1
    result=[]
    for family,path,seeds in [('matched',root(cfg)/'test',list(cfg.augmentation.seeds)),('historical',absolute(cfg.matched.historical_run)/'test',[0,1,2])]:
        for seed in seeds:
            saved=load(path/f'seed{seed}_id.pt');prediction=saved['logits'][indices].argmax(-1)
            for group,mask in [('all_labels_including_O',torch.ones(len(gold),dtype=torch.bool)),('entity_vs_entity',gold!=0)]:
                w,y,s=word[mask],gold[mask],sentence[mask];counts=torch.bincount(w*K+y,minlength=len(lookup)*K).reshape(-1,K);_,local=torch.unique(torch.stack((w,s),1),dim=0,return_inverse=True);within=torch.bincount(local*K+y,minlength=(int(local.max())+1)*K).reshape(-1,K)
                partners=counts[w].sum(-1)-counts[w,y]-within[local].sum(-1)+within[local,y];participates=partners>0
                for h,name in enumerate(saved['heads']):
                    pred=prediction[mask,h];metrics=pair_counts(w,y,pred,s,K,len(lookup));result.append({'family':family,'seed':seed,'head':name,'group':group,'occurrences':int(participates.sum()),'per_occurrence_accuracy':float((pred[participates]==y[participates]).double().mean()),**metrics})
    write_csv(output/'per_seed.csv',result);summary=[]
    for family,head,group in sorted({(r['family'],r['head'],r['group']) for r in result}):
        selected=[r for r in result if (r['family'],r['head'],r['group'])==(family,head,group)]
        for metric in ('per_occurrence_accuracy','pair_accuracy','word_macro_pair_accuracy'):
            values=torch.tensor([r[metric] for r in selected],dtype=torch.double);delta=torch.tensor([r[metric]-next(b[metric] for b in result if (b['family'],b['seed'],b['head'],b['group'])==(family,r['seed'],'baseline',group)) for r in selected],dtype=torch.double)
            summary.append({'family':family,'head':head,'group':group,'metric':metric,'mean':float(values.mean()),'seed_sd':float(values.std()),'paired_mean':float(delta.mean()),'paired_sd':float(delta.std()),'paired_se':float(delta.std()/math.sqrt(len(delta))),'seeds':len(values),'eligible_pairs':selected[0]['eligible_unordered_pairs'],'participating_occurrences':selected[0]['occurrences']})
    write_csv(output/'summary.csv',summary);return {'students':len(result)}
