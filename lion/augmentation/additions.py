"""Additional reporting from the original held-out draws and saved scorer predictions."""
import torch
from ..certification import data,selector,calibrate
from ..certification.classify import evaluate
from ..partition.train import load_model
from ..diagnostics.common import read_csv,write_csv
from .data import paths


@torch.no_grad()
def experiment(cfg,output):
    task=data.task_names(cfg)[0];root=paths(cfg);bank=data.load(cfg);raw=selector.raw_rows(cfg,bank);lookup={r['sentence']:r for r in raw}
    scorer=load_model(data.paths(cfg)/'scorer'/task,cfg).requires_grad_(False);labels=bank['labels'][task];groups,_=selector.label_groups(labels)
    enriched=[];increase=read_csv(root/'bounds'/'bounds.csv')
    for fold in ('g1','g2'):
        arrays=torch.load(root/'bounds'/f'{fold}.pt',weights_only=False);saved={}
        for group,name in enumerate(groups):
            values=arrays[name];rows=[]
            for sentence,focal in zip(values['sentence'],values['focal']):
                original=lookup[sentence];f=torch.tensor([focal])
                rows.append({**original,'focals':f,'gold':{task:y[f] for task,y in original['gold'].items()},'group':torch.tensor([group])})
            parts=[]
            for batch in selector.focal_batches(data.loader(rows,cfg),task,cfg):
                vectors=batch['hidden'][batch['owner'],batch['focal']]
                logp=selector.scorer_log_probs(scorer,vectors,cfg).gather(-1,batch['gold'][:,None]).squeeze(-1)
                parts.append(-calibrate.smooth(logp.double().exp(),len(labels),cfg.certification.gamma).cpu())
            raw_ce=torch.cat(parts) if parts else torch.empty(0,dtype=torch.float64)
            absolute=raw_ce+values['losses'];n=len(absolute)
            mean=float(absolute.mean()) if n else None;sd=float(absolute.std()) if n>1 else None
            upper=mean+1.645*sd/n**.5 if sd is not None else None
            record=next(r for r in increase if r['selector']==fold and r['group']==name)
            enriched.append({**record,'absolute_ce_mean':mean,'absolute_ce_sd':sd,'absolute_ce_upper_bound':upper,'raw_smoothed_ce_mean':float(raw_ce.mean()) if n else None})
            saved[name]={'raw_smoothed_ce':raw_ce,'absolute_filled_ce':absolute,'increase':values['losses'],'sentence':values['sentence'],'focal':values['focal']}
        torch.save(saved,output/f'{fold}_absolute_ce.pt')
    write_csv(output/'enriched_bounds.csv',enriched)
    cuts=read_csv(data.paths(cfg)/'calibrate'/'fractions.csv');write_csv(output/'calibrated_cuts.csv',cuts)
    full=torch.load(data.paths(cfg)/'scorer'/'full_features.pt',mmap=True,weights_only=False)
    native=bank['sets']['train'];mask=torch.tensor([r['fold']=='scorer_validation' for r in native]);counts=torch.tensor([len(r['focals']) for r in native])
    validation=data.readout_data([r for r in native if r['fold']=='scorer_validation'],full['train'][mask.repeat_interleave(counts)],task)
    logits=torch.load(data.paths(cfg)/'scorer'/task/'predictions.pt',mmap=True,weights_only=False)['validation']
    metrics=evaluate(logits,validation,cfg.certification.source,cfg.certification.source,labels,'native')
    token_ce=-logits[:,0].log_softmax(-1).gather(1,validation['y'][:,None]).squeeze(-1)
    sentences,inverse=validation['sentence'].unique(return_inverse=True)
    summed=torch.zeros(len(sentences)).index_add_(0,inverse,token_ce)
    score={key:float(value[0]) for key,value in metrics.items()}
    score.update(sentences=len(sentences),words=len(token_ce),mean_sentence_summed_ce=float(summed.mean()))
    write_csv(output/'scorer_validation.csv',[score]);return enriched,cuts,score
