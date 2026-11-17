"""Train exactly three independent MLPs on every eligible EWT training window."""
import torch
from torch.utils.data import Dataset,DataLoader
from torch.nn import functional as F
from tagger.crt import CRTReadouts
from ..diagnostics.common import save_json,write_csv
from .data import paths,CLASSES,TASKS


class Sentences(Dataset):
    def __init__(self,data):
        self.data=data;_,counts=data['sentence'].unique_consecutive(return_counts=True);self.offsets=torch.cat([torch.zeros(1,dtype=torch.long),counts.cumsum(0)])
    def __len__(self):return len(self.offsets)-1
    def __getitem__(self,index):
        a,b=self.offsets[index:index+2].tolist();return {'x':self.data['features'][a:b].float(),'y':self.data['y'][a:b]}


def collate(rows):return {k:torch.cat([r[k] for r in rows]) for k in rows[0]}


def loader(data,cfg,train=False):return DataLoader(Sentences(data),batch_size=cfg.train.batch_size if train else cfg.runtime.batch_size,shuffle=train,collate_fn=collate,num_workers=cfg.runtime.workers)


@torch.no_grad()
def validation(core,data,cfg):
    core.eval();confusion=[torch.zeros(c,c) for c in CLASSES]
    for batch in loader(data,cfg):
        values=core(batch['x'].to(cfg.runtime.device))
        for h,c in enumerate(CLASSES):
            prediction=values[h].argmax(-1).cpu();y=batch['y'][:,h]
            confusion[h]+=torch.bincount(y*c+prediction,minlength=c*c).reshape(c,c)
    result=[]
    for matrix in confusion:
        diagonal=matrix.diagonal();result.append({'accuracy':float(diagonal.sum()/matrix.sum()),'macro_f1':float((2*diagonal/(matrix.sum(0)+matrix.sum(1)).clamp_min(1)).mean())})
    return result


def experiment(cfg,output):
    _,cache=paths(cfg);data={s:torch.load(cache/(s+'.pt'),mmap=True,weights_only=False) for s in ('train','validation')}
    core=CRTReadouts(cfg.tagger,CLASSES,cfg.seed).to(cfg.runtime.device)
    optimizer=torch.optim.AdamW(core.parameters(),lr=cfg.train.lr,weight_decay=cfg.train.weight_decay)
    history=[]
    for epoch in range(1,cfg.train.epochs+1):
        core.train()
        for batch in loader(data['train'],cfg,True):
            optimizer.zero_grad(set_to_none=True);values=core(batch['x'].to(cfg.runtime.device));y=batch['y'].to(cfg.runtime.device)
            sum(F.cross_entropy(value,y[:,h]) for h,value in enumerate(values)).backward()
            if cfg.train.clip is not None:
                for readout in core.models:torch.nn.utils.clip_grad_norm_(readout.parameters(),cfg.train.clip)
            optimizer.step()
        scores=validation(core,data['validation'],cfg)
        for h,task in enumerate(TASKS):
            history.append({'epoch':epoch,'task':task,**scores[h]})
        print(f'Readout epoch {epoch}/{cfg.train.epochs}: '+', '.join(f'{t} dev acc {s["accuracy"]:.4f}' for t,s in zip(TASKS,scores)),flush=True)
    state={k:v.detach().cpu().clone() for k,v in core.state_dict().items()};epochs=torch.full((3,),cfg.train.epochs,dtype=torch.long)
    torch.save({'state':state,'epochs':epochs,'classes':CLASSES,'seed':cfg.seed},output/'model.pt')
    scores=validation(core,data['validation'],cfg);save_json(output/'dev.json',{t:{**s,'selected_epoch':int(epochs[h])} for h,(t,s) in enumerate(zip(TASKS,scores))});write_csv(output/'history.csv',history)
    return {'dev_accuracy':[s['accuracy'] for s in scores]}


def readouts(cfg):
    results,_=paths(cfg);core=CRTReadouts(cfg.tagger,CLASSES,cfg.seed).to(cfg.runtime.device)
    core.load_state_dict(torch.load(results/'train'/'model.pt',map_location='cpu',weights_only=False)['state'])
    return core.eval().requires_grad_(False)
