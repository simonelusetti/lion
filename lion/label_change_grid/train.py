"""Shared baseline plus nine heads at equal labelled-token/optimizer budgets; dev only."""
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from tagger.ensemble import IndependentReadouts,clip_heads
from ..label_change.data import parts
from ..augmentation.train import schedule,collate
from ..partition.train import predict,metric_vectors
from ..certification.classify import evaluate
from ..runtime import autocast
from ..diagnostics.common import write_csv
from .generate import root,cells


class Steps(Dataset):
    def __init__(self,raw,augmented,batches,plan,seed):
        self.raw,self.augmented,self.batches=raw,augmented,batches;self.original_stream=plan['original_stream'];self.slices=[]
        sizes=plan['labelled_tokens_per_step'];total=int(sizes.sum());nraw=len(raw['y'])
        for head,f in enumerate(augmented):
            n=len(f['y']);ratio=n/(nraw+n);rng=torch.Generator().manual_seed(seed+1000);needed=round(total*ratio)
            stream=torch.cat([torch.randperm(n,generator=rng) for _ in range((needed+n-1)//n)])[:needed] if n else torch.empty(0,dtype=torch.long)
            self.slices.append([]);seen=old_aug=original=0
            for size in sizes.tolist():
                seen+=size;end=round(seen*ratio);nv=end-old_aug;no=size-nv
                self.slices[-1].append((self.original_stream[original:original+no],stream[old_aug:end]));original+=no;old_aug=end
    def __len__(self):return len(self.batches)
    def __getitem__(self,i):
        b=self.batches[i];a=b['baseline'];size=len(a);x=torch.empty(size,1+len(self.augmented),768);y=torch.empty(size,1+len(self.augmented),dtype=torch.long)
        x[:,0]=self.raw['features'][a,0].float();y[:,0]=self.raw['y'][a]
        for h,f in enumerate(self.augmented):
            original,variant=self.slices[h][i];n=len(original);x[:n,h+1]=self.raw['features'][original,0].float();y[:n,h+1]=self.raw['y'][original]
            x[n:,h+1]=f['features'][variant].float();y[n:,h+1]=f['y'][variant]
        return {'x':x,'y':y,'epoch':b['epoch']}


def experiment(cfg,output):
    bank,p=parts(cfg);raw=p['fit'];validation=p['validation'];labels=bank['labels']['ner'];grid=cells(cfg);aug=[]
    caches={f:torch.load(root(cfg)/'generate'/f'nearby{round(f*100)}.pt',mmap=True,weights_only=False) for f in cfg.grid_pilot.fractions}
    for cell in grid:
        saved=caches[cell['fraction']];keep=saved['confidence']>=cell['tau'];aug.append({'features':saved['features'][keep],'y':saved['y'][keep]})
    records=[];budgets=[]
    for seed in cfg.augmentation.seeds:
        folder=output/f'seed{seed}';folder.mkdir(exist_ok=True);torch.manual_seed(seed);core=IndependentReadouts(cfg.tagger,768,len(labels),len(grid)+1).to(cfg.runtime.device)
        optimizer=torch.optim.AdamW(core.parameters(),lr=cfg.train.lr,weight_decay=cfg.train.weight_decay);batches,plan=schedule(raw['sentence'],cfg,seed);dataset=Steps(raw,aug,batches,plan,seed);loader=DataLoader(dataset,batch_size=1,collate_fn=collate,num_workers=cfg.runtime.workers)
        best=torch.full((len(grid)+1,),-torch.inf);epoch_best=torch.zeros(len(grid)+1,dtype=torch.long);state={k:v.detach().cpu().clone() for k,v in core.state_dict().items()};history=[]
        for number,b in enumerate(loader):
            core.train();optimizer.zero_grad(set_to_none=True)
            with autocast(cfg):logits=core(b['x'].to(cfg.runtime.device)).float()
            loss=F.cross_entropy(logits.permute(1,2,0),b['y'].to(cfg.runtime.device).T,reduction='none').mean(-1).sum();loss.backward();clip_heads(core,cfg.train.clip);optimizer.step();epoch=b['epoch']
            if number+1<len(batches) and batches[number+1]['epoch']==epoch:continue
            scores=metric_vectors(predict(core,validation,cfg),validation['y'],len(labels));criterion=scores['macro_f1'];improved=criterion>best
            for key,value in core.state_dict().items():state[key][improved]=value.detach().cpu()[improved]
            best[improved]=criterion[improved];epoch_best[improved]=epoch
            for h,name in enumerate(['baseline']+[c['name'] for c in grid]):history.append({'head':h,'name':name,'epoch':epoch,'dev_macro_f1':float(criterion[h])})
            print(f'seed{seed} epoch{epoch}: dev macro-F1 range {float(criterion.min()):.4f}–{float(criterion.max()):.4f}',flush=True)
        core.load_state_dict(state);logits=predict(core,validation,cfg);metrics=evaluate(logits,validation,cfg.certification.source,cfg.certification.source,labels,'native')
        torch.save({'state':state,'labels':len(labels),'heads':['baseline']+[c['name'] for c in grid],'epoch':epoch_best},folder/'model.pt');write_csv(folder/'history.csv',history)
        for h,name in enumerate(['baseline']+[c['name'] for c in grid]):
            records.append({'seed':seed,'head':h,'name':name,'epoch':int(epoch_best[h]),**{k:float(v[h]) for k,v in metrics.items()}})
            augmented=sum(len(dataset.slices[h-1][i][1]) for i in range(len(batches))) if h else 0
            budgets.append({'seed':seed,'name':name,'optimizer_steps':len(batches),'labelled_token_presentations':int(plan['baseline_original_presentations']),'original_token_presentations':int(plan['baseline_original_presentations'])-augmented,'augmented_token_presentations':augmented,'training_augmented_pool':len(aug[h-1]['y']) if h else 0})
        write_csv(output/'dev_per_seed.csv',records);write_csv(output/'budgets.csv',budgets)
    return {'heads':len(records)}
