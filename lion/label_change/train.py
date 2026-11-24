"""Two matched readouts: identical original batches, appended focal-only pseudo-labels."""
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from tagger.ensemble import IndependentReadouts,clip_heads
from ..augmentation.train import schedule,collate,evaluation_data
from ..partition.train import predict,metric_vectors
from ..certification.classify import evaluate,span_counts
from ..labels import LABELS
from ..runtime import autocast
from ..diagnostics.common import write_csv,save_json
from . import data

ARMS=['baseline','augmented']


class Steps(Dataset):
    def __init__(self,features,batches,seed):
        self.f,self.batches=features,batches;original=features['raw'];n=len(features['y'])
        needed=round(sum(len(b['baseline']) for b in batches)*n/len(original['y']));g=torch.Generator().manual_seed(seed+1000)
        self.stream=torch.cat([torch.randperm(n,generator=g) for _ in range((needed+n-1)//n)])[:needed] if n else torch.empty(0,dtype=torch.long)
        offset=seen=0;self.indices=[]
        for b in batches:
            seen+=len(b['baseline']);end=round(seen*n/len(original['y']));self.indices.append(self.stream[offset:end]);offset=end
    def __len__(self):return len(self.batches)
    def __getitem__(self,i):
        b=self.batches[i];a=b['baseline'];v=self.indices[i];raw=self.f['raw'];n=len(a);x=torch.zeros(n+len(v),2,768);y=torch.full((n+len(v),2),-100,dtype=torch.long)
        x[:n]=raw['features'][a].float().expand(-1,2,-1);y[:n]=raw['y'][a,None].expand(-1,2)
        x[n:,1]=self.f['features'][v].float();y[n:,1]=self.f['y'][v]
        return {'x':x,'y':y,'epoch':b['epoch']}


def experiment(cfg,output):
    f=torch.load(data.root(cfg)/'generate'/'features.pt',mmap=True,weights_only=False);evaluation=evaluation_data(cfg,output);labels=evaluation['labels'];validation=evaluation['sets']['validation']['data'];records=[];budgets=[]
    for seed in cfg.augmentation.seeds:
        folder=output/f'seed{seed}';folder.mkdir(exist_ok=True);torch.manual_seed(seed);core=IndependentReadouts(cfg.tagger,768,len(labels),2).to(cfg.runtime.device)
        optimizer=torch.optim.AdamW(core.parameters(),lr=cfg.train.lr,weight_decay=cfg.train.weight_decay);batches,plan=schedule(f['raw']['sentence'],cfg,seed);dataset=Steps(f,batches,seed);steps=DataLoader(dataset,batch_size=1,collate_fn=collate,num_workers=cfg.runtime.workers)
        best=torch.full((2,),-torch.inf);state={k:v.detach().cpu().clone() for k,v in core.state_dict().items()};history=[]
        for number,b in enumerate(steps):
            core.train();optimizer.zero_grad(set_to_none=True);y=b['y'].to(cfg.runtime.device)
            with autocast(cfg):logits=core(b['x'].to(cfg.runtime.device)).float()
            losses=F.cross_entropy(logits.permute(1,2,0),y.T,ignore_index=-100,reduction='none').sum(-1)/(y!=-100).sum(0)
            losses.sum().backward();clip_heads(core,cfg.train.clip);optimizer.step();epoch=b['epoch']
            if number+1<len(batches) and batches[number+1]['epoch']==epoch:continue
            scores=metric_vectors(predict(core,validation,cfg),validation['y'],len(labels));criterion=scores['macro_f1'];improved=criterion>best
            for k,v in core.state_dict().items():state[k][improved]=v.detach().cpu()[improved]
            best[improved]=criterion[improved]
            for h,arm in enumerate(ARMS):history.append({'epoch':epoch,'arm':arm,'dev_macro_f1':float(criterion[h])})
            print(f'seed{seed} epoch{epoch}: baseline/augmented dev macro-F1 {criterion.tolist()}',flush=True)
        core.load_state_dict(state);torch.save({'state':state,'labels':len(labels),'dimension':768,'heads':2,'linear':False,'seed':seed},folder/'model.pt');write_csv(folder/'history.csv',history)
        for split,item in evaluation['sets'].items():
            if split=='validation':continue
            part=item['data'];target=item['target'];logits=predict(core,part,cfg)
            for space in (['native','shared_ner'] if split=='id' else ['shared_ner']):
                metrics=evaluate(logits,part,'conll2003',target,labels,space);c,p,e=span_counts(logits.argmax(-1),part['y'],part['sentence'],labels,LABELS[target] if space=='shared_ner' else labels,shared=space=='shared_ner')
                for h,arm in enumerate(ARMS):records.append({'seed':seed,'arm':arm,'target':target,'space':space,'span_f1':float(metrics['span_f1'][h]),'accuracy':float(metrics['accuracy'][h]),'precision':float(c[h]/p[h].clamp_min(1)),'recall':float(c[h]/e.clamp_min(1))})
            torch.save({'logits':logits,'gold':part['y'],'sentence':part['sentence']},folder/f'{split}.pt')
        # Mechanism check on the independent acceptance-calibration inputs; pseudo-label agreement is not accuracy.
        check=torch.load(data.root(cfg)/'generate'/'changed_probe.pt',weights_only=False)
        for kind,x in [('original',check['original']),('modified',check['modified'])]:
            if len(x):
                pred=predict(core,{'features':x[:,None],'y':check['gold'],'sentence':torch.arange(len(x))},cfg).argmax(-1);torch.save(pred,folder/f'probe_{kind}.pt')
        budgets.append({'seed':seed,'optimizer_steps_both':len(batches),'original_label_presentations_both':int(plan['baseline_original_presentations']),'extra_focal_labels_augmented':len(dataset.stream)})
        write_csv(output/'per_seed.csv',records)
    write_csv(output/'budgets.csv',budgets);return {'evaluations':len(records)}
