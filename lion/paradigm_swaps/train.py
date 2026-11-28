"""Existing paired student recipe; train only four new heads, reuse baseline/MENTION."""
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tagger.ensemble import IndependentReadouts,clip_heads
from ..certification import data as source
from ..augmentation.train import schedule,collate
from ..label_change_grid.train import Steps
from ..partition.train import predict,metric_vectors
from ..runtime import autocast
from ..diagnostics.common import write_csv
from .generate import root,load


def train(cfg,out):
    raw=load(root(cfg)/'prepare/raw.pt');bank=source.load(cfg);full=load(source.paths(cfg)/'scorer/full_features.pt');validation=source.readout_data(bank['sets']['validation'],full['validation'],'ner');arms=list(cfg.paradigm_swaps.arms);aug=[load(root(cfg)/'generate'/a/'accepted_features_draw0.pt') for a in arms];budgets=[];records=[]
    for seed in cfg.augmentation.seeds:
        folder=out/f'seed{seed}';folder.mkdir(exist_ok=True);torch.manual_seed(seed);core=IndependentReadouts(cfg.tagger,768,9,len(arms)).to(cfg.runtime.device);optimizer=torch.optim.AdamW(core.parameters(),lr=cfg.train.lr,weight_decay=cfg.train.weight_decay);batches,plan=schedule(raw['sentence'],cfg,seed);dataset=Steps(raw,aug,batches,plan,seed);loader=DataLoader(dataset,batch_size=1,collate_fn=collate,num_workers=cfg.runtime.workers);best=torch.full((len(arms),),-torch.inf);epoch_best=torch.zeros(len(arms),dtype=torch.long);state={k:v.detach().cpu().clone() for k,v in core.state_dict().items()};history=[]
        for number,b in enumerate(loader):
            core.train();optimizer.zero_grad(set_to_none=True)
            with autocast(cfg):logits=core(b['x'][:,1:].to(cfg.runtime.device)).float()
            loss=F.cross_entropy(logits.permute(1,2,0),b['y'][:,1:].to(cfg.runtime.device).T,reduction='none').mean(-1).sum();loss.backward();clip_heads(core,cfg.train.clip);optimizer.step();epoch=b['epoch']
            if number+1<len(batches) and batches[number+1]['epoch']==epoch:continue
            criterion=metric_vectors(predict(core,validation,cfg),validation['y'],9)['macro_f1'];improved=criterion>best
            for k,v in core.state_dict().items():state[k][improved]=v.detach().cpu()[improved]
            best[improved]=criterion[improved];epoch_best[improved]=epoch
            for h,a in enumerate(arms):history.append({'seed':seed,'head':a,'epoch':epoch,'dev_token_macro_f1':float(criterion[h])})
            print(f'seed{seed} epoch{epoch}: {criterion.tolist()}',flush=True)
        torch.save({'state':state,'labels':9,'heads':arms,'epoch':epoch_best},folder/'model.pt');write_csv(folder/'history.csv',history)
        for h,a in enumerate(arms):
            variant=sum(len(x[1]) for x in dataset.slices[h]);budgets.append({'seed':seed,'head':a,'augmentation_pool_size':len(aug[h]['y']),'optimizer_steps':len(batches),'labelled_token_presentations':int(plan['baseline_original_presentations']),'original_token_presentations':int(plan['baseline_original_presentations'])-variant,'augmented_token_presentations':variant,'mixing_proportion':len(aug[h]['y'])/(len(raw['y'])+len(aug[h]['y'])),'status':'computed; existing Steps/schedule, identical original ordering and initialization'});records.append({'seed':seed,'head':a,'selected_epoch':int(epoch_best[h]),'dev_token_macro_f1':float(best[h])})
        write_csv(out/'budgets.csv',budgets);write_csv(out/'dev_selection.csv',records)
    return {'new_heads':len(records)}
