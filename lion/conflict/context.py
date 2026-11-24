"""The only new predictor: a layer-8 masked-slot MLP, with native dev held out."""
import torch
from ..partition import train
from ..diagnostics.common import write_csv
from . import data,encode


def features(cfg,output):
    for name in cfg.coverage.datasets:
        folder=output/name;folder.mkdir(exist_ok=True);bank=data.load(cfg,name);src=data.source(cfg,name);encoder=encode.model(src,bank)
        for split in ('train','validation','id'):
            target=folder/f'{split}.pt'
            if target.exists():continue
            dataset=encode.Slots(bank,split);x=torch.empty(len(dataset),768,dtype=torch.float16)
            for batch in encode.loader(dataset,cfg):x[batch['index']]=encode.forward(encoder,batch,src)['features'][:,0].cpu().half()
            torch.save(x,target);print(f'{name} context features {split}: {len(x)} slots',flush=True)
        del encoder;torch.cuda.empty_cache()
    return {'datasets':len(cfg.coverage.datasets)}


def fit(cfg,output):
    for name in cfg.coverage.datasets:
        folder=output/name;folder.mkdir(exist_ok=True);src=data.source(cfg,name);bank=data.load(cfg,name);s=bank['slots']['train'];features=torch.load(data.root(cfg)/'features'/name/'train.pt',mmap=True,weights_only=False)
        def part(mask):return {'features':features[mask,None],'y':s['y'][mask],'sentence':s['sentence'][mask]}
        if not (folder/'model.pt').exists():train.fit({'train':part(~s['holdout']),'validation':part(s['holdout'])},len(bank['labels']),src,cfg.conflict.context_seed,folder)
        model=train.load_model(folder,src)
        for split in ('validation','id'):
            x=torch.load(data.root(cfg)/'features'/name/f'{split}.pt',mmap=True,weights_only=False);slots=bank['slots'][split]
            logits=train.predict(model,{'features':x[:,None],'y':slots['y'],'sentence':slots['sentence']},src)[:,0]
            torch.save({'logits':logits,'probability':logits.double().softmax(-1),'ce':-logits.double().log_softmax(-1).gather(1,slots['y'][:,None])[:,0]},folder/f'{split}.pt')
    return {'context_predictors':len(cfg.coverage.datasets)}
