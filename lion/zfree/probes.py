"""Balanced E3 probes: document CV on ID test, separate native ID validation."""
import torch
from ..partition.data import folds
from ..partition.train import fit
from ..diagnostics.common import write_csv


def conditional(features,validation_features,pool,validation,labels,genres,cfg,seed,fold_seed,output):
    documents={}
    for doc,z in zip(pool['document'],pool['z'].tolist()):documents[doc]=z
    counts=[sum(z==g for z in documents.values()) for g in sorted(set(documents.values()))]
    count=min(cfg.zfree.folds,min(counts))
    fold=folds(pool,count,fold_seed);eligible=torch.ones(len(labels),len(genres),dtype=torch.bool)
    val_count=torch.bincount(validation['y']*len(genres)+validation['z'],minlength=len(labels)*len(genres)).reshape(len(labels),len(genres))
    eligible&=val_count>=cfg.diagnostics.probe_min.validation
    for test in range(count):
        for split,mask in [('train',fold!=test),('test',fold==test)]:
            support=torch.bincount(pool['y'][mask]*len(genres)+pool['z'][mask],minlength=len(labels)*len(genres)).reshape(len(labels),len(genres))
            eligible&=support>=cfg.diagnostics.probe_min[split]
    eligible[eligible.sum(-1)<2]=False
    predicted=torch.full(features.shape[:2],-1,dtype=torch.long);skips=[]
    for label,name in enumerate(labels):
        domains=eligible[label].nonzero().flatten()
        if len(domains)<2:
            skips.append({'label':name,'reason':'fewer than two genres meet train/validation/test support in every document fold'});continue
        mapping=torch.full((len(genres),),-1,dtype=torch.long);mapping[domains]=torch.arange(len(domains))
        allowed=(pool['y']==label)&eligible[label,pool['z']]
        val_ids=((validation['y']==label)&eligible[label,validation['z']]).nonzero().flatten()
        for test in range(count):
            ids={'train':(allowed&(fold!=test)).nonzero().flatten(),'test':(allowed&(fold==test)).nonzero().flatten()}
            local={split:{'features':features[index],'y':mapping[pool['z'][index]],'sentence':pool['sentence'][index]} for split,index in ids.items()}
            local['validation']={'features':validation_features[val_ids],'y':mapping[validation['z'][val_ids]],'sentence':validation['sentence'][val_ids]}
            _,values=fit(local,len(domains),cfg,seed,output/f'label{label}'/f'fold{test}',linear=True,balanced=True)
            predicted[ids['test']]=domains[values['test'].argmax(-1)]
    output.mkdir(parents=True,exist_ok=True)
    torch.save({'predicted':predicted,'eligible':eligible,'fold':fold,'folds':count},output/'oof.pt');write_csv(output/'skips.csv',skips)
    return predicted,eligible,count


def bootstrap(bank,ids,seed,count):
    names=sorted({bank['document'][i] for i in ids.tolist()});lookup={doc:i for i,doc in enumerate(names)}
    inverse=torch.tensor([lookup[bank['document'][i]] for i in ids.tolist()])
    genre=torch.zeros(len(names),dtype=torch.long);genre[inverse]=bank['z'][ids]
    generator=torch.Generator().manual_seed(seed);weights=torch.zeros(count+1,len(names));weights[0]=1
    for g in genre.unique():
        docs=(genre==g).nonzero().flatten();draws=docs[torch.randint(len(docs),(count,len(docs)),generator=generator)]
        weights[1:].scatter_add_(1,draws,torch.ones_like(draws,dtype=torch.float32))
    return inverse,weights
