"""Estimate before fitting: time real masked queries and a worst-case room budget."""
import time
import torch
from ..diagnostics.common import save_json
from . import data,encode


def experiment(cfg,output):
    datasets={}
    for name in cfg.coverage.datasets:
        src=data.source(cfg,name);bank=data.load(cfg,name);encoder=encode.model(src,bank);slots=bank['slots']['train'];n=min(cfg.conflict.estimate_sequences,len(slots['y']))
        index=torch.randperm(len(slots['y']),generator=torch.Generator().manual_seed(cfg.seed))[:n];dataset=encode.Slots(bank,'train',index)
        for batch in encode.loader(encode.Slots(bank,'train',index[:128]),cfg):encode.forward(encoder,batch,src,mlm=True)
        torch.cuda.synchronize();started=time.monotonic()
        for batch in encode.loader(dataset,cfg):encode.forward(encoder,batch,src,mlm=True)
        torch.cuda.synchronize();seconds=time.monotonic()-started;rate=float(dataset.length.sum())/seconds
        feature_tokens=sum(float(encode.Slots(bank,split).length.sum()) for split in ('train','validation','id'))
        test_tokens=float(encode.Slots(bank,'id').length.sum());original_tokens=float(encode.Slots(bank,'id',mode='original').length.sum())
        # Conservative: every test context passes tau=.9 and has all eleven insertion candidates.
        candidates=cfg.conflict.natural_draws+2*cfg.conflict.top_each
        stages={'context_features':feature_tokens/rate,'context_head':45.,'supply':test_tokens/rate,'room':(candidates*test_tokens+original_tokens)/rate,'analysis_report':60.}
        datasets[name]={'slots':{s:len(v['y']) for s,v in bank['slots'].items()},'timed_sequences':n,'timed_seconds':seconds,'tokens_per_second':rate,'seconds':stages,'upper_estimate_seconds':sum(stages.values())}
        print(name,datasets[name],flush=True);del encoder;torch.cuda.empty_cache()
    total=sum(d['upper_estimate_seconds'] for d in datasets.values());save_json(output/'estimate.json',{'datasets':datasets,'upper_estimate_seconds':total,'minutes':total/60,'assumptions':'All test slots selected at tau=.9, five natural draws plus top three of each class. Full MLM forward timed for every query, including context-feature passes; CPU/I/O and fit/report allowance included. No sampling reduction.'})
    return {'minutes':total/60}
