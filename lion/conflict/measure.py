"""Streaming lexical supply and frozen layer-head predictions; no insertion training."""
import torch
from tagger.ensemble import stack_readouts
from ..layer_sensitivity.train import Head
from ..runtime import autocast
from ..diagnostics.common import absolute
from . import data,encode


@torch.no_grad()
def supply(cfg,output):
    for name in cfg.coverage.datasets:
        target=output/f'{name}.pt'
        if target.exists():continue
        bank=data.load(cfg,name);src=data.source(cfg,name);slots=bank['slots']['id'];n=len(slots['y']);context=torch.load(data.root(cfg)/'context'/name/'id.pt',weights_only=False)['probability']
        selected=(context.max(-1).values>=min(cfg.conflict.thresholds)).nonzero().flatten();lex=bank['lexicon'];encoder=encode.model(src,bank);device=src.runtime.device
        counts=lex['counts'].to(device);total=lex['total'].to(device);majority=lex['majority'].to(device);eligible=(total>=cfg.conflict.minimum_occurrences)&lex['unique_majority'].to(device)
        k=cfg.conflict.top_each;draws=cfg.conflict.natural_draws;columns=draws+2*k
        result={key:torch.full((n,),torch.nan,dtype=torch.float64) for key in ('allowed_mass','conflict_mass','nonconflict_mass','max_conflict_probability','context_gold_probability','priced_bound')}
        result.update(words=torch.full((n,columns),-1,dtype=torch.long),kind=torch.full((n,columns),-1,dtype=torch.long),word_probability=torch.zeros(n,columns,dtype=torch.float64),selected=selected)
        generator=torch.Generator(device=device).manual_seed(cfg.conflict.sample_seed+sum(map(ord,name)))
        for batch in encode.loader(encode.Slots(bank,'id',selected),cfg):
            rows=batch['slot'];y=slots['y'][rows].to(device);p=encode.forward(encoder,batch,src,mlm=True)['probability'];gold_count=counts.T[y]
            conflict=eligible[None]&(majority[None]!=y[:,None])&(gold_count<=cfg.conflict.maximum_gold_share*total[None])
            nonconflict=eligible[None]&(majority[None]==y[:,None]);allowed_mass=p.sum(-1)
            natural=torch.multinomial(p,draws,replacement=True,generator=generator)
            choices=[natural];kinds=[torch.where(conflict.gather(1,natural),0,torch.where(nonconflict.gather(1,natural),1,-1))]
            for mask,kind in ((conflict,0),(nonconflict,1)):
                value,index=p.masked_fill(~mask,-1).topk(k,dim=1);choices.append(index);kinds.append(torch.full_like(index,kind).masked_fill(value<0,-1))
            choices=torch.cat(choices,1);kind=torch.cat(kinds,1);word=lex['ids'].to(device)[choices]
            # Natural words remain stored even when neither label definition applies.
            valid=torch.ones_like(kind,dtype=torch.bool);valid[:,draws:]=kind[:,draws:]>=0
            result['words'][rows]=word.masked_fill(~valid,-1).cpu();result['kind'][rows]=kind.cpu();result['word_probability'][rows]=p.gather(1,choices).masked_fill(~valid,0).cpu()
            best=p.masked_fill(~conflict,0).max(-1).values;pc=context[rows].gather(1,slots['y'][rows,None])[:,0]
            for key,value in [('allowed_mass',allowed_mass),('conflict_mass',(p*conflict).sum(-1)),('nonconflict_mass',(p*nonconflict).sum(-1)),('max_conflict_probability',best),('context_gold_probability',pc),('priced_bound',torch.where(best.cpu()>0,(1-pc)/best.cpu(),torch.inf))]:result[key][rows]=value.cpu()
        torch.save(result,target);print(f'{name} supply: {len(selected)} contexts selected at lowest threshold',flush=True);del encoder;torch.cuda.empty_cache()
    return {'datasets':len(cfg.coverage.datasets)}


def heads(cfg,name,src,bank):
    models=[];identifiers=[]
    for layer in cfg.conflict.layers:
        for seed in cfg.conflict.seeds:
            path=absolute(cfg.conflict.layer_run)/'train'/name/f'layer{layer}'/f'seed{seed}'/'model.pt';saved=torch.load(path,weights_only=False,map_location='cpu')
            model=Head(src,saved['labels'],layer);model.load_state_dict(saved['state']);models.append(model.core);identifiers.append({'layer':layer,'seed':seed,'path':str(path)})
    return stack_readouts(models).to(src.runtime.device).eval().requires_grad_(False),identifiers


@torch.no_grad()
def predict(core,features,bank,src,layers):
    index=torch.tensor([0 if layer==8 else 1 for layer in layers],device=features.device)
    with autocast(src):logits=core(features[:,index]).double()
    probability=logits.softmax(-1);mapping=bank['mapping'].to(probability.device)
    projected=probability.new_zeros(*probability.shape[:2],len(bank['labels']))
    projected.scatter_add_(-1,mapping[None,None].expand_as(probability),probability)
    return projected.argmax(-1).cpu()


@torch.no_grad()
def room(cfg,output):
    for name in cfg.coverage.datasets:
        target=output/f'{name}.pt'
        if target.exists():continue
        bank=data.load(cfg,name);src=data.source(cfg,name);sup=torch.load(data.root(cfg)/'supply'/f'{name}.pt',weights_only=False);selected=sup['selected'];n,columns=sup['words'].shape
        core,identifiers=heads(cfg,name,src,bank);encoder=encode.model(src,bank);layers=[h['layer'] for h in identifiers]
        original=torch.full((n,len(layers)),-1,dtype=torch.int16)
        for batch in encode.loader(encode.Slots(bank,'id',selected,mode='original'),cfg):original[batch['slot']]=predict(core,encode.forward(encoder,batch,src)['features'],bank,src,layers).short()
        slots,choice=(sup['words']>=0).nonzero(as_tuple=True);dataset=encode.Slots(bank,'id',slots,mode='insert',word=sup['words'][slots,choice]);prediction=torch.full((n,columns,len(layers)),-1,dtype=torch.int16)
        for batch in encode.loader(dataset,cfg):
            i=batch['index'];prediction[slots[i],choice[i]]=predict(core,encode.forward(encoder,batch,src)['features'],bank,src,layers).short()
        torch.save({'prediction':prediction,'original':original,'heads':identifiers},target);print(f'{name} room: {len(slots)} insertions; all six existing heads',flush=True);del encoder,core;torch.cuda.empty_cache()
    return {'datasets':len(cfg.coverage.datasets)}
