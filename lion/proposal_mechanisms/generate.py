"""Unchanged sampling primitives, exact input identity, separately timed generation/scoring."""
import time
import torch
from ..certification import data,selector
from ..encoder import load_masked_encoder
from ..partition.train import load_model
from ..diagnostics.common import absolute,save_json
from ..label_change.data import sampled_rows

ARMS=('local70','full_one_pass','full_refined')


def root(cfg):return absolute(cfg.mechanisms.results)


def inputs(cfg,limit=None):
    bank=data.load(cfg);held=[r for r in selector.raw_rows(cfg,bank) if r['fold']=='calibration']
    rows=sampled_rows(held,cfg.mechanisms.sources,cfg.label_change.seed+3)
    if limit is not None:rows=rows[:limit]
    return bank,rows


def sync(cfg):
    if cfg.runtime.device.startswith('cuda'):torch.cuda.synchronize(cfg.runtime.device)


def cohort(rows,labels):
    lengths=torch.tensor([len(r['ids']) for r in rows]);offsets=torch.cat((torch.zeros(1,dtype=torch.long),lengths.cumsum(0)));focals=torch.tensor([int(r['focals'][0]) for r in rows]);starts=[];ends=[]
    for r,focal in zip(rows,focals):
        pos=(r['word_ids']==focal).nonzero().flatten();starts.append(int(pos[0]));ends.append(int(pos[-1])+1)
    return {'input_ids':torch.cat([r['ids'].int() for r in rows]),'offsets':offsets,'lengths':lengths,'focal':focals,'focal_start':torch.tensor(starts),'focal_end':torch.tensor(ends),
            'sentence':torch.tensor([r['sentence'] for r in rows]),'source_id':[r['source_id'] for r in rows],'gold':torch.tensor([int(r['gold']['ner'][0]) for r in rows]),'labels':labels}


@torch.no_grad()
def arm(cfg,bank,rows,encoder,scorer,name,output=None):
    meta=cohort(rows,bank['labels']['ner']);n=len(rows);timings=[];started=time.monotonic()
    if output is not None:output.mkdir(parents=True,exist_ok=True)
    for draw in range(cfg.mechanisms.draws):
        rank_rng=torch.Generator(device=cfg.runtime.device).manual_seed(cfg.generators.seed+draw*100003)
        local_rng=torch.Generator(device=cfg.runtime.device).manual_seed(cfg.generators.seed+12+draw*100003)
        # Refinement initialization exactly matches Arm 2, but is physically rerun and timed.
        full_rng=torch.Generator(device=cfg.runtime.device).manual_seed(cfg.mechanisms.full_seed+draw*100003)
        refine_mask_rng=torch.Generator(device=cfg.runtime.device).manual_seed(cfg.mechanisms.full_seed+50000+draw*100003)
        refine_fill_rng=torch.Generator(device=cfg.runtime.device).manual_seed(cfg.mechanisms.full_seed+70000+draw*100003)
        candidate=torch.empty_like(meta['input_ids']);initial_candidate=torch.empty_like(candidate) if name=='full_refined' else None;prediction=torch.empty(n,dtype=torch.long);confidence=torch.empty(n);source_prediction=torch.empty(n,dtype=torch.long);source_confidence=torch.empty(n)
        generation_seconds=scoring_seconds=0.;cursor=0;accepted_parts=[];accepted_labels=[];accepted_indices=[];all_features=[]
        for b in selector.focal_batches(data.loader(rows,cfg),'ner',cfg):
            ids,attention,words,focal=(b[k] for k in ('ids','attention','word_ids','focal'));size=len(focal);ordinary=attention&(words>=0)&(words!=focal[:,None])
            sync(cfg);clock=time.monotonic()
            if name=='local70':
                p=torch.arange(int(b['counts'].max()),device=ids.device)[None];d=(p-focal[:,None]).abs();eligible=(p<b['counts'][:,None])&(d>0)&(d<=5);count=torch.floor(eligible.sum(-1)*.7+.5).long()
                noise=torch.rand(eligible.shape,device=ids.device,generator=rank_rng).masked_fill(~eligible,torch.inf);dropped=eligible&(noise.argsort(-1).argsort(-1)<count[:,None]);drop=data.word_to_pieces(dropped,words,False)
                filled=selector.independent_fills(encoder,ids,attention,drop,1,bank['special'],cfg,local_rng)
            else:
                filled=selector.independent_fills(encoder,ids,attention,ordinary,1,bank['special'],cfg,full_rng)
                if name=='full_refined':
                    initialization=filled
                    number=torch.ceil(ordinary.sum(-1)*cfg.mechanisms.refinement_fraction).long()
                    for _ in range(cfg.mechanisms.refinement_rounds):
                        noise=torch.rand(ordinary.shape,device=ids.device,generator=refine_mask_rng).masked_fill(~ordinary,torch.inf)
                        remask=ordinary&(noise.argsort(-1).argsort(-1)<number[:,None])
                        filled=selector.independent_fills(encoder,filled[:,0],attention,remask,1,bank['special'],cfg,refine_fill_rng)
            sync(cfg);generation_seconds+=time.monotonic()-clock;clock=time.monotonic()
            features=selector.draw_features(encoder,ids,attention,words,focal,filled,cfg)[:,0];p=selector.scorer_log_probs(scorer,features,cfg).exp();s,h=p.max(-1)
            original=b['hidden'][b['owner'],focal];os,oh=selector.scorer_log_probs(scorer,original,cfg).exp().max(-1)
            sync(cfg);scoring_seconds+=time.monotonic()-clock
            if cfg.mechanisms.get('cache_all_features',False):all_features.append(features.cpu().half())
            if cfg.mechanisms.get('cache_accepted_features',False):
                accepted=(oh==b['gold'])&(h!=oh)&(s>=cfg.mechanisms.tau)&(filled[:,0]!=ids).any(-1)
                accepted_parts.append(features[accepted].cpu().half());accepted_labels.append(h[accepted].cpu());accepted_indices.append(accepted.nonzero().flatten().cpu()+cursor)
            start,end=int(meta['offsets'][cursor]),int(meta['offsets'][cursor+size]);candidate[start:end]=filled[:,0][attention].int().cpu()
            if initial_candidate is not None:initial_candidate[start:end]=initialization[:,0][attention].int().cpu()
            prediction[cursor:cursor+size]=h.cpu();confidence[cursor:cursor+size]=s.cpu();source_prediction[cursor:cursor+size]=oh.cpu();source_confidence[cursor:cursor+size]=os.cpu();cursor+=size
        timing={'arm':name,'draw_id':draw,'proposals':n,'generation_seconds':generation_seconds,'scoring_seconds':scoring_seconds,'compute_seconds':generation_seconds+scoring_seconds,'mlm_passes_per_proposal':5 if name=='full_refined' else 1}
        timings.append(timing)
        if output is not None:
            torch.save({'arm':name,'draw_id':draw,'input_ids':candidate,'offsets':meta['offsets'],'prediction':prediction,'confidence':confidence,'source_prediction':source_prediction,'source_confidence':source_confidence,'initial_input_ids':initial_candidate,'timing':timing},output/f'draw{draw}.pt')
            if cfg.mechanisms.get('cache_accepted_features',False):torch.save({'features':torch.cat(accepted_parts),'y':torch.cat(accepted_labels),'example_id':torch.cat(accepted_indices)},output/f'accepted_features_draw{draw}.pt')
            if cfg.mechanisms.get('cache_all_features',False):torch.save({'features':torch.cat(all_features),'y':prediction,'anchor_id':torch.arange(n)},output/f'all_features_draw{draw}.pt')
            print(f'{name} draw{draw}: generation {generation_seconds:.2f}s, scoring {scoring_seconds:.2f}s',flush=True)
    return {'arm':name,'proposals':n*cfg.mechanisms.draws,'generation_seconds':sum(t['generation_seconds'] for t in timings),'scoring_seconds':sum(t['scoring_seconds'] for t in timings),'wall_seconds':time.monotonic()-started,'draws':timings}


def benchmark(cfg,output):
    setup=time.monotonic();bank,rows=inputs(cfg,cfg.mechanisms.benchmark_sources);encoder=load_masked_encoder(cfg);scorer=load_model(data.paths(cfg)/'scorer'/'ner',cfg).requires_grad_(False);setup=time.monotonic()-setup
    results=[]
    for name in ARMS:
        result=arm(cfg,bank,rows,encoder,scorer,name);factor=cfg.mechanisms.sources/len(rows);result['projected_generation_seconds']=result['generation_seconds']*factor;result['projected_scoring_seconds']=result['scoring_seconds']*factor;result['projected_wall_seconds']=result['wall_seconds']*factor;results.append(result)
    total=setup+sum(r['projected_wall_seconds'] for r in results)+30
    save_json(output/'estimate.json',{'benchmark_sources':len(rows),'draws_per_source':cfg.mechanisms.draws,'setup_seconds':setup,'arms':results,'projected_whole_run_seconds':total,'projected_minutes':total/60,'allowance_seconds':30})
    print({'projected_minutes':total/60,'arms':[{k:r[k] for k in ('arm','projected_generation_seconds','projected_scoring_seconds','projected_wall_seconds')} for r in results]},flush=True)
    return {'projected_minutes':total/60}


def experiment(cfg,output):
    bank,rows=inputs(cfg);meta=cohort(rows,bank['labels']['ner']);torch.save(meta,output/'sources.pt');encoder=load_masked_encoder(cfg);scorer=load_model(data.paths(cfg)/'scorer'/'ner',cfg).requires_grad_(False);timings=[]
    for name in ARMS:timings.append(arm(cfg,bank,rows,encoder,scorer,name,output/name))
    save_json(output/'timing.json',timings);return {'arms':len(timings),'proposals_per_arm':len(rows)*cfg.mechanisms.draws}
