"""Frozen MLM/scorer trajectories and exact finite-simplex selection probabilities."""
import time
import torch
import pyarrow.parquet as pq
from ..certification import data,selector
from ..partition.train import load_model
from ..encoder import load_masked_encoder
from ..diagnostics.common import absolute,save_json
from ..label_change.fewnerd_diagnostic import parquet
from ..proposal_mechanisms import generate as previous


def root(cfg):return absolute(cfg.guided.results)


def probabilities(reward,rule):
    r=reward.double();m=r.shape[-1];error=r.new_zeros(len(r))
    if rule=='uniform':p=torch.full_like(r,1/m)
    elif rule=='greedy':
        p=(r==r.max(-1,keepdim=True).values).double();p/=p.sum(-1,keepdim=True)
    else:
        lam=float(rule.removeprefix('lambda'));values=r.sort(dim=-1,descending=True).values;k=torch.arange(1,m+1,device=r.device,dtype=r.dtype)[None]
        thresholds=(values.cumsum(-1)-2*lam*m)/k
        active=torch.where(values>thresholds,k,0).amax(-1).long()
        nu=thresholds.gather(1,(active-1)[:,None]);p=(r-nu).clamp_min(0)/(2*lam*m);error=(p.sum(-1)-1).abs();p/=p.sum(-1,keepdim=True)
    divergence=((p-1/m).square()*m).sum(-1);effective=1/p.square().sum(-1)
    return p,divergence,effective,error


def target_metadata(cfg,bank,rows):
    meta=previous.cohort(rows,bank['labels']['ner']);native=pq.read_table(parquet('train'),columns=['tokens','ner_tags']).to_pylist();needed=set();forms=[]
    for row in rows:
        source=int(row['source_id'].split('/')[-1]);focal=int(row['focals'][0]);forms.append(native[source]['tokens'][focal]);needed.add(forms[-1])
    known={w:set() for w in needed}
    for row in bank['sets']['train']:
        if row['fold']!='fit':continue
        source=int(row['source_id'].split('/')[-1]);tokens=native[source]['tokens']
        for word,label in zip(tokens[:len(row['focals'])],row['gold']['ner'].tolist()):
            if word in known:known[word].add(label)
    alternatives=[sorted(known[w]-{int(y)}) for w,y in zip(forms,meta['gold'])];supported=torch.tensor([bool(a) for a in alternatives]);generator=torch.Generator().manual_seed(cfg.guided.target_seed);targets=torch.empty(cfg.mechanisms.draws,len(rows),dtype=torch.long)
    for draw in range(cfg.mechanisms.draws):
        for i,(alt,y) in enumerate(zip(alternatives,meta['gold'].tolist())):
            options=alt if alt else [j for j in range(len(bank['labels']['ner'])) if j!=y];targets[draw,i]=options[int(torch.randint(len(options),(1,),generator=generator))]
    meta.update(forms=forms,supported_alternative=supported,alternative_labels=alternatives,targets=targets)
    return meta


def prepare(cfg,output):
    bank,rows=previous.inputs(cfg);meta=target_metadata(cfg,bank,rows);supported=meta['supported_alternative']
    torch.save(meta,output/'sources_targets.pt');save_json(output/'target_policy.json',{'source_examples':len(rows),'supported_alternative_sources':int(supported.sum()),'fallback_sources':int((~supported).sum()),'replicates':cfg.mechanisms.draws,'coarse_labels':meta['labels'],'policy':'One fixed target per source/replicate, uniform among non-source labels observed for the exact case-sensitive word in scorer fit data; otherwise uniform over all other labels including O. No other held-out labels or scorer predictions choose targets.'})
    return {'supported_sources':int(supported.sum()),'sources':len(rows)}


def score(cfg,encoder,scorer,batch,candidates):
    x=selector.draw_features(encoder,batch['ids'],batch['attention'],batch['word_ids'],batch['focal'],candidates,cfg)
    n,k,d=x.shape
    return selector.scorer_log_probs(scorer,x.flatten(0,1),cfg).exp().reshape(n,k,-1),x


def exact_original(current,original,attention):return ((current==original)|~attention).all(-1)


@torch.no_grad()
def arm(cfg,bank,rows,meta,encoder,scorer,rule,replicates,output=None):
    n=len(rows);lengths=meta['lengths'][:n];offsets=torch.cat((torch.zeros(1,dtype=torch.long),lengths.cumsum(0)));timings=[];arm_start=time.monotonic()
    final_only=rule=='uniform' and not cfg.guided.get('log_uniform_scores',True)
    if output is not None:output.mkdir(parents=True,exist_ok=True)
    for replicate in range(replicates):
        replicate_started=time.monotonic()
        full_rng=torch.Generator(device=cfg.runtime.device).manual_seed(cfg.mechanisms.full_seed+replicate*100003)
        mask_rng=torch.Generator(device=cfg.runtime.device).manual_seed(cfg.mechanisms.full_seed+50000+replicate*100003)
        completion_rng=torch.Generator(device=cfg.runtime.device).manual_seed(cfg.mechanisms.full_seed+70000+replicate*100003)
        select_rng=torch.Generator(device=cfg.runtime.device).manual_seed(cfg.guided.selection_seed+replicate*100003)
        remask_bits=torch.empty(4,int(offsets[-1]),dtype=torch.bool);state_ids=torch.empty(5,int(offsets[-1]),dtype=torch.int32);state_prob=torch.full((n,5,len(meta['labels'])),torch.nan);source_prediction=torch.empty(n,dtype=torch.long);source_confidence=torch.empty(n)
        chosen_occurrence=torch.empty(n,4,dtype=torch.long);text_unchanged=torch.empty(n,4,dtype=torch.bool);divergence=torch.empty(n,4,dtype=torch.float64);effective=torch.empty(n,4,dtype=torch.float64);normalization_error=torch.empty(n,4,dtype=torch.float64)
        rewards=torch.full((n,4,9),torch.nan);selection_prob=torch.empty(n,4,9,dtype=torch.float64);generation_time=scoring_time=selection_time=0.;cursor=0;mlm_evaluations=0;masked_distributions=0;accepted_features=[];accepted_labels=[];accepted_anchors=[];all_features=[]
        for batch in selector.focal_batches(data.loader(rows,cfg),'ner',cfg):
            ids,attention,words,focal=(batch[k] for k in ('ids','attention','word_ids','focal'));b=len(ids);sl=slice(cursor,cursor+b);ordinary=attention&(words>=0)&(words!=focal[:,None]);target=meta['targets'][replicate,sl].to(ids.device)
            previous.sync(cfg);clock=time.monotonic();current=selector.independent_fills(encoder,ids,attention,ordinary,1,bank['special'],cfg,full_rng)[:,0]
            previous.sync(cfg);generation_time+=time.monotonic()-clock;mlm_evaluations+=int(ordinary.any(-1).sum());masked_distributions+=int(ordinary.sum())
            clock=time.monotonic()
            if not final_only:
                p,_=score(cfg,encoder,scorer,batch,current[:,None]);state_prob[sl,0]=p[:,0].cpu()
            orig=selector.scorer_log_probs(scorer,batch['hidden'][batch['owner'],focal],cfg).exp();os,oh=orig.max(-1)
            previous.sync(cfg);scoring_time+=time.monotonic()-clock;source_prediction[sl]=oh.cpu();source_confidence[sl]=os.cpu();start,end=int(offsets[cursor]),int(offsets[cursor+b]);state_ids[0,start:end]=current[attention].int().cpu()
            count=torch.ceil(ordinary.sum(-1)*cfg.guided.mask_fraction).long()
            for step in range(cfg.guided.rounds):
                previous.sync(cfg);clock=time.monotonic();noise=torch.rand(ordinary.shape,device=ids.device,generator=mask_rng).masked_fill(~ordinary,torch.inf);remask=ordinary&(noise.argsort(-1).argsort(-1)<count[:,None])
                proposals=selector.independent_fills(encoder,current,attention,remask,cfg.guided.completions,bank['special'],cfg,completion_rng);candidates=torch.cat((current[:,None],proposals),1)
                previous.sync(cfg);generation_time+=time.monotonic()-clock;mlm_evaluations+=int(remask.any(-1).sum());masked_distributions+=int(remask.sum())
                if final_only:r=torch.zeros(b,candidates.shape[1],device=ids.device)
                else:
                    clock=time.monotonic();p,candidate_features=score(cfg,encoder,scorer,batch,candidates)
                    previous.sync(cfg);scoring_time+=time.monotonic()-clock;r=p.gather(2,target[:,None,None].expand(-1,9,1))[:,:,0]
                clock=time.monotonic();pi,d,e,err=probabilities(r,rule);choice=torch.multinomial(pi,1,replacement=True,generator=select_rng)[:,0];owner=torch.arange(b,device=ids.device);next_state=candidates[owner,choice]
                unchanged=exact_original(next_state,current,attention)
                previous.sync(cfg);selection_time+=time.monotonic()-clock
                remask_bits[step,start:end]=remask[attention].cpu()
                chosen_occurrence[sl,step]=choice.cpu();text_unchanged[sl,step]=unchanged.cpu();divergence[sl,step]=d.cpu();effective[sl,step]=e.cpu();normalization_error[sl,step]=err.cpu();selection_prob[sl,step]=pi.cpu()
                if not final_only:
                    next_p=p[owner,choice];rewards[sl,step]=r.cpu();state_prob[sl,step+1]=next_p.cpu()
                current=next_state;state_ids[step+1,start:end]=current[attention].int().cpu()
            if final_only:
                clock=time.monotonic();p,features=score(cfg,encoder,scorer,batch,current[:,None]);next_p=p[:,0];selected_features=features[:,0]
                previous.sync(cfg);scoring_time+=time.monotonic()-clock;state_prob[sl,-1]=next_p.cpu()
            else:selected_features=candidate_features[owner,choice]
            if cfg.guided.get('cache_accepted_features',False):
                confidence,assigned=next_p.max(-1);accept=(oh==batch['gold'])&(assigned!=batch['gold'])&(confidence>=cfg.guided.tau)&~exact_original(current,ids,attention)
                accepted_features.append(selected_features[accept].cpu().half());accepted_labels.append(assigned[accept].cpu());accepted_anchors.append(accept.nonzero().flatten().cpu()+cursor)
            if cfg.guided.get('cache_all_features',False):all_features.append(selected_features.cpu().half())
            cursor+=b
        encoded=1 if final_only else 1+cfg.guided.rounds*(cfg.guided.completions+1)
        timing={'arm':rule,'replicate':replicate,'trajectories':n,'mlm_sequence_evaluations':mlm_evaluations,'mlm_masked_position_distributions':masked_distributions,'scorer_encoded_sequence_evaluations':n*encoded,
            'scorer_readout_evaluations_including_source':n*(encoded+1),'intermediate_scorer_logging':not final_only,'generation_seconds':generation_time,'scoring_seconds':scoring_time,'selection_seconds':selection_time}
        timings.append(timing)
        if output is not None:
            torch.save({'arm':rule,'replicate':replicate,'state_input_ids':state_ids,'offsets':offsets,'state_probabilities':state_prob,'source_prediction':source_prediction,'source_confidence':source_confidence,
                'refinement_mask_positions':remask_bits,'selected_candidate_occurrence':chosen_occurrence,'selected_text_unchanged':text_unchanged,'chi2':divergence,'effective_candidates':effective,'normalization_residual':normalization_error,'candidate_target_rewards':rewards,'selection_probabilities':selection_prob,'timing':timing},output/f'replicate{replicate}.pt')
            if cfg.guided.get('cache_accepted_features',False):torch.save({'features':torch.cat(accepted_features),'y':torch.cat(accepted_labels),'anchor_id':torch.cat(accepted_anchors)},output/f'accepted_features_draw{replicate}.pt')
            if cfg.guided.get('cache_all_features',False):torch.save({'features':torch.cat(all_features),'y':state_prob[:,-1].argmax(-1),'anchor_id':torch.arange(n)},output/f'all_features_draw{replicate}.pt')
            print(f'{rule} replicate{replicate}: generation {generation_time:.1f}s, scoring {scoring_time:.1f}s, selection {selection_time:.1f}s',flush=True)
        timing['wall_seconds']=time.monotonic()-replicate_started
    return {'arm':rule,'trajectories':n*replicates,'wall_seconds':time.monotonic()-arm_start,'replicates':timings}


def benchmark(cfg,output):
    started=time.monotonic();bank,rows=previous.inputs(cfg,cfg.guided.benchmark_sources);meta=torch.load(root(cfg)/'prepare'/'sources_targets.pt',weights_only=False);encoder=load_masked_encoder(cfg);scorer=load_model(data.paths(cfg)/'scorer'/'ner',cfg).requires_grad_(False);setup=time.monotonic()-started;results=[]
    for rule in cfg.guided.arms:
        r=arm(cfg,bank,rows,meta,encoder,scorer,rule,cfg.guided.benchmark_replicates);r['projected_wall_seconds']=r['wall_seconds']*cfg.mechanisms.sources/len(rows)*cfg.mechanisms.draws/cfg.guided.benchmark_replicates;results.append(r)
    seconds=setup+sum(r['projected_wall_seconds'] for r in results)+120;save_json(output/'estimate.json',{'sources':len(rows),'replicates':cfg.guided.benchmark_replicates,'arms':results,'projected_seconds':seconds,'projected_minutes':seconds/60,'setup_seconds':setup,'analysis_allowance_seconds':120});print({'projected_minutes':seconds/60,'arm_minutes':{r['arm']:r['projected_wall_seconds']/60 for r in results}},flush=True);return {'projected_minutes':seconds/60}


def experiment(cfg,output):
    bank,rows=previous.inputs(cfg);meta=torch.load(root(cfg)/'prepare'/'sources_targets.pt',weights_only=False);encoder=load_masked_encoder(cfg);scorer=load_model(data.paths(cfg)/'scorer'/'ner',cfg).requires_grad_(False);timings=[]
    for rule in cfg.guided.arms:timings.append(arm(cfg,bank,rows,meta,encoder,scorer,rule,cfg.mechanisms.draws,output/rule))
    save_json(output/'timing.json',timings);return {'arms':len(timings),'trajectories_per_arm':len(rows)*cfg.mechanisms.draws}
