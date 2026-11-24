"""Yield, trajectories, diversity and sentence-cluster paired uncertainty."""
from collections import Counter,defaultdict
import json,math,time
import torch
from ..diagnostics.common import absolute,write_csv,save_json
from .generate import root


def distribution(x,prefix):
    x=x.double().flatten()
    if not len(x):return {prefix+'_mean':None,prefix+'_q25':None,prefix+'_median':None,prefix+'_q75':None}
    q=torch.quantile(x,torch.tensor([.25,.5,.75],dtype=x.dtype)).tolist()
    return {prefix+'_mean':float(x.mean()),prefix+'_q25':q[0],prefix+'_median':q[1],prefix+'_q75':q[2]}


def strata(meta):return [('overall',torch.ones(len(meta['gold']),dtype=torch.bool)),('gold_supported_alternative',meta['supported_alternative']),('fallback_target',~meta['supported_alternative'])]


def bootstrap_difference(x,sentence,seed,draws):
    # One record per trajectory; resample source sentences with all their focal/replicate records.
    if not len(x):return {'difference':None,'ci95_low':None,'ci95_high':None,'source_sentence_clusters':0}
    _,group=sentence.unique(return_inverse=True);n=int(group.max())+1;count=torch.bincount(group,minlength=n).double();total=torch.zeros(n,dtype=torch.float64).index_add_(0,group,x.double())
    index=torch.randint(n,(draws,n),generator=torch.Generator().manual_seed(seed));value=total[index].sum(-1)/count[index].sum(-1);q=torch.quantile(value,torch.tensor([.025,.975],dtype=torch.float64))
    return {'difference':float(x.double().mean()),'ci95_low':float(q[0]),'ci95_high':float(q[1]),'source_sentence_clusters':n,'bootstrap_draws':draws}


def load_arm(cfg,name,meta):
    saved=[torch.load(root(cfg)/'generate'/name/f'replicate{d}.pt',weights_only=False) for d in range(cfg.mechanisms.draws)];p=torch.stack([r['state_probabilities'] for r in saved]);prediction=p.argmax(-1);confidence=p.max(-1).values;gold=meta['gold'][None,:,None];target=meta['targets'][:,:,None];source_prediction=torch.stack([r['source_prediction'] for r in saved]);correct=source_prediction==meta['gold'][None]
    identity=torch.empty(p.shape[:3],dtype=torch.bool);keys=[]
    for d,r in enumerate(saved):
        offsets=r['offsets'].tolist();rep_keys=[]
        for i in range(len(meta['gold'])):
            a,b=offsets[i:i+2];original=meta['input_ids'][int(meta['offsets'][i]):int(meta['offsets'][i+1])]
            for stage in range(5):identity[d,i,stage]=torch.equal(r['state_input_ids'][stage,a:b],original)
            rep_keys.append(tuple(r['state_input_ids'][-1,a:b].tolist()))
        keys.append(rep_keys)
    eligible=correct[:,:,None]&~identity;any_accept=eligible&(prediction!=gold)&(confidence>=cfg.guided.tau);target_probability=p.gather(-1,target[:,:,:,None].expand(-1,-1,5,1))[:,:,:,0];target_accept=eligible&(prediction==target)&(target_probability>=cfg.guided.tau)
    return {'saved':saved,'probability':p,'prediction':prediction,'confidence':confidence,'source_correct':correct,'identity':identity,'eligible':eligible,'any_accept':any_accept,'target_accept':target_accept,'target_probability':target_probability,'keys':keys}


def summarize(cfg,name,a,meta,timing,scope,stratum,mask):
    reps=1 if scope=='first_replicate' else cfg.mechanisms.draws;index=mask.nonzero().flatten().tolist();n=len(index)*reps;final_accept=a['any_accept'][:reps,:,4][:,mask];target_accept=a['target_accept'][:reps,:,4][:,mask];eligible=a['eligible'][:reps,:,4][:,mask];identity=a['identity'][:reps,:,4][:,mask];correct=a['source_correct'][:reps,mask];pred=a['prediction'][:reps,:,4][:,mask];conf=a['confidence'][:reps,:,4][:,mask];gold=meta['gold'][mask][None]
    selected_keys=[a['keys'][d][i] for d in range(reps) for i in index];counts=Counter(selected_keys);within=sum(len({a['keys'][d][i] for d in range(reps)}) for i in index)
    accepted_keys={(a['keys'][d][i],int(meta['focal_start'][i]),int(meta['focal_end'][i])) for d in range(reps) for i in index if a['any_accept'][d,i,4]}
    accepted_indices={i for i in index if a['any_accept'][:reps,i,4].any()};canonical={(meta['source_id'][i],int(meta['focal'][i])) for i in accepted_indices}
    times=timing['replicates'][:reps];wall=timing['wall_seconds'] if reps==cfg.mechanisms.draws else times[0]['wall_seconds'];total_compute=sum(r['generation_seconds']+r['scoring_seconds']+r['selection_seconds'] for r in times)
    ne=int(eligible.sum());na=int(final_accept.sum());nt=int(target_accept.sum());bad=int((~correct).sum());same=int(identity.sum());nunch=int((eligible&(pred==gold)).sum());low=int((eligible&(pred!=gold)&(conf<cfg.guided.tau)).sum())
    r={'arm':name,'prespecified_main_guided':name==cfg.guided.main_arm,'scope':scope,'stratum':stratum,'source_anchors':len(index),'attempts':n,'any_flip_accepted':na,'any_flip_rate':na/n if n else None,'target_accepted':nt,'target_acceptance_rate':nt/n if n else None,
       'incorrect_source_exclusions':bad,'final_unchanged_text':same,'final_unchanged_rate':same/n if n else None,'incorrect_source_and_unchanged_overlap':int(((~correct)&identity).sum()),'unchanged_correct_source_exclusions':int((correct&identity).sum()),'eligible_final_outputs':ne,'any_acceptance_given_eligible':na/ne if ne else None,'target_acceptance_given_eligible':nt/ne if ne else None,
       'eligible_prediction_stayed_source':nunch,'eligible_changed_below_tau':low,'eligible_accepted_target':nt,'eligible_accepted_other_label':na-nt,'distinct_accepted_focal_examples':len(accepted_keys),'anchors_with_any_acceptance':len(accepted_indices),'canonical_source_focal_slots_with_any_acceptance':len(canonical),
       'visited_original_at_any_stage':int(a['identity'][:reps,mask].any(-1).sum()),'visited_original_rate':float(a['identity'][:reps,mask].any(-1).double().mean()) if n else None,
       'distinct_final_sentences_global':len(counts),'global_duplicate_outputs_excess':n-len(counts),'global_duplicate_rate':(n-len(counts))/n if n else None,'distinct_outputs_within_anchor_sum':within,'within_anchor_duplicate_outputs_excess':n-within,'within_anchor_duplicate_rate':(n-within)/n if n else None,'cross_anchor_duplicates_after_within_anchor_dedup':within-len(counts),
       'mlm_sequence_evaluations_full_arm':sum(t['mlm_sequence_evaluations'] for t in times),'scorer_encoded_sequence_evaluations_full_arm':sum(t['scorer_encoded_sequence_evaluations'] for t in times),'scorer_readout_evaluations_including_source_full_arm':sum(t['scorer_readout_evaluations_including_source'] for t in times),
       'generation_seconds_full_arm':sum(t['generation_seconds'] for t in times),'scoring_seconds_full_arm':sum(t['scoring_seconds'] for t in times),'selection_seconds_full_arm':sum(t['selection_seconds'] for t in times),'compute_seconds_full_arm':total_compute,'total_wall_seconds_full_arm':wall,'distinct_accepted_focal_examples_per_total_arm_second':len(accepted_keys)/wall,'attempts_per_total_arm_second':n/wall}
    return r


def experiment(cfg,output):
    started=time.monotonic();meta=torch.load(root(cfg)/'prepare'/'sources_targets.pt',weights_only=False);timings=json.loads((root(cfg)/'generate'/'timing.json').read_text());summaries=[];stages=[];regularization=[];comparison={};pair_rows=[];init_match=mask_match=compared=0;uniform_initial=[];uniform_masks=[]
    for name in cfg.guided.arms:
        a=load_arm(cfg,name,meta);timing=next(t for t in timings if t['arm']==name)
        if name=='uniform':
            uniform_initial=[r['state_input_ids'][0].clone() for r in a['saved']];uniform_masks=[r['refinement_mask_positions'].clone() for r in a['saved']]
        else:
            for d,r in enumerate(a['saved']):
                off=r['offsets'].tolist()
                for i in range(len(meta['gold'])):
                    start,end=off[i:i+2];init_match+=torch.equal(r['state_input_ids'][0,start:end],uniform_initial[d][start:end]);mask_match+=torch.equal(r['refinement_mask_positions'][:,start:end],uniform_masks[d][:,start:end]);compared+=1
        for scope in ('first_replicate','pooled_five_replicates'):
            reps=1 if scope=='first_replicate' else cfg.mechanisms.draws
            for label,mask in strata(meta):
                summaries.append(summarize(cfg,name,a,meta,timing,scope,label,mask))
                for step in range(5):
                    p=a['target_probability'][:reps,mask,step];n=p.numel();record={'arm':name,'scope':scope,'stratum':label,'stage':step,'attempts':n,'mean_target_probability':float(p.mean()) if n else None,'predicted_target_fraction':float((a['prediction'][:reps,mask,step]==meta['targets'][:reps,mask]).double().mean()) if n else None,'target_specific_acceptance_fraction':float(a['target_accept'][:reps,mask,step].double().mean()) if n else None,'original_reconstruction_fraction':float(a['identity'][:reps,mask,step].double().mean()) if n else None}
                    if step:
                        parent=torch.stack([r['selected_candidate_occurrence'][:,step-1] for r in a['saved'][:reps]])[:,mask];same=torch.stack([r['selected_text_unchanged'][:,step-1] for r in a['saved'][:reps]])[:,mask]
                        record.update(parent_candidate_selected_fraction=float((parent==0).double().mean()) if n else None,unchanged_text_transition_fraction=float(same.double().mean()) if n else None)
                    stages.append(record)
                if name.startswith('lambda'):
                    chi=torch.stack([r['chi2'] for r in a['saved'][:reps]])[:,mask];effective=torch.stack([r['effective_candidates'] for r in a['saved'][:reps]])[:,mask];residual=torch.stack([r['normalization_residual'] for r in a['saved'][:reps]])[:,mask]
                    regularization.append({'arm':name,'scope':scope,'stratum':label,'transitions':chi.numel(),**distribution(chi,'chi2'),**distribution(effective,'effective_candidates'),'max_normalization_residual':float(residual.max()) if residual.numel() else None})
        records=[]
        for d in range(cfg.mechanisms.draws):
            for i in range(len(meta['gold'])):
                records.append({'arm':name,'replicate':d,'anchor_id':i,'source_id':meta['source_id'][i],'sentence_id':int(meta['sentence'][i]),'focal_word':meta['forms'][i],'focal_start':int(meta['focal_start'][i]),'focal_end':int(meta['focal_end'][i]),'source_gold_label':meta['labels'][int(meta['gold'][i])],'target_label':meta['labels'][int(meta['targets'][d,i])],'gold_supported_alternative':bool(meta['supported_alternative'][i]),'source_correct':bool(a['source_correct'][d,i]),'assigned_pseudo_label':meta['labels'][int(a['prediction'][d,i,4])],'final_confidence':float(a['confidence'][d,i,4]),'target_probability':float(a['target_probability'][d,i,4]),'identical_to_source':bool(a['identity'][d,i,4]),'visited_source':bool(a['identity'][d,i].any()),'eligible':bool(a['eligible'][d,i,4]),'any_flip_accepted':bool(a['any_accept'][d,i,4]),'target_specific_accepted':bool(a['target_accept'][d,i,4]),'final_nonpadding_input_ids':json.dumps(a['keys'][d][i])})
        write_csv(output/f'{name}_trajectories.csv',records)
        if name in ('uniform',cfg.guided.main_arm):comparison[name]={k:a[k] for k in ('any_accept','target_accept','target_probability','prediction')}
        print(f'Summarised {name}',flush=True)
    cluster_lookup={};off=meta['offsets'].tolist();cluster_id=torch.tensor([cluster_lookup.setdefault(tuple(meta['input_ids'][off[i]:off[i+1]].tolist()),len(cluster_lookup)) for i in range(len(meta['gold']))])
    control=comparison['uniform'];guided=comparison[cfg.guided.main_arm]
    for scope in ('first_replicate','pooled_five_replicates'):
        reps=1 if scope=='first_replicate' else cfg.mechanisms.draws
        for label,mask in strata(meta):
            sentence=cluster_id[None].expand(reps,-1)[:,mask].flatten()
            for metric,key in [('any_flip_acceptance','any_accept'),('target_specific_acceptance','target_accept'),('mean_target_probability','target_probability')]:
                delta=guided[key][:reps,mask,4].double()-control[key][:reps,mask,4].double();r=bootstrap_difference(delta.flatten(),sentence,cfg.guided.bootstrap_seed,cfg.guided.bootstrap_draws)
                pair_rows.append({'guided_arm':cfg.guided.main_arm,'control':'uniform','scope':scope,'stratum':label,'metric':metric,**r})
    historical=[];old=absolute(cfg.mechanisms.results);historical_timing=next(t for t in json.loads((old/'generate'/'timing.json').read_text()) if t['arm']=='full_refined')
    old_meta=torch.load(old/'generate'/'sources.pt',weights_only=False)
    for scope in ('first_replicate','pooled_five_replicates'):
        reps=1 if scope=='first_replicate' else cfg.mechanisms.draws;accepted_count=target_count=0;distinct=set()
        for d in range(reps):
            r=torch.load(old/'generate'/'full_refined'/f'draw{d}.pt',weights_only=False);offset=r['offsets'].tolist()
            for i in range(len(meta['gold'])):
                start,end=offset[i:i+2];original=old_meta['input_ids'][int(old_meta['offsets'][i]):int(old_meta['offsets'][i+1])];changed=not torch.equal(r['input_ids'][start:end],original);ok=int(r['source_prediction'][i])==int(meta['gold'][i]);prediction=int(r['prediction'][i]);confidence=float(r['confidence'][i]);accepted=ok and changed and prediction!=int(meta['gold'][i]) and confidence>=cfg.guided.tau
                if accepted:
                    accepted_count+=1;target_count+=prediction==int(meta['targets'][d,i]);distinct.add((tuple(r['input_ids'][start:end].tolist()),int(meta['focal_start'][i]),int(meta['focal_end'][i])))
        times=historical_timing['draws'][:reps];compute=sum(t['generation_seconds']+t['scoring_seconds'] for t in times);attempts=reps*len(meta['gold'])
        historical.append({'scope':scope,'mechanism':'historical single-completion refinement, no retained-parent candidate and no reward selection','attempts':attempts,'any_flip_accepted':accepted_count,'any_flip_rate':accepted_count/attempts,'target_accepted_posthoc_fixed_targets':target_count,'target_acceptance_rate':target_count/attempts,'distinct_accepted_focal_examples':len(distinct),'mlm_sequence_evaluations':5*attempts,'scorer_encoded_sequence_evaluations':attempts,'generation_seconds':sum(t['generation_seconds'] for t in times),'scoring_seconds':sum(t['scoring_seconds'] for t in times),'total_compute_seconds':compute,'distinct_accepted_per_compute_second':len(distinct)/compute,'comparison_status':'Historical, not matched selection control. Target success is posthoc against this experiment’s fixed targets; historical generation did not use targets. Historical timing conditions differ.'})
    for filename,records in [('summary',summaries),('stages',stages),('regularization',regularization),('paired_main_comparison',pair_rows),('historical_comparison',historical)]:write_csv(output/f'{filename}.csv',records)
    save_json(output/'summary.json',summaries);save_json(output/'matching.json',{'initializations_matched':init_match,'mask_position_sets_matched':mask_match,'trajectories_compared_to_uniform':compared,'main_guided_arm':cfg.guided.main_arm,'paired_uncertainty':'2000 source-sentence cluster bootstrap resamples, paired trajectory differences; draws and focal anchors with the same exact original non-padding input sequence stay together, including duplicate sentence records. Pointwise percentile 95% intervals.'})
    write_report(cfg,output,summaries,stages,regularization,pair_rows,historical)
    return {'arms':len(cfg.guided.arms),'trajectories_per_arm':cfg.mechanisms.sources*cfg.mechanisms.draws,'analysis_seconds':time.monotonic()-started}


def write_report(cfg,output,summary,stages,regularization,paired,historical):
    def fmt(value,scale=1):return '—' if value is None else f'{float(value)*scale:.3f}'
    rows=[r for r in summary if r['scope']=='pooled_five_replicates' and r['stratum']=='overall'];lines=['# Matched scorer-guided refinement diagnostic','',f'Frozen full-data scorer and MLM; 4,000 original held-out cohort anchors × five replicates per arm. Main guided arm prespecified: **{cfg.guided.main_arm}**. Uniform control also scores all nine candidate occurrences. No student training or new annotations.','',
      '| Arm | Attempts | Any-flip % | Target % | Any / target % among eligible | Distinct accepted | Anchors with acceptance | Unchanged % | Global duplicate % | Total wall s | Distinct accepted / wall s |','|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['arm']} | {r['attempts']} | {fmt(r['any_flip_rate'],100)} | {fmt(r['target_acceptance_rate'],100)} | {fmt(r['any_acceptance_given_eligible'],100)} / {fmt(r['target_acceptance_given_eligible'],100)} | {r['distinct_accepted_focal_examples']} | {r['anchors_with_any_acceptance']} | {fmt(r['final_unchanged_rate'],100)} | {fmt(r['global_duplicate_rate'],100)} | {fmt(r['total_wall_seconds_full_arm'])} | {fmt(r['distinct_accepted_focal_examples_per_total_arm_second'])} |")
    lines+=['','All acceptance rates above use all attempts unless explicitly conditional. Distinct accepted identity is exact non-padding token IDs plus focal start/end positions. Reconstruction is separate from duplication.','', '| Arm | First-replicate attempts | Any-flip % | Target % |','|---|---:|---:|---:|']
    for r in summary:
        if r['scope']=='first_replicate' and r['stratum']=='overall':lines.append(f"| {r['arm']} | {r['attempts']} | {fmt(r['any_flip_rate'],100)} | {fmt(r['target_acceptance_rate'],100)} |")
    lines+=['','| Arm | Target support stratum | Attempts | Any-flip % | Target % |','|---|---|---:|---:|---:|']
    for r in summary:
        if r['scope']=='pooled_five_replicates' and r['stratum']!='overall':lines.append(f"| {r['arm']} | {r['stratum']} | {r['attempts']} | {fmt(r['any_flip_rate'],100)} | {fmt(r['target_acceptance_rate'],100)} |")
    lines+=['','| Arm | Eligible | Incorrect source | Final unchanged | Incorrect ∩ unchanged | Stayed source label | Changed below tau | Accepted target | Accepted other | Ever visited original |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['arm']} | {r['eligible_final_outputs']} | {r['incorrect_source_exclusions']} | {r['final_unchanged_text']} | {r['incorrect_source_and_unchanged_overlap']} | {r['eligible_prediction_stayed_source']} | {r['eligible_changed_below_tau']} | {r['eligible_accepted_target']} | {r['eligible_accepted_other_label']} | {r['visited_original_at_any_stage']} |")
    lines+=['','Source exclusions and reconstruction overlap; neither is silently dropped from all-attempt rates. The four outcome columns partition eligible final states. Stage visits count initialization and selected editing states, not unselected proposals. Full duplicate accounting and first-replicate/stratum counts are in summary.csv.','', '| Main guided minus uniform | Difference, pp | Source-sentence-cluster 95% interval, pp |','|---|---:|---|']
    for r in paired:
        if r['scope']=='pooled_five_replicates' and r['stratum']=='overall':lines.append(f"| {r['metric']} | {fmt(r['difference'],100)} | [{fmt(r['ci95_low'],100)}, {fmt(r['ci95_high'],100)}] |")
    lines+=['','| Arm | MLM sequence evaluations | Scorer encoded sequence evaluations | Generation s | Scoring s | Selection s |','|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['arm']} | {r['mlm_sequence_evaluations_full_arm']} | {r['scorer_encoded_sequence_evaluations_full_arm']} | {fmt(r['generation_seconds_full_arm'])} | {fmt(r['scoring_seconds_full_arm'])} | {fmt(r['selection_seconds_full_arm'])} |")
    lines+=['','The scoring count includes initialization and all nine occurrences in each round, without deduplication. Cached source-vector readouts are additional and recorded in summary.csv. Initialization is physically regenerated and charged to every arm. Equal trajectory counts do not imply equal costs relative to the historical generator.','', '| Arm | Stage | Mean target p | Predict target % | Target acceptance % | Unchanged transition % |','|---|---:|---:|---:|---:|']
    for r in stages:
        if r['scope']=='pooled_five_replicates' and r['stratum']=='overall' and r['arm'] in ('uniform',cfg.guided.main_arm):lines.append(f"| {r['arm']} | {r['stage']} | {fmt(r['mean_target_probability'])} | {fmt(r['predicted_target_fraction'],100)} | {fmt(r['target_specific_acceptance_fraction'],100)} | {fmt(r.get('unchanged_text_transition_fraction'),100)} |")
    lines+=['','| Positive λ arm | Mean local χ² divergence | Mean effective candidate occurrences |','|---|---:|---:|']
    for r in regularization:
        if r['scope']=='pooled_five_replicates' and r['stratum']=='overall':lines.append(f"| {r['arm']} | {fmt(r['chi2_mean'])} | {fmt(r['effective_candidates_mean'])} |")
    lines+=['','The positive-λ solver uses the exact simplex threshold solution in float64, then samples an occurrence. Greedy uses uniform tie breaking. The parent and all duplicate occurrences remain in the candidate distribution; effective candidate counts refer to occurrences, not distinct strings. No intermediate acceptance threshold or best-intermediate/replicate selection is used.','',
      'Historical single-completion refinement results and timing are retained in historical_comparison.csv. It is not the matched uniform selection control: it has one completion, no retained-parent candidate and no intermediate reward scoring. Its posthoc target success uses the newly fixed targets.','',
      'These are scorer prediction changes, not verified true label changes. The local χ² penalty is relative only to a nine-occurrence uniform distribution; it is not natural-data divergence or a pseudo-label precision certificate. No fluency, grammaticality or plausibility filters/metrics were introduced.','',
      'Raw selected states, targets, masks, rewards, selection probabilities, per-step scorer distributions and final trajectory CSVs are saved under results/guided_refinement/.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
