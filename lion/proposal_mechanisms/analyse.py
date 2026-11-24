"""Identity/diversity and outcome accounting; counts only, no independent-draw CIs."""
from collections import Counter,defaultdict
import json,time
import torch
from ..diagnostics.common import read_csv,write_csv,save_json,absolute
from .generate import root,ARMS


def summary(records,scope,timing,meta):
    selected=[r for r in records if scope=='pooled_five_draws' or r['draw_id']==0];n=len(selected)
    groups=defaultdict(list)
    for r in selected:groups[r['example_id']].append(r)
    counts=Counter(r['_ids'] for r in selected);within_unique=sum(len({r['_ids'] for r in rows}) for rows in groups.values());unique_sizes=Counter(len({r['_ids'] for r in rows}) for rows in groups.values())
    bad=[r for r in selected if not r['source_correct']];unchanged=[r for r in selected if r['identical_to_source']];eligible=[r for r in selected if r['eligible']]
    outcome=Counter(r['outcome'] for r in eligible);accepted=[r for r in selected if r['accepted']];distinct_accepted=len({(r['_ids'],r['focal_start'],r['focal_end']) for r in accepted})
    times=timing['draws'] if scope=='pooled_five_draws' else timing['draws'][:1];g=sum(t['generation_seconds'] for t in times);s=sum(t['scoring_seconds'] for t in times);compute=g+s
    result={'arm':timing['arm'],'scope':scope,'source_examples':len(groups),'raw_proposals':n,'incorrect_source_prediction_exclusions':len(bad),'incorrect_source_prediction_fraction':len(bad)/n,
        'identical_to_source_draws':len(unchanged),'identical_to_source_fraction':len(unchanged)/n,'incorrect_source_and_identical_overlap':sum(r['identical_to_source'] for r in bad),
        'unchanged_input_exclusions_among_correct_sources':sum(r['source_correct'] for r in unchanged),'eligible_proposals':len(eligible),'eligible_fraction':len(eligible)/n,
        'eligible_unchanged_label':outcome['unchanged_label'],'eligible_changed_label_below_tau':outcome['changed_below_tau'],'eligible_accepted':len(accepted),'acceptance_per_raw_proposal':len(accepted)/n,
        'distinct_outputs_within_source_sum':within_unique,'within_source_duplicate_draws_excess':n-within_unique,'mean_distinct_outputs_per_source':within_unique/len(groups),'sources_with_duplicate_draws':sum(len({r['_ids'] for r in rows})<len(rows) for rows in groups.values()),
        'distinct_outputs_per_source_histogram':json.dumps(dict(sorted(unique_sizes.items()))),'global_distinct_generated_sentences':len(counts),'global_duplicate_generated_draws_excess':n-len(counts),'global_duplicate_sentence_groups':sum(c>1 for c in counts.values()),
        'global_draws_in_duplicate_groups':sum(c for c in counts.values() if c>1),'cross_source_duplicate_excess_after_within_source_dedup':within_unique-len(counts),
        'distinct_accepted_focal_examples':distinct_accepted,'generation_seconds':g,'scoring_seconds':s,'generation_plus_scoring_seconds':compute,'proposals_per_compute_second':n/compute,
        'distinct_accepted_focal_examples_per_compute_second':distinct_accepted/compute,'logical_mlm_passes_per_proposal':5 if timing['arm']=='full_refined' else 1}
    rejected=len(eligible)-len(accepted)
    for key,label in [('eligible_unchanged_label','unchanged_label'),('eligible_changed_label_below_tau','changed_below_tau'),('eligible_accepted','accepted')]:
        result[key+'_fraction_of_eligible']=result[key]/len(eligible) if eligible else None
        if label!='accepted':result[key+'_fraction_of_eligible_rejections']=result[key]/rejected if rejected else None
    if scope=='pooled_five_draws':result.update(arm_wall_seconds=timing['wall_seconds'],proposals_per_arm_wall_second=n/timing['wall_seconds'],distinct_accepted_focal_examples_per_arm_wall_second=distinct_accepted/timing['wall_seconds'])
    return result


def experiment(cfg,output):
    folder=root(cfg)/'generate';meta=torch.load(folder/'sources.pt',weights_only=False);n=len(meta['gold']);labels=meta['labels'];offsets=meta['offsets'].tolist();sourcekeys=[tuple(meta['input_ids'][offsets[i]:offsets[i+1]].tolist()) for i in range(n)];timings=json.loads((folder/'timing.json').read_text());all_summary=[];initial_matches=0;initial_total=0
    for arm in ARMS:
        records=[]
        for draw in range(cfg.mechanisms.draws):
            path=folder/arm/f'draw{draw}.pt';saved=torch.load(path,weights_only=False);off=saved['offsets'].tolist()
            if arm=='full_refined':
                initial=torch.load(folder/'full_one_pass'/f'draw{draw}.pt',weights_only=False)['input_ids'];initial_total+=n
                initial_matches+=sum(torch.equal(saved['initial_input_ids'][off[i]:off[i+1]],initial[off[i]:off[i+1]]) for i in range(n))
            for i in range(n):
                key=tuple(saved['input_ids'][off[i]:off[i+1]].tolist());identical=key==sourcekeys[i];prediction=int(saved['prediction'][i]);confidence=float(saved['confidence'][i]);source_prediction=int(saved['source_prediction'][i]);gold=int(meta['gold'][i]);correct=source_prediction==gold;eligible=correct and not identical;changed=prediction!=source_prediction;accepted=eligible and changed and confidence>=cfg.mechanisms.tau
                if not correct:outcome='excluded_incorrect_source'
                elif identical:outcome='excluded_identical_input'
                elif not changed:outcome='unchanged_label'
                elif confidence<cfg.mechanisms.tau:outcome='changed_below_tau'
                else:outcome='accepted'
                records.append({'arm':arm,'example_id':i,'source_id':meta['source_id'][i],'sentence_id':int(meta['sentence'][i]),'focal_word_index':int(meta['focal'][i]),'focal_start':int(meta['focal_start'][i]),'focal_end':int(meta['focal_end'][i]),'draw_id':draw,
                    'source_gold_label':labels[gold],'source_prediction':labels[source_prediction],'source_confidence':float(saved['source_confidence'][i]),'assigned_label':labels[prediction],'scorer_confidence':confidence,'source_correct':correct,
                    'identical_to_source':identical,'eligible':eligible,'accepted':accepted,'outcome':outcome,'nonpadding_input_ids':json.dumps(key),'candidate_file':str(path.relative_to(root(cfg))),'_ids':key})
        timing=next(t for t in timings if t['arm']==arm)
        for scope in ('first_draw','pooled_five_draws'):all_summary.append(summary(records,scope,timing,meta))
        write_csv(output/f'{arm}_candidates.csv',[{k:v for k,v in r.items() if k!='_ids'} for r in records])
    write_csv(output/'summary.csv',all_summary);save_json(output/'summary.json',all_summary)
    previous=read_csv(absolute(cfg.generators.results)/'nearby_70_proposals.csv');local=torch.load(folder/'local70'/'draw0.pt',weights_only=False)
    reference_match=sum(int(p['sentence_id'])==int(meta['sentence'][i]) and int(p['focal'])==int(meta['focal'][i]) for i,p in enumerate(previous));prediction_match=sum(p['proposed']==labels[int(local['prediction'][i])] for i,p in enumerate(previous))
    protocol={'identity':'Exact tuple of all non-padding model-input token IDs, including boundaries; no decoding/normalization. Original reconstruction is counted separately from duplicate draws.',
        'global_duplicate_domain':'All generated sentences within the named arm and summary scope, across all cohort examples. Excess duplicates=N−number of distinct sentences; draws participating in repeated groups are additionally counted.',
        'focal_example_identity':'(generated non-padding token-ID tuple, focal start-inclusive/end-exclusive WordPiece span). Source IDs are not part of this deduplication key.',
        'within_source_group':'One original cohort example_id and its five draws. The fixed held-out cohort was originally sampled with replacement; repeated source/focal slots retain separate example IDs and are additionally covered by global duplicates.',
        'cohort_examples':n,'unique_original_source_focal_slots':len(set(zip(meta['source_id'],meta['focal'].tolist()))),'draws_per_example':cfg.mechanisms.draws,
        'sampling':'Existing selector.independent_fills and mlm_probabilities, temperature 1, special token IDs excluded, no top-k/top-p truncation, independent masked-position sampling. Same vocabulary rules in all arms, no redraw/filtering.',
        'local_generator':'Existing 70% ±5 word mask, half-up rounded word count, focal protected. First-draw rank/fill seeds match previous nearby_70 diagnostic; each subsequent draw resamples both random subset and fills.',
        'full_generator':'Every ordinary non-focal WordPiece position, punctuation included; boundaries and padding fixed. Position and length unchanged. No source context visible.',
        'refinement':'Arm-2 initialization recomputed with identical RNG stream, then four rounds remasking ceil(.25*ordinary_nonfocal_piece_count) uniformly chosen positions, conditioned only on the current generated context and unchanged focal. Initialization is physically rerun and included in generation time.',
        'refinement_initial_candidates_exactly_matching_arm2':initial_matches,'refinement_initial_candidates_compared':initial_total,
        'timing':'Generation includes subset selection and all logical MLM passes; scoring includes frozen re-encoding of the candidate plus focal mean-WordPiece aggregation/readout and original readout. GPU synchronized at timing boundaries. Proposals/compute-second and distinct accepted/compute-second use generation+scoring as denominator. Pooled arm-wall throughput also reported, including candidate transfers/saving. Empty editable contexts require no actual MLM forward.',
        'uncertainty':'Descriptive counts/rates only. No population confidence intervals; five draws share a source, and cohort samples can repeat source/focal slots.',
        'interpretation':'Acceptance is a scorer prediction change, never a verified change in the true label. No fluency, grammaticality, plausibility or malformation filters/metrics.',
        'previous_first_draw_cohort_id_matches':reference_match,'previous_first_draw_prediction_matches':prediction_match,'previous_first_draw_max_absolute_confidence_difference':max(abs(float(p['confidence'])-float(local['confidence'][i])) for i,p in enumerate(previous))}
    save_json(output/'protocol.json',protocol)
    for r in all_summary:print({k:r[k] for k in ('arm','scope','raw_proposals','identical_to_source_draws','eligible_proposals','eligible_accepted','distinct_accepted_focal_examples','generation_seconds','scoring_seconds')},flush=True)
    return {'summaries':len(all_summary),'candidates':3*n*cfg.mechanisms.draws}
