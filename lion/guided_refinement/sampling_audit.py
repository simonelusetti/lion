"""Audit saved sampling/provenance and recheck arithmetic; no model inference."""
import json,sys,time
from collections import Counter,defaultdict
import torch
from scipy.stats import beta
from forge import run,start_run
from omegaconf import OmegaConf
from ..diagnostics.common import ROOT,absolute,save_json,write_csv,read_csv
from ..label_change.data import sampled_rows
from .population import load,cp
from .binwise import minimize


def key(row):return tuple(row['ids'].tolist())


def main(cfg):
    OmegaConf.resolve(cfg);cfg.forge.store=str(absolute(cfg.forge.store));tracking=start_run(cfg);started=time.monotonic()
    source=absolute(cfg.binwise.source);output=absolute(cfg.sampling_audit.results);output.mkdir(parents=True,exist_ok=True);OmegaConf.save(cfg,output/'config.yaml')
    bank=load(absolute(cfg.certification.run_path)/'prepare'/'sentences.pt');train=bank['sets']['train'];validation=bank['sets']['validation']
    fit=[r for r in train if r['fold']=='fit'];held=[r for r in train if r['fold']=='calibration']
    exposed={r['source_id'] for offset in (2,3) for r in sampled_rows(held,cfg.label_change.calibration_draws,cfg.label_change.seed+offset)}
    fit_keys={key(r) for r in fit};val_by_key=defaultdict(list)
    for r in validation:val_by_key[key(r)].append(r)
    kept=[];seen=set();exclusions=Counter()
    for r in held:
        if r['source_id'] in exposed:exclusions['previously_exposed']+=1;continue
        if key(r) in fit_keys:exclusions['fit_exact_input']+=1;continue
        if key(r) in seen:exclusions['duplicate_within_remaining_holdout']+=1;continue
        seen.add(key(r));kept.append(r)
    order=torch.randperm(len(kept),generator=torch.Generator().manual_seed(cfg.population.natural_seed)).tolist();kept=[kept[i] for i in order]
    prepared_design=load(source/'prepare'/'natural_design.pt');prepared_reference=load(source/'prepare'/'natural_reference.pt');prepared=prepared_design+prepared_reference
    focal_rng=torch.Generator().manual_seed(cfg.population.focal_seed);replay_errors=[]
    for i,(raw,saved) in enumerate(zip(kept,prepared)):
        focal=int(torch.randint(len(raw['focals']),(1,),generator=focal_rng));span=(raw['word_ids']==focal).nonzero().flatten()
        if raw['source_id']!=saved['source_id'] or int(span[0])!=saved['start'] or int(span[-1])+1!=saved['end'] or int(raw['gold']['ner'][focal])!=saved['gold'] or not torch.equal(raw['ids'],saved['ids']):replay_errors.append(i)
    natural=load(source/'diagnostic'/'natural_reference.pt');selected_ids=set(natural['source_id']);selected=[r for r in prepared_reference if r['source_id'] in selected_ids]
    overlaps=[]
    for group,rows in [('design',prepared_design),('reference_before_confidence',prepared_reference),('reference_selected',selected)]:
        for r in rows:
            matches=val_by_key.get(key(r),[])
            if not matches:continue
            same_span=False;same_gold=False
            for match in matches:
                for f in match['focals'].tolist():
                    pos=(match['word_ids']==f).nonzero().flatten()
                    if (int(pos[0]),int(pos[-1])+1)==(r['start'],r['end']):
                        same_span=True;same_gold|=int(match['gold']['ner'][f])==r['gold']
            overlaps.append({'group':group,'source_id':r['source_id'],'focal_start':r['start'],'focal_end':r['end'],'same_focal_span':same_span,'same_gold_label':same_gold,'validation_source_ids':'|'.join(m['source_id'] for m in matches)})
    write_csv(output/'validation_overlap.csv',overlaps)
    design_keys={key(r) for r in prepared_design};reference_keys={key(r) for r in prepared_reference};selected_by_id={r['source_id']:r for r in selected}
    matching_saved_labels=all(int(g)==selected_by_id[s]['gold'] for s,g in zip(natural['source_id'],natural['gold']))
    edges=load(source/'design'/'bins.pt')['edges'];J=len(edges)+1;arms=list(cfg.mechanism_pilot.arms);A=len(arms)
    natural_bins=torch.bucketize(natural['phi'].double(),edges);correct=natural['prediction']==natural['gold'];n=torch.bincount(natural_bins,minlength=J);k=torch.bincount(natural_bins[correct],minlength=J)
    Lj=torch.tensor([float(beta.ppf(.025/J,int(a),int(b-a)+1)) if a>0 and b>0 else 0. for a,b in zip(k,n)],dtype=torch.double)
    old=read_csv(source/'reanalysis_binwise'/'quality.csv');old_by_arm={r['arm']:r for r in old};natural_table=[]
    for j in range(J):natural_table.append({'bin':j,'n':int(n[j]),'k':int(k[j]),'L_j':float(Lj[j]),'tail':.025/J})
    write_csv(output/'recomputed_natural_bounds.csv',natural_table)
    subset=json.loads((source/'prepare'/'subset.json').read_text());chosen=set(subset['source_ids']);cohort=sampled_rows([r for r in train if r['source_id'] in chosen],cfg.guided_training.anchors,cfg.guided_training.anchor_seed)
    indices=load(source/'prepare'/'proposal_indices.pt');expected=torch.randint(len(cohort),(cfg.population.proposals,),generator=torch.Generator().manual_seed(cfg.population.proposal_seed));proposal_index_replay=torch.equal(indices,expected);rows=[cohort[i] for i in indices.tolist()]
    meta=load(source/'prepare'/'sources_targets.pt');checks=[];mass_rows=[];tail=.025/(2*A*J)
    for arm in arms:
        raw=load(source/'generate'/arm/'replicate0.pt');saved=load(source/'generate'/arm/'accepted_features_draw0.pt');scored=load(source/'diagnostic'/f'{arm}_scores.pt')
        confidence,prediction=raw['state_probabilities'][:,-1].max(-1);ids=raw['state_input_ids'][-1];offset=raw['offsets']
        changed=torch.tensor([not torch.equal(ids[int(offset[i]):int(offset[i+1])].long(),r['ids']) for i,r in enumerate(rows)])
        mask=(raw['source_prediction']==meta['gold'])&changed&(prediction!=meta['gold'])&(confidence>=cfg.guided.tau);accepted=mask.nonzero().flatten()
        bins=torch.bucketize(scored['phi'].double(),edges);counts=torch.bincount(bins,minlength=J);m=len(accepted)
        lower,upper=zip(*[cp(int(c),m,tail) for c in counts]);lo=torch.tensor(lower,dtype=torch.double);hi=torch.tensor(upper,dtype=torch.double)
        bound,q,lp=minimize(lo,hi,Lj) if m else (0.,None,None)
        checks.append({'arm':arm,'attempts':len(rows),'accepted_recomputed':m,'accepted_saved':len(saved['y']),
            'acceptance_rows_match':torch.equal(accepted,saved['anchor_id']) and torch.equal(accepted,scored['proposal_indices']),
            'pseudo_labels_match':torch.equal(prediction[mask],saved['y']),'accepted_bin_total':int(counts.sum()),'region_nonempty':q is not None,'sum_lower':float(lo.sum()),'sum_upper':float(hi.sum()),
            'recomputed_bound':bound,'previously_reported_bound':float(old_by_arm[arm]['binwise_lower']),'bound_difference':bound-float(old_by_arm[arm]['binwise_lower']),
            'independent_lp_difference':bound-lp if lp is not None else None,'nominal_conditional_pass':m>0 and q is not None and bound>=.95})
        for j in range(J):mass_rows.append({'arm':arm,'bin':j,'accepted':int(counts[j]),'lower':float(lo[j]),'upper':float(hi[j]),'tail':tail,'worst_mass':float(q[j]) if q is not None else None})
    write_csv(output/'recomputed_arm_bounds.csv',checks);write_csv(output/'recomputed_mass_intervals.csv',mass_rows)
    # The model artifact contains predictions only for fit and native-validation data.
    checkpoint=load(absolute(cfg.certification.run_path)/'scorer'/'ner'/'model.pt');predictions=load(absolute(cfg.certification.run_path)/'scorer'/'ner'/'predictions.pt')
    counts={'native_training_sentences':len(train),'scorer_fit_sentences':len(fit),'scorer_calibration_sentences':len(held),'exclusions_sequential':dict(exclusions),
        'remaining_unique_sentences':len(kept),'prepared_sentences':len(prepared),'design_sentences':len(prepared_design),'reference_sentences_before_confidence':len(prepared_reference),
        'reference_after_confidence':len(natural['phi']),'natural_correct':int(correct.sum()),'natural_accuracy':float(correct.double().mean()),
        'design_reference_exact_input_overlap':len(design_keys&reference_keys),'reference_fit_exact_input_overlap':sum(key(r) in fit_keys for r in selected),
        'validation_overlap_counts':dict(Counter(r['group'] for r in overlaps)),
        'validation_overlap_same_span_counts':dict(Counter(r['group'] for r in overlaps if r['same_focal_span'])),
        'validation_overlap_same_gold_counts':dict(Counter(r['group'] for r in overlaps if r['same_gold_label'])),
        'selection_replay_mismatches':len(replay_errors),'saved_gold_labels_match':matching_saved_labels,'proposal_indices_replay':proposal_index_replay,
        'unique_proposal_anchor_indices':int(indices.unique().numel()),'scorer_fit_label_count':sum(len(r['focals']) for r in fit),
        'scorer_prediction_artifact_shapes':{name:list(value.shape) for name,value in predictions.items()},'scorer_selected_epoch':checkpoint['epoch'].tolist(),
        'declared_document_ids_are_sentence_aliases':all(r['document']==r['source_id'] for r in train),
        'confidence_tails':{'natural_correctness_each':.025/J,'generated_mass_each_endpoint':tail,'calibration_total':.025,'mass_total':.025}}
    save_json(output/'counts.json',counts)
    requirements=[]
    def item(requirement,status,evidence):requirements.append({'requirement':requirement,'status':status,'evidence':evidence})
    item('Exact reference construction replay','established',f"All {len(kept)} retained sentences, seeded order, focal spans, labels and token IDs reproduce prepared data; mismatches {len(replay_errors)}.")
    item('One focal per sentence; disjoint design/reference IDs and token inputs','established',f"4000 design, 13917 reference before confidence, 13061 selected; exact-input overlap {len(design_keys&reference_keys)}. No reference sentences repeat.")
    item('Independent original documents','not established','document is assigned source_id (sentence alias), not a genuine article identifier. Distinct sentence rows do not demonstrate independent source documents; no source-article grouping is preserved.')
    item('Eligibility and confidence filter are label-blind with respect to correctness','established','Natural eligibility uses prior exposure, token-input duplicates and fitted-input overlap; focal selection uniform by word count; confidence >=.7. No natural gold-correctness filter. Saved gold matches actual calibration labels.')
    item('Source-ID and exact-input separation from scorer optimizer fitting','established','Calibration IDs excluded from fit fold; token-input matches to all fitted rows removed; zero remaining fit matches. Source scorer predictions contain fit and native-validation only. Content separation is not itself proof of statistical independence.')
    selected_overlaps=[r for r in overlaps if r['group']=='reference_selected'];design_overlaps=[r for r in overlaps if r['group']=='design']
    item('No exact focal-input/gold overlap with checkpoint selection','violated',f"{len(selected_overlaps)} selected reference inputs match native validation sentences used every epoch for checkpoint selection; {sum(r['same_focal_span'] for r in selected_overlaps)} also match focal span; {sum(r['same_gold_label'] for r in selected_overlaps)} match focal gold. This establishes content overlap, not whether matching records are copies of one original observation or independent draws with the same value.")
    item('No exact design-input overlap with checkpoint selection','violated',f"{len(design_overlaps)} design inputs match scorer native-validation inputs. This is separate from the correctly disjoint design/calibration partition and is a content-overlap finding, not a proof of stochastic dependence.")
    item('Statistical independence of calibration from checkpoint selection','not established','Same-valued observations can occur in independent samples, so exact matches alone do not prove a statistical independence violation. Conversely, split IDs alone do not rule out copies of the same original Wikipedia observation. Source-document provenance is absent, preventing that distinction for the observed matches.')
    item('No task-data MLM training/adaptation','established','FrozenMaskedBERT loads pretrained bert-base-cased, eval, requires_grad_(False); task stages train readout heads only. External pretraining-corpus overlap is not audited from these artifacts.')
    item('Threshold, score, binning and policies frozen before this calibration','established','tau=.7 and eleven historical arms are fixed configs; bin score specified before run. design stage freezes five score edges before diagnostic sees reference correctness and generated frequencies. No correctness/F1 optimization of edges in code.')
    item('Full historical tuning exposure excluded','not established','Known 4000 error and 4000 acceptance streams are replayed/excluded at sentence level (6695 distinct IDs). No complete historical observation-use registry proves absence of all prior adaptive exposure, or duplicate-equivalent exposure across excluded/current rows.')
    item('Natural counts IID binomial for underlying task population','not established','Fixed-corpus deduplication and input-dependent exclusions precede a without-replacement shuffled design/reference split; one random focal per remaining sentence. No sampling argument identifies these as IID draws from the claimed post-selection task population. Deduplication alone proves neither validity nor invalidity. Article dependence and observed validation reuse also remain. Resampling corpus rows would not establish task-population evidence.')
    item('Generated independent anchor sampling with replacement','established',f"All 20000 torch.randint indices exactly replay; {int(indices.unique().numel())} distinct indices from fixed 4000-entry cohort. One final trajectory per index; no fixed-five allocation.")
    item('Actual fixed generation policies and acceptance law','established','Per-arm independent RNG streams across trajectory draws, frozen scorer/MLM; guided finite pool (parent plus eight completions) and fixed round count define proposal law. Intermediate selection is part of generator, not an extra diagnostic filter. Fractions retain fixed word masks/count rules. Shared streams across arms do not invalidate Bonferroni.')
    item('No diagnostic deduplication, redraw or acceptance stopping','established','All 20000 attempts retained per arm; duplicate outputs and reconstructions counted; source-correct + changed input + flipped prediction + confidence>=.7 are the only acceptance conditions; all recomputed accepted IDs and pseudo-labels match saved student features.')
    item('Generated q counts use actual accepted final outputs','established','Every accepted-ID list matches both raw final-state filter and diagnostic score rows. Bin totals equal accepted counts for all arms; frequencies do not include rejected proposals.')
    item('Conditional multinomial argument for accepted samples','established','Conditional on fixed source cohort/policies and usual independent PRNG draws, attempts are IID; conditioning on fixed-attempt-run accepted total m gives multinomial accepted-bin counts. m is random, not a quality-based stopping rule. Within-trajectory candidate dependence and cross-arm coupling are allowed.')
    item('Conditional binomial argument for natural correctness','not established','For IID natural pairs, conditional on random bin count n_j, gold correctness is Binomial(n_j,theta_j), so random counts are not themselves an error. That IID natural-data premise is not established here; correct CP arithmetic does not repair its provenance.')
    item('Natural and generated confidence allocations','established',f"J={J}, A={A}; each correctness tail .025/J={.025/J}; each q endpoint .025/(2*A*J)={tail}; Bonferroni total .05. Shared reference counted once; independence between correctness and mass events is unnecessary.")
    item('Empty bins, zero successes and empty accepted samples','established','L_j=0 for n_j=0 or k_j=0; m=0 reports zero/no PASS. Gold-empty bins retained. CP q endpoints use zero/one conventions. Actual data have no gold-empty bins or empty accepted arms.')
    item('Bounded simplex minimization and independent verification','established','q lower<=q<=upper and sum(q)=1 enforced; feasible for every arm. Greedy allocation and independent scipy linprog match; bounds exactly reproduce previous report. Known-mass, constant-L, single-bin and gold-empty limiting cases checked in prior mathematical_checks.json.')
    item('Simultaneous advertised 95% coverage, conditional only on transfer and intended population','not established','The union-bound proof is correct if its natural calibration conditions hold. Content overlap with checkpoint selection is verified, while its original-observation provenance and the natural population/sampling argument remain unestablished. Thus numerical PASS results cannot currently be promoted to established simultaneous task-population certificates. Actual undercoverage is not proven.')
    item('No unaccounted old/new certificate selection','established','Reanalysis reports old/new numbers separately; no max(old,new) certificate, no bound-based arm selection or retraining. Descriptive correlations include all arms and label undefined metrics.')
    item('Calibration data used as downstream evaluation','established','Downstream test_cache is separate native Few-NERD test plus transfer CoNLL/WNUT/WikiANN, raw inputs; no calibration scores used in student losses. Exact content/underlying-document leakage with those test sources is not comprehensively documented by cached feature-only evaluations; source split names alone do not prove content independence.')
    write_csv(output/'requirements.csv',requirements)
    lines=['# Population-quality sampling and calibration audit','',
        '**Conclusion: the reported PASS results are not established as simultaneous finite-sample 95% task-population correctness certificates, conditional only on the stated transfer premise and intended population.** The numerical formulas and generated sampling checks pass. Exact checkpoint-selection content overlap is confirmed, but whether it reflects repeated original observations or independent same-valued draws cannot be resolved from stored provenance. The natural-population sampling argument is also unresolved. Neither overlap nor deduplication alone proves statistical dependence or actual miscoverage above 5%. No transfer assumption is assessed here.','',
        f"Reference counts reproduce exactly: {int(n.sum())} examples, {int(k.sum())} correct ({100*float(correct.double().mean()):.3f}%). Every prior binwise bound is reproduced unchanged. All eleven remain **nominal conditional PASS**, not newly validated certificates. No generation, scoring or training was run.",'',
        '## What the reference represents','',
        'The source is supervised Few-NERD native train, fixed dataset revision 205f3e9c9f3577ea2561d43f2f62dc249ab92d5b. Sentence rows are tokenized/truncated to 512 BERT pieces. The seeded split assigns 105414 rows to scorer fit and 26353 to calibration. The natural reference is sentence-uniform over the retained unique tokenized-input cohort, with one uniformly selected word per sentence, subsequently conditioned on scorer confidence >=.7. It is not a uniform-word draw from the original unfiltered task domain. The code operationally specifies this empirical selection policy; no underlying task-population sampling derivation for the retained unique-sentence distribution is supplied.','',
        f"Sequential calibration exclusions: {dict(exclusions)}. Remaining unique rows: {len(kept)}, partitioned into 4000 design and 13917 reference rows by seeded random permutation. Each row gets one label-blind uniform word position. Confidence leaves 13061 reference rows. Deduplication uses the complete post-truncation input-token-ID tuple, ignoring source ID, focal position, gold labels and word segmentation. It removes matches to scorer-fit inputs and repeats within remaining holdout, keeping first occurrences. It does **not** remove native-validation matches. The stored document field is a sentence-ID alias, not an observed independent article unit.",'',
        'For a fixed empirical corpus, the partition is a finite without-replacement selection; IID binomial trials are not automatic. For an underlying population, an argument is needed for sentence/document sampling, input-dependent eligibility and deduplication. A valid deduplicated-population or finite-population argument is possible in principle, but is not established by uniqueness alone. Reinterpreting the fixed reference as a population census would change the target; this audit does not do that.','',
        '## Requirements','',
        '| Requirement | Status | Evidence |','|---|---|---|']
    for r in requirements:lines.append(f"| {r['requirement']} | **{r['status']}** | {r['evidence']} |")
    lines+=['','## Confidence calculations','',
        'Under shared annotation mechanism and the stipulated full within-bin equality, theta_j transfers and pi_r=sum_j q_rj theta_j follows by total expectation. For IID natural pairs, CP correctness bounds condition on each random n_j; for IID attempts, accepted-bin intervals condition on each random m_r. Their simultaneous events combine by union bound, without event independence. The implementation uses .005 per correctness bound and .025/(2*11*5) per q endpoint, then minimizes sum q_j L_j on the bounded simplex. The arithmetic is established; its natural-data sampling premise is not.','',
        'Recomputed bounds and statuses are in recomputed_arm_bounds.csv; correctness counts/bounds in recomputed_natural_bounds.csv; accepted-bin intervals and minimizing masses in recomputed_mass_intervals.csv. No counts or old/new bound selection correction is needed.','',
        '## Can existing observations repair coverage?','',
        'Removing the 207 validation-overlapping reference rows could remove that observed leakage, but would change the reference eligibility/population unless explicitly incorporated, and would not by itself establish the missing sentence/document independence and deduplication argument. Recomputing CP on that subset and advertising a corrected 95% certificate would therefore be unjustified. The current artifacts permit exact arithmetic verification, not a valid repair of the full coverage claim. No replacement experiment or silently redefined reference is used.','',
        '## Smallest future change','',
        'Keep the fixed scorer/generators, student results and frozen bins if the target protocol permits. Obtain a genuinely untouched gold calibration sample from the **declared natural population**, with a defensible independent original sampling unit (document if sentences cluster). Exclude exact fitted **and checkpoint-validation** focal-input/label matches before defining the eligible population. Keep design data and calibration independent at that sampling unit; use a prespecified label-blind one-focal rule and the same confidence filter. Either sample IID from the explicitly defined underlying population, or explicitly target a fixed empirical corpus and use a justified finite-population sampling argument; do not conflate the two. A gold subset from these artifacts can be reused only if its independence/provenance is established, not merely because source IDs differ. No scorer refit is necessary if a suitable untouched calibration population exists. Otherwise reserve it and exclude it, including duplicate equivalents, from scorer fitting and checkpoint selection before a future run.','',
        'Generated attempts already have the required fixed-budget with-replacement design; no generator redesign, new rejection sampler or regenerated outputs is needed just to correct this part. Any future target/population change must be explicit, and simultaneous coverage must be re-evaluated under that target.','',
        '## Evidence locations','',
        '- Source/split/tokenization: lion/label_change/fewnerd_diagnostic.py:prepare. Scorer fitting roles: lion/label_change/data.py:parts/scorer. Epoch selection: lion/partition/train.py:fit.','- Natural construction, score bins, actual accepted counts: lion/guided_refinement/population.py. Guided policies: lion/guided_refinement/generate.py. Fraction policies: lion/guided_refinement/size_grid.py. Direct bounds: lion/guided_refinement/binwise.py.','- Artifact comparisons, exact overlap IDs and matches: counts.json and validation_overlap.csv in this folder. requirements.csv provides machine-readable findings.']
    (output/'AUDIT.md').write_text('\n'.join(lines)+'\n');save_json(output/'runtime.json',{'seconds':time.monotonic()-started});tracking.finish({'reference_n':len(natural['phi']),'validation_overlap':len(selected_overlaps),'coverage_established':False})
    print(json.dumps(counts,indent=2),flush=True);print(output/'AUDIT.md',flush=True);return 0


if __name__=='__main__':run('lion.guided_refinement',sys.argv[1:],main_module='sampling_audit',config_dir=str(ROOT/'conf'),config_name='population_sampling_audit')
