"""Conditional certification, generation yield and matched raw-test comparisons."""
import json,shutil,time
import torch
from scipy.stats import beta
from ..diagnostics.common import ROOT,read_csv,write_csv,save_json
from . import data


def certify(cfg,output):
    error=torch.load(data.root(cfg)/'generate'/'error_sample.pt',weights_only=False);high=error['confidence']>=cfg.label_change.tau;wrong=error['prediction']!=error['gold'];n=len(high);k=int((high&wrong).sum())
    proposals=read_csv(data.root(cfg)/'generate'/'acceptance_sample.csv');m=len(proposals);accepted=sum(r['accepted']=='True' for r in proposals);tail=cfg.label_change.alpha/2
    e_U=float(beta.ppf(1-tail,k+1,n-k)) if k<n else 1.;a_L=float(beta.ppf(tail,accepted,m-accepted+1)) if accepted else 0.
    result={'tau':cfg.label_change.tau,'alpha':cfg.label_change.alpha,'joint_confidence_at_least':1-cfg.label_change.alpha,'individual_one_sided_confidence':1-tail,'error_draws':n,'acceptance_draws':m,'high_confidence_errors':k,'accepted':accepted,
            'scorer_accuracy':float((~wrong).double().mean()),'scorer_error':float(wrong.double().mean()),'high_confidence_share':float(high.double().mean()),'conditional_high_confidence_error':float(wrong[high].double().mean()) if high.any() else None,'e_hat':k/n,'a_hat':accepted/m,'e_U':e_U,'a_L':a_L,'ideal_sampler_pi_L':max(0.,1-e_U/a_L) if a_L>0 else 0.,'desired_precision':cfg.label_change.desired_precision,'delta_max':(1-cfg.label_change.desired_precision)*a_L-e_U,'vacuous_denominator':a_L<=0,
            'status':'CONDITIONAL ONLY: delta_Q unknown; ideal-sampler result does not establish MLM precision.',
            'reference':'Exact one-sided Clopper-Pearson bounds, Bonferroni alpha/2 per quantity, conditional on the fixed held-out empirical corpus and independent with-replacement sampling streams. Extending to unseen populations or the training-candidate distribution requires additional assumptions; delta_Q must cover the joint distribution of text and true modified labels, not text alone.'}
    save_json(output/'certification.json',result);write_csv(output/'certification.csv',[result]);return {'ideal_sampler_pi_L':result['ideal_sampler_pi_L'],'delta_max':result['delta_max']}


def experiment(cfg,output):
    root=data.root(cfg);tables=ROOT/'report'/'tables'/'label_change';tables.mkdir(parents=True,exist_ok=True);scores=read_csv(root/'train'/'per_seed.csv');cert=json.loads((root/'certify'/'certification.json').read_text());gen=json.loads((root/'generate'/'generation.json').read_text());bank=data.data.load(cfg);labels=bank['labels']['ner'];proposals=read_csv(root/'generate'/'proposals.csv');accepted=[r for r in proposals if r['accepted']=='True'];transitions=[]
    for a in range(len(labels)):
        for b in range(len(labels)):
            count=sum(int(r['gold'])==a and int(r['proposed'])==b for r in accepted)
            if count:transitions.append({'original':labels[a],'proposed':labels[b],'accepted':count,'type_change':labels[a].removeprefix('B-').removeprefix('I-')!=labels[b].removeprefix('B-').removeprefix('I-')})
    write_csv(output/'label_transitions.csv',transitions);summary=[]
    for target,space in sorted({(r['target'],r['space']) for r in scores}):
        a=torch.tensor([float(r['span_f1']) for r in scores if (r['target'],r['space'],r['arm'])==(target,space,'baseline')]);b=torch.tensor([float(r['span_f1']) for r in scores if (r['target'],r['space'],r['arm'])==(target,space,'augmented')]);d=b-a
        summary.append({'target':target,'space':space,'baseline_mean':float(a.mean()),'baseline_sd':float(a.std()),'augmented_mean':float(b.mean()),'augmented_sd':float(b.std()),'difference_mean':float(d.mean()),'difference_sd':float(d.std())})
    write_csv(output/'comparison.csv',summary);mechanism=[];probe=torch.load(root/'generate'/'changed_probe.pt',weights_only=False)
    if len(probe['gold']):
        for seed in cfg.augmentation.seeds:
            a=torch.load(root/'train'/f'seed{seed}'/'probe_original.pt',weights_only=False);b=torch.load(root/'train'/f'seed{seed}'/'probe_modified.pt',weights_only=False)
            for h,arm in enumerate(('baseline','augmented')):mechanism.append({'seed':seed,'arm':arm,'rows':len(a),'prediction_flip_rate':float((a[:,h]!=b[:,h]).double().mean()),'agreement_with_scorer_proposed_label':float((b[:,h]==probe['proposed']).double().mean())})
    write_csv(output/'mechanism.csv',mechanism)
    for p in [root/'prepare'/'partitions.csv',root/'train'/'per_seed.csv',root/'train'/'budgets.csv',root/'certify'/'certification.csv',root/'certify'/'certification.json',root/'generate'/'generation.json',*output.glob('*.csv')]:shutil.copy2(p,tables/p.name)
    timing=json.loads((root/'timing.json').read_text());timing['ended_unix']=time.time();timing['wall_seconds']=timing['ended_unix']-timing['started_unix'];save_json(root/'timing.json',timing);save_json(tables/'timing.json',timing)
    typechanges=sum(int(r['accepted']) for r in transitions if r['type_change']);parts=read_csv(root/'prepare'/'partitions.csv');fit=next(r for r in parts if r['role']=='fit');held=next(r for r in parts if r['role']=='calibration')
    lines=['# Pilot: context-changing, focal-label-changing augmentation','',f"CoNLL-2003 pilot: {fit['sentences']} fit sentences, {held['sentences']} held-out training sentences for calibration. The scorer and both arms use exactly the same original fit sentences; calibration sentences are excluded from every fitted model. Native dev selects checkpoints. All CoNLL/WNUT/WikiANN evaluations use untouched raw test inputs and the existing span/label-mapping code.",'',
       'Frozen bert-base-cased block 12, existing MLP and ten-epoch recipe, seeds 0/1/2 for the comparison. Identical initialization, original batches/order, original label presentations and optimizer steps in both arms. The augmented arm appends focal-only examples in their observed dataset proportion; extra labelled-token counts are recorded in budgets.csv. Mean loss is over labelled tokens; original gold labels are unchanged.',
       '',f"One proposal per fit focal: uniformly choose a valid neighbour at word distance 1–5, mask all pieces of that neighbour and sample once with the existing one-pass MLM fill implementation. Piece counts and focal pieces are preserved. Multi-piece neighbour fills use independent piece samples from the jointly masked pass; this is an approximate word sampler. Accept only an actual case-sensitive word change, a different native BIO prediction and confidence ≥{cfg.label_change.tau}. No retries or learned selectors.",'',
       f"Generation: **{gen['accepted']:,}/{gen['proposals']:,} accepted ({100*gen['acceptance_rate']:.3f}%)**. Of these, {typechanges} change entity type/O and {gen['accepted']-typechanges} change only B/I. [Label transitions](tables/label_change/label_transitions.csv).",'',
       f"Scorer calibration accuracy {100*cert['scorer_accuracy']:.2f}%; overall error {100*cert['scorer_error']:.2f}%; high-confidence share {100*cert['high_confidence_share']:.2f}%; conditional error among high-confidence predictions {100*(cert['conditional_high_confidence_error'] if cert['conditional_high_confidence_error'] is not None else float('nan')):.2f}%.",
       '',f"**Conditional bounds:** e_U={cert['e_U']:.6f}, a_L={cert['a_L']:.6f}, ideal-sampler π_L={cert['ideal_sampler_pi_L']:.4f}; δ_max={cert['delta_max']:.6f} for desired precision .95. Joint confidence at least .95 (one-sided .975 Clopper–Pearson bounds for each quantity). {'The zero acceptance lower bound makes the precision bound vacuous.' if cert['vacuous_denominator'] else ''}",
       '', 'The two calibration streams independently sample uniform focal slots with replacement from the same fixed held-out corpus. Bounds are exact for those empirical sampling risks conditional on that corpus; repeated slots are not extra evidence about a broader population. δ_Q is unknown and must cover the relevant joint distribution of text and true modified labels, including any reference-to-generator shift. A text-marginal discrepancy alone is insufficient. **No unconditional certification is claimed, and δ_Q=0 is not established.** Positive δ_max describes allowable discrepancy under the stated inequality, not a measured discrepancy.',
       '', '| Test | Label space | Baseline span F1 % ± SD | Augmented span F1 % ± SD | Paired Δ percentage points ± SD |','|---|---|---|---|---|']
    for r in summary:lines.append(f"| {r['target']} | {r['space']} | {100*r['baseline_mean']:.2f} ± {100*r['baseline_sd']:.2f} | {100*r['augmented_mean']:.2f} ± {100*r['augmented_sd']:.2f} | {100*r['difference_mean']:+.2f} ± {100*r['difference_sd']:.2f} |")
    lines+=['',f"Runtime: {timing['wall_seconds']/60:.1f} min. [Per-seed scores and budgets](tables/label_change/). [Mechanism diagnostic](tables/label_change/mechanism.csv) measures flips and agreement with the scorer on accepted held-out calibration candidates; this is descriptive pseudo-label agreement, **not gold accuracy**.",'','[Implementation](IMPLEMENTATION_18.md). Proposals and accepted feature arrays: `results/label_change/generate/`.']
    (ROOT/'report'/'RESULTS_18.md').write_text('\n'.join(lines)+'\n');return {'report':'report/RESULTS_18.md'}
