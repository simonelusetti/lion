"""Compact per-task result tables and predeclared hypothesis verdicts."""
import csv,json,shutil,time,math
from ..diagnostics.common import ROOT,save_json
from ..invariance.data import paths


def experiment(cfg,output):
    results,_=paths(cfg);report=ROOT/'report';tables=report/'tables'/'pilot';tables.mkdir(parents=True,exist_ok=True)
    for stage in ('prepare','select','analyze'):
        for path in (results/stage).glob('*.csv'):shutil.copy2(path,tables/path.name)
    shutil.copy2(results/'select'/'selections.json',tables/'selections.json')
    rows=list(csv.DictReader((results/'analyze'/'summary.csv').open()));contrasts=list(csv.DictReader((results/'analyze'/'contrasts.csv').open()));validity=list(csv.DictReader((results/'analyze'/'scorer_validity.csv').open()));calibration=list(csv.DictReader((results/'analyze'/'calibration.csv').open()))
    timing=json.loads((results/'timing.json').read_text());timing['ended_unix']=time.time();timing['wall_clock_seconds']=timing['ended_unix']-timing['started_unix'];save_json(results/'timing.json',timing);shutil.copy2(results/'timing.json',tables/'timing.json')
    choices=json.loads((results/'select'/'selections.json').read_text())
    def summary(task,rule,reg,metric):return next(r for r in rows if (r['task'],r['rule'],r['regularizer'],r['metric'])==(task,rule,reg,metric))
    def display(row,percent=True):
        factor=100 if percent else 1
        return f"{factor*float(row['mean']):.2f} ± {factor*float(row['seed_sd']):.2f} [{factor*float(row['ci_low']):.2f}, {factor*float(row['ci_high']):.2f}]"
    def point(task,rule,reg,metric,scale=100):return scale*float(summary(task,rule,reg,metric)['mean'])
    def contrast(task,name,reg='none'):return next(r for r in contrasts if (r['task'],r['contrast'],r['regularizer'])==(task,name,reg))
    def verdict(records):
        if all(float(r['ci_low'])>0 for r in records):return 'supported'
        if any(float(r['ci_high'])<0 for r in records):return 'contradicted'
        return 'inconclusive'
    lines=['# Mask-rule pilot grid: results','',f"Wall-clock: **{timing['wall_clock_seconds']/3600:.2f} hours**. Four tasks × {len(cfg.pilot.rules)} rules × {len(cfg.pilot.regularizers)} regularizers × five rotations × three seeds. Scorers, λ pilots and the 20-subset validity readouts are additional fits.",'','Accuracy/F1/dependence/leakage values are percentages; gains are percentage points. Brackets on ID/OOD accuracy are paired document-bootstrap 95% intervals; other columns give rotation-equal means. All metrics with seed SD, intervals, contributing rotations and valid-bootstrap counts are in summary.csv. “p” is a descriptive mean of cell-level Bonferroni p-values, not an aggregate significance test. H1–H5 use the unregularized grid; H6 compares regularizers. The best-fixed comparator is the per-rotation/seed OOD oracle maximum, not a test-selected deployed model.']
    for task in cfg.invariance.datasets:
        lines+=['',f'## {task}','','| Rule × penalty | ID accuracy / F1 | OOD accuracy / F1 | Gain ID / OOD / shrink | Invariance F / p / reject % | Genre BA | Mask BA |','|---|---|---|---|---|---:|---:|']
        for rule in cfg.pilot.rules:
            for reg in cfg.pilot.regularizers:
                ia=summary(task,rule,reg,'id_accuracy');oa=summary(task,rule,reg,'ood_accuracy')
                lines.append(f"| {rule} × {reg} | {display(ia)} / {point(task,rule,reg,'id_macro_f1'):.2f} | {display(oa)} / {point(task,rule,reg,'ood_macro_f1'):.2f} | {point(task,rule,reg,'gain_id'):+.2f} / {point(task,rule,reg,'gain_ood'):+.2f} / {point(task,rule,reg,'gain_shrink'):+.2f} | {point(task,rule,reg,'invariance_statistic',1):.2f} / {point(task,rule,reg,'invariance_p',1):.3f} / {point(task,rule,reg,'rejected'):.1f} | {point(task,rule,reg,'genre_dependence'):.2f} | {point(task,rule,reg,'mask_leakage'):.2f} |")
        ranks=[float(r['spearman']) for r in validity if r['task']==task];rho=sum(ranks)/len(ranks);null=[float(r['rejection_rate']) for r in calibration if r['task']==task and r['regularizer']=='none'];rate=sum(null)/len(null)
        unreliable=rho<.5 or not math.isfinite(rho) or rate>.1
        h1=[contrast(task,n) for n in ('tree1_full','lin1_full','tree1_rand','lin1_rand')]
        if task=='ud_deprel':h1.append(contrast(task,'tree1_lin1'))
        h3a=contrast(task,'prev_full');h3b=contrast(task,'prev_rejection_minus_full')
        h3='supported' if float(h3a['ci_low'])>0 and float(h3b['ci_high'])<0 else ('contradicted' if float(h3a['ci_high'])<0 or float(h3b['ci_low'])>0 else 'inconclusive')
        h4=verdict([contrast(task,'inv_suff'),contrast(task,'inv_best_fixed')]);h5=contrast(task,'free_genre');margin=cfg.pilot.equivalence_margin
        equivalence='supported' if float(h5['ci_low'])>=-margin and float(h5['ci_high'])<=margin else ('contradicted' if float(h5['ci_low'])>margin or float(h5['ci_high'])<-margin else 'inconclusive')
        if unreliable:h4=equivalence='inconclusive'
        lines+=['',f"- H1 locality/syntax: **{verdict(h1)}**.",'- H2 covariate shift vs non-invariance: **inconclusive as a causal distinction**; fitted-residual pass/reject patterns are in H2_patterns.csv.',f'- H3 previous-sentence context: **{h3}**.',f'- H4 invariance-based selection: **{h4}**.',f'- H5 annotation-free environments: **{equivalence}**, equivalence margin ±{100*margin:.1f} pp.',f"- H6 conditional penalty: **{verdict([contrast(task,'penalty_full_minus_tree','cond')]) if 'cond' in cfg.pilot.regularizers else 'inconclusive (not run)'}**.",f'\nScorer-validity mean Spearman: **{rho:.3f}** (poor if <.5); random-document calibration rejection rate: **{100*rate:.1f}%** (α=5%, conservative Bonferroni). Selection evidence is discounted if mean ranking <.5 or null rejection >10%.']
        fallback={rule:sum(r['fallback'][rule] for r in choices[task].values()) for rule in ('sel_inv_genre','sel_inv_free')}
        lines.append(f"Fallback to word, genre/free: {fallback['sel_inv_genre']}/5 and {fallback['sel_inv_free']}/5. Chosen subsets and descriptive ICP intersections are in selections.json.")
    lines+=['','## Scope and limitations','','All 11 mandatory rules and both regularizers were run; no cost-driven rule drops. The optional per-word-class selector was omitted. The prompt’s 12-rule total includes that optional rule. This pilot uses fixed document-spread focal samples, at most two per sentence, with caps 256/64/256 per genre for native train/validation/test; actual counts are in samples.csv. Probe minimum supports are reduced to 8/4/4 for this pilot and coverage is explicit. Native validation documents select E3 probe checkpoints; held-document folds use five EWT/two GUM folds. All masks are re-encoded, with identical masked strings deduplicated. Scorers use eight cached uniformly sampled subsets per focal as their training mask pool. Full-prev prepends as much of the previous sentence as fits in 512 WordPieces, preserving the complete current cached sentence; changed absolute positions and unavailable/truncated previous context limit H3 interpretation.','','A residual-mean rejection can arise from estimation or calibration error, not only a changing true conditional. Non-rejection does not establish invariance, and ICP intersections are descriptive, not proven necessary slots. Bonferroni controls across classes, not the search over 256 subsets. Genre-free clustering uses no genre annotation as its environment labels, but the shared rotation and balanced sampling design still uses genres. Gold syntax itself can reveal annotation-related structure; the mask-shape probe measures some of this. No test labels select subsets or λ. Intervals condition on fitted readouts, frozen scorers and selections; models/probes are not refitted in bootstrap. Nominal intervals accompany only the declared H1–H6 contrast family; other measurements are descriptive.','','[Implementation](IMPLEMENTATION_5.md); [tables](tables/pilot/).']
    (report/'RESULTS_5.md').write_text('\n'.join(lines)+'\n')
    return {'report':str(report/'RESULTS_5.md'),'wall_clock_seconds':timing['wall_clock_seconds']}
