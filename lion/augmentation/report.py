"""Seed-paired span results and single-draw augmentation diagnostics."""
import csv,json,shutil,time
from collections import defaultdict
import torch
from ..diagnostics.common import ROOT,write_csv,save_json
from .data import paths,ARMS
from .additions import experiment as additions


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def experiment(cfg,output):
    bounds,cuts,scorer=additions(cfg,output)
    root=paths(cfg);rows=read(root/'train'/'per_seed.csv');groups=defaultdict(list);summaries=[];paired=[];differences=[]
    for row in rows:groups[(row['target'],row['label_space'],row['arm'])].append(row)
    for (target,space,arm),values in groups.items():
        record={'target':target,'label_space':space,'arm':arm,'seeds':len(values)}
        for metric in ('span_f1','precision','recall'):
            x=torch.tensor([float(r[metric]) for r in values]);record[metric+'_mean']=float(x.mean());record[metric+'_sd']=float(x.std())
        summaries.append(record)
    for target,space in sorted({(r['target'],r['label_space']) for r in rows}):
        for control in ('baseline','random_augmented'):
            records=[]
            for seed in cfg.augmentation.seeds:
                a=next(r for r in rows if (r['target'],r['label_space'],r['arm'],int(r['seed']))==(target,space,'selector_augmented',seed))
                b=next(r for r in rows if (r['target'],r['label_space'],r['arm'],int(r['seed']))==(target,space,control,seed))
                record={'target':target,'label_space':space,'seed':seed,'comparison':'selector_minus_'+control,
                        **{m:float(a[m])-float(b[m]) for m in ('span_f1','precision','recall')}}
                paired.append(record);records.append(record)
            summary={'target':target,'label_space':space,'comparison':'selector_minus_'+control}
            for metric in ('span_f1','precision','recall'):
                x=torch.tensor([r[metric] for r in records]);summary[metric+'_mean']=float(x.mean());summary[metric+'_sd']=float(x.std())
            differences.append(summary)
    for name,values in [('summary',summaries),('paired_seeds',paired),('paired_summary',differences)]:write_csv(output/(name+'.csv'),values)
    report=ROOT/'report';tables=report/'tables'/'augmentation';tables.mkdir(parents=True,exist_ok=True)
    for stage in ('prepare','bounds','generate','train','report'):
        for p in (root/stage).glob('*.csv'):shutil.copy2(p,tables/(stage+'_'+p.name))
    partitions=read(root/'prepare'/'partitions.csv');diagnostics=read(root/'generate'/'diagnostics.csv')
    source=json.loads((root/'prepare'/'source.json').read_text());generation=json.loads((root/'generate'/'timing.json').read_text());training=json.loads((root/'train'/'timing.json').read_text())
    timing=json.loads((root/'timing.json').read_text());timing.setdefault('ended_unix',time.time());timing['wall_seconds']=timing['ended_unix']-timing['started_unix'];save_json(root/'timing.json',timing)
    for name,value in [('timing',timing),('generation_timing',generation),('training_timing',training),('source',source)]:save_json(tables/(name+'.json'),value)
    chosen=[p for p in partitions if p['partition'] in ('g1','g2')];sentences=sum(int(p['sentences']) for p in chosen);words=sum(int(p['words']) for p in chosen)
    def display(r,m):return f"{100*r[m+'_mean']:.2f} ± {100*r[m+'_sd']:.2f}"
    lines=['# Experiment 8: original-plus-fill augmentation','',f'All three arms train on the same **{sentences:,} original CoNLL-2003 sentences ({words:,} labelled words): T1 ∪ T2**. Both scorer partitions are excluded. Exact sentence IDs are in prepare_training_sentences.csv.','',f"The scorer, cross-fitted word-ranking selectors and cuts were rebuilt once using {source['source_preset']} (seed 0), then frozen. Source sentence cap: {cfg.certification.train_sentence_cap}; selector-training focal cap: {cfg.certification.selector_train_focals_per_fold}; certificate sentence cap: {cfg.certification.test_documents}. None means uncapped. The matched classifier population is T1 union T2.",'','| Partition | Sentences | Words |','|---|---:|---:|']
    for p in partitions:lines.append(f"| {p['partition']} | {p['sentences']} | {p['words']} |")
    lines+=['','Each augmented arm contains all originals and exactly one cached, focal-labelled variant per training word. Selector masks use the opposite fold and scorer-predicted raw group. Random masks drop the same number of context words at uniformly random positions. Every dropped piece is drawn once from the single all-dropped-masked MLM pass; no draw is selected or rejected by its label score. Focal words and piece counts remain intact.','','The three MLPs share initialization within each seed and use the same ordered original-sentence stream. The matched baseline’s 64-sentence batches define a common labelled-token budget per optimizer step. Baseline spends it on originals; augmented arms spend half on originals and half on variants (one-token rounding when odd). Thus all arms have equal steps and labelled-token counts; augmented originals progress through the shared stream at half speed. Frozen whole-sentence features allow an original sentence’s word rows to span adjacent optimizer batches. Ten baseline-budget epochs equal five passes over each original-plus-variant pool. AdamW lr .001, weight decay .01, dropout .1 and per-head clip 1 match the source baseline. Each arm selects its checkpoint on raw CoNLL dev macro-F1: the unweighted mean of token-level F1 across the nine BIO tags, including O (zero-denominator tags contribute zero). This is not span F1. The existing matched-baseline code uses this same token macro-F1 criterion, as do all three current arms; checkpoint selection is unchanged.','','## Raw test results','','Mean ± sample SD over three seeds; percentages. Span counting and native/shared label mapping call the existing certification evaluation code. CoNLL’s shared result is retained alongside its native result.','','| Dataset | Label space | Arm | Span F1 | Precision | Recall |','|---|---|---|---:|---:|---:|']
    for r in summaries:lines.append(f"| {r['target']} | {r['label_space']} | {r['arm']} | {display(r,'span_f1')} | {display(r,'precision')} | {display(r,'recall')} |")
    lines+=['','## Seed-paired differences','','Percentage points, mean ± sample SD of paired differences. Per-arm/per-seed values and each seed’s differences are in the accompanying CSVs.','','| Dataset | Label space | Selector minus | Span F1 | Precision | Recall |','|---|---|---|---:|---:|---:|']
    for r in differences:lines.append(f"| {r['target']} | {r['label_space']} | {r['comparison']} | {display(r,'span_f1')} | {display(r,'precision')} | {display(r,'recall')} |")
    lines+=['','## Variant diagnostics','','Scorer disagreement is a biased proxy for label flips, measured after generation without changing the draw. Unchanged/zero-drop draws still count as variants.','','| Arm | Raw group | Variants | Variant token share | Scorer disagreement | Input changed |','|---|---|---:|---:|---:|---:|']
    for r in diagnostics:lines.append(f"| {r['arm']} | {r.get('group') or 'all'} | {r['variants']} | {100*float(r['variant_token_share']):.1f}% | {100*float(r['scorer_disagreement']):.2f}% | {100*float(r['changed_input_share']):.2f}% |")
    lines+=['','## Frozen scorer validation','',f"On its own scorer-validation split ({scorer['sentences']} sentences, {scorer['words']} words), the frozen scorer has **{100*scorer['span_f1']:.2f}% native span F1** and **{scorer['cross_entropy']:.4f} nats per word** of raw, unsmoothed cross-entropy, with full-sentence inputs. The mean sum of word NLLs per sentence is {scorer['mean_sentence_summed_ce']:.4f} nats. This is the scorer’s A-validation partition, not the native CoNLL dev set used to select the three downstream classifiers.",'','## Recalibrated kept fractions','','| Selector | Applied to | Raw group | rho | Calibration rows |','|---|---|---|---:|---:|']
    for r in cuts:lines.append(f"| {r['selector']} | {r['held_out']} | {r['group']} | {float(r['rho']):.2f} | {r['rows']} |")
    lines+=['','## Held-out bounds for selector fills','','These are the existing certificate calculation on the rebuilt pilot’s held-out test rows and calibrated cuts: one-sided 95% normal upper bounds on **smoothed fitted-scorer excess cross-entropy, in nats**. They use the same calibration-style independent fills. They are not guaranteed label-error probabilities. Interpreting them as information-loss bounds requires accurate reference posteriors and the relevant fill conditional; selection-conditioned and calibration-style fills are different. No probability conversion is asserted. All arms run regardless of these diagnostics.','','| Selector | Group | Kept fraction | n | Absolute CE mean | Absolute CE SD | Absolute CE UCB | CE increase mean | Increase SD | Increase UCB |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def number(value):return f"{float(value):.4f}" if value not in (None,'') else '—'
    for r in bounds:
        values=[r['absolute_ce_mean'],r['absolute_ce_sd'],r['absolute_ce_upper_bound'],r['mean_loss'],r['sd'],r['upper_bound']]
        lines.append(f"| {r['selector']} | {r['group']} | {float(r['rho']):.2f} | {r['rows']} | "+' | '.join(number(v) for v in values)+' |')
    lines+=['','Absolute CE is −log[(1−gamma) × mean_fill p_scorer(gold) + gamma/C], with gamma=.01. Its UCB is mean + 1.645 × sample SD / sqrt(n). It is reconstructed on the exact existing test rows and fills as the saved CE increase plus raw gamma-smoothed CE; no new fill-ins are sampled. The increase and absolute CE have separate standard deviations and bounds.']
    lines+=['','## Timing and artifacts','',f"Variant generation: **{generation['fold_generation_seconds']/60:.1f} minutes**. Three-seed fitting of all three arms: **{training['fit_seconds_all_seeds']/60:.1f} minutes**, including dev checkpoint evaluations. Training plus raw test evaluation: {training['training_evaluation_seconds_all_seeds']/60:.1f} minutes. End-to-end including rebuilt prerequisites: **{timing['wall_seconds']/3600:.2f} hours**.",'',f"Raw run: `{root}`. Frozen source run: `{source['run']}`. Cached focal features are reused for all seeds/epochs; schedules and predictions are saved per seed. [Implementation](IMPLEMENTATION_8.md), [tables](tables/augmentation/)."]
    (report/'RESULTS_8.md').write_text('\n'.join(lines)+'\n');return {'report':str(report/'RESULTS_8.md')}
