"""Compact results and the three requested figure families."""
import csv
import json
import shutil
import time
from collections import defaultdict
from ..diagnostics.common import ROOT,save_json
from ..invariance.data import paths


def figures(results,target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=list(csv.DictReader((results/'analyze'/'models.csv').open()));tasks=list(dict.fromkeys(r['dataset'] for r in rows))
    for metric in ('ood_accuracy','genre_dependence'):
        fig,axes=plt.subplots(2,len(tasks),figsize=(5*len(tasks),7),squeeze=False)
        for p,policy in enumerate(('id','genre')):
            for i,task in enumerate(tasks):
                ax=axes[p,i];subset=[r for r in rows if r['dataset']==task and r['policy']==policy]
                for rule in ('Rz','Rg','Rrand'):
                    values=defaultdict(list)
                    for r in subset:
                        if r['rule'] in (rule,'R0'):values[float(r['lambda'])].append(float(r[metric]))
                    xs=sorted(values);ax.plot(xs,[100*sum(values[x])/len(values[x]) for x in xs],marker='o',label=rule)
                positive=[float(r['lambda']) for r in subset if float(r['lambda'])>0]
                ax.set_xscale('symlog',linthresh=min(positive)/10)
                ax.set(title=f'{task}: {policy} validation',xlabel='λ',ylabel=metric+' (%)');ax.legend()
        fig.tight_layout();fig.savefig(target/(metric+'_lambda.png'),dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,len(tasks),figsize=(5*len(tasks),7),squeeze=False)
    for p,policy in enumerate(('id','genre')):
        for i,task in enumerate(tasks):
            ax=axes[p,i]
            for rule in ('R0','Rz','Rg','Rrand'):
                subset=[r for r in rows if r['dataset']==task and r['policy']==policy and r['rule']==rule]
                ax.scatter([float(r['zfree_violation']) for r in subset],[100*float(r['genre_dependence']) for r in subset],s=12,alpha=.5,label=rule)
            ax.set_xscale('symlog',linthresh=1e-8);ax.set(title=f'{task}: {policy} validation',xlabel='Z-free violation',ylabel='Within-label genre BA (%)');ax.legend()
    fig.tight_layout();fig.savefig(target/'violation_dependence.png',dpi=160);plt.close(fig)


def experiment(cfg,output):
    results,_=paths(cfg);report=ROOT/'report';tables=report/'tables'/'zfree';plots=report/'figures'/'zfree'
    tables.mkdir(parents=True,exist_ok=True);plots.mkdir(parents=True,exist_ok=True)
    for stage in ('analyze','contexts','calibration'):
        for file in (results/stage).glob('*.csv'):shutil.copy2(file,tables/file.name)
    shutil.copy2(results/'calibration'/'grids.json',tables/'grids.json')
    figures(results,plots)
    timing=json.loads((results/'timing.json').read_text());timing['ended_unix']=time.time();timing['wall_clock_seconds']=timing['ended_unix']-timing['started_unix'];timing['fits']=len(list((results/'train').glob('*/*/seed*/timing.json')))
    save_json(results/'timing.json',timing);shutil.copy2(results/'timing.json',tables/'timing.json')
    rows=list(csv.DictReader((tables/'summary.csv').open()));correlations=list(csv.DictReader((tables/'correlations.csv').open()))
    coverage=list(csv.DictReader((tables/'coverage.csv').open()))
    lines=['# Document-context nuisance penalty: results','',f"Wall-clock: **{timing['wall_clock_seconds']/3600:.2f} hours**, {timing['fits']} fits. Includes context preparation, calibration, training, probes and analysis; queue time is excluded.",'','The penalty and residualizer for Rz use no genre labels. Genre labels still define this study’s rotations and balanced batches, the Rg comparator, diagnostic probes, and validation policy (b). Policy (a) uses native in-domain validation. Residualization removes a fitted linear label contribution and does not prove the remaining context is exclusively nuisance.','','Differences below are percentage points: mean ± seed SD [95% paired document-block bootstrap interval]. Rotations are weighted equally. Probe intervals condition on fitted probes and classifier/λ choices; probes are not refitted during bootstrap.','','| Task | Validation | Primary comparison | Difference | Rotations |','|---|---|---|---:|---:|']
    for r in rows:
        if r['kind']!='primary':continue
        value=f"{100*float(r['mean']):+.2f} ± {100*float(r['seed_sd']):.2f} [{100*float(r['ci_low']):+.2f}, {100*float(r['ci_high']):+.2f}]"
        label=r['comparison']+' '+r['metric']
        lines.append(f"| {r['dataset']} | {r['policy']} | {label} | {value} | {r['rotations']} |")
    lines+=['','Primary 1 favours Rz when genre-dependence differences are negative. Primary 2 favours Rz when OOD accuracy differences are positive. Zero-crossing intervals remain inconclusive.','','| Task | Validation | Primary 3: Spearman ρ | Seed SD | 95% CI | Models |','|---|---|---:|---:|---|---:|']
    for r in correlations:lines.append(f"| {r['dataset']} | {r['policy']} | {float(r['spearman']):+.3f} | {float(r['seed_sd']):.3f} | [{float(r['ci_low']):+.3f}, {float(r['ci_high']):+.3f}] | {r['models']} |")
    lines+=['','Rank correlations include every rule/λ candidate and one shared λ=0 baseline per rotation/seed, pooled across tasks and separately per task, for each checkpoint policy. Genre chance levels and label support differ across tasks, so the pooled correlation needs the per-task checks. Positive correlation means lower Z-free violation tracks lower genre dependence; it does not establish a causal relationship. The two checkpoint policies share optimization trajectories and are not independent replications.','','| Task | Training coverage | ID-test coverage |','|---|---:|---:|']
    for task in dict.fromkeys(r['dataset'] for r in coverage):
        values={}
        for split in ('train','id_test'):
            cells=[r for r in coverage if r['dataset']==task and r['split']==split];values[split]=sum(int(r['covered']) for r in cells)/sum(int(r['tokens']) for r in cells)
        lines.append(f"| {task} | {100*values['train']:.1f}% | {100*values['id_test']:.1f}% |")
    lines+=['','Deviations/operational choices: GUM has only two native ID-test documents per genre, so probes use two document folds (EWT uses five), with separate native ID-validation documents for probe checkpoint selection. Balanced accuracy averages eligible genres within labels, then labels equally; support thresholds are the E3 defaults. ID bootstrap resamples documents within genres. PCA/ridge use only labelled native-training tokens from training genres; context means use all cached words in other sentences within the same native partition. Ridge strength is .001 with an unpenalized intercept. A fixed donor-document derangement with random donor-token matching defines Rrand. Pilot calibration uses a 100-step R0 trajectory and 12 batches; its shared four-value logarithmic grid brackets the rules’ median CE/penalty ratios with its middle values. See calibration.csv for actual penalty/CE ratios. No training-token or MMD subsampling is used.','','Descriptive OOD macro-F1, in-domain accuracy, violation, λ selections, probe support and all candidate measurements are in [tables](tables/zfree/). [Figures](figures/zfree/); [implementation](IMPLEMENTATION_4.md).']
    (report/'RESULTS_4.md').write_text('\n'.join(lines)+'\n')
    return {'report':str(report/'RESULTS_4.md'),'wall_clock_seconds':timing['wall_clock_seconds']}
