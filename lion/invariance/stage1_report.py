"""Compact stage-1 report: two primary comparisons and pilot/neighbor checks."""
import csv
import json
import shutil
import time
from pathlib import Path
from ..diagnostics.common import ROOT,save_json
from .data import paths


def experiment(cfg,output):
    results,_=paths(cfg);report=ROOT/'report';tables=report/'tables'/'invariance';tables.mkdir(parents=True,exist_ok=True)
    for source in (results/'analyze').glob('*.csv'):shutil.copy2(source,tables/source.name)
    selection=json.loads((results/'pilot'/'selection.json').read_text())
    shutil.copy2(results/'pilot'/'selection.json',tables/'pilot_selection.json')
    figures(cfg,results,report)
    timing=json.loads((results/'timing.json').read_text());timing.update(ended_unix=time.time())
    timing['wall_clock_seconds']=timing['ended_unix']-timing['started_unix'];save_json(results/'timing.json',timing)
    fits=[json.loads(p.read_text()) for stage in ('pilot','train') for p in (results/stage).glob('*/*/seed*/timing.json')]
    timing['fit_seconds']=sum(f['seconds'] for f in fits);timing['fits']=len(fits);save_json(results/'timing.json',timing);shutil.copy2(results/'timing.json',tables/'timing.json')
    rows=list(csv.DictReader((tables/'summary.csv').open()))
    lines=['# Stage 1: fixed invariance modes','',f"Stage-1 wall-clock: **{timing['wall_clock_seconds']/3600:.2f} hours** ({timing['fits']} fits; includes pilots, training and analysis). Kernel: **{cfg.invariance.kernel}**, cap {cfg.invariance.token_cap} tokens per genre/cell. Superseded full-grid time is excluded.",'','Only R0/R1/R2 were trained. Both entropy thresholds below are post-hoc cuts of the same predictions. Main λ is selected once per task/mode on the pilot rotation’s mean validation macro-F1 across three seeds; all other rotations keep it fixed. The neighbouring λ is a sensitivity check, never a second test-based selection.','','| Task | Pilot test / validation | R1 λ / neighbour | R2 λ / neighbour |','|---|---|---:|---:|']
    for task,s in selection.items():lines.append(f"| {task} | {s['rotation']} / {s['validation_genre']} | {s['R1']['lambda']:g} / {s['R1']['neighbour']:g} | {s['R2']['lambda']:g} / {s['R2']['neighbour']:g} |")
    lines+=['','Values are accuracy percentage-point differences: mean ± seed SD [95% document-block bootstrap interval]. Rotations receive equal weight; intervals condition on the three fitted seeds and pilot choices. Unsupported rotations are excluded per contrast; support is retained in the CSVs.','','| Task | Policy | H | Comparison | Shift tertile | Difference | Rotations |','|---|---|---:|---|---|---:|---:|']
    def display(r):return f"{100*float(r['mean']):+.2f} ± {100*float(r['seed_sd']):.2f} [{100*float(r['ci_low']):+.2f}, {100*float(r['ci_high']):+.2f}]"
    for r in rows:
        primary1=r['kind']=='mode_delta' and r['group']=='-1'
        primary2=r['kind']=='interaction' and r['group']=='3'
        if not (primary1 or primary2):continue
        label='1: R1−R2 overall' if primary1 else '2: unambiguous − ambiguous/rare'
        lines.append(f"| {r['dataset']} | {r['policy']} | {r['threshold']} | {label} | {r['tertile']} | {display(r)} | {r['rotations']} |")
    lines+=['','Tertile −1 means all tokens; 0/1/2 are low/medium/high training-defined label shift. Ties in training label scores can leave strata empty. Positive interactions within supported strata are consistent with H_annot; attenuation after stratification is compatible with H_shift. A zero-crossing interval does not establish equivalence. Comparison 1 alone distinguishes neither causal explanation.']
    for task in selection:
        relevant=[r for r in rows if r['dataset']==task and r['policy']=='main' and r['kind']=='interaction' and r['group']=='3' and r['tertile']!='-1' and int(r['rotations'])>0]
        positive=sum(float(r['ci_low'])>0 for r in relevant);negative=sum(float(r['ci_high'])<0 for r in relevant)
        assessment='H_annot-consistent positive interactions appear in some strata, with mixed evidence across strata.' if positive else 'Neither explanation is established by the within-tertile intervals.'
        if negative:assessment+=' Some intervals favour the reverse interaction.'
        lines.append(f'\n**{task}, comparison 2:** {assessment} Positive / negative intervals: {positive} / {negative} of {len(relevant)} supported threshold×tertile cells; thresholds reuse predictions.')
    lines+=['','Deviations from the initial request are the user’s cost revision: Stage 1 only; no GUM discourse or routing; pilot-fixed λ plus one neighbour; 600-step maximum with early stopping; capped, batch-resampled MMD; entropy thresholds applied only in evaluation. The RBF bandwidth uses the capped, genre-stratified batch sample. No linear-kernel fallback was used.' if cfg.invariance.kernel=='rbf' else 'The user-authorized linear-kernel MMD fallback was used; penalty equals squared distances of genre mean probability vectors. Other cost revisions are documented in IMPLEMENTATION_3.md.','','See [implementation](IMPLEMENTATION_3.md), [tables](tables/invariance/), and [figures](figures/invariance/). Stage 2 remains deferred.']
    (report/'RESULTS_3.md').write_text('\n'.join(lines)+'\n')
    return {'report':str(report/'RESULTS_3.md'),'wall_clock_seconds':timing['wall_clock_seconds']}


def figures(cfg,results,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from collections import defaultdict
    target=report/'figures'/'invariance';target.mkdir(parents=True,exist_ok=True)
    curves=list(csv.DictReader((results/'analyze'/'lambda_curves.csv').open()))
    rows=list(csv.DictReader((results/'analyze'/'summary.csv').open()))
    tasks=list(dict.fromkeys(r['dataset'] for r in curves))
    fig,axes=plt.subplots(1,len(tasks),figsize=(5*len(tasks),4),squeeze=False)
    for task,ax in zip(tasks,axes[0]):
        for rule in ('R1','R2'):
            values=defaultdict(list)
            for r in curves:
                if r['dataset']==task and r['rule']==rule and r['pilot']=='True':values[float(r['lambda'])].append(float(r['ood_accuracy']))
            xs=sorted(values);ax.plot(xs,[100*sum(values[x])/len(values[x]) for x in xs],marker='o',label=rule)
        ax.set_xscale('symlog',linthresh=.1);ax.set(title=task,xlabel='λ (pilot rotation only)',ylabel='OOD accuracy (%)');ax.legend()
    fig.tight_layout();fig.savefig(target/'pilot_lambda_curves.png',dpi=160);plt.close(fig)
    for threshold in cfg.invariance.thresholds:
        fig,axes=plt.subplots(len(tasks),4,figsize=(15,3.3*len(tasks)),squeeze=False)
        for i,task in enumerate(tasks):
            for j,tertile in enumerate((-1,0,1,2)):
                ax=axes[i,j]
                selected=[r for r in rows if r['dataset']==task and r['policy']=='main' and float(r['threshold'])==threshold and r['kind']=='mode_delta' and int(r['tertile'])==tertile and r['group'] in ('0','1','2')]
                for r in selected:
                    x=int(r['group']);mean=100*float(r['mean']);lo=100*float(r['ci_low']);hi=100*float(r['ci_high'])
                    if int(r['rotations']):ax.vlines(x,lo,hi);ax.plot(x,mean,'o')
                ax.axhline(0,color='gray',lw=.7);ax.set_xticks(range(3),['Unamb.','Ambig.','Rare']);ax.set(title=f'{task}: shift {tertile}',ylabel='R1−R2 accuracy (pp)')
        fig.tight_layout();fig.savefig(target/f'group_shift_H{threshold:g}.png',dpi=160);plt.close(fig)
