"""Report the three primary comparisons and requested descriptive figures."""
from collections import defaultdict
import math
import shutil

import torch

from ..diagnostics.common import ROOT,read_csv
from .data import paths


def fmt(value):return f'{100*float(value):.2f}' if math.isfinite(float(value)) else 'undefined'


def estimate(row):return f"{fmt(row['mean'])} ± {fmt(row['seed_sd'])} [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]"


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'---|'*len(headers),*['| '+' | '.join(map(str,row))+' |' for row in rows]])+'\n'


def plot(cfg,curves,summary,scores,destination):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    for dataset in cfg.invariance.datasets:
        fig,axes=plt.subplots(1,len(cfg.invariance.thresholds),figsize=(11,4),squeeze=False)
        for ax,threshold in zip(axes[0],cfg.invariance.thresholds):
            baseline=[float(r['ood_accuracy']) for r in curves if r['dataset']==dataset and float(r['threshold'])==threshold and float(r['lambda'])==0]
            ax.axhline(100*float(torch.tensor(baseline).mean()),label='R0',color='black',ls=':')
            for rule in ('R1','R2','R3','R4','R5'):
                rows=[r for r in curves if r['dataset']==dataset and float(r['threshold'])==threshold and r['rule']==rule]
                x=sorted({float(r['lambda']) for r in rows});y=[];sd=[]
                for strength in x:
                    values=[]
                    for seed in cfg.diagnostics.seeds:
                        chosen=[float(r['ood_accuracy']) for r in rows if float(r['lambda'])==strength and int(r['seed'])==seed]
                        values.append(torch.tensor(chosen).mean())
                    values=torch.stack(values);y.append(100*float(values.mean()));sd.append(100*float(values.std()))
                ax.errorbar(range(len(x)),y,yerr=sd,label=rule,marker='o',capsize=2)
                ax.set_xticks(range(len(x)),[str(v) for v in x],rotation=45)
            ax.set(title=f'{dataset}, H={threshold}',xlabel='lambda',ylabel='OOD accuracy (%)');ax.legend()
        fig.tight_layout();fig.savefig(destination/f'{dataset}_lambda.png',dpi=150);plt.close(fig)
        for threshold in cfg.invariance.thresholds:
            fig,axes=plt.subplots(1,4,figsize=(14,4),squeeze=False)
            for ax,stratum in zip(axes[0],(-1,0,1,2)):
                chosen=[r for r in summary if r['dataset']==dataset and float(r['threshold'])==threshold and r['kind']=='mode_delta' and int(r['group_or_tertile'])==stratum]
                for i,r in enumerate(chosen):
                    mean=float(r['mean'])*100;low=float(r['ci_low'])*100;high=float(r['ci_high'])*100
                    ax.errorbar(i,mean,yerr=[[max(0,mean-low)],[max(0,high-mean)]],fmt='o')
                ax.set_xticks(range(len(chosen)),[r['comparison'] for r in chosen],rotation=55)
                ax.axhline(0,color='grey',lw=.7);ax.set(title='all shifts' if stratum==-1 else f'shift tertile {stratum+1}',ylabel='R1 minus R2 accuracy (pp)')
            fig.tight_layout();fig.savefig(destination/f'{dataset}_interaction_H{threshold}.png',dpi=150);plt.close(fig)
            selected=[r for r in scores if r['dataset']==dataset and float(r['threshold'])==threshold and int(r['group'])==-1]
            def model_mean(model):
                return 100*float(torch.tensor([float(r['accuracy']) for r in selected if r['model']==model]).mean())
            draws=[model_mean(f'R5_{d}') for d in range(cfg.invariance.random_draws)]
            fig,ax=plt.subplots(figsize=(5,4));ax.scatter(range(len(draws)),draws,label='R5 draws')
            ax.axhline(model_mean('R3'),label='R3',color='red');ax.set(xlabel='Random type assignment draw',ylabel='OOD accuracy (%)',title=f'{dataset}, H={threshold}');ax.legend()
            fig.tight_layout();fig.savefig(destination/f'{dataset}_R3_R5_H{threshold}.png',dpi=150);plt.close(fig)
            fig,ax=plt.subplots(figsize=(7,4));names=['R1','R2','routing','ensemble']+[f'random_routing_{d}' for d in range(cfg.invariance.random_draws)]
            ax.bar(range(len(names)),[model_mean(n) for n in names]);ax.set_xticks(range(len(names)),names,rotation=55)
            ax.set(ylabel='OOD accuracy (%)',title=f'{dataset}, H={threshold}')
            fig.tight_layout();fig.savefig(destination/f'{dataset}_routing_H{threshold}.png',dpi=150);plt.close(fig)


def experiment(cfg,output):
    results,_=paths(cfg);source=results/'analyze'
    summary=read_csv(source/'summary.csv');curves=read_csv(source/'lambda_curves.csv');scores=read_csv(source/'per_rotation_scores.csv')
    tables=ROOT/'report/tables/invariance';figures=ROOT/'report/figures/invariance'
    tables.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
    for path in source.glob('*.csv'):shutil.copy2(path,tables/path.name)
    plot(cfg,curves,summary,scores,figures)
    lines=['# Fixed invariance signatures: results','',
           'Setup and all operational choices: [IMPLEMENTATION_3.md](IMPLEMENTATION_3.md). Entries are accuracy differences in percentage points: mean ± seed SD [95% document-block bootstrap interval]. Rotations are weighted equally; seeds are 0/1/2. Lambda and checkpoints are selected only on the held-out validation genre.','',
           '## Primary 1: fixed conditional versus marginal mode','',
           table(['Dataset','R1 minus R2 OOD accuracy','Interpretation'],[[r['dataset'],estimate(r),'conditional better; neither hypothesis identified' if float(r['ci_low'])>0 else 'marginal better; neither hypothesis identified' if float(r['ci_high'])<0 else 'neither / inconclusive'] for r in summary if r['kind']=='primary' and r['comparison']=='R1_minus_R2' and float(r['threshold'])==cfg.invariance.thresholds[0]]),
           'An overall fixed-mode advantage alone cannot distinguish H_annot from H_shift.','',
           '## Primary 2: interaction after controlling label shift','',
           'The primary interaction is (R1−R2 on unambiguous tokens) minus (R1−R2 on ambiguous and rare/unseen tokens combined). Positive differences are the direction predicted by H_annot.','',
           table(['Dataset','Entropy threshold','Shift stratum','Interaction','Supported rotations','Assessment'],[[r['dataset'],r['threshold'], 'all' if int(r['group_or_tertile'])==-1 else str(int(r['group_or_tertile'])+1),estimate(r),r['rotations'],
             'H_annot-consistent within stratum' if float(r['ci_low'])>0 and int(r['group_or_tertile'])>=0 else 'H_annot direction overall' if float(r['ci_low'])>0 else 'opposite interaction' if float(r['ci_high'])<0 else 'neither established'] for r in summary if r['kind']=='interaction' and r['comparison']=='combined']),
           'A positive interaction that weakens after stratification is compatible with H_shift, but intervals containing zero do not establish that it vanishes. No equivalence test is claimed. Component interactions against ambiguous and rare/unseen groups separately, group-specific mode differences and supports are in the CSVs.','',
           '## Primary 3: structure of the type split','',
           table(['Dataset','Entropy threshold','Comparison','OOD accuracy difference','Assessment'],[[r['dataset'],r['threshold'],r['comparison'],estimate(r),
             'supports useful assignment structure' if float(r['ci_low'])>0 else 'opposes R3 advantage' if float(r['ci_high'])<0 else 'neither / inconclusive'] for r in summary if r['kind']=='primary' and r['comparison']!='R1_minus_R2']),
           'R3 must beat all three comparators, not just one. The better-fixed comparator is the per-rotation, per-seed oracle maximum of R1/R2 test accuracy; it is used only for analysis, never prediction or tuning. The R5 comparator averages draw-level metrics rather than ensembling their probabilities. Evidence for H_annot additionally requires the within-shift interaction; a structured split can help for other reasons.','',
           '## R3 rank and routing (descriptive)','']
    rank=[]
    for dataset in cfg.invariance.datasets:
        for threshold in cfg.invariance.thresholds:
            chosen=[r for r in scores if r['dataset']==dataset and float(r['threshold'])==threshold and int(r['group'])==-1]
            mean=lambda model:float(torch.tensor([float(r['accuracy']) for r in chosen if r['model']==model]).mean())
            r3=mean('R3');draws=[mean(f'R5_{d}') for d in range(cfg.invariance.random_draws)]
            rank.append([dataset,threshold,1+sum(v>r3 for v in draws),sum(v==r3 for v in draws)])
    lines += [table(['Dataset','Entropy threshold','R3 rank among R3 + five R5 draws','Tied R5 draws'],rank),
              'Ranks compare equal-rotation, three-seed mean accuracies; rank 1 with ties is not a strict win. Per-rotation ranks are also retained. Routing uses only validation group accuracy; ties prefer R1, and empty validation groups fall back to global validation accuracy. The probability ensemble and five exact-share random-type routes are controls.','',
              table(['Dataset','Threshold','Model','OOD accuracy (%)'],[[r['dataset'],r['threshold'],r['comparison'],estimate(r)] for r in summary if r['kind']=='score' and int(r['group_or_tertile'])==-1 and r['metric']=='accuracy' and r['comparison'] in {'routing','ensemble','random_routing_mean'}]),
              '## Files and interpretation limits','',
              'Full tables, including selected lambdas, complete lambda curves, per-group accuracy/F1, in-domain costs, label-shift scores and all random controls: [tables/invariance](tables/invariance/). Requested figures: [figures/invariance](figures/invariance/).','',
              'All words are used after baseline tokenization/truncation; this is not the old two-focal sample. The five GUM test genres were fixed in advance, not selected from results. Word groups and label-shift cutoffs use training genres only. Gold test labels are used only for scoring and label-shift stratification.','',
              'Conditional cells below eight tokens/genre are skipped exactly as configured; training histories record supported mass and pair counts. Therefore a mode advantage can reflect penalty coverage as well as the hypothesized causal direction. The R5 matching algorithm is a constrained random type allocation with singleton completion, not a uniform draw over all exactly matched subsets.','',
              'Label shift is equal-genre generalized JS of Bernoulli(label versus other) frequencies. Tertiles use the training-token-weighted score distribution, with ties preserved; unsupported interaction strata are undefined. A label absent from training has zero observed training shift, which cannot diagnose its unseen-domain shift.','',
              'The bootstrap resamples held-out test documents, keeping fitted models, validation choices and type assignments fixed. It does not include retraining or validation-selection uncertainty. This is a predictive diagnostic; neither MMD signature is established as causally correct by these comparisons alone.','']
    (ROOT/'report/RESULTS_3.md').write_text('\n'.join(lines))
    return {'summary_rows':len(summary)}
