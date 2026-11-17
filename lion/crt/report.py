"""Requested raw-array summaries and factual CRT results, without extra arms."""
import csv,json,math,shutil,time
import torch
from ..diagnostics.common import ROOT,save_json,write_csv
from .data import paths,TASKS


def experiment(cfg,output):
    results,_=paths(cfg);tables=ROOT/'report'/'tables'/'crt';tables.mkdir(parents=True,exist_ok=True)
    records=[];terciles=[];histograms=[]
    for file in sorted((results/'test'/'all').glob('*.pt')):
        data=torch.load(file,weights_only=False);n=len(data['row']);offsets=data['offsets']
        for task_index,task in enumerate(TASKS):
            if not n:continue
            original=data['T'][:,task_index].double();random=data['resampled'][:,:,task_index].double();masked=data['masked'][:,task_index].double()
            delta=original-random.mean(1);p=(1+(random>=original[:,None]).sum(1)).double()/(cfg.crt.draws+1)
            # Sum paired differences: algebraically S^(k)-S, with exact zeros for identical draws.
            aggregate=(random-original[:,None]).sum(0);sd=aggregate.std(unbiased=True)
            z=float(-aggregate.mean()/sd) if sd>0 else (float('nan') if aggregate.mean()==0 else math.copysign(float('inf'),float(-aggregate.mean())))
            p_aggregate=float((1+(aggregate>=0).sum()).double()/(cfg.crt.draws+1))
            rate05=float((p<=.05).double().mean());rate01=float((p<=.01).double().mean())
            record={'condition':file.stem,'offsets':','.join(map(str,offsets)),'pair':len(offsets)==2,'task':task,'rows':n,'delta':float(delta.mean()),'z':z,'p':p_aggregate,
                    'mask_effect':float((original-masked).mean()),'reject_05':rate05,'se_05':math.sqrt(rate05*(1-rate05)/n),'reject_01':rate01,'se_01':math.sqrt(rate01*(1-rate01)/n),
                    'tie_fraction':float((random==original[:,None]).double().mean()),'entropy_mean':float(data['entropy'].mean()),
                    'synthetic_null':task=='synthetic' and not any(d in (-2,1) for d in offsets)}
            records.append(record)
            cuts=torch.quantile(data['entropy'],torch.tensor([1/3,2/3]));group=torch.bucketize(data['entropy'],cuts)
            for j in range(3):
                mask=group==j;terciles.append({'condition':file.stem,'task':task,'tercile':j,'rows':int(mask.sum()),'delta':float(delta[mask].mean()),'cut_low':float(cuts[0]),'cut_high':float(cuts[1])})
            if record['synthetic_null'] and len(offsets)==1:
                count=torch.histc(p.float(),bins=10,min=0,max=1).long()
                for j,value in enumerate(count):histograms.append({'condition':file.stem,'bin_low':j/10,'bin_high':(j+1)/10,'count':int(value),'rows':n})
            torch.save({'row':data['row'],'p':p.float(),'delta':delta.float()},output/(file.stem+'_'+task+'.pt'))
    for name,rows in [('summary',records),('entropy_terciles',terciles),('null_histograms',histograms)]:write_csv(output/(name+'.csv'),rows);shutil.copy2(output/(name+'.csv'),tables/(name+'.csv'))
    for file in [results/'prepare'/'samples.csv',results/'train'/'dev.json',results/'budget.json',results/'benchmark.json']:
        shutil.copy2(file,tables/file.name)
    histogram_figure(histograms,ROOT/'report'/'figures'/'crt')
    timing=json.loads((results/'timing.json').read_text());timing['ended_unix']=time.time();timing['wall_clock_seconds']=timing['ended_unix']-timing['started_unix'];save_json(results/'timing.json',timing);shutil.copy2(results/'timing.json',tables/'timing.json')
    dev=json.loads((results/'train'/'dev.json').read_text());budget=json.loads((results/'budget.json').read_text())
    lines=['# Experiment 6: conditional randomization tests','',f"Wall-clock: **{timing['wall_clock_seconds']/3600:.2f} hours**. **{budget['selected_rows']}/{budget['available_rows']} sentence rows**, K={cfg.crt.draws}; {budget['halvings']} row-budget halvings. The measured 1% benchmark and projection are in benchmark.json and budget.json.",'','One fp16 bert-base-cased backbone and its own pretrained MLM head supply both representations and samples. Three independent seed-0 MLPs use every eligible training word’s ±16-subword window, with CLS/SEP. Training uses a fixed ten epochs; dev accuracy covers all eligible dev words and does not select the scorer, since dev rows enter the CRT.','','| Task | Dev accuracy | Dev macro-F1 | Epoch |','|---|---:|---:|---:|']
    for task in TASKS:lines.append(f"| {task} | {100*dev[task]['accuracy']:.2f}% | {100*dev[task]['macro_f1']:.2f}% | {dev[task]['selected_epoch']} |")
    lines+=['','## Arm A: synthetic labels','','−2 and +1 are non-null; the other single offsets are null by the specified label rule. Rejection fractions have binomial SEs treating sentence rows as independent. Ties use exact equality and count against rejection.','','| d | Rows | Null | p≤.05 ± SE | p≤.01 ± SE | Ties | Aggregate p | z | Δ |','|---|---:|---|---:|---:|---:|---:|---:|---:|']
    singles=[r for r in records if not r['pair']]
    for r in sorted([r for r in singles if r['task']=='synthetic'],key=lambda r:int(r['offsets'])):
        lines.append(f"| {r['offsets']} | {r['rows']} | {r['synthetic_null']} | {r['reject_05']:.3f} ± {r['se_05']:.3f} | {r['reject_01']:.3f} ± {r['se_01']:.3f} | {r['tie_fraction']:.3f} | {r['p']:.5f} | {r['z']:.2f} | {r['delta']:+.4f} |")
    lines+=['','[Ten-bin per-row p histograms for each null offset](figures/crt/synthetic_null_histograms.png). Histogram counts are also in null_histograms.csv.']
    for task in ('upos','deprel'):
        lines+=['',f'## Arm B: {task}','','| d | Rows | Δ | z | Aggregate p | Masking effect | Per-row p≤.05 |','|---|---:|---:|---:|---:|---:|---:|']
        for r in sorted([r for r in singles if r['task']==task],key=lambda r:int(r['offsets'])):lines.append(f"| {r['offsets']} | {r['rows']} | {r['delta']:+.4f} | {r['z']:.2f} | {r['p']:.5f} | {r['mask_effect']:+.4f} | {r['reject_05']:.3f} |")
    lines+=['','## Arm C: adjacent pairs','','| Task | Pair | Rows | Δ | z | Pair p | Single p (first / second) | Masking effect | Per-row p≤.05 |','|---|---|---:|---:|---:|---:|---|---:|---:|']
    for r in records:
        if not r['pair']:continue
        single_p=[next(v['p'] for v in singles if v['task']==r['task'] and v['offsets']==str(d)) for d in map(int,r['offsets'].split(','))]
        lines.append(f"| {r['task']} | {r['offsets']} | {r['rows']} | {r['delta']:+.4f} | {r['z']:.2f} | {r['p']:.5f} | {single_p[0]:.5f} / {single_p[1]:.5f} | {r['mask_effect']:+.4f} | {r['reject_05']:.3f} |")
    null_pair=next(r for r in records if r['task']=='synthetic' and r['offsets']=='2,3')
    lines.append(f"\nSynthetic null pair (+2,+3): per-row rejection {null_pair['reject_05']:.3f} ± {null_pair['se_05']:.3f} at .05 and {null_pair['reject_01']:.3f} ± {null_pair['se_01']:.3f} at .01; tie fraction {null_pair['tie_fraction']:.3f}.")
    lines+=['','## Effects by sampler entropy','','Entropy is the single-position conditional entropy; for pairs it is first-position entropy plus the draw-average second-position conditional entropy. Tercile cut ties remain together; empty groups are shown as nan.','','| Task | Offset/pair | Low Δ (n) | Middle Δ (n) | High Δ (n) |','|---|---|---:|---:|---:|']
    for r in records:
        values=[next(v for v in terciles if v['task']==r['task'] and v['condition']==r['condition'] and v['tercile']==j) for j in range(3)]
        lines.append(f"| {r['task']} | {r['offsets']} | "+' | '.join(f"{v['delta']:+.4f} ({v['rows']})" for v in values)+' |')
    null=[r for r in singles if r['task']=='synthetic' and r['synthetic_null']]
    lines+=['','## Factual summary','',f"Across the eight synthetic-null single offsets, per-row .05 rejection fractions range from {min(r['reject_05'] for r in null):.3f} to {max(r['reject_05'] for r in null):.3f}; aggregate p-values range from {min(r['p'] for r in null):.5f} to {max(r['p'] for r in null):.5f}."]
    for task in ('upos','deprel'):
        values=[r for r in singles if r['task']==task];largest=max(values,key=lambda r:r['delta'])
        lines.append(f"\n{task}: the largest measured single-offset mean CRT effect is {largest['delta']:+.4f} at d={largest['offsets']}; {sum(r['p']<=.05 for r in values)}/10 aggregate tests have p≤.05.")
    lines+=['','## Implementation and raw data','','[Implementation](IMPLEMENTATION_6.md). Raw arrays are under `results/crt/test/all/*.pt`: original/resampled/masked statistics for all three tasks, token draws, conditional entropy, sentence-row IDs and original replacement tokens. `results/crt/test/baseline.pt` retains the original rows and their sentence/document metadata. Report-stage files retain per-row p-values and effects.','','The three arms share identical encoder work and randomizations. Identical sampled windows are scored once and expanded back to all K draws; unchanged replacements copy the original score exactly to preserve ties. No top-k, top-p or temperature change is used. Joint sampling follows the requested two-mask/first-fill chain and batches each distinct first value once per row. No additional arms, datasets, encoders, sweeps or seed repetitions were added. Binomial SEs use sentence rows; source sentences may share documents. Finite-sample CRT validity requires the sampling conditional to match the data conditional; the reported calibration numbers do not assume this equality. No multiplicity correction was applied to these requested diagnostic p-values.']
    (ROOT/'report'/'RESULTS_6.md').write_text('\n'.join(lines)+'\n');return {'report':str(ROOT/'report'/'RESULTS_6.md'),'wall_clock_seconds':timing['wall_clock_seconds']}


def histogram_figure(rows,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder.mkdir(parents=True,exist_ok=True);names=sorted({r['condition'] for r in rows});fig,axes=plt.subplots(2,4,figsize=(14,6),squeeze=False)
    for ax,name in zip(axes.flatten(),names):
        values=[r for r in rows if r['condition']==name];ax.bar([r['bin_low']+.05 for r in values],[r['count']/r['rows'] for r in values],width=.095)
        ax.set(title=name,xlabel='Per-row p',ylabel='Fraction',xlim=(0,1));ax.axhline(.1,color='gray',linestyle='--',lw=.7)
    fig.tight_layout();fig.savefig(folder/'synthetic_null_histograms.png',dpi=160);plt.close(fig)
