"""CoNLL-only diagnostic tables; augmented arms and transfer stay in CSVs."""
import json,time,shutil
from pathlib import Path
import torch
from ..diagnostics.common import ROOT,read_csv,save_json
from ..sensitivity.data import paths


def experiment(cfg,output):
    root=paths(cfg);analysis=root/'analyse';tables=ROOT/'report'/'tables'/'sensitivity_1b';tables.mkdir(parents=True,exist_ok=True)
    for p in analysis.glob('*.csv'):shutil.copy2(p,tables/p.name)
    summary=read_csv(analysis/'summary.csv');fills=read_csv(analysis/'A_fills.csv');distance=read_csv(analysis/'B_distance.csv');matched=read_csv(analysis/'B_matched.csv');concentration=read_csv(analysis/'D_concentration.csv');words=read_csv(analysis/'D_words.csv');repro=read_csv(analysis/'reproduction.csv')
    report_rows=[r for r in summary if r['dataset']=='conll2003' and r['classifier'] in ('baseline','scorer')]
    def find(model,analysis,group,bin,mask,metric):return next(r for r in report_rows if (r['classifier'],r['analysis'],r['group'],r['bin'],r['mask'],r['metric'])==(model,analysis,group,bin,mask,metric))
    def scalar(x,digits=4):return f'{float(x):.{digits}f}' if x not in ('',None) else '—'
    def formatted(r,percent=False):
        factor=100 if percent else 1;mean=f"{factor*float(r['mean']):.4f}";sd=r.get('seed_sd');se=r.get('row_se')
        return mean+(f" ± {factor*float(sd):.4f}" if sd else '')+(f" (SE {factor*float(se):.4f})" if se else '')
    def pair(model,analysis,group,bin,metric):return ' / '.join(formatted(find(model,analysis,group,bin,k,metric),metric in ('flip','error')) for k in ('selector','random'))
    generated=sum(torch.load(p,mmap=True,weights_only=False)['seconds'] for p in (root/'generate').glob('*/g*.pt'))
    measured=json.loads((root/'measure'/'timing.json').read_text())['measurement_seconds'];analysed=json.loads((analysis/'timing.json').read_text())['analysis_seconds']
    timing=json.loads((root/'timing.json').read_text());timing.update(ended_unix=time.time(),generation_seconds=generated,measurement_seconds=measured,analysis_seconds=analysed);timing['wall_seconds']=timing['ended_unix']-timing['started_unix'];save_json(root/'timing.json',timing);save_json(tables/'timing.json',timing)
    lines=['# Step 1b: diagnostics of in-domain sensitivity','',f'Fills were **regenerated** with the original procedure, seeds, batch layout and four draws, because step 1 saved only focal vectors. Filled token IDs are now saved. Generation: **{generated/60:.1f} min**; measurement: **{measured/60:.1f} min**; analysis: **{analysed/60:.1f} min**; wall-clock: **{timing["wall_seconds"]/3600:.2f} h**. Nothing was trained or selected. All analyses below use this single regenerated set.','','Only CoNLL test, baseline (three seeds) and scorer appear in this report. Augmented arms and WNUT/WikiANN are in the CSVs; transfer rows are marked UNCERTIFIED. All tables put selector/random side by side. S is gamma=.01 KL over nine native tags, in nats. Flip/error/restoration/concentration percentages use a factor of 100. Baseline values are mean ± sample SD across seeds, followed by row SE. Scorer has no seed SD. Neither folds nor draws count as additional independent rows. Bins below 50 paired rows have an asterisk; matched ratios are not computed for those cells.','','## Reproduction of step 1','','| Mask | Rows | Step 1 saved S | Regenerated S | Difference |','|---|---:|---:|---:|---:|']
    for r in repro:lines.append(f"| {r['mask']} | {r['rows']} | {float(r['step1_S']):.8f} | {float(r['regenerated_S']):.8f} | {float(r['difference_from_step1']):+.3e} |")
    deviation=max(abs(float(r['difference_from_step1'])) for r in repro)
    lines.append(f'\nThe supplied rounded baseline-entity references were selector 0.1024 and random 0.1283. The largest absolute difference from the saved full-precision step-1 values is {deviation:.3e}; differences from the rounded references are also saved in reproduction.csv.')
    lines+=['','## A. Actual changes and non-null sensitivity','','| Model | Group | Full / paired non-null rows | Excluded sel / rand | Dropped words sel / rand | Changed words sel / rand | Restored % sel / rand | Null swaps % sel / rand | Non-null S sel / rand | Non-null flips % sel / rand |','|---|---|---:|---:|---:|---:|---:|---:|---|---|']
    for model in ('baseline','scorer'):
        for group in ('O','entity'):
            a=find(model,'A_nonnull',group,'non_null','selector','S');f=[next(r for r in fills if (r['dataset'],r['group'],r['mask'])==('conll2003',group,k)) for k in ('selector','random')]
            def fillpair(key,scale=1):return ' / '.join(f'{scale*float(r[key]):.3f}' for r in f)
            lines.append(f"| {model} | {group} | {f[0]['rows']} / {a['rows']} | {a['excluded_selector']} / {a['excluded_random']} | {fillpair('mean_dropped')} | {fillpair('mean_changed')} | {fillpair('restored_fraction',100)} | {fillpair('null_swap_fraction',100)} | {pair(model,'A_nonnull',group,'non_null','S')} | {pair(model,'A_nonnull',group,'non_null','flip')} |")
    lines+=['','Restored percentages count identical restored word occurrences among all dropped-word occurrences. Null swaps mean every dropped word was restored, including zero-drop draws. Non-null estimates average eligible draws within each fold, then available fold means within the word. The table uses words with non-null support for both masks; the two mask-specific all-null exclusions are shown before this additional pairing. No null draw contributes zero to the non-null average.']
    lines+=['','## B1. Changed-count and nearest-change bins','','| Model | Group | Bin | Paired rows | S sel / rand | Flips % sel / rand |','|---|---|---|---:|---|---|']
    for model in ('baseline','scorer'):
        for group in ('O','entity'):
            for axis,bins in [('changed_count',['0','1','2','3','4','5–9','10+']),('nearest_changed',['1','2','3','4–5','6–10','>10'])]:
                for b in bins:
                    r=find(model,'B_'+axis,group,b,'selector','S');mark='*' if r['small_bin']=='True' else ''
                    lines.append(f"| {model} | {group} | {axis}: {b}{mark} | {r['rows']} | {pair(model,'B_'+axis,group,b,'S')} | {pair(model,'B_'+axis,group,b,'flip')} |")
    lines+=['','Each estimate uses the same words for both masks: a word needs at least one draw in that bin for each mask (after fold averaging). Nearest-change bins omit null swaps. Paired row counts vary by bin and are recorded above; no draw is counted as an independent observation.']
    lines+=['','## B2. Distances of dropped words','','| Group | Dropped positions sel / rand | Mean distance sel / rand | Median sel / rand | Within 2 words % sel / rand |','|---|---:|---:|---:|---:|']
    for group in ('O','entity'):
        f=[next(r for r in distance if (r['dataset'],r['group'],r['mask'])==('conll2003',group,k)) for k in ('selector','random')]
        def dp(key,scale=1):return ' / '.join(f'{scale*float(r[key]):.3f}' for r in f)
        lines.append(f"| {group} | {dp('dropped_positions')} | {dp('mean')} | {dp('median')} | {dp('within_two',100)} |")
    lines+=['','These distributions pool dropped word positions from both folds; every dropped position receives equal weight. They are mask properties and do not depend on classifier seed.']
    lines+=['','## B3. Size- and distance-matched S ratios','','| Model | Group | Matched ratio | Seed SD | Row SE | Cells | Selector non-null mass retained % |','|---|---|---:|---:|---:|---:|---:|']
    for r in matched:
        if r['dataset']=='conll2003' and r['classifier'] in ('baseline','scorer') and r['group'] in ('O','entity'):
            lines.append(f"| {r['classifier']} | {r['group']} | {scalar(r['matched_ratio_mean'])} | {scalar(r['matched_ratio_seed_sd'])} | {scalar(r['ratio_row_se_fixed_weights'])} | {r['supported_cells']} | {100*float(r['selector_nonnull_mass_retained']):.2f} |")
    lines+=['','Ratios are computed within changed-count × nearest-distance cells on common words, then averaged using selector cell frequencies. Cells below 50 paired words or with zero random S are omitted; weights are renormalized over retained cells. Row SE uses a delta method with cell weights held fixed, accounting for words shared across cells. Retained mass and cell counts describe the support used.']
    lines+=['','## C. Gold entity spans seen in training','','| Model | Category | Rows | Original error % | S sel / rand | Flips % sel / rand |','|---|---|---:|---|---|---|']
    for model in ('baseline','scorer'):
        for category in ('unseen','seen_different_type','seen_same_type','no_gold_entity'):
            r=find(model,'C_entity_seen','entity',category,'selector','S');e=find(model,'C_entity_seen','entity',category,'original','error');mark='*' if r['small_bin']=='True' else ''
            lines.append(f"| {model} | {category}{mark} | {r['rows']} | {formatted(e,True)} | {pair(model,'C_entity_seen','entity',category,'S')} | {pair(model,'C_entity_seen','entity',category,'flip')} |")
    lines+=['','Groups remain the scorer-predicted groups from step 1. A scorer-predicted entity word whose gold tag is O has no gold entity span and is shown separately. Seen membership uses exact case-sensitive gold span word strings anywhere in T1 ∪ T2, excluding scorer split A. Original error compares the full native BIO tag, including B/I.']
    lines+=['','## D1. Training word-form ambiguity','','| Model | Group | Word-form bin | Rows | Share of group % | S sel / rand | Flips % sel / rand |','|---|---|---|---:|---:|---|---|']
    for model in ('baseline','scorer'):
        for group in ('O','entity'):
            for b in ('unseen_word','one_type','two_or_more_types'):
                r=find(model,'D_ambiguity',group,b,'selector','S');mark='*' if r['small_bin']=='True' else ''
                lines.append(f"| {model} | {group} | {b}{mark} | {r['rows']} | {100*float(r['share_of_group']):.2f} | {pair(model,'D_ambiguity',group,b,'S')} | {pair(model,'D_ambiguity',group,b,'flip')} |")
    lines+=['','Ambiguity counts distinct training types O/PER/LOC/ORG/MISC per case-sensitive focal word, ignoring B/I. The share denominator is the corresponding scorer-predicted group.']
    lines+=['','## D2. Baseline flip concentration','','| Ranking by S | Top rows % | Top row count | Selector flip share % | Random flip share % |','|---|---:|---:|---|---|']
    for rank in ('selector','random'):
        for fraction in (.01,.05,.1):
            f=[next(r for r in concentration if (r['dataset'],r['classifier'],r['group'],r['ranking'],float(r['top_fraction']),r['mask'])==('conll2003','baseline','all',rank,fraction,k)) for k in ('selector','random')]
            def cp(r):return f"{100*float(r['flip_share_mean']):.2f} ± {100*float(r['flip_share_seed_sd']):.2f} (SE {100*float(r['row_se_fixed_ranking']):.2f})"
            lines.append(f"| {rank} | {fraction*100:.0f} | {f[0]['top_rows']} | {cp(f[0])} | {cp(f[1])} |")
    lines+=['','Rows are ranked separately by each mask’s S for each seed. For each ranking, both masks are evaluated on the same selected rows. These are shares of all flips, using draw-averaged flips as weights. Row SE conditions on the ranking; CSVs also contain O/entity-specific concentration.']
    lines+=['','## D3. Frequent forms in the baseline’s top 1%','','| Rank | Selector-ranked word (count) | Random-ranked word (count) |','|---|---|---|']
    lists=[[r for r in words if (r['dataset'],r['classifier'],r['group'],r['ranking'],r['seed'])==('conll2003','baseline','all',kind,'mean')] for kind in ('selector','random')]
    for i in range(20):
        value=[]
        for items in lists:value.append(items[i]['form'].replace('|','\\|')+f" ({items[i]['count']})" if i<len(items) else '—')
        lines.append(f'| {i+1} | {value[0]} | {value[1]} |')
    lines+=['','These counts use the top 1% under seed-averaged S, once per word row. Seed-specific lists and the augmented-arm results are retained in the CSVs. No model or mask is selected from these rankings.','','[Implementation](IMPLEMENTATION_10.md); [all CSVs](tables/sensitivity_1b/). Filled IDs and word-slot offsets are under `results/sensitivity_1b/generate/`; per-draw measurements under `results/sensitivity_1b/measure/`.']
    (ROOT/'report'/'RESULTS_10.md').write_text('\n'.join(lines)+'\n');return {'report':str(ROOT/'report'/'RESULTS_10.md')}
