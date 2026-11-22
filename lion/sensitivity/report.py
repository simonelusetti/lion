"""Compact step-1 tables; all seed/fold row uncertainties remain in CSVs."""
import time,shutil,json
import torch
from ..diagnostics.common import ROOT,absolute,read_csv,write_csv,save_json
from .data import paths,source_root


def experiment(cfg,output):
    root=paths(cfg);rows=read_csv(root/'measure'/'seed_summary.csv');per_seed=read_csv(root/'measure'/'per_classifier_seed.csv');masks=read_csv(root/'measure'/'mask_diagnostics.csv')
    tables=ROOT/'report'/'tables'/'sensitivity';tables.mkdir(parents=True,exist_ok=True)
    for p in (root/'measure').glob('*.csv'):shutil.copy2(p,tables/p.name)
    cuts=read_csv(source_root(cfg)/'calibrate'/'fractions.csv');write_csv(tables/'cuts.csv',cuts)
    source_bounds=read_csv(absolute(cfg.sensitivity.augmentation_run)/'report'/'enriched_bounds.csv');write_csv(tables/'label_references.csv',source_bounds)
    generation=sum(torch.load(p,map_location='cpu',mmap=True,weights_only=False)['seconds'] for p in (root/'generate').glob('*/g*.pt'))
    measured=json.loads((root/'measure'/'timing.json').read_text())['measurement_seconds'];timing=json.loads((root/'timing.json').read_text());timing['ended_unix']=time.time();timing['wall_seconds']=timing['ended_unix']-timing['started_unix'];timing['generation_seconds']=generation;timing['measurement_seconds']=measured
    save_json(root/'timing.json',timing);save_json(tables/'timing.json',timing)
    lines=['# Step 1: sensitivity to selector-droppable context','',f'Generation: **{generation/60:.1f} min**; measurement: **{measured/60:.1f} min**; end-to-end: **{timing["wall_seconds"]/3600:.2f} h**. No model was trained or selected. Steps 2 and 3 were not implemented.','',f'All words in each raw test set use both frozen selectors, {cfg.sensitivity.draws} independent calibration-style fills per mask kind, and the existing full-scale classifiers (three arms × three seeds) plus the frozen scorer. All distributions retain the native nine BIO tags and gamma=.01 uniform mixing; no shared-label projection is applied to these sensitivity measures.','','Main tables use the **row-wise mean of g1 and g2**, then average over rows. S is KL(original || filled), in nats; flips are percentages. Classifier ± values are sample SD across three seeds; scorer has one fitted model. Complete per-classifier/per-seed g1, g2 and mean results, including standard errors over rows, are in [per_classifier_seed.csv](tables/sensitivity/per_classifier_seed.csv). The seed summary also reports row SE after seed-averaging; folds and draws are never treated as independent extra rows.']
    def value(r,metric):
        mean=float(r[metric+'_mean']);sd=r[metric+'_seed_sd'];factor=100 if metric=='flip' else 1
        return f'{factor*mean:.4f}'+(f' ± {factor*float(sd):.4f}' if sd else '')
    for dataset in sorted({r['dataset'] for r in rows}):
        lines+=['',f'## {dataset}'+(' — UNCERTIFIED transfer' if dataset!='conll2003' else ' — in-domain reference'),'','| Classifier | Group | Rows | Selector S | Random S | S ratio | Selector flips % | Random flips % | ID label reference |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for name in ('baseline','selector_augmented','random_augmented','scorer'):
            for group in ('O','entity','all'):
                a=next(r for r in rows if (r['dataset'],r['classifier'],r['fold'],r['group'],r['mask'])==(dataset,name,'mean',group,'selector'))
                b=next(r for r in rows if (r['dataset'],r['classifier'],r['fold'],r['group'],r['mask'])==(dataset,name,'mean',group,'random'))
                ref=float(a['ID_label_reference']);reference=f'{ref:.4f}'+(' (>1, loose)' if ref>1 else '')
                lines.append(f"| {name} | {group} | {a['rows']} | {value(a,'S')} | {value(b,'S')} | {float(a['ratio_mean']):.3f} | {value(a,'flip')} | {value(b,'flip')} | {reference} |")
    lines+=['','## Existing label references and cuts','','The reference is the existing held-out **absolute gamma-smoothed CE UCB**, not the CE increase. Under deterministic labels, appropriate oracle posteriors and selection-conditioned fills, such a loss can bound label flips. The actual fitted scorer and calibration-style independent fills do not guarantee those assumptions. Values below 1 are numerically non-vacuous references, not proof of a valid label-error certificate. Transfer results are uncertified; their numbers in this column are ID references only. “All” uses the target group-frequency-weighted references; the mean-fold column averages the fold references. These derived references are not new joint 95% certificates.','','| Selector | Group | rho | Reference UCB | Below 1 |','|---|---|---:|---:|---|']
    for r in source_bounds:
        lines.append(f"| {r['selector']} | {r['group']} | {float(r['rho']):.2f} | {float(r['absolute_ce_upper_bound']):.4f} | {float(r['absolute_ce_upper_bound'])<1} |")
    lines+=['','## Dropped-set diagnostics','','Coincidence means exact equality of the selector and random masks, including zero-drop and accidental matches. The all-context-dropped share is listed separately (excluding rows with no context). Random and selector masks receive independent fills even when their sets coincide.','','| Dataset | Fold | Group | Mean dropped words | Same-set % | All context dropped % |','|---|---|---|---:|---:|---:|']
    for r in masks:lines.append(f"| {r['dataset']} | {r['fold']} | {r['group']} | {float(r['mean_dropped_words']):.2f} | {100*float(r['same_set_share']):.2f} | {100*float(r['all_context_dropped_share']):.2f} |")
    lines+=['','Per-row draw-averaged measurements and sentence/focal keys are saved under `results/sensitivity/measure/*_rows.pt`; all four filled focal vectors per mask type are cached under `results/sensitivity/generate/`. Full per-seed ratios and row SEs are in ratios.csv. [Implementation](IMPLEMENTATION_9.md).']
    (ROOT/'report'/'RESULTS_9.md').write_text('\n'.join(lines)+'\n');return {'report':str(ROOT/'report'/'RESULTS_9.md')}
