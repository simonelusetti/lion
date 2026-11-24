"""Focused coverage/supply/room report; layer 12 and per-seed detail stay in CSVs."""
import json,shutil,time
from ..diagnostics.common import ROOT,read_csv,save_json
from . import data


def f(v,scale=1):
    if v in ('',None):return '—'
    x=float(v)*scale
    return '∞' if x==float('inf') else f'{x:.4f}'


def classifier(r):return f(r['mean'],100)+' ± '+f(r['seed_sd'],100)+' (SE '+f(r['row_se'],100)+')'


def experiment(cfg,output):
    root=data.root(cfg);tables=ROOT/'report'/'tables'/'conflict';tables.mkdir(parents=True,exist_ok=True)
    for p in (root/'analyse').glob('*.csv'):shutil.copy2(p,tables/p.name)
    for p in [root/'prepare'/'slot_counts.csv',root/'prepare'/'provenance.json',root/'estimate'/'estimate.json']:shutil.copy2(p,tables/p.name)
    rows={name:read_csv(root/'analyse'/f'{name}.csv') for name in ('predictor','coverage','supply','room','paired_gap','natural_draws')};counts=read_csv(root/'prepare'/'slot_counts.csv')
    timing=json.loads((root/'timing.json').read_text());timing['ended_unix']=time.time();timing['stages']={s:json.loads((root/s/'runtime.json').read_text())['seconds'] for s in ('features','context','supply','room','analyse')};timing['wall_seconds']=timing['ended_unix']-timing['started_unix'];save_json(root/'timing.json',timing);save_json(tables/'timing.json',timing)
    projection=json.loads((root/'estimate'/'estimate.json').read_text());audit=read_csv(root/'analyse'/'audit.csv')
    lines=['# Conflict-insertion feasibility: coverage, supply, room','',
      'Full native CoNLL-2003 and EWT train/dev/test splits. CoNLL labels collapse B/I to five entity/O classes; eligible slots are O words and singleton gold entities. EWT uses every word and base dependency relations. Only one new context-only MLP per task was fitted; existing layer-run classifiers and the frozen BERT/MLM are reused. No augmentation training.',
      '', 'Context input removes all focal WordPieces and inserts one [MASK]. Its layer-8 MLP selects a checkpoint using a fixed 10% document holdout inside native train, leaving native dev for calibration. Selection uses max-label confidence ≥τ, regardless of whether the top label equals gold. Confident errors remain in every analysis.',
      '', 'Supply uses the original MLM softmax probabilities on whole single-piece alphanumeric tokens (no ##, punctuation or special tokens); it does not renormalize them. Natural draws alone renormalize over this vocabulary. A word needs ≥5 native-training occurrences and a unique most frequent label; tied maxima are excluded. Conflict additionally requires this label differ from gold and gold share ≤10%. Counts include all native-training occurrences, including CoNLL multiword entity words.',
      '', 'The priced quantity (1−p_ctx(y_c))/p_MLM(w*) is a plug-in estimate, not a certificate. A missing conflict candidate gives price ∞. Quartiles use empirical nearest-rank order statistics, including ∞. Dev CE upper bounds are pointwise one-sided 95% CLT bounds (z=1.645), not simultaneous bounds across thresholds/groups and not guarantees for individual contexts.',
      '', f"Pre-run conservative estimate: {projection['minutes']:.1f} min, assuming all test contexts selected and all eleven insertion slots available. Actual wall-clock: {timing['wall_seconds']/60:.1f} min; feature extraction {timing['stages']['features']/60:.1f}, context fitting {timing['stages']['context']/60:.1f}, supply {timing['stages']['supply']/60:.1f}, room {timing['stages']['room']/60:.1f}, analysis {timing['stages']['analyse']/60:.1f} min.",
      '', 'Classifier estimates are mean ± sample seed SD over the three saved heads, plus word-row SE. Insertions are averaged within each context before averaging contexts. Bins below 50 supporting contexts have *. Layer 8 appears below; layer 12 and per-seed results are in CSVs. Existing BIO-head probabilities are summed over B/I before the CoNLL type argmax.']
    def table(title,headers,values,note):
        lines.extend(['','## '+title,'','| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |'])
        lines.extend('| '+' | '.join(str(x).replace('|','\\|') for x in row)+' |' for row in values);lines.extend(['',note])
    def n(r,key='rows'):return str(r[key])+('*' if r.get('small_bin')=='True' else '')
    table('Eligible slots',['Dataset','Split','Slots','All words','Entity words kept','Entity retention %'],[[r['dataset'],r['split'],r['slots'],r['all_words'],r['retained_entity_words'],f(r['retained_entity_share'],100)] for r in counts],'Entity retention counts original gold entity words; singleton eligibility is applied consistently to fitting, calibration and measurement.')
    table('Context-only predictor',['Dataset','Split','Group','Rows','Accuracy % (SE)','CE (SE)'],[[r['dataset'],r['split'],r['group'],n(r),f(r['accuracy'],100)+' ('+f(r['accuracy_row_se'],100)+')',f(r['ce_mean'])+' ('+f(r['ce_row_se'])+')'] for r in rows['predictor']],'Dev and test are evaluated after the checkpoint is fixed using training data only.')
    table('1: dev coverage and held-out CE bound',['Dataset','τ','Group','Selected / population','Coverage %','Error % (SE)','Mean CE','CE upper 95%'],[[r['dataset'],r['tau'],r['group'],n(r,'selected')+' / '+r['population'],f(r['coverage'],100),f(r['selected_error'],100)+' ('+f(r['selected_error_row_se'],100)+')',f(r['ce_mean']),f(r['ce_upper_95'])] for r in rows['coverage'] if r['split']=='validation'],'Cross-entropy upper-bounds conditional entropy under the stated sampling assumptions. The reported uncertainty concerns mean CE on selected rows, not whether each row fixes a unique label.')
    table('1: test coverage',['Dataset','τ','Group','Subset','Selected / population','Coverage % (SE)'],[[r['dataset'],r['tau'],r['group'],r['subset'],n(r,'selected')+' / '+r['population'],f(r['coverage'],100)+' ('+f(r['coverage_row_se'],100)+')'] for r in rows['coverage'] if r['split']=='id'],'Conflict-prone means the original word has ≥2 training labels, or has training occurrences but none with the current gold label. Unseen words are not included in that subset.')
    values=[]
    for r in rows['supply']:
        values.append([r['dataset'],r['tau'],r['group'],n(r),*[f(r[k+'_mean'])+' ['+f(r[k+'_q25'])+', '+f(r[k+'_median'])+', '+f(r[k+'_q75'])+']' for k in ('conflict_mass','nonconflict_mass')],f(r['plausible_conflict_0.01'],100),f(r['plausible_conflict_0.05'],100)])
    table('2: lexical supply',['Dataset','τ','Group','Rows','Conflict mass mean [Q1, median, Q3]','Non-conflict mass mean [Q1, median, Q3]','Any conflict p≥.01 %','Any conflict p≥.05 %'],values,'Masses and thresholds use raw full-vocabulary MLM probabilities. CSVs also include retained vocabulary mass, row SEs and masses conditional on the allowed vocabulary.')
    table('2: price of the most plausible conflict',['Dataset','τ','Group','Rows','Mean price','Q1 / median / Q3','Price ≤.1 %','Price ≤.2 %','No conflict %'],[[r['dataset'],r['tau'],r['group'],n(r),f(r['priced_bound_mean']),' / '.join(f(r['priced_bound_'+q]) for q in ('q25','median','q75')),f(r['priced_le_0.1'],100),f(r['priced_le_0.2'],100),f(r['no_conflict_word_share'],100)] for r in rows['supply']],'The price numerator uses probability assigned to gold y_c, rather than top-label confidence. Contexts without a conflict candidate remain in the denominator at infinite price.')
    values=[]
    for r in rows['room']:
        if r['layer']!='8' or r['metric']!='context_label':continue
        def other(metric):return next(v for v in rows['room'] if all(v[k]==r[k] for k in ('dataset','tau','group','scheme','layer','kind')) and v['metric']==metric)
        original=other('original_accuracy');word=other('word_label_when_different')
        values.append([r['dataset'],r['tau'],r['group'],r['scheme'],r['kind'],n(r),classifier(r) if r['mean'] else '—',classifier(word) if word['mean'] else '—',classifier(original) if original['mean'] else '—'])
    table('3: layer-8 room',['Dataset','τ','Group','Draw scheme','Kind','Contexts','Predict y_c %','Predict word label % (different)','Original accuracy %'],values,'Natural uses five independent draws with replacement; unclassified draws are logged and never relabelled or retried. Top uses up to three candidates per class. The original reference uses precisely the contexts supporting that row. Predict-word-label is undefined when word label equals y_c.')
    table('3: main paired gap, non-conflict minus conflict',['Dataset','τ','Group','Scheme','Paired contexts','Gap percentage points ± SD (SE)','Original accuracy %'],[[r['dataset'],r['tau'],r['group'],r['scheme'],n(r),classifier(r) if r['mean'] else '—',f(r['original_accuracy'],100)] for r in rows['paired_gap'] if r['layer']=='8'],'Each paired gap uses contexts with at least one insertion from both classes under that scheme; each context receives equal weight. Marginal class supports above can differ, so their raw difference is not the main gap.')
    lines+=['',f"Manual audit: [{len(audit)} random most-plausible conflict insertions](tables/conflict/audit.csv) from τ={min(cfg.conflict.thresholds)}. It includes original/inserted sentences, slot, gold label, full training-label counts/distributions, raw MLM probability and all seed predictions. If fewer than 100 eligible contexts exist, all are included.",'','[Implementation](IMPLEMENTATION_17.md) · [All CSVs](tables/conflict/). Raw arrays: `results/conflict/`.']
    (ROOT/'report'/'RESULTS_17.md').write_text('\n'.join(lines)+'\n');return {'report':str(ROOT/'report'/'RESULTS_17.md')}
