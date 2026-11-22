"""Compact dependency-relation sensitivity results; full detail stays in CSVs."""
import json,time,shutil
import torch
from ..diagnostics.common import ROOT,absolute,read_csv,write_csv,save_json
from ..sensitivity_diagnostics.analyze import reduce


def f(value):return '—' if value in ('',None) else f'{float(value):.4f}'


def estimate(r,scale=1,mean='mean',sd='seed_sd',se='row_se'):
    text=f(scale*float(r[mean]))
    if r.get(sd) not in ('',None):text+=' ± '+f(scale*float(r[sd]))
    if r.get(se) not in ('',None):text+=' (SE '+f(scale*float(r[se]))+')'
    return text


def experiment(cfg,output):
    root=absolute(cfg.augmentation.run_path);tables=ROOT/'report'/'tables'/'deprel';tables.mkdir(parents=True,exist_ok=True)
    gate=read_csv(root/'gate'/'gate.csv');passed=all(r['passed']=='True' for r in gate);status='gate_passed' if passed else 'UNCERTIFIED'
    summary=read_csv(root/'analyse'/'summary.csv');performance=read_csv(root/'train'/'performance.csv');bounds=read_csv(root/'gate'/'enriched_bounds.csv')
    provenance=json.loads((root/'source'/'prepare'/'provenance.json').read_text());projection=json.loads((root/'estimate.json').read_text());gamma=json.loads((root/'source'/'gamma.json').read_text())['gamma']
    for p in (root/'analyse').glob('*.csv'):
        records=read_csv(p)
        for r in records:r['status']=status
        write_csv(p,records);shutil.copy2(p,tables/p.name)
    for p in [root/'train'/'performance.csv',root/'source'/'prepare'/'counts.csv',root/'source'/'prepare'/'sample.csv',root/'source'/'prepare'/'provenance.json',root/'gate'/'enriched_bounds.csv',root/'gate'/'calibrated_cuts.csv',root/'gate'/'scorer_validation.csv',root/'gate'/'gate.csv',root/'estimate.json']:shutil.copy2(p,tables/p.name)
    timing=json.loads((root/'timing.json').read_text());timing['ended_unix']=time.time();timing['part1_seconds']=sum(json.loads((p/'runtime.json').read_text())['seconds'] for p in [root/'source'/s for s in ('scorer','selector','calibrate')]+[root/'gate',root/'train']);timing['part2_seconds']=sum(json.loads((root/s/'runtime.json').read_text())['seconds'] for s in ('metadata','generate','measure','analyse'));timing['wall_seconds']=timing['ended_unix']-timing['started_unix'];save_json(root/'timing.json',timing);save_json(tables/'timing.json',timing)
    counts=read_csv(root/'source'/'prepare'/'counts.csv');c={r['split']:r for r in counts}
    lines=['# EWT dependency relations: context sensitivity','',provenance['task'],'',f"Pinned `{provenance['dataset']}`, `en_ewt`, revision `{provenance['revision']}`. {provenance['label_mapping']} File hashes and source IDs: [tables](tables/deprel/).",'',
       f"Source train: {c['train']['sentences']} sentences. Scorer fit/validation: {c['scorer_train']['sentences']}/{c['scorer_validation']['sentences']}; classifier T1/T2: {c['g1']['sentences']}/{c['g2']['sentences']}. Roles are document-disjoint. Native dev/test: {c['validation']['sentences']}/{c['id']['sentences']} sentences, {c['id']['words']} test words. Test {'halved' if projection['test_halved'] else 'not halved'} under the inherited six-hour budget rule.",'',
       f"Part 1 estimate/actual: {projection['used']['part1_seconds']/3600:.2f}/{timing['part1_seconds']/3600:.2f} h. Part 2 estimate/actual: {projection['used']['part2_seconds']/60:.1f}/{timing['part2_seconds']/60:.1f} min. Wall-clock: {timing['wall_seconds']/3600:.2f} h.",'',
       f"One all-token calibration group; both folds: {status}. Calibration γ={gamma:.6f} comes from the scorer's own validation error rate. Measurement S is unchanged KL(original || filled), γ=.01 over {len(provenance['labels'])} native relations. Each test word has four independent fills per mask and fold, saved once with token IDs. Baseline results are mean ± seed SD (three seeds), with word-row SEs. Folds/draws are not additional independent observations. * marks bins below fifty paired rows; these cells do not enter ratios."]
    def table(title,headers,rows,note):
        lines.extend(['','## '+title,'','| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |'])
        lines.extend('| '+' | '.join(str(v).replace('|','\\|') for v in r)+' |' for r in rows);lines.extend(['',note])
    def find(model,analysis,bin,mask,metric):return next(r for r in summary if (r['classifier'],r['analysis'],r['bin'],r['mask'],r['metric'])==(model,analysis,bin,mask,metric))
    def pair(model,analysis,bin,metric):return ' / '.join(estimate(find(model,analysis,bin,k,metric),100 if metric in ('flip','error') else 1) for k in ('selector','random'))
    values=[]
    for model in ('baseline','scorer'):
        records=[r for r in performance if r['classifier']==model]
        for metric in ('accuracy','macro_f1'):
            x=torch.tensor([float(r[metric]) for r in records]);values.append([model,metric,f(float(x.mean())*100),f(float(x.std())*100) if len(x)>1 else '—'])
    table('Raw test performance',['Model','Metric','Mean %','Seed SD'],values,'Checkpoints use raw native-dev token macro-F1, with the same ten-epoch recipe as the previous baseline. No head annotations are input to the classifier and no span score is reported.')
    table('Cuts and held-out reference',['Fold','Retained ρ','Gate','n','ΔCE','Absolute CE mean','SD','Absolute CE UCB'],[[r['selector'],r['rho'],r['status'],r['rows'],f(r['mean_loss']),f(r['absolute_ce_mean']),f(r['absolute_ce_sd']),f(r['absolute_ce_upper_bound'])] for r in gate],"The unchanged gate rejects if the CE-increase lower bound exceeds ε=.05; ρ=1 passes by identity. Absolute CE UCB is mean + 1.645 SD/√n. It is an exact label-error reference only for fills conditioned on selection; these use the calibration-style procedure.")
    ratios=[]
    for model in ('baseline','scorer'):
        measured=[torch.load(root/'measure'/f'en_ewt_deprel_{model}_{s}.pt',weights_only=False) for s in ([0,1,2] if model=='baseline' else [-1])]
        x=torch.stack([v['kl'].double().mean(-1).mean(1) for v in measured]);point=x.mean(1);r=point[:,0]/point[:,1];influence=((x[:,:,0]-r[:,None]*x[:,:,1])/point[:,1,None]).mean(0)
        ratios.append({'classifier':model,'mean':float(r.mean()),'seed_sd':float(r.std()) if len(r)>1 else None,'row_se':float(influence.std()/len(influence)**.5)})
    write_csv(tables/'raw_ratios.csv',ratios)
    table('All-draw sensitivity',['Model','Rows','S sel / rand','Flips % sel / rand','Raw ratio'],[[m,find(m,'headline','all_draws','selector','S')['rows'],pair(m,'headline','all_draws','S'),pair(m,'headline','all_draws','flip'),estimate(next(r for r in ratios if r['classifier']==m))] for m in ('baseline','scorer')],'Each statistic averages draws within fold and folds within word. The masks have the same dropped-word count for each row and fold.')
    fills=read_csv(root/'analyse'/'A_fills.csv')
    table('A: restoration',['Mask','Mean dropped','Mean changed','Restored %','Null swaps %'],[[r['mask'],f(r['mean_dropped']),f(r['mean_changed']),f(100*float(r['restored_fraction'])),f(100*float(r['null_swap_fraction']))] for r in fills],'Restoration is case-sensitive at word level; null swaps include zero-drop draws.')
    table('A: non-null sensitivity',['Model','Paired rows','Excluded sel / rand','Non-null S sel / rand','Flips % sel / rand'],[[m,find(m,'A_nonnull','non_null','selector','S')['rows'],find(m,'A_nonnull','non_null','selector','S')['excluded_selector']+' / '+find(m,'A_nonnull','non_null','selector','S')['excluded_random'],pair(m,'A_nonnull','non_null','S'),pair(m,'A_nonnull','non_null','flip')] for m in ('baseline','scorer')],'Non-null estimates exclude all-null words, average eligible draws within each fold and use common support across masks.')
    for analysis,title in [('B_changed_count','B: changed-word count'),('B_nearest_changed','B: nearest-change distance'),('C_word_seen','C: word and label familiarity'),('D_ambiguity','D: training word-form ambiguity')]:
        values=[]
        for r in summary:
            if r['analysis']!=analysis or r['mask']!='selector' or r['metric']!='S':continue
            m,b=r['classifier'],r['bin'];row=[m,b,r['rows']+('*' if r['small_bin']=='True' else ''),pair(m,analysis,b,'S'),pair(m,analysis,b,'flip')]
            if analysis=='C_word_seen':row.append(estimate(find(m,analysis,b,'original','error'),100))
            if analysis=='D_ambiguity':row.append(f(100*float(r['share_of_group'])))
            values.append(row)
        table(title,['Model','Bin','Rows','S sel / rand','Flips % sel / rand']+(['Original error %'] if analysis=='C_word_seen' else ['Share %'] if analysis=='D_ambiguity' else []),values,'Both masks use the same words within each bin. Familiarity and ambiguity use exact case-sensitive words and base dependency labels in T1∪T2 only; entity-span familiarity is inapplicable.')
    distance=read_csv(root/'analyse'/'B_distance.csv');matched=read_csv(root/'analyse'/'B_matched.csv')
    table('B: dropped-position distances',['Mask','Mean distance','Median','Within two words %'],[[r['mask'],f(r['mean']),f(r['median']),f(100*float(r['within_two']))] for r in distance],'Every dropped position is weighted equally, pooling both folds.')
    table('B: size/distance-matched selector/random ratio',['Model','Ratio ± SD (row SE)','Supported cells','Retained mass %'],[[r['classifier'],estimate(r,mean='matched_ratio_mean',sd='matched_ratio_seed_sd',se='ratio_row_se_fixed_weights'),r['supported_cells'],f(100*float(r['selector_nonnull_mass_retained']))] for r in matched],'Compute ratios within common changed-count × nearest-distance cells, then weight by selector cell frequencies. Omit cells with fewer than fifty rows or zero random S; renormalize remaining weights. Row SE fixes cell weights and accounts for repeated words across cells.')
    concentration=read_csv(root/'analyse'/'D_concentration.csv');values=[]
    for ranking in ('selector','random'):
        for fraction in (.01,.05,.1):
            records=[next(r for r in concentration if (r['classifier'],r['ranking'],float(r['top_fraction']),r['mask'])==('baseline',ranking,fraction,k)) for k in ('selector','random')]
            values.append([ranking,int(100*fraction),records[0]['top_rows'],*[estimate(r,100,'flip_share_mean','flip_share_seed_sd','row_se_fixed_ranking') for r in records]])
    table('D: baseline flip concentration',['Ranking','Top %','Rows','Selector flip share %','Random flip share %'],values,'Rank within seed by draw-averaged S. Each ranking evaluates both masks on the same selected words; row SE conditions on the ranking.')
    words=read_csv(root/'analyse'/'D_words.csv');lists=[[r for r in words if (r['classifier'],r['ranking'],r['seed'])==('baseline',k,'mean')] for k in ('selector','random')]
    table('D: frequent forms in top 1%',['Rank','Selector ranking','Random ranking'],[[i+1,*[a[i]['form']+' ('+a[i]['count']+')' if i<len(a) else '—' for a in lists]] for i in range(20)],'Counts use seed-averaged S, once per focal-word row. Seed-specific lists and sensitivity by gold dependency relation are in CSVs.')
    lines+=['','Differences from NER: relation labels and token metrics; one calibration group; document-disjoint source roles; native EWT dev/test sizes; word-label familiarity replaces entity-span familiarity. The scorer, selector, fill and sensitivity algorithms are reused.','', '[Implementation](IMPLEMENTATION_12.md) · [CSV tables](tables/deprel/) · [Previous Few-NERD results](old/11_fewnerd_replication/RESULTS_11.md).']
    (ROOT/'report'/'RESULTS_12.md').write_text('\n'.join(lines)+'\n')
    return {'report':str(ROOT/'report'/'RESULTS_12.md')}
