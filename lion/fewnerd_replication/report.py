"""Paired-mask tables with the existing CoNLL values next to Few-NERD."""
import json,time,shutil
import torch
from ..diagnostics.common import ROOT,absolute,read_csv,write_csv,save_json
from ..sensitivity_diagnostics.analyze import reduce


def number(value):return '—' if value in ('',None) else f'{float(value):.4f}'


def estimate(row,mean='mean',sd='seed_sd',se='row_se',scale=1):
    if row is None:return '—'
    def f(key):return number(float(row[key])*scale) if row.get(key) not in ('',None) else None
    text=f(mean) or '—'
    if f(sd):text+=' ± '+f(sd)
    if f(se):text+=' (SE '+f(se)+')'
    return text


def experiment(cfg,output):
    root=absolute(cfg.augmentation.run_path);tables=ROOT/'report'/'tables'/'fewnerd';tables.mkdir(parents=True,exist_ok=True)
    old=absolute(cfg.replication.conll_analysis);new=root/'analyse';datasets=['conll2003','fewnerd'];rows={}
    gates=read_csv(root/'gate'/'gate.csv');status={g:all(r['passed']=='True' for r in gates if r['group']==g) for g in ('O','entity')};status['all']=all(status.values())
    for path in new.glob('*.csv'):
        records=read_csv(path)
        for r in records:
            r['status']='gate_passed' if status.get(r.get('group','all'),False) else 'UNCERTIFIED'
        write_csv(path,records)
    for name in ('summary','A_fills','B_distance','B_matched','D_concentration','D_words'):
        rows[name]=[[r for r in read_csv(p/(name+'.csv')) if r['dataset']==d and r.get('classifier','baseline') in ('baseline','scorer')] for p,d in zip((old,new),datasets)]
        write_csv(tables/(name+'.csv'),rows[name][0]+rows[name][1])
    for name in ('per_seed','B_cells'):
        write_csv(tables/(name+'.csv'),[r for p,d in zip((old,new),datasets) for r in read_csv(p/(name+'.csv')) if r['dataset']==d and r.get('classifier','baseline') in ('baseline','scorer')])
    provenance=json.loads((root/'source'/'prepare'/'provenance.json').read_text());projection=json.loads((root/'estimate.json').read_text())
    gamma=json.loads((root/'source'/'gamma.json').read_text())['gamma']
    part1=sum(json.loads((p/'runtime.json').read_text())['seconds'] for p in [root/'source'/s for s in ('scorer','selector','calibrate')]+[root/'gate',root/'train'])
    part2=sum(json.loads((root/s/'runtime.json').read_text())['seconds'] for s in ('metadata','generate','measure','analyse'))
    timing=json.loads((root/'timing.json').read_text());timing.update(ended_unix=time.time(),part1_seconds=part1,part2_seconds=part2);timing['wall_seconds']=timing['ended_unix']-timing['started_unix'];save_json(root/'timing.json',timing)
    for p in [root/'gate'/'enriched_bounds.csv',root/'gate'/'calibrated_cuts.csv',root/'gate'/'scorer_validation.csv',root/'gate'/'gate.csv',root/'train'/'performance.csv',root/'source'/'prepare'/'counts.csv',root/'source'/'prepare'/'provenance.json',root/'estimate.json',root/'timing.json']:shutil.copy2(p,tables/p.name)
    lines=['# Few-NERD replication: context sensitivity','',f"Pinned `DFKI-SLT/few-nerd`, supervised split, revision `{provenance['revision']}`. Exact file names and SHA-256 values are in [provenance.json](tables/fewnerd/provenance.json). {provenance['tagging']}",
           '',f"Seed {cfg.replication.seed}: 10,531 source training sentences → 2,106 scorer fit + 527 scorer validation + 3,949 T1 + 3,949 T2. Classifiers use exactly T1∪T2 (7,898 sentences), excluding scorer split A. Native validation has 2,438 sentences. Test: {len(torch.load(root/'source'/'prepare'/'sentences.pt',weights_only=False)['sets']['id']):,} sentences, {provenance['test_words']:,} words. CoNLL used 14,041 source / 10,531 classifier sentences. Calibration/held-out proportions remain 70%/30% of the opposite selector half.",
           '',f"Test gold spans seen anywhere in T1∪T2: {provenance['seen_test_gold_spans']:,}/{provenance['test_gold_spans']:,} ({100*provenance['seen_test_span_share']:.2f}%); seen with same type: {100*provenance['seen_same_type_test_span_share']:.2f}%. This is a span-weighted statistic; C below is focal-word weighted.",
           '',f"Before training: Part 1 {projection['used']['part1_seconds']/3600:.2f} h, Part 2 {projection['used']['part2_seconds']/60:.1f} min; total {projection['used']['total_seconds']/3600:.2f} h. Test {'halved' if projection['test_halved'] else 'not halved'} under the six-hour rule. Actual Part 1 {part1/3600:.2f} h; Part 2 {part2/60:.1f} min; pipeline wall-clock {timing['wall_seconds']/3600:.2f} h. Fresh fills were generated once, with token IDs saved from the start; all diagnostics reuse these four draws per row, mask and fold.",
           '',f"Calibration γ = {gamma:.6f}, the new scorer's token error rate on its own validation split. Measurement S retains the step-1 definition: KL(original || filled), γ=.01 over native BIO tags (CoNLL nine, Few-NERD seventeen). A flip changes argmax. Folds and draws are averaged within each focal word; they are not extra observations. Baselines show mean ± sample SD over seeds 0/1/2 plus row SE. Selector/random always use common rows, bins and weights. * marks fewer than 50 paired rows; these cells do not contribute ratios. Failed-gate groups remain in every analysis as UNCERTIFIED.",
           '',"Checkpoint selection is raw native-dev token macro-F1, including O and all BIO tags, exactly the existing matched baseline criterion; this differs from reported span F1. No augmented arms are trained."]
    lines+=['', 'Few-NERD group status: '+', '.join(g+': '+('gate passed' if status[g] else 'UNCERTIFIED') for g in ('O','entity'))+'. These statuses apply to every table below.']
    def table(title,headers,values,note):
        lines.extend(['','## '+title,'','| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |'])
        lines.extend('| '+' | '.join(str(x).replace('|','\\|') for x in row)+' |' for row in values);lines.extend(['',note])
    def find(index,analysis,model,group,bin,mask,metric):
        return next(r for r in rows['summary'][index] if (r['analysis'],r['classifier'],r['group'],r['bin'],r['mask'],r['metric'])==(analysis,model,group,bin,mask,metric))
    def pair(index,analysis,model,group,bin,metric):
        return ' / '.join(estimate(find(index,analysis,model,group,bin,k,metric),scale=100 if metric in ('flip','error') else 1) for k in ('selector','random'))
    def sizes(r):return str(r['rows'])+('*' if r['small_bin']=='True' else '')
    def masks(data,index,group,key,scale=1):
        return ' / '.join(number(scale*float(next(r[key] for r in rows[data][index] if r['group']==group and r['mask']==k))) for k in ('selector','random'))
    # Per-fold calibration and held-out references; the gate is the same lower-bound failure screen as CoNLL.
    oldbounds=read_csv(absolute(cfg.replication.conll_run)/'report'/'enriched_bounds.csv');values=[]
    for fold in ('g1','g2'):
        for group in ('O','entity'):
            cells=[]
            for records in (oldbounds,gates):
                r=next(r for r in records if r['selector']==fold and r['group']==group)
                passed=float(r['rho'])==1 or float(r['lower_bound'])<=cfg.certification.epsilon
                cells.append(f"ρ={float(r['rho']):.2f}; {'pass' if passed else 'UNCERTIFIED'}; ΔCE={number(r['mean_loss'])}; abs CE={number(r['absolute_ce_mean'])} ± {number(r['absolute_ce_sd'])}; n={r['rows']}; UCB={number(r['absolute_ce_upper_bound'])}")
            values.append([fold,group,*cells])
    table('Pipeline: cuts, gate and label-flip reference',['Fold','Group','CoNLL','Few-NERD'],values,"The gate rejects when the held-out CE-increase lower bound exceeds ε=.05; ρ=1 passes by identity. Absolute CE UCB is mean + 1.645 SD/√n. It is an exact label-error reference only for fill-ins conditioned on selection; these are the calibration-style fill-ins. Calibration uses Learn then Test, δ=.1 and the unchanged 0.05 cut grid.")
    oldperformance=read_csv(absolute(cfg.replication.conll_run)/'train'/'per_seed.csv');performance=read_csv(root/'train'/'performance.csv');values=[]
    for model in ('baseline','scorer'):
        for seed in ([0,1,2] if model=='baseline' else [-1]):
            a=next((r for r in oldperformance if r['arm']==model and int(r['seed'])==seed and r['target']=='conll2003' and r['label_space']=='native'),None)
            b=next(r for r in performance if r['classifier']==model and int(r['seed'])==seed)
            values.append([model,seed,number(a['span_f1']) if a else number(b.get('conll_span_f1')),number(b['span_f1']),number(b['precision']),number(b['recall'])])
    for model in ('baseline',):
        a=torch.tensor([float(r['span_f1']) for r in oldperformance if r['arm']==model and r['target']=='conll2003' and r['label_space']=='native']);b=torch.tensor([float(r['span_f1']) for r in performance if r['classifier']==model])
        values.append([model,'mean ± SD',f'{a.mean():.4f} ± {a.std():.4f}',f'{b.mean():.4f} ± {b.std():.4f}','—','—'])
    table('Raw test span F1',['Model','Seed','CoNLL F1','Few-NERD F1','Few precision','Few recall'],values,'All F1 values use the existing exact BIO span implementation. The scorer is one fitted model; baseline summaries use three seeds.')
    # Raw ratios use row-level, fold/draw-averaged influences, not independent fill observations.
    ratios=[]
    for index,(dataset,base) in enumerate(zip(datasets,(old.parent,root))):
        for model in ('baseline','scorer'):
            measured=[torch.load(base/'measure'/f'{dataset}_{model}_{s}.pt',weights_only=False) for s in ([0,1,2] if model=='baseline' else [-1])]
            for group,groupid in [('O',0),('entity',1)]:
                mask=measured[0]['group']==groupid;x=torch.stack([m['kl'][mask].double().mean(-1).mean(1) for m in measured]);p=x.mean(1);r=p[:,0]/p[:,1]
                influence=((x[:,:,0]-r[:,None]*x[:,:,1])/p[:,1,None]).mean(0)
                ratios.append({'dataset':dataset,'classifier':model,'group':group,'rows':int(mask.sum()),'mean':float(r.mean()),'seed_sd':float(r.std()) if len(r)>1 else None,'row_se':float(influence.std()/len(influence)**.5)})
    write_csv(tables/'raw_ratios.csv',ratios);values=[]
    for model in ('baseline','scorer'):
        for group in ('O','entity'):
            cells=[]
            for i in range(2):
                r=find(i,'headline',model,group,'all_draws','selector','S');ratio=next(r for r in ratios if (r['dataset'],r['classifier'],r['group'])==(datasets[i],model,group))
                ref=sum(float(r['absolute_ce_upper_bound']) for r in (oldbounds if i==0 else gates) if r['group']==group)/2
                cells.extend([sizes(r),pair(i,'headline',model,group,'all_draws','S'),pair(i,'headline',model,group,'all_draws','flip'),estimate(ratio),number(ref)])
            values.append([model,group,*cells])
    table('Step 1: all draws',['Model','Group','CoNLL n','CoNLL S sel / rand','CoNLL flips % sel / rand','CoNLL raw ratio','CoNLL reference','Few n','Few S sel / rand','Few flips % sel / rand','Few raw ratio','Few reference'],values,'Raw ratios divide selector S by random S within seed. Label-flip references average the two fold UCBs for the corresponding group.')
    values=[]
    for model in ('baseline','scorer'):
        for group in ('O','entity'):
            cells=[]
            for i in range(2):
                r=find(i,'A_nonnull',model,group,'non_null','selector','S')
                cells.extend([sizes(r),f"{r['excluded_selector']} / {r['excluded_random']}",*[masks('A_fills',i,group,k,scale) for k,scale in [('mean_dropped',1),('mean_changed',1),('restored_fraction',100),('null_swap_fraction',100)]],pair(i,'A_nonnull',model,group,'non_null','S'),pair(i,'A_nonnull',model,group,'non_null','flip')])
            values.append([model,group,*cells])
    table('1b-A: restoration and non-null draws',['Model','Group',*[f'{d} {s}' for d in ('CoNLL','Few') for s in ('paired n','excluded sel / rand','dropped sel / rand','changed sel / rand','restored % sel / rand','null % sel / rand','non-null S sel / rand','non-null flips % sel / rand')]],values,'Restoration counts identical words among dropped positions, case-sensitive. Null swaps include zero-drop draws. Each non-null statistic averages eligible draws per fold, then available folds per word, on common non-null row support; exclusions are mask-specific all-null rows before pairing.')
    for analysis,title in [('B_changed_count','1b-B: changed-word counts'),('B_nearest_changed','1b-B: distance to nearest change'),('C_entity_seen','1b-C: seen gold entities'),('D_ambiguity','1b-D: word-form ambiguity')]:
        values=[]
        for r in rows['summary'][0]:
            if r['analysis']!=analysis or r['mask']!='selector' or r['metric']!='S' or r['group'] not in ('O','entity'):continue
            model,group,bin=r['classifier'],r['group'],r['bin'];cells=[]
            for i in range(2):
                q=find(i,analysis,model,group,bin,'selector','S');cells.extend([sizes(q),pair(i,analysis,model,group,bin,'S'),pair(i,analysis,model,group,bin,'flip')])
                if analysis=='C_entity_seen':cells.append(estimate(find(i,analysis,model,group,bin,'original','error'),scale=100))
                if analysis=='D_ambiguity':cells.append(number(100*float(q['share_of_group'])))
            values.append([model,group,bin,*cells])
        fields=['n','S sel / rand','flips % sel / rand']+(['original error %'] if analysis=='C_entity_seen' else ['share %'] if analysis=='D_ambiguity' else [])
        table(title,['Model','Group','Bin',*[f'{d} {s}' for d in ('CoNLL','Few') for s in fields]],values,{'B_changed_count':'Bins count changed words per draw. Statistics use common rows with draw support in the same bin for both masks.','B_nearest_changed':'Distances count words from the focal position; null draws are excluded. Paired row counts are shown for every bin.','C_entity_seen':'Seen means an exact case-sensitive gold span string in classifier training. Same type additionally matches the entity type. Scorer-predicted entity rows with gold O appear as no_gold_entity.','D_ambiguity':'Ambiguity counts distinct training types per case-sensitive word, ignoring B/I. Few-NERD has O plus eight coarse types; CoNLL has O plus four.'}[analysis])
    values=[]
    for group in ('O','entity'):
        values.append([group,*[masks('B_distance',i,group,k,scale) for i in range(2) for k,scale in [('mean',1),('median',1),('within_two',100)]]])
    table('1b-B: dropped-position distances',['Group',*[f'{d} {s}' for d in ('CoNLL','Few') for s in ('mean sel / rand','median sel / rand','within 2 words % sel / rand')]],values,'Distributions pool dropped word positions over both selector folds, weighting every dropped position equally.')
    values=[]
    for model in ('baseline','scorer'):
        for group in ('O','entity'):
            cells=[]
            for i in range(2):
                r=next(r for r in rows['B_matched'][i] if r['classifier']==model and r['group']==group)
                cells.extend([estimate(r,'matched_ratio_mean','matched_ratio_seed_sd','ratio_row_se_fixed_weights'),r['supported_cells'],number(100*float(r['selector_nonnull_mass_retained']))])
            values.append([model,group,*cells])
    table('1b-B: size- and distance-matched S ratio',['Model','Group',*[f'{d} {s}' for d in ('CoNLL','Few') for s in ('ratio','cells','retained mass %')]],values,'Cell ratios use common rows, then selector cell-frequency weights. Cells below 50 rows or with zero random S are omitted; remaining weights are renormalized. Row SE holds weights fixed and accounts for rows appearing in multiple cells.')
    values=[]
    for ranking in ('selector','random'):
        for fraction in (.01,.05,.1):
            cells=[]
            for i in range(2):
                selected=[next(r for r in rows['D_concentration'][i] if (r['classifier'],r['group'],r['ranking'],float(r['top_fraction']),r['mask'])==('baseline','all',ranking,fraction,k)) for k in ('selector','random')]
                cells.extend([selected[0]['top_rows'],' / '.join(estimate(r,'flip_share_mean','flip_share_seed_sd','row_se_fixed_ranking',100) for r in selected)])
            values.append([ranking,int(fraction*100),*cells])
    table('1b-D: baseline flip concentration',['Ranking','Top %','CoNLL n','CoNLL flip share % sel / rand','Few n','Few flip share % sel / rand'],values,'Each seed ranks rows by draw-averaged S. Both masks are evaluated on the same top rows for each ranking. Row SE conditions on the ranking.')
    lists=[[ [r for r in rows['D_words'][i] if (r['classifier'],r['group'],r['ranking'],r['seed'])==('baseline','all',kind,'mean')] for kind in ('selector','random')] for i in range(2)];values=[]
    for rank in range(20):
        values.append([rank+1,*[(items[rank]['form']+' ('+items[rank]['count']+')') if rank<len(items) else '—' for dataset in lists for items in dataset]])
    table('1b-D: frequent forms in the baseline top 1%',['Rank','CoNLL selector','CoNLL random','Few selector','Few random'],values,'Lists use seed-averaged S and count each focal-word row once. Seed-specific lists are retained in CSVs.')
    lines+=['','[Implementation](IMPLEMENTATION_11.md). [All tables](tables/fewnerd/). Cached fills and per-row arrays: `results/fewnerd_replication/generate/` and `measure/`. Native label inventories, training populations and calibration γ differ as specified; measurement definitions are unchanged.']
    (ROOT/'report'/'RESULTS_11.md').write_text('\n'.join(lines)+'\n')
    return {'report':str(ROOT/'report'/'RESULTS_11.md')}
