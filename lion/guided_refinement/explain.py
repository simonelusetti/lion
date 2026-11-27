"""Text, exposure, label and baseline-difficulty descriptors of saved augmentations."""
import json,sys,time
from collections import Counter,defaultdict
import torch
from torch.utils.data import DataLoader
from forge import run,start_run
from omegaconf import OmegaConf
from tagger.ensemble import IndependentReadouts
from ..diagnostics.common import ROOT,absolute,save_json,write_csv,read_csv
from ..augmentation.train import schedule
from ..label_change_grid.train import Steps
from ..metrics import shared_label
from .population import load,correlation


def collate(rows):return torch.stack(rows)


def distribution(values):
    t=torch.tensor(list(values),dtype=torch.double)
    q=torch.quantile(t,torch.tensor([.5,.9,.99],dtype=torch.double))
    return {'mean':float(t.mean()),'median':float(q[0]),'p90':float(q[1]),'p99':float(q[2]),'max':int(t.max())}


def lexical(examples,indices,vocabulary,bigrams):
    chosen=[examples[i] for i in indices];tokens=[t for e in chosen for t in e['tokens']];pairs=[t for e in chosen for t in e['bigrams']]
    token_types=set(tokens);pair_types=set(pairs);word_count=Counter(e['word'] for e in chosen)
    return {'distinct_sources':len({e['source'] for e in chosen}),'distinct_focal_forms':len(word_count),'top10_focal_share':sum(c for _,c in word_count.most_common(10))/len(chosen),
        'distinct_focal_examples':len({e['key'] for e in chosen}),'nonfocal_token_positions':len(tokens),'token_types':len(token_types),'token_type_token_ratio':len(token_types)/len(tokens) if tokens else None,
        'bigram_types':len(pair_types),'bigram_type_token_ratio':len(pair_types)/len(pairs) if pairs else None,
        'novel_token_position_share':sum(t not in vocabulary for t in tokens)/len(tokens) if tokens else None,
        'novel_token_type_share':len(token_types-vocabulary)/len(token_types) if token_types else None,
        'novel_bigram_position_share':sum(t not in bigrams for t in pairs)/len(pairs) if pairs else None,
        'novel_bigram_type_share':len(pair_types-bigrams)/len(pair_types) if pair_types else None,
        'mean_piece_edit_fraction':sum(e['piece_edit_fraction'] for e in chosen)/len(chosen),
        'mean_word_slot_edit_fraction':sum(e['word_slot_edit_fraction'] for e in chosen)/len(chosen)}


def weighted(values,weights):return float(values.double()@weights.double()/weights.sum()) if weights.sum()>0 else None


def report(cfg):
    output=absolute(cfg.explain.results);source=absolute(cfg.explain.source);arms=list(cfg.mechanism_pilot.arms)
    summaries=read_csv(output/'arm_summary.csv');examples=read_csv(output/'accepted_examples.csv');labels=read_csv(output/'label_composition.csv');matched=read_csv(output/'matched_summary.csv');exposures=read_csv(output/'exposure_per_seed.csv');joined=read_csv(output/'downstream_joined.csv');correlations=read_csv(output/'correlations.csv')
    label_names=[r['label'] for r in labels if r['arm']=='baseline' and r['scope']=='pool'];extra=[]
    for r in summaries:
        pool=[e for e in examples if e['arm']==r['arm']];n=len(pool)
        r['shared_space_label_change_share']=sum(shared_label(e['source_gold'])!=shared_label(e['assigned_label']) for e in pool)/n
        r['shared_space_assigned_O_share']=sum(shared_label(e['assigned_label'])=='O' for e in pool)/n
        for label in label_names:r[f'assigned_{label}_share']=sum(e['assigned_label']==label for e in pool)/n
        for label in ['all',*label_names]:
            chosen=pool if label=='all' else [e for e in pool if e['assigned_label']==label]
            extra.append({'arm':r['arm'],'assigned_native_label':label,'examples':len(chosen),'shared_label':shared_label(label) if label!='all' else '',
                'shared_label_changed_count':sum(shared_label(e['source_gold'])!=shared_label(e['assigned_label']) for e in chosen),'shared_label_change_proportion':sum(shared_label(e['source_gold'])!=shared_label(e['assigned_label']) for e in chosen)/len(chosen) if chosen else None})
    write_csv(output/'shared_label_changes.csv',extra);write_csv(output/'arm_summary.csv',summaries);by_arm={r['arm']:r for r in summaries}
    metrics=['shared_space_label_change_share','shared_space_assigned_O_share',*[f'assigned_{label}_share' for label in label_names]]
    correlations=[r for r in correlations if r['metric'] not in metrics]
    for r in joined:r.update({k:by_arm[r['arm']][k] for k in metrics})
    write_csv(output/'downstream_joined.csv',joined)
    for group,names in [('all_11',arms),('refinement_5',list(cfg.guided.arms))]:
        for dataset,space in sorted({(r['target'],r['space']) for r in joined}):
            subset=[r for r in joined if r['arm'] in names and (r['target'],r['space'])==(dataset,space)];y=torch.tensor([float(r['paired_delta_vs_baseline']) for r in subset],dtype=torch.double)
            for metric in metrics:
                x=torch.tensor([float(r[metric]) for r in subset],dtype=torch.double)
                for kind in ('Pearson','Spearman'):
                    value,reason=correlation(x,y,kind=='Spearman');correlations.append({'group':group,'outcome':dataset+'/'+space,'metric':metric,'kind':kind,'value':value,'undefined_reason':reason,'arms':len(subset)})
    write_csv(output/'correlations.csv',correlations)
    def value(row,key):return float(row[key])
    def corr(group,outcome,metric):
        rows=[r for r in correlations if r['group']==group and r['outcome']==outcome and r['metric']==metric]
        return ' / '.join(f"{float(r['value']):+.3f}" if r['value'] not in (None,'') else 'undefined ('+r['undefined_reason']+')' for r in rows)
    def match(arm,count,metric):return next(r for r in matched if r['arm']==arm and int(r['count'])==count and r['metric']==metric)
    lines=['# Why quality scores and downstream gains differ','',
        'Analysis of the completed eleven-arm, 20,000-proposal small experiment. No new generation or training. Only existing baseline-head inference on cached accepted features was added (CPU float32, three existing seed-matched checkpoints). No encoder/scorer inference or geometry. IID sampling remains an explicit unverified working assumption; this report does not repeat the provenance audit.','',
        'The data support quantity/exposure, label composition and baseline difficulty as distinct variables that the quality formulas do not jointly measure. They do not establish generated gold correctness or a causal correctness-versus-diversity trade-off.','',
        '## Quantity and exact exposure','',
        'All arms run 380 optimizer steps over ten epochs, 603,210 supervised-token presentations. Equal total budgets do **not** equalize gold or augmented supervision. Augmented draws replace original-token presentations within the fixed budget. Mixtures are proportional to pool size, implemented by deterministic rounded quotas. Each head loss is mean CE across its labelled tokens; every original/augmented token has coefficient 1 before batch normalization. No separate augmentation weight, quality weight, class weight or generator-lambda coefficient enters student loss. Per-example sum of 1/batch-size coefficients and actual effective augmentation loss shares are saved.','',
        '| Arm | Accepted / distinct | Original-gold draws | Augmented draws | Reuse per pool entry | Mean selected epoch |','|---|---:|---:|---:|---:|---:|']
    for r in summaries:lines.append(f"| {r['arm']} | {r['accepted']} / {r['distinct_accepted']} | {value(r,'gold_draws_full_run'):,.0f} | {value(r,'augmented_draws_full_run'):,.0f} | {value(r,'mean_reuse_full_run'):.3f} | {value(r,'mean_checkpoint_epoch'):.2f} |")
    lines+=['','Baseline: 603,210 original-gold draws, zero augmented draws; checkpoint epoch 10 for all seeds. Every augmentation pool entry is encountered, including before its selected checkpoint. The sole exact repeated focal example is in window70 (236 entries, 235 distinct). Reuse is essentially equal across pools, not a large-arm advantage in reduced repetition.','',
        'Exposure is reconstructed exactly from the original schedule and Steps RNG, not estimated. Full-run counts reproduce budgets.csv for every seed. Checkpoints used for evaluation can predate epoch 10: e.g. lambda0.01 uses epochs 9/8/9, versus baseline 10/10/10. Its mean checkpoint-prefix augmented exposure is 16,590, rather than 19,143 full-run draws. Separate per-seed full-run and checkpoint-prefix original/augmented counts are in exposure_per_seed.csv; presentation counts and loss coefficients per pool entry in exposure_per_example.csv.','',
        '## Coverage, text and baseline difficulty','',
        '| Arm | Sources | Case-sensitive forms | Top-10 form share | Changed nonfocal pieces | Baseline disagreement | Baseline pseudo-label CE | Baseline pseudo-label probability |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries:lines.append(f"| {r['arm']} | {r['distinct_sources']} | {r['distinct_focal_forms']} | {100*value(r,'top10_focal_share'):.1f}% | {100*value(r,'mean_piece_edit_fraction'):.1f}% | {100*value(r,'baseline_disagreement'):.1f}% | {value(r,'baseline_cross_entropy'):.3f} | {value(r,'baseline_pseudo_probability'):.3f} |")
    lines+=['','Baseline difficulty is averaged across three existing baseline students; it compares their predictions with assigned pseudo-labels, not unknown gold labels. Most accepted examples already agree with those students: uniform 85.1%, lambda0.01 68.7%, greedy 64.6%. Guidance supplies more baseline-disagreeing examples, but these are not established true counterexamples. Pool-weighted and exactly exposure-weighted metrics, overall and per assigned class, are in baseline_difficulty.csv. Their differences are small because complete shuffled passes over each pool make per-entry reuse nearly uniform.','',
        'Source and form multiplicity distributions (mean/median/p90/p99/max), distinct assigned labels per form and multi-label form counts are in arm_summary.csv. Full source and case-sensitive form frequency/label tables: coverage_by_source.csv and coverage_by_word.csv.','',
        'Text descriptors use original aligned nonfocal WordPiece positions, excluding focal/special/padding positions. Original-word-slot edit fractions are also saved; generated word boundaries are not revalidated. Lexical statistics are WordPiece tokens and contiguous bigrams, never bridging focal/special holes, with novelty relative to the original 2400-sentence student corpus. They are not fluency or learning-usefulness scores.','',
        '## Matched accepted counts','',
        '100 repeated subsamples without replacement at every feasible size among 25/100/200/500/1000. No small-pool upsampling. Below, 1000-example refinement subsets show that broader source/form coverage persists after matching count, whereas token-vocabulary size is similar. Values are subsample mean ± SD, not independent-run confidence intervals.','',
        '| Arm | Sources at n=1000 | Forms at n=1000 | Token types at n=1000 | Novel-bigram position share |','|---|---:|---:|---:|---:|']
    for arm in cfg.guided.arms:
        cells=[]
        for metric in ('distinct_sources','distinct_focal_forms','token_types','novel_bigram_position_share'):
            r=match(arm,1000,metric);factor=100 if metric=='novel_bigram_position_share' else 1;cells.append(f"{factor*value(r,'mean'):.2f} ± {factor*value(r,'subsample_sd'):.2f}"+('%' if factor==100 else ''))
        lines.append('| '+arm+' | '+' | '.join(cells)+' |')
    lines+=['','All eleven arms can be compared at n=25. The mean distinct focal forms range from 18.70 (sentence20) to 23.91 (lambda0.01); the refinement arms are similar at this very small count. At n=1000, lambda0.01 covers about 511 forms versus uniform 426, but has about 8907 token types versus uniform 9051. Thus word coverage improves without a general increase in matched-count vocabulary diversity. Full counts and variability for every feasible arm/size/descriptor are in matched_summary.csv; individual draws in matched_subsamples.csv.','',
        '## Label composition and transfer mapping','',
        'Original training labels are 78.93% O and 1.46% art. The coarse-label mixture changes substantially under guidance; pool label proportions, original labels, combined training label proportions and full 9×9 source-gold→assigned transitions are saved for pools and each seed/exposure scope.','',
        '| Arm | Assigned O | Assigned art | Assigned shared-space O | Source→assigned label changes in shared space |','|---|---:|---:|---:|---:|']
    for r in summaries:lines.append(f"| {r['arm']} | {100*value(r,'pool_O_share'):.2f}% | {100*value(r,'assigned_art_share'):.2f}% | {100*value(r,'shared_space_assigned_O_share'):.2f}% | {100*value(r,'shared_space_label_change_share'):.2f}% |")
    lines+=['','Native art, event, other and product labels map to O in the existing PER/ORG/LOC transfer evaluation; building maps to LOC. All accepted examples change native labels, but many do not change the projected label. Uniform has 110 O→art examples; lambda0.01 has 515; greedy has 560. This class-composition difference is a plausible reason native and transfer effects diverge, not proof of causation.','',
        'The shared-mapping diagnostic is descriptive only; student training still uses the original nine native classes. exposure-weighted class/transition tables preserve the actual training counts. No evaluation mapping was changed.','',
        '## Quality formulations and descriptive associations','',
        'All five quality columns remain separate: direct binwise plug-in/lower, previous worst-ratio plug-in/lower, and original G=max(0,1−.028/a). Definitions and original confidence allocations are unchanged; no maximum of certificates is used. The direct plug-in spans only 97.17–97.40%, while its lower bound is strongly associated with accepted count (Pearson +.936 across 11 arms, +.940 within refinements). G is an acceptance transformation (+.988/+ .996 correlation with accepted count). The ratio plug-in instead correlates negatively with count (−.921 across 11), reflecting its different distribution-ratio penalty. None is measured generated-label accuracy.','',
        '| Arm | Direct plug-in | Direct lower | Ratio plug-in | Ratio lower | Old G |','|---|---:|---:|---:|---:|---:|']
    for r in summaries:lines.append('| '+r['arm']+' | '+' | '.join(f'{100*value(r,k):.4f}%' for k in ('binwise_plugin','binwise_lower','ratio_plugin','ratio_lower','old_G'))+' |')
    for group in ('all_11','refinement_5'):
        lines+=['',f'{group}: correlations with seed-paired F1 improvement; Pearson / Spearman. All descriptors and quality measures are included in correlations.csv.','',
            '| Descriptor | Few-NERD native | CoNLL | WNUT | WikiANN |','|---|---:|---:|---:|---:|']
        for metric in ('binwise_plugin','binwise_lower','ratio_plugin','ratio_lower','old_G','accepted','distinct_focal_forms','baseline_disagreement','assigned_art_share','mean_checkpoint_epoch'):
            cells=[corr(group,dataset,metric) for dataset in ('fewnerd/native','conll2003/shared_ner','wnut/shared_ner','wikiann/shared_ner')]
            lines.append('| '+metric+' | '+' | '.join(cells)+' |')
    lines+=['','Across eleven arms, WNUT gains correlate with accepted count (+.911), form coverage (+.914), changed-piece fraction (+.928) and the direct lower bound (+.951). Within the five refinements, the direct-bound/WNUT correlation drops to +.136; native Few-NERD is −.955. These related configurations do not separate quantity, labels, generator family and checkpoint selection causally.','',
        '## Existing downstream changes','',
        '| Arm | Few-NERD native ΔF1 | CoNLL ΔF1 | WNUT ΔF1 | WikiANN ΔF1 |','|---|---:|---:|---:|---:|']
    for arm in arms:
        cells=[next(r for r in joined if r['arm']==arm and (r['target'],r['space'])==dataset) for dataset in [('fewnerd','native'),('conll2003','shared_ner'),('wnut','shared_ner'),('wikiann','shared_ner')]]
        lines.append('| '+arm+' | '+' | '.join(f"{100*float(r['paired_delta_vs_baseline']):+.3f}" for r in cells)+' |')
    lines+=['','Differences are percentage points. Existing span precision and recall, seed SDs and paired differences are in downstream_joined.csv. WNUT illustrates distinct operating-point changes: uniform Δprecision +3.756 / Δrecall −.494 points, whereas lambda0.01 +.765 / +1.893 and greedy −.562 / +2.263. Similar F1 gains therefore need not reflect the same mechanism. Native Few-NERD precision falls for lambda0.01 and greedy while recall increases.','',
        '## Supported explanations, limits and next controlled comparison','',
        'Supported: the direct lower-bound ranking substantially reflects sample-size uncertainty; accepted counts determine augmented exposure; exact duplicates and unequal reuse are negligible; guidance changes case-sensitive focal coverage, assigned class balance and baseline difficulty, including at matched pool size; coarse-label changes and transfer-label changes are different; checkpoint-prefix exposure differs even with equal maximum budgets. Quality alone does not measure these training ingredients.','',
        'Still speculative: which of quantity, label composition, difficulty or checkpoint selection causes each domain gain, and whether assigned labels are actually correct. There is no evidence here for a correctness-versus-diversity trade-off. High lexical novelty is a descriptor, not a benefit or a fluency assessment. Correlations are exploratory, unadjusted and not eleven independent causal interventions.','',
        'Smallest useful controlled comparison (not launched): uniform versus lambda0.01 with 1000 accepted examples per arm, the same original-gold and augmented-token presentation counts, the same initialization/order/seeds and fixed checkpoint step; stratify accepted sampling by assigned native label on common support. Reuse the existing matched baseline. This removes the current quantity, repetition, class-mixture and selected-step differences; any remaining difference still includes source/context composition and cannot be interpreted as true-label quality. If label-stratified common support cannot supply 1000 entries without replacement, lower the common count instead of duplicating a small stratum.','',
        'Files: arm_summary.csv; exposure_per_seed.csv and exposure_per_example.csv; coverage_by_source.csv and coverage_by_word.csv; label_composition.csv and label_transitions.csv; shared_label_changes.csv; matched_summary.csv; baseline_difficulty.csv; downstream_joined.csv; correlations.csv. Numerical and sampling conventions: method.json.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n');return 0


def main(cfg):
    if cfg.stage.experiment=='explain_report':return report(cfg)
    OmegaConf.resolve(cfg);cfg.forge.store=str(absolute(cfg.forge.store));tracking=start_run(cfg);torch.set_num_threads(4);started=time.monotonic()
    source=absolute(cfg.explain.source);output=absolute(cfg.explain.results);output.mkdir(parents=True,exist_ok=True);OmegaConf.save(cfg,output/'config.yaml')
    print('Loading saved pools and textual metadata',flush=True)
    raw=load(source/'prepare'/'raw.pt');meta=load(source/'prepare'/'sources_targets.pt');bank=load(absolute(cfg.certification.run_path)/'prepare'/'sentences.pt');labels=meta['labels'];C=len(labels)
    selected=set(json.loads((source/'prepare'/'subset.json').read_text())['source_ids']);original=[r for r in bank['sets']['train'] if r['source_id'] in selected];lookup={r['source_id']:r for r in original};special=set(bank['special']['excluded']);vocabulary=set();original_bigrams=set()
    for row in original:
        ids=row['ids'].tolist();valid=[t not in special for t in ids];vocabulary.update(t for t,k in zip(ids,valid) if k);original_bigrams.update((ids[i],ids[i+1]) for i in range(len(ids)-1) if valid[i] and valid[i+1])
    quality_rows=read_csv(source/'reanalysis_binwise'/'quality.csv');quality={r['arm']:r for r in quality_rows};arms=list(cfg.mechanism_pilot.arms)
    pools={};examples={};summaries=[];coverage_sources=[];coverage_words=[];example_rows=[];label_rows=[];transition_rows=[]
    base_counts=torch.bincount(raw['y'],minlength=C).double()
    for c,label in enumerate(labels):label_rows.append({'arm':'baseline','seed':'pool','scope':'pool','label':label,'augmented_count':0,'augmented_proportion':0.,'original_gold_count':int(base_counts[c]),'original_gold_proportion':float(base_counts[c]/base_counts.sum())})
    for arm in arms:
        saved=load(source/'generate'/arm/'accepted_features_draw0.pt');generated=load(source/'generate'/arm/'replicate0.pt');pools[arm]=saved;rows=[];source_forms=defaultdict(set);word_labels=defaultdict(set);word_sources=defaultdict(set)
        for index,anchor in enumerate(saved['anchor_id'].tolist()):
            a,b=generated['offsets'][anchor:anchor+2].tolist();ids=generated['state_input_ids'][-1,a:b].long();source_id=meta['source_id'][anchor];focal=int(meta['focal'][anchor]);fstart=int(meta['focal_start'][anchor]);fend=int(meta['focal_end'][anchor]);original_row=lookup[source_id];old=original_row['ids']
            ordinary=(original_row['word_ids']>=0)&(original_row['word_ids']!=focal)
            eligible=ordinary&~torch.isin(old,torch.tensor(list(special)))&~torch.isin(ids,torch.tensor(list(special)))
            changed=(ids!=old)&eligible;word_ids=original_row['word_ids'];slot_ids=word_ids[eligible].unique();slots_changed=torch.zeros(len(original_row['focals']),dtype=torch.bool).scatter_reduce_(0,word_ids[changed],torch.ones(int(changed.sum()),dtype=torch.bool),reduce='amax',include_self=True)
            tokens=ids[eligible].tolist();positions=(eligible[:-1]&eligible[1:]).nonzero().flatten().tolist();pairs=[(int(ids[p]),int(ids[p+1])) for p in positions];word=meta['forms'][anchor];y=int(saved['y'][index]);gold=int(meta['gold'][anchor]);source_forms[source_id].add(word);word_labels[word].add(y);word_sources[word].add(source_id)
            piece_fraction=float(changed.sum()/eligible.sum()) if eligible.any() else 0.;word_fraction=float(slots_changed[slot_ids].float().mean()) if len(slot_ids) else 0.
            key=(tuple(ids.tolist()),fstart,fend)
            rows.append({'key':key,'source':source_id,'word':word,'y':y,'gold':gold,'tokens':tokens,'bigrams':pairs,'piece_edit_fraction':piece_fraction,'word_slot_edit_fraction':word_fraction})
            example_rows.append({'arm':arm,'pool_index':index,'proposal_index':anchor,'source_id':source_id,'focal_word':word,'source_gold':labels[gold],'assigned_label':labels[y],'piece_edit_fraction':piece_fraction,'original_word_slot_edit_fraction':word_fraction,'focal_pieces_unchanged':torch.equal(ids[fstart:fend],old[fstart:fend])})
        examples[arm]=rows;m=len(rows);sc=Counter(e['source'] for e in rows);wc=Counter(e['word'] for e in rows);source_dist=distribution(sc.values());word_dist=distribution(wc.values());labdiv=distribution(len(v) for v in word_labels.values())
        q=quality[arm];desc=lexical(rows,range(m),vocabulary,original_bigrams)
        summary={'arm':arm,'proposals':cfg.population.proposals,'accepted':m,'acceptance':m/cfg.population.proposals,'distinct_accepted':len({r['key'] for r in rows}),
            'duplicate_focal_rate':1-len({r['key'] for r in rows})/m,'duplicate_sentence_rate':1-len({r['key'][0] for r in rows})/m,
            **desc,**{f'accepted_per_source_{k}':v for k,v in source_dist.items()},**{f'accepted_per_form_{k}':v for k,v in word_dist.items()},**{f'assigned_labels_per_form_{k}':v for k,v in labdiv.items()},
            'forms_with_multiple_assigned_labels':sum(len(v)>1 for v in word_labels.values()),'pool_O_share':float((saved['y']==labels.index('O')).double().mean()),
            'binwise_plugin':float(q['plugin_quality']),'binwise_lower':float(q['binwise_lower']),'ratio_plugin':float(q['previous_worst_ratio_plugin']),'ratio_lower':float(q['previous_worst_ratio_lower']),
            'old_G':float(q['old_G']),'binwise_mass_uncertainty_loss':float(q['mass_uncertainty_loss']),'binwise_calibration_uncertainty_loss':float(q['calibration_uncertainty_loss'])}
        summaries.append(summary)
        for s,count in sc.items():coverage_sources.append({'arm':arm,'source_id':s,'accepted':count,'distinct_focal_forms':len(source_forms[s])})
        for word,count in wc.items():coverage_words.append({'arm':arm,'focal_word':word,'accepted':count,'distinct_sources':len(word_sources[word]),'distinct_assigned_labels':len(word_labels[word]),'assigned_labels':'|'.join(labels[v] for v in sorted(word_labels[word]))})
        y=saved['y'];g=torch.tensor([e['gold'] for e in rows]);counts=torch.bincount(y,minlength=C);trans=torch.bincount(g*C+y,minlength=C*C).reshape(C,C)
        for c,label in enumerate(labels):label_rows.append({'arm':arm,'seed':'pool','scope':'pool','label':label,'augmented_count':int(counts[c]),'augmented_proportion':float(counts[c]/m),'original_gold_count':int(base_counts[c]),'original_gold_proportion':float(base_counts[c]/base_counts.sum())})
        for s in range(C):
            for t in range(C):transition_rows.append({'arm':arm,'seed':'pool','scope':'pool','source_gold':labels[s],'assigned':labels[t],'count':int(trans[s,t]),'proportion':float(trans[s,t]/m)})
        print(f'{arm}: {m} accepted, {desc["distinct_sources"]} sources, {desc["distinct_focal_forms"]} forms',flush=True)
    write_csv(output/'pool_summary.csv',summaries);write_csv(output/'accepted_examples.csv',example_rows);write_csv(output/'coverage_by_source.csv',coverage_sources);write_csv(output/'coverage_by_word.csv',coverage_words)
    print('Reconstructing full-run and selected-checkpoint exposure',flush=True)
    exposures=[];weights={};loss_weights={};counts_by_example=[];augmented=[pools[a] for a in arms];budget_rows=read_csv(source/'train'/'budgets.csv');checkpoints={}
    for seed in cfg.augmentation.seeds:
        saved=load(source/'train'/f'seed{seed}'/'model.pt');checkpoints[seed]=saved;batches,plan=schedule(raw['sentence'],cfg,seed);dataset=Steps(raw,augmented,batches,plan,seed)
        for h,name in enumerate(['baseline',*arms]):
            selected_epoch=int(saved['epoch'][h]);scopes=[('full_run',cfg.train.epochs),('selected_checkpoint',selected_epoch)]
            for scope,epoch in scopes:
                stop=sum(b['epoch']<=epoch for b in batches);sizes=plan['labelled_tokens_per_step'][:stop];total=int(sizes.sum());original_ids=torch.cat([b['baseline'] for b in batches[:stop]]) if h==0 else torch.cat([dataset.slices[h-1][i][0] for i in range(stop)])
                if h:
                    variants=torch.cat([dataset.slices[h-1][i][1] for i in range(stop)]);w=torch.bincount(variants,minlength=len(pools[name]['y']));mass=torch.zeros(len(w),dtype=torch.double)
                    for i in range(stop):
                        v=dataset.slices[h-1][i][1];mass.scatter_add_(0,v,torch.full((len(v),),1/float(sizes[i]),dtype=torch.double))
                    weights[seed,name,scope]=w;loss_weights[seed,name,scope]=mass
                    distinct_keys={examples[name][i]['key'] for i in w.nonzero().flatten().tolist()};distinct_indices=int((w>0).sum());nv=int(w.sum());pool_size=len(w)
                    for i,(count,loss_mass) in enumerate(zip(w.tolist(),mass.tolist())):counts_by_example.append({'seed':seed,'arm':name,'scope':scope,'pool_index':i,'presentations':count,'sum_per_step_loss_coefficients':loss_mass})
                else:w=torch.empty(0,dtype=torch.long);mass=torch.empty(0,dtype=torch.double);distinct_keys=set();distinct_indices=nv=pool_size=0
                nr=len(original_ids);actual_ratio=nv/total;pool_ratio=pool_size/(len(raw['y'])+pool_size)
                entry={'seed':seed,'arm':name,'scope':scope,'checkpoint_epoch':selected_epoch,'epochs_in_scope':epoch,'optimizer_steps':stop,'total_supervised_tokens':total,'original_gold_tokens':nr,'augmented_tokens':nv,'augmentation_pool_size':pool_size,
                    'distinct_pool_indices_encountered':distinct_indices,'distinct_focal_examples_encountered':len(distinct_keys),'mean_reuse_per_pool_index':nv/pool_size if pool_size else 0.,'mean_reuse_per_distinct_focal_example':nv/len(distinct_keys) if distinct_keys else 0.,
                    'target_augmented_probability':pool_ratio,'actual_augmented_token_share':actual_ratio,'augmented_loss_mass_share':float(mass.sum()/stop),'per_token_loss_weight':1.,'counts_type':'exact reconstruction from original seeded schedule, not expectation'}
                exposures.append(entry)
                if scope=='full_run':
                    budget=next(r for r in budget_rows if int(r['seed'])==seed and r['head']==name)
                    if (int(budget['original_token_presentations']),int(budget['augmented_token_presentations']))!=(nr,nv):raise ValueError('Exposure does not reproduce saved budgets')
                original_counts=torch.bincount(raw['y'][original_ids],minlength=C).double()
                if h:
                    assigned=pools[name]['y'];gold=torch.tensor([e['gold'] for e in examples[name]]);augcounts=torch.zeros(C,dtype=torch.double).scatter_add_(0,assigned,w.double());trans=torch.zeros(C*C,dtype=torch.double).scatter_add_(0,gold*C+assigned,w.double()).reshape(C,C)
                    for s in range(C):
                        for t in range(C):transition_rows.append({'arm':name,'seed':seed,'scope':scope,'source_gold':labels[s],'assigned':labels[t],'count':int(trans[s,t]),'proportion':float(trans[s,t]/nv) if nv else None})
                else:augcounts=torch.zeros(C,dtype=torch.double)
                for c,label in enumerate(labels):label_rows.append({'arm':name,'seed':seed,'scope':scope,'label':label,'augmented_count':int(augcounts[c]),'augmented_proportion':float(augcounts[c]/nv) if nv else 0.,'original_gold_count':int(original_counts[c]),'original_gold_proportion':float(original_counts[c]/nr),'combined_count':int(original_counts[c]+augcounts[c]),'combined_proportion':float((original_counts[c]+augcounts[c])/total)})
    write_csv(output/'exposure_per_seed.csv',exposures);write_csv(output/'exposure_per_example.csv',counts_by_example);write_csv(output/'label_composition.csv',label_rows);write_csv(output/'label_transitions.csv',transition_rows)
    print('Existing baseline-head inference on cached accepted features (CPU float32)',flush=True)
    difficulty=[];difficulty_by_example=[];baseline_prob={}
    with torch.inference_mode():
        for seed,saved in checkpoints.items():
            baseline=IndependentReadouts(cfg.tagger,768,C,1).eval().requires_grad_(False);baseline.load_state_dict({k:v[:1] for k,v in saved['state'].items()})
            for arm in arms:
                pool=pools[arm];parts=[]
                for x in DataLoader(pool['features'],batch_size=512,collate_fn=collate,num_workers=0):parts.append(baseline(x.float()[:,None])[:,0].softmax(-1))
                prob=torch.cat(parts);baseline_prob[seed,arm]=prob;pseudo=pool['y'];p=prob.gather(1,pseudo[:,None])[:,0];ce=-p.log();disagreement=(prob.argmax(-1)!=pseudo).double()
                for i in range(len(p)):
                    difficulty_by_example.append({'seed':seed,'arm':arm,'pool_index':i,'assigned':labels[int(pseudo[i])],'baseline_prediction':labels[int(prob[i].argmax())],'pseudo_label_probability':float(p[i]),'cross_entropy':float(ce[i]),'disagreement':bool(disagreement[i])})
                for scope in ('pool','full_run','selected_checkpoint'):
                    w=torch.ones(len(p)) if scope=='pool' else weights[seed,arm,scope]
                    for label in ['all',*labels]:
                        mask=torch.ones(len(p),dtype=torch.bool) if label=='all' else pseudo==labels.index(label);wt=w*mask
                        difficulty.append({'seed':seed,'arm':arm,'weighting':scope,'label':label,'pool_examples':int(mask.sum()),'weighted_presentations':int(wt.sum()),'disagreement':weighted(disagreement,wt),'cross_entropy':weighted(ce,wt),'pseudo_label_probability':weighted(p,wt)})
    torch.save(baseline_prob,output/'baseline_probabilities.pt');write_csv(output/'baseline_difficulty.csv',difficulty);write_csv(output/'baseline_difficulty_per_example.csv',difficulty_by_example)
    print('Matched-count textual subsamples (without replacement)',flush=True)
    matched=[];matched_detail=[];rng=torch.Generator().manual_seed(cfg.explain.seed)
    for arm in arms:
        for count in cfg.explain.matched_counts:
            if len(examples[arm])<count:continue
            draws=[]
            for repeat in range(cfg.explain.subsamples):
                indices=torch.randperm(len(examples[arm]),generator=rng)[:count].tolist();stats=lexical(examples[arm],indices,vocabulary,original_bigrams);draws.append(stats);matched_detail.append({'arm':arm,'count':count,'repeat':repeat,**stats})
            for metric in draws[0]:
                values=torch.tensor([d[metric] for d in draws],dtype=torch.double);matched.append({'arm':arm,'count':count,'metric':metric,'mean':float(values.mean()),'subsample_sd':float(values.std()),'p05':float(torch.quantile(values,.05)),'p95':float(torch.quantile(values,.95)),'repeats':cfg.explain.subsamples})
    write_csv(output/'matched_subsamples.csv',matched_detail);write_csv(output/'matched_summary.csv',matched)
    print('Combining quality, exposure, composition and downstream results',flush=True)
    for r in summaries:
        arm=r['arm'];full=[e for e in exposures if e['arm']==arm and e['scope']=='full_run'];selected_exp=[e for e in exposures if e['arm']==arm and e['scope']=='selected_checkpoint'];d=[v for v in difficulty if v['arm']==arm and v['label']=='all' and v['weighting']=='pool']
        r.update({'augmented_draws_full_run':sum(e['augmented_tokens'] for e in full)/len(full),'gold_draws_full_run':sum(e['original_gold_tokens'] for e in full)/len(full),'mean_reuse_full_run':sum(e['mean_reuse_per_pool_index'] for e in full)/len(full),
            'mean_checkpoint_epoch':sum(e['epochs_in_scope'] for e in selected_exp)/len(selected_exp),'augmented_draws_selected':sum(e['augmented_tokens'] for e in selected_exp)/len(selected_exp),
            'baseline_disagreement':sum(v['disagreement'] for v in d)/len(d),'baseline_cross_entropy':sum(v['cross_entropy'] for v in d)/len(d),'baseline_pseudo_probability':sum(v['pseudo_label_probability'] for v in d)/len(d)})
        tv=[]
        for seed in cfg.augmentation.seeds:
            selected_labels=[x for x in label_rows if x['arm']==arm and x['seed']==seed and x['scope']=='selected_checkpoint'];tv.append(sum(abs(x['combined_proportion']-float(base_counts[labels.index(x['label'])]/base_counts.sum())) for x in selected_labels)/2)
        r['combined_label_shift_TV']=sum(tv)/len(tv)
    write_csv(output/'arm_summary.csv',summaries)
    per_seed=read_csv(source/'test'/'per_seed.csv');scores=read_csv(source/'test'/'summary.csv');joined=[]
    for r in summaries:
        for s in scores:
            if s['head']!=r['arm']:continue
            subset=[v for v in per_seed if (v['head'],v['target'],v['space'])==(r['arm'],s['target'],s['space'])];baseline=[v for v in per_seed if (v['head'],v['target'],v['space'])==('baseline',s['target'],s['space'])]
            precision=torch.tensor([float(v['precision']) for v in subset]);recall=torch.tensor([float(v['recall']) for v in subset]);dp=torch.tensor([float(v['precision'])-float(next(b['precision'] for b in baseline if b['seed']==v['seed'])) for v in subset]);dr=torch.tensor([float(v['recall'])-float(next(b['recall'] for b in baseline if b['seed']==v['seed'])) for v in subset])
            joined.append({**r,**s,'mean_precision':float(precision.mean()),'precision_seed_sd':float(precision.std()),'mean_recall':float(recall.mean()),'recall_seed_sd':float(recall.std()),'paired_precision_delta':float(dp.mean()),'paired_recall_delta':float(dr.mean())})
    write_csv(output/'downstream_joined.csv',joined)
    correlations=[];metrics=['binwise_plugin','binwise_lower','ratio_plugin','ratio_lower','old_G','acceptance','accepted','distinct_accepted','distinct_sources','distinct_focal_forms','top10_focal_share','pool_O_share','mean_piece_edit_fraction','token_types','bigram_types','novel_token_position_share','novel_bigram_position_share','mean_reuse_full_run','augmented_draws_full_run','augmented_draws_selected','mean_checkpoint_epoch','baseline_disagreement','baseline_cross_entropy','baseline_pseudo_probability','combined_label_shift_TV','binwise_mass_uncertainty_loss']
    for group,names in [('all_11',arms),('refinement_5',list(cfg.guided.arms))]:
        qrows=[r for r in summaries if r['arm'] in names]
        for target in ('binwise_plugin','binwise_lower','ratio_plugin','ratio_lower','old_G'):
            y=torch.tensor([r[target] for r in qrows],dtype=torch.double)
            for metric in ('accepted','binwise_mass_uncertainty_loss','distinct_sources','distinct_focal_forms'):
                x=torch.tensor([r[metric] for r in qrows],dtype=torch.double)
                for kind in ('Pearson','Spearman'):
                    value,reason=correlation(x,y,kind=='Spearman');correlations.append({'group':group,'outcome':target,'metric':metric,'kind':kind,'value':value,'undefined_reason':reason,'arms':len(qrows)})
        for dataset,space in sorted({(r['target'],r['space']) for r in joined}):
            subset=[r for r in joined if r['arm'] in names and (r['target'],r['space'])==(dataset,space)];y=torch.tensor([float(r['paired_delta_vs_baseline']) for r in subset],dtype=torch.double)
            for metric in metrics:
                x=torch.tensor([r[metric] for r in subset],dtype=torch.double)
                for kind in ('Pearson','Spearman'):
                    value,reason=correlation(x,y,kind=='Spearman');correlations.append({'group':group,'outcome':dataset+'/'+space,'metric':metric,'kind':kind,'value':value,'undefined_reason':reason,'arms':len(subset)})
    write_csv(output/'correlations.csv',correlations)
    save_json(output/'method.json',{'sampling':'IID remains an explicit unverified working assumption; no provenance re-audit. Quality scores retain original definitions/allocations; none is measured generated-label accuracy.',
        'exposure':'Exactly reconstruct schedule and Steps seeded indices for each seed, checking full-run counts against saved budgets. Both ten-epoch run and selected-head checkpoint prefixes. Count distinct generated token-input plus focal-span keys, retaining duplicate pool entries in loss. Counts are not expectations.',
        'loss':'Each head uses mean CE over all labelled tokens of its batch, no separate augmentation coefficient, class reweighting, quality weighting or generator lambda in student loss. Head losses sum across independent parameter blocks; clipping per head. Pool-size-proportional mixtures with deterministic rounded quotas. Per-example loss coefficient is 1/batch-labelled-token-count; effective augmented coefficient mass logged separately.',
        'text':'Use saved token IDs and original word alignment. Edit fraction excludes focal, special and padding positions; original-word slots are aligned by original WordPiece slots, not newly validated linguistic words. Lexical unigrams and contiguous bigrams are tokenizer-position statistics; bigrams never bridge focal/special holes. Novelty relative to all ordinary WordPieces/bigrams in 2400 original training sentences, not the full scorer corpus. No encoder geometry, fluency scoring or correctness measurements.',
        'matched':'100 repeated simple random subsamples without replacement per feasible count/arm. Pool entries sampled; duplicates retained and counted. Counts 25/100/200/500/1000, no upsampling of small arms. Subsample SD describes variability, not independent experiment uncertainty.',
        'difficulty':'Existing seed-matched baseline checkpoint heads only; CPU float32 inference on cached accepted frozen-encoder features, no encoder/scorer inference. Overall and assigned-label metrics, pool and exact exposure weighting. Pseudo-label CE/disagreement are student difficulty, not gold accuracy.',
        'evaluation':'Existing native/shared label spaces and span metrics, per-seed precision/recall reused. No test evaluation rerun, no new model or data generation.',
        'unavailable':'Actual unknown generated gold labels; replayed stochastic training gradients and dropout masks are unavailable and not needed for exact scheduled presentation counts. No causal attribution is possible across eleven coupled configurations.'})
    save_json(output/'runtime.json',{'seconds':time.monotonic()-started});tracking.finish({'arms':len(arms),'exposure_rows':len(exposures),'matched_subsamples':len(matched_detail)})
    report(cfg);print(output,flush=True);return 0


if __name__=='__main__':run('lion.guided_refinement',sys.argv[1:],main_module='explain',config_dir=str(ROOT/'conf'),config_name='population_explain')
