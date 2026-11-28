"""Reuse raw test features, baseline/MENTION predictions and established metrics."""
import math
from collections import defaultdict
import torch
from tagger.ensemble import IndependentReadouts
from ..diagnostics.common import absolute,write_csv,read_csv
from ..partition.train import predict
from ..independent_judge.run import evaluation_from_logits
from ..matched_regime.evaluation import paired,rowkeys
from ..label_change.natural_pairs import pair_counts
from ..variance_decomposition import run as previous
from ..language_checks.run import metric_row,summarize
from .generate import root,load


def test(cfg,out):
    cache=load(absolute(cfg.grid_pilot.evaluation_cache));arms=list(cfg.paradigm_swaps.arms)
    for seed in cfg.augmentation.seeds:
        saved=load(root(cfg)/'train'/f'seed{seed}/model.pt');core=IndependentReadouts(cfg.tagger,768,9,len(arms)).to(cfg.runtime.device);core.load_state_dict(saved['state']);core.eval()
        for split,item in cache['sets'].items():
            if split=='validation':continue
            logits=predict(core,item['data'],cfg);old=load(absolute(cfg.matched.results)/'test'/f'seed{seed}_{split}.pt');ix=[list(old['heads']).index(a) for a in ['baseline','MENTION']];combined=torch.cat((old['logits'][:,ix],logits),1);torch.save({'logits':combined,'gold':old['gold'],'sentence':old['sentence'],'heads':['baseline','MENTION']+arms},out/f'seed{seed}_{split}.pt')
            print(f'Test seed{seed} {split}',flush=True)
    evaluation_from_logits(cfg,out,out);paired(out);return {'new_heads':len(arms),'reused_heads':['baseline','MENTION']}


def slices(cfg,out):
    ref=load(absolute('results/language_checks/prepare/reference.pt'));rows=[]
    for seed in cfg.augmentation.seeds:
        for split,d in ref['datasets'].items():
            saved=load(root(cfg)/'test'/f'seed{seed}_{split}.pt');pred=previous.predict_tokens(saved['logits'],None if d['space']=='native' else ref['mapping']);gold=d['gold'];freq=d['frequency'];heads=saved['heads']
            for definition,key in [('raw','raw_category'),('labels_at_least_10pct','robust_category')]:
                group=torch.where(freq==0,0,torch.where(freq<=7,1,torch.where(d[key]==2,3,2)));names=['unseen','rare_1_7','frequent_unambiguous','frequent_ambiguous'];c=previous.grouped(pred,gold,group,4)
                for h,head in enumerate(heads):
                    for g,name in enumerate(names):rows.append({'seed':seed,'arm':head,'dataset':d['dataset'],'definition':definition,'token_type':name,'small_slice':int(c['n'][g])<50,'status':'computed; baseline/MENTION predictions reused',**metric_row(c,h,g,0)})
    write_csv(out/'per_seed.csv',rows);summary=summarize(rows,['arm','dataset','definition','token_type'],['accuracy','token_micro_f1','accuracy_gain','token_f1_gain']);write_csv(out/'summary.csv',summary);return {'rows':len(rows)}


def pairs(cfg,out):
    reference=load(absolute(cfg.natural_pairs.results)/'predictions.pt');cache=load(absolute(cfg.grid_pilot.evaluation_cache))['sets']['id']['data'];keys=rowkeys(cache['sentence'],cfg.encoder.max_length);wanted=rowkeys(reference['sentence'],cfg.encoder.max_length);indices=torch.searchsorted(keys,wanted);lookup={};word=torch.tensor([lookup.setdefault(w,len(lookup)) for w in reference['forms']]);gold=reference['gold'];sentence=reference['sentence'];K=9;rows=[]
    for seed in cfg.augmentation.seeds:
        saved=load(root(cfg)/'test'/f'seed{seed}_id.pt');pred=saved['logits'][indices].argmax(-1)
        for group,mask in [('all_labels_including_O',torch.ones(len(gold),dtype=torch.bool)),('entity_vs_entity',gold!=0)]:
            for h,a in enumerate(saved['heads']):rows.append({'seed':seed,'arm':a,'group':group,**pair_counts(word[mask],gold[mask],pred[mask,h],sentence[mask],K,len(lookup)),'status':'computed using reused exact natural-pair cohort; baseline/MENTION logits reused'})
    summary=[]
    for a,group in sorted(set((r['arm'],r['group']) for r in rows)):
        selected=[r for r in rows if (r['arm'],r['group'])==(a,group)];x=torch.tensor([r['word_macro_pair_accuracy'] for r in selected],dtype=torch.double);delta=torch.tensor([r['word_macro_pair_accuracy']-next(b['word_macro_pair_accuracy'] for b in rows if (b['seed'],b['group'],b['arm'])==(r['seed'],group,'baseline')) for r in selected],dtype=torch.double);summary.append({'arm':a,'group':group,'mean':float(x.mean()),'seed_sd':float(x.std()),'paired_mean':float(delta.mean()),'paired_se':float(delta.std()/math.sqrt(len(delta))),'ambiguous_word_types':selected[0]['ambiguous_word_types'],'eligible_pairs':selected[0]['eligible_unordered_pairs'],'status':'computed from reused cohort and cached logits'})
    write_csv(out/'per_seed.csv',rows);write_csv(out/'summary.csv',summary);return {'rows':len(rows)}


CONTRASTS=[('B vs A','PARADIGM_SPAN','MENTION'),('C vs B','PARADIGM_SPAN_STRICT','PARADIGM_SPAN'),('D vs E','PARADIGM_TOKEN','LWTR_TOKEN'),('D vs A','PARADIGM_TOKEN','MENTION')]
def report(cfg,out):
    spans=read_csv(root(cfg)/'test/seqeval_per_seed.csv');slices_rows=read_csv(root(cfg)/'slices/per_seed.csv');pools=read_csv(root(cfg)/'stats/pool_statistics.csv');pairs_rows=read_csv(root(cfg)/'pairs/per_seed.csv');contrasts=[];summary=[]
    def add(contrast,a,b,target,space,metric,definition,kind,source,akey,bkey,value):
        deltas=[]
        for seed in cfg.augmentation.seeds:
            av=next(r for r in source if r['seed']==str(seed) and akey(r,a));bv=next(r for r in source if r['seed']==str(seed) and bkey(r,b));delta=float(av[value])-float(bv[value]);deltas.append(delta);contrasts.append({'contrast':contrast,'arm_a':a,'arm_b':b,'dataset':target,'space':space,'metric':metric,'definition':definition,'token_type':kind,'seed':seed,'difference':delta,'status':'computed seed-paired; A=MENTION reused unchanged'})
        d=torch.tensor(deltas,dtype=torch.double);summary.append({'contrast':contrast,'arm_a':a,'arm_b':b,'dataset':target,'space':space,'metric':metric,'definition':definition,'token_type':kind,'mean_difference':float(d.mean()),'seed_SE':float(d.std()/math.sqrt(len(d))),'status':'computed from fixed checkpoints; five paired seeds'})
    for contrast,a,b in CONTRASTS:
        for target,space in sorted(set((r['target'],r['space']) for r in spans)):
            key=lambda r,arm:r['head']==arm and r['target']==target and r['space']==space
            add(contrast,a,b,target,space,'span_f1','none','all',spans,key,key,'span_f1')
        for target,definition,kind in sorted(set((r['dataset'],r['definition'],r['token_type']) for r in slices_rows)):
            key=lambda r,arm:r['arm']==arm and r['dataset']==target and r['definition']==definition and r['token_type']==kind
            for metric in ('accuracy','token_micro_f1'):add(contrast,a,b,target,'native' if target=='fewnerd' else 'shared',metric,definition,kind,slices_rows,key,key,metric)
        for group in sorted(set(r['group'] for r in pairs_rows)):
            key=lambda r,arm:r['arm']==arm and r['group']==group
            add(contrast,a,b,'fewnerd','native','word_macro_pair_accuracy','none',group,pairs_rows,key,key,'word_macro_pair_accuracy')
    write_csv(out/'contrasts_per_seed.csv',contrasts);write_csv(out/'contrasts_summary.csv',summary)
    lines=['# Paradigm swaps','','Four prespecified contrasts; all differences are left minus right, percentage points, mean ± paired seed SE (five seeds). Test sets and checkpoints are fixed. Values below computed from new heads and reused unchanged baseline/MENTION logits.','','| Contrast | Few-NERD | CoNLL | WNUT | WikiANN |','|---|---:|---:|---:|---:|']
    for contrast,a,b in CONTRASTS:
        values=[next(r for r in summary if (r['contrast'],r['dataset'],r['space'],r['metric'])==(contrast,t,s,'span_f1')) for t,s in [('fewnerd','native'),('conll2003','shared_ner'),('wnut','shared_ner'),('wikiann','shared_ner')]];lines.append('| '+contrast+f' ({a} − {b}) | '+' | '.join(f"{100*r['mean_difference']:+.3f} ± {100*r['seed_SE']:.3f}" for r in values)+' |')
    lines+=['','Every negative cell quantifies where the first arm underperforms its comparator. These are conditional model-performance comparisons, not validation of the assumed gold-label preservation.','','| Pool | Augmented labels | New word types | Label-risk proxy | MLM mean piece PLL change |','|---|---:|---:|---:|---:|']
    for r in pools:lines.append(f"| {r['arm']} | {r['labelled_augmented_tokens']} | {r['new_word_types']} | {100*float(r['label_risk_share']):.2f}% | {float(r['paired_PLL_change']):+.3f} |")
    lines+=['','All pool statistics above are computed on the exact selected 1932 labels. The risk proxy counts seen focal words whose training-majority native label differs from the assigned label; unseen words are counted separately. Positive PLL change means higher MLM pseudo-log-likelihood. Neither proxy is used to select checkpoints or filter pools (apart from the explicitly specified STRICT lexicon rule).','','## Where each contrast lands','','| Contrast | Dataset | Largest positive token-F1 difference, raw types | Most negative token-F1 difference, raw types |','|---|---|---|---|']
    descriptions=[]
    for contrast,a,b in CONTRASTS:
        for target in ['fewnerd','conll2003','wnut','wikiann']:
            rs=[r for r in summary if (r['contrast'],r['dataset'],r['metric'],r['definition'])==(contrast,target,'token_micro_f1','raw')];best=max(rs,key=lambda r:r['mean_difference']);worst=min(rs,key=lambda r:r['mean_difference']);lines.append(f"| {contrast} | {target} | {best['token_type']}: {100*best['mean_difference']:+.3f} ± {100*best['seed_SE']:.3f} | {worst['token_type']}: {100*worst['mean_difference']:+.3f} ± {100*worst['seed_SE']:.3f} |")
        pa=next(r for r in pools if r['arm']==a);pb=next(r for r in pools if r['arm']==b)
        descriptions.append(f"{contrast}: new-type counts {pa['new_word_types']} vs {pb['new_word_types']}; risk {100*float(pa['label_risk_share']):.2f}% vs {100*float(pb['label_risk_share']):.2f}%. These pool descriptors accompany the above gains; four fixed contrasts cannot establish that novelty or risk caused them.")
    lines+=['']+descriptions
    lines+=['','## Reuse and deviations','','- Same 2400 originals and all original labels; four new heads only. Baseline and MENTION test logits/results reused unchanged. Same five seeds, initial MLP weights, shared dropout convention, original stream, optimizer steps, total labelled-token budgets and raw native-dev token macro-F1 checkpoint selection. Existing Steps and schedule code are used directly. Exact per-seed counts in train/budgets.csv; historical baseline/MENTION rows in reused_budgets.csv.','- Span anchors recovered from the historical MENTION pool, including repeated anchor draws. Historical MENTION used source-correctness eligibility; this is inherited only by the required same-mention B/C anchors. No scorer is loaded or called, and D/E have no source filter. B/C initially traverse identical recovered anchors; rejected proposals advance, and additional anchors are drawn from that list until the size is reached. Thus rejection can change the final accepted anchor composition.','- The old MLM routine was parallel, not sequential. Added minimal exact left-to-right conditional fill, one non-continuation/special-free whole WordPiece per word slot; future slots masked, prior sampled slots visible. Temperature 1, full restricted vocabulary, no top-k/p. This limits MLM fillers to single-piece words; entries must round-trip as one WordPiece (so unused/normalization-inconsistent vocabulary strings cannot alter the sampled input on re-encoding). Punctuation is retained. This is slightly stricter than the prior counting analysis’s non-continuation/non-special vocabulary. LWTR donors can have multiple pieces and come from same-label token occurrences.','- Native training uses existing coarse IO labels, merging consecutive same-type words; BIO is re-derived by the established evaluation code. Augmentation labels cover replaced positions only; overshoot is sampled down to exactly 1932 focal labels. Some nonselected replaced positions can remain context, as in MENTION. No new mention list is learned, but span arms necessarily use gold source-mention boundaries.','- Pools are generated once with a fixed seed and reused across all training seeds. Identity draws are rejected as specified for paradigm arms; LWTR also rejects identity for a matched actual-swap comparison. PARADIGM_TOKEN proposes occurrence-uniform anchors; identity rejections draw fresh anchors until 1932 accepted swaps. LWTR uses that exact accepted anchor set and redraws identity donors there. Thus the shared final anchor set is conditioned on MLM swap acceptance, rather than being uniform after rejection. An initial implementation retried identity MLM fills at the same anchor and stalled; it was interrupted and generation rerun with this fresh-anchor rule.','- PLL is full-vocabulary leave-one-WordPiece-out log-probability, mean over ordinary pieces then over labelled pool items; identical input sequences are evaluated once. Source/augmented PLLs are paired. It is a naturalness proxy, not an acceptance filter.','- Test encoding is entirely reused. Token transfer predictions sum native softmax probabilities into shared classes before argmax; span metrics use existing native-argmax/BIO/shared mapping. Raw and ≥10% ambiguity slices use the corresponding native/shared training word counts. Slices below 50 tokens are flagged. Pair accuracy reuses the held-out distinct-sentence, different-label cohort and existing exact word-macro code.','- Every performance/contrast/PLL/pool statistic is computed; baseline and MENTION predictions, original/test/validation features, training conventions and natural-pair cohort are reused. No external notes modified. No missing required input.','','Files: stats/pool_statistics.csv; stats/label_distribution.csv; stats/examples.csv (30 seeded labelled examples per arm); test/paired_summary.csv; slices/per_seed.csv and summary.csv; pairs/per_seed.csv and summary.csv; report/contrasts_per_seed.csv and contrasts_summary.csv; train/budgets.csv; config.yaml.']
    text='\n'.join(lines)+'\n';(out/'RESULTS.md').write_text(text);(root(cfg)/'RESULTS.md').write_text(text);return {'contrasts':len(summary)}
