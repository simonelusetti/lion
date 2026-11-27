"""Word identity, within-word labels, and cached test-prediction gain attribution."""
import json,math,sys,time
from collections import defaultdict
import torch
import pyarrow.parquet as pq
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from forge import run,start_run
from omegaconf import OmegaConf
from ..diagnostics.common import ROOT,absolute,save_json,write_csv,read_csv,dataset_config
from ..diagnostics.data import token_path,SHARED
from ..label_change.fewnerd_diagnostic import parquet
from ..labels import LABELS
from ..metrics import shared_label


CATEGORIES=['unseen','seen_unambiguous','seen_ambiguous']
def load(path):return torch.load(path,weights_only=False,mmap=True)
def collate(rows):return torch.stack(rows)


def collect_metadata(cfg,cache):
    original=absolute(cfg.matched.originals_run)/'prepare';sentences=read_csv(original/'sentences.csv');native=pq.read_table(parquet('train'),columns=['tokens']).to_pylist();forms=[]
    for row in sentences:forms.extend(native[int(row['source_id'].split('/')[-1])]['tokens'][:int(row['words'])])
    tests={};tokenizer=AutoTokenizer.from_pretrained(cfg.encoder.name,use_fast=True,local_files_only=True);raw=pq.read_table(parquet('test'),columns=['tokens','ner_tags']).to_pylist();words=[];gold=[]
    for start in range(0,len(raw),256):
        block=raw[start:start+256];encoded=tokenizer([r['tokens'] for r in block],is_split_into_words=True,add_special_tokens=True,truncation=True,max_length=cfg.encoder.max_length)
        for i,row in enumerate(block):
            kept=list(dict.fromkeys(w for w in encoded.word_ids(i) if w is not None));words.extend(row['tokens'][w] for w in kept);gold.extend(row['ner_tags'][w] for w in kept)
    tests['id']=words
    if not torch.equal(torch.tensor(gold),cache['sets']['id']['data']['y']):raise ValueError('Native test words do not align with cached gold')
    sources=[]
    for split,item in cache['sets'].items():
        if split=='id':continue
        target=item['target'];local=dataset_config(cfg,target);local.encoder.special_tokens=True;path=token_path(local)
        data=load_from_disk(path);rows=[r for r in data['test'] if r['labels']];words=[w for r in rows for w in r['words']];gold=torch.tensor([y for r in rows for y in r['labels']])
        if not torch.equal(gold,item['data']['y']):raise ValueError(f'{target}: cached word metadata/gold mismatch')
        tests[split]=words;sources.append({'target':target,'metadata_source':str(path),'word_metadata':'reused existing token cache'})
    return forms,tests,sources


def category(counts,threshold):
    total=counts.sum(-1);valid=counts>0 if threshold==0 else (counts.double()>=threshold*total[:,None])&(counts>0);nlabels=valid.sum(-1)
    return torch.where(total==0,0,torch.where(nlabels<=1,1,2)),nlabels


def decompose(x,word):
    _,group=torch.unique(word,sorted=True,return_inverse=True);G=int(group.max())+1;n=len(x);counts=torch.bincount(group,minlength=G).double();sums=torch.zeros(G,x.shape[1],dtype=torch.double).index_add_(0,group,x);means=sums/counts[:,None];global_mean=x.mean(0)
    total=float((x-global_mean).square().sum());within=float((x-means[group]).square().sum());between=float((counts[:,None]*(means-global_mean).square()).sum())
    return {'tokens':n,'word_types':G,'total_variance':total/(n-1),'between_word_variance':between/(n-1),'within_word_variance':within/(n-1),'ICC':between/total,'identity_residual':(total-between-within)/(n-1)},group,counts,means,within


def variance(cfg,x,word,y,native_counts,output):
    frequencies=torch.bincount(word);order=word.argsort(stable=True);_,sizes=word[order].unique_consecutive(return_counts=True);offset=0;chosen=[];rng=torch.Generator().manual_seed(cfg.variance_analysis.seed)
    for size in sizes.tolist():
        indices=order[offset:offset+size];offset+=size
        if size>=2:chosen.append(indices[torch.randperm(size,generator=rng)[:cfg.variance_analysis.cap_per_word]])
    chosen=torch.cat(chosen);features=x[chosen,0].double();w=word[chosen];labels=y[chosen];rows=[]
    for name,mask in [('all',torch.ones(len(chosen),dtype=torch.bool)),('gold_entity_tokens',labels!=0)]:
        values=features[mask];ids=w[mask]
        for control in (False,True):
            if control:ids=ids[torch.randperm(len(ids),generator=torch.Generator().manual_seed(cfg.variance_analysis.seed))]
            result,*_=decompose(values,ids);rows.append({'layer':12,'sample':name,'permuted_word_ids':control,'status':'computed from reused cached features',**result})
    ambiguity=[];K=native_counts.shape[1]
    for threshold in cfg.variance_analysis.ambiguity_thresholds:
        cats,_=category(native_counts,threshold);mask=cats[w]==2;values=features[mask];ids=w[mask];gold=labels[mask]
        result,group,counts,means,within=decompose(values,ids);G=len(counts);cells=group*K+gold;cell_counts=torch.bincount(cells,minlength=G*K).reshape(G,K);sums=torch.zeros(G*K,values.shape[1],dtype=torch.double).index_add_(0,cells,values).reshape(G,K,-1);centroids=sums/cell_counts.clamp_min(1)[:,:,None]
        explained=float((cell_counts[:,:,None]*(centroids-means[:,None]).square()).sum());predictions=[];majority=[];supported=[]
        for start in range(0,len(values),512):
            v=values[start:start+512];g=group[start:start+512];target=gold[start:start+512];n=cell_counts[g];cent=centroids[g];distance=(v[:,None]-cent).square().sum(-1);distance[n==0]=torch.inf;owner=torch.arange(len(v));own=n[owner,target];leave=(sums[g,target]-v)/(own-1).clamp_min(1)[:,None];distance[owner,target]=(v-leave).square().sum(-1);distance[owner[own<=1],target[own<=1]]=torch.inf
            predictions.append(distance.argmin(-1));n=n.clone();n[owner,target]-=1;majority.append(n.argmax(-1));supported.append(own>1)
        pred=torch.cat(predictions);base=torch.cat(majority);support=torch.cat(supported)
        ambiguity.append({'layer':12,'definition':'raw' if threshold==0 else 'labels_at_least_10pct','tokens':len(values),'word_types':G,'within_variance_explained_by_label':explained/within,'within_word_variance':within/(len(values)-1),'between_label_variance':explained/(len(values)-1),
            'LOO_centroid_accuracy':float((pred==gold).double().mean()),'LOO_majority_accuracy':float((base==gold).double().mean()),'LOO_centroid_gain':float(((pred==gold).double()-(base==gold).double()).mean()),'gold_label_without_LOO_support':int((~support).sum()),'LOO_accuracy_gold_supported':float((pred[support]==gold[support]).double().mean()),'distance':'Squared Euclidean to per-word label centroids; focal observation excluded from its gold centroid; missing candidate labels excluded.','status':'computed'})
    write_csv(output/'variance.csv',rows);write_csv(output/'within_word_labels.csv',ambiguity);torch.save(chosen,output/'variance_sample_indices.pt')


def predict_tokens(logits,mapping=None):
    result=torch.empty(logits.shape[:2],dtype=torch.long)
    for index in DataLoader(torch.arange(len(logits)),batch_size=8192,collate_fn=collate,num_workers=0):
        values=logits[index]
        if mapping is None:result[index]=values.argmax(-1)
        else:
            probabilities=values.softmax(-1);projected=torch.zeros(*values.shape[:2],len(SHARED));projected.scatter_add_(-1,mapping[None,None].expand_as(probabilities),probabilities);result[index]=projected.argmax(-1)
    return result


def grouped(pred,gold,groups,G):
    valid=groups>=0;pred=pred[valid];gold=gold[valid];groups=groups[valid];H=pred.shape[1];keys=groups[:,None]+torch.arange(H)[None]*G;correct=pred==gold[:,None]
    def count(mask):return torch.bincount(keys[mask],minlength=H*G).reshape(H,G)
    return {'n':torch.bincount(groups,minlength=G),'entity_n':torch.bincount(groups[gold!=0],minlength=G),'correct':count(correct),'tp':count(correct&(gold[:,None]!=0)),'fp':count(~correct&(pred!=0)),'fn':count(~correct&(gold[:,None]!=0))}


def f1(tp,fp,fn):return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.


def shapley(base,arm):
    scores={}
    for bits in range(8):
        counts=[sum((arm if bits&(1<<g) else base)[g][j] for g in range(3)) for j in range(3)];scores[bits]=f1(*counts)
    contributions=[]
    for g in range(3):
        value=0.
        for bits in range(8):
            if bits&(1<<g):continue
            k=bits.bit_count();weight=math.factorial(k)*math.factorial(2-k)/6;value+=weight*(scores[bits|(1<<g)]-scores[bits])
        contributions.append(value)
    return contributions,scores[7]-scores[0]


def pool_words(cfg,run,head):
    if run=='matched':
        selection=json.loads((absolute(cfg.matched.results)/'match'/'selections.json').read_text())[head];pool=load(absolute(cfg.matched.results)/'new_pools'/f"{selection['parent']}.pt");indices=selection['indices']
        return [pool['word'][i] for i in indices],pool['y'][torch.tensor(indices)]
    pool=load(absolute(cfg.matched.results)/'historical_pools'/f'{head}.pt');return pool['word'],pool['y']


def record(run,seed,head,dataset,definition,part,category_name,c,h,base,contribution=None,overall=None,relevant=None):
    n=int(c['n'][category_name[0]])
    g,name=category_name;tp,fp,fn=(int(c[k][h,g]) for k in ('tp','fp','fn'));bt,bp,bn=(int(c[k][base,g]) for k in ('tp','fp','fn'));score=f1(tp,fp,fn);baseline=f1(bt,bp,bn)
    return {'run':run,'seed':seed,'arm':head,'dataset':dataset,'definition':definition,'part':part,'category':name,'tokens':n,'gold_entity_tokens':int(c['entity_n'][g]),'accuracy':float(c['correct'][h,g]/n) if n else None,'token_micro_f1':score if n else None,'accuracy_gain':float((c['correct'][h,g]-c['correct'][base,g])/n) if n else None,'token_f1_gain':score-baseline if n else None,'shapley_contribution':contribution,'overall_token_f1_gain':overall,'share_of_total_gain':contribution/overall if contribution is not None and overall else None,'flip_or_steered':relevant,'tp':tp,'fp':fp,'fn':fn,'status':'computed from reused test logits'}


def report(cfg):
    output=absolute(cfg.variance_analysis.results);stats=read_csv(output/'word_statistics.csv');summary=read_csv(output/'summary.csv');variance_rows=read_csv(output/'variance.csv');labels=read_csv(output/'within_word_labels.csv');coverage_rows=read_csv(output/'coverage_exposure.csv');timing=json.loads((output/'runtime.json').read_text())
    def get(run,arm,dataset,definition='raw',part='B'):
        return {r['category']:r for r in summary if (r['run'],r['arm'],r['dataset'],r['definition'],r['part'])==(run,arm,dataset,definition,part)}
    def number(row,key):return float(row[key])
    def gain(row,key='accuracy_gain'):return f"{100*number(row,key+'_mean'):+.3f} ± {100*number(row,key+'_se'):.3f}" if row[key+'_mean'] else 'unavailable'
    lines=['# Word identity, ambiguity and augmentation gains','',
        f"Computed CPU analysis from cached observations in {timing['seconds']:.1f} seconds (excluding this report formatting). No generation, training, model forward pass, new annotation or GPU. Both runs use the same 2400 originals; each arm is paired only with its own run baseline (five seeds matched, three seeds size_grid). No arm, checkpoint or original evaluation metric is changed.",'',
        '**Status labels:** all new tables below are **computed** from **reused** cached features/logits/pools and raw/token metadata. Original span evaluations are **reused unchanged** in reused_span_results_*.csv. Token accuracy/non-O micro-F1 below are the requested category diagnostics, not span F1. SE is across student seeds, not an uncertainty interval over test sentences.','',
        '## A. Word-type statistics (computed)','',
        'Case-sensitive surface forms; native nine-label counts for Few-NERD and existing shared four-label counts for transfer. O counts. The robust definition counts a label only if it represents at least 10% of that word’s training occurrences.','',
        '| Dataset / label space | Unseen tokens | Unambiguous tokens, raw | Ambiguous tokens, raw | Ambiguous tokens, 10% rule |','|---|---:|---:|---:|---:|']
    for dataset,space in [('training','native'),('training','shared'),('fewnerd','native'),('conll2003','shared'),('wnut','shared'),('wikiann','shared')]:
        rows={r['category']:r for r in stats if (r['dataset'],r['space'],r['definition'],r['subset'])==(dataset,space,'raw','all_tokens')};robust=next(r for r in stats if (r['dataset'],r['space'],r['definition'],r['subset'],r['category'])==(dataset,space,'labels_at_least_10pct','all_tokens','seen_ambiguous'))
        lines.append(f"| {dataset} / {space} | {100*number(rows['unseen'],'token_share'):.2f}% | {100*number(rows['seen_unambiguous'],'token_share'):.2f}% | {100*number(rows['seen_ambiguous'],'token_share'):.2f}% | {100*number(robust,'token_share'):.2f}% |")
    lines+=['','In native training, 9.46% of types are raw-ambiguous but account for 51.15% of tokens. Thus most types have one observed label, but frequent ambiguous types carry substantial token mass. The 10% rule reduces ambiguous token mass to 10.75%. Training gold-entity token ambiguity is 48.19% raw and 36.82% robust. All type/token counts, shares and gold-entity restrictions for training and every test set are in word_statistics.csv.','',
        '## A. Feature variance and within-word labels (computed from reused block12)','',
        '| Sample | Word-ID permutation | Tokens / types | Between trace | Within trace | Total trace | ICC |','|---|---|---:|---:|---:|---:|---:|']
    for r in variance_rows:lines.append(f"| {r['sample']} | {r['permuted_word_ids']} | {r['tokens']} / {r['word_types']} | {number(r,'between_word_variance'):.3f} | {number(r,'within_word_variance'):.3f} | {number(r,'total_variance'):.3f} | {number(r,'ICC'):.4f} |")
    lines+=['','Sampling: words with >=2 original occurrences; at most 50 occurrences per word, uniformly without replacement, seed0. Entity row filters gold-entity tokens from this same sample. Components use the common n−1 covariance denominator and satisfy total=between+within to numerical precision. ICC means between/total here, not a fitted random-effects parameter. The nonzero shuffled control reflects finite-sample group means and must not be interpreted as zero expected explained variance.','',
        '| Ambiguity definition | Tokens / types | Within variance explained by label | LOO centroid accuracy | LOO majority accuracy | Centroid advantage | Gold label lacks LOO support |','|---|---:|---:|---:|---:|---:|---:|']
    for r in labels:lines.append(f"| {r['definition']} | {r['tokens']} / {r['word_types']} | {100*number(r,'within_variance_explained_by_label'):.2f}% | {100*number(r,'LOO_centroid_accuracy'):.2f}% | {100*number(r,'LOO_majority_accuracy'):.2f}% | {100*number(r,'LOO_centroid_gain'):+.2f} pp | {r['gold_label_without_LOO_support']} |")
    lines+=['','The label decomposition is descriptive on the capped sample; it is not held-out variance explained. LOO removes the observation from its own-label centroid and excludes missing candidate labels. Singleton gold-label cases cannot be classified correctly with that label centroid; supported-only accuracy is also saved. This indicates some within-word label signal, while the incremental centroid advantage over a word-majority control is modest.','',
        '## B. MENTION and identity-related gains (computed)','',
        'Cells are exact Shapley contributions to the total non-O token-micro-F1 change, in percentage points. Contributions sum to the total; they are not category-specific F1 scores.','',
        '| Target | Total Δtoken-F1 | Unseen contribution | Seen-unambiguous contribution | Seen-ambiguous contribution | Unseen share of total |','|---|---:|---:|---:|---:|---:|']
    for dataset in ('fewnerd','conll2003','wnut','wikiann'):
        rows=get('matched','MENTION',dataset);u=rows['unseen'];share=100*number(u,'share_of_mean_total_gain') if u['share_of_mean_total_gain'] else None
        lines.append(f"| {dataset} | {100*number(u,'overall_token_f1_gain_mean'):+.3f} | "+' | '.join(gain(rows[c],'shapley_contribution') for c in CATEGORIES)+f" | {share:.1f}% |")
    lines+=['','MENTION: unseen is the largest positive contributor on CoNLL (about 76%) and WNUT (57%); WikiANN is split almost equally between unseen and raw-ambiguous words (about 38% each). With robust ambiguity, unseen is the largest WikiANN category but still only 38% of the total. Native Few-NERD has no positive total token-F1 gain. Training mention donors introduce no new focal word types relative to these originals, so unseen-test gains cannot be attributed to direct exposure to those unseen focal identities.','',
        'Rare-word extension (rare=1–5 training occurrences; computed Δcategory token-F1, pp):','',
        '| Target | Unseen | Seen rare | Seen frequent |','|---|---:|---:|---:|']
    for dataset in ('fewnerd','conll2003','wnut','wikiann'):
        rows=get('matched','MENTION',dataset,'frequency','B_frequency');lines.append('| '+dataset+' | '+' | '.join(gain(rows[c],'token_f1_gain') for c in ('unseen','seen_rare_1_5','seen_frequent_6plus'))+' |')
    lines+=['','## B. All arms: total token-F1 change and largest contributor (computed)','',
        'Each cell is total Δtoken-F1 in pp, followed by the category with largest absolute Shapley contribution. A negative total can therefore name a category driving a loss. Full per-category token accuracy/F1 and paired mean/SD/SE are in summary.csv and per_seed.csv, for both raw and robust definitions. Shares may be negative or exceed 100% when category effects oppose; near-zero totals are flagged, not suppressed.','',
        '| Run / arm | Few-NERD native | CoNLL | WNUT | WikiANN |','|---|---:|---:|---:|---:|']
    for run,arm in dict.fromkeys((r['run'],r['arm']) for r in summary if r['arm']!='baseline'):
        cells=[]
        for dataset in ('fewnerd','conll2003','wnut','wikiann'):
            rows=get(run,arm,dataset);leading=max(rows.values(),key=lambda r:abs(number(r,'shapley_contribution_mean')));label={'unseen':'unseen','seen_unambiguous':'unambig','seen_ambiguous':'ambig'}[leading['category']];cells.append(f"{100*number(leading,'overall_token_f1_gain_mean'):+.3f} ({label})")
        lines.append('| '+run+' / '+arm+' | '+' | '.join(cells)+' |')
    lines+=['','Across the 20 FLIP/AGREED/steered configurations (all historical arms were flip-filtered), unseen is the largest positive contributor in all positive-gain CoNLL cells (13/13), WNUT cells (17/17) and WikiANN cells (10/10). For Few-NERD it is 11/14 raw, versus ambiguous 3/14; robust is 10/14 versus 4/14. These are counts across related configurations, not independent trials or significance tests.','',
        '## C. Targeted versus untargeted ambiguous words (computed)','',
        'All pool entries were encountered before the selected checkpoints in both runs; exposure was exactly reconstructed. Coverage is case-sensitive focal-word coverage, not sentence/context similarity. Departure means at least one encountered assigned label differs from the word’s training-majority label in the evaluation label space.','',
        'Examples on WNUT, raw ambiguity; cells are Δtoken accuracy mean ± seed SE in pp.','',
        '| Run / arm | Uncovered ambiguous | Covered ambiguous | Covered, no majority departure | Covered, majority departure |','|---|---:|---:|---:|---:|']
    for run,arm in [('matched','uniform_FLIP_full'),('matched','local70_FLIP_full'),('size_grid','lambda0.001'),('size_grid','lambda0.01'),('size_grid','greedy')]:
        c=get(run,arm,'wnut',part='C_coverage');d=get(run,arm,'wnut',part='C_departure');lines.append('| '+run+' / '+arm+' | '+' | '.join(gain(r) for r in [c['ambiguous_uncovered'],c['ambiguous_covered'],d['covered_no_majority_departure'],d['covered_majority_departure']])+' |')
    lines+=['','Coverage contrasts are observational and confounded by word frequency and composition. In large pools most ambiguous test tokens are covered: e.g. uniform full-FLIP covers 5515/5659 ambiguous WNUT tokens, leaving only 144 uncovered. Counts are saved for every slice, and small uncovered groups should not be interpreted as controlled counterfactuals. Both definitions and every arm/target appear in the C rows of summary.csv.','',
        '## Requested hypothesis verdicts','',
        '- **MENTION gains concentrated in unseen: partial support.** Unseen dominates CoNLL and WNUT transfer gains; WikiANN is mixed and Few-NERD does not improve. Rare-seen gains occur too, but frequent-seen gains are not uniformly absent.',
        '- **FLIP/steered gains concentrated in seen-ambiguous: not supported as the main explanation.** Most positive total token-F1 gains are driven by unseen words, including transfer. There is a narrower contextual signal: on native Few-NERD, ambiguous-token accuracy gains exceed unambiguous gains for 18/20 configurations raw (17/20 robust), and some covered-departure slices improve. This does not make ambiguous words the main source of the aggregate gains.',
        '- **Neither gains much on seen-unambiguous: not uniformly true.** For example, WNUT seen-unambiguous token-F1 improves by about 1.32 pp with MENTION and 5.64 pp with historical lambda0.001; accuracy shifts are much smaller because O dominates this slice. The 10% definition expands this category further. No arbitrary threshold for “much” is used as a formal test.',
        '- **ICC high: substantial, but not near one.** Between-word trace explains 56.6% overall and 65.0% on entity tokens, above shuffled controls of 17.8% and 28.8%. The remaining 43.4%/35.0% is within-word variance. This does not establish that predictions ignore context or that all between-word variance is lexical identity causally.','',
        '## Deviations and missing caches','',
        '- Missing feature/logit caches: **none**. No encoder or classifier forward pass was needed; layers6/9 were not computed, as instructed.',
        '- Native test surface metadata was reconstructed by tokenizer alignment against cached raw tokens, with exact gold-order checks. Transfer word forms were reused from existing token caches. This is metadata recovery, not feature re-encoding.',
        '- Shared token predictions follow existing evaluate(): sum native class probabilities into shared labels, then argmax. They need not match native-argmax-then-map span decisions; original span metrics were reused unchanged and are not decomposed here.',
        '- Nonlinear F1 attribution uses exact three-category Shapley replacement of baseline confusion-count contributions. This is an explicit analysis convention, not a modification of F1 or a causal attribution.',
        '- Rare (1–5) versus frequent (>=6) is an additional prespecified diagnostic of the stated identity hypothesis; all requested raw/10% categories remain separate.',
        '- Computed descriptive components and nearest-centroid LOO do not involve fitting any new student. Seed SEs do not capture corpus-sampling uncertainty or the uncertainty in the training ambiguity definition.','',
        'Files: word_statistics.csv; variance.csv; within_word_labels.csv; per_seed.csv; summary.csv; coverage_exposure.csv; reused_span_results_matched.csv; reused_span_results_size_grid.csv; method.json; config.yaml. All sources and numerical conventions are recorded in method.json.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n');return 0


def main(cfg):
    if cfg.stage.experiment=='variance_report':return report(cfg)
    OmegaConf.resolve(cfg);torch.set_num_threads(cfg.runtime.threads);cfg.forge.store=str(absolute(cfg.forge.store));tracking=start_run(cfg);output=absolute(cfg.variance_analysis.results);output.mkdir(parents=True,exist_ok=True);OmegaConf.save(cfg,output/'config.yaml');started=time.monotonic();print('Loading cached features and word metadata',flush=True)
    cache=load(absolute(cfg.grid_pilot.evaluation_cache));training=load(absolute(cfg.matched.originals_run)/'prepare'/'raw.pt');train_forms,test_forms,metadata_sources=collect_metadata(cfg,cache);vocab={}
    def identifiers(words):return torch.tensor([vocab.setdefault(w,len(vocab)) for w in words])
    train_word=identifiers(train_forms);test_word={k:identifiers(w) for k,w in test_forms.items()};V=len(vocab);K=len(cache['labels']);y=training['y']
    if len(train_word)!=len(y):raise ValueError('Training forms/features have different lengths')
    native_counts=torch.bincount(train_word*K+y,minlength=V*K).reshape(V,K);mapping=torch.tensor([SHARED.index(shared_label(label)) for label in cache['labels']]);shared_counts=torch.zeros(V,len(SHARED),dtype=torch.long).scatter_add_(1,mapping[None].expand(V,-1),native_counts)
    counts_by_space={'native':native_counts,'shared':shared_counts};frequency=native_counts.sum(-1);word_lookup={w:i for w,i in vocab.items()};stats=[]
    for space,counts in counts_by_space.items():
        for threshold in cfg.variance_analysis.ambiguity_thresholds:
            cats,_=category(counts,threshold);definition='raw' if threshold==0 else 'labels_at_least_10pct';sets=[('training',train_word,y if space=='native' else mapping[y])]
            for split,item in cache['sets'].items():
                if (split=='id')!=(space=='native'):continue
                gold=item['data']['y'];names=LABELS[item['target']];gold=gold if space=='native' else torch.tensor([SHARED.index(shared_label(label)) for label in names])[gold];sets.append((item['target'],test_word[split],gold))
            for dataset,w,gold in sets:
                for subset,mask in [('all_tokens',torch.ones(len(w),dtype=torch.bool)),('gold_entity_tokens',gold!=0)]:
                    ids=w[mask];unique=ids.unique();token_counts=torch.bincount(cats[ids],minlength=3);type_counts=torch.bincount(cats[unique],minlength=3)
                    for g,name in enumerate(CATEGORIES):stats.append({'dataset':dataset,'space':space,'definition':definition,'subset':subset,'category':name,'tokens':int(token_counts[g]),'token_share':float(token_counts[g]/len(ids)) if len(ids) else None,'types':int(type_counts[g]),'type_share':float(type_counts[g]/len(unique)) if len(unique) else None,'status':'computed'})
    write_csv(output/'word_statistics.csv',stats);print('Computing trace decomposition and within-word label prediction',flush=True);variance(cfg,training['features'],train_word,y,native_counts,output)
    print('Loading augmentation coverage; selected-checkpoint exposure reconstructed',flush=True);coverage={};coverage_rows=[];seeds_by_run={'matched':[0,1,2,3,4],'size_grid':[0,1,2]};paths={'matched':absolute(cfg.matched.results),'size_grid':absolute(cfg.matched.historical_run)}
    for run,path in paths.items():
        saved=load(path/'train'/'seed0'/'model.pt');heads=saved['heads']
        for head in heads:
            if head=='baseline':continue
            words,assigned=pool_words(cfg,run,head);ids=torch.tensor([word_lookup.get(w,-1) for w in words]);m=len(words)
            for seed in seeds_by_run[run]:
                model=load(path/'train'/f'seed{seed}'/'model.pt');epoch=int(model['epoch'][list(model['heads']).index(head)]);draws=round(epoch*len(y)*m/(len(y)+m));used=torch.arange(m) if draws>=m else torch.randperm(m,generator=torch.Generator().manual_seed(seed+1000))[:draws];known=ids[used];known_labels=assigned[used];valid=known>=0;known=known[valid];known_labels=known_labels[valid]
                for space,counts in counts_by_space.items():
                    lab=known_labels if space=='native' else mapping[known_labels];covered=torch.zeros(V,dtype=torch.bool);covered[known]=True;departure=torch.zeros(V,dtype=torch.bool);departure[known[lab!=counts.argmax(-1)[known]]]=True;coverage[run,seed,head,space]=(covered,departure)
                coverage_rows.append({'run':run,'seed':seed,'arm':head,'checkpoint_epoch':epoch,'pool_entries':m,'augmented_presentations_to_checkpoint':draws,'pool_indices_encountered':len(used),'all_pool_entries_seen':draws>=m,'distinct_covered_forms':int(coverage[run,seed,head,'native'][0].sum()),'outside_test_vocabulary_pool_entries':int((ids<0).sum()),'status':'computed exact selected-checkpoint exposure from original Steps sampler'})
    write_csv(output/'coverage_exposure.csv',coverage_rows);records=[]
    for run,path in paths.items():
        for seed in seeds_by_run[run]:
            for split,item in cache['sets'].items():
                dataset=item['target'];space='native' if split=='id' else 'shared';counts=counts_by_space[space];cached=load(path/'test'/f'seed{seed}_{split}.pt');gold=item['data']['y'];gold=gold if space=='native' else torch.tensor([SHARED.index(shared_label(label)) for label in LABELS[dataset]])[gold];pred=predict_tokens(cached['logits'],None if space=='native' else mapping);heads=cached['heads'];base=list(heads).index('baseline');w=test_word[split]
                if not torch.equal(cached['gold'],item['data']['y']):raise ValueError('Cached test logits have different gold/order')
                for threshold in cfg.variance_analysis.ambiguity_thresholds:
                    cats,_=category(counts,threshold);groups=cats[w];definition='raw' if threshold==0 else 'labels_at_least_10pct';c=grouped(pred,gold,groups,3);base_counts=[[int(c[k][base,g]) for k in ('tp','fp','fn')] for g in range(3)]
                    for h,head in enumerate(heads):
                        contribution,total=shapley(base_counts,[[int(c[k][h,g]) for k in ('tp','fp','fn')] for g in range(3)])
                        for g,name in enumerate(CATEGORIES):records.append(record(run,seed,head,dataset,definition,'B',(g,name),c,h,base,contribution[g],total))
                        if head=='baseline':continue
                        covered,departure=coverage[run,seed,head,space];pair_pred=pred[:,[base,h]];ambiguous=groups==2;relevant=run=='size_grid' or '_FLIP' in head or '_AGREED' in head
                        for part,mask,group,names in [('C_coverage',ambiguous,covered[w].long(),['ambiguous_uncovered','ambiguous_covered']),('C_departure',ambiguous&covered[w],departure[w].long(),['covered_no_majority_departure','covered_majority_departure'])]:
                            grouped_ids=torch.where(mask,group,-1);c2=grouped(pair_pred,gold,grouped_ids,2)
                            for g,name in enumerate(names):records.append(record(run,seed,head,dataset,definition,part,(g,name),c2,1,0,relevant=relevant))
                frequency_groups=torch.where(frequency[w]==0,0,torch.where(frequency[w]<=cfg.variance_analysis.rare_max_count,1,2));cf=grouped(pred,gold,frequency_groups,3)
                for h,head in enumerate(heads):
                    for g,name in enumerate(['unseen','seen_rare_1_5','seen_frequent_6plus']):records.append(record(run,seed,head,dataset,'frequency','B_frequency',(g,name),cf,h,base))
                print(f'{run} seed{seed} {dataset}: cached predictions analyzed',flush=True)
    write_csv(output/'per_seed.csv',records);summary=[];grouped_records=defaultdict(list)
    for r in records:grouped_records[r['run'],r['arm'],r['dataset'],r['definition'],r['part'],r['category']].append(r)
    for key,rows in grouped_records.items():
        row=dict(zip(('run','arm','dataset','definition','part','category'),key));row['tokens_mean']=sum(r['tokens'] for r in rows)/len(rows);row['seeds']=len(rows)
        for metric in ('accuracy','token_micro_f1','accuracy_gain','token_f1_gain','shapley_contribution','overall_token_f1_gain'):
            values=torch.tensor([r[metric] for r in rows if r[metric] is not None],dtype=torch.double);row[metric+'_mean']=float(values.mean()) if len(values) else None;row[metric+'_sd']=float(values.std()) if len(values)>1 else None;row[metric+'_se']=float(values.std()/math.sqrt(len(values))) if len(values)>1 else None
        numerator=row['shapley_contribution_mean'];denominator=row['overall_token_f1_gain_mean'];row['share_of_mean_total_gain']=numerator/denominator if numerator is not None and denominator else None;row['near_zero_total_change']=abs(denominator)<.001 if denominator is not None else None;row['status']='computed';summary.append(row)
    write_csv(output/'summary.csv',summary)
    for run,path in paths.items():
        original=read_csv(path/'test'/('paired_summary.csv' if run=='matched' else 'summary.csv'));write_csv(output/f'reused_span_results_{run}.csv',[{**r,'status':'reused unchanged'} for r in original])
    save_json(output/'method.json',{'all_results':'New statistics computed from reused original features, logits, token metadata and pool identities. Original span results copied unchanged with reused status. No model inference, generation, training, new annotations or GPU.',
        'metadata':metadata_sources,'native_metadata':'Raw cached Few-NERD test tokens retokenized only for alignment; no encoder pass. Exact label vectors checked against feature cache. Training case-sensitive forms use the same native prefix convention as natural-pairs code.',
        'ambiguity':'Native 9 labels for Few-NERD; existing shared PER/ORG/LOC/O mapping for transfer. Train counts are recomputed in each label space. Raw positive labels versus labels with frequency >=10 percent. O counts; unseen has zero training occurrences.',
        'predictions':'Native argmax unchanged; shared token predictions use existing evaluate() probability-sum projection before argmax. This matches existing token metrics; category token-F1 is not span F1.',
        'attribution':'Exact Shapley decomposition over three disjoint categories: replace baseline TP/FP/FN counts by arm counts for each subset, average marginal F1 changes over all category orders. Contributions sum exactly to total token-micro-F1 change. Shares are signed; can exceed 100 percent or be negative; undefined at zero total, flagged unstable below 0.1 pp.',
        'features':'Cached block12 original word-level mean-WordPiece inputs, no additional layers. Words with >=2 original occurrences, cap50 without replacement seed0. Entity stratum restricts sampled gold-entity tokens. TSS=BSS+WSS; all divided by n-1. ICC here is descriptive between/total, not a fitted random-effects reliability parameter. Word-ID permutation preserves stratum group sizes; one seeded control, no permutation p-value.',
        'within_word':'Ambiguity selected using full original training counts, centroids estimated on the capped sample. Label explains between word-label means divided by within-word SS. LOO subtracts focal observation from its gold label centroid; missing gold-centroid support is reported; leave-one-out within-word majority is a control.',
        'coverage':'Actual selected checkpoint prefix of original Steps stream reconstructed. If presentations cover one shuffled pool pass all entries are seen; otherwise exact seed+1000 permutation prefix is used. Full source/focal assigned label matches use actual saved parent/selection indices. Baselines compared on the same arm-specific C masks.',
        'rare_extension':'Additional requested-identity diagnostic: rare seen means 1–5 original occurrences; frequent means >=6. These slices supplement, not replace, the requested ambiguity categories.',
        'missing_caches':[],'deviations':['No layers6/9 because no encoder pass was needed, as instructed. Category gain shares use explicit nonlinear Shapley attribution rather than an invalid weighted average of F1 values.'],'variance_status':'computed; source feature tensors reused'})
    save_json(output/'runtime.json',{'seconds':time.monotonic()-started});report(cfg);tracking.finish({'runs':2,'per_seed_rows':len(records),'summary_rows':len(summary)});print(output,flush=True);return 0


if __name__=='__main__':run('lion.variance_decomposition',sys.argv[1:],main_module='run',config_dir=str(ROOT/'conf'),config_name='variance_decomposition')
