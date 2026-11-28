"""Counting-only information audit of saved labelled augmentation pools."""
import json,math,sys,time
from collections import defaultdict
import torch
from transformers import AutoTokenizer
from forge import run,start_run
from omegaconf import OmegaConf
from ..diagnostics.common import ROOT,absolute,read_csv,write_csv,save_json


def load(p):return torch.load(p,mmap=True,weights_only=False)
def entropy(keys):
    counts=torch.unique(keys,return_counts=True)[1].double();p=counts/counts.sum()
    return float(-(p*p.log()).sum()+(len(counts)-1)/(2*counts.sum()))
def cmi(w,c,y,K,C):return entropy(w*K+y)+entropy(c*K+y)-entropy(y)-entropy((w*C+c)*K+y)
def excess(w,c,y,K,C,seed,shuffles):
    rng=torch.Generator().manual_seed(seed);groups=[(y==g).nonzero().flatten() for g in y.unique()];observed=cmi(w,c,y,K,C);null=[]
    for _ in range(shuffles):
        perm=w.clone()
        for ix in groups:perm[ix]=w[ix[torch.randperm(len(ix),generator=rng)]]
        null.append(cmi(perm,c,y,K,C))
    return observed-sum(null)/len(null),observed,sum(null)/len(null)
def label_entropy(counts):
    n=counts.sum(1);p=counts.double()/n.clamp_min(1)[:,None];h=-(p*p.clamp_min(1e-300).log()).sum(1);seen=n>0
    return float(h[seen].mean()),float((h*n).sum()/n.sum())
def rank(x):
    v,inv,n=x.unique(sorted=True,return_inverse=True,return_counts=True);r=(n.cumsum(0)-n.double()/2)[inv];return r
def corr(a,b):
    a=a-a.mean();b=b-b.mean();d=a.norm()*b.norm();return float(a@b/d) if d>0 else None
def standard(x):
    sd=x.std(0,unbiased=False);return (x-x.mean(0))/sd.clamp_min(1e-12),sd


def contexts(words,radius):
    return [tuple(words[j] if 0<=j<len(words) else '<BOS>' if j<0 else '<EOS>' for j in list(range(i-radius,i))+list(range(i+1,i+radius+1))) for i in range(len(words))]


def count(cfg,out):
    ref=load(absolute(cfg.language_checks.results)/'prepare/reference.pt');raw=load(absolute(cfg.matched.results)/'prepare/raw.pt');forms=ref['forms'];train_w=ref['train_word'];lookup=dict(ref['lookup']);mapping=ref['mapping'];native_y=raw['y'];original_c={1:[],2:[]};position=0
    for r in read_csv(absolute(cfg.matched.results)/'prepare/sentences.csv'):
        n=int(r['words']);words=forms[position:position+n];position+=n
        for radius in original_c:original_c[radius].extend(contexts(words,radius))
    tokenizer=AutoTokenizer.from_pretrained(cfg.encoder.name,local_files_only=True);vocab=tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    selections=json.loads((absolute(cfg.matched.results)/'match/selections.json').read_text());exposure=read_csv(absolute(cfg.variance_analysis.results)/'coverage_exposure.csv');arms=list(dict.fromkeys((r['run'],r['arm']) for r in exposure));stats=[]
    for run_name,arm in [('originals','baseline')]+arms:
        print(f'Count {run_name}/{arm}',flush=True)
        if run_name=='originals':words=[];aug_y=torch.empty(0,dtype=torch.long);aug_c={1:[],2:[]}
        else:
            if run_name=='matched':
                selection=selections[arm];pool=load(absolute(cfg.matched.results)/'new_pools'/f"{selection['parent']}.pt");indices=selection['indices']
            else:pool=load(absolute(cfg.matched.results)/'historical_pools'/f'{arm}.pt');indices=range(len(pool['y']))
            # Existing exposure reconstruction establishes every entry was encountered in every seed.
            relevant=[r for r in exposure if r['run']==run_name and r['arm']==arm]
            if not all(r['all_pool_entries_seen']=='True' for r in relevant):raise ValueError('Exposure is not the full pool; reconstruct the saved sampler first')
            words=[pool['word'][i] for i in indices];aug_y=pool['y'][torch.tensor(list(indices))];aug_c={1:[],2:[]}
            for i in indices:
                lo,hi=map(int,pool['offsets'][i:i+2]);ids=pool['input_ids'][lo:hi].tolist();wid=pool['word_ids'][lo:hi].tolist();slots=defaultdict(list)
                for token,j in zip(ids,wid):
                    if j>=0:slots[j].append(vocab[token])
                focal=int(pool['focal'][i]);decoded={j:tokenizer.convert_tokens_to_string(t) for j,t in slots.items()};last=max(slots)
                for radius in aug_c:aug_c[radius].append(tuple(decoded.get(j,'<BOS>' if j<0 else '<EOS>' if j>last else '<EMPTY>') for j in list(range(focal-radius,focal))+list(range(focal+1,focal+radius+1))))
            del pool
        aug_w=torch.tensor([lookup.setdefault(w,len(lookup)) for w in words],dtype=torch.long);w=torch.cat((train_w,aug_w));N=len(aug_w);V=len(lookup)
        for space in ('native','shared'):
            y0=native_y if space=='native' else mapping[native_y];ya=aug_y if space=='native' else mapping[aug_y];y=torch.cat((y0,ya));K=9 if space=='native' else 4;base=torch.bincount(train_w*K+y0,minlength=V*K).reshape(V,K);added=torch.bincount(aug_w*K+ya,minlength=V*K).reshape(V,K);total=base+added;seen=base.sum(1)>0;majority=base.argmax(1);base_h=label_entropy(base);all_h=label_entropy(total)
            common={'run':run_name,'arm':arm,'space':space,'N':N,'mixing_proportion':N/len(w),'filter':'AGREED' if 'AGREED' in arm else 'none' if arm=='MENTION' or 'ALL' in arm or run_name=='originals' else 'FLIP','S2a_departures':int(((ya!=majority[aug_w])&seen[aug_w]).sum()),'S2b_new_multilabel_types':int(((base.gt(0).sum(1)==1)&(total.gt(0).sum(1)>1)).sum()),'S2c_entropy_type_delta':all_h[0]-base_h[0],'S2c_entropy_token_delta':all_h[1]-base_h[1],'S3a_majority_occurrences':int(((ya==majority[aug_w])&seen[aug_w]).sum()),'S3b_unseen_occurrences':int((~seen[aug_w]).sum()),'status':'computed from reused pools; actual labels and unique encountered pool items'}
            for radius in (1,2):
                clook={};c=torch.tensor([clook.setdefault(v,len(clook)) for v in original_c[radius]+aug_c[radius]]);C=len(clook);pairs0=set((c[:len(y0)]*K+y0).tolist());pairs=set((c*K+y).tolist());new=pairs-pairs0;sample=torch.randperm(len(w),generator=torch.Generator().manual_seed(cfg.information_audit.seed))[:cfg.information_audit.sample_size];s1,plugin,null=excess(w[sample],c[sample],y[sample],K,C,cfg.information_audit.seed,cfg.information_audit.shuffles)
                row=common|{'context_radius':radius,'sample_n':len(sample),'S1a_excess_CMI':s1,'CMI_MM':plugin,'shuffle_mean_CMI_MM':null,'S1b_new_context_label_pairs':len(new),'S1b_new_contexts':len(set(c[len(y0):].tolist())-set(c[:len(y0)].tolist()))}
                for label in range(K):row[f'new_contexts_label_{label}']=sum(v%K==label for v in new)
                stats.append(row)
    base={(r['space'],r['context_radius']):r for r in stats if r['run']=='originals'}
    numeric=['S1a_excess_CMI','S1b_new_context_label_pairs','S1b_new_contexts','S2a_departures','S2b_new_multilabel_types','S2c_entropy_type_delta','S2c_entropy_token_delta','S3a_majority_occurrences','S3b_unseen_occurrences']
    for r in stats:
        for key in numeric:
            r[key+'_delta_vs_originals']=r[key]-base[r['space'],r['context_radius']][key];r[key+'_per_augmented_token']=r[key]/r['N'] if r['N'] else None
        r['delta_CMI']=r['S1a_excess_CMI_delta_vs_originals'];r['S1_decoupling']=-r['delta_CMI'];r['logN']=math.log(max(1,r['N']))
    write_csv(out/'arm_statistics.csv',stats);return stats


def merge(cfg,out,stats):
    gains=read_csv(absolute(cfg.language_checks.results)/'tail/summary.csv');rows=[];by={(r['run'],r['arm'],r['space'],r['context_radius']):r for r in stats}
    for g in gains:
        if g['arm']=='baseline' or g['subset']!='all_tokens':continue
        if g['partition']=='bands' and g['bin'] in ('unseen','rare_1_7'):definitions=['raw','labels_at_least_10pct'];kind='unseen' if g['bin']=='unseen' else 'rare'
        elif g['partition']=='frequency_ambiguity' and g['bin'] in ('frequent_unambiguous','frequent_ambiguous'):definitions=[g['definition']];kind=g['bin']
        else:continue
        for definition in definitions:
            for radius in (1,2):
                s=by[g['run'],g['arm'],'native' if g['dataset']=='fewnerd' else 'shared',radius];rows.append(s|{'dataset':g['dataset'],'definition':definition,'token_type':kind,'test_n':float(g['n']),'small_slice':float(g['n'])<cfg.information_audit.min_slice,'accuracy_gain':float(g['accuracy_gain_mean']),'accuracy_seed_SE':float(g['accuracy_gain_se']),'token_f1_gain':float(g['token_f1_gain_mean']),'token_f1_seed_SE':float(g['token_f1_gain_se']),'gain_status':'reused language_checks/tail/summary.csv'})
    write_csv(out/'merged_gains.csv',rows);return rows


S1=['S1_decoupling','S1b_new_context_label_pairs'];S2=['S2a_departures','S2b_new_multilabel_types','S2c_entropy_type_delta','S2c_entropy_token_delta'];S3=['S3a_majority_occurrences','S3b_unseen_occurrences']
def analyze(cfg,out,rows):
    correlations=[];regressions=[];errors=[]
    for scope in ('matched','size_grid','pooled'):
      for radius in (1,2):
       for definition in ('raw','labels_at_least_10pct'):
        for kind in ('unseen','rare','frequent_unambiguous','frequent_ambiguous'):
         data=[r for r in rows if (scope=='pooled' or r['run']==scope) and r['context_radius']==radius and r['definition']==definition and r['token_type']==kind and not r['small_slice']];groups=[(r['run'],r['arm']) for r in data];unique=list(dict.fromkeys(groups));groupid=torch.tensor([unique.index(g) for g in groups]);datasets=sorted(set(r['dataset'] for r in data));z=torch.tensor([[r[k] for k in S1+S2+S3+['logN']] for r in data],dtype=torch.double);nuisance=torch.tensor([[1.]+[float(r['dataset']==d) for d in datasets[1:]]+([float(r['run']=='matched')] if scope=='pooled' else []) for r in data],dtype=torch.double);meta={'scope':scope,'context_radius':radius,'definition':definition,'token_type':kind,'rows':len(data),'arms':len(unique),'status':'computed descriptive; seeds summarized, not independent replicates'}
         for metric in ('accuracy_gain','token_f1_gain'):
          y=torch.tensor([r[metric] for r in data],dtype=torch.double)
          for j,key in enumerate(S1+S2+S3+['logN']):correlations.append(meta|{'metric':metric,'statistic':key,'spearman':corr(rank(z[:,j]),rank(y))})
          for version in ('composites','sub_statistics','logN_only'):
           def design(train,test):
            zx,sd=standard(z[train]);zt=(z[test]-z[train].mean(0))/sd.clamp_min(1e-12);ys=y[train].std(unbiased=False).clamp_min(1e-12);yt=(y[train]-y[train].mean())/ys;names=[];a=[];b=[]
            if version=='composites':
                pos=0
                for name,keys in [('S1',S1),('S2',S2),('S3',S3)]:
                    ix=[j for j in range(pos,pos+len(keys)) if sd[j]>1e-12];pos+=len(keys)
                    if not ix:continue
                    _,_,v=torch.linalg.svd(zx[:,ix],full_matrices=False);weight=v[0];weight*=1 if float(weight.sum())>=0 else -1;a.append(zx[:,ix]@weight);b.append(zt[:,ix]@weight);names.append(name)
            elif version=='sub_statistics':
                for j,key in enumerate(S1+S2+S3):
                    if sd[j]>1e-12:a.append(zx[:,j]);b.append(zt[:,j]);names.append(key)
            if sd[-1]>1e-12:a.append(zx[:,-1]);b.append(zt[:,-1]);names.append('logN')
            A=torch.stack(a,1) if a else torch.empty(len(train),0,dtype=torch.double);B=torch.stack(b,1) if b else torch.empty(len(test),0,dtype=torch.double)
            if A.shape[1]:
                scale=A.std(0,unbiased=False).clamp_min(1e-12);A=A/scale;B=B/scale
            X=torch.cat((nuisance[train],A),1);T=torch.cat((nuisance[test],B),1);beta=torch.linalg.pinv(X)@yt
            return T@beta*ys+y[train].mean(),beta,X,names
           ix=torch.arange(len(data));prediction,beta,X,names=design(ix,ix);rankx=int(torch.linalg.matrix_rank(X));p=X.shape[1];offset=p-len(names)
           for j,name in enumerate(names):
                column=X[:,offset+j];others=torch.cat((X[:,:offset+j],X[:,offset+j+1:]),1);residual=column-others@(torch.linalg.pinv(others)@column);den=float((column-column.mean()).square().sum());r2=1-float(residual.square().sum())/den if den else 1;vif=1/(1-r2) if r2<1-1e-10 else float('inf');regressions.append(meta|{'metric':metric,'version':version,'term':name,'standardized_beta':float(beta[offset+j]),'VIF':vif,'design_rank':rankx,'design_columns':p,'rank_deficient':rankx<p})
           for gid,g in enumerate(unique):
                train=(groupid!=gid).nonzero().flatten();test=(groupid==gid).nonzero().flatten();pred,*_=design(train,test);err=pred-y[test]
                errors.append(meta|{'metric':metric,'version':version,'held_out_run':g[0],'held_out_arm':g[1],'squared_error':float(err.square().mean()),'absolute_error':float(err.abs().mean())})
    write_csv(out/'correlations.csv',correlations);write_csv(out/'regressions.csv',regressions);write_csv(out/'leave_one_arm_out.csv',errors)
    summary=[];bucket=defaultdict(list)
    for r in errors:bucket[tuple(r[k] for k in ('scope','context_radius','definition','token_type','metric','version'))].append(r)
    for key,rs in bucket.items():summary.append(dict(zip(('scope','context_radius','definition','token_type','metric','version'),key))|{'RMSE':math.sqrt(sum(r['squared_error'] for r in rs)/len(rs)),'MAE':sum(r['absolute_error'] for r in rs)/len(rs),'held_out_arms':len(rs),'status':'computed; whole arm held out across all datasets; transformations fitted inside fold'})
    write_csv(out/'prediction_errors.csv',summary)
    lines=['# Information audit','','All statistics are computed from cached pools; all test gains and seed SEs are reused. No generation, training, scoring or model forward passes.','','## P1–P4 (primary radius 1, raw ambiguity; descriptive)','','| Prediction | Verdict | Numbers |','|---|---|---|']
    for kind,pid,term in [('unseen','P1 unseen','S1'),('rare','P1 rare','S1'),('frequent_ambiguous','P2','S2'),('frequent_unambiguous','P3','S3')]:
        details=[];flags=[]
        for metric in ('accuracy_gain','token_f1_gain'):
            rs=[r for r in regressions if (r['scope'],r['context_radius'],r['definition'],r['token_type'],r['metric'],r['version'])==('pooled',1,'raw',kind,metric,'composites')];coef={r['term']:r['standardized_beta'] for r in rs};err={r['version']:r['RMSE'] for r in summary if (r['scope'],r['context_radius'],r['definition'],r['token_type'],r['metric'])==('pooled',1,'raw',kind,metric)};ok=coef.get(term,0)>0 and err['composites']<err['logN_only']
            if pid=='P2':ok=ok and coef.get('S2',0)>max(coef.get('S1',0),coef.get('S3',0))
            if pid=='P3':ok=ok and sum(abs(r[metric]) for r in rows if r['context_radius']==1 and r['definition']=='raw' and r['token_type']==kind)/len([r for r in rows if r['context_radius']==1 and r['definition']=='raw' and r['token_type']==kind])<.005
            flags.append(ok);details.append(f"{metric}: β({term})={coef.get(term,float('nan')):+.3f}; LOAO RMSE {100*err['composites']:.3f} vs size-only {100*err['logN_only']:.3f} pp")
        verdict='PASS' if all(flags) else 'FAIL' if not any(flags) else 'MIXED';lines.append(f'| {pid} | {verdict} | '+ '; '.join(details)+' |')
    paired=[]
    for metric in ('accuracy_gain','token_f1_gain'):
        ds=[r for r in rows if r['run']=='matched' and r['context_radius']==1 and r['definition']=='raw' and r['token_type']=='unseen' and r['N']==1932];a=torch.tensor([r['delta_CMI'] for r in ds],dtype=torch.double);b=torch.tensor([r[metric] for r in ds],dtype=torch.double);rho=corr(rank(-a),rank(b));paired.append(rho);lines.append(f"\nP4 {metric}: same-N (1932) Spearman(−ΔCMI, unseen gain)={rho:+.3f}. MENTION and nine same-size generated arms, four datasets; repeats are descriptive.")
    lines.insert(10,f"| P4 | {'PASS' if all(r>0 for r in paired) else 'FAIL' if all(r<=0 for r in paired) else 'MIXED'} | Same-N correlations below. |")
    lines+=['','## Conventions and limitations','','CMI is in nats: Miller–Madow-corrected H(W,Y)+H(C,Y)−H(Y)−H(W,C,Y), minus the mean of 20 within-label word permutations. Every arm uses a fixed-seed 60,000-token sample without replacement, including originals. ΔCMI subtracts the originals-only estimate. Sparse exact contexts make this estimator exploratory; negative corrected or excess values are retained. Context-label novelty counts all distinct pairs absent from originals. Per-label new contexts mean contexts newly paired with that label.','',
    'Original contexts use case-sensitive saved surface words. Generated contexts use saved original word-slot IDs and detokenize the generated pieces within each slot; no retokenization or sentence repair. Thus generated WordPiece strings that cross lexical boundaries remain in their original slots. Missing neighbour slots use explicit boundaries/EMPTY. This operational deviation is necessary because arbitrary piece replacement need not produce ordinary words.','',
    'Unique encountered pool entries are counted once, preserving duplicates stored as separate items. Repeated epoch presentations do not multiply N. Existing coverage_exposure establishes all entries were encountered for every seed. Mixing is the existing token-proportional sampler expectation N/(60321+N), not a realised draw fraction. Entropy is plug-in within-word entropy (not Miller–Madow); type and token means include newly introduced words. Unseen-majority departures are excluded and reported separately.','',
    'Statistics are computed separately in native and shared label spaces. The ≥10% ambiguity definition changes test slices only; loss labels and pool entropy remain the actual labels. Small slices <50 tokens are excluded from fitting and flagged in merged tables. Seed SEs are retained, not treated as additional independent rows or used as inverse-variance weights.','',
    'Regressions include dataset indicators in addition to the requested pooled run indicator. Slopes/response are standardized; S PCs are oriented towards positive summed loadings (S1 uses −ΔCMI). Constant predictors are omitted. VIF and matrix rank expose collinearity; pseudoinverse coefficients in deficient designs are not separately identifiable. LOAO holds out the entire run–arm, including all four tests; scaling/PCA is fitted only on remaining arms. These are descriptive small-arm associations, not significance tests or causal identification. P3 operationalizes small as mean absolute gain <0.5 percentage points.','',
    'Missing optional input: independent_judge has pools and headline test gains, but no existing word-type sliced gain tables or coverage_exposure reconstruction in the requested sources. It is excluded rather than silently recomputing test slices or assuming exposure. No required input for matched/size_grid is missing. All original statistics, augmented counts, CMI, correlations and regression numbers are computed; paired gains/seed SEs and checkpoint exposure are reused.','',
    'If the composite LOAO RMSE is at least the size-only RMSE, these descriptors add no predictive value beyond pool size for that outcome. See prediction_errors.csv for every run, definition and context radius. P1–P4 are operational descriptive verdicts, not a test of a formal random-effects model.','',
    'Files: arm_statistics.csv (including per-augmented-token values and deltas); merged_gains.csv; correlations.csv; regressions.csv; leave_one_arm_out.csv; prediction_errors.csv; config.yaml.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main(cfg):
    OmegaConf.resolve(cfg);torch.set_num_threads(cfg.runtime.threads);cfg.forge.store=str(absolute(cfg.forge.store));tracking=start_run(cfg);out=absolute(cfg.information_audit.results);out.mkdir(parents=True,exist_ok=True);OmegaConf.save(cfg,out/'config.yaml');start=time.monotonic();stats=count(cfg,out);rows=merge(cfg,out,stats);analyze(cfg,out,rows);save_json(out/'runtime.json',{'seconds':time.monotonic()-start,'model_forward_passes':0});tracking.finish({'arms':len(stats)//4-1});return 0

if __name__=='__main__':run('lion.information_audit',sys.argv[1:],main_module='run',config_dir=str(ROOT/'conf'),config_name='information_audit')
