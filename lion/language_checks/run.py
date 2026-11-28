"""Paradigm, frequency-tail and ambiguity checks; no augmentation or student fitting."""
import csv,json,math,sys,time
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from forge import run,start_run
from omegaconf import OmegaConf
from ..diagnostics.common import ROOT,absolute,save_json,write_csv,read_csv
from ..diagnostics.data import SHARED
from ..labels import LABELS
from ..metrics import shared_label
from ..runtime import configure,autocast
from ..encoder import load_masked_encoder
from ..variance_decomposition import run as previous
from ..guided_refinement.population import correlation


def root(cfg):return absolute(cfg.language_checks.results)
def load(path):return torch.load(path,mmap=True,weights_only=False)
def bins(f):return torch.bucketize(f,torch.tensor([0,1,3,7,15,31]),right=False)


def prepare(cfg,output):
    cache=load(absolute(cfg.grid_pilot.evaluation_cache));training=load(absolute(cfg.matched.originals_run)/'prepare'/'raw.pt');forms,tests,sources,rows=previous.collect_metadata(cfg,cache,return_rows=True);vocab={}
    def ids(words):return torch.tensor([vocab.setdefault(w,len(vocab)) for w in words])
    train=ids(forms);test_ids={k:ids(w) for k,w in tests.items()};V=len(vocab);K=len(cache['labels']);counts=torch.bincount(train*K+training['y'],minlength=V*K).reshape(V,K);mapping=torch.tensor([SHARED.index(shared_label(label)) for label in cache['labels']]);shared=torch.zeros(V,len(SHARED),dtype=torch.long).scatter_add_(1,mapping[None].expand(V,-1),counts);frequency=counts.sum(-1);samples=[];datasets={};rng=torch.Generator().manual_seed(cfg.language_checks.seed)
    scorer=load(absolute(cfg.natural_pairs.results)/'predictions.pt');native_sentence=cache['sets']['id']['data']['sentence']
    from ..matched_regime.evaluation import rowkeys
    native_keys=rowkeys(native_sentence,cfg.encoder.max_length);oldkeys=rowkeys(scorer['sentence'],cfg.encoder.max_length);oldpositions=torch.searchsorted(oldkeys,native_keys);available=oldpositions<len(oldkeys);safe=oldpositions.clamp_max(len(oldkeys)-1);available&=oldkeys[safe]==native_keys;A=torch.full((len(native_keys),),-1,dtype=torch.long);A[available]=scorer['prediction'][safe[available]]
    for split,item in cache['sets'].items():
        target=item['target'];space='native' if split=='id' else 'shared';local_counts=counts if space=='native' else shared;majority=local_counts.argmax(-1);fallback=int(local_counts.sum(0).argmax());word=test_ids[split];freq=frequency[word];raw,_=previous.category(local_counts,0.);robust,_=previous.category(local_counts,.1);raw=raw[word];robust=robust[word];fb=bins(freq);stratum=fb*4+(raw==2).long()*2+(robust==2).long();groups=stratum.unique(sorted=True);size=min(cfg.language_checks.sample_per_test,len(word));quota=size//len(groups);selected=[]
        for g in groups:
            candidates=(stratum==g).nonzero().flatten();selected.append(candidates[torch.randperm(len(candidates),generator=rng)[:quota]])
        selected=torch.cat(selected);remaining=torch.ones(len(word),dtype=torch.bool);remaining[selected]=False;extra=remaining.nonzero().flatten();selected=torch.cat((selected,extra[torch.randperm(len(extra),generator=rng)[:size-len(selected)]]));selected=selected.sort().values
        lengths=torch.tensor([len(r['words']) for r in rows[split]]);end=lengths.cumsum(0);start=end-lengths;owner=torch.searchsorted(end,selected,right=True);gold=item['data']['y'];names=cache['labels'] if space=='native' else SHARED;gold=gold if space=='native' else torch.tensor([SHARED.index(shared_label(label)) for label in LABELS[target]])[gold]
        for index,o in zip(selected.tolist(),owner.tolist()):
            r=rows[split][o];f=index-int(start[o]);wid=torch.tensor(r['word_ids']);positions=(wid==f).nonzero().flatten();original=torch.tensor(r['ids']);lo,hi=int(positions[0]),int(positions[-1])+1;masked=torch.cat((original[:lo],torch.tensor([103]),original[hi:]))
            samples.append({'sample_id':len(samples),'split':split,'dataset':target,'space':space,'token_index':index,'sentence_index':o,'focal':f,'word':tests[split][index],'frequency':int(freq[index]),'frequency_bin':int(fb[index]),'raw_ambiguous':bool(raw[index]==2),'robust_ambiguous':bool(robust[index]==2),'gold':int(gold[index]),'identity':int(majority[word[index]]) if frequency[word[index]] else fallback,'A_prediction':int(A[index]) if split=='id' else -1,'sentence':' '.join(r['words']),'masked_sentence':' '.join(r['words'][:f]+['[MASK]']+r['words'][f+1:]),'ids':masked,'mask_position':lo})
        datasets[split]={'dataset':target,'space':space,'word_ids':word,'frequency':freq,'raw_category':raw,'robust_category':robust,'gold':gold,'stratum':stratum,'selected':selected}
    torch.save({'train_word':train,'forms':forms,'lookup':vocab,'counts':counts,'shared_counts':shared,'frequency':frequency,'mapping':mapping,'labels':cache['labels'],'datasets':datasets},output/'reference.pt');torch.save(samples,output/'sample.pt')
    write_csv(output/'token_sample.csv',[{k:v for k,v in r.items() if k!='ids'} for r in samples]);write_csv(output/'sampling.csv',[{'dataset':d['dataset'],'stratum':int(g),'population_n':int((d['stratum']==g).sum()),'sample_n':int((d['stratum'][d['selected']]==g).sum())} for d in datasets.values() for g in d['stratum'].unique()])
    save_json(output/'sources.json',{'status':'computed metadata from reused caches','metadata_sources':sources,'scorer_A':'Actual full-data A cached natural_pairs predictions used for native occurrences where present. No matching cached A transfer predictions found; existing student baselines are not A and are not substituted.','missing_A_transfer':['conll2003','wnut','wikiann'],'model_inference_except_MLM':False,'stratification':'Equal initial quotas over nonempty frequency × raw-ambiguity × robust-ambiguity strata, then random unused-token fill to cap. Without replacement. Metrics describe this stratified sample, not a self-weighting test-population sample.'})
    return {'samples':len(samples)}


def collate(rows):
    width=max(len(r['ids']) for r in rows);ids=torch.zeros(len(rows),width,dtype=torch.long);attention=torch.zeros_like(ids,dtype=torch.bool)
    for i,r in enumerate(rows):ids[i,:len(r['ids'])]=r['ids'];attention[i,:len(r['ids'])]=True
    return ids,attention,torch.tensor([r['mask_position'] for r in rows]),[r['sample_id'] for r in rows]


def metrics(pred,gold):
    pred=torch.as_tensor(pred);gold=torch.as_tensor(gold);correct=pred==gold;tp=int((correct&(gold!=0)).sum());fp=int((~correct&(pred>0)).sum());fn=int((~correct&(gold!=0)).sum())
    return {'n':len(gold),'accuracy':float(correct.double().mean()) if len(gold) else None,'non_O_micro_f1':previous.f1(tp,fp,fn) if len(gold) else None,'tp':tp,'fp':fp,'fn':fn,'abstentions':int((pred<0).sum())}


@torch.no_grad()
def paradigm(cfg,output):
    reference=load(root(cfg)/'prepare'/'reference.pt');sample=load(root(cfg)/'prepare'/'sample.pt');tokenizer=AutoTokenizer.from_pretrained(cfg.encoder.name,local_files_only=True);encoder=load_masked_encoder(cfg);tokens=tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))));allowed=torch.tensor([not t.startswith('##') and i not in tokenizer.all_special_ids for i,t in enumerate(tokens)]);known=torch.tensor([reference['lookup'].get(t,-1) for t in tokens]);voting=allowed&(known>=0)
    # The lookup includes test-only words; only positive TRAIN counts may vote.
    voting&=reference['frequency'][known.clamp_min(0)]>0;labels_by_space={};out={}
    for space,counts in [('native',reference['counts']),('shared',reference['shared_counts'])]:labels_by_space[space]=counts.argmax(-1)[known.clamp_min(0)]
    fh=(output/'mlm_top50.csv').open('w',newline='');writer=csv.DictWriter(fh,fieldnames=['sample_id','dataset','rank','filler','whole_word_probability','full_vocab_probability','training_frequency','majority_native','majority_shared','receives_vote']);writer.writeheader()
    loader=DataLoader(sample,batch_size=cfg.language_checks.mlm_batch_size,collate_fn=collate,num_workers=cfg.runtime.workers)
    for number,(ids,attention,positions,indices) in enumerate(loader):
        with autocast(cfg):logits=encoder(ids.to(cfg.runtime.device),attention.to(cfg.runtime.device),positions.to(cfg.runtime.device),mlm=True)['logits'].float().cpu()
        full=logits.softmax(-1);whole=logits.masked_fill(~allowed,-torch.inf).softmax(-1);mass=whole[:,voting].sum(-1);whole_mass=full[:,allowed].sum(-1);top_probability,top_index=whole.topk(cfg.language_checks.top_k,-1);top_vote=voting[top_index];top_mass=(top_probability*top_vote).sum(-1)
        for space,K in [('native',len(reference['labels'])),('shared',len(SHARED))]:
            lab=labels_by_space[space];score=torch.zeros(len(ids),K).scatter_add_(1,lab[None].expand(len(ids),-1),whole*voting);top_score=torch.zeros(len(ids),K).scatter_add_(1,lab[top_index],top_probability*top_vote)
            score[mass>0]/=mass[mass>0,None];top_score[top_mass>0]/=top_mass[top_mass>0,None];prediction=score.argmax(-1);top_prediction=top_score.argmax(-1);prediction[mass==0]=-1;top_prediction[top_mass==0]=-1
            for i,index in enumerate(indices):
                if sample[index]['space']==space:out[index]={'paradigm':int(prediction[i]),'top50':int(top_prediction[i]),'vote_mass':float(mass[i]),'top50_vote_mass':float(top_mass[i]),'top50_total_mass':float(top_probability[i].sum()),'top50_voting_fraction_within_top50':float(top_mass[i]/top_probability[i].sum()),'whole_word_mass_of_full_vocab':float(whole_mass[i]),'full_vocab_vote_mass':float(full[i,voting].sum()),'most_likely_filler':tokens[int(top_index[i,0])]}
        for i,index in enumerate(indices):
            for rank,(token,p) in enumerate(zip(top_index[i].tolist(),top_probability[i].tolist()),1):writer.writerow({'sample_id':index,'dataset':sample[index]['dataset'],'rank':rank,'filler':tokens[token],'whole_word_probability':p,'full_vocab_probability':float(full[i,token]),'training_frequency':int(reference['frequency'][known[token]]) if known[token]>=0 else 0,'majority_native':int(labels_by_space['native'][token]) if voting[token] else '', 'majority_shared':int(labels_by_space['shared'][token]) if voting[token] else '', 'receives_vote':bool(voting[token])})
        if (number+1)%100==0:print(f'MLM sample batches {number+1}/{len(loader)}',flush=True)
    fh.close();rows=[{k:v for k,v in r.items() if k!='ids'}|out[r['sample_id']] for r in sample];write_csv(output/'token_predictions.csv',rows);tables=[];failures=[];confusions=[];examples=[];hypotheses=[];rng=torch.Generator().manual_seed(cfg.language_checks.seed)
    for dataset in dict.fromkeys(r['dataset'] for r in rows):
        ds=[r for r in rows if r['dataset']==dataset];gold=torch.tensor([r['gold'] for r in ds]);p=torch.tensor([r['paradigm'] for r in ds]);word=torch.tensor([r['identity'] for r in ds]);names=reference['labels'] if ds[0]['space']=='native' else SHARED
        for definition,ambkey in [('raw','raw_ambiguous'),('labels_at_least_10pct','robust_ambiguous')]:
            ambiguity=torch.tensor([r[ambkey] for r in ds]);error=p!=gold;fb=torch.tensor([r['frequency_bin'] for r in ds]);unseen=fb==0;unambiguous=(~ambiguity)&(~unseen);gaps=[]
            for g in range(1,7):
                mask=(fb==g)&unambiguous
                if mask.any():gaps.append((g,abs(float((p[mask]==gold[mask]).double().mean()-(word[mask]==gold[mask]).double().mean()))))
            frequent=next((v for g,v in gaps if g==6),None);closest=frequent is not None and frequent<=min(v for _,v in gaps)+1e-12;unseen_gain=float(((p[unseen]==gold[unseen]).double()-(word[unseen]==gold[unseen]).double()).mean());ambig_share=float(ambiguity.double().mean());error_ambig_share=float(ambiguity[error].double().mean()) if error.any() else 0.
            hypotheses.append({'dataset':dataset,'definition':definition,'unseen_accuracy_gain':unseen_gain,'unseen_beats_identity':unseen_gain>0,'frequent_unambiguous_abs_accuracy_gap':frequent,'closest_gap_bin':','.join(str(g) for g,v in gaps if v==min(t for _,t in gaps)),'frequent_unambiguous_closest':closest,'ambiguous_share_sample':ambig_share,'ambiguous_share_paradigm_errors':error_ambig_share,'error_overrepresentation':error_ambig_share>ambig_share,'PASS':unseen_gain>0 and closest and error_ambig_share>ambig_share})
            for partition in ['all','frequency','ambiguity']:
                groups=[('all',torch.ones(len(ds),dtype=torch.bool))] if partition=='all' else [(cfg.language_checks.frequency_names[g],fb==g) for g in range(7)] if partition=='frequency' else [('unseen',unseen),('seen_unambiguous',unambiguous),('seen_ambiguous',ambiguity)]
                for group,mask in groups:
                    for model,key in [('paradigm','paradigm'),('top50','top50'),('identity','identity'),('A','A_prediction')]:
                        pred=torch.tensor([r[key] for r in ds]);available=pred>=0 if model=='A' else torch.ones(len(ds),dtype=torch.bool);selected=mask&available
                        mass_key='top50_vote_mass' if model=='top50' else 'vote_mass';tables.append({'dataset':dataset,'definition':definition,'partition':partition,'group':group,'model':model,'sample_n':int(mask.sum()),'available_n':int(selected.sum()),'status':'computed; A predictions reused' if model=='A' else 'computed',**metrics(pred[selected],gold[selected]),'mean_vote_mass':sum(ds[i][mass_key] for i in selected.nonzero().flatten().tolist())/int(selected.sum()) if selected.any() else None})
        for pright in (False,True):
            for wright in (False,True):
                mask=((p==gold)==pright)&((word==gold)==wright);indices=mask.nonzero().flatten();cell=f'paradigm_{"right" if pright else "wrong"}__word_{"right" if wright else "wrong"}'
                failures.append({'dataset':dataset,'cell':cell,'n':len(indices),'raw_ambiguous_fraction':sum(ds[i]['raw_ambiguous'] for i in indices.tolist())/len(indices) if len(indices) else None,'robust_ambiguous_fraction':sum(ds[i]['robust_ambiguous'] for i in indices.tolist())/len(indices) if len(indices) else None})
                for model,pred in [('paradigm',p),('identity',word)]:
                    for (g,z),count in zip(*torch.unique(torch.stack((gold[mask],pred[mask]),1),dim=0,return_counts=True)):confusions.append({'dataset':dataset,'cell':cell,'model':model,'gold':names[int(g)],'prediction':names[int(z)] if z>=0 else 'ABSTAIN','n':int(count)})
                for i in indices[torch.randperm(len(indices),generator=rng)[:20]].tolist():examples.append({**ds[i],'cell':cell,'gold_name':names[int(gold[i])],'paradigm_name':names[int(p[i])] if p[i]>=0 else 'ABSTAIN','identity_name':names[int(word[i])]})
    write_csv(output/'metrics.csv',tables);write_csv(output/'failure_cells.csv',failures);write_csv(output/'confusions.csv',confusions);write_csv(output/'failure_examples.csv',examples);write_csv(output/'hypothesis_A.csv',hypotheses);return {'samples':len(rows)}


def metric_row(c,h,g,base):
    n=int(c['n'][g]);v={key:int(c[key][h,g]) for key in ('tp','fp','fn')};b={key:int(c[key][base,g]) for key in ('tp','fp','fn')}
    return {'n':n,'accuracy':float(c['correct'][h,g]/n) if n else None,'accuracy_gain':float((c['correct'][h,g]-c['correct'][base,g])/n) if n else None,
        'token_micro_f1':previous.f1(**v) if n else None,'token_f1_gain':previous.f1(**v)-previous.f1(**b) if n else None,**v,**{'baseline_'+k:t for k,t in b.items()}}


def summarize(rows,keys,metrics):
    groups=defaultdict(list);result=[]
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    for key,values in groups.items():
        row=dict(zip(keys,key));row['seeds']=len(values);row['n']=sum(r['n'] for r in values)/len(values)
        for metric in metrics:
            x=torch.tensor([r[metric] for r in values if r[metric] is not None],dtype=torch.double);row[metric+'_mean']=float(x.mean()) if len(x) else None;row[metric+'_se']=float(x.std()/math.sqrt(len(x))) if len(x)>1 else None
        row['status']='computed from reused logits';result.append(row)
    return result


def isotonic(y):
    blocks=[]
    for i,value in enumerate(y.tolist()):
        blocks.append([i,i+1,value,1])
        while len(blocks)>1 and blocks[-2][2]<blocks[-1][2]:
            right=blocks.pop();left=blocks.pop();weight=left[3]+right[3];blocks.append([left[0],right[1],(left[2]*left[3]+right[2]*right[3])/weight,weight])
    fitted=torch.empty_like(y)
    for start,end,value,_ in blocks:fitted[start:end]=value
    return fitted


def tail(cfg,output):
    reference=load(root(cfg)/'prepare'/'reference.pt');cache=load(absolute(cfg.grid_pilot.evaluation_cache));rows=[];rates=[];training_frequency=reference['frequency'];singleton_types=int((training_frequency==1).sum());N=int(training_frequency.sum())
    for split,d in reference['datasets'].items():
        freq=d['frequency'];rates.append({'dataset':d['dataset'],'tokens':len(freq),'unseen_rate':float((freq==0).double().mean()),'singleton_rate':float((freq==1).double().mean()),'Good_Turing_N1':singleton_types,'training_N':N,'Good_Turing_missing_mass':singleton_types/N if split=='id' else None,'status':'computed from reused training/test words'})
    write_csv(output/'coverage.csv',rates)
    runs=[('matched',absolute(cfg.matched.results),[0,1,2,3,4]),('size_grid',absolute(cfg.matched.historical_run),[0,1,2])]
    for run,path,seeds in runs:
        for seed in seeds:
            for split,d in reference['datasets'].items():
                cached=load(path/'test'/f'seed{seed}_{split}.pt');prediction=previous.predict_tokens(cached['logits'],None if d['space']=='native' else reference['mapping']);gold=d['gold'];fb=bins(d['frequency']);heads=cached['heads'];base=list(heads).index('baseline')
                partitions=[('frequency','none',fb,list(cfg.language_checks.frequency_names)),('tail_endpoints','none',torch.where(d['frequency']<=1,0,torch.where(d['frequency']>=32,1,-1)),['freq_0_1','freq_32plus']),('bands','none',torch.where(d['frequency']==0,0,torch.where(d['frequency']<=7,1,2)),['unseen','rare_1_7','frequent_8plus'])]
                for definition,key in [('raw','raw_category'),('labels_at_least_10pct','robust_category')]:
                    category=d[key];group=torch.where(d['frequency']==0,0,torch.where(d['frequency']<=7,torch.where(category==2,2,1),torch.where(category==2,4,3)))
                    partitions.append(('frequency_ambiguity',definition,group,['unseen','rare_unambiguous','rare_ambiguous','frequent_unambiguous','frequent_ambiguous']))
                for partition,definition,group,names in partitions:
                    for subset in ('all_tokens','gold_entity_tokens'):
                        selected=group if subset=='all_tokens' else torch.where(gold!=0,group,-1);c=previous.grouped(prediction,gold,selected,len(names))
                        for h,head in enumerate(heads):
                            for g,name in enumerate(names):rows.append({'run':run,'seed':seed,'arm':head,'dataset':d['dataset'],'partition':partition,'definition':definition,'subset':subset,'bin':name,**metric_row(c,h,g,base)})
                print(f'Cached tails: {run} seed{seed} {d["dataset"]}',flush=True)
    write_csv(output/'per_seed.csv',rows);summary=summarize(rows,['run','arm','dataset','partition','definition','subset','bin'],['accuracy','token_micro_f1','accuracy_gain','token_f1_gain']);write_csv(output/'summary.csv',summary)
    fits=[];fit_rows=[];hyp=[];groups=defaultdict(list)
    for r in summary:
        if r['arm']=='baseline':continue
        if r['partition']=='frequency':groups[r['run'],r['arm'],r['dataset'],r['subset']].append(r)
    datasets={v['dataset']:v for v in reference['datasets'].values()}
    for key,values in groups.items():
        values.sort(key=lambda r:list(cfg.language_checks.frequency_names).index(r['bin']));d=datasets[key[2]];freq=d['frequency'];midpoints=[0.,1.,2.5,5.5,11.5,23.5,float(freq[freq>=32].double().mean())]
        for metric in ('accuracy_gain','token_f1_gain'):
            available=[r for r in values if r[metric+'_mean'] is not None];y=torch.tensor([r[metric+'_mean'] for r in available],dtype=torch.double);fit=isotonic(y);violations=int((y[1:]>y[:-1]).sum())
            fits.append({'run':key[0],'arm':key[1],'dataset':key[2],'subset':key[3],'metric':metric,'fit':'unweighted decreasing isotonic / PAVA','RMSE':float((y-fit).square().mean().sqrt()),'observed_adjacent_increases':violations,'non_monotone_observed':violations>0,'status':'computed descriptive fit'})
            for r,observed,fitted in zip(available,y.tolist(),fit.tolist()):fit_rows.append({'run':key[0],'arm':key[1],'dataset':key[2],'subset':key[3],'metric':metric,'bin':r['bin'],'frequency_representative':midpoints[list(cfg.language_checks.frequency_names).index(r['bin'])],'observed_gain':observed,'fitted_gain':fitted,'residual':observed-fitted})
    write_csv(output/'fits.csv',fits);write_csv(output/'fit_residuals.csv',fit_rows)
    for run,arm,dataset in sorted({(r['run'],r['arm'],r['dataset']) for r in summary if r['arm']!='baseline'}):
        for metric in ('accuracy_gain','token_f1_gain'):
            selected=[r for r in summary if (r['run'],r['arm'],r['dataset'],r['partition'],r['subset'])==(run,arm,dataset,'tail_endpoints','all_tokens')];by={r['bin']:r for r in selected};low=by['freq_0_1'][metric+'_mean'];high=by['freq_32plus'][metric+'_mean'];hyp.append({'run':run,'arm':arm,'dataset':dataset,'metric':metric,'gain_0_1':low,'gain_32plus':high,'tail_exceeds_head':low is not None and high is not None and low>high})
    write_csv(output/'hypothesis_B.csv',hyp)
    reused=read_csv(absolute(cfg.variance_analysis.results)/'summary.csv');unseen=[];associations=[];rates_by={r['dataset']:r['unseen_rate'] for r in rates}
    for r in reused:
        if r['part']=='B' and r['definition']=='raw' and r['category']=='unseen' and r['arm']!='baseline':unseen.append({'run':r['run'],'arm':r['arm'],'dataset':r['dataset'],'unseen_token_rate':rates_by[r['dataset']],'unseen_shapley_contribution':float(r['shapley_contribution_mean']),'unseen_share_total_gain':r['share_of_mean_total_gain'],'status':'contribution reused; unseen rate computed'})
    for run,arm in sorted({(r['run'],r['arm']) for r in unseen}):
        selected=[r for r in unseen if (r['run'],r['arm'])==(run,arm)];x=torch.tensor([r['unseen_token_rate'] for r in selected],dtype=torch.double);y=torch.tensor([r['unseen_shapley_contribution'] for r in selected],dtype=torch.double)
        for rank in (False,True):
            value,reason=correlation(x,y,rank);associations.append({'run':run,'arm':arm,'targets':len(selected),'correlation':'Spearman' if rank else 'Pearson','value':value,'undefined_reason':reason,'interpretation':'descriptive only; four domains; no significance claim'})
    write_csv(output/'unseen_contributions.csv',unseen);write_csv(output/'unseen_domain_associations.csv',associations);return {'rows':len(rows)}


def frequency(cfg,output):
    reference=load(root(cfg)/'prepare'/'reference.pt');raw=load(absolute(cfg.matched.originals_run)/'prepare'/'raw.pt');word=reference['train_word'];V=int(word.max())+1;counts=reference['counts'][:V];n=counts.sum(-1);sums=torch.zeros(V,768,dtype=torch.double);squares=torch.zeros(V,dtype=torch.double)
    for start in range(0,len(word),4096):
        x=raw['features'][start:start+4096,0].double();w=word[start:start+4096];sums.index_add_(0,w,x);squares.index_add_(0,w,x.square().sum(-1))
    dispersion=(squares/n-sums.div(n[:,None]).square().sum(-1)).clamp_min(0);fb=bins(n);rows=[];correlations=[]
    for space,counts in [('native',reference['counts'][:V]),('shared',reference['shared_counts'][:V])]:
        for threshold in (0.,.1):
            cats,_=previous.category(counts,threshold);trim=counts.double();trim=trim if threshold==0 else trim*(trim>=threshold*n[:,None]);p=trim/trim.sum(-1,keepdim=True);entropy=-torch.special.xlogy(p,p).sum(-1)
            for g in range(1,7):
                mask=fb==g;eligible=mask&(n>=2);rows.append({'space':space,'definition':'raw' if threshold==0 else 'labels_at_least_10pct','bin':cfg.language_checks.frequency_names[g],'types':int(mask.sum()),'ambiguous_share':float((cats[mask]==2).double().mean()),'mean_label_entropy_nats':float(entropy[mask].mean()),'dispersion_eligible_types':int(eligible.sum()),'mean_within_word_squared_distance':float(dispersion[eligible].mean()) if eligible.any() else None,'status':'computed from reused words/gold/block12 features; full occurrences, no cap'})
            chosen=[r for r in rows if r['space']==space and r['definition']==('raw' if threshold==0 else 'labels_at_least_10pct')]
            for metric in ('ambiguous_share','mean_label_entropy_nats','mean_within_word_squared_distance'):
                valid=[(i,r) for i,r in enumerate(chosen) if r[metric] is not None];x=torch.tensor([i for i,_ in valid],dtype=torch.double);y=torch.tensor([r[metric] for _,r in valid],dtype=torch.double);corr,reason=correlation(x,y,True);correlations.append({'space':space,'definition':'raw' if threshold==0 else 'labels_at_least_10pct','metric':metric,'frequency_bin_Spearman':corr,'undefined_reason':reason,'strictly_nondecreasing_observed':bool((y[1:]>=y[:-1]).all()),'status':'computed; descriptive 5–6 frequency bins'})
    write_csv(output/'training_frequency.csv',rows);write_csv(output/'frequency_trends.csv',correlations)
    wordnet={'available_offline':False,'reason':'No offline WordNet resource found; no download attempted.'}
    try:
        from nltk.corpus import wordnet as wn
        wn.ensure_loaded();lookup={}
        for word_string,index in reference['lookup'].items():
            if index<V:lookup[index]=len(wn.synsets(word_string.lower()))
        values=torch.tensor([lookup[i] for i in range(V)],dtype=torch.double);ambiguity=(reference['counts'][:V]>0).sum(-1)>1;c1,_=correlation(n.double(),values,True);c2,_=correlation(ambiguity.double(),values,True);wordnet={'available_offline':True,'frequency_vs_synsets_Spearman':c1,'raw_ambiguity_vs_synsets_Spearman':c2,'types':V};write_csv(output/'wordnet.csv',[{'word':w,'frequency':int(n[i]),'synsets':lookup[i]} for w,i in reference['lookup'].items() if i<V])
    except LookupError:pass
    save_json(output/'wordnet.json',wordnet);return {'training_types':V}


def truth(value):return value is True or value=='True'
def verdict(flags):return 'PASS' if all(flags) else 'FAIL' if not any(flags) else 'MIXED'


def report(cfg,output):
    a=read_csv(root(cfg)/'paradigm'/'hypothesis_A.csv');b=read_csv(root(cfg)/'tail'/'hypothesis_B.csv');summary=read_csv(root(cfg)/'tail'/'summary.csv');rows=read_csv(root(cfg)/'tail'/'per_seed.csv');trends=read_csv(root(cfg)/'frequency'/'frequency_trends.csv');training=read_csv(root(cfg)/'frequency'/'training_frequency.csv');metrics_a=read_csv(root(cfg)/'paradigm'/'metrics.csv');coverage=read_csv(root(cfg)/'tail'/'coverage.csv');fit=read_csv(root(cfg)/'tail'/'fits.csv');context_contrasts=[];mention_bands=[];decisions={}
    raw_a=[r for r in a if r['definition']=='raw'];status_A=verdict([truth(r['PASS']) for r in raw_a]);A_pass=sum(truth(r['PASS']) for r in raw_a)
    outcomes_B=[]
    for metric in ('accuracy_gain','token_f1_gain'):
        arms=sorted({(r['run'],r['arm']) for r in b});success=[]
        for run,arm in arms:
            selected=[r for r in b if (r['run'],r['arm'],r['metric'])==(run,arm,metric)];success.append(sum(truth(r['tail_exceeds_head']) for r in selected)>len(selected)/2)
        outcomes_B.append(sum(success)>len(success)/2);decisions[metric]={'arms_passing_majority_targets':sum(success),'arms':len(success),'passes':outcomes_B[-1]}
    status_B=verdict(outcomes_B)
    first=[r for r in trends if r['space']=='native'];trend_pass=all(r['frequency_bin_Spearman'] and float(r['frequency_bin_Spearman'])>0 for r in first)
    selected_arms=sorted({(r['run'],r['arm']) for r in summary if (r['run']=='matched' and '_FLIP' in r['arm']) or (r['run']=='size_grid' and r['arm'] in ('uniform','lambda0.001','lambda0.01','greedy'))})
    for run,arm in selected_arms:
        for dataset in ('fewnerd','conll2003','wnut','wikiann'):
            for definition in ('raw','labels_at_least_10pct'):
                for metric in ('accuracy_gain','token_f1_gain'):
                    selected=[r for r in rows if (r['run'],r['arm'],r['dataset'],r['partition'],r['definition'],r['subset'])==(run,arm,dataset,'frequency_ambiguity',definition,'all_tokens') and r['bin'] in ('frequent_ambiguous','frequent_unambiguous')];diff=[]
                    for seed in sorted({r['seed'] for r in selected}):
                        one={r['bin']:r for r in selected if r['seed']==seed}
                        if one['frequent_ambiguous'][metric] and one['frequent_unambiguous'][metric]:diff.append(float(one['frequent_ambiguous'][metric])-float(one['frequent_unambiguous'][metric]))
                    values=torch.tensor(diff,dtype=torch.double);context_contrasts.append({'run':run,'arm':arm,'dataset':dataset,'definition':definition,'metric':metric,'ambiguous_minus_unambiguous_gain':float(values.mean()) if len(values) else None,'seed_se':float(values.std()/math.sqrt(len(values))) if len(values)>1 else None,'seeds':len(values),'ambiguous_gain_larger':bool(values.mean()>0) if len(values) else None})
    counts_C=[]
    for definition in ('raw','labels_at_least_10pct'):
        for metric in ('accuracy_gain','token_f1_gain'):
            selected=[r for r in context_contrasts if r['definition']==definition and r['metric']==metric];count=sum(r['ambiguous_gain_larger'] is True for r in selected);counts_C.append({'definition':definition,'metric':metric,'comparisons':len(selected),'ambiguous_better':count,'passes_majority':count>len(selected)/2})
    context_pass=all(r['passes_majority'] for r in counts_C)
    for dataset in ('fewnerd','conll2003','wnut','wikiann'):
        for metric in ('accuracy_gain','token_f1_gain'):
            selected=[r for r in summary if (r['run'],r['arm'],r['dataset'],r['partition'],r['subset'])==('matched','MENTION',dataset,'bands','all_tokens')];available=[r for r in selected if r[metric+'_mean']];winner=max(available,key=lambda r:float(r[metric+'_mean']));mention_bands.append({'dataset':dataset,'metric':metric,'best_band':winner['bin'],'best_gain':float(winner[metric+'_mean']),'best_is_unseen_or_rare':winner['bin']!='frequent_8plus'})
    mention_pass=sum(r['best_is_unseen_or_rare'] for r in mention_bands if r['metric']=='token_f1_gain')>2;status_C=verdict([trend_pass,context_pass,mention_pass]);write_csv(output/'hypothesis_C_contrasts.csv',context_contrasts);write_csv(output/'hypothesis_C_counts.csv',counts_C);write_csv(output/'MENTION_bands.csv',mention_bands)
    save_json(output/'decisions.json',{'A':{'status':status_A,'datasets_passing':A_pass,'datasets':len(raw_a)},'B':{'status':status_B,**decisions},'C':{'status':status_C,'frequency_trends_positive':trend_pass,'context_gain_condition':context_pass,'mention_band_condition':mention_pass},'interpretation':'Prespecified descriptive criteria, not significance tests. A raw definition primary, robust separate. C context set is matched FLIP/full-FLIP and size-grid uniform/lambda0.001/lambda0.01/greedy; full tables contain every arm. C compares a strict majority of arm-target cells for each metric/ambiguity definition; MENTION token-F1 is primary with accuracy sensitivity.'})
    lines=[f'H_A: {status_A} — {A_pass}/{len(raw_a)} test sets meet all three primary paradigm criteria.',f'H_B: {status_B} — tail-to-head endpoint rule: '+', '.join(f"{metric}: {v['arms_passing_majority_targets']}/{v['arms']} arms" for metric,v in decisions.items())+'.',f'H_C: {status_C} — rising trends={trend_pass}, frequent-ambiguous gain condition={context_pass}, MENTION unseen/rare band condition={mention_pass}.','',
        '# Language checks','',
        'Every new statistic and fit below is **computed**; student logits/features, scorer-A predictions where available, word metadata and prior Shapley contributions are **reused**. The only new model forward passes are frozen BERT MLM calls for Part A. No augmentation generation, student training, new annotations or GPU. The hypothesis rules are saved in config before execution; these are descriptive checks, not p-values.','',
        '## A. Paradigm','',
        'At most 8000 natural focal tokens per target, fixed seed0, without replacement; frequency × raw/robust ambiguity stratification. Sample frequencies therefore are not self-weighting population rates. Each focal span is replaced by one MASK. Whole-word candidates are tokens not starting ##, excluding tokenizer specials; punctuation is retained. Vocabulary entries absent from original training have no vote. Their mass is reported rather than assigned a label. Native and shared-space majority labels follow variance_decomposition conventions.','',
        '| Target | Full paradigm accuracy | Word identity accuracy | Top50 accuracy | Cached A accuracy / available n | Vote mass full / top50 |','|---|---:|---:|---:|---:|---:|']
    for dataset in ('fewnerd','conll2003','wnut','wikiann'):
        selected={r['model']:r for r in metrics_a if (r['dataset'],r['definition'],r['partition'])==(dataset,'raw','all')}
        def percent(row,key):return f"{100*float(row[key]):.2f}%" if row[key] else 'unavailable'
        lines.append('| '+dataset+' | '+percent(selected['paradigm'],'accuracy')+' | '+percent(selected['identity'],'accuracy')+' | '+percent(selected['top50'],'accuracy')+' | '+percent(selected['A'],'accuracy')+' / '+selected['A']['available_n']+' | '+percent(selected['paradigm'],'mean_vote_mass')+' / '+percent(selected['top50'],'mean_vote_mass')+' |')
    lines+=['','A-only columns use the subset with actual cached A predictions; do not interpret unavailable rows as zeros or student baselines as A. No-vote top50 cases abstain: accuracy counts them wrong, entity micro-F1 counts entity false negatives and does not invent a positive class. Coverage is relative to the whole-word-restricted MLM distribution; full-vocabulary masses are also saved. The primary full-vocabulary paradigm and top50 results remain separate.','',
        '| Target / ambiguity | Unseen Δaccuracy | Frequent-unambiguous gap | Closest at 32+? | Ambiguous share: sample / paradigm errors | All criteria? |','|---|---:|---:|---|---:|---|']
    for r in a:lines.append(f"| {r['dataset']} / {r['definition']} | {100*float(r['unseen_accuracy_gain']):+.3f} pp | {100*float(r['frequent_unambiguous_abs_accuracy_gap']):.3f} pp | {r['frequent_unambiguous_closest']} | {100*float(r['ambiguous_share_sample']):.2f}% / {100*float(r['ambiguous_share_paradigm_errors']):.2f}% | {r['PASS']} |")
    lines+=['','The unseen criterion uses accuracy; the frequent-unambiguous criterion compares the absolute paradigm/identity accuracy gap with other seen-unambiguous frequency bins. Error overrepresentation compares ambiguous shares in all paradigm errors versus the whole sampled set. Primary decisions use raw ambiguity; the 10% version is displayed separately. Per-bin/non-O F1 results: ../paradigm/metrics.csv. Complete failure 2×2 counts and ambiguous fractions: failure_cells.csv; label confusions: confusions.csv; up to 20 seeded sentence examples per cell/target: failure_examples.csv.','',
        '## B. Tail coverage and gains','',
        '| Target | Observed unseen-token rate | Test singleton rate | Good–Turing N1/N (in-domain only) |','|---|---:|---:|---:|']
    for r in coverage:lines.append(f"| {r['dataset']} | {100*float(r['unseen_rate']):.3f}% | {100*float(r['singleton_rate']):.3f}% | "+(f"{100*float(r['Good_Turing_missing_mass']):.3f}%" if r['Good_Turing_missing_mass'] else 'not in-domain')+' |')
    lines+=['','Good–Turing is the empirical singleton-type count divided by 60321 training tokens, not a domain-shift correction. Per-bin gains reuse exactly the original matched/size-grid logits, each against its own baseline; all arms, both all-token and gold-entity-token restrictions are retained with paired mean/seed SE. Gold-entity-only precision/F1 excludes O-gold tokens by definition and must not be interpreted as overall entity precision.','',
        '| Endpoint metric | Arms with larger 0–1 than 32+ gain on >2 targets | Strict majority of arms? |','|---|---:|---|']
    for metric,r in decisions.items():lines.append(f"| {metric} | {r['arms_passing_majority_targets']} / {r['arms']} | {r['passes']} |")
    primary_fit=[r for r in fit if r['subset']=='all_tokens' and r['metric']=='token_f1_gain'];nonmono=sum(truth(r['non_monotone_observed']) for r in primary_fit)
    lines +=['',f'{nonmono}/{len(primary_fit)} all-token arm/target token-F1 curves have at least one observed adjacent-bin increase. Endpoint concentration and smoothly decreasing gains are different claims. Unweighted decreasing isotonic/PAVA fits pool violations and report residuals; they do not assert that the noisy observed curves are monotone. Closed-bin representatives are midpoints; 32+ uses that target’s observed mean training frequency. No fitted hyperparameter or checkpoint is selected from these descriptive curves.','',
        'Per-seed and summary data: ../tail/per_seed.csv and summary.csv. Fits/violations: fits.csv; fitted values and residuals: fit_residuals.csv. Prior unseen Shapley contributions are reused unchanged from variance_decomposition, joined to computed unseen rates in unseen_contributions.csv. Four-target Pearson/Spearman values in unseen_domain_associations.csv are descriptive only; no significance claims.','',
        '## C. Ambiguity by frequency','',
        '| Space / definition | Metric | Frequency Spearman | Nondecreasing observed bins? |','|---|---|---:|---|']
    for r in trends:lines.append(f"| {r['space']} / {r['definition']} | {r['metric']} | {float(r['frequency_bin_Spearman']):+.3f} | {r['strictly_nondecreasing_observed']} |")
    lines+=['','Training statistics are type-weighted in six nonempty training frequency bands. Unseen is undefined as a training-word-type bin, not assigned zero entropy or dispersion. Entropy uses native labels (shared-space sensitivity also provided); robust entropy drops <10% labels and renormalizes. Dispersion is the mean per-word squared distance to its own cached block12 centroid using ALL original occurrences, then averaged across eligible types (frequency>=2), not the prior capped sample. A singleton dispersion is unestimated. These are proxies/descriptors, not WordNet senses or proof of semantic mechanisms.','',
        '| Ambiguity definition | Gain metric | Frequent ambiguous gain exceeds frequent unambiguous | Majority? |','|---|---|---:|---|']
    for r in counts_C:lines.append(f"| {r['definition']} | {r['metric']} | {r['ambiguous_better']} / {r['comparisons']} | {r['passes_majority']} |")
    lines+=['','Frequent means 8+. Comparisons use matched FLIP/full-FLIP plus historical uniform/lambda0.001/lambda0.01/greedy; all other arm/band tables are retained too. These counts are across related configurations and domains, not independent experiments. Per-seed paired contrasts and SEs: hypothesis_C_contrasts.csv.','',
        '| MENTION target | Largest token-F1 gain band | Largest accuracy gain band |','|---|---|---|']
    for dataset in ('fewnerd','conll2003','wnut','wikiann'):
        one={r['metric']:r for r in mention_bands if r['dataset']==dataset};lines.append(f"| {dataset} | {one['token_f1_gain']['best_band']} ({100*one['token_f1_gain']['best_gain']:+.3f} pp) | {one['accuracy_gain']['best_band']} ({100*one['accuracy_gain']['best_gain']:+.3f} pp) |")
    wordnet=json.loads((root(cfg)/'frequency'/'wordnet.json').read_text());lines+=['','Optional WordNet: '+json.dumps(wordnet)+'. No network access or download was attempted.','',
        '## Deviations / unavailable checks','',
        '- Actual scorer-A test predictions are cached only for the original filtered native natural-pair occurrences. Other sampled native tokens and all transfer A baselines are marked unavailable. Full-data student checkpoints were checked and are not identical to A; no proxy substitution or additional scorer/head inference was performed.',
        '- All requested student test logits and original block12 features were available. Metadata/tokenizer alignment is not an encoder pass. No additional feature layers were computed.',
        '- Scores are computed on a saved stratified 8000-token-or-smaller sample per target in A; no unreported weighting turns that into a random-population accuracy estimate. B/C use the full cached test populations.',
        '- The decreasing fit is isotonic/PAVA rather than the optional example rational function. It is a descriptive shape-constrained fit, not student training. Endpoint decisions and strict monotonicity flags are separately reported.',
        '- Majority/ambiguity are recomputed in each evaluation label space as in variance_decomposition; the native lexicon remains the common source. Whole-word vocabulary means not-## and not-special, without punctuation or plausibility filtering.',
        '- Top50 no-vote examples abstain rather than invent a label. Empty statistical slices are unavailable. Example cells with fewer than 20 entries contain all available entries.',
        '- H_A is adjudicated per dataset and combined as MIXED when domains disagree. H_B uses literal strict majorities of arms and targets, with both gain metrics checked. H_C positive rank trends, majority frequent-ambiguous gain contrasts, and MENTION token-F1 band concentration must all pass for PASS; mixed components yield MIXED. These are explicit operational/descriptive checks, not causal or significance tests.','',
        'Primary source definitions, seeds and runtime configs: config.yaml in run/stage folders. All newly computed tables include computed/reused status; cached contributions and A predictions are labelled reused. Labels of generated examples are not analyzed as known gold here.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n');return {'A':status_A,'B':status_B,'C':status_C}


def main(cfg):
    OmegaConf.resolve(cfg);configure(cfg);cfg.forge.store=str(absolute(cfg.forge.store));tracking=start_run(cfg);folder=root(cfg);folder.mkdir(parents=True,exist_ok=True);stage=cfg.stage.experiment;output=folder/stage;output.mkdir(exist_ok=True);OmegaConf.save(cfg,folder/'config.yaml');OmegaConf.save(cfg,output/'config.yaml');start=time.monotonic();print(f'Language checks {stage}: {output} ({cfg.runtime.device})',flush=True)
    if not (folder/'timing.json').exists():save_json(folder/'timing.json',{'started_unix':time.time()})
    result={'prepare':prepare,'frequency':frequency,'tail':tail,'paradigm':paradigm,'report':report}[stage](cfg,output);save_json(output/'runtime.json',{'seconds':time.monotonic()-start});tracking.finish(result)
    if stage=='report':
        import shutil
        shutil.copyfile(output/'RESULTS.md',folder/'RESULTS.md');timing=json.loads((folder/'timing.json').read_text());timing['ended_unix']=time.time();timing['wall_seconds']=timing['ended_unix']-timing['started_unix'];save_json(folder/'timing.json',timing)
    return 0


if __name__=='__main__':
    stages=[None] if any(a.startswith('stage.experiment=') for a in sys.argv[1:]) else ['prepare','frequency','tail','paradigm','report']
    for stage in stages:
        code=run('lion.language_checks',sys.argv[1:]+([] if stage is None else [f'stage.experiment={stage}']),main_module='run',config_dir=str(ROOT/'conf'),config_name='language_checks')
        if code:sys.exit(code)
