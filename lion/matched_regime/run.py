"""One fixed original-data regime; reuse historical pools, match new training cells."""
import json,sys,time
from collections import defaultdict
import torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from forge import run,start_run
from omegaconf import OmegaConf
from ..runtime import configure
from ..diagnostics.common import ROOT,absolute,save_json,write_csv,read_csv
from ..certification import data as source,selector
from ..partition.train import load_model
from ..encoder import load_masked_encoder
from ..proposal_mechanisms import generate as mechanisms,train_pilot as students
from ..guided_refinement import generate as guided,size_grid
from ..independent_judge import representations
from ..label_change.fewnerd_diagnostic import parquet
from . import data,certificate,evaluation


def root(cfg):return data.root(cfg)
def load(path):return data.load(path)


def judges(cfg):return (*representations.load_encoder(cfg),load_model(absolute(cfg.matched.judge_run)/'judge_train'/'ner',cfg).requires_grad_(False))


def historical_judge(cfg,output):
    model,tokenizer,b=judges(cfg);counts=[]
    for path in sorted((root(cfg)/'historical_pools').glob('*.pt')):
        pool=load(path);pred=data.judge(cfg,pool,torch.arange(len(pool['y'])),model,tokenizer,b);torch.save(pred,output/path.name);counts.append({'arm':path.stem,'accepted':len(pred),'agreed':int((pred==pool['y']).sum())});write_csv(output/'counts.csv',counts)
    return {'arms':len(counts)}


def historical_certificates(cfg,output):
    reference,lookup=certificate.natural(cfg);summary=[];groups=[];f1=read_csv(absolute(cfg.matched.historical_run)/'test'/'summary.csv');joined=[]
    for path in sorted((root(cfg)/'historical_pools').glob('*.pt')):
        pool=load(path);pred=load(root(cfg)/'historical_judge'/path.name);s,g=certificate.evaluate(cfg,path.stem,pool,torch.arange(len(pred)),pred,reference,lookup,output);summary+=s;groups+=g
        for row in s:
            for score in f1:
                if score['head']==path.stem:joined.append({**row,**score})
    certificate.save(cfg,output,summary,groups);write_csv(output/'certificates_with_existing_F1.csv',joined);return {'certificates':len(summary)}


def generate_stage(cfg,output):
    bank,rows=data.anchors(cfg);meta=load(root(cfg)/'prepare'/'sources_targets.pt');encoder=load_masked_encoder(cfg);a=load_model(source.paths(cfg)/'scorer'/'ner',cfg).requires_grad_(False);timings=[]
    for design in cfg.matched.designs:
        if design=='uniform':timing=guided.arm(cfg,bank,rows,meta,encoder,a,'uniform',cfg.mechanisms.draws,output/design)
        elif design=='sentence70':timing=size_grid.fraction_arm(cfg,bank,rows,encoder,a,design,output/design)
        else:timing=mechanisms.arm(cfg,bank,rows,encoder,a,'local70',output/design)
        timings.append(timing);save_json(output/'timing.json',timings)
    return {'designs':len(timings)}


def new_pools(cfg,output):
    result=[data.collect_new(cfg,design,output) for design in cfg.matched.designs];save_json(output/'counts.json',result);return {'designs':len(result)}


def flip(pool,cfg):return pool['changed']&(pool['a_prediction']!=pool['source_gold'])&(pool['a_confidence']>=cfg.joint.tau)


def judge_flip(cfg,output):
    model,tokenizer,b=judges(cfg);counts=[]
    for design in cfg.matched.designs:
        pool=load(root(cfg)/'new_pools'/f'{design}.pt');indices=flip(pool,cfg).nonzero().flatten();prediction=torch.full((len(pool['y']),),-1,dtype=torch.long);prediction[indices]=data.judge(cfg,pool,indices,model,tokenizer,b);torch.save(prediction,output/f'{design}.pt');counts.append({'design':design,'ALL':len(prediction),'FLIP':len(indices),'AGREED':int((prediction[indices]==pool['y'][indices]).sum())});write_csv(output/'counts.csv',counts)
    return {'designs':len(counts)}


@torch.no_grad()
def mention(cfg,n,output):
    bank,originals=data.originals(cfg);labels=bank['labels']['ner'];outside=labels.index('O');native=pq.read_table(parquet('train'),columns=['tokens']).to_pylist();spans=[];by_type=defaultdict(list);eligible_sources=[];source_prediction=load(root(cfg)/'prepare'/'source_predictions.pt');word_offset=0
    for row in originals:
        tokens=native[int(row['source_id'].split('/')[-1])]['tokens'][:len(row['focals'])];gold=row['gold']['ner'];i=0
        while i<len(gold):
            if int(gold[i])==outside:i+=1;continue
            end=i+1
            while end<len(gold) and gold[end]==gold[i]:end+=1
            index=len(spans);spans.append((row,tokens,i,end,int(gold[i])));by_type[int(gold[i])].append(index)
            if torch.equal(source_prediction[word_offset+i:word_offset+end],gold[i:end]):eligible_sources.append(index)
            i=end
        word_offset+=len(gold)
    rng=torch.Generator().manual_seed(cfg.matched.mention_seed);tokenizer=AutoTokenizer.from_pretrained(cfg.encoder.name,use_fast=True,local_files_only=True);records=[];metadata=[];total=0;skipped=0
    while total<n:
        index=eligible_sources[int(torch.randint(len(eligible_sources),(1,),generator=rng))];row,tokens,start,end,label=spans[index];options=by_type[label]
        source_position=options.index(index);pick=int(torch.randint(len(options)-1,(1,),generator=rng));donor_index=options[pick+int(pick>=source_position)];donor=spans[donor_index];replacement=donor[1][donor[2]:donor[3]];new_tokens=tokens[:start]+replacement+tokens[end:]
        encoded=tokenizer(new_tokens,is_split_into_words=True,add_special_tokens=True,truncation=True,max_length=cfg.encoder.max_length);wid=encoded.word_ids();kept=len({w for w in wid if w is not None})
        if start+len(replacement)>kept:skipped+=1;continue
        gold=torch.full((kept,),-1,dtype=torch.long);gold[start:start+len(replacement)]=label;records.append({'ids':torch.tensor(encoded['input_ids']),'word_ids':torch.tensor([w if w is not None else -1 for w in wid]),'focals':torch.arange(kept),'gold':{'ner':gold},'sentence':len(records)});metadata.append((row['source_id'],new_tokens,start,start+len(replacement),label));total+=len(replacement)
    encoder=load_masked_encoder(cfg);features=source.full_features(encoder,{'mention':records},cfg)['mention'];a=load_model(source.paths(cfg)/'scorer'/'ner',cfg).requires_grad_(False);pool_records=[];selected=[];offset=0
    for r,(source_id,tokens,start,end,label) in zip(records,metadata):
        for f in range(start,end):pool_records.append({'ids':r['ids'],'word_ids':r['word_ids'],'focal':f,'source_id':source_id,'word':tokens[f],'y':label,'gold':label,'draw':0,'anchor':len(pool_records)});selected.append(offset+f)
        offset+=len(r['focals'])
    permutation=torch.randperm(len(selected),generator=rng)[:n];selected=torch.tensor(selected)[permutation];pool_records=[pool_records[i] for i in permutation.tolist()];features=features[selected];p=torch.cat([selector.scorer_log_probs(a,x.to(cfg.runtime.device),cfg).exp().cpu() for x in features.split(1024)]);confidence,prediction=p.max(-1)
    for r,pred,conf in zip(pool_records,prediction.tolist(),confidence.tolist()):r['a_prediction']=pred;r['confidence']=conf
    pool=data.pack(pool_records,features);torch.save(pool,output/'MENTION.pt');save_json(output/'mention_method.json',{'reference':'Dai & Adel (2020), https://aclanthology.org/2020.coling-main.343/','method':'One uniformly sampled entity mention replaced per constructed sentence, donor uniformly sampled among training mention occurrences of the same coarse type, retaining identical-string draws. This is the task-specified single-mention version, not the paper full multi-mention Bernoulli grid.','donor_pool':'Only 2400 originals; coarse IO consecutive same-type spans follow existing dataset convention, so adjacent same-type mentions without a boundary are merged.','loss':'Exactly N labelled replaced-token focal features sampled without replacement from complete generated mention-token pool; other positions unlabelled, including any nonselected replaced tokens.','requested_N':n,'constructed_mentions':len(records),'replaced_tokens_before_subsample':total,'truncation_skips':skipped})
    return pool


def match(cfg,output):
    counts=read_csv(root(cfg)/'judge_flip'/'counts.csv');n=min(int(r['AGREED']) for r in counts)
    if n==0:raise ValueError('An AGREED pool is empty; no nonzero matched-size grid can be formed')
    rng=torch.Generator().manual_seed(cfg.matched.subsample_seed);selections={};rows=[]
    for design in cfg.matched.designs:
        pool=load(root(cfg)/'new_pools'/f'{design}.pt');b=load(root(cfg)/'judge_flip'/f'{design}.pt');f=flip(pool,cfg);sets={'ALL':torch.arange(len(b)),'FLIP':f.nonzero().flatten(),'AGREED':(f&(b==pool['y'])).nonzero().flatten()}
        for variant,indices in sets.items():
            chosen=indices[torch.randperm(len(indices),generator=rng)[:n]];name=f'{design}_{variant}';selections[name]={'parent':design,'indices':chosen.tolist()};save_training(cfg,name,pool,chosen);rows.append({'head':name,'natural_size':len(indices),'training_size':len(chosen),'matched':True})
        name=f'{design}_FLIP_full';indices=sets['FLIP'];selections[name]={'parent':design,'indices':indices.tolist()};save_training(cfg,name,pool,indices);rows.append({'head':name,'natural_size':len(indices),'training_size':len(indices),'matched':False})
    pool=mention(cfg,n,root(cfg)/'new_pools');indices=torch.arange(n);selections['MENTION']={'parent':'MENTION','indices':indices.tolist()};save_training(cfg,'MENTION',pool,indices);rows.append({'head':'MENTION','natural_size':n,'training_size':n,'matched':True})
    save_json(output/'selections.json',selections);save_json(output/'matched_size.json',{'N':n,'rule':'Smallest natural AGREED pool across three designs; fixed-seed subsampling without replacement for ALL/FLIP/AGREED, exactly N supervised mention tokens.'});write_csv(output/'sizes.csv',rows);return {'N':n,'students':1+len(rows)}


def save_training(cfg,name,pool,indices):
    folder=root(cfg)/'generate'/name;folder.mkdir(parents=True,exist_ok=True);torch.save({'features':pool['features'][indices],'y':pool['y'][indices],'parent_pool_index':indices},folder/'accepted_features_draw0.pt')


def judge_matched(cfg,output):
    model,tokenizer,b=judges(cfg);selection=json.loads((root(cfg)/'match'/'selections.json').read_text());names=list(cfg.matched.designs)+['MENTION']
    for name in names:
        pool=load(root(cfg)/'new_pools'/f'{name}.pt');known=load(root(cfg)/'judge_flip'/f'{name}.pt').clone() if name!='MENTION' else torch.full((len(pool['y']),),-1,dtype=torch.long)
        wanted=torch.tensor(sorted({i for r in selection.values() if r['parent']==name for i in r['indices']}));missing=wanted[known[wanted]<0];known[missing]=data.judge(cfg,pool,missing,model,tokenizer,b);torch.save(known,output/f'{name}.pt')
    return {'parents':len(names)}


def new_certificates(cfg,output):
    reference,lookup=certificate.natural(cfg);selection=json.loads((root(cfg)/'match'/'selections.json').read_text());summary=[];groups=[]
    for name,r in selection.items():
        pool=load(root(cfg)/'new_pools'/f"{r['parent']}.pt");indices=torch.tensor(r['indices']);b=load(root(cfg)/'judge_matched'/f"{r['parent']}.pt")[indices];s,g=certificate.evaluate(cfg,name,pool,indices,b,reference,lookup,output);summary+=s;groups+=g
    certificate.save(cfg,output,summary,groups);return {'certificates':len(summary)}


def report(cfg,output):
    old=read_csv(root(cfg)/'historical_certificates'/'certificates.csv');new=read_csv(root(cfg)/'new_certificates'/'certificates.csv');scores=read_csv(root(cfg)/'test'/'paired_summary.csv');pairs=read_csv(root(cfg)/'pairs'/'summary.csv');sizes=read_csv(root(cfg)/'match'/'sizes.csv');joined=[]
    for r in new:
        for s in scores:
            if s['head']==r['pool']:joined.append({**r,**s})
    write_csv(output/'matched_comparison.csv',joined);write_csv(output/'historical_comparison.csv',read_csv(root(cfg)/'historical_certificates'/'certificates_with_existing_F1.csv'));write_csv(output/'ambiguous_pairs.csv',pairs)
    n=json.loads((root(cfg)/'match'/'matched_size.json').read_text())['N'];lines=['# One-regime augmentation comparison','',f'Fixed 2400 originals (60321 gold words); A/G BERT-base-cased, reused ELECTRA B; exact independent_judge natural focal sample and predictions reused. CPU-only. New shared 40000 source-correct word-occurrence anchors × five attempts per design. Matched augmentation size N={n}. Seeds 0/1/2/3/4; raw native-dev token macro-F1 checkpoint selection, no test-based selection.','',
        'Majority labels use exactly the 2400 originals, case-sensitive, fixed label-order ties. refined_departure is the requested majority→A grouping; label_only_departure holds natural rows fixed; label_only_original reproduces the old high-confidence-only reference. Generated returns to majority are retained with unsupported-group w=1. Certificates concern the A=B=assigned subset, not every ALL/FLIP training item. For MENTION that subset can be smaller than A=B.','',
        'Natural units are one focal per sentence. Multiple attempts/anchors and outputs share source sentences; sentence counts and repeats are reported. Token-count CP mass calculations are nominal when this dependence is not justified; no unconditional 95% claim is made. No false independence of A/B errors is assumed.','',
        '| Historical arm | Accepted | A–B agreement | Refined lower δ=0 | Old-label lower δ=0 | Unsupported agreed mass |','|---|---:|---:|---:|---:|---:|']
    for r in old:
        if r['scheme']!='refined_departure':continue
        previous=next(s for s in old if s['pool']==r['pool'] and s['scheme']=='label_only_original');lines.append(f"| {r['pool']} | {r['pool_size']} | {100*float(r['agreement_AB']):.2f}% | {100*float(r['certificate_delta0']):.2f}% | {100*float(previous['certificate_delta0']):.2f}% | {100*float(r['unsupported_agreed_mass']):.2f}% |")
    lines+=['','| Student | Pool size | Few-NERD native ΔF1 | CoNLL ΔF1 | WNUT ΔF1 | WikiANN ΔF1 |','|---|---:|---:|---:|---:|---:|']
    for r in sizes:
        values=[next(s for s in scores if s['head']==r['head'] and (s['target'],s['space'],s['metric'])==(target,space,'span_f1')) for target,space in [('fewnerd','native'),('conll2003','shared_ner'),('wnut','shared_ner'),('wikiann','shared_ner')]]
        lines.append('| '+r['head']+' | '+r['training_size']+' | '+' | '.join(f"{100*float(s['paired_mean']):+.3f} ± {100*float(s['paired_se']):.3f}" for s in values)+' |')
    lines+=['','Cells are paired mean ± seed SE in percentage points. Absolute F1/P/R, seed SDs and paired SD/SE: ../test/paired_summary.csv; per-label F1/P/R: ../test/per_label_per_seed.csv and per_label_summary.csv. Historical size-grid seqeval results were computed from cached test logits, with existing three seeds (not represented as five). Natural-pair outcomes include per-occurrence accuracy on participating occurrences and exact pair-weighted both-labels-correct rates, preserving the original held-out occurrence construction.','',
        '## Reuse and deviations','',
        '- Historical accepted token IDs and focal features were all present; no historical regeneration or scorer retraining. Recovery counts/mismatches: ../historical_pools/reuse.json. Historical pools retain their original source-correct post-filter and attempts; only accepted outputs are reused in Part 1.',
        '- Part 2 cannot reuse historical generation: the old anchor set contains incorrect-source anchors, and ALL must now exclude those before any generation. Three new designs therefore share a new eligible occurrence-uniform anchor stream. Existing generation functions are reused; uniform uses final-only A scoring and unchanged finite-pool sampling.',
        '- B checkpoint and all natural A/B calibration predictions/focal identities are reused; only new/generated accepted and selected ALL/MENTION inputs receive fresh B inference.',
        '- Mention replacement uses one same-type training-mention donor per constructed sentence, retaining identical-string draws; coarse IO same-type consecutive spans follow the existing convention. Exactly N replaced-token focal features are selected without replacement. This adapts the [Dai & Adel (2020)](https://aclanthology.org/2020.coling-main.343/) mention operation to the specified focal-loss budget, not their full Bernoulli/multi-mention grid.',
        '- Existing students cannot replace new five-seed heads because the source eligibility, matched pools and MENTION/full-FLIP cells differ. Their test logits are reused for old per-label metrics and ambiguous pairs; no encoder/test-set re-encoding is needed.',
        '- Nominal CP coverage and group-support limitations are explicit in certificate method.json files. Pointwise 95% per pool/scheme is not simultaneous selection over all schemes/pools. Matched ALL/FLIP/AGREED have identical maximum original-gold and augmented token counts; full-FLIP cells have natural-sized mixtures. Selected checkpoint epochs can differ, per unchanged recipe.',
        '- Full config, seeds, source eligibility, majority pool and calibration reuse are saved; no external notes were edited.','',
        'Tables: historical_comparison.csv; matched_comparison.csv; ambiguous_pairs.csv. Full group counts, agreement, flip composition and slack breakdowns are in ../historical_certificates and ../new_certificates.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n');return {'N':n,'matched_students':len(sizes)+1}


def main(cfg):
    OmegaConf.resolve(cfg);configure(cfg);cfg.forge.store=str(absolute(cfg.forge.store));tracking=start_run(cfg);folder=root(cfg);folder.mkdir(parents=True,exist_ok=True);stage=cfg.stage.experiment;output=folder/stage;output.mkdir(exist_ok=True);OmegaConf.save(cfg,folder/'config.yaml');OmegaConf.save(cfg,output/'config.yaml')
    if not (folder/'timing.json').exists():save_json(folder/'timing.json',{'started_unix':time.time()})
    started=time.monotonic();print(f'Matched regime {stage}: {output} ({cfg.runtime.device})',flush=True)
    operation={'prepare':data.prepare,'historical_pools':data.historical_pools,'historical_judge':historical_judge,'historical_certificates':historical_certificates,'new_generate':generate_stage,'new_pools':new_pools,'judge_flip':judge_flip,'match':match,'judge_matched':judge_matched,'new_certificates':new_certificates,'train':students.train,'test':evaluation.new,'old_evaluation':evaluation.old,'pairs':evaluation.pairs,'report':report}[stage]
    result=operation(cfg,output);save_json(output/'runtime.json',{'seconds':time.monotonic()-started});tracking.finish(result)
    if stage=='report':
        timing=json.loads((folder/'timing.json').read_text());timing['ended_unix']=time.time();timing['wall_seconds']=timing['ended_unix']-timing['started_unix'];save_json(folder/'timing.json',timing)
    return 0


if __name__=='__main__':
    stages=[None] if any(a.startswith('stage.experiment=') for a in sys.argv[1:]) else ['prepare','historical_pools','historical_judge','historical_certificates','new_generate','new_pools','judge_flip','match','judge_matched','new_certificates','train','test','old_evaluation','pairs','report']
    for stage in stages:
        code=run('lion.matched_regime',sys.argv[1:]+([] if stage is None else [f'stage.experiment={stage}']),main_module='run',config_dir=str(ROOT/'conf'),config_name='matched_regime')
        if code:sys.exit(code)
