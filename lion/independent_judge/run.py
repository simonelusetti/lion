"""CPU uniform-refinement experiment with an independent-family agreement judge."""
import json,math,sys,time
import torch
from scipy.stats import beta
from transformers import AutoTokenizer
from forge import run,start_run
from omegaconf import OmegaConf
from ..runtime import configure
from ..diagnostics.common import ROOT,absolute,save_json,write_csv,read_csv
from ..certification import data,selector
from ..partition.train import load_model,fit,predict
from ..label_change.data import sampled_rows
from ..proposal_mechanisms import train_pilot as students
from ..guided_refinement import generate
from ..guided_refinement.binwise import minimize
from ..metrics import bio,shared_label
from ..labels import LABELS
from . import representations


def root(cfg):return absolute(cfg.joint.results)
def load(path):return torch.load(path,mmap=True,weights_only=False)
def task(cfg):return cfg.joint.task


def words(row,tokenizer,ids=None):
    ids=row['ids'] if ids is None else ids
    return [tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids[row['word_ids']==w].tolist())) for w in range(len(row['focals']))]


def sources(cfg):
    bank=data.load(cfg);rows=[r for r in selector.raw_rows(cfg,bank) if r['fold']=='fit']
    return bank,sampled_rows(rows,cfg.joint.proposal_count,cfg.joint.source_seed)


def readout(records,features):
    return {'features':features[:,None],'y':torch.cat([r['gold'] for r in records]),'sentence':torch.repeat_interleave(torch.arange(len(records)),torch.tensor([len(r['focals']) for r in records]))}


@torch.no_grad()
def prepare(cfg,output):
    bank=data.load(cfg);rows=selector.raw_rows(cfg,bank);fitted=[r for r in rows if r['fold']=='fit'];validation=bank['sets']['validation'];labels=bank['labels'][task(cfg)]
    tokenizer=AutoTokenizer.from_pretrained(cfg.encoder.name,local_files_only=True)
    raw=data.readout_data(fitted,torch.cat([r['hidden'] for r in fitted]),task(cfg));torch.save(raw,output/'raw.pt')
    native={}
    for name,part in [('train',fitted),('validation',validation)]:
        native[name]=[{'words':words(r,tokenizer),'focals':r['focals'],'gold':r['gold'][task(cfg)],'source_id':r['source_id']} for r in part]
    torch.save(native,output/'judge_native.pt')
    # Fix a disjoint, one-focal-per-sentence natural calibration sample before B fitting.
    excluded={tuple(r['ids'].tolist()) for r in [*fitted,*validation]};seen=set();cal=[];a_features=[]
    rng=torch.Generator().manual_seed(cfg.joint.calibration_seed)
    for r in rows:
        if r['fold']!='calibration':continue
        key=tuple(r['ids'].tolist())
        if key in excluded or key in seen:continue
        seen.add(key);f=int(torch.randint(len(r['focals']),(1,),generator=rng));cal.append({'words':words(r,tokenizer),'focals':torch.tensor([f]),'gold':r['gold'][task(cfg)][f:f+1].clone(),'source_id':r['source_id']});a_features.append(r['hidden'][f].clone())
    torch.save(cal,output/'calibration.pt');a=load_model(data.paths(cfg)/'scorer'/task(cfg),cfg).requires_grad_(False)
    p=torch.cat([selector.scorer_log_probs(a,x.to(cfg.runtime.device),cfg).exp().cpu() for x in torch.stack(a_features).split(1024)])
    torch.save({'probability':p,'gold':torch.cat([r['gold'] for r in cal]),'source_id':[r['source_id'] for r in cal]},output/'calibration_A.pt')
    sampled=sampled_rows(fitted,cfg.joint.proposal_count,cfg.joint.source_seed);meta=generate.target_metadata(cfg,bank,sampled);torch.save(meta,output/'sources_targets.pt')
    save_json(output/'partitions.json',{'train_sentences':len(fitted),'train_labelled_tokens':len(raw['y']),'checkpoint_validation_sentences':len(validation),'calibration_sentences':len(cal),'focals_per_calibration_sentence':1,'generation_trajectories':len(sampled),
        'roles':'A existing scorer, B new head, and fresh students share exact original fit pool. Native dev selects A/B/student checkpoints, never calibration. Native and existing transfer tests reserved for final F1.',
        'calibration_selection':'All unique reserved-fold tokenized sentences excluding exact inputs in fit or checkpoint validation; one seeded label-blind uniformly random focal per sentence; no correctness filtering.',
        'dependence':'Real source-document identifiers are unavailable (bank document IDs alias sentences). Independent sentences and IID generated attempts conditional on fixed source corpus/policies are working assumptions, not verified. This reserved corpus was used in earlier research diagnostics; it is not newly collected gold evidence.',
        'text':'Judge reads canonical word-slot detokenization of the G/A model input. Focal WordPiece IDs stay unchanged for G/A/students; B uses its own tokenizer. This may differ from pre-tokenization raw spelling for unknown or normalized tokens.'})
    return {'training_sentences':len(fitted),'calibration_sentences':len(cal)}


def benchmark(cfg,output):
    bank,rows=sources(cfg);rows=rows[:cfg.joint.benchmark_sources];meta=load(root(cfg)/'prepare'/'sources_targets.pt');encoder=generate.load_masked_encoder(cfg);a=load_model(data.paths(cfg)/'scorer'/task(cfg),cfg).requires_grad_(False)
    timing=generate.arm(cfg,bank,rows,meta,encoder,a,'uniform',1)
    projected=timing['wall_seconds']*cfg.joint.proposal_count/len(rows);del encoder,a
    model,tokenizer=representations.load_encoder(cfg);native=load(root(cfg)/'prepare'/'judge_native.pt');sample=native['train'][:cfg.joint.benchmark_judge_sentences];started=time.monotonic();representations.encode(cfg,model,tokenizer,sample);seconds=time.monotonic()-started
    estimate={'uniform_generation_seconds':projected,'judge_native_encoding_seconds':seconds/sum(len(r['focals']) for r in sample)*sum(len(r['focals']) for rs in native.values() for r in rs),'benchmark_uniform':timing,'benchmark_judge_seconds':seconds,'note':'Projection uses actual CPU bf16 timings; full-data head training, calibration/judging and evaluation are additional.'}
    estimate['projected_major_inference_hours']=(estimate['uniform_generation_seconds']+estimate['judge_native_encoding_seconds'])/3600;save_json(output/'estimate.json',estimate);print(estimate,flush=True);return {'inference_hours':estimate['projected_major_inference_hours']}


def judge_features(cfg,output):
    native=load(root(cfg)/'prepare'/'judge_native.pt');model,tokenizer=representations.load_encoder(cfg);summary={}
    for split,records in native.items():
        started=time.monotonic();features,info=representations.encode(cfg,model,tokenizer,records);torch.save(readout(records,features),output/f'{split}.pt');summary[split]={**info,'seconds':time.monotonic()-started};save_json(output/'encoding.json',summary)
    return {'splits':len(summary)}


def judge_train(cfg,output):
    parts={s:load(root(cfg)/'judge_features'/f'{s}.pt') for s in ('train','validation')};labels=data.load(cfg)['labels'][task(cfg)]
    fit(parts,len(labels),cfg,cfg.joint.judge_seed,output/task(cfg));return {'judge_heads':1}


def generate_stage(cfg,output):
    bank,rows=sources(cfg);meta=load(root(cfg)/'prepare'/'sources_targets.pt');encoder=generate.load_masked_encoder(cfg);a=load_model(data.paths(cfg)/'scorer'/task(cfg),cfg).requires_grad_(False)
    result=generate.arm(cfg,bank,rows,meta,encoder,a,'uniform',1,output/'uniform');save_json(output/'timing.json',result);return {'proposals':len(rows)}


def final_filter(cfg,raw,meta):
    p=raw['state_probabilities'][:,-1];confidence,label=p.max(-1);difference=raw['state_input_ids'][-1]!=meta['input_ids'];prefix=torch.cat((torch.zeros(1,dtype=torch.long),difference.long().cumsum(0)));offset=raw['offsets'];changed=prefix[offset[1:]]>prefix[offset[:-1]]
    accept=changed&(label!=meta['gold'])&(confidence>=cfg.joint.tau)
    if cfg.joint.source_correct:accept&=raw['source_prediction']==meta['gold']
    return accept,label,confidence,changed


def judge_outputs(cfg,output):
    bank,rows=sources(cfg);meta=load(root(cfg)/'prepare'/'sources_targets.pt');raw=load(root(cfg)/'generate'/'uniform'/'replicate0.pt');all_features=load(root(cfg)/'generate'/'uniform'/'all_features_draw0.pt');accept,label,confidence,changed=final_filter(cfg,raw,meta);indices=accept.nonzero().flatten()
    tokenizer_A=AutoTokenizer.from_pretrained(cfg.encoder.name,local_files_only=True);generated=[]
    for i in indices.tolist():
        lo,hi=raw['offsets'][i:i+2].tolist();ids=raw['state_input_ids'][-1,lo:hi].long();focal=int(rows[i]['focals'][0])
        generated.append({'words':words({**rows[i],'focals':torch.arange(len(rows[i]['hidden']))},tokenizer_A,ids),'focals':torch.tensor([focal]),'gold':label[i:i+1]})
    model,tokenizer=representations.load_encoder(cfg);calibration=load(root(cfg)/'prepare'/'calibration.pt');cal_x,cal_info=representations.encode(cfg,model,tokenizer,calibration)
    gen_x,gen_info=representations.encode(cfg,model,tokenizer,generated) if generated else (torch.empty(0,model.config.hidden_size),{})
    del model
    b=load_model(root(cfg)/'judge_train'/task(cfg),cfg).requires_grad_(False)
    def probabilities(x):return torch.cat([selector.scorer_log_probs(b,chunk.to(cfg.runtime.device),cfg).exp().cpu() for chunk in x.split(1024)]) if len(x) else torch.empty(0,len(meta['labels']))
    cal_p=probabilities(cal_x);gen_p=probabilities(gen_x);agree=gen_p.argmax(-1)==label[indices];agreed_indices=indices[agree]
    torch.save({'calibration_probability':cal_p,'generated_probability':gen_p,'accepted_indices':indices,'agreed_indices':agreed_indices,'accepted_A_label':label[indices]},output/'predictions.pt')
    for name,index in [('ALL',torch.arange(len(label))),('FLIP',indices),('AGREED',agreed_indices)]:
        folder=root(cfg)/'generate'/name;folder.mkdir(exist_ok=True);torch.save({'features':all_features['features'][index],'y':label[index],'anchor_id':index},folder/'accepted_features_draw0.pt')
    save_json(output/'counts.json',{'ALL':len(label),'FLIP':len(indices),'AGREED':len(agreed_indices),'unchanged_inputs':int((~changed).sum()),'source_incorrect':int((raw['source_prediction']!=meta['gold']).sum()),'calibration_encoding':cal_info,'accepted_encoding':gen_info,'B_used_in_generation_or_FLIP_filter':False,'H_a_source_correctness_requirement':cfg.joint.source_correct})
    return {'accepted':len(indices),'agreed':len(agreed_indices)}


def certificate(cfg,output):
    a=load(root(cfg)/'prepare'/'calibration_A.pt');b=load(root(cfg)/'judge_outputs'/'predictions.pt');labels=load(root(cfg)/'prepare'/'sources_targets.pt')['labels'];K=len(labels)
    confidence,pred=a['probability'].max(-1);high=confidence>=cfg.joint.tau;cal_agree=high&(pred==b['calibration_probability'].argmax(-1));wrong=pred!=a['gold'];accepted=b['accepted_A_label'];gen_agree=b['generated_probability'].argmax(-1)==accepted;generated=accepted[gen_agree];m=len(generated);mass=torch.bincount(generated,minlength=K);bounds=[];empirical=[];table=[];ql=[];qu=[]
    for g,label in enumerate(labels):
        selected=cal_agree&(pred==g);n=int(selected.sum());k=int((selected&wrong).sum());upper=float(beta.ppf(1-cfg.joint.alpha_cal/K,k+1,n-k)) if n and k<n else 1.
        count=int(mass[g]);lower_q=float(beta.ppf(cfg.joint.alpha_mass/(2*K),count,m-count+1)) if m and count else 0.;upper_q=float(beta.ppf(1-cfg.joint.alpha_mass/(2*K),count+1,m-count)) if m and count<m else 1.
        bounds.append(upper);empirical.append(k/n if n else None);ql.append(lower_q);qu.append(upper_q)
        table.append({'group':label,'natural_A_confident':int((high&(pred==g)).sum()),'natural_agreed':n,'natural_agreed_wrong':k,'agreed_error_plugin':k/n if n else None,'agreed_error_upper':upper,'generated_agreed':count,'q_plugin':count/m if m else None,'q_lower':lower_q,'q_upper':upper_q})
    w=torch.tensor(bounds,dtype=torch.double);lo=torch.tensor(ql,dtype=torch.double);hi=torch.tensor(qu,dtype=torch.double)
    # max q.w = -min q.(-w), preserving sum(q)=1.
    minimum,q,lp=minimize(lo,hi,-w) if m else (0.,None,None);worst=-minimum if q is not None else 1.;bound=max(0.,1-worst) if m and q is not None else 0.
    plugin_error=sum(int(mass[g])/m*empirical[g] for g in range(K) if mass[g] and empirical[g] is not None) if m else None
    if any(mass[g] and empirical[g] is None for g in range(K)):plugin_error=None
    result={'delta0_correctness_lower':bound,'worst_error_delta0':worst,'delta95_breakdown':max(0.,bound-.95),'delta90_breakdown':max(0.,bound-.90),'signed_margin95':bound-.95,'signed_margin90':bound-.90,
        'passes95_at_delta0':m>0 and q is not None and bound>=.95,'passes90_at_delta0':m>0 and q is not None and bound>=.90,'agreed_outputs':m,'accepted_outputs':len(accepted),
        'agreement_generated_accepted':float(gen_agree.double().mean()) if len(accepted) else None,'agreement_natural_A_confident':float(cal_agree.sum()/high.sum()),
        'natural_A_confident_n':int(high.sum()),'natural_agreed_n':int(cal_agree.sum()),'naive_A_only_accuracy':float((high&~wrong).sum()/high.sum()),
        'plugin_agreed_error':plugin_error,'plugin_agreed_correctness':1-plugin_error if plugin_error is not None else None,'region_nonempty':q is not None,'alpha_cal':cfg.joint.alpha_cal,'alpha_mass':cfg.joint.alpha_mass,
        'delta_rule':'With sum q=1, correctness lower(delta)=max(0,delta0_lower-delta). At delta*=margin the threshold is attained; any larger slack fails. Breakdown 0 when already below threshold.',
        'assumption':'For every A-predicted group, generated agreed joint-error rate <= natural A-confident agreed error rate + delta. No independence of A and B errors is assumed. Error dependence may remain despite different pretrained families.',
        'coverage':'Simultaneous CP group events by Bonferroni: K error upper bounds share .025, 2K mass endpoints share .025; union bound needs no event independence. Natural observations use one focal per sentence. Sentence independence/source-document dependence remains an unverified working assumption. Conditional certificate only, not measured generated gold accuracy.'}
    if q is not None:
        for row,value in zip(table,q.tolist()):row['worst_case_mass']=value
    write_csv(output/'groups.csv',table);save_json(output/'certificate.json',result);print(result,flush=True);return {'correctness_lower':bound,'agreed':m}


def evaluation(cfg,output):
    students.test(cfg,output)
    return evaluation_from_logits(cfg,output,output)


def evaluation_from_logits(cfg,output,logits_root):
    from seqeval.metrics import classification_report
    cache=load(absolute(cfg.grid_pilot.evaluation_cache));labels=cache['labels'];records=[];per_label=[]
    for seed in cfg.augmentation.seeds:
        for split,item in cache['sets'].items():
            if split=='validation':continue
            saved=load(logits_root/f'seed{seed}_{split}.pt');gold=item['data']['y'];sentence=item['data']['sentence'];_,counts=torch.unique_consecutive(sentence,return_counts=True);target=item['target']
            for space in (['native','shared_ner'] if split=='id' else ['shared_ner']):
                def tags(sequence,names,is_gold):
                    tags=[names[int(i)] for i in sequence]
                    if space=='shared_ner':tags=[('B-' if t.startswith('B-') else 'I-' if t.startswith('I-') else '')+shared_label(t) if shared_label(t)!='O' else 'O' for t in tags]
                    return tags if any(t.startswith(('B-','I-')) for t in tags) else bio(tags)
                target_names=labels if space=='native' else LABELS[target];truth=[tags(part,target_names,True) for part in gold.split(counts.tolist())]
                for h,name in enumerate(saved['heads']):
                    predictions=saved['logits'][:,h].argmax(-1);guess=[tags(part,labels,False) for part in predictions.split(counts.tolist())];report=classification_report(truth,guess,output_dict=True,zero_division=0);micro=report['micro avg']
                    records.append({'seed':seed,'head':name,'target':target,'space':space,'span_f1':micro['f1-score'],'precision':micro['precision'],'recall':micro['recall']})
                    for label,value in report.items():per_label.append({'seed':seed,'head':name,'target':target,'space':space,'label':label,**value})
    summary=[]
    for target,space in sorted({(r['target'],r['space']) for r in records}):
        for name in dict.fromkeys(r['head'] for r in records):
            selected=[r for r in records if (r['target'],r['space'],r['head'])==(target,space,name)];scores=torch.tensor([r['span_f1'] for r in selected]);delta=torch.tensor([r['span_f1']-next(b['span_f1'] for b in records if (b['seed'],b['target'],b['space'],b['head'])==(r['seed'],target,space,'baseline')) for r in selected]);summary.append({'head':name,'target':target,'space':space,'mean_span_f1':float(scores.mean()),'seed_sd':float(scores.std()),'paired_delta_vs_baseline':float(delta.mean()),'paired_delta_seed_sd':float(delta.std())})
    label_summary=[]
    for name,target,space,label in sorted({(r['head'],r['target'],r['space'],r['label']) for r in per_label}):
        selected=[r for r in per_label if (r['head'],r['target'],r['space'],r['label'])==(name,target,space,label)];values=torch.tensor([r['f1-score'] for r in selected]);label_summary.append({'head':name,'target':target,'space':space,'label':label,'mean_f1':float(values.mean()),'seed_sd':float(values.std()) if len(values)>1 else None,'seeds':len(values)})
    write_csv(output/'seqeval_per_seed.csv',records);write_csv(output/'per_label_per_seed.csv',per_label);write_csv(output/'per_label_summary.csv',label_summary);write_csv(output/'seqeval_summary.csv',summary);return {'evaluations':len(records)}


def report(cfg,output):
    c=json.loads((root(cfg)/'certificate'/'certificate.json').read_text());counts=json.loads((root(cfg)/'judge_outputs'/'counts.json').read_text());parts=json.loads((root(cfg)/'prepare'/'partitions.json').read_text());scores=read_csv(root(cfg)/'test'/'seqeval_summary.csv');groups=read_csv(root(cfg)/'certificate'/'groups.csv')
    lines=['# Independent-judge uniform-refinement CPU experiment','',
        f"G and A: frozen {cfg.encoder.name}, same BERT encoder family. B: frozen {cfg.joint.judge_name}, ELECTRA family, new head trained on A's same {parts['train_sentences']} original fitting sentences. Head recipe and students unchanged; full-data students, three seeds. One final generated context per trajectory, exact focal pieces and G positions/length retained. Existing uniform finite-pool code is reused, with all final focal vectors cached in addition to accepted ones.",'',
        f"Calibration has one fixed random focal per each of {parts['calibration_sentences']} held-out unique sentences, excluding exact fitted and checkpoint-validation inputs. A errors retained after confidence >=.7. Natural A-only accuracy: {100*c['naive_A_only_accuracy']:.3f}%. A–B agreement: natural {100*c['agreement_natural_A_confident']:.3f}%, accepted generated {100*c['agreement_generated_accepted']:.3f}%.",'',
        '| Quantity | Value |','|---|---:|',f"| Agreed correctness lower, delta=0 | {100*c['delta0_correctness_lower']:.3f}% |",f"| Plug-in agreed correctness | {c['plugin_agreed_correctness']} |",f"| Delta breakdown at 95% | {c['delta95_breakdown']:.6f} |",f"| Delta breakdown at 90% | {c['delta90_breakdown']:.6f} |",f"| Pass 95% at zero slack | {c['passes95_at_delta0']} |",f"| Pass 90% at zero slack | {c['passes90_at_delta0']} |",'',
        'Certificate is conditional on joint-error transfer, not observed augmentation gold accuracy. Different encoder families do not establish independence of errors. Bonferroni allocations and the mass-box simplex optimization are saved with an independent LP check. If a calibration group has no agreements its error upper bound is one; no probability floors or invented correctness.','',
        '| Group | Natural agreed | Wrong agreements | Error upper | Generated agreed |','|---|---:|---:|---:|---:|']
    for r in groups:lines.append(f"| {r['group']} | {r['natural_agreed']} | {r['natural_agreed_wrong']} | {float(r['agreed_error_upper']):.5f} | {r['generated_agreed']} |")
    lines+=['','| Student | Augmented focal examples |','|---|---:|','| baseline | 0 |',*[f'| {name} | {counts[name]} |' for name in cfg.mechanism_pilot.arms],'',
        'ALL includes every final context, without confidence, source-correctness, text-change or flip filtering. FLIP uses H_a; AGREED adds B=A only after final acceptance. B never guides generation. Variant losses cover focal pseudo-labels only; original labels preserved. Omega=1, mean CE over labelled tokens, proportional mixtures, equal optimizer-step/labelled-token budgets with exact counts in train/budgets.csv. Raw native dev token macro-F1 selects every checkpoint.','',
        '| Target | Space | Student | Span F1 mean ± seed SD | Delta pp |','|---|---|---|---:|---:|']
    for r in scores:lines.append(f"| {r['target']} | {r['space']} | {r['head']} | {100*float(r['mean_span_f1']):.3f} ± {100*float(r['seed_sd']):.3f} | {100*float(r['paired_delta_vs_baseline']):+.3f} |")
    lines+=['','Per-label seqeval F1/precision/recall: ../test/per_label_per_seed.csv; micro F1: ../test/seqeval_per_seed.csv. Existing shared label mapping is unchanged.','',
        'Limitations/deviations:','',
        '- The user supplied an explicit arm/filter specification replacing the missing notes reference. Notes were not modified.',
        '- Uniform generation disables intermediate scorer logging: only the final selected context is encoded/scored by A; intermediate score/reward fields are NaN, not inferred or fabricated. Candidate sampling and uniform-selection RNG streams are unchanged.',
        f"- H_a includes the established source-correctness check: {cfg.joint.source_correct}. This is recorded explicitly because the prompt abbreviates H_a to final confidence/new-label conditions.",
        '- Known remaining discrepancy with the clarification: anchors are drawn before checking source correctness, which is enforced in FLIP rather than before generation. Consequently ALL includes incorrect-source anchors. The logging-only optimization does not change that sampling policy.',
        '- The number of proposals is 20000, the established small-run budget. Generation anchors are IID uniform-word draws with replacement from the full fit pool, not the previous 2400-sentence student subset. A, B and fresh students now share the full original fit pool as requested; compare to this fresh full-data baseline, not the previous small-data baseline.',
        '- Canonical G word-slot detokenization feeds B. Its own tokenizer may split words differently; nonoverlapping overflow chunks retain all requested word pieces, with mean pooling across pieces. G/A/student focal vectors retain original G input IDs.',
        '- Sentence independence and missing original-document provenance remain flagged; one focal per calibration sentence avoids treating multiple tokens in a sentence as independent. Historical calibration reuse is disclosed in partitions.json. No A/B error-independence assumption is made.',
        '- CPU only, bf16 inference, 16 configured CPU threads. Exact configs/seeds and per-stage timings are saved.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n');return {'certificate':c['delta0_correctness_lower']}


def main(cfg):
    OmegaConf.resolve(cfg);configure(cfg);cfg.forge.store=str(absolute(cfg.forge.store));tracking=start_run(cfg);folder=root(cfg);folder.mkdir(parents=True,exist_ok=True);stage=cfg.stage.experiment;output=folder/stage;output.mkdir(exist_ok=True)
    OmegaConf.save(cfg,folder/'config.yaml')
    if not (folder/'timing.json').exists():save_json(folder/'timing.json',{'started_unix':time.time()})
    print(f'Independent judge {stage}: {output} ({cfg.runtime.device})',flush=True);start=time.monotonic()
    operation={'prepare':prepare,'benchmark':benchmark,'judge_features':judge_features,'judge_train':judge_train,'generate':generate_stage,'judge_outputs':judge_outputs,'certificate':certificate,'train':students.train,'test':evaluation,'report':report}[stage]
    result=operation(cfg,output);save_json(output/'runtime.json',{'seconds':time.monotonic()-start});tracking.finish(result)
    if stage=='report':
        timing=json.loads((folder/'timing.json').read_text());timing['ended_unix']=time.time();timing['wall_seconds']=timing['ended_unix']-timing['started_unix'];save_json(folder/'timing.json',timing)
    return 0


if __name__=='__main__':
    stages=[None] if any(a.startswith('stage.experiment=') for a in sys.argv[1:]) else ['prepare','benchmark','judge_features','judge_train','generate','judge_outputs','certificate','train','test','report']
    for stage in stages:
        code=run('lion.independent_judge',sys.argv[1:]+([] if stage is None else [f'stage.experiment={stage}']),main_module='run',config_dir=str(ROOT/'conf'),config_name='independent_judge')
        if code:sys.exit(code)
