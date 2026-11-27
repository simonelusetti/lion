"""Frozen five-bin population-correctness diagnostic and small matched students."""
import json
import math
import sys
import time
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from scipy.stats import beta
from forge import run,start_run
from omegaconf import OmegaConf
from ..certification import data,selector
from ..partition.train import load_model
from ..encoder import load_masked_encoder
from ..runtime import configure,autocast
from ..diagnostics.common import ROOT,absolute,save_json,write_csv,read_csv
from ..label_change.data import sampled_rows
from ..proposal_mechanisms import train_pilot as downstream
from . import generate,train_pilot,size_grid


def root(cfg):return downstream.root(cfg)
def load(path):return torch.load(path,weights_only=False,mmap=True)


def proposal_rows(cfg):
    bank,cohort=train_pilot.anchors(cfg)
    indices=load(root(cfg)/'prepare'/'proposal_indices.pt')
    return bank,[cohort[i] for i in indices.tolist()]


def packed_row(row):
    focal=int(row['focals'][0]);positions=(row['word_ids']==focal).nonzero().flatten()
    return {'ids':row['ids'].clone(),'ordinary':row['word_ids']>=0,'start':int(positions[0]),'end':int(positions[-1])+1,
            'source_id':row['source_id'],'gold':int(row['gold']['ner'][0]),'hidden':row['hidden'][focal].clone()}


def prepare(cfg,output):
    result=downstream.prepare(cfg,output)
    indices=torch.randint(cfg.guided_training.anchors,(cfg.population.proposals,),generator=torch.Generator().manual_seed(cfg.population.proposal_seed))
    torch.save(indices,output/'proposal_indices.pt')
    bank,rows=proposal_rows(cfg);torch.save(generate.target_metadata(cfg,bank,rows),output/'sources_targets.pt')
    raw=selector.raw_rows(cfg,bank);held=[r for r in raw if r['fold']=='calibration']
    # Reconstruct both earlier labelled calibration streams, excluding whole sentences.
    exposed={r['source_id'] for offset in (2,3) for r in sampled_rows(held,cfg.label_change.calibration_draws,cfg.label_change.seed+offset)}
    fitted={tuple(r['ids'].tolist()) for r in raw if r['fold']=='fit'};seen=set();natural=[]
    for row in held:
        key=tuple(row['ids'].tolist())
        if row['source_id'] in exposed or key in fitted or key in seen:continue
        seen.add(key);natural.append(row)
    order=torch.randperm(len(natural),generator=torch.Generator().manual_seed(cfg.population.natural_seed)).tolist()
    focal_rng=torch.Generator().manual_seed(cfg.population.focal_seed);natural=[natural[i] for i in order];focals=[]
    for row in natural:
        f=torch.tensor([int(torch.randint(len(row['focals']),(1,),generator=focal_rng))])
        focals.append(packed_row({**row,'focals':f,'gold':{'ner':row['gold']['ner'][f]}}))
    n=cfg.population.design_sentences
    torch.save(focals[:n],output/'natural_design.pt');torch.save(focals[n:],output/'natural_reference.pt')
    audit={'scorer_fit_sentences':sum(r['fold']=='fit' for r in raw),'scorer_reserved_sentences':len(held),
        'previously_sampled_calibration_sentences_excluded':len(exposed),'natural_design_sentences':len(focals[:n]),'natural_frequency_correctness_sentences':len(focals[n:]),
        'scorer_training_implementation':'lion.label_change.data.scorer -> parts: fit fold only; native dev used for checkpoint selection, reserved calibration excluded.',
        'scorer_retrained':False,'natural_policy':'One uniformly random focal word per sentence, label-blind, no vocabulary restriction. Exact tokenized duplicates and fitted-sentence overlap excluded before random design/reference partition; confidence >=.7, never conditioned on correctness.',
        'independence':'Disjoint natural sentences assumed independent task-domain observations. Design independent of reference. Generation: 20000 IID source indices sampled with replacement from the fixed original 4000-anchor pilot cohort; independent trajectory randomness. Shared sequence across arms, no five-draw clusters. Bounds condition on frozen scorer/cohort/policies; eta shared across natural/generated inputs is assumed.'}
    save_json(output/'independence_audit.json',audit);print(audit,flush=True)
    return {**result,'proposals':len(rows),'reference_sentences':len(focals[n:])}


def collate(rows):
    width=max(len(r['ids']) for r in rows);ids=torch.zeros(len(rows),width,dtype=torch.long);ordinary=torch.zeros_like(ids,dtype=torch.bool);attention=torch.zeros_like(ordinary);focal=torch.zeros_like(ordinary)
    for i,r in enumerate(rows):
        n=len(r['ids']);ids[i,:n]=r['ids'];ordinary[i,:n]=r['ordinary'];attention[i,:n]=True;focal[i,r['start']:r['end']]=True
    return {'ids':ids,'attention':attention,'drop':ordinary&~focal}


@torch.no_grad()
def phi(cfg,encoder,rows,mask_id):
    # Raw MLM logits, without the sampling-vocabulary renormalization; observed-token NLL.
    loader=DataLoader(rows,batch_size=max(1,min(cfg.runtime.batch_size,cfg.runtime.mlm_position_budget//cfg.encoder.max_length)),collate_fn=collate,num_workers=cfg.runtime.workers)
    values=[]
    for b in loader:
        ids,attention,drop=(b[k].to(cfg.runtime.device) for k in ('ids','attention','drop'))
        if not drop.any():values.append(torch.zeros(len(ids)));continue
        with autocast(cfg):logits=encoder(ids.masked_fill(drop,mask_id),attention,drop,mlm=True)['logits'].float()
        loss=F.cross_entropy(logits,ids[drop],reduction='none');owner=drop.nonzero()[:,0]
        score=torch.zeros(len(ids),device=ids.device).scatter_add_(0,owner,loss)/drop.sum(-1).clamp_min(1)
        values.append(score.cpu())
    return torch.cat(values)


@torch.no_grad()
def natural_scores(cfg,encoder,scorer,rows,mask_id):
    hidden=torch.stack([r['hidden'] for r in rows]);prob=torch.cat([selector.scorer_log_probs(scorer,x.to(cfg.runtime.device),cfg).exp().cpu() for x in hidden.split(1024)])
    confidence,pred=prob.max(-1);keep=confidence>=cfg.guided.tau
    chosen=[r for r,k in zip(rows,keep.tolist()) if k]
    return {'phi':phi(cfg,encoder,chosen,mask_id),'prediction':pred[keep],'gold':torch.tensor([r['gold'] for r in chosen]),'source_id':[r['source_id'] for r in chosen],'total_sentences':len(rows)}


def design(cfg,output):
    bank=data.load(cfg);encoder=load_masked_encoder(cfg);scorer=load_model(data.paths(cfg)/'scorer'/'ner',cfg).requires_grad_(False)
    sample=natural_scores(cfg,encoder,scorer,load(root(cfg)/'prepare'/'natural_design.pt'),bank['special']['mask'])
    # Stable right-closed assignment: tied quantile boundaries collapse before freezing.
    edges=torch.quantile(sample['phi'].double(),torch.arange(1,cfg.population.bins,dtype=torch.double)/cfg.population.bins).unique(sorted=True)
    torch.save({'edges':edges,'design_phi':sample['phi']},output/'bins.pt')
    save_json(output/'bins.json',{'interior_edges':edges.tolist(),'actual_bins':len(edges)+1,'range':'(-infinity, first], ..., (last, infinity)','design_selected':len(sample['phi']),'design_sentences':sample['total_sentences'],'ties':'Identical boundaries collapse; torch.bucketize right=False; empty nonfocal context has score zero.','score':'Single parallel masked-reconstruction mean observed nonfocal-token NLL, focal pieces and boundaries preserved. Not a generator likelihood.'})
    return {'bins':len(edges)+1}


def candidates(cfg,output):
    bank,rows=proposal_rows(cfg);meta=load(root(cfg)/'prepare'/'sources_targets.pt');encoder=load_masked_encoder(cfg);scorer=load_model(data.paths(cfg)/'scorer'/'ner',cfg).requires_grad_(False);timings=[]
    for rule in cfg.mechanism_pilot.arms:
        if rule in cfg.guided.arms:t=generate.arm(cfg,bank,rows,meta,encoder,scorer,rule,1,output/rule)
        else:t=size_grid.fraction_arm(cfg,bank,rows,encoder,scorer,rule,output/rule)
        timings.append(t);save_json(output/'timing.json',timings)
    return {'arms':len(timings),'proposals_per_arm':len(rows)}


def cp(k,n,tail):
    if n==0:return 0.,1.
    return (float(beta.ppf(tail,k,n-k+1)) if k else 0.,float(beta.ppf(1-tail,k+1,n-k)) if k<n else 1.)


def ratio_max(p,q):
    # A sample zero is not evidence that the population bin is known null.
    return max(1.,max(max(float(a/b) if b>0 else math.inf,float(b/a) if a>0 else math.inf) for a,b in zip(p,q)))


def quality(correct,m):return correct/(correct+m*m*(1-correct)) if math.isfinite(m) else 0.
def finite(value):return value if math.isfinite(value) else 'inf'


def diagnostic(cfg,output):
    bank,rows=proposal_rows(cfg);meta=load(root(cfg)/'prepare'/'sources_targets.pt');edges=load(root(cfg)/'design'/'bins.pt')['edges'];j=len(edges)+1
    encoder=load_masked_encoder(cfg);scorer=load_model(data.paths(cfg)/'scorer'/'ner',cfg).requires_grad_(False)
    sample=natural_scores(cfg,encoder,scorer,load(root(cfg)/'prepare'/'natural_reference.pt'),bank['special']['mask']);torch.save(sample,output/'natural_reference.pt')
    n=len(sample['phi']);counts=torch.bincount(torch.bucketize(sample['phi'].double(),edges),minlength=j);correct=int((sample['prediction']==sample['gold']).sum());lower=cp(correct,n,cfg.population.alpha_cal)[0];empirical=correct/n
    arms=list(cfg.mechanism_pilot.arms);tail=cfg.population.alpha_in/(2*j*(1+len(arms)));p=counts.double()/n;intervals=[cp(int(c),n,tail) for c in counts]
    binrows=[{'population':'natural','bin':b,'n':n,'count':int(c),'frequency':float(p[b]),'lower':intervals[b][0],'upper':intervals[b][1]} for b,c in enumerate(counts)];summaries=[]
    for rule in arms:
        raw=load(root(cfg)/'generate'/rule/'replicate0.pt');final=raw['state_input_ids'][-1];prob=raw['state_probabilities'][:,-1];confidence,pred=prob.max(-1)
        offsets=raw['offsets'];proposal_keys=[];sentence_keys=[];accepted=[];reconstructed=0;source_wrong=0;accepted_rows=[]
        for i,row in enumerate(rows):
            ids=final[int(offsets[i]):int(offsets[i+1])].long();key=ids.numpy().tobytes();span=(int(meta['focal_start'][i]),int(meta['focal_end'][i]));sentence_keys.append(key);proposal_keys.append((key,*span))
            identical=torch.equal(ids,row['ids']);reconstructed+=int(identical);original_correct=int(raw['source_prediction'][i])==int(meta['gold'][i]);source_wrong+=int(not original_correct)
            take=original_correct and not identical and int(pred[i])!=int(meta['gold'][i]) and float(confidence[i])>=cfg.guided.tau
            if take:
                accepted.append(i);accepted_rows.append({'ids':ids,'ordinary':row['word_ids']>=0,'start':span[0],'end':span[1]})
        saved=load(root(cfg)/'generate'/rule/'accepted_features_draw0.pt')
        if saved['anchor_id'].tolist()!=accepted:raise ValueError(f'{rule}: cached student inputs do not match diagnostic acceptance')
        scores=phi(cfg,encoder,accepted_rows,bank['special']['mask']) if accepted else torch.empty(0)
        qcounts=torch.bincount(torch.bucketize(scores.double(),edges),minlength=j);m=len(accepted);q=qcounts.double()/m if m else torch.zeros(j,dtype=torch.double);qintervals=[cp(int(c),m,tail) for c in qcounts]
        plug_m=ratio_max(p,q);upper_m=max(1.,max(max(qintervals[b][1]/intervals[b][0] if intervals[b][0]>0 else math.inf,intervals[b][1]/qintervals[b][0] if qintervals[b][0]>0 else math.inf) for b in range(j)))
        bound=quality(lower,upper_m);plug=quality(empirical,plug_m);a=m/len(rows);keys=[proposal_keys[i] for i in accepted]
        for b,c in enumerate(qcounts):binrows.append({'population':rule,'bin':b,'n':m,'count':int(c),'frequency':float(q[b]),'lower':qintervals[b][0],'upper':qintervals[b][1]})
        summaries.append({'arm':rule,'proposals':len(rows),'accepted':m,'acceptance':a,'duplicate_focal_proposals':len(rows)-len(set(proposal_keys)),'duplicate_generated_sentences':len(rows)-len(set(sentence_keys)),
            'distinct_accepted_focal_examples':len(set(keys)),'duplicate_accepted_focal_examples':m-len(set(keys)),'source_reconstructions':reconstructed,'incorrect_source_exclusions':source_wrong,
            'natural_n':n,'natural_correct':correct,'natural_correctness':empirical,'L':lower,'M_plugin':finite(plug_m),'M_upper':finite(upper_m),'Pi_lower':bound,'Pi_plugin':plug,'strict_check':'PASS' if bound>=.95 and math.isfinite(upper_m) else 'FAIL','margin':bound-.95,'old_G':max(0.,1-cfg.population.old_error/a) if a else 0.})
        torch.save({'phi':scores,'proposal_indices':torch.tensor(accepted),'bin_counts':qcounts},output/f'{rule}_scores.pt')
        write_csv(output/'quality.csv',summaries);write_csv(output/'bin_counts.csv',binrows)
        print(rule,summaries[-1],flush=True)
    save_json(output/'confidence.json',{'alpha_cal':cfg.population.alpha_cal,'alpha_in':cfg.population.alpha_in,'input_one_sided_tail':tail,'input_intervals':2*j*(1+len(arms)),
        'natural_n':n,'natural_correct':correct,'L':lower,'bins':j,'confidence_at_least':.95,'assumption':'P0(.|Bj)=Qr(.|Bj), with matching support, for every bin and arm; shared annotation mechanism eta. Neither equality nor support is established by these counts. Bounds are conditional on this strong assumption, frozen policies and independent sampling.',
        'duplicate_definition':'Exact non-padding token IDs; focal examples also include focal WordPiece start/end. Duplicate count is excess occurrences beyond unique values. No deduplication or redraw in generation or training.',
        'empty_bins':'Any zero lower denominator yields infinite M_upper and zero primary quality; no floors or pseudocounts.'})
    return {'arms':len(summaries),'natural_n':n}


def ranks(x):
    _,inverse,counts=torch.unique(x,sorted=True,return_inverse=True,return_counts=True)
    return (counts.cumsum(0)-counts+(counts.double()-1)/2)[inverse]


def correlation(x,y,rank=False):
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():return None,'non-finite values'
    if x.unique().numel()<2 or y.unique().numel()<2:return None,'constant metric'
    if rank:x,y=ranks(x),ranks(y)
    return float(torch.corrcoef(torch.stack((x,y)))[0,1]),''


def report(cfg,output):
    quality_rows=read_csv(root(cfg)/'diagnostic'/'quality.csv');scores=read_csv(root(cfg)/'test'/'summary.csv');joined=[];correlations=[]
    for r in quality_rows:
        for s in scores:
            if s['head']==r['arm']:joined.append({**r,**s})
    write_csv(output/'arm_results.csv',joined)
    for group,names in [('all_11',list(cfg.mechanism_pilot.arms)),('refinement_5',list(cfg.guided.arms))]:
        for target,space in sorted({(r['target'],r['space']) for r in joined}):
            selected=[r for r in joined if r['arm'] in names and (r['target'],r['space'])==(target,space)]
            y=torch.tensor([float(r['paired_delta_vs_baseline']) for r in selected],dtype=torch.double)
            for metric in ('Pi_lower','Pi_plugin','acceptance','old_G'):
                x=torch.tensor([float(r[metric]) for r in selected],dtype=torch.double)
                for kind in ('Pearson','Spearman'):
                    value,reason=correlation(x,y,kind=='Spearman');correlations.append({'group':group,'target':target,'space':space,'metric':metric,'correlation':kind,'value':value,'undefined_reason':reason,'arms':len(selected)})
    write_csv(output/'correlations.csv',correlations)
    native={r['arm']:r for r in joined if r['target']=='fewnerd' and r['space']=='native'}
    lines=['# Binned population-correctness diagnostic','',
        'Small generation budget: 20,000 IID proposal trajectories per arm; same 2,400 original student sentences, fixed full-data scorer, three seeds and equal training budgets. Certification is diagnostic and never blocks training. Natural bin design is frozen before reference correctness/frequencies or downstream evaluation.','',
        'The conditional certificate assumes identical natural/generated input distributions within each bin, matching support, shared annotation mechanism, and independent observations. Bin counts do not establish these assumptions. The score is parallel masked-reconstruction NLL, not a generator density. Historical guided lambda rules retain their original finite-pool policy.','',
        '| Arm | Accepted | M upper | Quality lower | Plug-in quality | Check | Native F1 | Delta pp |','|---|---:|---:|---:|---:|---|---:|---:|']
    for r in quality_rows:
        s=native[r['arm']];lines.append(f"| {r['arm']} | {r['accepted']} | {r['M_upper']} | {float(r['Pi_lower']):.4f} | {float(r['Pi_plugin']):.4f} | {r['strict_check']} | {100*float(s['mean_span_f1']):.2f} | {100*float(s['paired_delta_vs_baseline']):+.2f} |")
    lines+=['','Full per-seed classification results: ../test/per_seed.csv. All dataset means, seed SDs, paired improvements and quality numbers: arm_results.csv. Pearson/Spearman comparisons for all eleven arms and the refinement subset: correlations.csv. Undefined correlations retain their reason. Exact bin counts and intervals: ../diagnostic/bin_counts.csv; inspect zero lower endpoints for collapsed bounds. Independence audit: ../prepare/independence_audit.json.','',
        'Old G uses the historical fixed 0.028 error reference and empirical acceptance; it remains a descriptive ideal-sampler comparator. Neither quality lower bounds nor correlations verify the labels of individual augmentations.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n');return {'comparisons':len(joined),'correlations':len(correlations)}


def main(cfg):
    OmegaConf.resolve(cfg);configure(cfg);cfg.forge.store=str(absolute(cfg.forge.store));tracking=start_run(cfg)
    folder=root(cfg);folder.mkdir(parents=True,exist_ok=True);stage=cfg.stage.experiment;output=folder/stage;output.mkdir(exist_ok=True)
    if not (folder/'timing.json').exists():save_json(folder/'timing.json',{'started_unix':time.time()})
    OmegaConf.save(cfg,folder/'config.yaml');started=time.monotonic();print(f'Population quality {stage}: {output}',flush=True)
    result={'prepare':prepare,'design':design,'generate':candidates,'diagnostic':diagnostic,'train':downstream.train,'test':train_pilot.evaluate,'report':report}[stage](cfg,output)
    save_json(output/'runtime.json',{'seconds':time.monotonic()-started});tracking.finish(result)
    if stage=='report':
        timing=json.loads((folder/'timing.json').read_text());timing['ended_unix']=time.time();timing['wall_seconds']=timing['ended_unix']-timing['started_unix'];save_json(folder/'timing.json',timing)
        with (output/'RESULTS.md').open('a') as stream:stream.write(f"\nTotal wall-clock time: {timing['wall_seconds']/3600:.2f} hours.\n")
    return 0


if __name__=='__main__':
    stages=[None] if any(a.startswith('stage.experiment=') for a in sys.argv[1:]) else ['prepare','design','generate','diagnostic','train','test','report']
    for stage in stages:
        code=run('lion.guided_refinement',sys.argv[1:]+([] if stage is None else [f'stage.experiment={stage}']),main_module='population',config_dir=str(ROOT/'conf'),config_name='population_quality')
        if code:sys.exit(code)
