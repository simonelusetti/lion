"""Small matched downstream comparison: same originals, fixed scorer, three mechanisms."""
import json,sys,time
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from forge import run,start_run
from omegaconf import OmegaConf
from tagger.ensemble import IndependentReadouts,clip_heads
from ..runtime import configure,autocast
from ..diagnostics.common import ROOT,absolute,save_json,write_csv
from ..certification import data,selector
from ..partition.train import load_model,predict,metric_vectors
from ..augmentation.train import schedule,collate
from ..label_change_grid.train import Steps
from ..certification.classify import evaluate,span_counts
from ..encoder import load_masked_encoder
from ..labels import LABELS
from . import generate

def head_names(cfg):return ('baseline',*cfg.mechanism_pilot.get('arms',generate.ARMS))


def augmentation_features(cfg,name):
    frames=[torch.load(root(cfg)/'generate'/name/f'accepted_features_draw{draw}.pt',mmap=True,weights_only=False) for draw in range(cfg.mechanisms.draws)]
    return {key:torch.cat([f[key] for f in frames]) for key in ('features','y')}


def root(cfg):return absolute(cfg.mechanism_pilot.results)


def prepare(cfg,output):
    bank=data.load(cfg);rows=[r for r in selector.raw_rows(cfg,bank) if r['fold']=='fit'];order=torch.randperm(len(rows),generator=torch.Generator().manual_seed(cfg.mechanism_pilot.subset_seed))[:cfg.mechanism_pilot.sentences].sort().values.tolist();rows=[rows[i] for i in order]
    raw=data.readout_data(rows,torch.cat([r['hidden'] for r in rows]),'ner');torch.save(raw,output/'raw.pt');write_csv(output/'sentences.csv',[{'sentence':r['sentence'],'source_id':r['source_id'],'words':len(r['focals'])} for r in rows])
    save_json(output/'subset.json',{'sentences':len(rows),'words':len(raw['y']),'subset_seed':cfg.mechanism_pilot.subset_seed,'source':'Full scorer fit pool only; calibration excluded','source_ids':[r['source_id'] for r in rows]})
    return {'sentences':len(rows),'words':len(raw['y'])}


def training_rows(cfg):
    selected=json.loads((root(cfg)/'prepare'/'subset.json').read_text())['source_ids'];chosen=set(selected);bank=data.load(cfg);rows=[r for r in selector.raw_rows(cfg,bank) if r['source_id'] in chosen];focal_rows=[]
    for r in rows:
        for focal in r['focals'].tolist():
            f=torch.tensor([focal]);focal_rows.append({**r,'focals':f,'gold':{'ner':r['gold']['ner'][f]}})
    return bank,focal_rows


@torch.no_grad()
def candidates(cfg,output):
    bank,rows=training_rows(cfg);encoder=load_masked_encoder(cfg);scorer=load_model(data.paths(cfg)/'scorer'/'ner',cfg).requires_grad_(False);torch.save(generate.cohort(rows,bank['labels']['ner']),output/'sources.pt');timings=[];counts=[]
    for name in generate.ARMS:
        folder=output/name;t=generate.arm(cfg,bank,rows,encoder,scorer,name,folder);timings.append(t);accepted=torch.load(folder/'accepted_features_draw0.pt',weights_only=False)
        counts.append({'mechanism':name,'proposals':len(rows),'accepted':len(accepted['y']),'acceptance_rate':len(accepted['y'])/len(rows),'generation_seconds':t['generation_seconds'],'scoring_seconds':t['scoring_seconds']});write_csv(output/'yield.csv',counts)
    save_json(output/'timing.json',timings);return {'proposals_per_mechanism':len(rows)}


def train(cfg,output):
    raw=torch.load(root(cfg)/'prepare'/'raw.pt',mmap=True,weights_only=False);bank=data.load(cfg);labels=bank['labels']['ner'];full=torch.load(data.paths(cfg)/'scorer'/'full_features.pt',mmap=True,weights_only=False);validation=data.readout_data(bank['sets']['validation'],full['validation'],'ner')
    aug=[augmentation_features(cfg,name) for name in head_names(cfg)[1:]];records=[];budgets=[]
    for seed in cfg.augmentation.seeds:
        folder=output/f'seed{seed}';folder.mkdir(exist_ok=True);torch.manual_seed(seed);core=IndependentReadouts(cfg.tagger,768,len(labels),len(head_names(cfg))).to(cfg.runtime.device);optimizer=torch.optim.AdamW(core.parameters(),lr=cfg.train.lr,weight_decay=cfg.train.weight_decay)
        batches,plan=schedule(raw['sentence'],cfg,seed);dataset=Steps(raw,aug,batches,plan,seed);loader=DataLoader(dataset,batch_size=1,collate_fn=collate,num_workers=cfg.runtime.workers)
        best=torch.full((len(head_names(cfg)),),-torch.inf);best_epoch=torch.zeros(len(head_names(cfg)),dtype=torch.long);state={k:v.detach().cpu().clone() for k,v in core.state_dict().items()};history=[]
        for number,b in enumerate(loader):
            core.train();optimizer.zero_grad(set_to_none=True)
            with autocast(cfg):logits=core(b['x'].to(cfg.runtime.device)).float()
            loss=F.cross_entropy(logits.permute(1,2,0),b['y'].to(cfg.runtime.device).T,reduction='none').mean(-1).sum();loss.backward();clip_heads(core,cfg.train.clip);optimizer.step();epoch=b['epoch']
            if number+1<len(batches) and batches[number+1]['epoch']==epoch:continue
            metrics=metric_vectors(predict(core,validation,cfg),validation['y'],len(labels));criterion=metrics['macro_f1'];improved=criterion>best
            for k,v in core.state_dict().items():state[k][improved]=v.detach().cpu()[improved]
            best[improved]=criterion[improved];best_epoch[improved]=epoch
            for h,name in enumerate(head_names(cfg)):history.append({'head':name,'epoch':epoch,'dev_macro_f1':float(criterion[h])})
            print(f'seed{seed} epoch{epoch}: dev macro-F1 {criterion.tolist()}',flush=True)
        core.load_state_dict(state);scores=evaluate(predict(core,validation,cfg),validation,'fewnerd','fewnerd',labels,'native')
        torch.save({'state':state,'labels':len(labels),'heads':head_names(cfg),'epoch':best_epoch},folder/'model.pt');write_csv(folder/'history.csv',history)
        for h,name in enumerate(head_names(cfg)):
            records.append({'seed':seed,'head':name,'epoch':int(best_epoch[h]),**{k:float(v[h]) for k,v in scores.items()}});variant=sum(len(dataset.slices[h-1][i][1]) for i in range(len(batches))) if h else 0
            budgets.append({'seed':seed,'head':name,'optimizer_steps':len(batches),'labelled_token_presentations':int(plan['baseline_original_presentations']),'original_token_presentations':int(plan['baseline_original_presentations'])-variant,'augmented_token_presentations':variant,'augmentation_pool_size':len(aug[h-1]['y']) if h else 0})
        write_csv(output/'dev_per_seed.csv',records);write_csv(output/'budgets.csv',budgets)
    return {'trained_heads':len(records)}


@torch.no_grad()
def test(cfg,output):
    evaluation=torch.load(absolute(cfg.grid_pilot.evaluation_cache),mmap=True,weights_only=False);labels=evaluation['labels'];records=[]
    for seed in cfg.augmentation.seeds:
        saved=torch.load(root(cfg)/'train'/f'seed{seed}'/'model.pt',weights_only=False);core=IndependentReadouts(cfg.tagger,768,len(labels),len(head_names(cfg))).to(cfg.runtime.device);core.load_state_dict(saved['state']);core.eval()
        for split,item in evaluation['sets'].items():
            if split=='validation':continue
            part=item['data'];target=item['target'];logits=predict(core,part,cfg)
            for space in (['native','shared_ner'] if split=='id' else ['shared_ner']):
                scores=evaluate(logits,part,'fewnerd',target,labels,space);c,p,e=span_counts(logits.argmax(-1),part['y'],part['sentence'],labels,LABELS[target] if space=='shared_ner' else labels,shared=space=='shared_ner')
                for h,name in enumerate(head_names(cfg)):records.append({'seed':seed,'head':name,'target':target,'space':space,'span_f1':float(scores['span_f1'][h]),'accuracy':float(scores['accuracy'][h]),'precision':float(c[h]/p[h].clamp_min(1)),'recall':float(c[h]/e.clamp_min(1))})
            torch.save({'logits':logits,'gold':part['y'],'sentence':part['sentence'],'heads':head_names(cfg)},output/f'seed{seed}_{split}.pt')
    summary=[]
    for target,space in sorted({(r['target'],r['space']) for r in records}):
        baseline=torch.tensor([r['span_f1'] for r in records if (r['target'],r['space'],r['head'])==(target,space,'baseline')])
        for name in head_names(cfg):
            score=torch.tensor([r['span_f1'] for r in records if (r['target'],r['space'],r['head'])==(target,space,name)]);d=score-baseline
            summary.append({'target':target,'space':space,'head':name,'mean_span_f1':float(score.mean()),'seed_sd':float(score.std()),'paired_delta_vs_baseline':float(d.mean()),'paired_delta_seed_sd':float(d.std())})
    write_csv(output/'per_seed.csv',records);write_csv(output/'summary.csv',summary);return {'evaluations':len(records)}


def main(cfg):
    OmegaConf.resolve(cfg);configure(cfg);cfg.forge.store=str(absolute(cfg.forge.store));tracking=start_run(cfg);folder=root(cfg);folder.mkdir(parents=True,exist_ok=True);stage=cfg.stage.experiment;output=folder/stage;output.mkdir(exist_ok=True);start=time.monotonic()
    if not (folder/'timing.json').exists():save_json(folder/'timing.json',{'started_unix':time.time()})
    OmegaConf.save(cfg,folder/'config.yaml');save_json(folder/'protocol.json',{'dataset':'2400 sentences, fixed random sample of full Few-NERD scorer fitting pool; no calibration or test examples in downstream training','scorer':'Existing full-data scorer, fixed; no scorer training','generation':'One fresh proposal per labelled training word per arm; original local70, full_one_pass, full_refined mechanisms reused. Focal pieces protected, same length/positions. Raw duplicate and identical candidates retained, no fluency/quality filters. Fixed mask/fill seeds; refined initialization matches full-one-pass.',
        'acceptance':'Source scorer correct; generated token IDs differ from original; new scorer argmax differs from source label; confidence >= .7. Only focal pseudo-label enters augmentation loss. No unconditional true-label-change claim.',
        'training':'Original labels untouched; same original training pool, frozen BERT block12, same MLP, ten epochs, raw native-dev token-macro-F1 checkpoint selection; seeds0/1/2. Shared original-token order and initial head weights, equal optimizer steps and total labelled-token counts; per-arm original/augmentation mixture is proportional to pool sizes, with exact counts in budgets.csv.',
        'evaluation':'Four predeclared heads, no parameter search or test-based selection. Raw native Few-NERD and existing raw CoNLL/WNUT/WikiANN transfer tests, same metrics/mapping; matched small-data baseline, not the prior full-data baseline.'})
    print(f'Mechanism training pilot {stage}: {output}',flush=True);result={'prepare':prepare,'generate':candidates,'train':train,'test':test}[stage](cfg,output);save_json(output/'runtime.json',{'seconds':time.monotonic()-start});tracking.finish(result)
    if stage=='test':
        timing=json.loads((folder/'timing.json').read_text());timing['ended_unix']=time.time();timing['wall_seconds']=timing['ended_unix']-timing['started_unix'];save_json(folder/'timing.json',timing)
    return 0


if __name__=='__main__':
    stages=[None] if any(a.startswith('stage.experiment=') for a in sys.argv[1:]) else ['prepare','generate','train','test']
    for stage in stages:
        code=run('lion.proposal_mechanisms',sys.argv[1:]+([] if stage is None else [f'stage.experiment={stage}']),main_module='train_pilot',config_dir=str(ROOT/'conf'),config_name='mechanism_training_pilot')
        if code:sys.exit(code)
