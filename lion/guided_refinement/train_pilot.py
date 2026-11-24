"""Matched downstream pilot using fresh training-only guided trajectories."""
import json
import sys
import time
import torch
from forge import run, start_run
from omegaconf import OmegaConf
from ..runtime import configure
from ..diagnostics.common import ROOT, save_json, write_csv, read_csv
from ..certification import data, selector
from ..partition.train import load_model
from ..encoder import load_masked_encoder
from ..label_change.data import sampled_rows
from ..proposal_mechanisms import train_pilot as downstream
from . import generate


def anchors(cfg):
    selected=set(json.loads((downstream.root(cfg)/'prepare'/'subset.json').read_text())['source_ids'])
    bank=data.load(cfg)
    rows=[r for r in selector.raw_rows(cfg,bank) if r['source_id'] in selected]
    return bank,sampled_rows(rows,cfg.guided_training.anchors,cfg.guided_training.anchor_seed)


def prepare(cfg,output):
    result=downstream.prepare(cfg,output)
    bank,rows=anchors(cfg)
    meta=generate.target_metadata(cfg,bank,rows)
    torch.save(meta,output/'sources_targets.pt')
    result.update(anchors=len(rows),supported_alternative_anchors=int(meta['supported_alternative'].sum()))
    return result


@torch.no_grad()
def candidates(cfg,output):
    bank,rows=anchors(cfg)
    meta=torch.load(downstream.root(cfg)/'prepare'/'sources_targets.pt',weights_only=False)
    encoder=load_masked_encoder(cfg)
    scorer=load_model(data.paths(cfg)/'scorer'/'ner',cfg).requires_grad_(False)
    timings=[];counts=[]
    for rule in cfg.mechanism_pilot.arms:
        timing=generate.arm(cfg,bank,rows,meta,encoder,scorer,rule,cfg.mechanisms.draws,output/rule)
        timings.append(timing)
        accepted=sum(len(torch.load(output/rule/f'accepted_features_draw{draw}.pt',weights_only=False)['y']) for draw in range(cfg.mechanisms.draws))
        proposals=len(rows)*cfg.mechanisms.draws
        counts.append({'arm':rule,'proposals':proposals,'accepted':accepted,'acceptance_rate':accepted/proposals,'wall_seconds':timing['wall_seconds']})
        write_csv(output/'yield.csv',counts)
        save_json(output/'timing.json',timings)
    return {'arms':len(timings),'proposals_per_arm':proposals}


def evaluate(cfg,output):
    result=downstream.test(cfg,output)
    records=read_csv(output/'per_seed.csv');comparisons=[]
    for target,space in sorted({(r['target'],r['space']) for r in records}):
        selected=[r for r in records if (r['target'],r['space'])==(target,space)]
        for reference in ('baseline','uniform'):
            differences=torch.tensor([float(next(r['span_f1'] for r in selected if r['head']==cfg.guided.main_arm and int(r['seed'])==seed))-float(next(r['span_f1'] for r in selected if r['head']==reference and int(r['seed'])==seed)) for seed in cfg.augmentation.seeds])
            comparisons.append({'target':target,'space':space,'arm':cfg.guided.main_arm,'reference':reference,'paired_span_f1_difference':float(differences.mean()),'seed_sd':float(differences.std())})
    write_csv(output/'primary_comparisons.csv',comparisons)
    return result


def main(cfg):
    OmegaConf.resolve(cfg);configure(cfg)
    from ..diagnostics.common import absolute
    cfg.forge.store=str(absolute(cfg.forge.store));tracking=start_run(cfg)
    folder=downstream.root(cfg);folder.mkdir(parents=True,exist_ok=True)
    stage=cfg.stage.experiment;output=folder/stage;output.mkdir(exist_ok=True)
    if not (folder/'timing.json').exists():save_json(folder/'timing.json',{'started_unix':time.time()})
    OmegaConf.save(cfg,folder/'config.yaml')
    save_json(folder/'protocol.json',{
        'originals':'Same fixed 2400 fit-only Few-NERD sentences as mechanism_training_pilot; calibration and test excluded.',
        'generation':'4000 uniform-word training anchors sampled with replacement, five trajectories per anchor per rule. Fixed full-data scorer and MLM; unchanged guided generation, fixed targets per trajectory, all five rules. Cache final accepted focal representations; retain duplicate attempts.',
        'acceptance':'Source prediction correct, generated token IDs differ, final prediction differs from gold and confidence >= 0.7. Final scorer label is the focal pseudo-label; other variant tokens excluded from loss. These are scorer prediction changes, not verified true-label changes.',
        'training':'Baseline plus five augmented heads, seeds 0/1/2. Same originals, initialization, shared original ordering, fixed optimizer steps and labelled-token budget; mixtures proportional to original and accepted-variant pool sizes. Original presentation counts differ by mixture and are logged. Existing frozen BERT block12 and MLP recipe; native raw-dev token macro-F1 checkpoint selection.',
        'evaluation':'All six predeclared heads evaluated on identical untouched cached Few-NERD, CoNLL, WNUT and WikiANN tests, existing metrics/mappings. Main guided rule lambda0.01 versus baseline and uniform; no test-based selection. This is a small downstream pilot, not full-data student training.'})
    print(f'Guided downstream pilot {stage}: {output}',flush=True)
    started=time.monotonic()
    result={'prepare':prepare,'generate':candidates,'train':downstream.train,'test':evaluate}[stage](cfg,output)
    save_json(output/'runtime.json',{'seconds':time.monotonic()-started});tracking.finish(result)
    if stage=='test':
        timing=json.loads((folder/'timing.json').read_text());timing['ended_unix']=time.time();timing['wall_seconds']=timing['ended_unix']-timing['started_unix'];save_json(folder/'timing.json',timing)
    return 0


if __name__=='__main__':
    stages=[None] if any(a.startswith('stage.experiment=') for a in sys.argv[1:]) else ['prepare','generate','train','test']
    for stage in stages:
        code=run('lion.guided_refinement',sys.argv[1:]+([] if stage is None else [f'stage.experiment={stage}']),main_module='train_pilot',config_dir=str(ROOT/'conf'),config_name='guided_training_pilot')
        if code:sys.exit(code)
