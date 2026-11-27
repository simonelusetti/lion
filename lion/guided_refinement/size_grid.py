"""Tenfold proposal-budget grid with matched downstream classifiers."""
import json
import sys
import time
import torch
from forge import run,start_run
from omegaconf import OmegaConf
from ..certification import data,selector
from ..partition.train import load_model
from ..encoder import load_masked_encoder
from ..diagnostics.common import ROOT,absolute,save_json,write_csv
from ..runtime import configure
from ..proposal_mechanisms import train_pilot as downstream
from . import generate,train_pilot


@torch.no_grad()
def fraction_arm(cfg,bank,rows,encoder,scorer,rule,output):
    output.mkdir(parents=True,exist_ok=True)
    nearby=rule.startswith('nearby');fraction=int(rule.removeprefix('nearby' if nearby else 'sentence'))/100
    start=time.monotonic();timings=[]
    for draw in range(cfg.mechanisms.draws):
        rank_rng=torch.Generator(device=cfg.runtime.device).manual_seed(cfg.generators.seed+(0 if nearby else 100)+draw*100003)
        fill_rng=torch.Generator(device=cfg.runtime.device).manual_seed(cfg.generators.seed+(0 if nearby else 100)+round(fraction*100)+10+draw*100003)
        features=[];labels=[];anchors=[];inputs=[];lengths=[];confidences=[];raw_inputs=[];raw_lengths=[];raw_probability=[];raw_original=[];all_vectors=[];all_labels=[];cursor=0;generation_seconds=scoring_seconds=0.
        for batch in selector.focal_batches(data.loader(rows,cfg),'ner',cfg):
            ids,attention,words,focal,gold=(batch[k] for k in ('ids','attention','word_ids','focal','gold'))
            positions=torch.arange(int(batch['counts'].max()),device=ids.device)[None]
            distance=(positions-focal[:,None]).abs();eligible=(positions<batch['counts'][:,None])&(distance>0)
            if nearby:eligible&=distance<=cfg.generators.radius
            count=torch.floor(eligible.sum(-1)*fraction+.5).long()
            noise=torch.rand(eligible.shape,device=ids.device,generator=rank_rng).masked_fill(~eligible,torch.inf)
            drop=data.word_to_pieces(eligible&(noise.argsort(-1).argsort(-1)<count[:,None]),words,False)
            generate.previous.sync(cfg);clock=time.monotonic()
            fills=selector.independent_fills(encoder,ids,attention,drop,1,bank['special'],cfg,fill_rng)
            generate.previous.sync(cfg);generation_seconds+=time.monotonic()-clock;clock=time.monotonic()
            vectors=selector.draw_features(encoder,ids,attention,words,focal,fills,cfg)[:,0]
            probability=selector.scorer_log_probs(scorer,vectors,cfg).exp();confidence,prediction=probability.max(-1)
            original=selector.scorer_log_probs(scorer,batch['hidden'][batch['owner'],focal],cfg).argmax(-1)
            accept=(original==gold)&(prediction!=gold)&(confidence>=cfg.guided.tau)&~generate.exact_original(fills[:,0],ids,attention)
            generate.previous.sync(cfg);scoring_seconds+=time.monotonic()-clock
            if cfg.mechanisms.get('cache_all_features',False):all_vectors.append(vectors.cpu().half());all_labels.append(prediction.cpu())
            features.append(vectors[accept].cpu().half());labels.append(prediction[accept].cpu());anchors.append(accept.nonzero().flatten().cpu()+cursor);confidences.append(confidence[accept].cpu())
            inputs.append(fills[:,0][accept][attention[accept]].cpu().int());lengths.append(attention[accept].sum(-1).cpu());cursor+=len(ids)
            if cfg.get('population',{}).get('save_raw',False):
                raw_inputs.append(fills[:,0][attention].cpu().int());raw_lengths.append(attention.sum(-1).cpu());raw_probability.append(probability.cpu());raw_original.append(original.cpu())
        saved={'features':torch.cat(features),'y':torch.cat(labels),'anchor_id':torch.cat(anchors),'confidence':torch.cat(confidences),'input_ids':torch.cat(inputs),'lengths':torch.cat(lengths)}
        torch.save(saved,output/f'accepted_features_draw{draw}.pt')
        if cfg.mechanisms.get('cache_all_features',False):torch.save({'features':torch.cat(all_vectors),'y':torch.cat(all_labels),'anchor_id':torch.arange(len(rows))},output/f'all_features_draw{draw}.pt')
        if cfg.get('population',{}).get('save_raw',False):
            length=torch.cat(raw_lengths);offsets=torch.cat((torch.zeros(1,dtype=torch.long),length.cumsum(0)))
            torch.save({'state_input_ids':torch.cat(raw_inputs)[None],'offsets':offsets,'state_probabilities':torch.cat(raw_probability)[:,None],'source_prediction':torch.cat(raw_original)},output/f'replicate{draw}.pt')
        timing={'draw':draw,'proposals':len(rows),'accepted':len(saved['y']),'generation_seconds':generation_seconds,'scoring_seconds':scoring_seconds};timings.append(timing)
        print(f'{rule} draw{draw}: {timing}',flush=True)
    return {'arm':rule,'trajectories':len(rows)*cfg.mechanisms.draws,'wall_seconds':time.monotonic()-start,'replicates':timings}


@torch.no_grad()
def candidates(cfg,output):
    bank,rows=train_pilot.anchors(cfg)
    meta=torch.load(downstream.root(cfg)/'prepare'/'sources_targets.pt',weights_only=False)
    encoder=load_masked_encoder(cfg);scorer=load_model(data.paths(cfg)/'scorer'/'ner',cfg).requires_grad_(False)
    timings=[];counts=[]
    for rule in cfg.mechanism_pilot.arms:
        if rule in cfg.guided.arms:
            timing=generate.arm(cfg,bank,rows,meta,encoder,scorer,rule,cfg.mechanisms.draws,output/rule)
        else:timing=fraction_arm(cfg,bank,rows,encoder,scorer,rule,output/rule)
        accepted=sum(len(torch.load(output/rule/f'accepted_features_draw{draw}.pt',weights_only=False)['y']) for draw in range(cfg.mechanisms.draws))
        proposals=len(rows)*cfg.mechanisms.draws
        counts.append({'arm':rule,'proposals':proposals,'accepted':accepted,'acceptance_rate':accepted/proposals,'augmented_pool_share':accepted/(60321+accepted),'wall_seconds':timing['wall_seconds']})
        timings.append(timing);write_csv(output/'yield.csv',counts);save_json(output/'timing.json',timings)
    return {'arms':len(timings),'proposals_per_arm':proposals}


def main(cfg):
    OmegaConf.resolve(cfg);configure(cfg);cfg.forge.store=str(absolute(cfg.forge.store));tracking=start_run(cfg)
    folder=downstream.root(cfg);folder.mkdir(parents=True,exist_ok=True);stage=cfg.stage.experiment;output=folder/stage;output.mkdir(exist_ok=True)
    if not (folder/'timing.json').exists():save_json(folder/'timing.json',{'started_unix':time.time()})
    OmegaConf.save(cfg,folder/'config.yaml')
    save_json(folder/'protocol.json',{
        'data':'Same 2400 fit-only Few-NERD sentences and 60321 original labels as previous downstream pilot. No calibration/dev/test observations enter augmented training.',
        'generation':'40000 uniform-word training anchors sampled with replacement, five draws per anchor: 200000 attempts per arm. Same anchors across all 11 rules. Five existing guided rules plus nearby +/-5 and whole-sentence replacement at 20/50/70 percent. Fractions use half-up word counts, fixed focal pieces, one parallel MLM pass; guided procedures unchanged. No redraws or deduplication.',
        'acceptance':'Fixed full-data scorer: original prediction correct, exact non-padding input token IDs changed, final prediction differs from original gold and confidence >=0.7. No fluency filter or certification gate. Only the accepted focal pseudo-label enters loss; prediction changes are not verified gold-label changes.',
        'training':'New baseline plus 11 augmented classifiers, three matched seeds; same original pool, initialization, original ordering, optimizer-step and total-labelled-token budgets as previous pilot. Augmented batches mix proportional to pool size; original presentation counts differ and are logged. Existing frozen block12 plus MLP. Raw native dev token macro-F1 selects checkpoints.',
        'evaluation':'All arms predeclared, no test-based selection; identical raw native Few-NERD and shared-label CoNLL/WNUT/WikiANN tests. Save per-seed results, mean/SD and paired differences. Main guided lambda0.01 comparisons also retained.'})
    print(f'Augmentation size grid {stage}: {output}',flush=True);started=time.monotonic()
    result={'prepare':train_pilot.prepare,'generate':candidates,'train':downstream.train,'test':train_pilot.evaluate}[stage](cfg,output)
    save_json(output/'runtime.json',{'seconds':time.monotonic()-started});tracking.finish(result)
    if stage=='test':
        timing=json.loads((folder/'timing.json').read_text());timing['ended_unix']=time.time();timing['wall_seconds']=timing['ended_unix']-timing['started_unix'];save_json(folder/'timing.json',timing)
    return 0


if __name__=='__main__':
    stages=[None] if any(a.startswith('stage.experiment=') for a in sys.argv[1:]) else ['prepare','generate','train','test']
    for stage in stages:
        code=run('lion.guided_refinement',sys.argv[1:]+([] if stage is None else [f'stage.experiment={stage}']),main_module='size_grid',config_dir=str(ROOT/'conf'),config_name='augmentation_size_grid')
        if code:sys.exit(code)
