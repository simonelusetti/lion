"""Generate once per fraction; thresholds reuse exactly the same fills."""
import math,json
import torch
from scipy.stats import beta
from transformers import AutoTokenizer
from ..certification import data,selector
from ..partition.train import load_model
from ..encoder import load_masked_encoder
from ..diagnostics.common import absolute,read_csv,write_csv


def root(cfg):return absolute(cfg.grid_pilot.results)


def cells(cfg):return [{'name':f'nearby{round(f*100)}_tau{t:.1f}','fraction':f,'tau':t} for f in cfg.grid_pilot.fractions for t in cfg.grid_pilot.thresholds]


@torch.no_grad()
def experiment(cfg,output):
    bank=data.load(cfg);rows=[r for r in selector.raw_rows(cfg,bank) if r['fold']=='fit'];scorer=load_model(data.paths(cfg)/'scorer'/'ner',cfg).requires_grad_(False);encoder=load_masked_encoder(cfg)
    tok=AutoTokenizer.from_pretrained(cfg.encoder.name,local_files_only=True);tokens=tok.convert_ids_to_tokens(list(range(len(tok))))
    for fi,fraction in enumerate(cfg.grid_pilot.fractions):
        saved={k:[] for k in ('features','y','confidence','sentence','focal','original_gold')};rank_rng=torch.Generator(device=cfg.runtime.device).manual_seed(cfg.grid_pilot.seed);fill_rng=torch.Generator(device=cfg.runtime.device).manual_seed(cfg.grid_pilot.seed+10+fi);proposals=0
        for b in selector.focal_batches(data.loader(rows,cfg),'ner',cfg):
            ids,attention,words,focal,gold=(b[k] for k in ('ids','attention','word_ids','focal','gold'));p=torch.arange(int(b['counts'].max()),device=ids.device)[None];distance=(p-focal[:,None]).abs();eligible=(p<b['counts'][:,None])&(distance>0)&(distance<=cfg.grid_pilot.radius)
            count=torch.floor(eligible.sum(-1)*fraction+.5).long();noise=torch.rand(eligible.shape,device=ids.device,generator=rank_rng).masked_fill(~eligible,torch.inf);drop_words=eligible&(noise.argsort(-1).argsort(-1)<count[:,None]);drop=data.word_to_pieces(drop_words,words,False)
            fills=selector.independent_fills(encoder,ids,attention,drop,1,bank['special'],cfg,fill_rng);vectors=selector.draw_features(encoder,ids,attention,words,focal,fills,cfg)[:,0]
            confidence,new=selector.scorer_log_probs(scorer,vectors,cfg).exp().max(-1);original=selector.scorer_log_probs(scorer,b['hidden'][b['owner'],focal],cfg).argmax(-1)
            # A token-id change is necessary; retained rows get word-level decoding below.
            accepted=(original==gold)&(new!=original)&(confidence>=min(cfg.grid_pilot.thresholds))&(fills[:,0]!=ids).any(-1);proposals+=len(gold)
            # Alternative WordPiece spellings of the same text are not actual context changes.
            for j in accepted.nonzero().flatten().tolist():
                changed=False
                for w in drop_words[j].nonzero().flatten().tolist():
                    positions=words[j]==w
                    a=''.join(tokens[i].removeprefix('##') for i in ids[j,positions].tolist());c=''.join(tokens[i].removeprefix('##') for i in fills[j,0,positions].tolist())
                    changed|=a!=c
                if not changed:accepted[j]=False
            for key,value in [('features',vectors[accepted].cpu().half()),('y',new[accepted].cpu()),('confidence',confidence[accepted].cpu()),('sentence',b['sentence'][accepted.cpu()]),('focal',focal[accepted].cpu()),('original_gold',gold[accepted].cpu())]:saved[key].append(value)
        result={k:torch.cat(v) for k,v in saved.items()};result['proposals']=proposals;torch.save(result,output/f'nearby{round(fraction*100)}.pt');print(f'fraction {fraction}: {len(result["y"])} fit candidates pass original-correct/flip and tau>=.7',flush=True)
    return {'fractions':len(cfg.grid_pilot.fractions)}


def calibrate(cfg,output):
    error=torch.load(absolute(cfg.label_change.results)/'generate'/'error_sample.pt',weights_only=False);rows=[];tail=cfg.label_change.alpha/2;n=len(error['gold'])
    for cell in cells(cfg):
        proposal=read_csv(absolute(cfg.grid_pilot.heldout_generators)/f"nearby_{round(cell['fraction']*100)}_proposals.csv");tau=cell['tau'];k=int(((error['prediction']!=error['gold'])&(error['confidence']>=tau)).sum());m=len(proposal)
        accepted=sum(r['original_correct']=='True' and r['context_changed']=='True' and r['original_prediction']!=r['proposed'] and float(r['confidence'])>=tau for r in proposal)
        e=k/n;a=accepted/m;e_U=float(beta.ppf(1-tail,k+1,n-k)) if k<n else 1.;a_L=float(beta.ppf(tail,accepted,m-accepted+1)) if accepted else 0.
        rows.append({**cell,'heldout_proposals':m,'heldout_accepted':accepted,'a':a,'error_draws':n,'high_confidence_errors':k,'e':e,'a_over_e':a/e if e else None,'e_U':e_U,'a_L':a_L,'ideal_sampler_pi_L':max(0.,1-e_U/a_L) if a_L>0 else 0.,'delta_max':(1-cfg.label_change.desired_precision)*a_L-e_U,'pointwise_joint_confidence':1-cfg.label_change.alpha,'certification_used_as_gate':False})
    write_csv(output/'calibration.csv',rows);return {'configurations':len(rows)}
