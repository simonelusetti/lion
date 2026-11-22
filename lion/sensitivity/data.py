"""Cache four independent calibration-style fills for each test word and selector."""
import time
import torch
from omegaconf import OmegaConf
from ..diagnostics.common import absolute,save_json,write_csv
from ..encoder import load_masked_encoder
from ..partition.train import load_model
from ..certification import data,selector


def paths(cfg):return absolute(cfg.sensitivity.results)


def source_config(cfg):
    source=OmegaConf.load(absolute(cfg.sensitivity.source_config))
    source.runtime=OmegaConf.merge(source.runtime,cfg.runtime)
    return source


def source_root(cfg):return absolute(cfg.sensitivity.source_config).parent


def selectors(cfg,source,fold):
    saved=torch.load(source_root(cfg)/'selector'/data.task_names(source)[0]/fold/'model.pt',map_location='cpu',weights_only=False)
    fitted=selector.Selector(saved['dimension'],source.certification.hidden);fitted.load_state_dict(saved['state'])
    return fitted.to(source.runtime.device).eval().requires_grad_(False)


def inputs(cfg,source):
    if cfg.sensitivity.get('adapter'):
        import importlib
        yield from importlib.import_module(cfg.sensitivity.adapter).inputs(cfg,source)
        return
    root=source_root(cfg);bank=torch.load(root/'prepare'/'sentences.pt',weights_only=False)
    sets={'conll2003':bank['sets']['id']};start=max(r['sentence'] for rows in bank['sets'].values() for r in rows)+1
    transfer,domains,_=data.ner_transfer_sets(source,'ner',start)
    for split,rows in transfer.items():sets[domains[split]]=rows
    evaluated=torch.load(absolute(cfg.sensitivity.augmentation_run)/'train'/'evaluation.pt',mmap=True,weights_only=False)
    for dataset,rows in sets.items():
        split='id' if dataset=='conll2003' else 'ood_'+dataset
        part=evaluated['sets'][split]['data'];attached=data.attach_features(rows,part['features'][:,0].float())
        yield dataset,attached,part,bank


@torch.no_grad()
def generate(cfg,output,recorder=None):
    source=source_config(cfg);task=data.task_names(source)[0];encoder=load_masked_encoder(source);scorer=load_model(source_root(cfg)/'scorer'/task,source).requires_grad_(False)
    started=time.monotonic();counts=[]
    for dataset,rows,raw,bank in inputs(cfg,source):
        folder=output/dataset;folder.mkdir(parents=True,exist_ok=True)
        torch.save(raw,folder/'raw.pt')
        for fold_index,fold in enumerate(('g1','g2')):
            target=folder/(fold+'.pt')
            if cfg.sensitivity.resume and target.exists():continue
            if recorder is not None:recorder.start(dataset,fold)
            fitted=selectors(cfg,source,fold);cuts=torch.load(source_root(cfg)/'calibrate'/task/(fold+'.pt'),weights_only=False)['fractions']
            fractions=torch.tensor(cuts,device=source.runtime.device,dtype=torch.float64);_,outside=selector.label_groups(bank['labels'][task])
            seed=cfg.sensitivity.seed+fold_index*10+sum(map(ord,dataset))*100
            fill_rng=[torch.Generator(device=source.runtime.device).manual_seed(seed+j) for j in (0,1)];mask_rng=torch.Generator(device=source.runtime.device).manual_seed(seed+2)
            result={k:[] for k in ('selector','random','group','sentence','focal','dropped','available','coincide')};clock=time.monotonic()
            for batch in selector.focal_batches(data.loader(rows,source),task,source):
                ids,attention,word_ids,focal=(batch[k] for k in ('ids','attention','word_ids','focal'))
                original=batch['hidden'][batch['owner'],focal]
                group=selector.predicted_group(selector.scorer_log_probs(scorer,original,source),outside)
                scores,eligible=selector.batch_scores(fitted,batch);available=eligible.sum(-1);budget=selector.word_budget(fractions[group],available)
                keep=selector.kept_words(selector.ranks(scores,eligible),eligible,budget);dropped=available-budget
                noise=torch.rand(eligible.shape,device=ids.device,generator=mask_rng).masked_fill(~eligible,torch.inf)
                random_keep=~(eligible&(noise.argsort(-1).argsort(-1)<dropped[:,None]))
                for j,(name,word_keep) in enumerate((('selector',keep),('random',random_keep))):
                    drop=data.word_to_pieces(~word_keep,word_ids,False)
                    fills=selector.independent_fills(encoder,ids,attention,drop,cfg.sensitivity.draws,bank['special'],source,fill_rng[j])
                    if recorder is not None:recorder.capture(batch,word_keep,fills,name)
                    features=selector.draw_features(encoder,ids,attention,word_ids,focal,fills,source,raw=original)
                    result[name].append(features.cpu().half())
                for k,value in [('group',group),('sentence',batch['sentence']),('focal',focal),('dropped',dropped),('available',available),('coincide',(keep==random_keep).all(-1))]:result[k].append(value.cpu())
            result={k:torch.cat(v) for k,v in result.items()}
            if recorder is not None:result.update(recorder.finish())
            result['seconds']=time.monotonic()-clock;torch.save(result,target)
            counts.append({'dataset':dataset,'selector':fold,'rows':len(result['group']),'seconds':result['seconds']})
            write_csv(output/'generation.csv',counts);print(f'{dataset}/{fold}: {len(result["group"])} test words, {result["seconds"]:.1f}s',flush=True)
    save_json(output/'timing.json',{'generation_wall_seconds':time.monotonic()-started});return {'cached_conditions':len(list(output.glob('*/g*.pt')))}
