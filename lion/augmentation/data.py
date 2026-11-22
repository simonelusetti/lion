"""Frozen source artifacts, opposite-fold single-draw variants and diagnostics."""
import time
import torch
from omegaconf import OmegaConf
from ..data import signature
from ..diagnostics.common import absolute,save_json,write_csv
from ..encoder import load_masked_encoder
from ..partition.train import load_model
from ..certification import data,selector,calibrate,certify

ARMS=['baseline','selector_augmented','random_augmented']


def paths(cfg):
    if cfg.augmentation.get('run_path') is not None:return absolute(cfg.augmentation.run_path)
    key=signature(OmegaConf.create({'protocol':'label-augmentation-v1','source':str(data.paths(cfg)),
                                   'augmentation':cfg.augmentation,'train':cfg.train,'tagger':cfg.tagger}))
    return absolute(cfg.augmentation.results)/key


def training(cfg):
    bank=data.load(cfg);full=torch.load(data.paths(cfg)/'scorer'/'full_features.pt',mmap=True,weights_only=False)
    raw=selector.raw_rows(cfg,bank);parts=[];selected=[];metadata=[]
    for applied in ('g2','g1'):
        include=torch.tensor([r['fold']==applied for r in bank['sets']['train']]);rows=[r for r in raw if r['fold']==applied]
        counts=torch.tensor([len(r['focals']) for r in bank['sets']['train']])
        parts.append(data.readout_data(rows,full['train'][include.repeat_interleave(counts)],'ner'));selected.extend(rows)
        metadata.extend({'sentence':r['sentence'],'source_id':r['source_id'],'fold':applied,'words':int((r['gold']['ner']>=0).sum())} for r in rows)
    combined={k:torch.cat([p[k] for p in parts]) for k in parts[0]}
    return bank,selected,combined,metadata


def prepare(cfg,output):
    source=data.paths(cfg);source.mkdir(parents=True,exist_ok=True);OmegaConf.save(cfg,source/'config.yaml')
    target=source/'prepare';target.mkdir(exist_ok=True)
    if not (cfg.augmentation.resume and (target/'sentences.pt').exists()):data.prepare(cfg,target)
    bank=data.load(cfg)
    rows=[]
    for group in ('scorer_train','scorer_validation','g1','g2'):
        part=[r for r in bank['sets']['train'] if r['fold']==group]
        rows.append({'partition':group,'sentences':len(part),'words':sum(int((r['gold']['ner']>=0).sum()) for r in part)})
    write_csv(output/'partitions.csv',rows)
    write_csv(output/'training_sentences.csv',[{'sentence':r['sentence'],'source_id':r['source_id'],'fold':r['fold'],'words':int((r['gold']['ner']>=0).sum())} for r in bank['sets']['train'] if r['fold'] in ('g1','g2')])
    save_json(output/'source.json',{'run':str(source),'training':'g1 union g2; scorer_train and scorer_validation excluded','source_preset':'full CoNLL-2003' if cfg.certification.train_sentence_cap is None else 'conll2003_pilot','retrained_prerequisites':True})
    return {'training_sentences':sum(r['sentences'] for r in rows if r['partition'] in ('g1','g2'))}


@torch.no_grad()
def bounds(cfg,output):
    """Reuse the word-rank held-out certificate, without the unrelated chained CRT diagnostics."""
    task=data.task_names(cfg)[0];bank=data.load(cfg);raw={r['sentence']:r for r in selector.raw_rows(cfg,bank)};encoder=load_masked_encoder(cfg)
    scorer=load_model(data.paths(cfg)/'scorer'/task,cfg).requires_grad_(False);names,outside=selector.label_groups(bank['labels'][task]);records=[]
    for fold in ('g1','g2'):
        fitted=selector.load_selector(cfg,task,fold).requires_grad_(False)
        saved=torch.load(data.paths(cfg)/'calibrate'/task/f'{fold}.pt',weights_only=False)
        held=[raw[s] for s in saved['test_sentences'][:cfg.certification.test_documents]]
        sampled=calibrate.sample_group_rows(held,task,scorer,outside,cfg,cfg.seed+1);arrays={}
        for group,name in enumerate(names):
            rows=certify.subgroup(sampled,group);rho=saved['fractions'][group]
            loss=certify.held_out_losses(encoder,fitted,scorer,rows,task,rho,bank['special'],cfg);bound=certify.certificate(loss,rho,cfg)
            records.append({'selector':fold,'group':name,'rho':rho,**bound,'units':'nats','label_error_probability_bound':None})
            arrays[name]={'losses':loss,'sentence':[r['sentence'] for r in rows],'focal':[int(r['focals'].item()) for r in rows]}
        torch.save(arrays,output/f'{fold}.pt')
    write_csv(output/'bounds.csv',records);return {'groups':len(records)}


@torch.no_grad()
def generate(cfg,output):
    bank,rows,raw,metadata=training(cfg);encoder=load_masked_encoder(cfg);scorer=load_model(data.paths(cfg)/'scorer'/'ner',cfg).requires_grad_(False)
    names,outside=selector.label_groups(bank['labels']['ner']);started=time.monotonic();parts=[];diagnostics=[]
    for fold,opposite in (('g1','g2'),('g2','g1')):
        target=output/f'{fold}.pt'
        if cfg.augmentation.resume and target.exists():parts.append(torch.load(target,weights_only=False));continue
        fitted=selector.load_selector(cfg,'ner',fold).requires_grad_(False)
        calibrated=torch.load(data.paths(cfg)/'calibrate'/'ner'/f'{fold}.pt',weights_only=False)
        fractions=torch.tensor(calibrated['fractions'],device=cfg.runtime.device,dtype=torch.float64)
        eligible_rows=[r for r in rows if r['fold']==opposite]
        seed=cfg.augmentation.generation_seed+(0 if fold=='g1' else 10)
        fill_rng=[torch.Generator(device=cfg.runtime.device).manual_seed(seed+j) for j in (0,1)]
        mask_rng=torch.Generator(device=cfg.runtime.device).manual_seed(seed+2)
        result={k:[] for k in ('selector','random','y','sentence','focal','group','drop_count','changed_selector','changed_random','disagree_selector','disagree_random')}
        fold_start=time.monotonic()
        for batch in selector.focal_batches(data.loader(eligible_rows,cfg), 'ner',cfg):
            ids,attention,word_ids,focal,gold=(batch[k] for k in ('ids','attention','word_ids','focal','gold'))
            original=batch['hidden'][batch['owner'],focal];group=selector.predicted_group(selector.scorer_log_probs(scorer,original,cfg),outside)
            scores,eligible=selector.batch_scores(fitted,batch);available=eligible.sum(-1);budget=selector.word_budget(fractions[group],available)
            keep=selector.kept_words(selector.ranks(scores,eligible),eligible,budget);count=available-budget
            # Uniform random context subsets, conditional only on the matched drop count.
            noise=torch.rand(eligible.shape,device=ids.device,generator=mask_rng).masked_fill(~eligible,torch.inf)
            drop_random=eligible&(noise.argsort(-1).argsort(-1)<count[:,None]);keep_random=~drop_random
            result['y'].append(gold.cpu());result['sentence'].append(batch['sentence']);result['focal'].append(focal.cpu());result['group'].append(group.cpu());result['drop_count'].append(count.cpu())
            for j,(arm,word_keep) in enumerate((('selector',keep),('random',keep_random))):
                dropped=data.word_to_pieces(~word_keep,word_ids,False)
                fills=selector.independent_fills(encoder,ids,attention,dropped,1,bank['special'],cfg,fill_rng[j])
                values=selector.draw_features(encoder,ids,attention,word_ids,focal,fills,cfg,raw=original)[:,0]
                # Diagnostics follow the single draw; they never select or regenerate it.
                disagreement=selector.scorer_log_probs(scorer,values,cfg).argmax(-1)!=gold
                result[arm].append(values.cpu().half());result['changed_'+arm].append((fills[:,0]!=ids).any(-1).cpu());result['disagree_'+arm].append(disagreement.cpu())
        result={k:torch.cat(v) for k,v in result.items()};result['seconds']=time.monotonic()-fold_start;torch.save(result,target);parts.append(result)
        print(f'{fold} applied to {opposite}: {len(result["y"])} one-draw variants per arm, {result["seconds"]:.1f}s',flush=True)
    values={k:torch.cat([p[k] for p in parts]) for k in parts[0] if isinstance(parts[0][k],torch.Tensor)}
    focals=torch.cat([r['focals'][r['gold']['ner']>=0] for r in rows])
    if not(torch.equal(values['y'],raw['y']) and torch.equal(values['sentence'],raw['sentence']) and torch.equal(values['focal'],focals)):raise ValueError('Original/variant word keys are not aligned.')
    torch.save({'raw':raw['features'][:,0].half(),**values},output/'features.pt')
    for arm in ('selector','random'):
        diagnostics.append({'arm':arm+'_augmented','variants':len(values['y']),'original_words':len(raw['y']),'variant_token_share':.5,
                            'scorer_disagreement':float(values['disagree_'+arm].float().mean()),'changed_input_share':float(values['changed_'+arm].float().mean()),'nonempty_drop_share':float((values['drop_count']>0).float().mean())})
        for group,name in enumerate(names):
            mask=values['group']==group
            diagnostics.append({'arm':arm+'_augmented','group':name,'variants':int(mask.sum()),'original_words':int(mask.sum()),'variant_token_share':.5,
                                'scorer_disagreement':float(values['disagree_'+arm][mask].float().mean()),'changed_input_share':float(values['changed_'+arm][mask].float().mean()),'nonempty_drop_share':float((values['drop_count'][mask]>0).float().mean())})
    write_csv(output/'diagnostics.csv',diagnostics);save_json(output/'timing.json',{'generation_wall_seconds':time.monotonic()-started,'fold_generation_seconds':sum(p['seconds'] for p in parts),'variants_per_arm':len(values['y'])})
    return {'variants_per_arm':len(values['y'])}
