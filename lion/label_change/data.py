"""Original features and one neighbouring-word replacement per focal; no selectors."""
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from ..certification import data,selector
from ..partition.train import fit,predict,load_model
from ..encoder import load_masked_encoder
from ..diagnostics.common import absolute,write_csv,save_json


def root(cfg):return absolute(cfg.label_change.results)


def prepare(cfg,output):
    source=absolute(cfg.label_change.source_bank);bank=torch.load(source/'prepare'/'sentences.pt',weights_only=False)
    rows=bank['sets']['train'];chosen=torch.randperm(len(rows),generator=torch.Generator().manual_seed(cfg.seed))[:cfg.certification.train_sentence_cap].sort().values
    mask=torch.zeros(len(rows),dtype=torch.bool);mask[chosen]=True;length=torch.tensor([len(r['focals']) for r in rows]);bank['sets']['train']=[rows[i] for i in chosen.tolist()]
    selected=bank['sets']['train'];order=torch.randperm(len(selected),generator=torch.Generator().manual_seed(cfg.label_change.seed));held=set(order[:round(len(selected)*cfg.label_change.calibration_fraction)].tolist())
    for i,r in enumerate(selected):r['fold']='calibration' if i in held else 'fit'
    source_out=data.paths(cfg);(source_out/'prepare').mkdir(parents=True,exist_ok=True);(source_out/'scorer').mkdir(exist_ok=True)
    torch.save(bank,source_out/'prepare'/'sentences.pt');full=torch.load(source/'scorer'/'full_features.pt',mmap=True,weights_only=False)
    torch.save({'train':full['train'][mask.repeat_interleave(length)],'validation':full['validation']},source_out/'scorer'/'full_features.pt');OmegaConf.save(cfg,source_out/'config.yaml')
    records=[{'role':role,'sentences':sum(r['fold']==role for r in selected),'words':sum(len(r['focals']) for r in selected if r['fold']==role)} for role in ('fit','calibration')]
    write_csv(output/'partitions.csv',records);write_csv(output/'training_sentences.csv',[{'sentence':r['sentence'],'source_id':r['source_id'],'role':r['fold']} for r in selected]);return {'fit_sentences':records[0]['sentences']}


def parts(cfg):
    bank=data.load(cfg);full=torch.load(data.paths(cfg)/'scorer'/'full_features.pt',mmap=True,weights_only=False);rows=bank['sets']['train'];length=torch.tensor([len(r['focals']) for r in rows]);result={}
    for role in ('fit','calibration'):
        keep=torch.tensor([r['fold']==role for r in rows]);result[role]=data.readout_data([r for r in rows if r['fold']==role],full['train'][keep.repeat_interleave(length)],'ner')
    result['validation']=data.readout_data(bank['sets']['validation'],full['validation'],'ner');return bank,result


def scorer(cfg,output):
    bank,p=parts(cfg);fit({'train':p['fit'],'validation':p['validation']},len(bank['labels']['ner']),cfg,cfg.seed,output/'ner');return {'scorers':1}


def sampled_rows(rows,n,seed):
    # IID uniform-word draws from a fixed held-out empirical population; repeated draws are intentional.
    lengths=torch.tensor([len(r['focals']) for r in rows]);ends=lengths.cumsum(0);starts=ends-lengths
    indices=torch.randint(int(ends[-1]),(n,),generator=torch.Generator().manual_seed(seed));owners=torch.searchsorted(ends,indices,right=True)
    result=[]
    for index,owner in zip(indices.tolist(),owners.tolist()):
        row=rows[owner];f=torch.tensor([index-int(starts[owner])]);result.append({**row,'focals':f,'gold':{'ner':row['gold']['ner'][f]}})
    return result


@torch.no_grad()
def proposals(cfg,rows,bank,encoder,scorer,seed):
    rng=torch.Generator(device=cfg.runtime.device).manual_seed(seed)
    tok=AutoTokenizer.from_pretrained(cfg.encoder.name,local_files_only=True);tokens=tok.convert_ids_to_tokens(list(range(len(tok))))
    records=[];features=[];labels=[]
    for batch in selector.focal_batches(data.loader(rows,cfg),'ner',cfg):
        ids,attention,words,focal,gold=(batch[k] for k in ('ids','attention','word_ids','focal','gold'));positions=torch.arange(int(batch['counts'].max()),device=ids.device)[None]
        distance=(positions-focal[:,None]).abs();eligible=(positions<batch['counts'][:,None])&(distance>=1)&(distance<=5);exists=eligible.any(-1)
        chance=torch.rand(eligible.shape,device=ids.device,generator=rng).masked_fill(~eligible,-1);neighbour=chance.argmax(-1);drop=(words==neighbour[:,None])&exists[:,None]
        fills=selector.independent_fills(encoder,ids,attention,drop,1,bank['special'],cfg,rng);vectors=selector.draw_features(encoder,ids,attention,words,focal,fills,cfg)[:,0]
        probability=selector.scorer_log_probs(scorer,vectors,cfg).exp();confidence,prediction=probability.max(-1)
        for j in range(len(focal)):
            pieces=drop[j];before=''.join(tokens[i].removeprefix('##') for i in ids[j,pieces].tolist());after=''.join(tokens[i].removeprefix('##') for i in fills[j,0,pieces].tolist());changed=bool(exists[j]) and before!=after
            accept=changed and int(prediction[j])!=int(gold[j]) and float(confidence[j])>=cfg.label_change.tau
            records.append({'sentence':int(batch['sentence'][j]),'focal':int(focal[j]),'neighbour':int(neighbour[j]) if exists[j] else -1,'distance':abs(int(neighbour[j])-int(focal[j])) if exists[j] else None,'original_neighbour':before,'replacement_neighbour':after,'gold':int(gold[j]),'proposed':int(prediction[j]),'confidence':float(confidence[j]),'changed':changed,'accepted':accept})
            if accept:features.append(vectors[j].cpu().half());labels.append(int(prediction[j]))
    return records,{'features':torch.stack(features) if features else torch.empty(0,768,dtype=torch.float16),'y':torch.tensor(labels,dtype=torch.long)}


@torch.no_grad()
def generate(cfg,output):
    bank,p=parts(cfg);rows=selector.raw_rows(cfg,bank);scorer=load_model(data.paths(cfg)/'scorer'/'ner',cfg).requires_grad_(False);encoder=load_masked_encoder(cfg)
    fit_rows=[r for r in rows if r['fold']=='fit'];held=[r for r in rows if r['fold']=='calibration'];n=cfg.label_change.calibration_draws
    gen,accepted=proposals(cfg,fit_rows,bank,encoder,scorer,cfg.label_change.seed+1);write_csv(output/'proposals.csv',gen);torch.save({'raw':p['fit'],**accepted},output/'features.pt')
    original=sampled_rows(held,n,cfg.label_change.seed+2);y=torch.cat([r['gold']['ner'] for r in original]);x=torch.cat([r['hidden'][r['focals']] for r in original]);prediction=predict(scorer,{'features':x[:,None],'y':y,'sentence':torch.arange(n)},cfg)[:,0].softmax(-1);s,h=prediction.max(-1)
    torch.save({'gold':y,'prediction':h,'confidence':s},output/'error_sample.pt')
    trial=sampled_rows(held,n,cfg.label_change.seed+3);calibration,probe=proposals(cfg,trial,bank,encoder,scorer,cfg.label_change.seed+4);write_csv(output/'acceptance_sample.csv',calibration)
    chosen=[r for r,record in zip(trial,calibration) if record['accepted']];original_vectors=torch.cat([r['hidden'][r['focals']] for r in chosen]) if chosen else torch.empty(0,768)
    torch.save({'original':original_vectors,'modified':probe['features'],'gold':torch.tensor([r['gold'] for r in calibration if r['accepted']]),'proposed':probe['y']},output/'changed_probe.pt')
    save_json(output/'generation.json',{'proposals':len(gen),'accepted':len(accepted['y']),'acceptance_rate':len(accepted['y'])/len(gen),'calibration_draws_per_quantity':n,'calibration_reference':'IID with-replacement uniform focal draws from the same fixed held-out empirical corpus, independent error and acceptance streams. Bounds are conditional on that corpus, not a population-level generalization guarantee.'})
    return {'accepted':len(accepted['y']),'proposals':len(gen)}
