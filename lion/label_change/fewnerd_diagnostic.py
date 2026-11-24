"""Labels-only ambiguity comparison and a scorer/generation-only Few-NERD pilot."""
import json,sys,time
import torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from forge import run,start_run
from omegaconf import OmegaConf
from ..diagnostics.common import ROOT,absolute,save_json,write_csv,read_csv
from ..runtime import configure
from ..certification import data as source
from ..encoder import load_masked_encoder
from ..labels import LABELS
from . import data,report

REV='205f3e9c9f3577ea2561d43f2f62dc249ab92d5b'


def parquet(split):
    return hf_hub_download('DFKI-SLT/few-nerd',f'supervised/{split}-00000-of-00001.parquet',revision=REV,repo_type='dataset',local_files_only=True)


def ambiguity(cfg,output):
    conll=pq.read_table(hf_hub_download('eriktks/conll2003','conll2003/train/0000.parquet',revision='refs/convert/parquet',repo_type='dataset',local_files_only=True),columns=['tokens','ner_tags']).to_pylist()
    few=pq.read_table(parquet('train'),columns=['tokens','ner_tags','fine_ner_tags']).to_pylist();records=[]
    native=LABELS['conll2003'];names=list(dict.fromkeys(x[2:] if x!='O' else x for x in native));mapping=[names.index(x[2:] if x!='O' else x) for x in native]
    for name,rows,column,map_labels in [('CoNLL entity types + O',conll,'ner_tags',mapping),('Few-NERD coarse + O',few,'ner_tags',None),('Few-NERD fine + O',few,'fine_ner_tags',None)]:
        lookup={};forms=[];gold=[]
        for row in rows:
            for w,y in zip(row['tokens'],row[column]):forms.append(lookup.setdefault(w,len(lookup)));gold.append(map_labels[y] if map_labels else y)
        forms=torch.tensor(forms);gold=torch.tensor(gold);classes=int(gold.max())+1;counts=torch.bincount(forms*classes+gold,minlength=len(lookup)*classes).reshape(-1,classes).double();freq=counts.sum(-1);entity=freq-counts[:,0];d=1-(counts/freq[:,None]).square().sum(-1);ambiguous=(counts>0).sum(-1)>1
        record={'dataset':name,'training_tokens':len(gold),'word_types':len(lookup),'labels':classes,'d_token_weighted':float((freq*d).sum()/freq.sum()),'d_entity_token_weighted':float((entity*d).sum()/entity.sum()),'d_type_weighted':float(d.mean()),'ambiguous_token_share':float(freq[ambiguous].sum()/freq.sum()),'ambiguous_entity_token_share':float(entity[ambiguous].sum()/entity.sum())}
        records.append(record);print(record,flush=True)
    write_csv(output/'ambiguity.csv',records);save_json(output/'ambiguity.json',records);return {'datasets':len(records)}


def prepare(cfg,output):
    tokenizer=AutoTokenizer.from_pretrained(cfg.encoder.name,use_fast=True,local_files_only=True);sets={};sentence=0;labels=LABELS['fewnerd'];g=torch.Generator().manual_seed(cfg.seed)
    for split,target in [('train','train'),('validation','validation')]:
        rows=pq.read_table(parquet(split),columns=['tokens','ner_tags']).to_pylist()
        cap=cfg.certification.train_sentence_cap
        indices=torch.randperm(len(rows),generator=g)[:cap].sort().values.tolist() if split=='train' and cap is not None else list(range(len(rows)))
        rows=[rows[i] for i in indices]
        encoded=tokenizer([r['tokens'] for r in rows],is_split_into_words=True,add_special_tokens=True,truncation=True,max_length=cfg.encoder.max_length);result=[]
        for i,r in enumerate(rows):
            word_ids=encoded.word_ids(i);kept=list(dict.fromkeys(w for w in word_ids if w is not None));packed={w:j for j,w in enumerate(kept)}
            result.append({'ids':torch.tensor(encoded['input_ids'][i]),'word_ids':torch.tensor([packed.get(w,-1) for w in word_ids]),'focals':torch.arange(len(kept)),'gold':{'ner':torch.tensor([r['ner_tags'][w] for w in kept])},'sentence':sentence,'source_id':f'fewnerd/{split}/{indices[i]}','document':f'fewnerd/{split}/{indices[i]}' });sentence+=1
        sets[target]=result
    selected=sets['train'];held=set(torch.randperm(len(selected),generator=torch.Generator().manual_seed(cfg.label_change.seed))[:round(len(selected)*cfg.label_change.calibration_fraction)].tolist())
    for i,r in enumerate(selected):r['fold']='calibration' if i in held else 'fit'
    bank={'sets':sets,'labels':{'ner':labels},'special':{'mask':tokenizer.mask_token_id,'excluded':tokenizer.all_special_ids},'domains':{s:'fewnerd' for s in sets},'group_unit':'sentence'}
    root=source.paths(cfg);(root/'prepare').mkdir(parents=True,exist_ok=True);(root/'scorer').mkdir(exist_ok=True);torch.save(bank,root/'prepare'/'sentences.pt');OmegaConf.save(cfg,root/'config.yaml')
    encoder=load_masked_encoder(cfg);full=source.full_features(encoder,sets,cfg,dtype=getattr(torch,cfg.label_change.get('feature_dtype','float32')));torch.save(full,root/'scorer'/'full_features.pt')
    records=[{'role':role,'sentences':sum(r['fold']==role for r in selected),'words':sum(len(r['focals']) for r in selected if r['fold']==role)} for role in ('fit','calibration')];write_csv(output/'partitions.csv',records)
    return {'fit_sentences':len(selected)-len(held),'calibration_sentences':len(held)}


def summarize(cfg,output):
    summaries=[]
    for name,root in [('CoNLL',absolute('results/label_change')),('Few-NERD coarse',data.root(cfg))]:
        gen=json.loads((root/'generate'/'generation.json').read_text());c=json.loads((root/'certify'/'certification.json').read_text());proposals=read_csv(root/'generate'/'proposals.csv');accepted=[r for r in proposals if r['accepted']=='True']
        transitions=read_csv(absolute('results/label_change/report/label_transitions.csv')) if name=='CoNLL' else []
        semantic=sum(int(r['accepted']) for r in transitions if r['type_change']=='True') if name=='CoNLL' else len(accepted)
        record={'dataset':name,**gen,'semantic_changes':semantic,'semantic_acceptance_rate':semantic/gen['proposals'],
                **{k:c[k] for k in ('scorer_accuracy','scorer_error','conditional_high_confidence_error','high_confidence_share','high_confidence_errors','error_draws','accepted','acceptance_draws','e_hat','a_hat','e_U','a_L','ideal_sampler_pi_L','delta_max')},
                'generation_accepted':gen['accepted'],'heldout_acceptance_to_error_ratio':c['a_hat']/c['e_hat'] if c['e_hat'] else None,
                'training_acceptance_to_error_ratio':gen['acceptance_rate']/c['e_hat'] if c['e_hat'] else None,'acceptance_needed_point_estimate':20*c['e_hat'],'acceptance_lower_bound_needed':20*c['e_U']}
        # 'accepted' above is the independent acceptance-calibration count; generation_accepted is the fitting-pool count.
        summaries.append(record);print(json.dumps(record,indent=2),flush=True)
    save_json(output/'checks.json',summaries);return {'datasets':2}


def main(cfg):
    OmegaConf.resolve(cfg);configure(cfg);cfg.forge.store=str(absolute(cfg.forge.store));tracking=start_run(cfg);root=data.root(cfg);stage=cfg.stage.experiment;output=(source.paths(cfg) if stage=='scorer' else root)/stage;output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();print(f'Few-NERD diagnostic {stage}',flush=True)
    operation={'ambiguity':ambiguity,'prepare':prepare,'scorer':data.scorer,'generate':data.generate,'certify':report.certify,'summarize':summarize}[stage];result=operation(cfg,output);save_json(output/'runtime.json',{'seconds':time.monotonic()-started});tracking.finish(result);return 0


if __name__=='__main__':
    stages=[None] if any(a.startswith('stage.experiment=') for a in sys.argv[1:]) else ['ambiguity','prepare','scorer','generate','certify','summarize']
    for stage in stages:
        run('lion.label_change',sys.argv[1:]+([] if stage is None else [f'stage.experiment={stage}']),main_module='fewnerd_diagnostic',config_dir=str(ROOT/'conf'),config_name='label_change_fewnerd_diagnostic')
