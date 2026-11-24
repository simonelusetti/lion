"""Full Few-NERD scorer, natural pairs and unchanged dev-selected augmentation grid."""
import sys,time
import torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from forge import run,start_run
from omegaconf import OmegaConf
from ..diagnostics.common import ROOT,absolute,save_json
from ..runtime import configure
from ..label_change import data as labelled,generators,natural_pairs
from ..label_change.fewnerd_diagnostic import prepare,parquet
from ..certification import data,selector
from ..partition.train import load_model,predict
from ..encoder import load_masked_encoder
from . import generate,train,select,evaluate


def error_sample(cfg,output):
    bank=data.load(cfg);rows=[r for r in selector.raw_rows(cfg,bank) if r['fold']=='calibration'];sample=labelled.sampled_rows(rows,cfg.label_change.calibration_draws,cfg.label_change.seed+2)
    y=torch.cat([r['gold']['ner'] for r in sample]);x=torch.cat([r['hidden'][r['focals']] for r in sample]);scorer=load_model(data.paths(cfg)/'scorer'/'ner',cfg)
    p=predict(scorer,{'features':x[:,None],'y':y,'sentence':torch.arange(len(y))},cfg)[:,0].softmax(-1);confidence,h=p.max(-1)
    generated=labelled.root(cfg)/'generate';generated.mkdir(exist_ok=True);torch.save({'gold':y,'prediction':h,'confidence':confidence},generated/'error_sample.pt')
    cert=labelled.root(cfg)/'certify';cert.mkdir(exist_ok=True);save_json(cert/'certification.json',{'e_hat':float(((h!=y)&(confidence>=cfg.label_change.tau)).double().mean()),'description':'Reference high-confidence error only; full per-grid conditional bounds are saved under grid/calibrate.'})
    return {'calibration_error_draws':len(y)}


@torch.no_grad()
def test_cache(cfg,output):
    # This stage runs only after select/selection.json has fixed the grid winner.
    bank=data.load(cfg);labels=bank['labels']['ner'];raw=pq.read_table(parquet('test'),columns=['tokens','ner_tags']).to_pylist();tok=AutoTokenizer.from_pretrained(cfg.encoder.name,use_fast=True,local_files_only=True);rows=[]
    encoded=tok([r['tokens'] for r in raw],is_split_into_words=True,add_special_tokens=True,truncation=True,max_length=cfg.encoder.max_length)
    for i,r in enumerate(raw):
        wid=encoded.word_ids(i);kept=list(dict.fromkeys(w for w in wid if w is not None));packed={w:j for j,w in enumerate(kept)}
        rows.append({'ids':torch.tensor(encoded['input_ids'][i]),'word_ids':torch.tensor([packed.get(w,-1) for w in wid]),'focals':torch.arange(len(kept)),'gold':{'ner':torch.tensor([r['ner_tags'][w] for w in kept])},'sentence':i,'source_id':f'fewnerd/test/{i}'})
    sets={'id':rows};domains={'id':'fewnerd'};transfer,target_domains,coverage=data.ner_transfer_sets(cfg,'ner',len(rows));sets.update(transfer);domains.update(target_domains)
    encoder=load_masked_encoder(cfg);full=data.full_features(encoder,sets,cfg)
    evaluated={split:{'data':data.readout_data(rs,full[split],'ner'),'target':domains[split]} for split,rs in sets.items()}
    torch.save({'sets':evaluated,'labels':labels},output/'evaluation.pt');save_json(output/'targets.json',{'native':'fewnerd','transfer_coverage':coverage});return {'test_datasets':len(evaluated)}


def main(cfg):
    OmegaConf.resolve(cfg);configure(cfg);cfg.forge.store=str(absolute(cfg.forge.store));root=labelled.root(cfg);root.mkdir(parents=True,exist_ok=True);stage=cfg.stage.experiment
    if stage=='pairs':return natural_pairs.main(cfg)
    tracking=start_run(cfg);started=time.monotonic()
    if not (root/'timing.json').exists():save_json(root/'timing.json',{'started_unix':time.time()})
    if stage=='scorer':output=data.paths(cfg)/'scorer'
    elif stage in ('generate','calibrate','train','select','test'):output=generate.root(cfg)/stage
    elif stage=='heldout':output=absolute(cfg.generators.results)
    else:output=root/stage
    output.mkdir(parents=True,exist_ok=True);OmegaConf.save(cfg,root/'config.yaml');print(f'Full Few-NERD {stage}: {output}',flush=True)
    save_json(root/'protocol.json',{'data':'Full supervised Few-NERD coarse native training split; no sentence cap. Fixed 20% native-training holdout excluded from scorer and all grid heads. Native dev is unchanged and selects checkpoints/configuration. Native test and CoNLL/WNUT/WikiANN transfer remain raw.',
        'scorer':'Fresh existing frozen bert-base-cased block12 mean-WordPiece + 768→256→9 MLP, same ten-epoch recipe; no encoder fine-tuning.',
        'natural_pairs':'Same exact case-sensitive distinct-label/different-sentence diagnostic, excluding exact fitted-sentence overlap and duplicate test sentences. Pilot and full scorers are also compared on the identical common held-out cohort. Diagnostic results cannot change the predeclared grid or its dev-only selection.',
        'grid':'Nearby ±5, .2/.5/.7 replacement × .7/.8/.9 threshold. Original-correct and actual post-edit prediction flip required. One fill/focal/fraction, reused across thresholds. All fit words proposed; no caps. Three seeds for every cell and shared baseline, identical optimizer-step and labelled-token budgets. Same original data pool/labels; proportionate original/variant mixture.',
        'calibration':'Fresh independent 4000 error and 4000 acceptance draws from reserved empirical corpus. Per-cell conditional ideal-sampler precision bounds and unknown-discrepancy caveats unchanged; never used as a gate.',
        'selection':'Mean native Few-NERD dev span F1 across three seeds, after existing dev token-macro-F1 checkpoint selection. Only winner and shared baseline receive test forwards.',
        'runtime':'Larger inference focal batches (64) and MLM-position budget (2048) to reduce overhead; unchanged architecture/formulation. Float16 frozen-feature cache, following existing project cache practice.'})
    operation={'prepare':prepare,'scorer':labelled.scorer,'error':error_sample,'heldout':generators.experiment,'generate':generate.experiment,'calibrate':generate.calibrate,'train':train.experiment,'select':select.experiment,'test_cache':test_cache,'test':evaluate.experiment}[stage]
    result=operation(cfg,output);save_json(output/'runtime.json',{'seconds':time.monotonic()-started});tracking.finish(result)
    if stage=='test':
        import json
        timing=json.loads((root/'timing.json').read_text());timing['ended_unix']=time.time();timing['wall_seconds']=timing['ended_unix']-timing['started_unix'];save_json(root/'timing.json',timing)
    return 0


if __name__=='__main__':
    stages=[None] if any(a.startswith('stage.experiment=') for a in sys.argv[1:]) else ['prepare','scorer','pairs','error','heldout','generate','calibrate','train','select','test_cache','test']
    for stage in stages:
        code=run('lion.label_change_grid',sys.argv[1:]+([] if stage is None else [f'stage.experiment={stage}']),main_module='full_fewnerd',config_dir=str(ROOT/'conf'),config_name='fewnerd_full_grid')
        if code:sys.exit(code)
