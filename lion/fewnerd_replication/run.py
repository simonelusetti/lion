"""Sequential Forge stages for a single Few-NERD replication."""
import json,sys,time,shutil
import torch
from forge import run,start_run
from omegaconf import OmegaConf
from ..diagnostics.common import ROOT,absolute,save_json,read_csv,write_csv
from ..runtime import configure
from ..certification import data as source_data,selector,calibrate
from ..augmentation import data as augmentation_data,additions
from ..sensitivity_diagnostics import metadata,generate,measure,analyze
from . import data,train,report


def archive():
    folder=ROOT/'report';old=folder/'old'/'10_sensitivity_step1b';old.mkdir(parents=True,exist_ok=True)
    for name in ('RESULTS_10.md','IMPLEMENTATION_10.md'):
        if (folder/name).exists():shutil.move(str(folder/name),old/name)
    p=folder/'tables'/'sensitivity_1b'
    if p.exists():(old/'tables').mkdir(exist_ok=True);shutil.move(str(p),old/'tables'/'sensitivity_1b')
    (old/'README.md').write_text('# CoNLL step 1b\n\n[Results](RESULTS_10.md), [implementation](IMPLEMENTATION_10.md).\n')
    index=folder/'old'/'README.md'
    if '10_sensitivity_step1b' not in index.read_text():index.write_text(index.read_text()+'\n- [CoNLL step 1b](10_sensitivity_step1b/README.md)\n')
    shutil.copy2(ROOT/'lion'/'fewnerd_replication'/'IMPLEMENTATION.md',folder/'IMPLEMENTATION_11.md')
    (folder/'RESULTS_11.md').write_text('# Few-NERD replication running\n\nResults will be written automatically at completion.\n')
    (folder/'README.md').write_text('# Latest: Few-NERD replication\n\n[Implementation](IMPLEMENTATION_11.md), [results](RESULTS_11.md), [earlier batches](old/README.md).\n')


def gamma(cfg):
    task=source_data.task_names(cfg)[0]
    logits=torch.load(source_data.paths(cfg)/'scorer'/task/'predictions.pt',weights_only=False)['validation']
    bank=source_data.load(cfg);gold=torch.cat([r['gold'][task] for r in bank['sets']['train'] if r['fold']=='scorer_validation'])
    value=float((logits[:,0].argmax(-1)!=gold).double().mean())
    save_json(source_data.paths(cfg)/'gamma.json',{'gamma':value,'split':'scorer_validation within source split A','words':len(gold)})
    cfg.certification.gamma=value;OmegaConf.save(cfg,source_data.paths(cfg)/'config.yaml')
    return {'gamma':value}


def gate(cfg,output):
    bounds=absolute(cfg.augmentation.run_path)/'bounds';bounds.mkdir(exist_ok=True);augmentation_data.bounds(cfg,bounds)
    additions.experiment(cfg,output)
    rows=read_csv(output/'enriched_bounds.csv')
    for r in rows:
        r['passed']=float(r['rho'])==1.0 or (r['lower_bound']!='' and float(r['lower_bound'])<=cfg.certification.epsilon)
        r['status']='gate_passed' if r['passed'] else 'UNCERTIFIED'
    write_csv(output/'gate.csv',rows)
    return {'passed_groups':sum(r['passed'] for r in rows),'groups':len(rows)}


def main(cfg):
    OmegaConf.resolve(cfg);configure(cfg);cfg.forge.store=str(absolute(cfg.forge.store));tracking=start_run(cfg)
    root=absolute(cfg.augmentation.run_path);root.mkdir(parents=True,exist_ok=True);stage=cfg.stage.experiment
    gp=source_data.paths(cfg)/'gamma.json'
    if gp.exists():cfg.certification.gamma=json.loads(gp.read_text())['gamma']
    target=source_data.paths(cfg) if stage in ('prepare','scorer','selector','calibrate') else root
    output=target/stage;output.mkdir(parents=True,exist_ok=True);started=time.monotonic();print(f'Few-NERD {stage}: {output}',flush=True)
    if not (root/'timing.json').exists():save_json(root/'timing.json',{'started_unix':time.time()})
    if stage=='prepare':
        result={'reused':True} if (output/'sentences.pt').exists() else data.prepare(cfg,output)
    else:
        if stage=='scorer' and not (ROOT/'report'/'IMPLEMENTATION_11.md').exists():archive()
        if stage=='scorer':
            if not (output/'ner'/'model.pt').exists():save_json(root/'timing.json',{'started_unix':time.time()})
            if not (output/'ner'/'model.pt').exists():selector.scorer_stage(cfg,output)
            result=gamma(cfg)
        elif stage=='selector':result=selector.experiment(cfg,output,started=started)
        elif stage=='calibrate':result=calibrate.experiment(cfg,output)
        else:result={'gate':gate,'train':train.experiment,'metadata':metadata.experiment,'generate':generate.experiment,'measure':measure.experiment,'analyse':analyze.experiment,'report':report.experiment}[stage](cfg,output)
    save_json(output/'runtime.json',{'seconds':time.monotonic()-started});save_json(tracking.path/'artifacts.json',{'results':str(output)});tracking.finish(result);return 0


if __name__=='__main__':
    explicit=any(x.startswith('stage.experiment=') for x in sys.argv[1:])
    stages=[None] if explicit else ['prepare','scorer','selector','calibrate','gate','train','metadata','generate','measure','analyse','report']
    for stage in stages:
        run('lion.fewnerd_replication',sys.argv[1:]+([] if stage is None else [f'stage.experiment={stage}']),main_module='run',config_dir=str(ROOT/'conf'),config_name='fewnerd_replication')
