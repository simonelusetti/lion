"""Reuse existing training/calibration/fill stages for EWT dependency relations."""
import json,sys,time,shutil
from forge import run,start_run
from omegaconf import OmegaConf
from ..runtime import configure
from ..diagnostics.common import ROOT,absolute,save_json
from ..certification import data as source,selector,calibrate
from ..fewnerd_replication.run import gamma,gate
from ..fewnerd_replication import train
from ..sensitivity_diagnostics import generate,measure,analyze
from . import data,report


def archive():
    folder=ROOT/'report';old=folder/'old'/'11_fewnerd_replication';old.mkdir(parents=True,exist_ok=True)
    for name in ('RESULTS_11.md','IMPLEMENTATION_11.md'):
        if (folder/name).exists():shutil.move(str(folder/name),old/name)
    p=folder/'tables'/'fewnerd'
    if p.exists():(old/'tables').mkdir(exist_ok=True);shutil.move(str(p),old/'tables'/'fewnerd')
    (old/'README.md').write_text('# Few-NERD replication\n\n[Results](RESULTS_11.md), [implementation](IMPLEMENTATION_11.md).\n')
    index=folder/'old'/'README.md'
    if '11_fewnerd_replication' not in index.read_text():index.write_text(index.read_text()+'\n- [Few-NERD replication](11_fewnerd_replication/README.md)\n')
    shutil.copy2(ROOT/'lion'/'deprel_replication'/'IMPLEMENTATION.md',folder/'IMPLEMENTATION_12.md')
    (folder/'RESULTS_12.md').write_text('# EWT dependency-relation replication running\n\nResults are generated automatically on completion.\n')
    (folder/'README.md').write_text('# Latest: dependency-relation context sensitivity\n\n[Implementation](IMPLEMENTATION_12.md), [results](RESULTS_12.md), [older batches](old/README.md).\n')


def main(cfg):
    OmegaConf.resolve(cfg);configure(cfg);cfg.forge.store=str(absolute(cfg.forge.store));tracking=start_run(cfg)
    root=absolute(cfg.augmentation.run_path);root.mkdir(parents=True,exist_ok=True);stage=cfg.stage.experiment;gp=source.paths(cfg)/'gamma.json'
    if gp.exists():cfg.certification.gamma=json.loads(gp.read_text())['gamma']
    output=(source.paths(cfg) if stage in ('prepare','scorer','selector','calibrate') else root)/stage;output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();print(f'EWT deprel {stage}: {output}',flush=True)
    if stage=='prepare':result={'reused':True} if (output/'sentences.pt').exists() else data.prepare(cfg,output)
    elif stage=='scorer':
        if not (ROOT/'report'/'IMPLEMENTATION_12.md').exists():archive()
        if not (output/'deprel'/'model.pt').exists():
            save_json(root/'timing.json',{'started_unix':time.time()});selector.scorer_stage(cfg,output)
        result=gamma(cfg)
    elif stage=='selector':result=selector.experiment(cfg,output,started=started)
    elif stage=='calibrate':result=calibrate.experiment(cfg,output)
    else:result={'gate':gate,'train':train.experiment,'metadata':data.metadata,'generate':generate.experiment,'measure':measure.experiment,'analyse':analyze.experiment,'report':report.experiment}[stage](cfg,output)
    save_json(output/'runtime.json',{'seconds':time.monotonic()-started});save_json(tracking.path/'artifacts.json',{'results':str(output)});tracking.finish(result);return 0


if __name__=='__main__':
    stages=[None] if any(a.startswith('stage.experiment=') for a in sys.argv[1:]) else ['prepare','scorer','selector','calibrate','gate','train','metadata','generate','measure','analyse','report']
    for stage in stages:run('lion.deprel_replication',sys.argv[1:]+([] if stage is None else [f'stage.experiment={stage}']),main_module='run',config_dir=str(ROOT/'conf'),config_name='deprel_replication')
