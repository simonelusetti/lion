"""Sequential Forge stages for the matched label-changing pilot."""
import sys,time,shutil
from forge import run,start_run
from omegaconf import OmegaConf
from ..diagnostics.common import ROOT,absolute,save_json
from ..runtime import configure
from ..certification import data as source
from . import data,train,report


def archive():
    folder=ROOT/'report';old=folder/'old'/'17_conflict_feasibility';old.mkdir(parents=True,exist_ok=True)
    for name in ('RESULTS_17.md','IMPLEMENTATION_17.md'):
        if (folder/name).exists():shutil.move(str(folder/name),old/name)
    p=folder/'tables'/'conflict'
    if p.exists():(old/'tables').mkdir(exist_ok=True);shutil.move(str(p),old/'tables'/'conflict')
    (old/'README.md').write_text('# Conflict-insertion feasibility\n\n[Results](RESULTS_17.md), [implementation](IMPLEMENTATION_17.md).\n');index=folder/'old'/'README.md'
    if '17_conflict_feasibility' not in index.read_text():index.write_text(index.read_text()+'\n- [Conflict-insertion feasibility](17_conflict_feasibility/README.md)\n')
    shutil.copy2(ROOT/'lion'/'label_change'/'IMPLEMENTATION.md',folder/'IMPLEMENTATION_18.md');(folder/'RESULTS_18.md').write_text('# Label-changing pilot running\n\nResults are generated automatically at completion.\n');(folder/'README.md').write_text('# Latest: context-changing, label-changing augmentation\n\n[Implementation](IMPLEMENTATION_18.md), [results](RESULTS_18.md), [older batches](old/README.md).\n')


def main(cfg):
    OmegaConf.resolve(cfg);configure(cfg);cfg.forge.store=str(absolute(cfg.forge.store));tracking=start_run(cfg);root=data.root(cfg);root.mkdir(parents=True,exist_ok=True);stage=cfg.stage.experiment
    if not (root/'timing.json').exists():save_json(root/'timing.json',{'started_unix':time.time()})
    if not (ROOT/'report'/'IMPLEMENTATION_18.md').exists():archive()
    output=(source.paths(cfg) if stage=='scorer' else root)/stage;output.mkdir(parents=True,exist_ok=True);started=time.monotonic();print(f'Label-changing pilot {stage}: {output}',flush=True)
    result={'prepare':data.prepare,'scorer':data.scorer,'generate':data.generate,'certify':report.certify,'train':train.experiment,'report':report.experiment}[stage](cfg,output)
    save_json(output/'runtime.json',{'seconds':time.monotonic()-started});save_json(tracking.path/'artifacts.json',{'results':str(output)});tracking.finish(result);return 0


if __name__=='__main__':
    stages=[None] if any(a.startswith('stage.experiment=') for a in sys.argv[1:]) else ['prepare','scorer','generate','certify','train','report']
    for stage in stages:
        code=run('lion.label_change',sys.argv[1:]+([] if stage is None else [f'stage.experiment={stage}']),main_module='run',config_dir=str(ROOT/'conf'),config_name='label_change')
        if code:sys.exit(code)
