"""Sequential Forge/Hydra feasibility stages; no augmentation training."""
import shutil,sys,time
from forge import run,start_run
from omegaconf import OmegaConf
from ..runtime import configure
from ..diagnostics.common import ROOT,absolute,save_json
from . import data,context,estimate,measure,analyse,report


def archive():
    folder=ROOT/'report';old=folder/'old'/'16_separability';old.mkdir(parents=True,exist_ok=True)
    for name in ('RESULTS_16.md','IMPLEMENTATION_16.md'):
        if (folder/name).exists():shutil.move(str(folder/name),old/name)
    tables=folder/'tables'/'separability'
    if tables.exists():(old/'tables').mkdir(exist_ok=True);shutil.move(str(tables),old/'tables'/'separability')
    (old/'README.md').write_text('# Per-word separability and alignment\n\n[Results](RESULTS_16.md), [implementation](IMPLEMENTATION_16.md).\n')
    index=folder/'old'/'README.md'
    if '16_separability' not in index.read_text():index.write_text(index.read_text()+'\n- [Separability and alignment](16_separability/README.md)\n')
    shutil.copy2(ROOT/'lion'/'conflict'/'IMPLEMENTATION.md',folder/'IMPLEMENTATION_17.md')
    (folder/'RESULTS_17.md').write_text('# Conflict-insertion feasibility running\n\nResults are written automatically on completion.\n')
    (folder/'README.md').write_text('# Latest: conflict-insertion feasibility\n\n[Implementation](IMPLEMENTATION_17.md), [results](RESULTS_17.md), [earlier batches](old/README.md).\n')


def main(cfg):
    OmegaConf.resolve(cfg);configure(cfg);cfg.forge.store=str(absolute(cfg.forge.store));tracking=start_run(cfg)
    root=data.root(cfg);root.mkdir(parents=True,exist_ok=True);stage=cfg.stage.experiment;output=root/stage;output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();print(f'Conflict feasibility {stage}: {output}',flush=True)
    if stage=='features':
        if not (ROOT/'report'/'IMPLEMENTATION_17.md').exists():archive()
        if not (root/'timing.json').exists():save_json(root/'timing.json',{'started_unix':time.time()})
    operation={'prepare':data.prepare,'estimate':estimate.experiment,'features':context.features,'context':context.fit,'supply':measure.supply,'room':measure.room,'analyse':analyse.experiment,'report':report.experiment}[stage]
    result=operation(cfg,output);save_json(output/'runtime.json',{'seconds':time.monotonic()-started});save_json(tracking.path/'artifacts.json',{'results':str(output)});tracking.finish(result);return 0


if __name__=='__main__':
    stages=[None] if any(a.startswith('stage.experiment=') for a in sys.argv[1:]) else ['prepare','estimate','features','context','supply','room','analyse','report']
    for stage in stages:
        code=run('lion.conflict',sys.argv[1:]+([] if stage is None else [f'stage.experiment={stage}']),main_module='run',config_dir=str(ROOT/'conf'),config_name='conflict')
        if code:sys.exit(code)
