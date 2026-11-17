"""Dependency-ordered Forge grid with independent task/rotation/seed cells."""
import importlib,shutil,sys,time
from forge import grid,start_run
from omegaconf import OmegaConf
from ..diagnostics.common import ROOT,absolute,save_json,write_csv
from ..runtime import configure
from ..invariance.data import paths


def archive():
    report=ROOT/'report';old=report/'old'/'04_document_context';old.mkdir(parents=True,exist_ok=True)
    for name in ('IMPLEMENTATION_4.md','RESULTS_4.md'):
        p=report/name
        if p.exists():shutil.move(str(p),old/name)
    for category in ('tables','figures'):
        p=report/category/'zfree'
        if p.exists():(old/category).mkdir(exist_ok=True);shutil.move(str(p),old/category/'zfree')
    for name in ('IMPLEMENTATION_4.md','RESULTS_4.md'):
        p=old/name
        if p.exists():p.write_text(p.read_text().replace('(../lion/','(../../../lion/').replace('(../conf/','(../../../conf/'))
    (old/'README.md').write_text('# Document-context experiment\n\n[Implementation](IMPLEMENTATION_4.md), [results](RESULTS_4.md).\n')
    index=report/'old'/'README.md';link='\n- [Batch 4: document context](04_document_context/README.md)\n'
    if '04_document_context' not in index.read_text():index.write_text(index.read_text()+link)
    shutil.copy2(ROOT/'lion'/'pilot'/'IMPLEMENTATION.md',report/'IMPLEMENTATION_5.md')
    (report/'README.md').write_text('# Latest: mask-rule pilot grid\n\n[Implementation](IMPLEMENTATION_5.md), [results/status](RESULTS_5.md), [tables](tables/pilot/). Earlier batches are in [old/](old/README.md).\n')
    (report/'RESULTS_5.md').write_text('# Pilot grid: results pending\n\nThe dependency-ordered grid is running. No measurements are available yet. See [implementation](IMPLEMENTATION_5.md).\n')


def main(cfg):
    OmegaConf.resolve(cfg);cfg.forge.store=str(absolute(cfg.forge.store));configure(cfg);tracking=start_run(cfg);results,_=paths(cfg);stage=cfg.stage.experiment;output=results/stage;output.mkdir(parents=True,exist_ok=True)
    if cfg.runtime.devices is not None and cfg.runtime.device.startswith('cuda'):cfg.runtime.device=f'cuda:{cfg.runtime.devices[0]}'
    print(f'Pilot {stage}: {cfg.pilot.task} rotation {cfg.pilot.rotation} seed {cfg.pilot.seed}',flush=True)
    if stage=='prepare':
        if not (results/'timing.json').exists():save_json(results/'timing.json',{'started_unix':time.time()})
        if not (ROOT/'report'/'IMPLEMENTATION_5.md').exists():archive()
        from .data import prepare
        result=prepare(cfg,output)
        write_csv(results/'grid.csv',[{'task':task,'rotation':rotation,'seed':seed,'rule':rule,'regularizer':reg} for task in cfg.invariance.datasets for rotation in range(5) for seed in cfg.diagnostics.seeds for rule in cfg.pilot.rules for reg in cfg.pilot.regularizers])
    elif stage in ('base','selected'):result=getattr(importlib.import_module('lion.pilot.encode'),stage)(cfg,output)
    elif stage in ('scorer','pilot','cell'):result=getattr(importlib.import_module('lion.pilot.train'),stage)(cfg,output)
    else:result=importlib.import_module('lion.pilot.'+stage).experiment(cfg,output)
    save_json(tracking.path/'artifacts.json',{'results':str(output)});tracking.finish(result);return 0


if __name__=='__main__':
    # Each fit contains independent rule×regularizer heads; cells can also be run separately.
    tasks=('ud_upos','ud_deprel','gum_upos','ud_discourse')
    stages=[['stage.experiment=prepare'],['stage.experiment=base']]
    stages += [[f'stage.experiment=scorer',f'pilot.task={task}',f'pilot.rotation={r}'] for task in tasks for r in range(5)]
    stages += [['stage.experiment=select'],['stage.experiment=selected']]
    stages += [['stage.experiment=pilot',f'pilot.task={task}'] for task in tasks]
    stages += [['stage.experiment=cell',f'pilot.task={task}',f'pilot.rotation={r}',f'pilot.seed={seed}'] for task in tasks for r in range(5) for seed in (0,1,2)]
    stages += [['stage.experiment=analyze'],['stage.experiment=report']]
    grid('lion.pilot',sys.argv[1:],stages,{},main_module='run',config_dir=str(ROOT/'conf'),config_name='pilot')
