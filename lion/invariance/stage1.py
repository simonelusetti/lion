"""Forge/Hydra pipeline for the cost-revised stage-1 study."""
import sys
import time
from forge import grid,start_run
from omegaconf import OmegaConf
from ..diagnostics.common import ROOT,absolute,save_json
from ..runtime import configure
from .data import paths


def main(cfg):
    OmegaConf.resolve(cfg);cfg.forge.store=str(absolute(cfg.forge.store))
    if cfg.runtime.devices is not None and cfg.runtime.device.startswith('cuda'):cfg.runtime.device=f'cuda:{cfg.runtime.devices[0]}'
    configure(cfg);tracking=start_run(cfg);results,_=paths(cfg);stage=cfg.stage.experiment
    output=results/stage;output.mkdir(parents=True,exist_ok=True)
    print(f'Stage 1 {stage}: {output}',flush=True)
    if stage=='prepare':
        if not (results/'timing.json').exists() or not cfg.invariance.resume:save_json(results/'timing.json',{'started_unix':time.time()})
        from .data import prepare
        result=prepare(cfg,output)
    elif stage in ('pilot','train'):
        from .stage1_train import experiment
        result=experiment(cfg,output,pilot=stage=='pilot')
    elif stage=='analyze':
        from .stage1_analyze import experiment
        result=experiment(cfg,output)
    else:
        from .stage1_report import experiment
        result=experiment(cfg,output)
    save_json(tracking.path/'artifacts.json',{'results':str(output)});tracking.finish(result)
    return 0


if __name__=='__main__':
    grid('lion.invariance',sys.argv[1:],[[f'stage.experiment={s}'] for s in ('prepare','pilot','train','analyze','report')],{},main_module='stage1',config_dir=str(ROOT/'conf'),config_name='invariance_stage1')
