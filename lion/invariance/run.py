"""Forge/Hydra entry point for invariance-signature training and analysis."""
import importlib
import sys

from forge import grid,start_run
from omegaconf import OmegaConf

from ..diagnostics.common import ROOT,absolute,save_json
from ..runtime import configure
from .data import paths


STAGES=('prepare','train','analyze','report')


def main(cfg):
    OmegaConf.resolve(cfg);cfg.forge.store=str(absolute(cfg.forge.store))
    if cfg.runtime.devices is not None and cfg.runtime.device.startswith('cuda'):cfg.runtime.device=f'cuda:{cfg.runtime.devices[0]}'
    configure(cfg);tracking=start_run(cfg);results,_=paths(cfg);stage=cfg.stage.experiment
    output=results/stage;output.mkdir(parents=True,exist_ok=True)
    print(f'Invariance stage {stage}: {output}',flush=True)
    if stage=='prepare':
        from .data import prepare
        result=prepare(cfg,output)
    else:result=importlib.import_module('lion.invariance.'+stage).experiment(cfg,output)
    save_json(tracking.path/'artifacts.json',{'results':str(output)});tracking.finish(result)
    return 0


if __name__=='__main__':
    grid('lion.invariance',sys.argv[1:],[[f'stage.experiment={s}'] for s in STAGES],{},main_module='run',config_dir=str(ROOT/'conf'),config_name='invariance')
