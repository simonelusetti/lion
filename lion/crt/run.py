"""Forge/Hydra execution: full-training readouts, runtime pilot, three CRT arms."""
import importlib,sys,time
from forge import grid,start_run
from omegaconf import OmegaConf
from ..diagnostics.common import ROOT,absolute,save_json
from ..runtime import configure
from .data import paths


def main(cfg):
    OmegaConf.resolve(cfg);cfg.forge.store=str(absolute(cfg.forge.store))
    if cfg.runtime.devices is not None and cfg.runtime.device.startswith('cuda'):cfg.runtime.device=f'cuda:{cfg.runtime.devices[0]}'
    configure(cfg);tracking=start_run(cfg);results,cache=paths(cfg);stage=cfg.stage.experiment;output=results/stage;output.mkdir(parents=True,exist_ok=True)
    print(f'CRT {stage}: {output}',flush=True)
    if stage=='prepare':
        if not (results/'timing.json').exists():save_json(results/'timing.json',{'started_unix':time.time()})
        from .data import prepare
        result=prepare(cfg,output)
    elif stage=='cache':
        from .data import extract
        result=extract(cfg,output)
    elif stage=='train' and cfg.crt.resume and (output/'dev.json').exists():result={'resume':True}
    else:result=importlib.import_module('lion.crt.'+('sample' if stage=='test' else stage)).experiment(cfg,output)
    save_json(tracking.path/'artifacts.json',{'results':str(output)});tracking.finish(result);return 0


if __name__=='__main__':
    grid('lion.crt',sys.argv[1:],[[f'stage.experiment={stage}'] for stage in ('prepare','cache','train','test','report')],{},main_module='run',config_dir=str(ROOT/'conf'),config_name='crt')
