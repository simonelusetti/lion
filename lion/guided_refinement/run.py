"""Prepare training-only targets, benchmark, then run the fixed five-arm diagnostic."""
import sys,time
from forge import run,start_run
from omegaconf import OmegaConf
from ..runtime import configure
from ..diagnostics.common import ROOT,absolute,save_json
from . import generate


def main(cfg):
    OmegaConf.resolve(cfg);configure(cfg);cfg.forge.store=str(absolute(cfg.forge.store));tracking=start_run(cfg);root=generate.root(cfg);root.mkdir(parents=True,exist_ok=True);stage=cfg.stage.experiment;output=root/stage;output.mkdir(exist_ok=True);started=time.monotonic();OmegaConf.save(cfg,root/'config.yaml')
    print(f'Guided refinement {stage}: {output}',flush=True)
    if stage=='generate':save_json(root/'timing.json',{'started_unix':time.time()})
    if stage=='analyse':
        from .analyse import experiment
        result=experiment(cfg,output)
        import json
        t=json.loads((root/'timing.json').read_text());t['ended_unix']=time.time();t['wall_seconds']=t['ended_unix']-t['started_unix'];save_json(root/'timing.json',t)
    else:result={'prepare':generate.prepare,'benchmark':generate.benchmark,'generate':generate.experiment}[stage](cfg,output)
    save_json(output/'runtime.json',{'seconds':time.monotonic()-started});tracking.finish(result);return 0


if __name__=='__main__':
    stages=[None] if any(a.startswith('stage.experiment=') for a in sys.argv[1:]) else ['generate','analyse']
    for stage in stages:
        code=run('lion.guided_refinement',sys.argv[1:]+([] if stage is None else [f'stage.experiment={stage}']),main_module='run',config_dir=str(ROOT/'conf'),config_name='guided_refinement')
        if code:sys.exit(code)
