"""Benchmark first; then generation and identity/outcome accounting only."""
import sys,time
from forge import run,start_run
from omegaconf import OmegaConf
from ..runtime import configure
from ..diagnostics.common import ROOT,absolute,save_json
from . import generate,analyse


def main(cfg):
    OmegaConf.resolve(cfg);configure(cfg);cfg.forge.store=str(absolute(cfg.forge.store));tracking=start_run(cfg);root=generate.root(cfg);root.mkdir(parents=True,exist_ok=True);stage=cfg.stage.experiment;output=root/stage;output.mkdir(exist_ok=True)
    OmegaConf.save(cfg,root/'config.yaml');started=time.monotonic();print(f'Proposal mechanisms {stage}: {output}',flush=True)
    if stage=='generate':save_json(root/'timing.json',{'started_unix':time.time()})
    result={'benchmark':generate.benchmark,'generate':generate.experiment,'analyse':analyse.experiment}[stage](cfg,output)
    save_json(output/'runtime.json',{'seconds':time.monotonic()-started});tracking.finish(result)
    if stage=='analyse':
        import json
        t=json.loads((root/'timing.json').read_text());t['ended_unix']=time.time();t['wall_seconds']=t['ended_unix']-t['started_unix'];save_json(root/'timing.json',t)
    return 0


if __name__=='__main__':
    stages=[None] if any(a.startswith('stage.experiment=') for a in sys.argv[1:]) else ['generate','analyse']
    for stage in stages:
        code=run('lion.proposal_mechanisms',sys.argv[1:]+([] if stage is None else [f'stage.experiment={stage}']),main_module='run',config_dir=str(ROOT/'conf'),config_name='proposal_mechanisms')
        if code:sys.exit(code)
