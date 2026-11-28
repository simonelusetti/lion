"""Forge/Hydra stages for fixed-size scorer-free paradigm swaps."""
import sys,time,shutil,json
from forge import run,start_run
from omegaconf import OmegaConf
from ..diagnostics.common import ROOT,absolute,save_json,read_csv,write_csv
from ..runtime import configure
from . import generate,train,stats,evaluate


def main(cfg):
    OmegaConf.resolve(cfg);configure(cfg);cfg.forge.store=str(absolute(cfg.forge.store));tracking=start_run(cfg);root=generate.root(cfg);root.mkdir(parents=True,exist_ok=True);stage=cfg.stage.experiment;out=root/stage;out.mkdir(exist_ok=True);OmegaConf.save(cfg,root/'config.yaml');OmegaConf.save(cfg,out/'config.yaml');started=time.monotonic();print(f'Paradigm swaps {stage} ({cfg.runtime.device})',flush=True)
    if not (root/'timing.json').exists():save_json(root/'timing.json',{'started_unix':time.time()})
    result={'prepare':generate.prepare,'generate':generate.generate,'stats':stats.stats,'train':train.train,'test':evaluate.test,'slices':evaluate.slices,'pairs':evaluate.pairs,'report':evaluate.report}[stage](cfg,out);save_json(out/'runtime.json',{'seconds':time.monotonic()-started});tracking.finish(result)
    if stage=='prepare':
        rows=[r|{'status':'reused unchanged historical budget'} for r in read_csv(absolute(cfg.matched.results)/'train/budgets.csv') if r['head'] in ('baseline','MENTION')];write_csv(root/'reused_budgets.csv',rows)
    if stage=='report':
        timing=json.loads((root/'timing.json').read_text());timing.update(ended_unix=time.time());timing['seconds']=timing['ended_unix']-timing['started_unix'];save_json(root/'timing.json',timing);text=(root/'RESULTS.md').read_text()+f"\nActual total wall-clock time (computed): {timing['seconds']/60:.2f} minutes. Stage timings: each stage/runtime.json.\n";(root/'RESULTS.md').write_text(text);(out/'RESULTS.md').write_text(text)
    return 0

if __name__=='__main__':
    stages=[None] if any(a.startswith('stage.experiment=') for a in sys.argv[1:]) else ['prepare','generate','stats','train','test','slices','pairs','report']
    for stage in stages:
        status=run('lion.paradigm_swaps',sys.argv[1:]+([] if stage is None else [f'stage.experiment={stage}']),main_module='run',config_dir=str(ROOT/'conf'),config_name='paradigm_swaps')
        if status:sys.exit(status)
