"""Inference-only step 1, reusing the full-scale augmentation artifacts."""
import sys,time,shutil
from forge import run,start_run
from omegaconf import OmegaConf
from ..diagnostics.common import ROOT,absolute,save_json
from ..runtime import configure
from . import data,measure,report


def archive():
    folder=ROOT/'report';old=folder/'old'/'08_augmentation_full';old.mkdir(parents=True,exist_ok=True)
    for name in ('RESULTS_8.md','IMPLEMENTATION_8.md'):
        p=folder/name
        if p.exists():shutil.move(str(p),old/name)
    p=folder/'tables'/'augmentation'
    if p.exists():(old/'tables').mkdir(exist_ok=True);shutil.move(str(p),old/'tables'/'augmentation')
    (old/'README.md').write_text('# Full-scale augmentation\n\n[Results](RESULTS_8.md), [implementation](IMPLEMENTATION_8.md). Raw models and caches remain in results/augmentation_full/.\n')
    index=folder/'old'/'README.md';link='\n- [Full-scale augmentation](08_augmentation_full/README.md)\n'
    if '08_augmentation_full' not in index.read_text():index.write_text(index.read_text()+link)
    shutil.copy2(ROOT/'lion'/'sensitivity'/'IMPLEMENTATION.md',folder/'IMPLEMENTATION_9.md')
    (folder/'RESULTS_9.md').write_text('# Step 1 sensitivity: results pending\n\nInference on existing classifiers only. No training. Steps 2 and 3 remain on hold.\n')
    (folder/'README.md').write_text('# Latest: step 1 sensitivity\n\n[Implementation](IMPLEMENTATION_9.md), [results](RESULTS_9.md), [tables](tables/sensitivity/). Earlier experiments: [old/](old/README.md).\n')


def main(cfg):
    OmegaConf.resolve(cfg);cfg.forge.store=str(absolute(cfg.forge.store));configure(cfg);tracking=start_run(cfg);root=data.paths(cfg);output=root/cfg.stage.experiment;output.mkdir(parents=True,exist_ok=True)
    if not (root/'timing.json').exists():save_json(root/'timing.json',{'started_unix':time.time()})
    if not (ROOT/'report'/'IMPLEMENTATION_9.md').exists():archive()
    print(f'Sensitivity step 1: {cfg.stage.experiment}',flush=True)
    result={'generate':data.generate,'measure':measure.experiment,'report':report.experiment}[cfg.stage.experiment](cfg,output)
    save_json(tracking.path/'artifacts.json',{'results':str(output)});tracking.finish(result);return 0


if __name__=='__main__':
    for stage in ('generate','measure','report'):
        run('lion.sensitivity',sys.argv[1:]+[f'stage.experiment={stage}'],main_module='run',config_dir=str(ROOT/'conf'),config_name='sensitivity')
