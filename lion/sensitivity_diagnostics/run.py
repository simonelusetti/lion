"""Step 1b only: regenerate, measure and explain the existing sensitivity outputs."""
import sys,time,shutil
from forge import run,start_run
from omegaconf import OmegaConf
from ..diagnostics.common import ROOT,absolute,save_json
from ..runtime import configure
from ..sensitivity.data import paths
from . import metadata,generate,measure,analyze,report


def archive():
    folder=ROOT/'report';old=folder/'old'/'09_sensitivity_step1';old.mkdir(parents=True,exist_ok=True)
    for name in ('RESULTS_9.md','IMPLEMENTATION_9.md'):
        if (folder/name).exists():shutil.move(str(folder/name),old/name)
    p=folder/'tables'/'sensitivity'
    if p.exists():(old/'tables').mkdir(exist_ok=True);shutil.move(str(p),old/'tables'/'sensitivity')
    (old/'README.md').write_text('# Sensitivity step 1\n\n[Results](RESULTS_9.md), [implementation](IMPLEMENTATION_9.md). Raw caches remain in results/sensitivity/.\n')
    index=folder/'old'/'README.md';link='\n- [Sensitivity step 1](09_sensitivity_step1/README.md)\n'
    if '09_sensitivity_step1' not in index.read_text():index.write_text(index.read_text()+link)
    shutil.copy2(ROOT/'lion'/'sensitivity_diagnostics'/'IMPLEMENTATION.md',folder/'IMPLEMENTATION_10.md')
    (folder/'RESULTS_10.md').write_text('# Step 1b diagnostics pending\n\nRegenerating the same seeded fills with token IDs, then performing inference and analysis only.\n')
    (folder/'README.md').write_text('# Latest: step 1b diagnostics\n\n[Implementation](IMPLEMENTATION_10.md), [results](RESULTS_10.md), [tables](tables/sensitivity_1b/). Earlier results: [old/](old/README.md).\n')


def main(cfg):
    OmegaConf.resolve(cfg);cfg.forge.store=str(absolute(cfg.forge.store));configure(cfg);tracking=start_run(cfg);root=paths(cfg);stage=cfg.stage.experiment;output=root/stage;output.mkdir(parents=True,exist_ok=True)
    if not (root/'timing.json').exists():save_json(root/'timing.json',{'started_unix':time.time()})
    if not (ROOT/'report'/'IMPLEMENTATION_10.md').exists():archive()
    print(f'Step 1b: {stage}',flush=True)
    operation={'metadata':metadata.experiment,'generate':generate.experiment,'measure':measure.experiment,'analyse':analyze.experiment,'report':report.experiment}[stage]
    result=operation(cfg,output);save_json(tracking.path/'artifacts.json',{'results':str(output)});tracking.finish(result);return 0


if __name__=='__main__':
    for stage in ('metadata','generate','measure','analyse','report'):
        run('lion.sensitivity_diagnostics',sys.argv[1:]+[f'stage.experiment={stage}'],main_module='run',config_dir=str(ROOT/'conf'),config_name='sensitivity_1b')
