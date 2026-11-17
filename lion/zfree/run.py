"""Forge/Hydra stages for the Z-free document-context penalty study."""
import shutil
import sys
import time
from forge import grid,start_run
from omegaconf import OmegaConf
from ..diagnostics.common import ROOT,absolute,save_json
from ..runtime import configure
from ..invariance.data import paths


def prepare_reports():
    report=ROOT/'report';archive=report/'old'/'03_invariance_stage1';archive.mkdir(parents=True,exist_ok=True)
    for name in ('IMPLEMENTATION_3.md','RESULTS_3.md','original_request_3.md'):
        if (report/name).exists():shutil.move(str(report/name),archive/name)
    for category in ('tables','figures'):
        source=report/category/'invariance'
        if source.exists():(archive/category).mkdir(exist_ok=True);shutil.move(str(source),archive/category/'invariance')
    (archive/'README.md').write_text('# Stage 1: fixed invariance modes\n\n[Implementation](IMPLEMENTATION_3.md), [results](RESULTS_3.md), [protocol](original_request_3.md). Raw outputs remain in results/invariance_stage1/.\n')
    # Adjust links in the archived documents to their new depth.
    for name in ('IMPLEMENTATION_3.md','RESULTS_3.md'):
        path=archive/name
        if path.exists():path.write_text(path.read_text().replace('(../lion/','(../../../lion/').replace('(../conf/','(../../../conf/').replace('(old/03_superseded_full_grid/','(../03_superseded_full_grid/'))
    index=report/'old'/'README.md';link='- [Batch 3: Stage 1 invariance](03_invariance_stage1/README.md)\n'
    if link not in index.read_text():index.write_text(index.read_text()+'\n'+link)
    shutil.copy2(ROOT/'lion'/'zfree'/'IMPLEMENTATION.md',report/'IMPLEMENTATION_4.md')
    (report/'README.md').write_text('# Latest experiment: document-context nuisance penalty\n\n- [Implementation](IMPLEMENTATION_4.md)\n- [Results / status](RESULTS_4.md)\n- [Tables](tables/zfree/) and [figures](figures/zfree/)\n\nPrevious batches are in [old/](old/README.md).\n')
    (report/'RESULTS_4.md').write_text('# Results pending\n\nThe document-context penalty study is running. Its report will include the three primary comparisons, both validation policies, coverage and wall-clock time. See [implementation](IMPLEMENTATION_4.md).\n')


def main(cfg):
    OmegaConf.resolve(cfg);cfg.forge.store=str(absolute(cfg.forge.store))
    if cfg.runtime.devices is not None and cfg.runtime.device.startswith('cuda'):cfg.runtime.device=f'cuda:{cfg.runtime.devices[0]}'
    configure(cfg);tracking=start_run(cfg);results,_=paths(cfg);stage=cfg.stage.experiment;output=results/stage;output.mkdir(parents=True,exist_ok=True)
    print(f'Z-free stage {stage}: {output}',flush=True)
    if stage=='prepare':
        if not (results/'timing.json').exists() or not cfg.invariance.resume:save_json(results/'timing.json',{'started_unix':time.time()})
        if not (ROOT/'report'/'IMPLEMENTATION_4.md').exists():prepare_reports()
        from ..invariance.data import prepare
        result=prepare(cfg,output)
    elif stage=='contexts':
        from .data import experiment
        result=experiment(cfg,output)
    elif stage=='calibration':
        from .train import calibration
        result=calibration(cfg,output)
    elif stage=='train':
        from .train import experiment
        result=experiment(cfg,output)
    elif stage=='analyze':
        from .analyze import experiment
        result=experiment(cfg,output)
    else:
        from .report import experiment
        result=experiment(cfg,output)
    save_json(tracking.path/'artifacts.json',{'results':str(output)});tracking.finish(result);return 0


if __name__=='__main__':
    grid('lion.zfree',sys.argv[1:],[[f'stage.experiment={s}'] for s in ('prepare','contexts','calibration','train','analyze','report')],{},main_module='run',config_dir=str(ROOT/'conf'),config_name='zfree')
