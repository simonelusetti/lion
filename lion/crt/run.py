"""Forge/Hydra execution: full-training readouts, runtime pilot, three CRT arms."""
import importlib,shutil,sys,time
from forge import grid,start_run
from omegaconf import OmegaConf
from ..diagnostics.common import ROOT,absolute,save_json
from ..runtime import configure
from .data import paths


def archive():
    report=ROOT/'report';old=report/'old'/'05_mask_pilot';old.mkdir(parents=True,exist_ok=True)
    for name in ('IMPLEMENTATION_5.md','RESULTS_5.md'):
        source=report/name
        if source.exists():shutil.move(str(source),old/name)
    for category in ('tables','figures'):
        source=report/category/'pilot'
        if source.exists():(old/category).mkdir(exist_ok=True);shutil.move(str(source),old/category/'pilot')
    for name in ('IMPLEMENTATION_5.md','RESULTS_5.md'):
        path=old/name
        if path.exists():path.write_text(path.read_text().replace('(../lion/','(../../../lion/').replace('(../conf/','(../../../conf/').replace('(../grids/','(../../../grids/'))
    (old/'README.md').write_text('# Mask-rule pilot grid\n\n[Implementation](IMPLEMENTATION_5.md), [results](RESULTS_5.md). Raw outputs remain in results/pilot/.\n')
    index=report/'old'/'README.md';link='\n- [Batch 5: mask pilot](05_mask_pilot/README.md)\n'
    if '05_mask_pilot' not in index.read_text():index.write_text(index.read_text()+link)
    shutil.copy2(ROOT/'lion'/'crt'/'IMPLEMENTATION.md',report/'IMPLEMENTATION_6.md')
    (report/'README.md').write_text('# Latest: conditional randomization test\n\n[Implementation](IMPLEMENTATION_6.md), [results/status](RESULTS_6.md), [tables](tables/crt/). Previous batches: [old/](old/README.md).\n')
    (report/'RESULTS_6.md').write_text('# CRT results pending\n\nThe three-arm experiment is running. The 1% timing pilot will select the sentence budget before the full CRT. K remains 200. See [implementation](IMPLEMENTATION_6.md).\n')


def main(cfg):
    OmegaConf.resolve(cfg);cfg.forge.store=str(absolute(cfg.forge.store))
    if cfg.runtime.devices is not None and cfg.runtime.device.startswith('cuda'):cfg.runtime.device=f'cuda:{cfg.runtime.devices[0]}'
    configure(cfg);tracking=start_run(cfg);results,cache=paths(cfg);stage=cfg.stage.experiment;output=results/stage;output.mkdir(parents=True,exist_ok=True)
    print(f'CRT {stage}: {output}',flush=True)
    if stage=='prepare':
        if not (results/'timing.json').exists():save_json(results/'timing.json',{'started_unix':time.time()})
        if not (ROOT/'report'/'IMPLEMENTATION_6.md').exists():archive()
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
