"""Rebuild frozen prerequisites once, then generate and fit the three arms."""
import sys,time,shutil
from forge import grid,start_run
from omegaconf import OmegaConf
from ..runtime import configure
from ..diagnostics.common import ROOT,absolute,save_json
from ..certification import data as source_data,selector,calibrate
from . import data,train,report


def prepare_report():
    folder=ROOT/'report';old=folder/'old'/'07_certification';old.mkdir(parents=True,exist_ok=True)
    if (folder/'RESULTS_7.md').exists():shutil.move(str(folder/'RESULTS_7.md'),old/'RESULTS_7.md')
    if (folder/'tables'/'certification').exists():(old/'tables').mkdir(exist_ok=True);shutil.move(str(folder/'tables'/'certification'),old/'tables'/'certification')
    (old/'README.md').write_text('# Historical certification run\n\n[Results](RESULTS_7.md). Its HardKuma implementation predates the current word-ranking selector.\n')
    index=folder/'old'/'README.md';text=index.read_text()
    if '07_certification' not in text:index.write_text(text+'\n- [Batch 7: historical certification](07_certification/README.md)\n')
    shutil.copy2(ROOT/'lion'/'augmentation'/'IMPLEMENTATION.md',folder/'IMPLEMENTATION_8.md')
    (folder/'RESULTS_8.md').write_text('# Augmentation results pending\n\nThe frozen word-rank prerequisites are being rebuilt, followed by single-draw generation and three-seed paired classifier fits.\n')
    (folder/'README.md').write_text('# Latest: label-preserving augmentation experiment\n\n[Implementation](IMPLEMENTATION_8.md), [results](RESULTS_8.md), [tables](tables/augmentation/). Earlier runs: [old/](old/README.md).\n')


def main(cfg):
    OmegaConf.resolve(cfg);cfg.forge.store=str(absolute(cfg.forge.store));cfg.paths.cache=str(absolute(cfg.paths.cache))
    if cfg.runtime.devices is not None and cfg.runtime.device.startswith('cuda'):cfg.runtime.device=f'cuda:{cfg.runtime.devices[0]}'
    configure(cfg);tracking=start_run(cfg);root=data.paths(cfg);stage=cfg.stage.experiment;output=root/stage;output.mkdir(parents=True,exist_ok=True)
    print(f'Augmentation {stage}: {output}',flush=True);started=time.monotonic()
    if stage=='prepare':
        if not (root/'timing.json').exists():save_json(root/'timing.json',{'started_unix':time.time()})
        if not (ROOT/'report'/'IMPLEMENTATION_8.md').exists():prepare_report()
        result=data.prepare(cfg,output)
    elif stage in ('scorer','selector','calibrate'):
        source_output=source_data.paths(cfg)/stage;source_output.mkdir(parents=True,exist_ok=True)
        marker={'scorer':source_output/'ner'/'model.pt','selector':source_output/'ner'/'g2'/'model.pt','calibrate':source_output/'ner'/'g2.pt'}[stage]
        if cfg.augmentation.resume and marker.exists():result={'reused':True}
        elif stage=='scorer':result=selector.scorer_stage(cfg,source_output)
        elif stage=='selector':result=selector.experiment(cfg,source_output,started=started)
        else:result=calibrate.experiment(cfg,source_output)
    elif stage=='bounds':result=data.bounds(cfg,output)
    elif stage=='generate':result=data.generate(cfg,output)
    elif stage=='train':result=train.experiment(cfg,output)
    else:result=report.experiment(cfg,output)
    save_json(output/'runtime.json',{'seconds':time.monotonic()-started});save_json(tracking.path/'artifacts.json',{'results':str(output)});tracking.finish(result);return 0


if __name__=='__main__':
    grid('lion.augmentation',sys.argv[1:],[[f'stage.experiment={s}'] for s in ('prepare','scorer','selector','calibrate','bounds','generate','train','report')],{},main_module='run',config_dir=str(ROOT/'conf'),config_name='augmentation')
