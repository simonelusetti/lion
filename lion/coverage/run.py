"""Sequential Forge stages: prepare, estimate, train, analyse, report."""
import json
import shutil
import sys
import time
from forge import run, start_run
from omegaconf import OmegaConf
from ..diagnostics.common import ROOT, absolute, save_json
from ..runtime import configure
from . import data, estimate, train, analyse, report


def archive():
    folder = ROOT / 'report'
    old = folder / 'old' / '13_layer_sensitivity'
    old.mkdir(parents=True, exist_ok=True)
    for name in ('RESULTS_13.md', 'IMPLEMENTATION_13.md'):
        if (folder / name).exists():
            shutil.move(str(folder / name), old / name)
    tables = folder / 'tables' / 'layers'
    if tables.exists():
        (old / 'tables').mkdir(exist_ok=True)
        shutil.move(str(tables), old / 'tables' / 'layers')
    (old / 'README.md').write_text('# Context sensitivity across encoder layers\n\n[Results](RESULTS_13.md), [implementation](IMPLEMENTATION_13.md).\n')
    index = folder / 'old' / 'README.md'
    if '13_layer_sensitivity' not in index.read_text():
        index.write_text(index.read_text() + '\n- [Context sensitivity across encoder layers](13_layer_sensitivity/README.md)\n')
    shutil.copy2(ROOT / 'lion' / 'coverage' / 'IMPLEMENTATION.md', folder / 'IMPLEMENTATION_14.md')
    (folder / 'RESULTS_14.md').write_text('# Coverage experiment running\n\nResults are written automatically on completion.\n')
    (folder / 'README.md').write_text('# Latest: is the layer gap a coverage effect?\n\n[Implementation](IMPLEMENTATION_14.md), [results](RESULTS_14.md), [older batches](old/README.md).\n')


def main(cfg):
    OmegaConf.resolve(cfg)
    configure(cfg)
    cfg.forge.store = str(absolute(cfg.forge.store))
    tracking = start_run(cfg)
    root = absolute(cfg.coverage.results)
    stage = cfg.stage.experiment
    output = root / stage
    output.mkdir(parents=True, exist_ok=True)
    if not (root / 'timing.json').exists():
        save_json(root / 'timing.json', {'started_unix': time.time()})
    if not (ROOT / 'report' / 'IMPLEMENTATION_14.md').exists():
        archive()
    started = time.monotonic()
    print(f'Coverage {stage}: {output}', flush=True)
    if stage == 'train':
        result = train.experiment(cfg, output, json.loads((root / 'estimate' / 'estimate.json').read_text())['fractions'])
    else:
        result = {'prepare': data.prepare, 'estimate': estimate.experiment, 'analyse': analyse.experiment, 'report': report.experiment}[stage](cfg, output)
    save_json(output / 'runtime.json', {'seconds': time.monotonic() - started})
    save_json(tracking.path / 'artifacts.json', {'results': str(output)})
    tracking.finish(result)
    return 0


if __name__ == '__main__':
    explicit = any(a.startswith('stage.experiment=') for a in sys.argv[1:])
    stages = [None] if explicit else ['prepare', 'estimate', 'train', 'analyse', 'report']
    for stage in stages:
        code = run('lion.coverage', sys.argv[1:] + ([] if stage is None else [f'stage.experiment={stage}']),
                   main_module='run', config_dir=str(ROOT / 'conf'), config_name='coverage')
        if code:
            sys.exit(code)
