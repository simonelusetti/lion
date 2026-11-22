"""Sequential Forge stages: estimate, train, measure, analyse, report."""
import json
import shutil
import sys
import time
from forge import run, start_run
from omegaconf import OmegaConf
from ..diagnostics.common import ROOT, absolute, save_json
from ..runtime import configure
from . import estimate, train, measure, analyse, report


def archive():
    folder = ROOT / 'report'
    old = folder / 'old' / '12_deprel_replication'
    old.mkdir(parents=True, exist_ok=True)
    for name in ('RESULTS_12.md', 'IMPLEMENTATION_12.md'):
        if (folder / name).exists():
            shutil.move(str(folder / name), old / name)
    tables = folder / 'tables' / 'deprel'
    if tables.exists():
        (old / 'tables').mkdir(exist_ok=True)
        shutil.move(str(tables), old / 'tables' / 'deprel')
    (old / 'README.md').write_text('# EWT dependency-relation replication\n\n[Results](RESULTS_12.md), [implementation](IMPLEMENTATION_12.md).\n')
    index = folder / 'old' / 'README.md'
    if '12_deprel_replication' not in index.read_text():
        index.write_text(index.read_text() + '\n- [EWT dependency-relation replication](12_deprel_replication/README.md)\n')
    shutil.copy2(ROOT / 'lion' / 'layer_sensitivity' / 'IMPLEMENTATION.md', folder / 'IMPLEMENTATION_13.md')
    (folder / 'RESULTS_13.md').write_text('# Layer sensitivity running\n\nResults are written automatically on completion.\n')
    (folder / 'README.md').write_text('# Latest: context sensitivity across encoder layers\n\n[Implementation](IMPLEMENTATION_13.md), [results](RESULTS_13.md), [older batches](old/README.md).\n')


def main(cfg):
    OmegaConf.resolve(cfg)
    configure(cfg)
    cfg.forge.store = str(absolute(cfg.forge.store))
    tracking = start_run(cfg)
    root = absolute(cfg.layers.results)
    stage = cfg.stage.experiment
    output = root / stage
    output.mkdir(parents=True, exist_ok=True)
    if not (root / 'timing.json').exists():
        save_json(root / 'timing.json', {'started_unix': time.time()})
    if not (ROOT / 'report' / 'IMPLEMENTATION_13.md').exists():
        archive()
    started = time.monotonic()
    print(f'Layer sensitivity {stage}: {output}', flush=True)
    if stage == 'estimate':
        result = estimate.experiment(cfg, output)
    else:
        chosen = json.loads((root / 'estimate' / 'estimate.json').read_text())['layers']
        cfg.layers.layers = chosen
        if stage == 'measure':
            result = measure.experiment(cfg, output, chosen)
        else:
            result = {'train': train.experiment, 'analyse': analyse.experiment, 'report': report.experiment}[stage](cfg, output)
    save_json(output / 'runtime.json', {'seconds': time.monotonic() - started})
    save_json(tracking.path / 'artifacts.json', {'results': str(output)})
    tracking.finish(result)
    return 0


if __name__ == '__main__':
    explicit = any(a.startswith('stage.experiment=') for a in sys.argv[1:])
    stages = [None] if explicit else ['estimate', 'train', 'measure', 'analyse', 'report']
    for stage in stages:
        code = run('lion.layer_sensitivity', sys.argv[1:] + ([] if stage is None else [f'stage.experiment={stage}']),
                   main_module='run', config_dir=str(ROOT / 'conf'), config_name='layer_sensitivity')
        if code:
            sys.exit(code)
