"""Sequential Forge stages: measure, report."""
import shutil
import sys
import time
from forge import run, start_run
from omegaconf import OmegaConf
from ..diagnostics.common import ROOT, absolute, save_json
from ..runtime import configure
from . import measure, report


def archive():
    folder = ROOT / 'report'
    old = folder / 'old' / '14_coverage'
    old.mkdir(parents=True, exist_ok=True)
    for name in ('RESULTS_14.md', 'IMPLEMENTATION_14.md'):
        if (folder / name).exists():
            shutil.move(str(folder / name), old / name)
    tables = folder / 'tables' / 'coverage'
    if tables.exists():
        (old / 'tables').mkdir(exist_ok=True)
        shutil.move(str(tables), old / 'tables' / 'coverage')
    (old / 'README.md').write_text('# Is the layer gap a coverage effect?\n\n[Results](RESULTS_14.md), [implementation](IMPLEMENTATION_14.md).\n')
    index = folder / 'old' / 'README.md'
    if '14_coverage' not in index.read_text():
        index.write_text(index.read_text() + '\n- [Is the layer gap a coverage effect?](14_coverage/README.md)\n')
    shutil.copy2(ROOT / 'lion' / 'label_share' / 'IMPLEMENTATION.md', folder / 'IMPLEMENTATION_15.md')
    (folder / 'RESULTS_15.md').write_text('# Label-share analysis running\n\nResults are written automatically on completion.\n')
    (folder / 'README.md').write_text('# Latest: label-aligned share of contextual variation\n\n[Implementation](IMPLEMENTATION_15.md), [results](RESULTS_15.md), [older batches](old/README.md).\n')


def main(cfg):
    OmegaConf.resolve(cfg)
    configure(cfg)
    cfg.forge.store = str(absolute(cfg.forge.store))
    tracking = start_run(cfg)
    root = absolute(cfg.label_share.results)
    stage = cfg.stage.experiment
    output = root / stage
    output.mkdir(parents=True, exist_ok=True)
    if not (root / 'timing.json').exists():
        save_json(root / 'timing.json', {'started_unix': time.time()})
    if not (ROOT / 'report' / 'IMPLEMENTATION_15.md').exists():
        archive()
    started = time.monotonic()
    print(f'Label share {stage}: {output}', flush=True)
    result = {'measure': measure.experiment, 'report': report.experiment}[stage](cfg, output)
    save_json(output / 'runtime.json', {'seconds': time.monotonic() - started})
    save_json(tracking.path / 'artifacts.json', {'results': str(output)})
    tracking.finish(result)
    return 0


if __name__ == '__main__':
    explicit = any(a.startswith('stage.experiment=') for a in sys.argv[1:])
    stages = [None] if explicit else ['measure', 'report']
    for stage in stages:
        code = run('lion.label_share', sys.argv[1:] + ([] if stage is None else [f'stage.experiment={stage}']),
                   main_module='run', config_dir=str(ROOT / 'conf'), config_name='label_share')
        if code:
            sys.exit(code)
