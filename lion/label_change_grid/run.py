"""Generate → diagnostics → fit on train/dev → freeze dev choice → test only selected cell."""
import sys,time
from forge import run,start_run
from omegaconf import OmegaConf
from ..diagnostics.common import ROOT,absolute,save_json
from ..runtime import configure
from . import generate,train,select,evaluate


def main(cfg):
    OmegaConf.resolve(cfg);configure(cfg);cfg.forge.store=str(absolute(cfg.forge.store));tracking=start_run(cfg);root=generate.root(cfg);root.mkdir(parents=True,exist_ok=True);stage=cfg.stage.experiment;output=root/stage;output.mkdir(exist_ok=True)
    if not (root/'timing.json').exists():save_json(root/'timing.json',{'started_unix':time.time()})
    OmegaConf.save(cfg,root/'config.yaml')
    save_json(root/'protocol.json',{'dataset':'CoNLL-2003 existing pilot: 2400 fit sentences, 600 calibration sentences, same native dev/test and existing WNUT/WikiANN tests',
        'scorer':'Fixed existing scorer from results/label_change/source/scorer/ner; no retraining',
        'generator':'Uniform random subset in ±5 word window, fractions .2/.5/.7, nearest-integer half-up count; one existing simultaneous MLM pass, preserved focal/piece counts; no retries. Generate once per fraction; thresholds reuse fills.',
        'acceptance':'Original scorer argmax equals original gold, actual case-sensitive context change, modified argmax differs from original argmax, modified confidence >= tau (.7/.8/.9). Original confidence is not an additional filter.',
        'diagnostics':'Original 4000 error draws rethresholded per tau; same held-out 4000 candidate rows as generator comparison, filtered by the stricter rule. e is joint error AND high confidence, not conditional error. Pointwise per-cell 95% joint Clopper-Pearson bounds (Bonferroni .025 per quantity). No simultaneous grid certification claim, no certification gate, unknown joint true-label generator discrepancy.',
        'training':'Existing frozen block-12 768→256→9 MLP/AdamW/dropout/clip, ten epochs, seeds0/1/2. Identical initial readouts, optimizer steps and total labelled-token counts for all ten heads. Shared original training pool and shuffled original-token order. Each augmented head samples originals and variants in proportion to their dataset sizes, so original presentations differ slightly and are explicitly recorded; all original labels remain untouched. Baseline uses only originals. This equal-token-budget schedule replaces the earlier append-extra-token schedule.',
        'selection':'Within each head: existing native-dev token macro-F1 checkpoint selection. Choose ONE grid cell by mean native-dev span F1 across three seeds; deterministic first-grid-cell tie break. Baseline is a shared reference. Save selection before opening test cache.',
        'evaluation':'Reconstruct only baseline and selected head, not all ten, for test forwards. Raw untouched CoNLL native/shared and WNUT/WikiANN shared evaluation, one pass per seed. Never use test performance for configuration or seed selection.'})
    started=time.monotonic();print(f'Label-change grid {stage}: {output}',flush=True)
    result={'generate':generate.experiment,'calibrate':generate.calibrate,'train':train.experiment,'select':select.experiment,'test':evaluate.experiment}[stage](cfg,output)
    save_json(output/'runtime.json',{'seconds':time.monotonic()-started});save_json(tracking.path/'artifacts.json',{'results':str(output)});tracking.finish(result)
    if stage=='test':
        import json
        timing=json.loads((root/'timing.json').read_text());timing['ended_unix']=time.time();timing['wall_seconds']=timing['ended_unix']-timing['started_unix'];save_json(root/'timing.json',timing)
    return 0


if __name__=='__main__':
    stages=[None] if any(a.startswith('stage.experiment=') for a in sys.argv[1:]) else ['generate','calibrate','train','select','test']
    for stage in stages:
        code=run('lion.label_change_grid',sys.argv[1:]+([] if stage is None else [f'stage.experiment={stage}']),main_module='run',config_dir=str(ROOT/'conf'),config_name='label_change_grid')
        if code:sys.exit(code)
