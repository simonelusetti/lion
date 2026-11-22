"""Pre-run estimate: timed steps and dev evaluations at real tensor sizes, times each run's step/epoch counts."""
import math
import time
import torch
from torch.nn import functional as F
from tagger.ensemble import clip_heads
from ..partition.train import metric_vectors
from ..runtime import autocast
from ..diagnostics.common import save_json
from ..layer_sensitivity.train import Head, MIX, logits
from . import data
from .train import epochs_for

FEATURE_SECONDS = 180.0  # all-layer vectors for train/dev/test; measured ~1–2 min per dataset in the layer run


def timed(src, labels, layer, words, dev_words, steps=60, evaluations=3):
    device = src.runtime.device
    x = torch.randn(words, 13, 768, device=device, dtype=torch.float16)
    y = torch.randint(len(labels), (words,), device=device)
    dev = torch.randn(dev_words, 13, 768, device=device, dtype=torch.float16)
    dev_y = torch.randint(len(labels), (dev_words,))
    model = Head(src, len(labels), layer).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=src.train.lr)
    per_batch = words // max(1, words // 900)
    for warm in (True, False):
        torch.cuda.synchronize()
        clock = time.monotonic()
        for _ in range(5 if warm else steps):
            ids = torch.randint(words, (per_batch,), device=device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(src):
                output = model(x[ids]).float()
            F.cross_entropy(output, y[ids]).backward()
            clip_heads(model.core, 1.0)
            optimizer.step()
        torch.cuda.synchronize()
        step = (time.monotonic() - clock) / steps
    clock = time.monotonic()
    for _ in range(evaluations):
        metric_vectors(logits(model, dev, src)[:, None], dev_y, len(labels))
    torch.cuda.synchronize()
    evaluation = (time.monotonic() - clock) / evaluations
    del x, dev
    torch.cuda.empty_cache()
    return step, evaluation


def experiment(cfg, output):
    result = {'datasets': {}}
    for name in cfg.coverage.datasets:
        src = data.source(cfg, name)
        bank = data.full_bank(cfg, name)
        rows = bank['sets']['train']
        words = sum(len(r['focals']) for r in rows)
        dev_words = sum(len(r['focals']) for r in bank['sets']['validation'])
        labels = bank['labels'][data.task(bank)]
        costs = {kind: timed(src, labels, layer, words, dev_words) for kind, layer in (('layer', 12), ('mixture', MIX))}
        per_fraction = {}
        for fraction in cfg.coverage.fractions:
            sentences = math.ceil(fraction * len(rows))
            epochs = epochs_for(sentences, len(rows), src.train)
            steps = epochs * math.ceil(sentences / src.train.batch_size)
            seconds = len(cfg.coverage.seeds) * (len(cfg.coverage.layers) * (steps * costs['layer'][0] + epochs * costs['layer'][1])
                                                 + steps * costs['mixture'][0] + epochs * costs['mixture'][1])
            per_fraction[str(fraction)] = {'sentences': sentences, 'epochs_cap': epochs, 'steps': steps, 'seconds': seconds}
        result['datasets'][name] = {'train_sentences': len(rows), 'train_words': words, 'dev_words': dev_words,
                                    'seconds_per_step': {k: v[0] for k, v in costs.items()}, 'seconds_per_dev_evaluation': {k: v[1] for k, v in costs.items()},
                                    'fractions': per_fraction, 'features_seconds': FEATURE_SECONDS,
                                    'total_seconds': FEATURE_SECONDS + sum(v['seconds'] for v in per_fraction.values())}
        print(name, result['datasets'][name], flush=True)
    analysis = 300.0
    result['total_seconds'] = analysis + sum(d['total_seconds'] for d in result['datasets'].values())
    result['total_hours'] = result['total_seconds'] / 3600
    result['dropped_smallest_fraction'] = result['total_seconds'] > cfg.coverage.runtime_budget_seconds
    result['fractions'] = [f for f in cfg.coverage.fractions if not (result['dropped_smallest_fraction'] and f == min(cfg.coverage.fractions))]
    save_json(output / 'estimate.json', result)
    return {'total_hours': result['total_hours'], 'fractions': result['fractions']}
