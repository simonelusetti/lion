"""Pre-run estimate: measured all-layer encoding throughput times the tokens each stage encodes."""
import time
import torch
from ..diagnostics.common import save_json
from . import data

SECONDS_PER_HEAD = 15.0  # existing baseline: ~7 s per ten-epoch fit; doubled for per-epoch validation and the mixture


def experiment(cfg, output):
    result = {'datasets': {}}
    for name in cfg.layers.datasets:
        src = data.source(cfg, name)
        sentences = data.bank(cfg, name)
        caches = data.fills(cfg, name)
        items = []
        for cache in caches:
            pieces = data.row_pieces(cache)
            for kind in data.KINDS:
                items += data.perturbed(cache, kind, pieces, sentences['special']['mask'])
        perturbed_tokens = sum(len(i[2]) for i in items)
        rows = sentences['sets']
        feature_tokens = sum(len(r['ids']) for r in rows['train'] if r['fold'] in ('g1', 'g2')) + sum(len(r['ids']) for split in ('validation', 'id') for r in rows[split])
        sample = [items[i] for i in torch.randperm(len(items), generator=torch.Generator().manual_seed(cfg.seed))[:cfg.layers.estimate_sequences].tolist()]
        bert = data.encoder(src)
        for _ in data.encode(bert, sample[:256], caches[0]['focal'], src, cfg.layers.token_budget):
            pass  # warm-up
        torch.cuda.synchronize()
        clock = time.monotonic()
        for _ in data.encode(bert, sample, caches[0]['focal'], src, cfg.layers.token_budget):
            pass
        torch.cuda.synchronize()
        throughput = sum(len(i[2]) for i in sample) / (time.monotonic() - clock)
        del bert
        torch.cuda.empty_cache()
        heads = (len(cfg.layers.layers) + 1) * len(cfg.layers.seeds)
        seconds = {'features': feature_tokens / throughput * 1.5, 'train': heads * SECONDS_PER_HEAD,
                   'measure': perturbed_tokens / throughput * 1.3, 'analyse': 120.0}
        result['datasets'][name] = {'perturbed_sequences': len(items), 'perturbed_tokens': perturbed_tokens, 'feature_tokens': feature_tokens,
                                    'tokens_per_second': throughput, 'seconds': seconds, 'total_seconds': sum(seconds.values())}
        print(name, result['datasets'][name], flush=True)
    result['total_seconds'] = sum(d['total_seconds'] for d in result['datasets'].values())
    result['total_hours'] = result['total_seconds'] / 3600
    result['restricted'] = result['total_seconds'] > cfg.layers.runtime_budget_seconds
    result['layers'] = list(cfg.layers.restricted_layers if result['restricted'] else cfg.layers.layers)
    save_json(output / 'estimate.json', result)
    return {'total_hours': result['total_hours'], 'restricted': result['restricted']}
