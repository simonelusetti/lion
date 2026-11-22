"""Stream perturbed sequences once; keep per-row KL, flips and focal-vector shifts for every head."""
import time
import torch
from torch.nn import functional as F
from ..runtime import autocast
from ..diagnostics.common import absolute, write_csv
from . import data
from .train import Head, MIX, head_name


def load_heads(cfg, name, src, layers):
    folder = absolute(cfg.layers.results) / 'train' / name
    heads = []
    for layer in [*layers, MIX]:
        for seed in cfg.layers.seeds:
            saved = torch.load(folder / head_name(layer) / f'seed{seed}' / 'model.pt', weights_only=False)
            model = Head(src, saved['labels'], layer)
            model.load_state_dict(saved['state'])
            heads.append((layer, seed, model.to(src.runtime.device).eval().requires_grad_(False)))
    return heads


@torch.no_grad()
def probabilities(model, x, src, gamma):
    with autocast(src):
        output = model(x)
    return (1 - gamma) * output.double().softmax(-1) + gamma / output.shape[-1]


@torch.no_grad()
def views(heads, x):
    """Vectors whose shift is measured: the 13 layers, then each mixture seed's mixed vector."""
    mixed = [model.vector(x) for layer, _, model in heads if layer == MIX]
    return torch.cat([x.float(), torch.stack(mixed, 1)], 1)


def cosine_distance(a, b):
    return 1 - F.cosine_similarity(a.float(), b.float(), dim=-1)


@torch.no_grad()
def experiment(cfg, output, layers):
    timing = []
    for name in cfg.layers.datasets:
        started = time.monotonic()
        src = data.source(cfg, name)
        device = src.runtime.device
        sentences = data.bank(cfg, name)
        caches = data.fills(cfg, name)
        test = torch.load(absolute(cfg.layers.results) / 'train' / name / 'test_features.pt', weights_only=False)
        original = test['features'].to(device)
        heads = load_heads(cfg, name, src, layers)
        n, gamma = len(original), cfg.layers.gamma
        p = torch.stack([torch.cat([probabilities(m, part, src, gamma) for part in original.split(8192)]) for _, _, m in heads])
        reference = torch.cat([views(heads, part).half() for part in original.split(8192)])
        del original
        V = reference.shape[1]

        # Anisotropy reference: random pairs from different sentences with different word forms.
        forms = torch.load(absolute(cfg.layers.datasets[name].metadata), weights_only=False)['forms']
        index = {}
        type_id = torch.tensor([index.setdefault(form, len(index)) for form in forms])
        generator = torch.Generator().manual_seed(cfg.seed)
        a = torch.randint(n, (4 * cfg.layers.random_pairs,), generator=generator)
        b = torch.randint(n, (4 * cfg.layers.random_pairs,), generator=generator)
        valid = (test['sentence'][a] != test['sentence'][b]) & (type_id[a] != type_id[b])
        a, b = a[valid][:cfg.layers.random_pairs], b[valid][:cfg.layers.random_pairs]
        pair_distance = cosine_distance(reference[a.to(device)], reference[b.to(device)]).cpu()

        # Contextuality: each occurrence's distance from its word type's mean vector.
        count = torch.bincount(type_id)
        frequent = count[type_id] >= cfg.layers.minimum_occurrences
        types = type_id.to(device)
        context = torch.full((V, n), torch.nan)
        for v in range(V):
            vectors = reference[:, v].float()
            means = torch.zeros(len(count), vectors.shape[1], device=device).index_add_(0, types, vectors) / count.to(device)[:, None]
            context[v] = cosine_distance(vectors, means[types]).cpu().masked_fill(~frequent, torch.nan)

        kl = torch.zeros(len(heads), n, 2, 2, data.VERSIONS)
        flip = torch.zeros(len(heads), n, 2, 2, data.VERSIONS, dtype=torch.bool)
        shift = torch.zeros(V, n, 2, 2, data.VERSIONS)
        bert = data.encoder(src)
        sequences = 0
        for f, cache in enumerate(caches):
            pieces = data.row_pieces(cache)
            for k, kind in enumerate(data.KINDS):
                clock = time.monotonic()
                items = data.perturbed(cache, kind, pieces, sentences['special']['mask'])
                sequences += len(items)
                for rows, versions, vectors in data.encode(bert, items, cache['focal'], src, cfg.layers.token_budget):
                    r = rows.to(device)
                    for h, (_, _, model) in enumerate(heads):
                        q = probabilities(model, vectors, src, gamma)
                        kl[h, rows, f, k, versions] = (p[h, r] * (p[h, r].log() - q.log())).sum(-1).clamp_min(0).float().cpu()
                        flip[h, rows, f, k, versions] = (q.argmax(-1) != p[h, r].argmax(-1)).cpu()
                    shift[:, rows, f, k, versions] = cosine_distance(views(heads, vectors), reference[r]).T.cpu()
                timing.append({'dataset': name, 'step': f'{data.FOLDS[f]}/{kind}', 'perturbed_sequences': len(items), 'seconds': time.monotonic() - clock})
                print(f'{name} {data.FOLDS[f]}/{kind}: {len(items)} perturbed sequences, {time.monotonic() - clock:.0f}s', flush=True)
        del bert
        torch.cuda.empty_cache()
        stack = lambda key: torch.stack([torch.stack([c[f'{kind}_{key}'] for kind in data.KINDS], 1) for c in caches], 1)
        torch.save({'kl': kl, 'flip': flip, 'shift': shift, 'heads': [(l, s) for l, s, _ in heads],
                    'views': [head_name(l) for l in range(13)] + [f'mixture_seed{s}' for l, s, _ in heads if l == MIX],
                    'pair_distance': pair_distance, 'context': context,
                    'changed': stack('changed_count'), 'nearest_changed': stack('nearest_changed'),
                    'nearest_dropped': torch.stack([torch.stack([data.nearest_dropped(c, kind) for kind in data.KINDS], 1) for c in caches], 1),
                    'group': caches[0]['group'], 'prediction': p.argmax(-1).cpu()}, output / f'{name}.pt')
        timing.append({'dataset': name, 'step': 'total', 'perturbed_sequences': sequences, 'seconds': time.monotonic() - started})
        write_csv(output / 'timing.csv', timing)
    return {'datasets': len(cfg.layers.datasets)}
