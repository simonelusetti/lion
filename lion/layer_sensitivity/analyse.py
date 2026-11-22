"""Per-head sensitivity, matched ratios, representation shift, gain and contextuality."""
import torch
from omegaconf import OmegaConf
from tagger.ensemble import IndependentReadouts
from ..diagnostics.common import absolute, write_csv, read_csv
from . import data
from .train import MIX, head_name

COUNT_BINS = ['0', '1', '2', '3', '4', '5–9', '10+']
DISTANCE_BINS = ['1', '2', '3', '4–5', '6–10', '>10']


def reduce(value, eligible=None):
    """[seeds, rows, fold, kind, draw] -> [seeds, rows, kind]: eligible draws within fold, then folds within row."""
    if eligible is None:
        return value.mean(-1).mean(2)
    count = eligible.sum(-1)
    mean = (value * eligible[None]).sum(-1) / count[None].clamp_min(1)
    return torch.nanmean(mean.masked_fill(count[None] == 0, torch.nan), dim=2)


def summary(x, mask):
    """x [seeds, rows]: mean ± seed SD and row SE of the seed-averaged row values, on common finite rows."""
    rows = mask & torch.isfinite(x).all(0)
    values = x[:, rows]
    n = int(rows.sum())
    point = values.mean(1)
    return {'rows': n, 'small_bin': n < 50, 'mean': float(point.mean()),
            'seed_sd': float(point.std()) if len(point) > 1 else 0.0,
            'row_se': float(values.mean(0).std() / n ** .5) if n > 1 else float('nan')}


def ratio(a, b, mask):
    rows = mask & torch.isfinite(a).all(0) & torch.isfinite(b).all(0)
    a, b = a[:, rows], b[:, rows]
    values = a.mean(1) / b.mean(1)
    pooled = float(a.mean() / b.mean())
    se = float(((a.mean(0) - pooled * b.mean(0)) / b.mean()).std() / int(rows.sum()) ** .5)
    return {'rows': int(rows.sum()), 'mean': float(values.mean()), 'seed_sd': float(values.std()) if len(values) > 1 else 0.0, 'row_se': se}


def matched(kl, changed, nearest, mask, minimum):
    """Step-1b size/distance-matched selector/random ratio over non-null natural draws."""
    counts = torch.where(changed < 5, changed, torch.where(changed < 10, 5, 6))
    distance = torch.bucketize(nearest, torch.tensor([1, 2, 3, 5, 10]))
    seeds, rows = kl.shape[0], kl.shape[1]
    total = int(((changed[:, :, 0] > 0) & mask[:, None, None]).sum())
    weights, ratios, influences, cells = [], [], [], []
    for c in range(1, 7):
        for d in range(6):
            eligible = (counts == c) & (distance == d) & (changed > 0)
            value = reduce(kl, eligible)
            common = mask & torch.isfinite(value).all((0, 2))
            n = int(common.sum())
            a, b = value[:, common, 0], value[:, common, 1]
            mass = int((eligible[:, :, 0] & mask[:, None, None]).sum())
            valid = n >= minimum and bool((b.mean(1) > 0).all())
            cells.append({'count_bin': COUNT_BINS[c], 'distance_bin': DISTANCE_BINS[d], 'paired_rows': n, 'small_bin': n < minimum, 'supported': valid})
            if valid:
                r = a.mean(1) / b.mean(1)
                weights.append(mass)
                ratios.append(r)
                influence = torch.zeros(seeds, rows, dtype=torch.float64)
                influence[:, common] = (a - r[:, None] * b) / (b.mean(1)[:, None] * n)
                influences.append(influence)
    mass = sum(weights)
    if not mass:  # e.g. layer 0 under natural fills: random S is exactly zero in every cell
        return {'mean': float('nan'), 'seed_sd': float('nan'), 'row_se': float('nan'), 'supported_cells': 0, 'retained_mass': 0.0}, cells
    w = torch.tensor(weights, dtype=torch.float64) / mass
    r = (torch.stack(ratios, 1) * w[None]).sum(1)
    influence = (torch.stack(influences, 1) * w[None, :, None]).sum(1).mean(0)
    se = float((influence.square().sum() * rows / (rows - 1)).sqrt())
    return {'mean': float(r.mean()), 'seed_sd': float(r.std()) if seeds > 1 else 0.0, 'row_se': se,
            'supported_cells': len(weights), 'retained_mass': mass / total}, cells


def baseline_check(cfg, name, src, train_folder):
    """Existing baseline (block 12) against the new layer-12 head; cached vectors against recomputed ones."""
    spec = cfg.layers.datasets[name]
    raw = torch.load(absolute(spec.fills) / 'raw.pt', mmap=True, weights_only=False)['features'][:, 0].float()
    new = torch.load(train_folder / 'test_features.pt', weights_only=False)['features'][:, 12].float()
    cosine = torch.nn.functional.cosine_similarity(raw, new, dim=-1)
    records = []
    performance = [r for r in read_csv(train_folder.parent / 'performance.csv') if r['dataset'] == name and r['head'] == 'layer12']
    for seed in cfg.layers.seeds:
        if 'ner' in data.data.task_names(src):
            saved = torch.load(absolute(spec.baseline) / f'seed{seed}' / 'model.pt', map_location='cpu', weights_only=False)
            model = IndependentReadouts(src.tagger, 768, len(saved['labels']), saved.get('heads', 3))
            model.load_state_dict(saved['state'])
            with torch.no_grad():
                old = torch.cat([model.eval()(part[:, None])[:, 0] for part in raw.split(8192)])
            existing = next(float(r['span_f1']) for r in read_csv(absolute(spec.baseline) / 'per_seed.csv')
                            if (r['seed'], r['arm'], r['target'], r['split'], r['label_space']) == (str(seed), 'baseline', 'conll2003', 'id', 'native'))
            metric = 'span_f1'
        else:
            old = torch.load(absolute(spec.baseline) / f'seed{seed}' / 'test.pt', weights_only=False)['logits'][:, 0]
            existing = next(float(r['macro_f1']) for r in read_csv(absolute(spec.baseline) / 'performance.csv') if r['classifier'] == 'baseline' and r['seed'] == str(seed))
            metric = 'macro_f1'
        new_logits = torch.load(train_folder / 'layer12' / f'seed{seed}' / 'test.pt')['logits']
        mine = next(float(r[metric]) for r in performance if r['seed'] == str(seed))
        records.append({'dataset': name, 'seed': seed, 'metric': metric, 'existing_baseline': existing, 'new_layer12_head': mine,
                        'difference': mine - existing, 'test_prediction_agreement': float((old.argmax(-1) == new_logits.argmax(-1)).double().mean()),
                        'cached_vs_recomputed_block12_min_cosine': float(cosine.min()), 'cached_vs_recomputed_block12_max_abs': float((raw - new).abs().max())})
    return records


def experiment(cfg, output):
    root = absolute(cfg.layers.results)
    rows_out = {k: [] for k in ('sensitivity', 'ratios', 'distance', 'cells', 'context', 'baseline_check')}
    for name in cfg.layers.datasets:
        src = data.source(cfg, name)
        m = torch.load(root / 'measure' / f'{name}.pt', weights_only=False)
        kl, flip, shift = m['kl'].double(), m['flip'].double(), m['shift'].double()
        changed, nearest, dropped = m['changed'], m['nearest_changed'], m['nearest_dropped']
        n = kl.shape[1]
        groups = {'all': torch.ones(n, dtype=torch.bool)} if list(cfg.layers.datasets[name].groups) == ['all'] else {'O': m['group'] == 0, 'entity': m['group'] == 1}
        pair = m['pair_distance'].mean(1).double()
        heads = {}
        for i, (layer, seed) in enumerate(m['heads']):
            heads.setdefault(layer, []).append(i)
        distance_bin = torch.bucketize(dropped, torch.tensor([1, 2, 3, 5, 10]))
        rows_out['baseline_check'] += baseline_check(cfg, name, src, root / 'train' / name)
        for layer, index in heads.items():
            head = head_name(layer)
            view = [m['views'].index(f'mixture_seed{s}') for l, s in m['heads'] if l == MIX] if layer == MIX else [layer] * len(index)
            k, f = kl[index], flip[index]
            norm = (shift[view] / pair[view][:, None, None, None, None])
            measures = {
                ('natural', 'all_draws'): (reduce(k[..., :4]), reduce(f[..., :4]), reduce(norm[..., :4])),
                ('natural', 'non_null'): (reduce(k[..., :4], changed > 0), reduce(f[..., :4], changed > 0), reduce(norm[..., :4], changed > 0)),
                ('masked', 'all_rows'): (reduce(k[..., 4:]), reduce(f[..., 4:]), reduce(norm[..., 4:])),
            }
            for group, mask in groups.items():
                base = {'dataset': name, 'group': group, 'head': head, 'layer': layer}
                for (perturbation, scope), (s, fl, sh) in measures.items():
                    for j, kind in enumerate(data.KINDS):
                        for metric, value in (('S', s), ('flip', fl), ('normalized_shift', sh), ('raw_shift', sh * pair[view][:, None, None])):
                            rows_out['sensitivity'].append({**base, 'perturbation': perturbation, 'scope': scope, 'mask': kind, 'metric': metric, **summary(value[..., j], mask)})
                        rows_out['sensitivity'].append({**base, 'perturbation': perturbation, 'scope': scope, 'mask': kind, 'metric': 'gain',
                                                        **ratio(s[..., j], sh[..., j], mask)})
                    rows_out['ratios'].append({**base, 'perturbation': perturbation, 'scope': scope, 'kind': 'raw_selector_random', **ratio(s[..., 0], s[..., 1], mask)})
                result, cells = matched(k[..., :4], changed, nearest, mask, cfg.layers.minimum_rows)
                rows_out['ratios'].append({**base, 'perturbation': 'natural', 'scope': 'non_null', 'kind': 'matched_selector_random', **result})
                rows_out['cells'] += [{**base, **c} for c in cells]
                for b, label in enumerate(DISTANCE_BINS):
                    eligible = ((distance_bin == b) & (dropped > 0))[..., None]
                    for metric, value in (('S', k[..., 4:]), ('flip', f[..., 4:])):
                        reduced = reduce(value, eligible)
                        for j, kind in enumerate(data.KINDS):
                            rows_out['distance'].append({**base, 'perturbation': 'masked', 'bin': label, 'mask': kind, 'metric': metric, **summary(reduced[..., j], mask)})
                context = m['context'][view].double() / pair[view][:, None]
                rows_out['context'].append({**base, 'raw_pair_distance': float(pair[view].mean()), **summary(context, mask)})
    for key, values in rows_out.items():
        write_csv(output / f'{key}.csv', values)
    return {k: len(v) for k, v in rows_out.items()}
