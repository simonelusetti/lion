"""Within-type total spread, between-label spread, permutation baseline; raw and per-dimension standardized."""
import time
from collections import Counter
import torch
from ..diagnostics.common import absolute, write_csv
from ..layer_sensitivity import data as layer_data
from omegaconf import OmegaConf


def collapse(labels):
    """BIO tags -> entity type plus O; other label sets unchanged."""
    names = [l[2:] if l[:2] in ('B-', 'I-') else l for l in labels]
    kept = list(dict.fromkeys(names))
    return torch.tensor([kept.index(n) for n in names]), kept


def eligible(forms, gold, minimum):
    """Forms with at least two labels that each occur ≥ minimum times; only those labels' occurrences are kept."""
    counts = Counter(zip(forms, gold.tolist()))
    labels = {}
    for (form, label), n in counts.items():
        if label >= 0 and n >= minimum:
            labels.setdefault(form, set()).add(label)
    types = sorted(f for f, ls in labels.items() if len(ls) >= 2)
    index = {f: i for i, f in enumerate(types)}
    rows = [i for i, (f, y) in enumerate(zip(forms, gold.tolist())) if f in index and y in labels[f]]
    return types, torch.tensor(rows), torch.tensor([index[forms[i]] for i in rows])


def spreads(x, tid, lab, types, classes):
    """Per type: T = mean ‖x − μ_w‖², B = Σ_y p(y|w) ‖μ_wy − μ_w‖²."""
    n = torch.bincount(tid, minlength=types).double()
    mean = torch.zeros(types, x.shape[1], dtype=torch.float64, device=x.device).index_add_(0, tid, x) / n[:, None]
    total = torch.zeros(types, dtype=torch.float64, device=x.device).index_add_(0, tid, (x - mean[tid]).square().sum(-1)) / n
    group = tid * classes + lab
    m = torch.bincount(group, minlength=types * classes).double()
    group_mean = torch.zeros(types * classes, x.shape[1], dtype=torch.float64, device=x.device).index_add_(0, group, x) / m.clamp_min(1)[:, None]
    owner = torch.arange(types * classes, device=x.device) // classes
    between = torch.zeros(types, dtype=torch.float64, device=x.device).index_add_(0, owner, m / n[owner] * (group_mean - mean[owner]).square().sum(-1))
    return total, between


def shuffled(tid, lab, generator):
    """Permute labels within each type, keeping every type's label counts."""
    grouped = torch.argsort(tid, stable=True)
    random = torch.argsort(tid.double() + torch.rand(len(tid), generator=generator, dtype=torch.float64).to(tid.device))
    out = torch.empty_like(lab)
    out[random] = lab[grouped]
    return out


def measure(x, tid, lab, types, classes, shuffles, seed):
    total, between = spreads(x, tid, lab, types, classes)
    generator = torch.Generator().manual_seed(seed)
    baseline = torch.stack([spreads(x, tid, shuffled(tid, lab, generator), types, classes)[1] for _ in range(shuffles)]).mean(0) / total
    return total, between / total, baseline


def experiment(cfg, output):
    started = time.monotonic()
    summary, per_type, timing = [], [], []
    spec = cfg.label_share
    for name in spec.datasets:
        clock = time.monotonic()
        src = OmegaConf.load(absolute(spec.datasets[name].source_config))
        src.runtime = OmegaConf.merge(src.runtime, cfg.runtime)
        device = src.runtime.device
        path = spec.datasets[name].full_bank
        bank = torch.load((absolute(spec.datasets[name].source_config).parent / 'prepare' if path is None else absolute(path)) / 'sentences.pt', weights_only=False)
        prepared = torch.load(absolute(spec.coverage) / 'prepare' / f'{name}.pt', weights_only=False)
        forms = [w for s in prepared['full_forms'] for w in s]
        mapping, classes = collapse(prepared['labels'])
        raw_gold = torch.cat(prepared['full_gold'])
        gold = torch.where(raw_gold >= 0, mapping[raw_gold.clamp_min(0)], -1)
        types, rows, tid = eligible(forms, gold, spec.minimum_per_label)
        lab = gold[rows]
        bert = layer_data.encoder(src)
        features = layer_data.layer_features(bert, bank['sets']['train'], src, labelled=False)
        del bert
        torch.cuda.empty_cache()
        assert len(features['features']) == len(forms)
        mixtures = [torch.load(absolute(spec.coverage) / 'train' / name / 'f1.0' / 'mixture' / f'seed{s}' / 'result.pt', weights_only=False) for s in spec.seeds]
        views = [(f'layer{l}', l, None) for l in range(13)] + [(f'mixture_seed{s}', None, m) for s, m in zip(spec.seeds, mixtures)]
        sentence = features['sentence']
        index = {f: i for i, f in enumerate(dict.fromkeys(forms))}
        form_id = torch.tensor([index[f] for f in forms])
        generator = torch.Generator().manual_seed(cfg.seed)
        a = torch.randint(len(forms), (4 * spec.random_pairs,), generator=generator)
        b = torch.randint(len(forms), (4 * spec.random_pairs,), generator=generator)
        keep = (sentence[a] != sentence[b]) & (form_id[a] != form_id[b])
        a, b = a[keep][:spec.random_pairs], b[keep][:spec.random_pairs]
        n_w = torch.bincount(tid, minlength=len(types)).double()
        top = set(n_w.argsort(descending=True, stable=True)[:spec.top_types].tolist())
        type_labels = ['|'.join(classes[int(y)] for y in lab[tid == t].unique().tolist()) for t in range(len(types))]
        weights = (n_w / n_w.sum()).to(device)
        tid_d, lab_d = tid.to(device), lab.to(device)
        for view, layer, mixture in views:
            if mixture is None:
                full = features['features'][:, layer]
            else:
                w = mixture['weights'].float()
                full = torch.cat([mixture['scale'] * torch.einsum('nld,l->nd', part.float(), w) for part in features['features'].split(16384)])
            # Per-dimension mean/SD over all training words at this view.
            mu = torch.zeros(full.shape[1], dtype=torch.float64)
            sq = torch.zeros(full.shape[1], dtype=torch.float64)
            for part in full.split(65536):
                part = part.double()
                mu += part.sum(0)
                sq += part.square().sum(0)
            mu /= len(full)
            sd = (sq / len(full) - mu.square()).clamp_min(1e-12).sqrt()
            pair = float((full[a].double() - full[b].double()).square().sum(-1).mean())
            pair_std = float(((full[a].double() - full[b].double()) / sd).square().sum(-1).mean())
            x = full[rows].double().to(device)
            raw = measure(x, tid_d, lab_d, len(types), len(classes), spec.shuffles, cfg.seed)
            z = measure((x - mu.to(device)) / sd.to(device), tid_d, lab_d, len(types), len(classes), spec.shuffles, cfg.seed)
            record = {'dataset': name, 'view': view, 'types': len(types), 'occurrences': len(rows)}
            for prefix, (total, share, baseline), pair_distance in (('', raw, pair), ('standardized_', z, pair_std)):
                # A type whose occurrences are identical at this view (e.g. layer 0, always in the same position) has no spread: share undefined.
                defined = total > 1e-9 * pair_distance
                w = weights * defined / (weights * defined).sum()
                record.update({f'{prefix}total_spread': float((weights * total).sum()), f'{prefix}random_pair_sq_distance': pair_distance,
                               f'{prefix}normalized_total_spread': float((weights * total).sum()) / pair_distance,
                               f'{prefix}undefined_types': int((~defined).sum()),
                               f'{prefix}share': float((w * share.nan_to_num()).sum()), f'{prefix}baseline': float((w * baseline.nan_to_num()).sum()),
                               f'{prefix}share_minus_baseline': float((w * (share - baseline).nan_to_num()).sum())})
                if prefix:
                    z_defined = defined
                else:
                    raw_defined = defined
            summary.append(record)
            for t, form in enumerate(types):
                per_type.append({'dataset': name, 'view': view, 'form': form, 'occurrences': int(n_w[t]), f'top_{spec.top_types}': t in top,
                                 'labels': type_labels[t],
                                 'total_spread': float(raw[0][t]), 'share_defined': bool(raw_defined[t]), 'share': float(raw[1][t]), 'baseline': float(raw[2][t]), 'share_minus_baseline': float(raw[1][t] - raw[2][t]),
                                 'standardized_share_defined': bool(z_defined[t]), 'standardized_share': float(z[1][t]), 'standardized_baseline': float(z[2][t]), 'standardized_share_minus_baseline': float(z[1][t] - z[2][t])})
            print(f'{name} {view}: share−baseline {record["share_minus_baseline"]:.4f}, standardized {record["standardized_share_minus_baseline"]:.4f}', flush=True)
        timing.append({'dataset': name, 'seconds': time.monotonic() - clock})
        del features
    write_csv(output / 'summary.csv', summary)
    write_csv(output / 'per_type.csv', per_type)
    write_csv(output / 'timing.csv', timing)
    return {'views': len(summary), 'seconds': time.monotonic() - started}
