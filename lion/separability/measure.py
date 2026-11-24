"""Per-word held-out linear separability, cross-word transfer, and alignment of label directions across words."""
import time
from itertools import combinations
import torch
from torch.nn import functional as F
from omegaconf import OmegaConf
from ..diagnostics.common import absolute, write_csv
from ..layer_sensitivity import data as layer_data
from ..label_share.measure import collapse, eligible


def occurrence_folds(tid, lab, folds, generator):
    """Stratified: within each (type, label) group, a random order assigned round-robin to folds."""
    group = tid * (int(lab.max()) + 1) + lab
    order = torch.argsort(group.double() + torch.rand(len(group), generator=generator, dtype=torch.float64))
    sorted_group = group[order]
    start = torch.searchsorted(sorted_group, sorted_group, right=False)
    fold = torch.empty_like(group)
    fold[order] = (torch.arange(len(group)) - start) % folds
    return fold


def lbfgs(parameters, loss, iterations):
    optimizer = torch.optim.LBFGS(parameters, max_iter=iterations, line_search_fn='strong_wolfe', tolerance_grad=1e-6, tolerance_change=1e-9)
    def closure():
        optimizer.zero_grad()
        value = loss()
        value.backward()
        return value
    optimizer.step(closure)


def per_word(x, tid, local, k, types, train, l2, iterations):
    """Independent multinomial logistic regression per type (joint objective = sum of per-type mean losses)."""
    device = x.device
    w = torch.zeros(types, k, x.shape[1], device=device, requires_grad=True)
    b = torch.zeros(types, k, device=device, requires_grad=True)
    xt, tt, yt = x[train], tid[train], local[train]
    n = torch.bincount(tt, minlength=types).float()
    weight = 1 / n[tt]
    valid = valid_labels(tid, local, k, types)
    def loss():
        logits = torch.einsum('nd,nkd->nk', xt, w[tt]) + b[tt]
        logits = logits.masked_fill(~valid[tt], -1e4)
        return (F.cross_entropy(logits, yt, reduction='none') * weight).sum() + l2 * w.square().sum()
    lbfgs([w, b], loss, iterations)
    return w.detach(), b.detach()


def valid_labels(tid, local, k, types):
    valid = torch.zeros(types, k, dtype=torch.bool, device=tid.device)
    valid[tid, local] = True
    return valid


def shared(x, lab, classes, train, l2, iterations):
    """One classifier over global labels, trained on other word types' occurrences."""
    w = torch.zeros(classes, x.shape[1], device=x.device, requires_grad=True)
    b = torch.zeros(classes, device=x.device, requires_grad=True)
    xt, yt = x[train], lab[train]
    lbfgs([w, b], lambda: F.cross_entropy(xt @ w.T + b, yt) + l2 * w.square().sum(), iterations)
    return w.detach(), b.detach()


def balanced_accuracy(prediction, tid, local, k, types):
    """Per type: mean over its labels of recall."""
    group = tid * k + local
    total = torch.bincount(group, minlength=types * k).double()
    hit = torch.bincount(group, weights=(prediction == local).double(), minlength=types * k)
    recall = (hit / total.clamp_min(1)).view(types, k)
    present = (total > 0).view(types, k)
    return (recall * present).sum(1) / present.sum(1)


def d_prime(score, tid, local, types):
    """Held-out separation-to-width along each fold's own direction between the type's two most frequent labels (local 0, 1)."""
    result = torch.full((types,), torch.nan, dtype=torch.float64)
    for t in range(types):
        a, b = score[(tid == t) & (local == 0)], score[(tid == t) & (local == 1)]
        result[t] = (a.mean() - b.mean()).abs() / ((a.var() + b.var()) / 2).sqrt()
    return result


def direction_alignment(x, tid, lab, types):
    """Mean cosine between different types' label-mean difference vectors, for every label pair shared by ≥2 types."""
    mean = {}
    for t in range(types):
        for y in lab[tid == t].unique().tolist():
            mean[(t, y)] = x[(tid == t) & (lab == y)].mean(0)
    pairs = {}
    for t in range(types):
        labels = sorted(lab[tid == t].unique().tolist())
        for a, b in combinations(labels, 2):
            pairs.setdefault((a, b), []).append(F.normalize(mean[(t, a)] - mean[(t, b)], dim=0))
    cosines, rows = [], []
    for (a, b), vectors in pairs.items():
        if len(vectors) < 2:
            continue
        v = torch.stack(vectors)
        c = v @ v.T
        off = c[~torch.eye(len(v), dtype=torch.bool, device=v.device)]
        cosines.append(off)
        rows.append({'label_pair': (a, b), 'types': len(v), 'mean_cosine': float(off.mean())})
    return float(torch.cat(cosines).mean()), rows


def experiment(cfg, output):
    spec = cfg.separability
    summary, per_type, pairs_out, timing = [], [], [], []
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
        T = len(types)
        # Local label index per type, ordered by frequency (local 0 and 1 = the two most frequent labels).
        local = torch.empty_like(lab)
        k = 0
        for t in range(T):
            mask = tid == t
            values, counts = lab[mask].unique(return_counts=True)
            order = values[counts.argsort(descending=True, stable=True)]
            local[mask] = (lab[mask][:, None] == order[None]).float().argmax(1)
            k = max(k, len(order))
        n_w = torch.bincount(tid, minlength=T).double()
        labels_per_type = torch.tensor([int(lab[tid == t].unique().numel()) for t in range(T)])
        generator = torch.Generator().manual_seed(cfg.seed)
        fold = occurrence_folds(tid, lab, spec.folds, generator)
        type_fold = torch.randperm(T, generator=generator) % spec.folds
        bert = layer_data.encoder(src)
        features = layer_data.layer_features(bert, bank['sets']['train'], src, labelled=False)
        del bert
        torch.cuda.empty_cache()
        mixtures = [torch.load(absolute(spec.coverage) / 'train' / name / 'f1.0' / 'mixture' / f'seed{s}' / 'result.pt', weights_only=False) for s in spec.seeds]
        views = [(f'layer{l}', l, None) for l in range(13)] + [(f'mixture_seed{s}', None, m) for s, m in zip(spec.seeds, mixtures)]
        tid_d, lab_d, local_d, fold_d = tid.to(device), lab.to(device), local.to(device), fold.to(device)
        held_type = type_fold.to(device)[tid_d]
        chance = (1 / labels_per_type.double())
        top = set(n_w.argsort(descending=True, stable=True)[:spec.top_types].tolist())
        for view, layer, mixture in views:
            if mixture is None:
                full = features['features'][:, layer]
            else:
                w = mixture['weights'].float()
                full = torch.cat([mixture['scale'] * torch.einsum('nld,l->nd', part.float(), w) for part in features['features'].split(16384)])
            mu = torch.zeros(full.shape[1], dtype=torch.float64)
            sq = torch.zeros(full.shape[1], dtype=torch.float64)
            for part in full.split(65536):
                part = part.double()
                mu += part.sum(0)
                sq += part.square().sum(0)
            mu /= len(full)
            sd = (sq / len(full) - mu.square()).clamp_min(1e-12).sqrt()
            # Per-dimension z-scores over all training words of this view, so one penalty scale fits every layer.
            x = ((full[rows].double() - mu) / sd).float().to(device)
            record = {'dataset': name, 'view': view, 'types': T, 'occurrences': len(rows)}
            results = {}
            for l2 in spec.l2:
                prediction = torch.empty_like(local_d)
                score = torch.empty(len(rows), dtype=torch.float64, device=device)
                for f in range(spec.folds):
                    w, b = per_word(x, tid_d, local_d, k, T, fold_d != f, l2, spec.max_iterations)
                    held = fold_d == f
                    logits = torch.einsum('nd,nkd->nk', x[held], w[tid_d[held]]) + b[tid_d[held]]
                    logits = logits.masked_fill(~valid_labels(tid_d, local_d, k, T)[tid_d[held]], -1e4)
                    prediction[held] = logits.argmax(-1)
                    direction = F.normalize(w[:, 0] - w[:, 1], dim=-1)
                    score[held] = (x[held] * direction[tid_d[held]]).sum(-1).double()
                word = balanced_accuracy(prediction, tid_d, local_d, k, T).cpu()
                separation = d_prime(score.cpu(), tid, local, T)
                transfer_prediction = torch.empty_like(local_d)
                for f in range(spec.folds):
                    w, b = shared(x, lab_d, len(classes), held_type != f, l2, spec.max_iterations)
                    held = held_type == f
                    logits = x[held] @ w.T + b
                    # Restrict to the held-out type's own labels, then map to its local label index.
                    allowed = torch.zeros(T, len(classes), dtype=torch.bool, device=device)
                    allowed[tid_d, lab_d] = True
                    choice = logits.masked_fill(~allowed[tid_d[held]], -1e4).argmax(-1)
                    order = torch.full((T, len(classes)), -1, dtype=torch.long, device=device)
                    order[tid_d, lab_d] = local_d
                    transfer_prediction[held] = order[tid_d[held], choice]
                transfer = balanced_accuracy(transfer_prediction, tid_d, local_d, k, T).cpu()
                results[l2] = (word, separation, transfer)
            weights = n_w / n_w.sum()
            agg = lambda v: float((weights * v).sum())
            best_word = max(spec.l2, key=lambda l2: agg(results[l2][0]))
            best_transfer = max(spec.l2, key=lambda l2: agg(results[l2][2]))
            word, separation, _ = results[best_word]
            transfer = results[best_transfer][2]
            cosine, label_pairs = direction_alignment(x, tid_d, lab_d, T)
            record.update({'chance_balanced_accuracy': agg(chance), 'per_word_balanced_accuracy': agg(word), 'per_word_l2': best_word,
                           'd_prime': agg(separation), 'transfer_balanced_accuracy': agg(transfer), 'transfer_l2': best_transfer,
                           'transfer_over_per_word_above_chance': agg(transfer - chance) / agg(word - chance), 'direction_cosine': cosine,
                           **{f'per_word_balanced_accuracy_l2_{l2}': agg(results[l2][0]) for l2 in spec.l2},
                           **{f'transfer_balanced_accuracy_l2_{l2}': agg(results[l2][2]) for l2 in spec.l2}})
            summary.append(record)
            for t, form in enumerate(types):
                per_type.append({'dataset': name, 'view': view, 'form': form, 'occurrences': int(n_w[t]), f'top_{spec.top_types}': t in top,
                                 'labels': int(labels_per_type[t]), 'chance': float(chance[t]), 'per_word_balanced_accuracy': float(word[t]),
                                 'd_prime': float(separation[t]), 'transfer_balanced_accuracy': float(transfer[t])})
            pairs_out += [{'dataset': name, 'view': view, 'label_pair': f'{classes[a]}|{classes[b]}', 'types': r['types'], 'mean_cosine': r['mean_cosine']}
                          for r in label_pairs for a, b in [r['label_pair']]]
            print(f'{name} {view}: per-word {record["per_word_balanced_accuracy"]:.4f} d\' {record["d_prime"]:.3f} transfer {record["transfer_balanced_accuracy"]:.4f} cos {cosine:.3f}', flush=True)
            write_csv(output / 'summary.csv', summary)
        timing.append({'dataset': name, 'seconds': time.monotonic() - clock})
        del features
    write_csv(output / 'summary.csv', summary)
    write_csv(output / 'per_type.csv', per_type)
    write_csv(output / 'label_pairs.csv', pairs_out)
    write_csv(output / 'timing.csv', timing)
    return {'views': len(summary)}
