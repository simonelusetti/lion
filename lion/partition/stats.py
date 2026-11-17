"""Paired document bootstrap and conditional nuisance scores."""
import torch
from torch.nn import functional as F


def documents(names):
    unique = sorted(set(names))
    lookup = {name: i for i, name in enumerate(unique)}
    return unique, torch.tensor([lookup[name] for name in names])


def bootstrap(data, cfg):
    names, inverse = documents(data["document"])
    doc_z = torch.empty(len(names), dtype=torch.long)
    doc_z[inverse] = data["z"]
    generator = torch.Generator().manual_seed(cfg.partition.bootstrap_seed)
    weights = torch.zeros(cfg.partition.bootstrap + 1, len(names))
    weights[0] = 1
    for domain in doc_z.unique():
        ids = (doc_z == domain).nonzero().flatten()
        draws = ids[torch.randint(len(ids), (cfg.partition.bootstrap, len(ids)), generator=generator)]
        weights[1:].scatter_add_(1, draws, torch.ones_like(draws, dtype=torch.float32))
    return inverse, weights


def label_metrics(logits, y, inverse, weights):
    n, conditions, classes = logits.shape
    docs = weights.shape[1]
    prediction = logits.argmax(-1)
    offset = inverse[:, None] * conditions * classes * classes + torch.arange(conditions)[None] * classes * classes
    flat = offset + y[:, None] * classes + prediction
    counts = torch.bincount(flat.flatten(), minlength=docs * conditions * classes * classes).reshape(docs, conditions, classes, classes).float()
    confusion = torch.einsum("bd,dckl->bckl", weights, counts)
    correct = confusion.diagonal(dim1=-2, dim2=-1)
    wanted, predicted = confusion.sum(-1), confusion.sum(-2)
    ce = -logits.log_softmax(-1).gather(-1, y[:, None, None].expand(-1, conditions, 1)).squeeze(-1)
    ce_docs = torch.zeros(docs, conditions).index_add_(0, inverse, ce)
    return {"accuracy": correct.sum(-1) / wanted.sum(-1),
            "macro_f1": (2 * correct / (wanted + predicted).clamp_min(1)).mean(-1),
            "cross_entropy": (weights @ ce_docs) / wanted.sum(-1)}


def conditional_accuracy(predicted, data, eligible, inverse, weights):
    # Each cell is one label/domain pair; fixed eligibility is shared by G and all conditions.
    n, conditions = predicted.shape
    labels, domains = eligible.shape
    totals = torch.zeros(weights.shape[1], labels * domains)
    correct = torch.zeros(weights.shape[1], conditions, labels * domains)
    cell = data["y"] * domains + data["z"]
    allowed = eligible[data["y"], data["z"]]
    ids = allowed.nonzero().flatten()
    totals.flatten().scatter_add_(0, inverse[ids] * labels * domains + cell[ids], torch.ones(len(ids)))
    offset = inverse[ids, None] * conditions * labels * domains + torch.arange(conditions)[None] * labels * domains + cell[ids, None]
    correct.flatten().scatter_add_(0, offset.flatten(), (predicted[ids] == data["z"][ids, None]).float().flatten())
    denominator = (weights @ totals).reshape(-1, labels, domains)
    numerator = torch.einsum("bd,dcl->bcl", weights, correct).reshape(-1, conditions, labels, domains)
    ratio = numerator / denominator[:, None]
    per_label = (ratio.masked_fill(~eligible[None, None], 0).sum(-1) /
                 eligible.sum(-1).clamp_min(1)[None, None])
    return per_label[:, :, eligible.sum(-1) >= 2].mean(-1)


def interval(samples):
    # Input [seed, observed+bootstrap, ...]. Bootstrap the seed mean, not the seed identities.
    point = samples[:, 0]
    boot = samples[:, 1:].mean(0)
    return {"mean": float(point.mean()), "seed_sd": float(point.std(unbiased=len(point) > 1)),
            "ci_low": float(torch.nanquantile(boot, .025)), "ci_high": float(torch.nanquantile(boot, .975)),
            "valid_bootstraps": int(torch.isfinite(boot).sum())}


def weighted_scalar(values, inverse, weights):
    count = torch.bincount(inverse, minlength=weights.shape[1]).float()
    total = torch.zeros(weights.shape[1], *values.shape[1:]).index_add_(0, inverse, values.float())
    return torch.einsum("bd,d...->b...", weights, total) / (weights @ count).reshape(-1, *([1] * (values.ndim - 1)))
