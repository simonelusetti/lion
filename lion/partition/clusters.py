"""X4: label-free inside/outside context clusters with held-document-out scoring."""
import math

import torch
from scipy.optimize import linear_sum_assignment

from ..diagnostics.common import write_csv
from .data import load_banks, load_features, pooled
from .stats import bootstrap, interval


def kmeans(x, clusters, seed, cfg):
    x = x.to(cfg.runtime.device).float()
    generator = torch.Generator(device=x.device).manual_seed(seed)
    centers = [x[torch.randint(len(x), (), generator=generator, device=x.device)]]
    minimum = torch.full((len(x),), torch.inf, device=x.device)
    for _ in range(1, clusters):
        minimum = torch.minimum(minimum, (x - centers[-1]).square().sum(-1))
        index = torch.multinomial(minimum, 1, generator=generator)[0] if minimum.sum() > 0 else torch.randint(len(x), (), generator=generator, device=x.device)
        centers.append(x[index])
    centers = torch.stack(centers)
    for iteration in range(cfg.partition.kmeans_iterations):
        assignment = (x.square().sum(-1, keepdim=True) + centers.square().sum(-1)[None] - 2 * x @ centers.T).argmin(-1)
        count = torch.bincount(assignment, minlength=clusters)
        updated = torch.zeros_like(centers).index_add_(0, assignment, x) / count[:, None].clamp_min(1)
        updated = torch.where(count[:, None] > 0, updated, centers)
        difference = (updated - centers).square().sum().sqrt()
        centers = updated
        if difference < cfg.partition.kmeans_tolerance: break
    return centers, iteration + 1


def assign(x, centers, cfg):
    x = x.to(cfg.runtime.device).float()
    return (x.square().sum(-1, keepdim=True) + centers.square().sum(-1)[None] - 2 * x @ centers.T).argmin(-1).cpu()


def matching(cluster, target, clusters, domains, allowed=None):
    allowed = torch.arange(domains) if allowed is None else allowed
    dense = torch.full((domains,), -1, dtype=torch.long)
    dense[allowed] = torch.arange(len(allowed))
    target = dense[target]
    domains = len(allowed)
    counts = torch.bincount(cluster * domains + target, minlength=clusters * domains).reshape(clusters, domains).double()
    balanced = counts / counts.sum(0, keepdim=True).clamp_min(1)
    capacity = math.ceil(clusters / domains)
    rows, columns = linear_sum_assignment((-balanced.repeat_interleave(capacity, dim=1)).numpy())
    mapping = torch.empty(clusters, dtype=torch.long)
    mapping[torch.as_tensor(rows)] = allowed[torch.as_tensor(columns) // capacity]
    return mapping


def nmi_boot(cluster, target, inverse, weights, clusters, domains):
    index = inverse * clusters * domains + cluster * domains + target
    by_doc = torch.bincount(index, minlength=weights.shape[1] * clusters * domains).reshape(weights.shape[1], -1).float()
    counts = (weights @ by_doc).reshape(-1, clusters, domains)
    total = counts.sum((-1, -2))
    probability = counts / total[:, None, None]
    a, b = probability.sum(-1), probability.sum(-2)
    information = (torch.special.xlogy(probability, probability) - torch.special.xlogy(probability, a[:, :, None] * b[:, None])).sum((-1, -2))
    entropy = -torch.special.xlogy(a, a).sum(-1) - torch.special.xlogy(b, b).sum(-1)
    value = torch.where(entropy > 0, 2 * information / entropy.clamp_min(1e-12), torch.ones_like(entropy))
    return torch.where(total > 0, value, torch.zeros_like(value)), total


def balanced_boot(prediction, target, mask, inverse, weights, domains):
    values = []
    for domain in domains.tolist():
        selected = mask & (target == domain)
        count = torch.bincount(inverse[selected], minlength=weights.shape[1]).float()
        correct = torch.zeros(weights.shape[1]).index_add_(0, inverse[selected], (prediction[selected] == target[selected]).float())
        values.append((weights @ correct) / (weights @ count))
    return torch.stack(values).mean(0)


def experiment(cfg, output):
    rows, counts, skips = [], [], []
    for name, bank in load_banks(cfg):
        features = load_features(cfg, bank, "pooling")
        x, valid = pooled(features), pooled(features, "valid")
        pool = bank["pool"]
        inverse, weights = bootstrap(pool, cfg)
        domains = len(bank["z_names"])
        counts.append({"dataset": name, "all_focals": len(valid), "paired_nonempty_inside_outside": int(valid.sum())})
        for multiplier in (1, 2):
            clusters = multiplier * domains
            for channel, channel_name in enumerate(("outside", "inside")):
                results = {}
                for seed in cfg.diagnostics.seeds:
                    sums = {}
                    oof = torch.full_like(pool["z"], -1)
                    conditional_oof = torch.full_like(pool["z"], -1)
                    for fold in range(cfg.partition.folds):
                        train = valid & (bank["fold"] != fold)
                        test = valid & (bank["fold"] == fold)
                        centers, iterations = kmeans(x[train, channel], clusters, seed, cfg)
                        train_cluster = assign(x[train, channel], centers, cfg)
                        test_cluster = assign(x[test, channel], centers, cfg)
                        mapping = matching(train_cluster, pool["z"][train], clusters, domains)
                        oof[test] = mapping[test_cluster]
                        for label in [-1, *range(len(bank["labels"]))]:
                            allowed_domains = torch.arange(domains) if label == -1 else bank["eligible"][label].nonzero().flatten()
                            if len(allowed_domains) < 2: continue
                            selected_train = torch.ones(int(train.sum()), dtype=torch.bool) if label == -1 else ((pool["y"][train] == label) & bank["eligible"][label, pool["z"][train]])
                            selected_test = torch.ones(int(test.sum()), dtype=torch.bool) if label == -1 else ((pool["y"][test] == label) & bank["eligible"][label, pool["z"][test]])
                            if int(selected_train.sum()) < clusters or not selected_test.any():
                                skips.append({"dataset": name, "label": label, "fold": fold, "reason": "insufficient nonempty paired pooling examples"})
                                continue
                            measured = nmi_boot(test_cluster[selected_test], pool["z"][test][selected_test], inverse[test][selected_test], weights, clusters, domains)
                            if label not in sums: sums[label] = [torch.zeros(weights.shape[0]), torch.zeros(weights.shape[0])]
                            sums[label][0] += measured[0] * measured[1]
                            sums[label][1] += measured[1]
                            if label >= 0:
                                local_mapping = matching(train_cluster[selected_train], pool["z"][train][selected_train], clusters, domains, allowed_domains)
                                ids = test.nonzero().flatten()[selected_test]
                                conditional_oof[ids] = local_mapping[test_cluster[selected_test]]
                        folder = output / name / f"seed{seed}" / channel_name
                        folder.mkdir(parents=True, exist_ok=True)
                        torch.save({"centers": centers.cpu(), "mapping": mapping, "iterations": iterations}, folder / f"K{clusters}_fold{fold}.pt")
                    for label, (numerator, denominator) in sums.items():
                        mask = valid if label == -1 else (valid & (pool["y"] == label) & bank["eligible"][label, pool["z"]] & (conditional_oof >= 0))
                        allowed = torch.arange(domains) if label == -1 else bank["eligible"][label].nonzero().flatten()
                        ba = balanced_boot(oof if label == -1 else conditional_oof, pool["z"], mask, inverse, weights, allowed)
                        results.setdefault(label, []).append({"NMI": numerator / denominator, "balanced_accuracy": ba})
                for label, values in results.items():
                    record = {"dataset": name, "channel": channel_name, "K": clusters,
                              "label": "all" if label == -1 else bank["labels"][label]}
                    for metric in ("NMI", "balanced_accuracy"):
                        record.update({metric + "_" + k: v for k, v in interval(torch.stack([v[metric] for v in values])).items()})
                    rows.append(record)
    write_csv(output / "scores.csv", rows)
    write_csv(output / "samples.csv", counts)
    write_csv(output / "skips.csv", skips)
    return {"cluster_conditions": len(rows)}
