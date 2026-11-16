"""Within-label probability/lexical dependence with document-block permutations."""
from itertools import combinations

import torch

from .common import VIEWS, bins, entropy, plot_table, seed_summary, write_csv
from .data import banks
from .readouts import load_y_predictions
from .run import cli


def mmd_statistics(kernel, membership):
    a = membership.double()
    b = 1 - a
    n, m = a.sum(-1), b.sum(-1)
    ka, kb = a @ kernel, b @ kernel
    diag = kernel.diag()
    aa = ((ka * a).sum(-1) - a @ diag) / (n * (n - 1))
    bb = ((kb * b).sum(-1) - b @ diag) / (m * (m - 1))
    ab = (ka * b).sum(-1) / (n * m)
    return aa + bb - 2 * ab


def blocked_mmd(probability, groups, documents, permutations, generator):
    distance = torch.cdist(probability.double(), probability.double()).square()
    positive = distance[distance > 0]
    bandwidth = positive.median().clamp_min(1e-12) if len(positive) else distance.new_tensor(1.)
    kernel = (-distance / (2 * bandwidth)).exp()
    names = {name: i for i, name in enumerate(sorted(set(documents)))}
    inverse = torch.tensor([names[name] for name in documents])
    doc_z = torch.empty(len(names), dtype=torch.long)
    doc_z[inverse] = groups
    shuffled = torch.stack([doc_z[torch.randperm(len(doc_z), generator=generator)]
                             for _ in range(permutations)])
    membership = torch.cat([(groups == 0)[None], shuffled[:, inverse] == 0])
    statistic = mmd_statistics(kernel, membership)
    p = (1 + (statistic[1:] >= statistic[0]).sum()).double() / (permutations + 1)
    return float(statistic[0]), float(p), float(bandwidth)


def lexical_js(words, groups):
    vocab = {word: i for i, word in enumerate(sorted(set(words)))}
    ids = torch.tensor([vocab[word] for word in words])
    counts = torch.bincount(groups * len(vocab) + ids, minlength=2 * len(vocab)).reshape(2, -1).double()
    distributions = counts / counts.sum(-1, keepdim=True)
    midpoint = distributions.mean(0)
    return float((torch.special.xlogy(distributions, distributions) -
                  torch.special.xlogy(distributions, midpoint)).sum() / 2)


def adjust_fdr(rows):
    for seed in sorted({row["seed"] for row in rows}):
        selected = [i for i, row in enumerate(rows) if row["seed"] == seed]
        p = torch.tensor([rows[i]["p_value"] for i in selected], dtype=torch.float64)
        order = p.argsort()
        q = p[order] * len(p) / torch.arange(1, len(p) + 1)
        q = q.flip(0).cummin(0).values.flip(0).clamp_max(1)
        for index, value in zip(order.tolist(), q.tolist()):
            rows[selected[index]]["q_value"] = value


def experiment(cfg, output):
    rows, skips, thresholds = [], [], []
    for name, bank in banks(cfg, nuisance=True):
        test = bank["data"]["test"]
        for seed in cfg.diagnostics.seeds:
            for view_name in VIEWS:
                predictions = load_y_predictions(cfg, name, view_name, seed)
                probability = predictions["test"].softmax(-1)
                train_entropy = entropy(predictions["train"].softmax(-1))
                cuts = torch.quantile(train_entropy, torch.tensor([1 / 3, 2 / 3]))
                predicted_entropy = entropy(probability)
                thresholds.append({"dataset": name, "seed": seed, "view": view_name,
                                   "lower": float(cuts[0]), "upper": float(cuts[1])})
                strata = [("all", "all", torch.ones(len(test["y"]), dtype=torch.bool))]
                strata += [("word_entropy", key, mask) for key, mask in bins(test, "word_entropy").items()]
                strata += [("predictive_entropy", "low", predicted_entropy <= cuts[0]),
                           ("predictive_entropy", "middle", (predicted_entropy > cuts[0]) & (predicted_entropy <= cuts[1])),
                           ("predictive_entropy", "high", predicted_entropy > cuts[1])]
                for label, label_name in enumerate(bank["labels"]):
                    for stratum_number, (kind, stratum, mask) in enumerate(strata):
                        for a, b in combinations(range(len(bank["z_names"])), 2):
                            prefix = {"dataset": name, "seed": seed, "view": view_name, "label": label_name,
                                      "stratum_kind": kind, "stratum": stratum,
                                      "domain_a": bank["z_names"][a], "domain_b": bank["z_names"][b]}
                            generator = torch.Generator().manual_seed(cfg.diagnostics.sample_seed +
                                                                      label * 10000 + stratum_number * 100 + a * 10 + b)
                            chunks = []
                            reason = None
                            for domain in (a, b):
                                ids = ((test["y"] == label) & (test["z"] == domain) & mask).nonzero().flatten()
                                ids = ids[torch.randperm(len(ids), generator=generator)[:cfg.diagnostics.mmd.per_group]]
                                docs = {test["document"][i] for i in ids.tolist()}
                                if len(ids) < cfg.diagnostics.mmd.min_group or len(docs) < cfg.diagnostics.mmd.min_documents:
                                    reason = "insufficient tokens/documents in label/domain/stratum"
                                chunks.append(ids)
                            if reason:
                                skips.append({**prefix, "reason": reason, "n_a": len(chunks[0]), "n_b": len(chunks[1])})
                                continue
                            ids = torch.cat(chunks)
                            groups = torch.cat([torch.zeros(len(chunks[0]), dtype=torch.long),
                                                torch.ones(len(chunks[1]), dtype=torch.long)])
                            documents = [test["document"][i] for i in ids.tolist()]
                            mmd, p, bandwidth = blocked_mmd(probability[ids], groups, documents,
                                                            cfg.diagnostics.mmd.permutations, generator)
                            rows.append({**prefix, "n_a": len(chunks[0]), "n_b": len(chunks[1]),
                                         "documents": len(set(documents)), "mmd_squared": mmd, "p_value": p,
                                         "bandwidth_squared": bandwidth,
                                         "word_js": lexical_js([test["words"][i] for i in ids.tolist()], groups),
                                         "permutation_unit": "sentence (exploratory)" if name == "ner_shared" else "document"})
    adjust_fdr(rows)
    write_csv(output / "scores.csv", rows)
    write_csv(output / "skips.csv", skips)
    write_csv(output / "entropy_thresholds.csv", thresholds)
    keys = ("dataset", "view", "label", "stratum_kind", "stratum", "domain_a", "domain_b")
    write_csv(output / "summary.csv", seed_summary(rows, keys, ("mmd_squared", "word_js", "p_value", "q_value")))
    plot_table(output / "dependence.png", [row for row in rows if row["stratum_kind"] == "all"], "mmd_squared")
    return {"dependence_tests": len(rows), "skipped_cells": len(skips)}


if __name__ == "__main__":
    cli("e5")
