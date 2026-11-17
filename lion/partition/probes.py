"""Held-document-out conditional nuisance probes, sharing condition batches."""
import torch

from ..diagnostics.common import write_csv
from .train import fit


def split_indices(bank, test, allowed):
    fold = bank["fold"]
    count = int(fold.max()) + 1
    return {"train": (allowed & (fold != test) & (fold != (test + 1) % count)).nonzero().flatten(),
            "validation": (allowed & (fold == (test + 1) % count)).nonzero().flatten(),
            "test": (allowed & (fold == test)).nonzero().flatten()}


def conditional(features, bank, cfg, seed, output, standardize=False, mixture=False):
    pool, eligible = bank["pool"], bank["eligible"]
    heads = 1 if mixture else features.shape[1]
    predicted = torch.full((len(pool["y"]), heads), -1, dtype=torch.long)
    models, skipped = {}, []
    for label, label_name in enumerate(bank["labels"]):
        domains = eligible[label].nonzero().flatten()
        if len(domains) < 2:
            skipped.append({"label": label_name, "reason": "fewer than two domains meet support in every document fold"})
            continue
        mapping = torch.full((len(bank["z_names"]),), -1, dtype=torch.long)
        mapping[domains] = torch.arange(len(domains))
        allowed = (pool["y"] == label) & eligible[label, pool["z"]]
        for test in range(cfg.partition.folds):
            ids = split_indices(bank, test, allowed)
            mean, scale = torch.zeros(1, features.shape[1], features.shape[2]), torch.ones(1, features.shape[1], features.shape[2])
            if standardize:
                mean = features[ids["train"]].float().mean(0, keepdim=True)
                scale = features[ids["train"]].float().std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
            local = {split: {"features": (features[index].float() - mean) / scale,
                             "y": mapping[pool["z"][index]], "sentence": pool["sentence"][index]}
                     for split, index in ids.items()}
            path = output / f"label{label}" / f"fold{test}"
            _, predictions = fit(local, len(domains), cfg, seed, path, linear=True, balanced=True, mixture=mixture)
            probability = predictions["test"].softmax(-1)
            if mixture: probability = probability.mean(1, keepdim=True)
            predicted[ids["test"]] = domains[probability.argmax(-1)]
            models[(label, test)] = {"path": str(path), "domains": domains, "mean": mean, "scale": scale}
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"predicted": predicted, "models": models, "eligible": eligible}, output / "oof.pt")
    write_csv(output / "skips.csv", skipped)
    return predicted, models


def global_probe(features, bank, cfg, seed, output):
    pool = bank["pool"]
    predictions = torch.empty(len(pool["y"]), features.shape[1], len(bank["z_names"]))
    for test in range(cfg.partition.folds):
        ids = split_indices(bank, test, torch.ones(len(pool["y"]), dtype=torch.bool))
        local = {split: {"features": features[index].float(), "y": pool["z"][index], "sentence": pool["sentence"][index]}
                 for split, index in ids.items()}
        _, predicted = fit(local, len(bank["z_names"]), cfg, seed, output / f"fold{test}", linear=True, balanced=True)
        predictions[ids["test"]] = predicted["test"]
    return predictions
