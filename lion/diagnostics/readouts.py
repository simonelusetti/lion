"""Baseline MLP training on cached focal vectors, batching complete sampled sentences."""
import json

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from tagger import build_tagger
from ..runtime import autocast, parallel
from .common import save_json, subset, write_csv


class FeatureSentences(Dataset):
    def __init__(self, data, view):
        self.data, self.view = data, view
        self.order = data["sentence"].argsort(stable=True)
        _, counts = data["sentence"][self.order].unique_consecutive(return_counts=True)
        self.offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        start, end = self.offsets[index:index + 2].tolist()
        indices = self.order[start:end]
        return {"x": self.data["features"][indices, self.view].float(),
                "y": self.data["y"][indices], "index": indices}


def collate_features(rows):
    return {key: torch.cat([row[key] for row in rows]) for key in rows[0]}


def feature_loader(data, view, cfg, train=False):
    return DataLoader(FeatureSentences(data, view), collate_fn=collate_features,
                      batch_size=cfg.train.batch_size if train else cfg.runtime.batch_size,
                      shuffle=train and cfg.train.shuffle, num_workers=cfg.runtime.workers,
                      pin_memory=cfg.runtime.device.startswith("cuda"),
                      persistent_workers=cfg.runtime.workers > 0)


def measurements(logits, targets, labels):
    n = len(labels)
    prediction = logits.argmax(-1)
    counts = torch.bincount(targets * n + prediction, minlength=n * n).reshape(n, n).float()
    gold, predicted, correct = counts.sum(-1), counts.sum(0), counts.diag()
    precision, recall = correct / predicted.clamp_min(1), correct / gold.clamp_min(1)
    f1 = 2 * correct / (gold + predicted).clamp_min(1)
    overall = {"n": len(targets), "accuracy": float(correct.sum() / len(targets)),
               "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall.mean()),
               "cross_entropy": float(F.cross_entropy(logits.float(), targets))}
    per_label = [{"label": label, "support": int(gold[i]), "predicted": int(predicted[i]),
                  "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
                 for i, label in enumerate(labels)]
    return overall, per_label


@torch.no_grad()
def predict(model, dataloader, n, classes, cfg):
    model.eval()
    output = torch.empty(n, classes)
    for batch in dataloader:
        with autocast(cfg):
            logits = model(batch["x"].to(cfg.runtime.device))
        output[batch["index"]] = logits.float().cpu()
    return output


def fit(data, labels, view, cfg, seed, output, linear=False, balanced=False):
    output.mkdir(parents=True, exist_ok=True)
    print(f"Readout {output}: {len(data['train']['y'])} train focal tokens, seed {seed}", flush=True)
    torch.manual_seed(seed)
    tagger_cfg = {"name": "linear"} if linear else cfg.tagger
    # build_tagger expects attribute-style config.
    if linear:
        from omegaconf import OmegaConf
        tagger_cfg = OmegaConf.create(tagger_cfg)
    core = build_tagger(tagger_cfg, data["train"]["features"].shape[-1], len(labels)).to(cfg.runtime.device)
    model = parallel(core, cfg)
    loaders = {split: feature_loader(rows, view, cfg, train=split == "train")
               for split, rows in data.items()}
    optimizer = torch.optim.AdamW(core.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    weights = None
    if balanced:
        counts = torch.bincount(data["train"]["y"], minlength=len(labels)).float()
        weights = (counts.sum() / (len(labels) * counts)).to(cfg.runtime.device)
    history, best = [], -float("inf")
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        total, count = 0., 0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            with autocast(cfg):
                logits = model(batch["x"].to(cfg.runtime.device))
                loss = F.cross_entropy(logits.float(), batch["y"].to(cfg.runtime.device), weight=weights)
            loss.backward()
            if cfg.train.clip is not None:
                nn.utils.clip_grad_norm_(core.parameters(), cfg.train.clip)
            optimizer.step()
            total += float(loss.detach()) * len(batch["y"])
            count += len(batch["y"])
        logits = predict(model, loaders["validation"], len(data["validation"]["y"]), len(labels), cfg)
        validation, _ = measurements(logits, data["validation"]["y"], labels)
        criterion = validation["balanced_accuracy" if balanced else "macro_f1"]
        history.append({"epoch": epoch, "train_ce": total / count, **validation})
        if criterion > best:
            best = criterion
            torch.save({"tagger": core.state_dict(), "labels": labels, "seed": seed,
                        "dimension": data["train"]["features"].shape[-1],
                        "linear": linear, "view": view, "epoch": epoch}, output / "model.pt")
    saved = torch.load(output / "model.pt", map_location=cfg.runtime.device, weights_only=False)
    core.load_state_dict(saved["tagger"])
    predictions = {split: predict(model, loaders[split], len(rows["y"]), len(labels), cfg)
                   for split, rows in data.items()}
    torch.save(predictions, output / "predictions.pt")
    write_csv(output / "history.csv", history)
    save_json(output / "selection.json", {"epoch": saved["epoch"], "validation_score": best,
                                           "criterion": "balanced_accuracy" if balanced else "macro_f1"})
    return predictions


def eligible_probe(bank, label, cfg, view):
    labels, domains = bank["labels"], bank["z_names"]
    support = {split: torch.bincount(rows["z"][rows["y"] == label], minlength=len(domains))
               for split, rows in bank["data"].items()}
    keep = torch.ones(len(domains), dtype=torch.bool)
    for split, counts in support.items():
        keep &= counts >= cfg.diagnostics.probe_min[split]
    selected = keep.nonzero().flatten()
    if len(selected) < 2:
        return None, [], {split: value.tolist() for split, value in support.items()}
    mapping = torch.full((len(domains),), -1, dtype=torch.long)
    mapping[selected] = torch.arange(len(selected))
    data = {}
    for split, rows in bank["data"].items():
        indices = ((rows["y"] == label) & keep[rows["z"]]).nonzero().flatten()
        data[split] = subset(rows, indices)
        data[split]["y"] = mapping[data[split]["z"]]
    return data, [domains[i] for i in selected], {split: value.tolist() for split, value in support.items()}


def probe_all(name, bank, cfg, output, views=(0, 1, 2)):
    rows, skipped = [], []
    for label, label_name in enumerate(bank["labels"]):
        for view in views:
            data, domains, support = eligible_probe(bank, label, cfg, view)
            if data is None:
                skipped.append({"dataset": name, "label": label_name, "view": view,
                                "reason": "fewer than two domains meet train/validation/test support",
                                "support": json.dumps(support)})
                continue
            for seed in cfg.diagnostics.seeds:
                predictions = fit(data, domains, view, cfg, seed,
                                  output / name / str(view) / str(label) / f"seed{seed}", linear=True, balanced=True)
                result, _ = measurements(predictions["test"], data["test"]["y"], domains)
                rows.append({"dataset": name, "label": label_name, "view": view, "seed": seed,
                             "domains": "|".join(domains), "chance": 1 / len(domains),
                             **result, "above_chance": result["balanced_accuracy"] - 1 / len(domains)})
    return rows, skipped


def load_y_predictions(cfg, name, view, seed):
    from .common import locations
    results, _ = locations(cfg)
    return torch.load(results / "e1" / name / view / f"seed{seed}" / "predictions.pt", weights_only=True)


def load_y_model(cfg, name, view, seed):
    from .common import locations
    results, _ = locations(cfg)
    saved = torch.load(results / "e1" / name / view / f"seed{seed}" / "model.pt", weights_only=False)
    model = build_tagger(cfg.tagger, saved["dimension"], len(saved["labels"])).to(cfg.runtime.device)
    model.load_state_dict(saved["tagger"])
    model.eval()
    return model
