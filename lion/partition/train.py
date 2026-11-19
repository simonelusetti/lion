"""Sentence-batched independent readouts with per-condition validation selection."""
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from tagger.ensemble import IndependentReadouts, clip_heads
from ..diagnostics.common import write_csv
from ..runtime import autocast, parallel


class Sentences(Dataset):
    def __init__(self, data, views=None, sample_draws=False):
        self.data = data
        self.views = views
        self.sample_draws = sample_draws
        self.order = data["sentence"].argsort(stable=True)
        _, counts = data["sentence"][self.order].unique_consecutive(return_counts=True)
        self.offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        a, b = self.offsets[index:index + 2].tolist()
        ids = self.order[a:b]
        if self.views is None:
            features = self.data["features"][ids]
        else:
            gathered = []
            for view in self.views:
                selected = view[ids]
                if view.ndim == 4:
                    # Optional cached draws are stored as [word, condition, draw, feature].
                    if self.sample_draws:
                        index = torch.randint(view.shape[2], selected.shape[:2])
                        selected = selected.gather(2, index[..., None, None].expand(-1, -1, 1, view.shape[-1]))
                    selected = selected[:, :, 0]
                gathered.append(selected)
            features = torch.cat(gathered, dim=1)
        return {"x": features.float(), "y": self.data["y"][ids], "index": ids}


def collate(rows):
    return {key: torch.cat([row[key] for row in rows]) for key in rows[0]}


def loader(data, cfg, train=False, views=None):
    if views is None:
        views = data.get("views")
    return DataLoader(Sentences(data, views, sample_draws=train), collate_fn=collate,
                      batch_size=cfg.train.batch_size if train else cfg.runtime.batch_size,
                      shuffle=train and cfg.train.shuffle, num_workers=cfg.runtime.workers,
                      pin_memory=cfg.runtime.device.startswith("cuda"))


def metric_vectors(logits, target, labels):
    n, conditions, _ = logits.shape
    prediction = logits.argmax(-1)
    offsets = torch.arange(conditions)[None] * labels * labels
    cells = target[:, None] * labels + prediction + offsets
    confusion = torch.bincount(cells.flatten(), minlength=conditions * labels * labels).reshape(conditions, labels, labels).float()
    correct = confusion.diagonal(dim1=-2, dim2=-1)
    wanted, predicted = confusion.sum(-1), confusion.sum(-2)
    return {"accuracy": correct.sum(-1) / n,
            "macro_f1": (2 * correct / (wanted + predicted).clamp_min(1)).mean(-1),
            "balanced_accuracy": (correct / wanted.clamp_min(1)).mean(-1),
            "cross_entropy": -logits.log_softmax(-1).gather(-1, target[:, None, None].expand(-1, conditions, 1)).squeeze(-1).mean(0)}


@torch.no_grad()
def predict(model, data, cfg, views=None):
    model.eval()
    output = None
    for batch in loader(data, cfg, views=views):
        with autocast(cfg):
            logits = model(batch["x"].to(cfg.runtime.device))
        if output is None:
            output = torch.empty(len(data["y"]), logits.shape[1], logits.shape[2])
        output[batch["index"]] = logits.float().cpu()
    return output


def save_fit(output, checkpoint, predictions, history):
    output.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output / "model.pt")
    torch.save(predictions, output / "predictions.pt")
    write_csv(output / "history.csv", history)


def fit(data, labels, cfg, seed, output, linear=False, balanced=False, mixture=False, condition_outputs=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    conditions, dimension = data["train"]["features"].shape[1:]
    if "views" in data["train"]:
        conditions = sum(view.shape[1] for view in data["train"]["views"])
    heads = 1 if mixture else conditions
    tagger = OmegaConf.create({"name": "linear"}) if linear else cfg.tagger
    core = IndependentReadouts(tagger, dimension, labels, heads).to(cfg.runtime.device)
    model = parallel(core, cfg)
    optimizer = torch.optim.AdamW(core.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    weights = None
    if balanced:
        counts = torch.bincount(data["train"]["y"], minlength=labels).float()
        weights = (counts.sum() / (labels * counts)).to(cfg.runtime.device)
    best = torch.full((heads,), -torch.inf)
    best_epoch = torch.zeros(heads, dtype=torch.long)
    history = []
    training = loader(data["train"], cfg, train=True)
    print(f"Fit {output}: {heads} heads, {len(data['train']['y'])} train focals, seed {seed}", flush=True)
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        for batch in training:
            optimizer.zero_grad(set_to_none=True)
            x, y = batch["x"].to(cfg.runtime.device), batch["y"].to(cfg.runtime.device)
            if mixture:
                choice = torch.randint(conditions, (len(x),), device=x.device)
                x = x[torch.arange(len(x), device=x.device), choice][:, None]
            with autocast(cfg):
                logits = model(x).float()
                ce = F.cross_entropy(logits.permute(1, 2, 0), y[None].expand(heads, -1), weight=weights, reduction="none")
                denominator = len(y) if weights is None else weights[y].sum()
                loss = (ce.sum(-1) / denominator).sum()
            loss.backward()
            if cfg.train.clip is not None:
                clip_heads(core, cfg.train.clip)
            optimizer.step()
        validation = metric_vectors(predict(model, data["validation"], cfg), data["validation"]["y"], labels)
        criterion = validation["balanced_accuracy" if balanced else "macro_f1"]
        if mixture:
            criterion = criterion.mean()[None]
        improved = criterion > best
        if improved.any():
            if epoch == 1:
                selected = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
            for key, value in core.state_dict().items():
                selected[key][improved] = value.detach().cpu()[improved]
            best[improved], best_epoch[improved] = criterion[improved], epoch
        for condition in range(conditions):
            history.append({"epoch": epoch, "condition": condition,
                            **{k: float(v[condition]) for k, v in validation.items()}})
    core.load_state_dict(selected)
    saved = {"state": selected, "labels": labels, "dimension": dimension, "heads": heads,
             "linear": linear, "mixture": mixture, "epoch": best_epoch,
             "validation_score": best, "seed": seed}
    predictions = {split: predict(model, part, cfg) for split, part in data.items()}
    if condition_outputs is None:
        save_fit(output, saved, predictions, history)
    else:
        for index, (path, validation_rows) in enumerate(condition_outputs):
            head = slice(index, index + 1)
            checkpoint = {**saved, "heads": 1, "epoch": best_epoch[head].clone(),
                          "validation_score": best[head].clone(),
                          "state": {key: value[head].clone() for key, value in selected.items()}}
            views = {name: (value[:validation_rows] if name == "validation" else value)[:, head].clone()
                     for name, value in predictions.items()}
            records = [{**row, "condition": 0} for row in history if row["condition"] == index]
            save_fit(path, checkpoint, views, records)
    return core.eval(), predictions


def load_model(path, cfg):
    saved = torch.load(Path(path) / "model.pt", map_location="cpu", weights_only=False)
    tagger = OmegaConf.create({"name": "linear"}) if saved["linear"] else cfg.tagger
    model = IndependentReadouts(tagger, saved["dimension"], saved["labels"], saved["heads"])
    model.load_state_dict(saved["state"])
    return model.to(cfg.runtime.device).eval()
