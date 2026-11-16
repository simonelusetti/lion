"""One training loop for sentence readouts and focal context interventions."""
import json
import logging
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from .cache import cached_loader, ensure_cache
from .data import loader, to_device
from .labels import LABELS
from .metrics import scores, transfer_scores
from .runtime import autocast, device_ids, parallel

log = logging.getLogger(__name__)


def make_loaders(cfg, data, tokenizer, encoder):
    cached = cfg.cache.enabled and cfg.method.kind == "layers"
    if cached:
        root = ensure_cache(cfg, data, tokenizer, encoder)
        return {name: cached_loader(root, name, cfg, train=name == "train") for name in data}
    return {name: loader(dataset, cfg, tokenizer, train=name == "train") for name, dataset in data.items()}


def load_parts(model, cfg):
    for part, source in (("tagger", cfg.train.classifier_source), ("method", cfg.train.selector_source)):
        if source is not None:
            saved = torch.load(source, map_location=cfg.runtime.device, weights_only=False)
            state = saved.get(part, saved.get("selector", saved))
            getattr(model, part).load_state_dict(state, strict=part == "tagger")
    for part in cfg.train.freeze:
        module = getattr(model, part)
        module.requires_grad_(False)
        for child in module.modules():
            if isinstance(child, nn.Dropout):
                child.p = 0


def teacher_batch(batch, records):
    """Sparse mask records are expanded on CPU at the data boundary."""
    valid = batch["word_mask"]
    width = valid.shape[1]
    teachers = valid[:, None, :].expand(-1, width, -1).float().clone()
    for row, sentence_id in enumerate(batch["sentence_ids"].tolist()):
        for focal, context in records.get(sentence_id, {}).items():
            teachers[row, focal] = 0
            teachers[row, focal, [focal, *context]] = 1
    return teachers


def batches(core, raw, cfg, records=None):
    batch = to_device(raw, cfg.runtime.device)
    if not core.masked:
        valid = batch["word_mask"] & (batch["labels"] >= 0)
        inputs = ({key: batch[key] for key in ("features", "heads") if key in batch}
                  if "features" in batch else
                  {key: batch[key] for key in ("ids", "attention_mask", "positions")})
        yield inputs, batch["labels"][valid], valid, None, batch
        return
    with autocast(cfg):
        words = core.prepare(batch).float()
    sentence, focal = (batch["word_mask"] & (batch["labels"] >= 0)).nonzero(as_tuple=True)
    teacher = teacher_batch(raw, records).to(cfg.runtime.device) if records is not None else None
    chunk = min(cfg.runtime.focal_chunk, max(1, cfg.runtime.attention_budget // batch["ids"].shape[1] ** 2))
    for start in range(0, len(sentence), chunk):
        source, target = sentence[start:start + chunk], focal[start:start + chunk]
        inputs = {key: batch[key][source] for key in ("ids", "attention_mask", "positions", "word_ids")}
        inputs.update(words=words[source], valid=batch["word_mask"][source], focal=target)
        if teacher is not None:
            inputs["teacher"] = teacher[source, target]
        context = inputs["valid"].clone()
        context.scatter_(1, target[:, None], False)
        yield inputs, batch["labels"][source, target], None, context, batch


def objective(output, targets, valid, context, inputs, cfg):
    logits = output["logits"] if valid is None else output["logits"][valid]
    ce = F.cross_entropy(logits.float(), targets, reduction="sum")
    auxiliary = ce.new_zeros(())
    if context is not None:
        gates = output["gates"].float()
        auxiliary = cfg.method.penalty * ((gates * context).sum(-1) / context.sum(-1).clamp_min(1)).sum()
        if cfg.method.name == "supervised_mask":
            teacher = inputs["teacher"]
            element = F.binary_cross_entropy(gates.clamp(1e-6, 1 - 1e-6), teacher, reduction="none")
            positive, negative = context & teacher.bool(), context & ~teacher.bool()
            pos = (element * positive).sum(-1) / positive.sum(-1).clamp_min(1)
            neg = (element * negative).sum(-1) / negative.sum(-1).clamp_min(1)
            categories = positive.any(-1).long() + negative.any(-1).long()
            auxiliary = auxiliary + cfg.method.supervision * ((pos + neg) / categories.clamp_min(1)).sum()
    return logits, ce, auxiliary


def train_epoch(model, core, dataloader, optimizer, cfg, records):
    model.train()
    totals = torch.zeros(3, device=cfg.runtime.device)
    for raw in tqdm(dataloader, desc="Train", leave=False):
        optimizer.zero_grad(set_to_none=True)
        count = ((raw["labels"] >= 0) & raw["word_mask"]).sum().item()
        for inputs, targets, valid, context, _ in batches(core, raw, cfg, records):
            with autocast(cfg):
                output = model(**inputs)
                _, ce, auxiliary = objective(output, targets, valid, context, inputs, cfg)
                loss = (ce + auxiliary) / count
            loss.backward()
            totals += torch.stack([ce.detach(), auxiliary.detach(), ce.new_tensor(len(targets))])
        if cfg.train.clip is not None:
            nn.utils.clip_grad_norm_(core.parameters(), cfg.train.clip)
        optimizer.step()
    return {"train_ce": float(totals[0] / totals[2]), "train_aux": float(totals[1] / totals[2])}


@torch.no_grad()
def evaluate(model, core, dataloader, cfg, labels, records=None, shared=False):
    model.eval()
    logits, targets, counts, sentences, positions, gates = [], [], [], [], [], []
    # A random-mask evaluation reuses its masks without consuming the training RNG.
    with torch.random.fork_rng(devices=device_ids(cfg)):
        torch.manual_seed(cfg.seed + 1)
        for raw in tqdm(dataloader, desc="Evaluate", leave=False):
            valid_raw = raw["word_mask"] & (raw["labels"] >= 0)
            count = valid_raw.sum(-1)
            counts.extend(count.tolist())
            sentences.append(raw["sentence_ids"].repeat_interleave(count))
            positions.append(valid_raw.nonzero(as_tuple=True)[1])
            for inputs, wanted, valid, context, _ in batches(core, raw, cfg, records):
                with autocast(cfg):
                    output = model(**inputs)
                prediction = output["logits"] if valid is None else output["logits"][valid]
                logits.append(prediction.float().cpu())
                targets.append(wanted.cpu())
                if context is not None:
                    gates.append(torch.stack([(output["gates"] * context).sum(), context.sum()]).cpu())
                elif cfg.method.name not in {"last", "layer"}:
                    gates.append(output["gates"][valid].float().sum(0).cpu())
    logits, targets = torch.cat(logits), torch.cat(targets)
    result = (transfer_scores(logits, targets, counts, labels, LABELS[cfg.data.name]) if shared else
              scores(logits, targets, counts, labels, cfg.data.name))
    if gates:
        total = torch.stack(gates).sum(0)
        result["context_mass" if core.masked else "weights"] = (
            float(total[0] / total[1].clamp_min(1)) if core.masked else (total / len(targets)).tolist())
    predictions = {"logits": logits, "targets": targets, "counts": counts,
                   "sentence_ids": torch.cat(sentences), "positions": torch.cat(positions),
                   "labels": labels, "target_labels": LABELS[cfg.data.name], "dataset": cfg.data.name}
    return result, predictions


def save_model(core, cfg, labels, path):
    torch.save({"method": core.method.state_dict(), "tagger": core.tagger.state_dict(),
                "labels": labels, "config": OmegaConf.to_container(cfg, resolve=True)}, path)


def fit(core, dataloaders, cfg, run, output, labels, records=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    load_parts(core, cfg)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in core.tagger.parameters() if p.requires_grad], "weight_decay": cfg.train.weight_decay},
        {"params": [p for p in core.method.parameters() if p.requires_grad], "weight_decay": cfg.train.method_decay},
    ], lr=cfg.train.lr)
    model = parallel(core, cfg)
    history, best = [], -float("inf")
    for epoch in range(1, cfg.train.epochs + 1):
        train_records = records.get("train") if records is not None else None
        metrics = train_epoch(model, core, dataloaders["train"], optimizer, cfg, train_records)
        validation_records = records.get("validation") if cfg.method.name == "oracle_mask" else None
        validation, _ = evaluate(model, core, dataloaders["validation"], cfg, labels, validation_records)
        metrics.update(epoch=epoch, val_macro_f1=validation["macro_f1"], val_accuracy=validation["accuracy"])
        if cfg.method.name == "layer":
            metrics["layer"] = cfg.method.layer
        history.append(metrics)
        run.push_log(metrics, step=epoch)
        log.info("epoch %d train CE %.4f val F1 %.4f", epoch, metrics["train_ce"], metrics["val_macro_f1"])
        save_model(core, cfg, labels, output / "last.pt")
        if validation["macro_f1"] > best:
            best = validation["macro_f1"]
            save_model(core, cfg, labels, output / "model.pt")
    # Final reports use the checkpoint selected on validation, never on test.
    saved = torch.load(output / "model.pt", map_location=cfg.runtime.device, weights_only=False)
    core.method.load_state_dict(saved["method"])
    core.tagger.load_state_dict(saved["tagger"])
    test_records = records.get("test") if cfg.method.name == "oracle_mask" else None
    result, prediction = evaluate(model, core, dataloaders["test"], cfg, labels, test_records)
    torch.save(prediction, output / "predictions.pt")
    (output / "report.json").write_text(json.dumps({"history": history, "test": result}, indent=2) + "\n")
    return result
