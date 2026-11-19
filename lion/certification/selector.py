"""HardKuma context gates trained through the frozen scorer."""
import math
import time
from itertools import islice

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..diagnostics.common import save_json, write_csv
from ..encoder import load_masked_encoder as masked_encoder
from ..partition.train import fit, load_model
from ..runtime import autocast
from .data import context, full_features, load, loader, paths, readout_data, task_names


class Selector(nn.Module):
    def __init__(self, dimension, widths):
        super().__init__()
        sizes = [2 * dimension, *widths, 2]
        layers = []
        for i, (source, target) in enumerate(zip(sizes, sizes[1:])):
            layers.append(nn.Linear(source, target))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.norm = nn.LayerNorm(dimension)
        self.scorer = nn.Sequential(*layers)

    def concentrations(self, hidden, focal, owner=None):
        hidden = self.norm(hidden.float())
        rows = torch.arange(len(hidden), device=hidden.device) if owner is None else owner
        first = self.scorer[0]
        focal_weight, candidate_weight = first.weight.chunk(2, dim=1)
        candidate = F.linear(hidden, candidate_weight, first.bias)
        if owner is not None:
            candidate = candidate[owner]
        projection = candidate + F.linear(hidden[rows, focal], focal_weight)[:, None]
        for layer in islice(self.scorer, 1, None):
            projection = layer(projection)
        return (F.softplus(projection) + 0.2).unbind(-1)

    @staticmethod
    def hard_gate(a, b, u):
        kuma = (-torch.expm1(torch.log1p(-u) / b)).pow(a.reciprocal())
        return (1.2 * kuma - 0.1).clamp(0, 1)

    def forward(self, hidden, focal, draw=False, owner=None):
        a, b = self.concentrations(hidden, focal, owner)
        u = torch.rand_like(a).clamp(1e-6, 1 - 1e-6) if draw else torch.full_like(a, 0.5)
        nonzero = (-torch.expm1(a * math.log(0.1 / 1.2))).pow(b)
        return self.hard_gate(a, b, u), nonzero

    def mask(self, hidden, focal, eligible, owner=None):
        a, b = self.concentrations(hidden, focal, owner)
        log_survival_at_zero = b * torch.log(-torch.expm1(a * math.log(0.1 / 1.2)))
        p_zero = -torch.expm1(log_survival_at_zero)
        p_one = (-torch.expm1(a * math.log(1.1 / 1.2))).pow(b)
        return ~eligible | (p_one >= p_zero)


def load_selector(cfg, task, fold):
    saved = torch.load(paths(cfg) / "selector" / task / fold / "model.pt",
                       map_location="cpu", weights_only=False)
    model = Selector(saved["dimension"], cfg.certification.hidden)
    model.load_state_dict(saved["state"])
    return model.to(cfg.runtime.device).eval()


def chunk_size(ids, cfg, max_rows=None):
    limit = cfg.runtime.focal_chunk if max_rows is None else max_rows
    return max(1, min(limit, cfg.runtime.attention_budget // ids.shape[1] ** 2))


def focal_batches(encoder, batches, cfg):
    device = cfg.runtime.device
    for batch in batches:
        ids, attention, focal, owners = (batch[key].to(device, non_blocking=True)
                                        for key in ("ids", "attention", "focal", "owner"))
        gold = {task: value.to(device, non_blocking=True) for task, value in batch["gold"].items()}
        fill = batch.get("fill")
        if fill is not None:
            fill = fill.to(device, non_blocking=True)
        with torch.no_grad(), autocast(cfg):
            hidden = encoder(ids, attention)["features"]
        lengths = batch["lengths"][batch["owner"]]
        sentence_owners = batch["owner"].tolist()
        chunk = chunk_size(ids, cfg)
        padded = F.pad(lengths, (0, -len(focal) % chunk))
        widths = padded.view(-1, chunk).amax(dim=1).tolist()
        for start, width in zip(range(0, len(focal), chunk), widths):
            stop = min(start + chunk, len(focal))
            part = slice(start, stop)
            owner = owners[part]
            first, last = sentence_owners[start], sentence_owners[stop - 1]
            yield (ids[owner, :width], attention[owner, :width], hidden[first:last + 1, :width],
                   focal[part], owner - first, batch["sentence"][part], {task: y[part] for task, y in gold.items()},
                   None if fill is None else fill[owner, :width].transpose(1, 2))


def replacement_positions(replace):
    indices = torch.arange(replace.shape[1], device=replace.device).expand_as(replace)
    width = int(replace.sum(-1).max())
    return indices.masked_fill(~replace, replace.shape[1]).sort(dim=-1).values[:, :width]


@torch.no_grad()
def mlm_probabilities(encoder, ids, attention, positions, special, cfg):
    with autocast(cfg):
        logits = encoder(ids, attention, positions, mlm=True)["logits"].float()
    logits[..., special["excluded"]] = -torch.inf
    return logits.softmax(-1)


@torch.no_grad()
def precompute_fills(encoder, rows, special, cfg):
    device = cfg.runtime.device
    draws = cfg.certification.training_draws
    rng = torch.Generator(device=device).manual_seed(cfg.seed)
    fills = []
    for batch in loader(rows, cfg):
        ids, attention = (batch[key].to(device, non_blocking=True) for key in ("ids", "attention"))
        columns = torch.arange(ids.shape[1], device=device)[None]
        eligible = attention & (columns > 0) & (columns < attention.sum(-1, keepdim=True) - 1)
        fill = ids[..., None].expand(-1, -1, draws).clone()
        owners, positions = eligible.nonzero(as_tuple=True)
        size = chunk_size(ids, cfg)
        for owner, position in zip(owners.split(size), positions.split(size)):
            if not len(owner):
                continue
            masked = ids[owner].clone()
            masked[torch.arange(len(owner), device=device), position] = special["mask"]
            probabilities = mlm_probabilities(encoder, masked, attention[owner], position, special, cfg)
            fill[owner, position] = torch.multinomial(probabilities, draws, replacement=True, generator=rng)
        fills.extend(values[:length].clone() for values, length in zip(fill.cpu(), batch["lengths"].tolist()))
    return [{**row, "fill": values} for row, values in zip(rows, fills)]


def score_draws(encoder, scorer, ids, attention, focal, fill, gate, cfg):
    def log_probabilities(ids, attention, focal, fill, gate):
        with autocast(cfg):
            features = encoder(ids, attention, focal, replacement_ids=fill, gates=gate)["features"]
            logits = scorer(features.float()[:, None]).squeeze(1).float()
        return logits.log_softmax(-1)

    n, draws, _ = fill.shape
    owners = torch.arange(n, device=fill.device).repeat_interleave(draws)
    score_batch = chunk_size(ids, cfg, cfg.runtime.selector_score_batch_size)
    scores = []
    for owner, replacements in zip(owners.split(score_batch), fill.flatten(0, 1).split(score_batch)):
        inputs = (ids[owner], attention[owner], focal[owner], replacements, gate[owner])
        scores.append(checkpoint(log_probabilities, *inputs, use_reentrant=False)
                      if cfg.runtime.activation_checkpointing else log_probabilities(*inputs))
    return torch.cat(scores).view(n, draws, -1)


def scorer_stage(cfg, output):
    bank = load(cfg)
    encoder = masked_encoder(cfg)
    full = full_features(encoder, {name: bank["sets"][name] for name in ("train", "validation")}, cfg)
    torch.save(full, output / "full_features.pt")
    train = bank["sets"]["train"]
    for task in task_names(cfg):
        data = {}
        for split in ("train", "validation"):
            selected = [row["fold"] == f"scorer_{split}" for row in train]
            mask = torch.repeat_interleave(torch.tensor(selected), torch.tensor([len(row["focals"]) for row in train]))
            rows = [row for row, keep in zip(train, selected) if keep]
            data[split] = readout_data(rows, full["train"][mask], task)
        fit(data, len(bank["labels"][task]), cfg, cfg.seed, output / task)
    return {"scorers": len(task_names(cfg))}


def sample_training_focals(rows, task, cap, seed):
    targets = torch.cat([row["gold"][task] for row in rows])
    eligible = (targets >= 0).nonzero().flatten()
    if len(eligible) <= cap:
        return rows
    order = torch.randperm(len(eligible), generator=torch.Generator().manual_seed(seed))
    selected = eligible[order[:cap]].sort().values
    lengths = torch.tensor([len(row["focals"]) for row in rows])
    ends = lengths.cumsum(0)
    owners = torch.searchsorted(ends, selected, right=True)
    sentence, counts = owners.unique_consecutive(return_counts=True)
    local = selected - (ends - lengths)[owners]
    sampled = []
    for index, keep in zip(sentence.tolist(), local.split(counts.tolist())):
        row = rows[index]
        sampled.append({**row, "focals": row["focals"][keep],
                        "gold": {name: values[keep] for name, values in row["gold"].items()}})
    return sampled


def experiment(cfg, output, started=None):
    bank = load(cfg)
    encoder = masked_encoder(cfg)
    draws = cfg.certification.training_draws
    reference = cfg.certification.timing_reference_seconds
    cap = cfg.certification.get("selector_train_focals_per_fold")
    if cap is not None and cap <= 0:
        raise ValueError("certification.selector_train_focals_per_fold must be positive.")
    timed = reference is not None
    tasks = task_names(cfg)
    folds = ("g1", "g2") if cfg.stage.fold is None else (cfg.stage.fold,)
    training_rows = {fold: [row for row in bank["sets"]["train"] if row["fold"] == fold] for fold in folds}
    remaining = 0
    if timed:
        for task in tasks:
            for rows in training_rows.values():
                candidates = int(torch.cat([row["gold"][task] for row in rows]).ge(0).sum())
                remaining += candidates if cap is None else min(candidates, cap)
        remaining *= cfg.certification.selector_epochs
    for task in tasks:
        scorer = load_model(paths(cfg) / "scorer" / task, cfg).requires_grad_(False)
        dimension = scorer.weight.shape[1]
        for fold in folds:
            torch.manual_seed(cfg.seed)
            selector = Selector(dimension, cfg.certification.hidden).to(cfg.runtime.device)
            optimizer = torch.optim.AdamW(selector.parameters(), lr=cfg.certification.selector_lr,
                                          weight_decay=cfg.train.weight_decay)
            rows = training_rows[fold]
            history = []
            folder = output / task / fold
            folder.mkdir(parents=True, exist_ok=True)
            if cap is not None:
                population = torch.cat([row["gold"][task] for row in rows])
                rows = sample_training_focals(rows, task, cap, cfg.seed)
                sampled = torch.cat([row["gold"][task] for row in rows])
                labels = bank["labels"][task]
                original_counts = torch.bincount(population[population >= 0], minlength=len(labels)).tolist()
                sampled_counts = torch.bincount(sampled[sampled >= 0], minlength=len(labels)).tolist()
                write_csv(folder / "training_sample.csv", [
                    {"label": label, "available_focals": available, "sampled_focals": count}
                    for label, available, count in zip(labels, original_counts, sampled_counts)
                ])
            print(f"Precomputing four single-position fill-ins for {task}/{fold} on {len(rows)} sentences", flush=True)
            rows = precompute_fills(encoder, rows, bank["special"], cfg)
            batches = loader(rows, cfg, train=True)
            print(f"Train selector {task}/{fold} on {len(rows)} sentences", flush=True)
            fitted_chunks = 0
            measured_words = 0
            if timed and cfg.runtime.device.startswith("cuda"):
                torch.cuda.synchronize()
            sampling_start = time.monotonic()
            for epoch in range(1, cfg.certification.selector_epochs + 1):
                totals = torch.zeros(4, device=cfg.runtime.device)
                for ids, attention, hidden, focal, owner, _, gold, fill in focal_batches(encoder, batches, cfg):
                    labeled = (gold[task] >= 0).nonzero().flatten()
                    if not len(labeled):
                        continue
                    ids, attention, focal, owner = ids[labeled], attention[labeled], focal[labeled], owner[labeled]
                    y = gold[task][labeled]
                    eligible = context(attention, focal)
                    gate, mass = selector(hidden, focal, draw=True, owner=owner)
                    gate = torch.where(eligible, gate, 1.0)
                    log_probability = score_draws(encoder, scorer, ids, attention, focal, fill[labeled], gate, cfg)
                    observed = log_probability.gather(-1, y[:, None, None].expand(-1, draws, 1)).squeeze(-1)
                    ce = (math.log(draws) - observed.logsumexp(-1)).mean()
                    size = (mass * eligible).sum(-1).mean()
                    loss = ce + cfg.certification.size_penalty * size
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if cfg.train.clip is not None:
                        torch.nn.utils.clip_grad_norm_(selector.parameters(), cfg.train.clip)
                    optimizer.step()
                    totals[:3] += torch.stack((loss.detach(), ce.detach(), size.detach())) * len(y)
                    totals[3] += len(y)
                    if timed:
                        remaining -= len(y)
                        fitted_chunks += 1
                        measured_words += len(y)
                        if fitted_chunks == cfg.certification.timing_chunks:
                            if cfg.runtime.device.startswith("cuda"):
                                torch.cuda.synchronize()
                            rate = (time.monotonic() - sampling_start) / measured_words
                            elapsed = time.monotonic() - started
                            estimate = elapsed + remaining * rate
                            timing = {"projected_through_selector_seconds": estimate, "elapsed_seconds": elapsed,
                                      "projected_folds": list(folds),
                                      "seconds_per_training_word": rate, "timed_words": measured_words,
                                      "reference_seconds": reference, "above_reference": estimate >= reference}
                            save_json(folder / "timing.json", timing)
                            print(f"{task}/{fold}: estimated {estimate / 3600:.2f}h through selector training "
                                  f"(reference {reference / 3600:g}h)", flush=True)
                loss, ce, size = (totals[:3] / totals[3]).tolist()
                history.append({"epoch": epoch, "loss": loss, "cross_entropy": ce, "expected_size": size,
                                "words": int(totals[3])})
                print(f"{task}/{fold} epoch {epoch}: CE {ce:.4f}, expected context {size:.2f}", flush=True)
            torch.save({"state": selector.state_dict(), "dimension": dimension}, folder / "model.pt")
            write_csv(folder / "history.csv", history)
    return {"selectors": len(folds) * len(tasks)}
