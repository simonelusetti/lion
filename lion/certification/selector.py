"""Rank context words through the frozen scorer and masked language model."""
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
from .data import (attach_features, context, focal_vectors, full_features, load, loader,
                   paths, readout_data, task_names, word_to_pieces)


class Selector(nn.Module):
    def __init__(self, dimension, widths):
        super().__init__()
        sizes = [2 * dimension, *widths, 1]
        layers = []
        for i, (source, target) in enumerate(zip(sizes, sizes[1:])):
            layers.append(nn.Linear(source, target))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.norm = nn.LayerNorm(dimension)
        self.scorer = nn.Sequential(*layers)

    def forward(self, hidden, focal, owner=None):
        hidden = self.norm(hidden.float())
        owners = torch.arange(len(hidden), device=hidden.device) if owner is None else owner
        first = self.scorer[0]
        focal_weight, context_weight = first.weight.chunk(2, dim=1)
        candidate = F.linear(hidden, context_weight, first.bias)
        if owner is not None:
            candidate = candidate[owner]
        projection = candidate + F.linear(hidden[owners, focal], focal_weight)[:, None]
        for layer in islice(self.scorer, 1, None):
            projection = layer(projection)
        return projection.squeeze(-1)


def ranks(scores, eligible):
    order = scores.masked_fill(~eligible, -torch.inf).argsort(dim=-1, descending=True, stable=True)
    return order.argsort(dim=-1)


def word_budget(fraction, available):
    return (fraction * available).round().long()


def kept_words(rank, eligible, budgets):
    return ~eligible | (rank < budgets[:, None])


def soft_keep(scores, eligible, budgets, cfg):
    strength = scores.sigmoid()
    differences = strength[:, :, None] - strength[:, None, :]
    wins = (differences / cfg.certification.rank_temperature).sigmoid()
    soft_rank = 0.5 + (wins * eligible[:, :, None]).sum(dim=1)
    soft = ((budgets[:, None] + 0.5 - soft_rank) / cfg.certification.gate_temperature).sigmoid()
    # The half added to soft_rank removes the included 0.5 self comparison.
    return torch.where(eligible, soft, 1.0)


def load_selector(cfg, task, fold):
    saved = torch.load(paths(cfg) / "selector" / task / fold / "model.pt",
                       map_location="cpu", weights_only=False)
    model = Selector(saved["dimension"], cfg.certification.hidden)
    model.load_state_dict(saved["state"])
    return model.to(cfg.runtime.device).eval()


def raw_rows(cfg, bank):
    full = torch.load(paths(cfg) / "scorer" / "full_features.pt", weights_only=False, mmap=True)
    return attach_features(bank["sets"]["train"], full["train"])


def chunk_size(ids, cfg, max_rows=None):
    limit = cfg.runtime.focal_chunk if max_rows is None else max_rows
    return max(1, min(limit, cfg.runtime.attention_budget // ids.shape[1] ** 2))


def focal_batches(batches, task, cfg):
    device = cfg.runtime.device
    for batch in batches:
        labeled = batch["gold"][task] >= 0
        focals = batch["focal"][labeled].to(device, non_blocking=True)
        owners = batch["owner"][labeled]
        lengths = batch["lengths"][owners]
        word_lengths = batch["word_counts"][owners]
        sentences = batch["sentence"][labeled]
        groups = batch["group"][labeled].to(device, non_blocking=True) if "group" in batch else None
        gold = batch["gold"][task][labeled].to(device, non_blocking=True)
        ids, attention, word_ids, hidden, counts = (batch[key].to(device, non_blocking=True)
            for key in ("ids", "attention", "word_ids", "hidden", "word_counts"))
        chunk = chunk_size(ids, cfg)
        pad = -len(focals) % chunk
        widths = F.pad(lengths, (0, pad)).view(-1, chunk).amax(dim=1).tolist()
        word_widths = F.pad(word_lengths, (0, pad)).view(-1, chunk).amax(dim=1).tolist()
        source = owners.tolist()
        owners = owners.to(device, non_blocking=True)
        for start, width, word_width in zip(range(0, len(focals), chunk), widths, word_widths):
            stop = min(start + chunk, len(focals))
            part = slice(start, stop)
            first, last = source[start], source[stop - 1]
            owner = owners[part]
            yield {"ids": ids[owner, :width], "attention": attention[owner, :width],
                   "word_ids": word_ids[owner, :width], "hidden": hidden[first:last + 1, :word_width],
                   "focal": focals[part], "owner": owner - first, "counts": counts[owner],
                   "sentence": sentences[part], "gold": gold[part],
                   "group": None if groups is None else groups[part]}


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
def independent_fills(encoder, ids, attention, drop, draws, special, cfg, rng):
    fill = ids[:, None].expand(-1, draws, -1).transpose(1, 2).clone()
    active = drop.any(-1).nonzero().flatten()
    # Bound the number of vocabulary distributions kept in memory at once.
    batch_rows = min(chunk_size(ids, cfg), max(1, cfg.runtime.mlm_position_budget // ids.shape[1]))
    for rows in active.split(batch_rows):
        if not len(rows):
            continue
        masked = ids[rows].masked_fill(drop[rows], special["mask"])
        positions = drop[rows]
        probabilities = mlm_probabilities(encoder, masked, attention[rows], positions, special, cfg)
        owner, column = positions.nonzero(as_tuple=True)
        fill[rows[owner], column] = torch.multinomial(probabilities, draws, replacement=True, generator=rng)
    return fill.transpose(1, 2)


def scorer_log_probs(scorer, features, cfg):
    with autocast(cfg):
        logits = scorer(features.float()[:, None]).squeeze(1).float()
    return logits.log_softmax(-1)


def predicted_group(log_probs, outside):
    label = log_probs.argmax(-1)
    return torch.zeros_like(label) if outside is None else label.ne(outside).long()


def label_groups(labels):
    return (("O", "entity"), labels.index("O")) if "O" in labels else (("all",), None)


def batch_scores(selector, batch):
    eligible = context(batch["counts"], batch["focal"], batch["hidden"].shape[1])
    scores = selector(batch["hidden"], batch["focal"], batch["owner"])
    return scores, eligible


def draw_features(encoder, ids, attention, word_ids, focal, fills, cfg, gates=None, raw=None):
    n, draws, _ = fills.shape
    owners = torch.arange(n, device=ids.device).repeat_interleave(draws)
    size = chunk_size(ids, cfg, cfg.runtime.selector_score_batch_size)
    parts = []

    def interpolate(ids, attention, word_ids, focal, replacements, gates):
        with autocast(cfg):
            return focal_vectors(encoder, ids, attention, word_ids, focal, replacements, gates)

    if raw is not None:
        features = raw[:, None].expand(-1, draws, -1).clone()
        owner, draw = (fills != ids[:, None]).any(-1).nonzero(as_tuple=True)
        if len(owner):
            keys = torch.cat((owner[:, None], fills[owner, draw]), dim=-1)
            queries, inverse = torch.unique(keys, dim=0, return_inverse=True)
            for query in queries.split(size):
                row = query[:, 0]
                with autocast(cfg):
                    parts.append(focal_vectors(encoder, query[:, 1:].contiguous(), attention[row],
                                               word_ids[row], focal[row]))
            features[owner, draw] = torch.cat(parts)[inverse]
        return features

    for owner, replacement in zip(owners.split(size), fills.flatten(0, 1).split(size)):
        if gates is None:
            with autocast(cfg):
                parts.append(focal_vectors(encoder, replacement, attention[owner], word_ids[owner], focal[owner]))
        else:
            args = (ids[owner], attention[owner], word_ids[owner], focal[owner], replacement, gates[owner])
            parts.append(checkpoint(interpolate, *args, use_reentrant=False)
                         if cfg.runtime.activation_checkpointing else interpolate(*args))
    return torch.cat(parts).reshape(n, draws, -1)


def gold_log_probabilities(scorer, features, gold, cfg):
    n, draws, _ = features.shape
    log_probs = scorer_log_probs(scorer, features.flatten(0, 1), cfg).view(n, draws, -1)
    return log_probs.gather(-1, gold[:, None, None].expand(-1, draws, 1)).squeeze(-1)


def scorer_stage(cfg, output):
    bank = load(cfg)
    encoder = masked_encoder(cfg)
    full = full_features(encoder, {name: bank["sets"][name] for name in ("train", "validation")}, cfg, dtype=torch.float32)
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
    cap = cfg.certification.selector_train_focals_per_fold
    tasks = task_names(cfg)
    folds = (("g1", "g2") if cfg.stage.fold is None else (cfg.stage.fold,))
    raw = raw_rows(cfg, bank)
    training_rows = {fold: [row for row in raw if row["fold"] == fold] for fold in folds}
    rng = torch.Generator(device=cfg.runtime.device).manual_seed(cfg.seed)
    for task in tasks:
        scorer = load_model(paths(cfg) / "scorer" / task, cfg).requires_grad_(False)
        dimension = scorer.weight.shape[1]
        for fold in folds:
            torch.manual_seed(cfg.seed)
            selector = Selector(dimension, cfg.certification.hidden).to(cfg.runtime.device)
            optimizer = torch.optim.AdamW(selector.parameters(), lr=cfg.certification.selector_lr,
                                          weight_decay=cfg.train.weight_decay)
            rows = training_rows[fold]
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
            batches = loader(rows, cfg, train=True)
            total = sum(int((row["gold"][task] >= 0).sum()) for row in rows) * cfg.certification.selector_epochs
            print(f"Train ranking selector {task}/{fold} on {total // cfg.certification.selector_epochs} word focals", flush=True)
            history = []
            fitted_chunks = measured_words = 0
            sampling_start = time.monotonic()
            for epoch in range(1, cfg.certification.selector_epochs + 1):
                totals = torch.zeros(4, device=cfg.runtime.device)
                for batch in focal_batches(batches, task, cfg):
                    ids, attention, word_ids, focal, y = (batch[key] for key in
                        ("ids", "attention", "word_ids", "focal", "gold"))
                    scores, eligible = batch_scores(selector, batch)
                    available = eligible.sum(-1)
                    rho = torch.rand(len(y), device=ids.device, generator=rng)
                    budgets = word_budget(rho, available)
                    keep = kept_words(ranks(scores.detach(), eligible), eligible, budgets)
                    drop = word_to_pieces(~keep, word_ids, False)
                    fill = independent_fills(encoder, ids, attention, drop, draws, bank["special"], cfg, rng)
                    gate = word_to_pieces(soft_keep(scores, eligible, budgets, cfg), word_ids, 1.0)
                    features = draw_features(encoder, ids, attention, word_ids, focal, fill, cfg, gates=gate)
                    observed = gold_log_probabilities(scorer, features, y, cfg)
                    loss = (math.log(draws) - observed.logsumexp(-1)).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if cfg.train.clip is not None:
                        torch.nn.utils.clip_grad_norm_(selector.parameters(), cfg.train.clip)
                    optimizer.step()
                    totals += torch.stack((loss.detach() * len(y), budgets.sum(), available.sum(),
                                           available.new_tensor(len(y))))
                    fitted_chunks += 1
                    measured_words += len(y)
                    if reference is not None and fitted_chunks == cfg.certification.timing_chunks:
                        if ids.is_cuda:
                            torch.cuda.synchronize()
                        rate = (time.monotonic() - sampling_start) / measured_words
                        elapsed = time.monotonic() - started
                        estimate = elapsed + (total - measured_words) * rate
                        save_json(folder / "timing.json", {"projected_through_selector_seconds": estimate,
                                  "elapsed_seconds": elapsed, "seconds_per_training_word": rate,
                                  "timed_words": measured_words, "reference_seconds": reference})
                        print(f"{task}/{fold}: projected through selector training {estimate / 3600:.2f}h", flush=True)
                loss, kept, available, words = (totals / totals[3]).tolist()
                history.append({"epoch": epoch, "cross_entropy": loss, "mean_training_kept": kept,
                                "mean_context_words": available, "words": int(totals[3])})
                print(f"{task}/{fold} epoch {epoch}: CE {loss:.4f}, mean budget {kept:.2f}/{available:.2f}", flush=True)
            torch.save({"state": selector.state_dict(), "dimension": dimension}, folder / "model.pt")
            write_csv(folder / "history.csv", history)
    return {"selectors": len(folds) * len(tasks)}
