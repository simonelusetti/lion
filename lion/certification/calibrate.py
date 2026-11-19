"""Calibrate kept-word fractions by scorer-predicted group."""
import math
import time

import torch

from ..diagnostics.common import read_csv, save_json, write_csv
from ..encoder import load_masked_encoder
from ..partition.train import load_model
from .data import load, loader, paths, task_names, word_to_pieces
from .selector import (batch_scores, chunk_size, draw_features, focal_batches, gold_log_probabilities,
                       independent_fills, kept_words, label_groups, load_selector, predicted_group,
                       ranks, raw_rows, scorer_log_probs, word_budget)


@torch.no_grad()
def sample_group_rows(rows, task, scorer, outside, cfg, seed):
    rng = torch.Generator(device=cfg.runtime.device).manual_seed(seed)
    width = 1 if outside is None else 2
    sampled = []
    start = 0
    for batch in loader(rows, cfg):
        device = cfg.runtime.device
        hidden = batch["hidden"].to(device, non_blocking=True)
        owner = batch["owner"].to(device, non_blocking=True)
        focal = batch["focal"].to(device, non_blocking=True)
        gold = batch["gold"][task].to(device, non_blocking=True)
        group = predicted_group(scorer_log_probs(scorer, hidden[owner, focal], cfg), outside)
        slot = owner * width + group
        chance = torch.rand(len(owner), device=device, dtype=torch.float64, generator=rng).masked_fill(gold < 0, -1)
        best = chance.new_full((len(hidden) * width,), -1)
        best.scatter_reduce_(0, slot, chance, reduce="amax")
        indices = torch.arange(len(owner), device=device)
        tied = chance.ge(0) & chance.eq(best[slot])
        winner = owner.new_full((len(hidden) * width,), len(owner))
        winner.scatter_reduce_(0, slot, torch.where(tied, indices, len(owner)), reduce="amin")
        rows_kept = indices[indices == winner[slot]]
        chosen_owner = owner[rows_kept].tolist()
        chosen_focal = focal[rows_kept].cpu()
        chosen_group = group[rows_kept].cpu()
        sizes = torch.bincount(owner[rows_kept], minlength=len(hidden)).tolist()
        start_focal = 0
        for index, count in zip(dict.fromkeys(chosen_owner), (count for count in sizes if count)):
            row = rows[start + index]
            which = chosen_focal[start_focal:start_focal + count]
            # The full input has one focal per word in its original order.
            sampled.append({**row, "focals": which, "group": chosen_group[start_focal:start_focal + count],
                            "gold": {key: labels[which] for key, labels in row["gold"].items()}})
            start_focal += count
        start += len(hidden)
    return sampled


def grid_points(step, device):
    stops = 1.0 - torch.arange(math.ceil(1 / step) + 1, device=device, dtype=torch.float64) * step
    return stops.round(decimals=10).clamp_min(0).unique(sorted=True).flip(0)


def smooth(probability, classes, gamma):
    return ((1 - gamma) * probability + gamma / classes).log()


@torch.no_grad()
def calibration_losses(encoder, selector, scorer, batch, grid, special, cfg, rng):
    ids, attention, word_ids, focal, gold = (batch[key] for key in
        ("ids", "attention", "word_ids", "focal", "gold"))
    scores, eligible = batch_scores(selector, batch)
    rank = ranks(scores, eligible)
    available = eligible.sum(-1)
    budgets = word_budget(grid, available[:, None])
    changed = torch.ones_like(budgets, dtype=torch.bool)
    changed[:, 1:] = budgets[:, 1:] != budgets[:, :-1]
    owners, point = (changed & (budgets < available[:, None])).nonzero(as_tuple=True)
    if not len(owners):
        return torch.zeros_like(budgets, dtype=torch.float64)
    base = batch["hidden"][batch["owner"], focal]
    original = scorer_log_probs(scorer, base, cfg)
    gamma = cfg.certification.gamma
    classes = original.shape[-1]
    raw = smooth(original.gather(-1, gold[:, None]).squeeze(-1).double().exp(), classes, gamma)
    losses = raw.new_zeros(budgets.shape)
    size = chunk_size(ids, cfg)
    draws = cfg.certification.calibration_draws
    for row, candidate in zip(owners.split(size), point.split(size)):
        keep = kept_words(rank[row], eligible[row], budgets[row, candidate])
        drop = word_to_pieces(~keep, word_ids[row], False)
        fills = independent_fills(encoder, ids[row], attention[row], drop, draws, special, cfg, rng)
        features = draw_features(encoder, ids[row], attention[row], word_ids[row], focal[row], fills, cfg, raw=base[row])
        logp = gold_log_probabilities(scorer, features, gold[row], cfg)
        average = logp.double().exp().mean(-1)
        losses[row, candidate] = raw[row] - smooth(average, classes, gamma)
    columns = torch.arange(len(grid), device=ids.device)[None]
    previous = torch.where(changed, columns, 0).cummax(-1).values
    return losses.gather(-1, previous)


def fixed_sequence(losses, grid, group, cfg, level):
    n = len(losses)
    epsilon = cfg.certification.epsilon
    mean = losses.mean(0) if n else grid.new_full(grid.shape, torch.nan)
    sd = losses.std(0) if n >= 2 else grid.new_full(grid.shape, torch.nan)
    se = sd / math.sqrt(max(1, n))
    p = 0.5 * torch.erfc((epsilon - mean) / (math.sqrt(2) * se))
    # A deterministic zero-loss all-kept policy is certified by construction.
    p[sd == 0] = (mean[sd == 0] >= epsilon).double()
    p[0] = 0.0
    rejected = (p <= level).long().cumprod(0).bool()
    rho = float(grid[rejected][-1])
    records = []
    for position, (fraction, risk, deviation, margin, value, passes) in enumerate(
        zip(grid.tolist(), mean.tolist(), sd.tolist(), (1.28 * se).tolist(), p.tolist(), rejected.tolist())
    ):
        records.append({"group": group, "rho": round(fraction, 10), "rows": n,
                        "mean_loss": risk if n else (0.0 if position == 0 else None),
                        "sd": deviation if n >= 2 else None,
                        "clt_margin_1_28": margin if n >= 2 else None,
                        "p": value if n >= 2 or position == 0 else None,
                        "rejected": passes, "selected": fraction == rho})
    return rho, records


@torch.no_grad()
def experiment(cfg, output):
    bank = load(cfg)
    raw = raw_rows(cfg, bank)
    encoder = load_masked_encoder(cfg)
    grid = grid_points(cfg.certification.grid_step, cfg.runtime.device)
    pairs = (("g1", "g2"), ("g2", "g1"))
    curves, summary = [], []
    for task in task_names(cfg):
        scorer = load_model(paths(cfg) / "scorer" / task, cfg)
        names, outside = label_groups(bank["labels"][task])
        folder = output / task
        folder.mkdir(parents=True, exist_ok=True)
        for fold, opposite in pairs:
            if cfg.stage.fold not in (None, fold):
                continue
            selector = load_selector(cfg, task, fold)
            held = [row for row in raw if row["fold"] == opposite]
            order = torch.randperm(len(held), generator=torch.Generator().manual_seed(cfg.seed)).tolist()
            boundary = round(cfg.certification.calibration_share * len(held))
            calibration = [held[index] for index in order[:boundary]]
            test = [held[index]["sentence"] for index in order[boundary:]]
            sampled = sample_group_rows(calibration, task, scorer, outside, cfg, cfg.seed)
            total = sum(len(row["focals"]) for row in sampled)
            print(f"Calibration {task}/{fold}: {total} word focals from {len(calibration)} sentences, "
                  f"{len(test)} held-out test sentences", flush=True)
            records, groups = [], []
            rng = torch.Generator(device=cfg.runtime.device).manual_seed(cfg.seed)
            started = time.monotonic()
            done = 0
            for batch in focal_batches(loader(sampled, cfg), task, cfg):
                records.append(calibration_losses(encoder, selector, scorer, batch, grid, bank["special"], cfg, rng).cpu())
                groups.append(batch["group"].cpu())
                previous, done = done, done + len(batch["gold"])
                if previous < min(256, total) <= done:
                    elapsed = time.monotonic() - started
                    estimate = elapsed * total / done
                    save_json(folder / f"{fold}_timing.json", {"pilot_rows": done, "total_rows": total,
                              "pilot_seconds": elapsed, "projected_seconds": estimate})
                    print(f"Calibration {task}/{fold}: projected {estimate / 3600:.2f}h from {done} focals", flush=True)
            losses, group_ids = torch.cat(records), torch.cat(groups)
            fractions = []
            for index, name in enumerate(names):
                chosen = group_ids == index
                rho, curve = fixed_sequence(losses[chosen], grid.cpu(), name, cfg, cfg.certification.delta / len(names))
                fractions.append(rho)
                meta = {"task": task, "selector": fold, "held_out": opposite}
                curves.extend({**meta, **record} for record in curve)
                summary.append({**meta, "group": name, "rows": int(chosen.sum()), "rho": round(rho, 10),
                                "calibration_sentences": len(calibration), "test_sentences": len(test)})
                print(f"Calibration {task}/{fold}/{name}: kept fraction {rho:.2f}", flush=True)
            torch.save({"fractions": fractions, "test_sentences": test}, folder / f"{fold}.pt")
    tables = output if cfg.stage.fold is None else output / cfg.stage.fold
    tables.mkdir(parents=True, exist_ok=True)
    write_csv(tables / "curve.csv", curves)
    write_csv(tables / "fractions.csv", summary)
    return {"calibrated_groups": len(summary)}


def merge_tables(cfg):
    output = paths(cfg) / "calibrate"
    for name in ("curve.csv", "fractions.csv"):
        write_csv(output / name, read_csv(output / "g1" / name) + read_csv(output / "g2" / name))
