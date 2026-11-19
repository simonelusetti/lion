"""Mask-conditional sufficiency, minimality and cross-entropy certification."""
import math
import time
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from ..diagnostics.common import read_csv, save_json, write_csv
from ..encoder import load_masked_encoder as masked_encoder
from ..partition.train import load_model
from ..runtime import autocast
from .data import collate, context, load, paths, task_names, unique_queries
from .selector import chunk_size, load_selector, mlm_probabilities, replacement_positions


def held_out(rows, task, cfg):
    documents = defaultdict(list)
    if rows:
        targets = [row["gold"][task] for row in rows]
        owner = torch.arange(len(rows)).repeat_interleave(torch.tensor([len(target) for target in targets]))
        eligible = owner[torch.cat(targets) >= 0].unique_consecutive().tolist()
        for index in eligible:
            row = rows[index]
            documents[row["document"]].append(row)
    rng = torch.Generator().manual_seed(cfg.seed)
    order = torch.randperm(len(documents), generator=rng)[:cfg.certification.test_documents]
    names = sorted(documents)
    selected = []
    for index in order.tolist():
        candidates = documents[names[index]]
        row = candidates[int(torch.randint(len(candidates), (), generator=rng))]
        options = (row["gold"][task] >= 0).nonzero().flatten()
        chosen = options[torch.randint(len(options), (1,), generator=rng)]
        selected.append({**row, "focals": row["focals"][chosen],
                         "gold": {key: value[chosen] for key, value in row["gold"].items()}})
    return selected


@torch.no_grad()
def batched_mlm(encoder, ids, attention, positions, special, cfg, batch_size):
    if len(ids) <= batch_size:
        return mlm_probabilities(encoder, ids, attention, positions, special, cfg)
    batches = zip(*(value.split(batch_size) for value in (ids, attention, positions)))
    return torch.cat([mlm_probabilities(encoder, *batch, special, cfg) for batch in batches])


def draw_tokens(cdf, queries, uniforms):
    """Search one vocabulary CDF per distinct input, with all of its requested uniforms."""
    columns = torch.arange(len(queries), device=cdf.device)
    upper = cdf[queries, -1]
    values = (uniforms * upper).minimum(torch.nextafter(upper, torch.zeros_like(upper)))
    packed = cdf.new_zeros((len(cdf), len(queries)))
    packed[queries, columns] = values
    return torch.searchsorted(cdf, packed, right=True)[queries, columns]


@torch.no_grad()
def chain(encoder, masked, attention, positions, special, cfg, uniforms, first_cdf, owner, batch_size):
    filled = masked.clone()
    for step, index in enumerate(positions.T):
        rows = (index < masked.shape[1]).nonzero().flatten()
        slots = index[rows]
        if not len(rows):
            break
        if step == 0:
            cumulative, group = first_cdf, owner[rows]
        else:
            inputs, valid, positions_now, group = unique_queries(filled[rows], attention[rows], slots)
            probability = batched_mlm(encoder, inputs, valid, positions_now, special, cfg, batch_size)
            cumulative = probability.cumsum(-1)
        filled[rows, slots] = draw_tokens(cumulative, group, uniforms[rows, step])
    return filled


@torch.no_grad()
def regeneration(encoder, ids, attention, replace, special, cfg, groups=1):
    errors = torch.zeros_like(replace)
    nll = torch.zeros_like(ids, dtype=torch.float32)
    entropy = torch.zeros_like(nll)
    rows, positions = replace.nonzero(as_tuple=True)
    if not len(rows):
        return errors, nll, entropy
    if groups > 1:
        # Keep queries for the same original sentence and position near each other.
        order = ((rows % (len(ids) // groups)) * ids.shape[1] + positions).argsort()
        rows, positions = rows[order], positions[order]
    columns = torch.arange(ids.shape[1], device=ids.device)[None]
    size = chunk_size(ids, cfg)
    for row, position in zip(rows.split(size), positions.split(size)):
        # The earlier true tokens are known, so every teacher-forced input is available at once.
        masked = replace[row] & (columns >= position[:, None])
        inputs = ids[row].masked_fill(masked, special["mask"])
        inputs, valid, focal, inverse = unique_queries(inputs, attention[row], position)
        probability = mlm_probabilities(encoder, inputs, valid, focal, special, cfg)
        value = ids[row, position]
        errors[row, position] = probability.argmax(-1)[inverse] != value
        nll[row, position] = -probability[inverse, value].log()
        entropy[row, position] = -torch.special.xlogy(probability, probability).sum(-1)[inverse]
    return errors, nll, entropy


@torch.no_grad()
def accepted_draws(encoder, selector, ids, attention, focal, keep, replace, special, cfg, rng, groups=1,
                   draws=None):
    n, width = ids.shape
    device = ids.device
    cleaning = draws is not None
    draws = cfg.certification.draws if draws is None else draws
    limit = cfg.certification.max_proposals
    attempts = torch.zeros(n, dtype=torch.long, device=device)
    passes = torch.zeros_like(attempts)
    found = torch.zeros_like(attempts)
    accepted = torch.empty(n, draws, width, dtype=ids.dtype, device=device)
    has_targets = replace.any(-1)
    if not has_targets.any():
        return accepted, found, passes, attempts
    positions = replacement_positions(replace)
    masked = ids.masked_fill(replace, special["mask"])
    batch_size = min(cfg.certification.proposal_batch_size, chunk_size(ids, cfg))
    starts = has_targets.nonzero().flatten()
    first_ids, first_attention, first_position, inverse = unique_queries(
        masked[starts], attention[starts], positions[starts, 0]
    )
    first_cdf = batched_mlm(encoder, first_ids, first_attention, first_position, special, cfg, batch_size).cumsum(-1)
    first_owner = torch.empty_like(found)
    first_owner[starts] = inverse
    rows_per_group = n // groups
    width_per_group = replace.sum(-1).view(groups, rows_per_group).amax(-1)
    # Use the same RNG shapes and order as separate conditions before sharing any encoder calls.
    if not cleaning:
        uniforms = torch.empty((n, limit, positions.shape[1]), device=device)
        for index, steps in enumerate(width_per_group.tolist()):
            if steps:
                uniforms[index * rows_per_group:(index + 1) * rows_per_group, :, :steps] = torch.rand(
                    (rows_per_group, limit, steps), device=device, generator=rng
                )
    repetitions = 1 if cleaning else max(1, batch_size // rows_per_group)
    window = max(1, cfg.certification.proposal_batch_size // (n * repetitions)) * repetitions
    short = (width_per_group <= cfg.runtime.crt_prefetch_max_positions).repeat_interleave(rows_per_group)
    lookahead = torch.where(short, window, repetitions)
    offsets = torch.arange(window, device=device)
    while True:
        active = ((found < draws) & (attempts < limit) & has_targets).nonzero().flatten()
        if not len(active):
            break
        remaining = draws - found[active]
        rounded = ((remaining + repetitions - 1) // repetitions) * repetitions
        requested = torch.minimum(torch.minimum(lookahead[active], rounded), limit - attempts[active])
        row, offset = (offsets[None] < requested[:, None]).nonzero(as_tuple=True)
        owners = active[row]
        draws_now = (torch.rand((len(owners), positions.shape[1]), device=device, generator=rng) if cleaning else
                     uniforms[owners, attempts[owners] + offset])
        candidates = chain(encoder, masked[owners], attention[owners], positions[owners],
                           special, cfg, draws_now, first_cdf, first_owner[owners], batch_size)
        same_mask = (candidates == ids[owners]).all(-1)
        changed = (~same_mask).nonzero().flatten()
        if len(changed):
            source = owners[changed]
            inputs, valid_positions, query_focal, inverse = unique_queries(
                candidates[changed], attention[source], focal[source]
            )
            with autocast(cfg):
                hidden = torch.cat([encoder(*batch)["features"] for batch in
                                    zip(inputs.split(batch_size), valid_positions.split(batch_size))])
            eligible = context(valid_positions, query_focal)
            mask = selector.mask(hidden, query_focal, eligible)
            same_mask[changed] = (mask[inverse] == keep[source]).all(-1)
        valid = torch.zeros((len(active), window), dtype=torch.bool, device=device)
        valid[row, offset] = same_mask
        cumulative = found[active, None] + valid.cumsum(dim=-1)
        # Ignore lookahead after the original round in which the last required draw was accepted.
        reached = cumulative >= draws
        first_hit = torch.where(reached.any(-1), reached.long().argmax(-1), window)
        consumed = torch.minimum((first_hit // repetitions + 1) * repetitions, requested)
        reported = valid & (offsets[None] < consumed[:, None])
        slot = cumulative[row, offset] - 1
        store = reported[row, offset] & (slot < draws)
        accepted[owners[store], slot[store]] = candidates[store]
        successes = reported.sum(-1)
        attempts[active] += consumed
        passes[active] += successes
        found[active] = (found[active] + successes).clamp_max(draws)
    return accepted, found, passes, attempts


@torch.no_grad()
def score_features(scorer, features, gold, cfg):
    with autocast(cfg):
        logits = scorer(features.float()[:, None]).squeeze(1).float()
    return logits.log_softmax(-1).gather(1, gold[:, None]).squeeze(1)


@torch.no_grad()
def log_probabilities(encoder, scorer, ids, attention, focal, gold, cfg):
    inputs, valid, positions, inverse = unique_queries(ids, attention, focal)
    size = chunk_size(ids, cfg)
    with autocast(cfg):
        features = torch.cat([encoder(*batch)["features"] for batch in
                              zip(inputs.split(size), valid.split(size), positions.split(size))])
    return score_features(scorer, features[inverse], gold, cfg)


@torch.no_grad()
def conditions(encoder, selector, scorer, ids, attention, focal, gold, keep, drop,
               target, original, special, cfg, rng):
    # Sufficiency redraws D. Minimality redraws j alone, leaving all of x_{-j} visible.
    points = len(ids)
    ids, attention, focal, gold, keep, original = (torch.cat((value, value)) for value in
                                                   (ids, attention, focal, gold, keep, original))
    replace = torch.cat((drop, target))
    accepted, found, passes, attempts = accepted_draws(
        encoder, selector, ids, attention, focal, keep, replace, special, cfg, rng, groups=2,
    )
    draws = cfg.certification.draws
    no_targets = ~replace.any(-1)
    scores = torch.full((len(ids), draws), torch.nan, device=ids.device)
    scores[no_targets] = original[no_targets, None]
    row, draw = (torch.arange(draws, device=ids.device)[None] < found[:, None]).nonzero(as_tuple=True)
    values = accepted[row, draw]
    same = (values == ids[row]).all(-1)
    scores[row[same], draw[same]] = original[row[same]]
    if (~same).any():
        changed = row[~same]
        scores[changed, draw[~same]] = log_probabilities(
            encoder, scorer, values[~same], attention[changed], focal[changed], gold[changed], cfg,
        )
    errors, nll, entropy = regeneration(encoder, ids, attention, replace, special, cfg, groups=2)
    results = {"original": original, "resampled": scores, "found": found, "passes": passes, "attempts": attempts,
               "mlm_tokens": replace.sum(-1), "mlm_errors": errors.sum(-1), "mlm_nll": nll.sum(-1),
               "mlm_entropy": entropy.sum(-1),
               "position": torch.where(replace.sum(-1) == 1, replace.long().argmax(-1), -1)}
    sufficiency = {key: value[:points].cpu() for key, value in results.items()}
    minimality = {key: value[points:].cpu() for key, value in results.items()}
    first = torch.where((found[:points] > 0)[:, None], accepted[:points, 0], ids[:points]).cpu()
    reusable = ((found[:points] > 0) | no_targets[:points]).cpu()
    return sufficiency, minimality, first, reusable


def summarize(values, cfg, sufficiency):
    has_targets = values["mlm_tokens"] > 0
    draws = cfg.certification.draws
    complete = has_targets & (values["found"] == draws)
    eligible = int(has_targets.sum())
    tokens = int(values["mlm_tokens"][has_targets].sum())
    attempts = int(values["attempts"][has_targets].sum())
    tested = complete | ~has_targets if sufficiency else complete
    result = {
        "eligible_rows": eligible, "complete_rows": int(complete.sum()), "pooled_rows": int(tested.sum()),
        "proposals": attempts,
        "mask_acceptance": float(values["passes"][has_targets].sum() / attempts) if attempts else None,
        "mlm_top1_error": float(values["mlm_errors"][has_targets].sum() / tokens) if tokens else None,
        "mlm_nll": float(values["mlm_nll"][has_targets].sum() / tokens) if tokens else None,
        "mlm_entropy": float(values["mlm_entropy"][has_targets].sum() / tokens) if tokens else None,
        "p": None, "effect": None,
    }
    if tested.any():
        original = values["original"][tested].double()
        resampled = values["resampled"][tested].double()
        difference = (original[:, None] - resampled).sum(0)
        result["p"] = (1 + int((difference <= 0).sum())) / (draws + 1)
        result["effect"] = float((original - resampled.logsumexp(-1) + math.log(draws)).mean())
    return result


def certificate(values, classes, cfg):
    no_drop = values["mlm_tokens"] == 0
    complete = (values["found"] == cfg.certification.draws) | no_drop
    n = int(complete.sum())
    result = {"rows": n, "no_drop_rows": int(no_drop.sum()), "incomplete_rows": len(complete) - n,
              "cross_entropy": None, "full_cross_entropy": None, "cross_entropy_gap": None,
              "ucb_clt_nats": None, "ucb_hoeffding_nats": None}
    if n:
        original = values["original"][complete].double()
        probabilities = values["resampled"][complete].double().exp().mean(-1)
        gamma = cfg.certification.gamma
        loss = -((1 - gamma) * probabilities + gamma / classes).log()
        mean = float(loss.mean())
        margin = math.log(classes / gamma) * math.sqrt(math.log(1 / cfg.certification.alpha) / (2 * n))
        result.update(cross_entropy=mean, full_cross_entropy=float(-original.mean()),
                      cross_entropy_gap=float((loss + original).mean()),
                      ucb_clt_nats=mean + 1.645 * float(loss.std()) / math.sqrt(n) if n > 1 else None,
                      ucb_hoeffding_nats=mean + margin)
    return result


@torch.no_grad()
def evaluate_rows(encoder, selector, scorer, rows, bank, task, cfg, rng):
    device = cfg.runtime.device
    special = bank["special"]
    rows = sorted(rows, key=lambda row: len(row["ids"]))
    records = ([], [])
    masks = []
    fills = []
    loader = DataLoader(rows, collate_fn=collate, batch_size=cfg.certification.test_batch_size,
                        num_workers=cfg.runtime.workers, pin_memory=device.startswith("cuda"))
    for batch in loader:
        ids, attention, focal = (batch[key].to(device, non_blocking=True) for key in ("ids", "attention", "focal"))
        gold = batch["gold"][task].to(device)
        with autocast(cfg):
            hidden = encoder(ids, attention)["features"]
        eligible = context(attention, focal)
        keep = selector.mask(hidden, focal, eligible)
        drop, retained = eligible & ~keep, eligible & keep
        random = torch.rand(ids.shape, device=device, generator=rng).masked_fill(~retained, -1)
        target = retained & torch.zeros_like(retained).scatter_(1, random.argmax(-1, keepdim=True), True)
        original = score_features(scorer, hidden[torch.arange(len(ids), device=device), focal], gold, cfg)
        sufficiency, minimality, first, reusable = conditions(
            encoder, selector, scorer, ids, attention, focal, gold, keep, drop, target, original, special, cfg, rng
        )
        records[0].append(sufficiency)
        records[1].append(minimality)
        kept = retained.nonzero(as_tuple=True)[1].cpu().split(retained.sum(-1).tolist())
        dropped = drop.nonzero(as_tuple=True)[1].cpu().split(drop.sum(-1).tolist())
        masks.extend({"kept": k, "dropped": d} for k, d in zip(kept, dropped))
        fills.extend(value[:length].clone() if valid else None
                     for value, length, valid in zip(first, batch["lengths"].tolist(), reusable.tolist()))
    return rows, records, masks, fills


def experiment(cfg, output):
    bank = load(cfg)
    encoder = masked_encoder(cfg)
    device = cfg.runtime.device
    summaries, bounds, sizes = [], [], []
    pairs = (("g1", "g2"), ("g2", "g1"))
    if cfg.stage.fold is not None:
        pairs = tuple((fold, opposite) for fold, opposite in pairs if fold == cfg.stage.fold)
    tasks = task_names(cfg)
    budget = cfg.certification.test_budget_seconds / (len(tasks) * len(pairs))
    for task in tasks:
        scorer = load_model(paths(cfg) / "scorer" / task, cfg).requires_grad_(False)
        folder = output / task
        folder.mkdir(parents=True, exist_ok=True)
        for fold, opposite in pairs:
            selector = load_selector(cfg, task, fold)
            rows = held_out([row for row in bank["sets"]["train"] if row["fold"] == opposite], task, cfg)
            if not rows:
                raise ValueError(f"No labeled diagnostic rows for {task}/{fold}.")
            rng = torch.Generator(device=device).manual_seed(cfg.seed)
            pilot_count = max(1, math.ceil(0.01 * len(rows)))
            start = time.monotonic()
            selected, records, masks, first_fills = evaluate_rows(
                encoder, selector, scorer, rows[:pilot_count], bank, task, cfg, rng
            )
            pilot_seconds = time.monotonic() - start
            estimate = pilot_seconds * len(rows) / pilot_count
            measured_rate = pilot_seconds / pilot_count
            retained_rows = max(pilot_count, min(len(rows), int(budget / measured_rate)))
            print(f"{task}/{fold}: 1% pilot {pilot_count}/{len(rows)} rows in {pilot_seconds:.1f}s. "
                  f"ETA {estimate / 3600:.2f}h for all rows. "
                  f"Using {retained_rows} rows with {cfg.certification.draws} draws.", flush=True)
            if retained_rows > pilot_count:
                more, additional, more_masks, fills = evaluate_rows(
                    encoder, selector, scorer, rows[pilot_count:retained_rows], bank, task, cfg, rng
                )
                selected.extend(more)
                for result, extra in zip(records, additional):
                    result.extend(extra)
                masks.extend(more_masks)
                first_fills.extend(fills)
            seconds = time.monotonic() - start
            meta = {"dataset": cfg.certification.source, "task": task, "selector": fold,
                    "held_out": opposite, "group_unit": bank["group_unit"]}
            save_json(folder / f"{fold}_timing.json", {
                "available_rows": len(rows), "pilot_rows": pilot_count, "pilot_seconds": pilot_seconds,
                "estimated_all_seconds": estimate, "selected_rows": retained_rows,
                "draws": cfg.certification.draws, "seconds": seconds,
            })
            kept = sum(len(mask["kept"]) for mask in masks)
            dropped = sum(len(mask["dropped"]) for mask in masks)
            fractions = [len(mask["kept"]) / (len(mask["kept"]) + len(mask["dropped"]))
                         for mask in masks if len(mask["kept"]) + len(mask["dropped"])]
            sizes.append({**meta, "rows": len(selected), "mean_kept_context": kept / len(selected),
                          "mean_dropped_context": dropped / len(selected),
                          "kept_fraction": sum(fractions) / len(fractions) if fractions else None})
            artifacts = {}
            for kind, batches in zip(("sufficiency", "minimality"), records):
                values = {key: torch.cat([batch[key] for batch in batches]) for key in batches[0]}
                summary = summarize(values, cfg, sufficiency=kind == "sufficiency")
                summaries.append({**meta, "test": kind, **summary})
                if kind == "sufficiency":
                    bounds.append({**meta, **certificate(values, len(bank["labels"][task]), cfg)})
                artifacts[kind] = values
                print(f"{task}/{fold} {kind}: {summary['complete_rows']}/{summary['eligible_rows']} complete",
                      flush=True)
            torch.save({"rows": selected, "masks": masks, "first_fills": first_fills, **artifacts}, folder / f"{fold}.pt")
    tables = output if cfg.stage.fold is None else output / cfg.stage.fold
    tables.mkdir(parents=True, exist_ok=True)
    write_csv(tables / "crt.csv", summaries)
    write_csv(tables / "certificate.csv", bounds)
    write_csv(tables / "mask_sizes.csv", sizes)
    return {"selectors": len(bounds)}


def merge_tables(cfg):
    output = paths(cfg) / "certify"
    order = {task: i for i, task in enumerate(task_names(cfg))}
    for name in ("crt.csv", "certificate.csv", "mask_sizes.csv"):
        rows = read_csv(output / "g1" / name) + read_csv(output / "g2" / name)
        rows.sort(key=lambda row: order[row["task"]])
        write_csv(output / name, rows)
