"""Word-level CRTs and held-out cross-entropy certificates."""
import math
import time

import torch
from torch.utils.data import DataLoader

from ..diagnostics.common import read_csv, save_json, write_csv
from ..encoder import load_masked_encoder as masked_encoder
from ..partition.train import load_model
from ..runtime import autocast
from .calibrate import sample_group_rows
from .data import collate, context, focal_vectors, load, paths, pool_words, task_names, unique_queries, word_to_pieces
from .selector import (batch_scores, chunk_size, focal_batches, kept_words, label_groups, load_selector,
                       mlm_probabilities, predicted_group, ranks, raw_rows, replacement_positions,
                       scorer_log_probs, word_budget)


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
def accepted_draws(encoder, selector, scorer, ids, attention, word_ids, counts, focal, gold, original,
                   keep, replace, expected_group, fractions, outside, special, cfg, rng, groups=1):
    n = len(ids)
    device = ids.device
    draws = cfg.certification.draws
    limit = cfg.certification.max_proposals
    attempts = torch.zeros(n, dtype=torch.long, device=device)
    passes = torch.zeros_like(attempts)
    found = torch.zeros_like(attempts)
    accepted = original.new_full((n, draws), torch.nan)
    has_targets = replace.any(-1)
    if not has_targets.any():
        return accepted, found, passes, attempts
    positions = replacement_positions(replace)
    masked = ids.masked_fill(replace, special["mask"])
    eligible_context = context(counts, focal, keep.shape[1])
    fixed_mask = (~eligible_context | keep).all(-1) | (~eligible_context | ~keep).all(-1)
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
    uniforms = torch.empty((n, limit, positions.shape[1]), device=device)
    for index, steps in enumerate(width_per_group.tolist()):
        if steps:
            uniforms[index * rows_per_group:(index + 1) * rows_per_group, :, :steps] = torch.rand(
                (rows_per_group, limit, steps), device=device, generator=rng
            )
    repetitions = max(1, batch_size // rows_per_group)
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
        draws_now = uniforms[owners, attempts[owners] + offset]
        candidates = chain(encoder, masked[owners], attention[owners], positions[owners],
                           special, cfg, draws_now, first_cdf, first_owner[owners], batch_size)
        same_mask = (candidates == ids[owners]).all(-1)
        candidate_scores = original[owners].clone()
        changed = (~same_mask).nonzero().flatten()
        if len(changed):
            source = owners[changed]
            keys = torch.cat((source[:, None], candidates[changed]), dim=-1)
            queries, inverse = torch.unique(keys, dim=0, return_inverse=True)
            query_source, query_ids = queries[:, 0], queries[:, 1:].contiguous()
            matches, scores = [], []
            fixed = bool(fixed_mask[query_source].all())
            for pool, base in zip(query_ids.split(batch_size), query_source.split(batch_size)):
                query_focal = focal[base]
                with autocast(cfg):
                    if fixed:
                        features = focal_vectors(encoder, pool, attention[base], word_ids[base], query_focal)
                    else:
                        vectors = encoder(pool, attention[base])["features"]
                        hidden = pool_words(vectors, word_ids[base], keep.shape[1])
                        features = hidden[torch.arange(len(base), device=device), query_focal]
                raw = scorer_log_probs(scorer, features, cfg)
                group = predicted_group(raw, outside)
                # Stratified CRT rows also condition on the scorer-predicted group.
                match = group == expected_group[base]
                if not fixed:
                    eligible = eligible_context[base]
                    budgets = word_budget(fractions[group], eligible.sum(-1))
                    mask = kept_words(ranks(selector(hidden, query_focal), eligible), eligible, budgets)
                    match &= (mask == keep[base]).all(-1)
                matches.append(match)
                scores.append(raw.gather(1, gold[base, None]).squeeze(1))
            same_mask[changed] = torch.cat(matches)[inverse]
            candidate_scores[changed] = torch.cat(scores)[inverse]
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
        accepted[owners[store], slot[store]] = candidate_scores[store]
        successes = reported.sum(-1)
        attempts[active] += consumed
        passes[active] += successes
        found[active] = (found[active] + successes).clamp_max(draws)
    return accepted, found, passes, attempts


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
def conditions(encoder, selector, scorer, batch, keep, drop, target, original, fractions, outside, special, cfg, rng):
    ids, attention, word_ids, counts, focal, gold, group = (batch[key] for key in
        ("ids", "attention", "word_ids", "counts", "focal", "gold", "group"))
    points = len(ids)
    ids, attention, word_ids, counts, focal, gold, group, keep, original = (
        torch.cat((value, value)) for value in (ids, attention, word_ids, counts, focal, gold, group, keep, original))
    replace = torch.cat((drop, target))
    scores, found, passes, attempts = accepted_draws(
        encoder, selector, scorer, ids, attention, word_ids, counts, focal, gold, original, keep, replace, group,
        fractions, outside, special, cfg, rng, groups=2,
    )
    no_targets = ~replace.any(-1)
    scores[no_targets] = original[no_targets, None]
    errors, nll, entropy = regeneration(encoder, ids, attention, replace, special, cfg, groups=2)
    results = {"original": original, "resampled": scores, "found": found, "passes": passes, "attempts": attempts,
               "mlm_tokens": replace.sum(-1), "mlm_errors": errors.sum(-1), "mlm_nll": nll.sum(-1),
               "mlm_entropy": entropy.sum(-1)}
    return ({key: value[:points].cpu() for key, value in results.items()},
            {key: value[points:].cpu() for key, value in results.items()})


@torch.no_grad()
def evaluate_rows(encoder, selector, scorer, rows, fractions, outside, special, task, cfg, rng):
    device = cfg.runtime.device
    rows = sorted(rows, key=lambda row: len(row["ids"]))
    records = ([], [])
    masks = []
    batches = DataLoader(rows, collate_fn=collate, batch_size=cfg.certification.test_batch_size,
                         num_workers=cfg.runtime.workers, pin_memory=device.startswith("cuda"))
    for batch in focal_batches(batches, task, cfg):
        scores, eligible = batch_scores(selector, batch)
        budgets = word_budget(fractions[batch["group"]], eligible.sum(-1))
        keep = kept_words(ranks(scores, eligible), eligible, budgets)
        retained, dropped = eligible & keep, eligible & ~keep
        random = torch.rand(keep.shape, device=device, generator=rng).masked_fill(~retained, -1)
        selected = random.argmax(-1)
        target = retained & torch.arange(keep.shape[1], device=device)[None].eq(selected[:, None])
        drop_pieces = word_to_pieces(dropped, batch["word_ids"], False)
        target_pieces = word_to_pieces(target, batch["word_ids"], False)
        original = scorer_log_probs(scorer, batch["hidden"][batch["owner"], batch["focal"]], cfg)
        gold = original.gather(1, batch["gold"][:, None]).squeeze(1)
        sufficiency, minimality = conditions(encoder, selector, scorer, batch, keep, drop_pieces, target_pieces,
                                             gold, fractions, outside, special, cfg, rng)
        minimality["position"] = torch.where(target.any(-1), selected, -1).cpu()
        records[0].append(sufficiency)
        records[1].append(minimality)
        kept = retained.nonzero(as_tuple=True)[1].cpu().split(retained.sum(-1).tolist())
        removed = dropped.nonzero(as_tuple=True)[1].cpu().split(dropped.sum(-1).tolist())
        masks.extend({"kept": kept_words, "dropped": dropped_words} for kept_words, dropped_words in zip(kept, removed))
    return [{key: value for key, value in row.items() if key != "hidden"} for row in rows], records, masks


def subgroup(rows, index):
    sampled = []
    for row in rows:
        keep = row["group"].eq(index)
        if keep.any():
            sampled.append({**row, "focals": row["focals"][keep], "group": row["group"][keep],
                            "gold": {key: labels[keep] for key, labels in row["gold"].items()}})
    return sampled


def group_test(encoder, selector, scorer, rows, fractions, outside, bank, cfg, meta, folder, budget):
    rng = torch.Generator(device=cfg.runtime.device).manual_seed(cfg.seed)
    pilot = max(1, math.ceil(0.01 * len(rows))) if rows else 0
    start = time.monotonic()
    def test(part):
        return evaluate_rows(encoder, selector, scorer, part, fractions, outside, bank["special"], meta["task"], cfg, rng)

    selected, records, masks = test(rows[:pilot]) if pilot else ([], ([], []), [])
    elapsed = time.monotonic() - start
    estimate = elapsed * len(rows) / pilot if pilot else 0.0
    retained = max(pilot, min(len(rows), int(budget * pilot / elapsed))) if pilot else 0
    label = f"{meta['task']}/{meta['selector']}/{meta['group']}"
    print(f"CRT {label}: pilot {pilot}/{len(rows)} rows, projected {estimate / 3600:.2f}h, "
          f"using {retained} rows with {cfg.certification.draws} draws", flush=True)
    if retained > pilot:
        more, additional, more_masks = test(rows[pilot:retained])
        selected.extend(more)
        masks.extend(more_masks)
        for saved, extra in zip(records, additional):
            saved.extend(extra)
    save_json(folder / f"{meta['selector']}_{meta['group']}_timing.json", {
        "available_rows": len(rows), "pilot_rows": pilot, "pilot_seconds": elapsed,
        "estimated_all_seconds": estimate, "selected_rows": retained, "draws": cfg.certification.draws,
        "seconds": time.monotonic() - start,
    })
    totals = []
    artifacts = {"rows": selected, "masks": masks}
    bound = {"rows": 0}
    for kind, batches in zip(("sufficiency", "minimality"), records):
        if batches:
            values = {key: torch.cat([batch[key] for batch in batches]) for key in batches[0]}
            summary = summarize(values, cfg, sufficiency=kind == "sufficiency")
            artifacts[kind] = values
            if kind == "sufficiency":
                bound = certificate(values, len(bank["labels"][meta["task"]]), cfg)
        else:
            summary = {"eligible_rows": 0, "complete_rows": 0, "pooled_rows": 0, "p": None, "effect": None}
        totals.append({**meta, "test": kind, **summary})
    counts = [(len(mask["kept"]), len(mask["dropped"])) for mask in masks]
    mean_kept = sum(k for k, _ in counts) / len(counts) if counts else None
    mean_dropped = sum(d for _, d in counts) / len(counts) if counts else None
    kept_fractions = [k / (k + d) for k, d in counts if k + d]
    size = {"rows": len(counts), "mean_kept_context": mean_kept, "mean_dropped_context": mean_dropped,
            "kept_fraction": sum(kept_fractions) / len(kept_fractions) if kept_fractions else None}
    p = totals[0]["p"]
    gap = bound.get("cross_entropy_gap")
    passed = p is not None and gap is not None and p > cfg.certification.alpha and gap <= cfg.certification.epsilon
    gate = {**meta, "p": p, "alpha": cfg.certification.alpha, "cross_entropy_gap": gap,
            "epsilon": cfg.certification.epsilon, "passed": passed}
    print(f"CRT {label}: sufficiency p={p}, certificate gap={gap}, gate {'passed' if passed else 'failed'}", flush=True)
    return totals, {**meta, **bound}, {**meta, **size}, gate, artifacts


@torch.no_grad()
def experiment(cfg, output):
    bank = load(cfg)
    raw = {row["sentence"]: row for row in raw_rows(cfg, bank)}
    encoder = masked_encoder(cfg)
    pairs = (("g1", "g2"), ("g2", "g1"))
    if cfg.stage.fold is not None:
        pairs = tuple(pair for pair in pairs if pair[0] == cfg.stage.fold)
    tasks = task_names(cfg)
    summaries, bounds, sizes, gates = [], [], [], []
    for task in tasks:
        names, outside = label_groups(bank["labels"][task])
        budget = cfg.certification.test_budget_seconds / (len(tasks) * len(pairs) * len(names))
        scorer = load_model(paths(cfg) / "scorer" / task, cfg)
        folder = output / task
        folder.mkdir(parents=True, exist_ok=True)
        for fold, opposite in pairs:
            selector = load_selector(cfg, task, fold)
            saved = torch.load(paths(cfg) / "calibrate" / task / f"{fold}.pt", weights_only=False)
            fractions = torch.tensor(saved["fractions"], device=cfg.runtime.device, dtype=torch.float64)
            sentences = saved["test_sentences"][:cfg.certification.test_documents]
            sampled = sample_group_rows([raw[sentence] for sentence in sentences], task, scorer, outside, cfg, cfg.seed + 1)
            artifacts = {}
            for index, name in enumerate(names):
                meta = {"dataset": cfg.certification.source, "task": task, "selector": fold, "held_out": opposite,
                        "group_unit": "sentence", "group": name, "rho": saved["fractions"][index],
                        "dropped_wordpiece_counts_visible": True}
                result, bound, size, gate, detail = group_test(
                    encoder, selector, scorer, subgroup(sampled, index), fractions, outside,
                    bank, cfg, meta, folder, budget,
                )
                summaries.extend(result)
                bounds.append(bound)
                sizes.append(size)
                gates.append(gate)
                artifacts[name] = detail
            torch.save(artifacts, folder / f"{fold}.pt")
    tables = output if cfg.stage.fold is None else output / cfg.stage.fold
    tables.mkdir(parents=True, exist_ok=True)
    for name, rows in (("crt", summaries), ("certificate", bounds), ("mask_sizes", sizes), ("gate", gates)):
        write_csv(tables / f"{name}.csv", rows)
    return {"groups": len(gates), "passed_groups": sum(gate["passed"] for gate in gates)}


def merge_tables(cfg):
    output = paths(cfg) / "certify"
    order = {task: i for i, task in enumerate(task_names(cfg))}
    for name in ("crt.csv", "certificate.csv", "mask_sizes.csv", "gate.csv"):
        rows = read_csv(output / "g1" / name) + read_csv(output / "g2" / name)
        rows.sort(key=lambda row: order[row["task"]])
        write_csv(output / name, rows)


def gate_passed(cfg):
    rows = read_csv(paths(cfg) / "certify" / "gate.csv")
    return bool(rows) and all(row["passed"] == "True" for row in rows)
