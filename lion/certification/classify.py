"""Cache calibrated word features and evaluate paired readouts on raw text."""
import time

import torch

from ..diagnostics.common import save_json, write_csv
from ..diagnostics.data import NER, SHARED
from ..encoder import load_masked_encoder as masked_encoder
from ..labels import LABELS
from ..metrics import shared_label
from ..partition.train import fit, load_model, metric_vectors, predict
from ..runtime import parallel
from .data import full_features, load, loader, ner_transfer_sets, paths, readout_data, task_names, word_to_pieces
from .selector import (batch_scores, draw_features, focal_batches, independent_fills, kept_words, label_groups,
                       load_selector, predicted_group, ranks, raw_rows, scorer_log_probs, word_budget)


@torch.no_grad()
def clean_fold(encoder, selector, scorer, rows, fractions, outside, task, special, cfg):
    rng = torch.Generator(device=cfg.runtime.device).manual_seed(cfg.seed + 1)
    draws = cfg.certification.cleaning_draws
    total = sum(int((row["gold"][task] >= 0).sum()) for row in rows)
    features, labels, sentences, focals = [], [], [], []
    written = altered = masked_words = 0
    started = time.monotonic()
    for batch in focal_batches(loader(rows, cfg), task, cfg):
        ids, attention, word_ids, focal, y = (batch[key] for key in
            ("ids", "attention", "word_ids", "focal", "gold"))
        raw = batch["hidden"][batch["owner"], focal]
        group = predicted_group(scorer_log_probs(scorer, raw, cfg), outside)
        score, eligible = batch_scores(selector, batch)
        budgets = word_budget(fractions[group], eligible.sum(-1))
        keep = kept_words(ranks(score, eligible), eligible, budgets)
        drop = word_to_pieces(~keep, word_ids, False)
        fills = independent_fills(encoder, ids, attention, drop, draws, special, cfg, rng)
        values = draw_features(encoder, ids, attention, word_ids, focal, fills, cfg, raw=raw)
        features.append(values.cpu())
        labels.append(y.cpu())
        sentences.append(batch["sentence"])
        focals.append(focal.cpu())
        altered += int(drop.any(-1).sum())
        masked_words += int((~keep).sum())
        previous, written = written, written + len(y)
        if previous < min(256, total) <= written:
            elapsed = time.monotonic() - started
            print(f"{task}: cached {written}/{total} word focals in {elapsed:.1f}s. "
                  f"Projected fold time {elapsed * total / written / 3600:.2f}h.", flush=True)
    return {"features": torch.cat(features), "y": torch.cat(labels), "sentence": torch.cat(sentences),
            "focal": torch.cat(focals), "focals_with_drops": altered,
            "dropped_words": masked_words, "seconds": time.monotonic() - started}


def clean(cfg, output):
    bank = load(cfg)
    raw = raw_rows(cfg, bank)
    encoder = masked_encoder(cfg)
    pairs = (("g1", "g2"), ("g2", "g1"))
    total = 0
    for task in task_names(cfg):
        _, outside = label_groups(bank["labels"][task])
        scorer = load_model(paths(cfg) / "scorer" / task, cfg)
        folder = output / task
        folder.mkdir(parents=True, exist_ok=True)
        for fold, opposite in pairs:
            if cfg.stage.fold not in (None, fold):
                continue
            selector = load_selector(cfg, task, fold)
            saved = torch.load(paths(cfg) / "calibrate" / task / f"{fold}.pt", weights_only=False)
            fractions = torch.tensor(saved["fractions"], device=cfg.runtime.device, dtype=torch.float64)
            rows = [row for row in raw if row["fold"] == opposite]
            print(f"Caching {cfg.certification.cleaning_draws} draws per word for {task}/{opposite} with {fold}", flush=True)
            cleaned = clean_fold(encoder, selector, scorer, rows, fractions, outside, task, bank["special"], cfg)
            torch.save(cleaned, folder / f"{fold}.pt")
            count = len(cleaned["y"])
            total += count
            save_json(folder / f"{fold}_timing.json", {"focals": count, "draws": cfg.certification.cleaning_draws, **{
                key: cleaned[key] for key in ("focals_with_drops", "dropped_words", "seconds")
            }})
    return {"cleaned_focals": total}


def concatenate(parts):
    return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}


def span_counts(predictions, gold, sentence, source_labels, target_labels, shared=False):
    def types(labels):
        if shared:
            return [shared_label(label) for label in labels]
        return [label[2:] if label.startswith(("B-", "I-")) else label for label in labels]

    source_types, target_types = types(source_labels), types(target_labels)
    vocabulary = {name: i + 1 for i, name in enumerate(sorted(set(source_types + target_types) - {"O"}))}
    vocabulary["O"] = 0
    source_type = torch.tensor([vocabulary[name] for name in source_types])
    target_type = torch.tensor([vocabulary[name] for name in target_types])
    ids = torch.cat((source_type[predictions], target_type[gold, None]), dim=1)
    begins = torch.cat((torch.tensor([label.startswith("B-") for label in source_labels])[predictions],
                        torch.tensor([label.startswith("B-") for label in target_labels])[gold, None]), dim=1)
    sentence_start = torch.ones(len(ids), dtype=torch.bool)
    sentence_start[1:] = sentence[1:] != sentence[:-1]
    begins |= sentence_start[:, None]
    begins[1:] |= ids[1:] != ids[:-1]
    begins &= ids != 0
    ends = torch.ones_like(begins)
    ends[:-1] = begins[1:] | (ids[:-1] != ids[1:]) | sentence_start[1:, None]
    ends &= ids != 0
    token = torch.arange(len(ids))[:, None]
    start = torch.where(begins, token, -1).cummax(dim=0).values
    matches = ends[:, :-1] & ends[:, -1:] & (ids[:, :-1] == ids[:, -1:]) & (start[:, :-1] == start[:, -1:])
    counts = ends.sum(0)
    return matches.sum(0), counts[:-1], counts[-1]


def span_scores(predictions, gold, sentence, source_labels, target_labels, shared=False):
    correct, predicted, expected = span_counts(predictions, gold, sentence, source_labels, target_labels, shared=shared)
    return 2 * correct.double() / (predicted + expected).clamp_min(1)


def evaluate(logits, data, source, target, source_labels, space):
    gold = data["y"]
    values = {}
    if space == "shared_ner":
        target_labels = LABELS[target]
        values["span_f1"] = span_scores(logits.argmax(-1), gold, data["sentence"], source_labels, target_labels, shared=True)
        mapping = torch.tensor([SHARED.index(shared_label(label)) for label in source_labels])
        probabilities = logits.softmax(-1)
        projected = probabilities.new_zeros(*logits.shape[:2], len(SHARED))
        projected.scatter_add_(-1, mapping[None, None].expand_as(probabilities), probabilities)
        logits = projected.clamp_min(torch.finfo(projected.dtype).tiny).log()
        labels = SHARED
        gold = torch.tensor([SHARED.index(shared_label(label)) for label in target_labels])[gold]
        pred = logits.argmax(-1)
        offsets = torch.arange(pred.shape[1])[None] * len(labels)
        cells = pred + offsets
        correct = torch.bincount(cells[pred == gold[:, None]], minlength=pred.shape[1] * len(labels)).view(-1, len(labels))
        denominator = (torch.bincount(gold, minlength=len(labels)) +
                       torch.bincount(cells.flatten(), minlength=pred.shape[1] * len(labels)).view(-1, len(labels)))
        values["entity_macro_f1"] = (2 * correct[:, 1:] / denominator[:, 1:].clamp_min(1)).mean(-1)
    else:
        labels = source_labels
        if source in (*NER, "conll2000"):
            values["span_f1"] = span_scores(logits.argmax(-1), gold, data["sentence"], labels, labels)
    values.update(metric_vectors(logits, gold, len(labels)))
    return values


def experiment(cfg, output):
    bank = load(cfg)
    full = torch.load(paths(cfg) / "scorer" / "full_features.pt", weights_only=False, mmap=True)
    encoder = masked_encoder(cfg)
    sets = bank["sets"]
    native_train = sets["train"]
    words = torch.tensor([len(row["focals"]) for row in native_train])
    comparisons = []
    source = cfg.certification.source
    if source in NER:
        start = max(row["sentence"] for rows in sets.values() for row in rows) + 1
        transfer, domains, coverage = ner_transfer_sets(cfg, "ner", start)
        sets.update(transfer)
        bank["domains"].update(domains)
        write_csv(output / "target_coverage.csv", coverage)
    full.update(full_features(encoder, {name: rows for name, rows in sets.items() if name not in full}, cfg, dtype=torch.float32))
    for task in task_names(cfg):
        labels = bank["labels"][task]
        cleaned, matched = [], []
        for fold, opposite in (("g1", "g2"), ("g2", "g1")):
            selected = torch.tensor([row["fold"] == opposite for row in native_train])
            rows = [row for row, include in zip(native_train, selected) if include]
            accepted = torch.load(paths(cfg) / "clean" / task / f"{fold}.pt", weights_only=False, mmap=True)
            raw = readout_data(rows, full["train"][selected.repeat_interleave(words)], task)
            focals = torch.cat([row["focals"][row["gold"][task] >= 0] for row in rows])
            if (not torch.equal(accepted["y"], raw["y"]) or not torch.equal(accepted["sentence"], raw["sentence"])
                    or not torch.equal(accepted["focal"], focals)):
                raise ValueError(f"Cleaned and raw training examples are misaligned for {task}/{fold}.")
            cleaned.append({key: accepted[key] for key in ("features", "y", "sentence")})
            matched.append(raw)
        training = concatenate(cleaned)
        raw_training = concatenate(matched)
        validation = readout_data(sets["validation"], full["validation"], task)
        task_path = output / task
        names = ("cleaned", "full_matched")
        paired = {
            "train": {**training, "views": (training["features"][:, None], raw_training["features"])},
            "validation": {**validation, "views": (validation["features"], validation["features"])},
        }
        readout, _ = fit(paired, len(labels), cfg, cfg.seed, task_path, condition_outputs=tuple(
            (task_path / name, len(validation["y"])) for name in names
        ))
        readouts = parallel(readout, cfg)
        for split in sets:
            if split in ("train", "validation"):
                continue
            target = bank["domains"][split]
            evaluation = readout_data(sets[split], full[split], task)
            features = (evaluation["features"], evaluation["features"])
            logits = predict(readouts, evaluation, cfg, views=features)
            predictions = {name: logits[:, i:i + 1].contiguous() for i, name in enumerate(names)}
            spaces = ["native"] if source not in NER else (["native", "shared_ner"] if split == "id" else ["shared_ner"])
            for space in spaces:
                evaluated = evaluate(logits, evaluation, source, target, labels, space)
                metrics = {name: {key: float(value[i]) for key, value in evaluated.items()}
                           for i, name in enumerate(names)}
                for name, values in metrics.items():
                    comparisons.append({
                        "task": task, "source": source, "target": target, "split": split, "label_space": space,
                        "classifier": name, "train_words": len(training["y"]),
                        "evaluation_words": len(evaluation["y"]), **values,
                        **{f"{metric}_gain_vs_full_matched": values[metric] - metrics["full_matched"][metric]
                           for metric in ("accuracy", "macro_f1", "entity_macro_f1", "span_f1") if metric in values},
                    })
            torch.save({"predictions": predictions, "labels": labels, "target": target,
                        "target_labels": LABELS[target] if target in NER else labels, "y": evaluation["y"],
                        "sentence": evaluation["sentence"]}, task_path / f"{split}_predictions.pt")
    write_csv(output / "performance.csv", comparisons)
    return {"comparisons": len(comparisons)}
