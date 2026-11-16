"""Length-matched, same-type mention substitutions drawn from training only."""
from collections import defaultdict

import torch
from datasets import load_from_disk
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from ..data import signature
from ..encoder import build_encoder
from .common import VIEWS, dataset_config, locations, plot_table, seed_summary, write_csv
from .data import banks, encode_views, NER, token_path
from .readouts import feature_loader, load_y_model, load_y_predictions, predict
from .run import cli


def mentions(row, labels):
    spans, start, previous = [], None, None
    for position, index in enumerate([*row["labels"], -1]):
        tag = labels[index] if index >= 0 else "O"
        kind = tag[2:] if tag.startswith(("B-", "I-")) else tag
        kind = None if kind == "O" else kind
        boundary = kind != previous or tag.startswith("B-")
        if boundary and start is not None:
            spans.append((start, position, previous))
            start = None
        if kind is not None and start is None:
            start = position
        previous = kind
    return spans


def piece_lengths(row):
    ids = torch.tensor(row["word_ids"])
    return torch.bincount(ids[ids >= 0], minlength=len(row["words"])).tolist()


def mention_bank(data, labels, cfg, generator):
    bank, seen, totals = defaultdict(list), set(), defaultdict(int)
    for row in data["train"]:
        lengths = piece_lengths(row)
        for start, end, kind in mentions(row, labels):
            if end == len(row["words"]) and len(row["ids"]) == cfg.encoder.max_length:
                continue
            key = (kind, tuple(lengths[start:end]))
            surface = tuple(row["words"][start:end])
            if (key, surface) in seen:
                continue
            seen.add((key, surface))
            totals[key] += 1
            if len(bank[key]) < cfg.diagnostics.substitution.bank_per_signature:
                bank[key].append(surface)
            else:
                index = int(torch.randint(totals[key], (), generator=generator))
                if index < len(bank[key]):
                    bank[key][index] = surface
    return bank


def changed_sentence(row, span, replacement, tokenizer, cfg, focal):
    start, end, _ = span
    words = row["words"][:start] + list(replacement) + row["words"][end:]
    encoded = tokenizer(words, is_split_into_words=True, truncation=True,
                        max_length=cfg.encoder.max_length, add_special_tokens=cfg.encoder.special_tokens)
    word_ids = encoded.word_ids()
    starts = [i for i, word in enumerate(word_ids) if word is not None and
              (i == 0 or word != word_ids[i - 1])]
    # Boundary checks: substitutions must preserve every word's packed alignment.
    if ([word if word is not None else -1 for word in word_ids] != row["word_ids"] or
            len(starts) != len(words)):
        return None
    return {**row, "words": words, "ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"],
            "word_ids": [-1 if word is None else word for word in word_ids],
            "positions": starts, "focals": [focal]}


def interventions(data, bank, tokenizer, cfg, name):
    generator = torch.Generator().manual_seed(cfg.diagnostics.sample_seed)
    replacements = mention_bank(data, bank["labels"], cfg, generator)
    selected, records, skips = [], [], []
    test = bank["data"]["test"]
    lookup = {(int(s), int(p)): i for i, (s, p) in enumerate(zip(test["sentence"], test["position"]))}
    candidates = [(row, focal) for row in bank["rows"]["test"] for focal in row["focals"]]
    chosen = 0
    for candidate in torch.randperm(len(candidates), generator=generator).tolist():
        row, focal = candidates[candidate]
        spans = mentions(row, bank["labels"])
        focal_span = next((span for span in spans if span[0] <= focal < span[1]), None)
        if focal_span is None:
            continue
        lengths = piece_lengths(row)
        neighbors = [span for span in spans if span != focal_span and
                     max(span[0] - focal_span[1], focal_span[0] - span[1], 0) <= cfg.diagnostics.substitution.radius]
        neighbor = min(neighbors, key=lambda span: min(abs(span[0] - focal_span[1]),
                                                        abs(focal_span[0] - span[1]))) if neighbors else None
        accepted = False
        for condition, span in (("neighbor", neighbor), ("focal", focal_span)):
            prefix = {"dataset": name, "sentence": row["sentence_id"], "position": focal, "intervention": condition}
            if span is None:
                skips.append({**prefix, "reason": "no neighboring entity in radius"})
                continue
            start, end, kind = span
            key = (kind, tuple(lengths[start:end]))
            original = tuple(row["words"][start:end])
            options = [surface for surface in replacements.get(key, []) if surface != original]
            if not options:
                skips.append({**prefix, "reason": "no distinct same-type train mention with matching per-word WordPiece lengths"})
                continue
            replacement = options[int(torch.randint(len(options), (), generator=generator))]
            changed = changed_sentence(row, span, replacement, tokenizer, cfg, focal)
            if changed is None:
                skips.append({**prefix, "reason": "retokenized sentence does not preserve packed alignment"})
                continue
            selected.append(changed)
            records.append({**prefix, "original_index": lookup[(row["sentence_id"], focal)],
                            "focal_words": focal_span[1] - focal_span[0], "entity_type": kind,
                            "old_mention": " ".join(original), "new_mention": " ".join(replacement)})
            accepted = True
        chosen += accepted
        if chosen >= cfg.diagnostics.substitution.focals:
            break
    return selected, records, skips


def experiment(cfg, output):
    results, cache = locations(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.encoder.name, use_fast=True)
    encoder = None
    scores, skipped, interventions_rows = [], [], []
    for name, bank in banks(cfg):
        if name not in NER:
            continue
        key = signature(OmegaConf.create({"encoder": OmegaConf.to_container(cfg.encoder, resolve=True),
                                          "sampling": bank["config"]["diagnostics"],
                                          "substitution": OmegaConf.to_container(cfg.diagnostics.substitution, resolve=True)}))
        path = cache / "substitutions" / name / key / "features.pt"
        if path.exists():
            saved = torch.load(path, weights_only=False)
            features, records, skips = saved["features"], saved["records"], saved["skips"]
        else:
            data = load_from_disk(token_path(dataset_config(cfg, name)))
            changed, records, skips = interventions(data, bank, tokenizer, cfg, name)
            if not changed:
                skipped.extend(skips)
                skipped.append({"dataset": name, "reason": "no eligible substitutions"})
                continue
            if encoder is None:
                encoder = build_encoder(cfg.encoder).to(cfg.runtime.device)
            features = encode_views(changed, encoder, tokenizer, cfg)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"features": features, "records": records, "skips": skips}, path)
        skipped.extend(skips)
        interventions_rows.extend(records)
        old = torch.tensor([row["original_index"] for row in records])
        test = {"features": features, "y": bank["data"]["test"]["y"][old],
                "sentence": torch.arange(len(records))}
        for seed in cfg.diagnostics.seeds:
            for view, view_name in enumerate(VIEWS):
                model = load_y_model(cfg, name, view_name, seed)
                after = predict(model, feature_loader(test, view, cfg), len(records), len(bank["labels"]), cfg)
                before = load_y_predictions(cfg, name, view_name, seed)["test"][old]
                flips = after.argmax(-1) != before.argmax(-1)
                delta = (after.softmax(-1) - before.softmax(-1)).abs().sum(-1)
                torch.save({"before": before, "after": after, "original_indices": old,
                            "target": test["y"]}, output / f"{name}_{view_name}_seed{seed}.pt")
                for condition in ("neighbor", "focal"):
                    for length in ("all", "single_word", "multiword"):
                        mask = torch.tensor([row["intervention"] == condition and
                                             (length == "all" or (row["focal_words"] == 1) == (length == "single_word"))
                                             for row in records])
                        if mask.any():
                            scores.append({"dataset": name, "seed": seed, "view": view_name,
                                           "intervention": condition, "focal_length": length, "n": int(mask.sum()),
                                           "flip_rate": float(flips[mask].float().mean()),
                                           "probability_l1": float(delta[mask].mean()),
                                           "before_accuracy": float((before.argmax(-1)[mask] == test["y"][mask]).float().mean()),
                                           "after_accuracy": float((after.argmax(-1)[mask] == test["y"][mask]).float().mean())})
    write_csv(output / "scores.csv", scores)
    write_csv(output / "interventions.csv", interventions_rows)
    write_csv(output / "skips.csv", skipped)
    write_csv(output / "summary.csv", seed_summary(scores, ("dataset", "view", "intervention", "focal_length"),
                                                    ("flip_rate", "probability_l1")))
    plot_table(output / "flips.png", [row for row in scores if row["intervention"] == "neighbor" and row["focal_length"] == "all"], "flip_rate")
    return {"substitutions": len(interventions_rows), "view_conditions": len(scores)}


if __name__ == "__main__":
    cli("e7")
