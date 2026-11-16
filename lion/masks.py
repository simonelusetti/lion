"""Gold-informed minimum context masks in same-word reference geometry."""
from collections import Counter, defaultdict
import json

import torch
from torch.nn import functional as F
from tqdm import tqdm

from .data import loader, to_device
from .encoder import expand_gates, word_features
from .runtime import autocast, parallel


def candidates(n, cfg, device):
    if n <= cfg.masks.exhaustive_context:
        return ((torch.arange(2 ** n, device=device)[:, None] >> torch.arange(n, device=device)) & 1).bool()
    sizes = torch.randint(n + 1, (cfg.masks.samples,), device=device)
    ranks = torch.rand(cfg.masks.samples, n, device=device).argsort(-1).argsort(-1)
    sampled = ranks < sizes[:, None]
    return torch.cat([torch.zeros(1, n, device=device, dtype=torch.bool), sampled,
                      torch.ones(1, n, device=device, dtype=torch.bool)]).unique(dim=0)


def geometry(hidden, references, labels, gold, cfg):
    if cfg.masks.metric == "cosine":
        distance = 1 - F.normalize(hidden.float(), dim=-1) @ F.normalize(references.float(), dim=-1).T
    else:
        distance = torch.cdist(hidden.float(), references.float())
    neighbors = labels[distance.topk(cfg.masks.k, largest=False).indices]
    probability = (neighbors == gold).float().mean(-1)
    same = distance.masked_fill(labels[None] != gold, torch.inf).amin(-1)
    other = distance.masked_fill(labels[None] == gold, torch.inf).amin(-1)
    margin = other - same
    supported = probability > 0.5 if cfg.masks.criterion == "knn" else margin > 0
    return supported, probability, margin


@torch.no_grad()
def search(cfg, data, tokenizer, encoder, output):
    counts = defaultdict(Counter)
    for row in data["train"]:
        for word, label in zip(row["words"], row["labels"]):
            if label >= 0:
                counts[word.lower()][label] += 1
    forms = [word for word, labels in counts.items()
             if sum(count >= cfg.masks.min_per_label for count in labels.values()) >= 2]
    forms.sort(key=lambda word: (-sum(counts[word].values()), word))
    if cfg.masks.max_types is not None:
        forms = forms[:cfg.masks.max_types]
    forms = set(forms)
    reference_rows = data["train"].filter(lambda row: any(w.lower() in forms for w in row["words"]))
    references = defaultdict(list)
    encode = parallel(encoder, cfg)
    key_width = cfg.encoder.max_length + 1
    for raw in tqdm(loader(reference_rows, cfg, tokenizer), desc="Mask references"):
        batch = to_device(raw, cfg.runtime.device)
        with autocast(cfg):
            hidden = word_features(encode(batch["ids"], batch["attention_mask"], all_layers=False),
                                   batch["positions"])["features"][:, :, -1].float().cpu()
        for b, sentence_id in enumerate(raw["sentence_ids"].tolist()):
            row = data["train"][sentence_id]
            for focal, word in enumerate(row["words"]):
                if word.lower() in forms and row["labels"][focal] >= 0:
                    references[word.lower()].append((hidden[b, focal], row["labels"][focal],
                                                      sentence_id * key_width + focal))
    references = {word: (torch.stack([r[0] for r in rows]).to(cfg.runtime.device),
                         torch.tensor([r[1] for r in rows], device=cfg.runtime.device),
                         torch.tensor([r[2] for r in rows], device=cfg.runtime.device))
                  for word, rows in references.items()}
    split = cfg.masks.split
    # Round-robin by form and label keeps the pilot from selecting only a majority reading.
    groups = defaultdict(list)
    for row in data[split]:
        for focal, (word, label) in enumerate(zip(row["words"], row["labels"])):
            if word.lower() in forms and label >= 0:
                groups[word.lower(), label].append((row["sentence_id"], focal))
    groups = list(groups.values())
    targets = [group[i] for i in range(max(map(len, groups))) for group in groups if i < len(group)]
    if cfg.masks.focals is not None:
        targets = targets[:cfg.masks.focals]
    records, details = defaultdict(dict), []
    for sentence_id, focal in tqdm(targets, desc="Minimum context masks"):
        row = data[split][sentence_id]
        batch = to_device(next(iter(loader(data[split].select([sentence_id]), cfg, tokenizer))), cfg.runtime.device)
        n = len(row["words"])
        context = torch.arange(n, device=cfg.runtime.device)
        context = context[context != focal]
        subsets = candidates(n - 1, cfg, cfg.runtime.device)
        gates = torch.ones(len(subsets), n, device=cfg.runtime.device)
        gates[:, context] = subsets.float()
        hidden = []
        chunk = min(cfg.runtime.focal_chunk, max(1, cfg.runtime.attention_budget // len(row["ids"]) ** 2))
        for start in range(0, len(gates), chunk):
            part = gates[start:start + chunk]
            attention = expand_gates(part, batch["word_ids"].expand(len(part), -1),
                                     batch["attention_mask"].expand(len(part), -1))
            with autocast(cfg):
                h = encode(batch["ids"].expand(len(part), -1), attention, all_layers=False)["features"]
            hidden.append(h[:, batch["positions"][0, focal], -1].float())
        ref, labels, keys = references[row["words"][focal].lower()]
        keep = keys != sentence_id * key_width + focal if split == "train" else torch.ones_like(keys, dtype=torch.bool)
        supported, probability, margin = geometry(torch.cat(hidden), ref[keep], labels[keep], row["labels"][focal], cfg)
        sizes = subsets.sum(-1)
        full = sizes.argmax()
        info = {"sentence_id": sentence_id, "focal": focal, "word": row["words"][focal],
                "label": row["labels"][focal], "full_supported": bool(supported[full]),
                "self_supported": bool(supported[sizes.argmin()]),
                "exhaustive": n - 1 <= cfg.masks.exhaustive_context,
                "full_probability": float(probability[full]), "full_margin": float(margin[full])}
        if supported[full]:
            minimum = sizes.masked_fill(~supported, n + 1).min()
            minima = subsets[supported & (sizes == minimum)]
            mask = {"representative": minima[0], "union": minima.any(0),
                    "intersection": minima.all(0)}[cfg.masks.policy]
            records[sentence_id][focal] = context[mask].tolist()
            info.update(minimum=int(minimum), minima=len(minima), context=context[mask].tolist())
        details.append(info)
    summary = {"targets": len(details), "supported": sum(row["full_supported"] for row in details),
               "exhaustive": sum(row["exhaustive"] for row in details), "split": split,
               "oracle": True}
    (output / "masks.jsonl").write_text("".join(json.dumps(row) + "\n" for row in details))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return {split: dict(records)}, summary
