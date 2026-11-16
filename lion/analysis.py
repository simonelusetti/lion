"""Layerwise evidence, gold-label oracles, and independent-seed controls."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import torch

from .metrics import scores


def word_metadata(data, num_labels):
    counts = defaultdict(Counter)
    for row in data["train"]:
        for word, label in zip(row["words"], row["labels"]):
            if label >= 0:
                counts[word.lower()][label] += 1
    words = [word.lower() for row in data["test"] for word, label in zip(row["words"], row["labels"])
             if label >= 0]
    return {"words": words, "num_labels": num_labels,
            "train_label_counts": {word: dict(labels) for word, labels in counts.items()}}


def form_summary(source, losses, layers, output, run_id):
    metadata = torch.load(source / "words.pt", weights_only=False)
    forms = sorted(set(metadata["words"]))
    index = {word: i for i, word in enumerate(forms)}
    inverse = torch.tensor([index[word] for word in metadata["words"]])
    n = torch.bincount(inverse, minlength=len(forms))
    average = torch.zeros(len(forms), len(layers)).index_add_(0, inverse, losses) / n[:, None]
    best = average.argmin(-1)
    counts = torch.tensor([[metadata["train_label_counts"].get(word, {}).get(label, 0)
                            for label in range(metadata["num_labels"])] for word in forms], dtype=torch.float32)
    train_n = counts.sum(-1)
    probability = counts / train_n[:, None]
    entropy = -torch.special.xlogy(probability, probability).sum(-1)
    with (output / f"forms_{run_id}.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["word", "test_count", "train_count", "train_label_entropy", "best_layer"])
        writer.writeheader()
        for i, word in enumerate(forms):
            writer.writerow({"word": word, "test_count": int(n[i]), "train_count": int(train_n[i]),
                             "train_label_entropy": float(entropy[i]),
                             "best_layer": layers[int(best[i])]})


def analyze(sources, output):
    runs = []
    for source in sources:
        source = Path(source)
        paths = sorted((source / "layers").glob("layer*/predictions.pt"))
        data = [torch.load(path, weights_only=False) for path in paths]
        # Verify alignment only at the saved-artifact boundary.
        for row in data[1:]:
            if not all(torch.equal(row[key], data[0][key]) for key in ("targets", "sentence_ids", "positions")):
                raise ValueError(f"Layer predictions refer to different tokens in {source}")
        layers = [int(path.parent.name.removeprefix("layer")) for path in paths]
        logits = torch.stack([row["logits"] for row in data], dim=1)
        gold = data[0]["targets"]
        loss = -logits.log_softmax(-1).gather(-1, gold[:, None, None].expand(-1, len(layers), 1)).squeeze(-1)
        gold_logit = logits.gather(-1, gold[:, None, None].expand(-1, len(layers), 1)).squeeze(-1)
        competitors = logits.clone().scatter(-1, gold[:, None, None].expand(-1, len(layers), 1), -torch.inf)
        margins = gold_logit - competitors.amax(-1)
        best = loss.argmin(-1)
        oracle = logits[torch.arange(len(gold)), best]
        per_layer = []
        for column, layer in enumerate(layers):
            result = scores(logits[:, column], gold, data[0]["counts"], data[0]["labels"], data[0]["dataset"])
            per_layer.append({"layer": layer, "macro_f1": result["macro_f1"],
                              "accuracy": result["accuracy"], "cross_entropy": float(loss[:, column].mean()),
                              "margin": float(margins[:, column].mean())})
        summary = {"source": str(source), "per_layer": per_layer,
                   "best_layer_counts": torch.bincount(best, minlength=len(layers)).tolist(),
                   "oracle": scores(oracle, gold, data[0]["counts"], data[0]["labels"], data[0]["dataset"]),
                   "oracle_uses_gold_labels": True}
        form_summary(source, loss, layers, output, len(runs))
        runs.append((layers, logits, loss, best, data[0], summary, margins))
    cross_seed = []
    for a, first in enumerate(runs):
        for b, second in enumerate(runs):
            if a != b:
                if first[0] != second[0] or not all(torch.equal(first[4][key], second[4][key])
                        for key in ("targets", "sentence_ids", "positions")):
                    raise ValueError("Independent-seed runs must cover identical layers and tokens")
                prediction = second[1][torch.arange(len(second[1])), first[3]]
                d = second[4]
                cross_seed.append({"select_run": a, "evaluate_run": b,
                                   **scores(prediction, d["targets"], d["counts"], d["labels"], d["dataset"])})
    report = {"runs": [row[5] for row in runs], "cross_seed_oracles": cross_seed,
              "note": "Cross-seed controls still use gold labels to choose the layer; these are oracle ceilings."}
    (output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "layers.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source", "layer", "macro_f1", "accuracy", "cross_entropy", "margin"])
        writer.writeheader()
        writer.writerows({"source": row[5]["source"], **layer} for row in runs for layer in row[5]["per_layer"])
    torch.save([{ "layers": row[0], "loss": row[2], "margins": row[6], "best_layer": torch.tensor(row[0])[row[3]],
                  "sentence_ids": row[4]["sentence_ids"], "positions": row[4]["positions"]} for row in runs],
               output / "per_token.pt")
    return {"oracle_macro_f1": runs[0][5]["oracle"]["macro_f1"]}
