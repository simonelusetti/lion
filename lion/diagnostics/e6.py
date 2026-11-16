"""Source-only readouts evaluated in domain and on every eligible target domain."""
import math

import torch

from .common import VIEWS, locations, plot_table, read_csv, seed_summary, subset, write_csv
from .data import banks
from .readouts import fit, measurements
from .run import cli


def experiment(cfg, output):
    results, _ = locations(cfg)
    dependence = read_csv(results / "e5" / "scores.csv") if (results / "e5" / "scores.csv").exists() else []
    lookup = {(row["dataset"], int(row["seed"]), row["view"], row["label"],
               tuple(sorted((row["domain_a"], row["domain_b"])))): float(row["mmd_squared"])
              for row in dependence if row["stratum_kind"] == "all"}
    rows, labels, skips, correlations = [], [], [], []
    for name, bank in banks(cfg, nuisance=True):
        data, domains = bank["data"], bank["z_names"]
        for source, source_name in enumerate(domains):
            counts = {split: int((part["z"] == source).sum()) for split, part in data.items()}
            if any(counts[split] < cfg.diagnostics.domain_min[split] for split in counts):
                skips.append({"dataset": name, "source": source_name, "reason": "insufficient source split support", **counts})
                continue
            local = {split: subset(part, (part["z"] == source).nonzero().flatten())
                     if split != "test" else part for split, part in data.items()}
            train_counts = torch.bincount(local["train"]["y"], minlength=len(bank["labels"]))
            for seed in cfg.diagnostics.seeds:
                for view, view_name in enumerate(VIEWS):
                    predicted = fit(local, bank["labels"], view, cfg, seed,
                                    output / name / source_name / view_name / f"seed{seed}")["test"]
                    source_mask = data["test"]["z"] == source
                    reference, source_labels = measurements(predicted[source_mask], data["test"]["y"][source_mask], bank["labels"])
                    for target, target_name in enumerate(domains):
                        mask = data["test"]["z"] == target
                        if int(mask.sum()) < cfg.diagnostics.domain_min.test:
                            skips.append({"dataset": name, "source": source_name, "target": target_name,
                                          "reason": "insufficient target test support"})
                            continue
                        measured, target_labels = measurements(predicted[mask], data["test"]["y"][mask], bank["labels"])
                        prefix = {"dataset": name, "seed": seed, "view": view_name,
                                  "source": source_name, "target": target_name}
                        rows.append({**prefix, **measured, "source_test_accuracy": reference["accuracy"],
                                     "source_test_macro_f1": reference["macro_f1"],
                                     "accuracy_drop": reference["accuracy"] - measured["accuracy"],
                                     "f1_drop": reference["macro_f1"] - measured["macro_f1"],
                                     "unseen_label_test_tokens": int((train_counts[data["test"]["y"][mask]] == 0).sum())})
                        related = []
                        for index, (a, b) in enumerate(zip(source_labels, target_labels)):
                            key = (name, seed, view_name, a["label"], tuple(sorted((source_name, target_name))))
                            mmd = lookup.get(key)
                            recall_drop = a["recall"] - b["recall"] if a["support"] and b["support"] else None
                            labels.append({**prefix, "label": a["label"], "train_support": int(train_counts[index]),
                                           "source_test_support": a["support"], "target_test_support": b["support"],
                                           "source_recall": a["recall"], "target_recall": b["recall"],
                                           "recall_drop": recall_drop, "target_f1": b["f1"], "e5_mmd_squared": mmd})
                            if mmd is not None and recall_drop is not None:
                                related.append((mmd, recall_drop))
                        if len(related) >= 3:
                            values = torch.tensor(related).T
                            if (values.std(-1) > 0).all():
                                correlations.append({**prefix, "labels": len(related),
                                                     "pearson_mmd_recall_drop": float(torch.corrcoef(values)[0, 1])})
    write_csv(output / "scores.csv", rows)
    write_csv(output / "per_label.csv", labels)
    write_csv(output / "e5_correlations.csv", correlations)
    write_csv(output / "skips.csv", skips)
    write_csv(output / "summary.csv", seed_summary(rows, ("dataset", "view", "source", "target"),
                                                    ("accuracy", "macro_f1", "accuracy_drop", "f1_drop")))
    plot_table(output / "degradation.png", [row for row in rows if row["source"] != row["target"]], "accuracy_drop")
    return {"source_target_conditions": len(rows), "skipped_cells": len(skips)}


if __name__ == "__main__":
    cli("e6")
