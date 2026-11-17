"""X3: does gold label context absorb the word-only representation's Z gain?"""
import torch
from torch.nn import functional as F

from ..diagnostics.common import write_csv
from .data import load_banks
from .probes import global_probe
from .stats import bootstrap, interval, label_metrics


def experiment(cfg, output):
    scores, trends = [], []
    for name, bank in load_banks(cfg):
        labels = len(bank["labels"])
        examples = [(row, focal) for rows in bank["rows"].values() for row in rows for focal in row["focals"]]
        inverse, weights = bootstrap(bank["pool"], cfg)
        target = bank["pool"]["z"]
        by_k = {}
        for k in (0, 1, 2):
            encoded = []
            for row, focal in examples:
                context = [row["labels"][i] if 0 <= i < len(row["labels"]) else labels for i in range(focal - k, focal + k + 1)]
                context = [value if value >= 0 else labels + 1 for value in context]
                encoded.append(F.one_hot(torch.tensor(context), labels + 2).flatten())
            a = torch.stack(encoded).float()[:, None]
            b = torch.cat([a, bank["pool"]["word_only"].float()[:, None]], dim=-1)
            seed_gains = []
            for seed in cfg.diagnostics.seeds:
                predictions = [global_probe(x, bank, cfg, seed, output / name / f"k{k}" / f"seed{seed}" / condition)
                               for x, condition in ((a, "A"), (b, "B"))]
                measures = [label_metrics(p, target, inverse, weights) for p in predictions]
                # Balanced accuracy uses per-domain recall, not ordinary token accuracy.
                ba = []
                for prediction in predictions:
                    recalls = []
                    for domain in range(len(bank["z_names"])):
                        mask = target == domain
                        recalls.append(label_metrics(prediction[mask], target[mask], inverse[mask], weights)["accuracy"])
                    ba.append(torch.stack(recalls).mean(0))
                seed_gains.append({"A": ba[0][:, 0], "B": ba[1][:, 0], "ba_gain": (ba[1] - ba[0])[:, 0],
                                   "ce_gain": (measures[0]["cross_entropy"] - measures[1]["cross_entropy"])[:, 0]})
            record = {"dataset": name, "k": k}
            for key in seed_gains[0]:
                record.update({key + "_" + field: value for field, value in interval(torch.stack([s[key] for s in seed_gains])).items()})
            scores.append(record)
            by_k[k] = {key: torch.stack([s[key] for s in seed_gains]) for key in seed_gains[0]}
        for metric in ("ba_gain", "ce_gain"):
            trends.append({"dataset": name, "metric": metric + "_k2_minus_k0",
                           **interval(by_k[2][metric] - by_k[0][metric])})
    write_csv(output / "gains.csv", scores)
    write_csv(output / "trend.csv", trends)
    return {"window_conditions": len(scores)}
