"""Three independently trained views on identical focal samples."""
import torch

from .common import VIEWS, bins, plot_table, seed_summary, write_csv
from .data import banks
from .readouts import fit, measurements
from .run import cli


def experiment(cfg, output):
    scores, per_label, gains = [], [], []
    for name, bank in banks(cfg):
        for seed in cfg.diagnostics.seeds:
            predictions = []
            for view, view_name in enumerate(VIEWS):
                predicted = fit(bank["data"], bank["labels"], view, cfg, seed,
                                output / name / view_name / f"seed{seed}")
                logits = predicted["test"]
                predictions.append(logits)
                overall, labels = measurements(logits, bank["data"]["test"]["y"], bank["labels"])
                prefix = {"dataset": name, "seed": seed, "view": view_name}
                scores.append({**prefix, **overall})
                per_label.extend({**prefix, **row} for row in labels)
            test = bank["data"]["test"]
            target = test["y"]
            correct = [p.argmax(-1) == target for p in predictions]
            losses = [torch.nn.functional.cross_entropy(p, target, reduction="none") for p in predictions]
            per_token = {"full_minus_word_accuracy": correct[0].float() - correct[1].float(),
                         "word_minus_full_ce": losses[1] - losses[0],
                         "target": target, "sentence": test["sentence"], "position": test["position"],
                         "frequency": test["frequency"], "word_entropy": test["word_entropy"]}
            torch.save(per_token, output / name / f"gains_seed{seed}.pt")
            for kind in ("frequency", "word_entropy"):
                for label, mask in bins(test, kind).items():
                    if mask.any():
                        gains.append({"dataset": name, "seed": seed, "bin_kind": kind, "bin": label,
                                      "n": int(mask.sum()),
                                      "accuracy_gain": float(per_token["full_minus_word_accuracy"][mask].mean()),
                                      "ce_gain": float(per_token["word_minus_full_ce"][mask].mean())})
    write_csv(output / "scores.csv", scores)
    write_csv(output / "per_label.csv", per_label)
    write_csv(output / "gains.csv", gains)
    write_csv(output / "summary.csv", seed_summary(scores, ("dataset", "view"), ("accuracy", "macro_f1")))
    write_csv(output / "gain_summary.csv", seed_summary(gains, ("dataset", "bin_kind", "bin"), ("accuracy_gain", "ce_gain")))
    plot_table(output / "scores.png", scores, "macro_f1")
    return {"trained_classifiers": len(scores)}


if __name__ == "__main__":
    cli("e1")
