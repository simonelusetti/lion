"""X1 source-only shift curves and simultaneous finite-window/full contrasts."""
import torch

from ..diagnostics.common import write_csv
from .data import load_banks, load_features, model_data
from .stats import bootstrap, interval, label_metrics
from .train import fit


def experiment(cfg, output):
    curves, contrasts, skips = [], [], []
    for name, bank in load_banks(cfg):
        all_names = bank["conditions"]
        selected = [i for i, s in enumerate(all_names) if not s.startswith("rand_") and s != "G"]
        names = [all_names[i] for i in selected]
        feature_parts = load_features(cfg, bank)
        data = model_data(bank, {s: {"features": p["features"][:, selected]} for s, p in feature_parts.items()})
        inverse, weights = bootstrap(bank["data"]["test"], cfg)
        seed_curves = []
        for seed in cfg.diagnostics.seeds:
            inside, outside = [], []
            for domain, domain_name in enumerate(bank["z_names"]):
                masks = {split: part["z"] == domain for split, part in bank["data"].items()}
                if any(int(mask.sum()) < cfg.diagnostics.domain_min[split] for split, mask in masks.items()):
                    skips.append({"dataset": name, "source": domain_name, "reason": "insufficient native source split support"})
                    continue
                local = {split: {key: value[masks[split]] if split != "test" else value for key, value in part.items()}
                         for split, part in data.items()}
                _, predicted = fit(local, len(bank["labels"]), cfg, seed, output / name / f"seed{seed}" / domain_name)
                for target in range(len(bank["z_names"])):
                    mask = bank["data"]["test"]["z"] == target
                    if int(mask.sum()) < cfg.diagnostics.domain_min.test: continue
                    measured = label_metrics(predicted["test"][mask], data["test"]["y"][mask], inverse[mask], weights)
                    (inside if target == domain else outside).append(measured["accuracy"])
            seed_curves.append({"in_domain": torch.stack(inside).mean(0), "out_of_domain": torch.stack(outside).mean(0)})
        saved = {kind: torch.stack([row[kind] for row in seed_curves]) for kind in ("in_domain", "out_of_domain")}
        word, full = names.index("lin0"), names.index("full")
        for kind, accuracy in saved.items():
            gain = accuracy - accuracy[:, :, word:word + 1]
            for i, condition in enumerate(names):
                curves.append({"dataset": name, "evaluation": kind, "condition": condition,
                               **{"accuracy_" + k: v for k, v in interval(accuracy[:, :, i]).items()},
                               **{"gain_" + k: v for k, v in interval(gain[:, :, i]).items()}})
        for family, prefix in (("linear", "lin"), ("tree", "tree")):
            finite = [i for i, condition in enumerate(names) if condition.startswith(prefix) and condition != "lin0"]
            if not finite: continue
            difference = saved["out_of_domain"][:, :, finite] - saved["out_of_domain"][:, :, full:full + 1]
            point = difference[:, 0].mean(0)
            boot = difference[:, 1:].mean(0)
            critical = torch.quantile((boot - point).abs().amax(-1), .95)
            peak = int(point.argmax())
            contrasts.append({"dataset": name, "family": family, "peak_condition": names[finite[peak]],
                              "peak_minus_full": float(point[peak]), "simultaneous_ci_low": float(point[peak] - critical),
                              "simultaneous_ci_high": float(point[peak] + critical),
                              "supports_finite_peak": bool(point[peak] - critical > 0),
                              "sources": len(bank["z_names"]), "bootstrap_replicates": cfg.partition.bootstrap})
        torch.save({"curves": saved, "conditions": names}, output / name / "curves.pt")
    write_csv(output / "curves.csv", curves)
    write_csv(output / "peaks.csv", contrasts)
    write_csv(output / "skips.csv", skips)
    return {"curves": len(curves)}
