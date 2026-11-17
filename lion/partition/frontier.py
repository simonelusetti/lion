"""X0/X1: dedicated and amortized readouts, conditional nuisance, fixed comparisons."""
import torch

from ..diagnostics.common import write_csv
from .data import load_banks, load_features, model_data, paths, pooled
from .probes import conditional
from .stats import bootstrap, conditional_accuracy, interval, label_metrics, weighted_scalar
from .train import fit


def families(bank):
    names = bank["conditions"]
    result = {"linear": [i for i, name in enumerate(names) if name.startswith("lin")],
              "random": [names.index("lin0")] + [i for i, name in enumerate(names) if name.startswith("rand_")]}
    if any(name.startswith("tree") for name in names):
        result["tree"] = [names.index("lin0")] + [i for i, name in enumerate(names) if name.startswith("tree")]
    for key in result: result[key].append(names.index("full"))
    return result


def crossing(accuracy, nuisance, indices, fraction, word, full):
    target = accuracy[word] + fraction * (accuracy[full] - accuracy[word])
    if accuracy[full] <= accuracy[word]: return torch.tensor(float("nan"))
    for a, b in zip(indices[:-1], indices[1:]):
        if accuracy[a] <= target <= accuracy[b] and accuracy[b] > accuracy[a]:
            weight = (target - accuracy[a]) / (accuracy[b] - accuracy[a])
            return nuisance[a] + weight * (nuisance[b] - nuisance[a])
    return torch.tensor(float("nan"))


def experiment(cfg, output):
    all_scores, nuisance_rows, amortized_rows, contrasts, knees = [], [], [], [], []
    for name, bank in load_banks(cfg):
        feature_parts = load_features(cfg, bank)
        data = model_data(bank, feature_parts)
        names = bank["conditions"]
        full, word, floor = (names.index(key) for key in ("full", "lin0", "G"))
        inverse, weights = bootstrap(bank["pool"], cfg)
        offset = len(data["train"]["y"]) + len(data["validation"]["y"])
        test_inverse = inverse[offset:]
        seed_y, seed_n, seed_mix = [], [], []
        counts = pooled(feature_parts, "geometry")[:, :, 2].mean(0)
        family_indices = families(bank)
        family_indices["random"] = sorted(family_indices["random"], key=lambda i: (float(counts[i]), i))
        for seed in cfg.diagnostics.seeds:
            folder = output / name / f"seed{seed}"
            _, predictions = fit(data, len(bank["labels"]), cfg, seed, folder / "dedicated")
            _, mixture = fit(data, len(bank["labels"]), cfg, seed, folder / "amortized", mixture=True)
            y_metrics = label_metrics(predictions["test"], data["test"]["y"], test_inverse, weights)
            mixed_metrics = label_metrics(mixture["test"], data["test"]["y"], test_inverse, weights)
            p, q = predictions["test"].log_softmax(-1), mixture["test"].log_softmax(-1)
            kl = (p.exp() * (p - q)).sum(-1)
            mix_stats = {"accuracy_loss": y_metrics["accuracy"] - mixed_metrics["accuracy"],
                         "f1_loss": y_metrics["macro_f1"] - mixed_metrics["macro_f1"],
                         "kl": weighted_scalar(kl, test_inverse, weights)}
            seed_y.append(y_metrics)
            seed_mix.append(mix_stats)
            features = {"representation": pooled(feature_parts),
                        "predictor": torch.cat([p.log_softmax(-1) for p in predictions.values()]),
                        "geometry": pooled(feature_parts, "geometry")}
            nuisance = {}
            for kind, x in features.items():
                predicted, _ = conditional(x, bank, cfg, seed, folder / kind, standardize=kind == "geometry")
                nuisance[kind] = conditional_accuracy(predicted, bank["pool"], bank["eligible"], inverse, weights)
            seed_n.append(nuisance)
        y_stats = {key: torch.stack([s[key] for s in seed_y]) for key in seed_y[0]}
        n_stats = {key: torch.stack([s[key] for s in seed_n]) for key in seed_n[0]}
        for condition, label in enumerate(names):
            row = {"dataset": name, "condition": label, "mean_context_words": float(counts[condition])}
            for metric, values in y_stats.items(): row.update({metric + "_" + k: v for k, v in interval(values[:, :, condition]).items()})
            all_scores.append(row)
            nr = {"dataset": name, "condition": label}
            for kind, values in n_stats.items():
                excess = values[:, :, condition] - values[:, :, floor]
                nr.update({kind + "_" + k: v for k, v in interval(excess).items()})
                nr[kind + "_floor_ba"] = float(values[:, 0, floor].mean())
            nuisance_rows.append(nr)
            mr = {"dataset": name, "condition": label}
            for metric in seed_mix[0]:
                values = torch.stack([s[metric][:, condition] for s in seed_mix])
                mr.update({metric + "_" + k: v for k, v in interval(values).items()})
            mr["within_tolerance"] = (mr["accuracy_loss_ci_high"] <= cfg.partition.amortized_accuracy_tolerance and
                                      mr["f1_loss_ci_high"] <= cfg.partition.amortized_f1_tolerance and
                                      mr["kl_ci_high"] <= cfg.partition.amortized_kl_tolerance)
            amortized_rows.append(mr)
        excess = n_stats["predictor"] - n_stats["predictor"][:, :, floor:floor + 1]
        matches = {}
        for family, indices in family_indices.items():
            matches[family] = torch.stack([torch.stack([crossing(y_stats["accuracy"][s, b], excess[s, b], indices,
                                                                  cfg.partition.gain_fraction, word, full)
                                                       for b in range(weights.shape[0])]) for s in range(len(seed_y))])
            contrasts.append({"dataset": name, "comparison": "matched_90pct", "family": family,
                              **interval(matches[family])})
            for condition in indices[1:-1]:
                recovered = y_stats["accuracy"][:, :, condition] >= (y_stats["accuracy"][:, :, word] +
                            cfg.partition.gain_fraction * (y_stats["accuracy"][:, :, full] - y_stats["accuracy"][:, :, word]))
                valid = (y_stats["accuracy"][:, :, full] > y_stats["accuracy"][:, :, word]) & (excess[:, :, full] > 0)
                small = excess[:, :, condition] <= cfg.partition.nuisance_fraction * excess[:, :, full]
                meets = torch.where(valid, (recovered & small).float(), torch.nan)
                knees.append({"dataset": name, "family": family, "condition": names[condition],
                              **interval(meets)})
        for a, b in (("linear", "random"), ("tree", "random"), ("tree", "linear")):
            if a in matches and b in matches:
                contrasts.append({"dataset": name, "comparison": a + "_minus_" + b,
                                  "family": "paired", **interval(matches[a] - matches[b])})
        torch.save({"Y": y_stats, "Z": n_stats, "counts": counts, "conditions": names,
                    "families": family_indices, "bootstrap_weights": weights}, output / name / "frontier.pt")
    write_csv(output / "label_scores.csv", all_scores)
    write_csv(output / "nuisance.csv", nuisance_rows)
    write_csv(output / "amortized.csv", amortized_rows)
    write_csv(output / "matched.csv", contrasts)
    write_csv(output / "knees.csv", knees)
    return {"conditions": len(all_scores)}
