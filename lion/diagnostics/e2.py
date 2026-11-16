"""Compare held-out word-only predictions with full-training lexical label counts."""
import torch
from torch.nn import functional as F

from .common import VIEWS, locations, plot_table, seed_summary, write_csv
from .data import banks
from .readouts import feature_loader, load_y_model, load_y_predictions, predict
from .run import cli


def ece(probability, targets, n_bins):
    confidence, prediction = probability.max(-1)
    groups = (confidence * n_bins).long().clamp_max(n_bins - 1)
    n = torch.bincount(groups, minlength=n_bins)
    correct = torch.zeros(n_bins).index_add_(0, groups, (prediction == targets).float())
    certainty = torch.zeros(n_bins).index_add_(0, groups, confidence)
    value = ((correct - certainty).abs() / len(targets)).sum()
    return float(value)


def experiment(cfg, output):
    scores, types, skips = [], [], []
    for name, bank in banks(cfg):
        if name == "ner_shared":
            continue
        test, stats = bank["data"]["test"], bank["stats"]
        counts = stats["counts"]
        eligible = (counts.sum(-1) >= cfg.diagnostics.min_type_frequency) & ((counts > 0).sum(-1) > 1)
        words = [word for word, index in stats["vocab"].items() if eligible[index]]
        observed = sorted(set(test["words"]) & set(words))
        if not observed:
            skips.append({"dataset": name, "reason": "no eligible ambiguous training type in held-out sample"})
            continue
        ids = {word: i for i, word in enumerate(observed)}
        inverse = torch.tensor([ids.get(word, -1) for word in test["words"]])
        mask = inverse >= 0
        group, target = inverse[mask], test["y"][mask]
        support = torch.bincount(group, minlength=len(observed))
        empirical = counts[torch.tensor([stats["vocab"][word] for word in observed])].float()
        empirical /= empirical.sum(-1, keepdim=True)
        for seed in cfg.diagnostics.seeds:
            matched = load_y_predictions(cfg, name, "word_only", seed)["test"]
            model = load_y_model(cfg, name, "full", seed)
            shifted = predict(model, feature_loader(test, 1, cfg), len(test["y"]), len(bank["labels"]), cfg)
            for condition, logits in (("trained_word_only", matched), ("full_on_word_only", shifted)):
                probability = logits.softmax(-1)[mask]
                average = torch.zeros_like(empirical).index_add_(0, group, probability) / support[:, None]
                kl = (torch.special.xlogy(empirical, empirical) - empirical * average.clamp_min(1e-9).log()).sum(-1)
                # Average occurrence-level KL distinguishes a mean match from variable outputs across positions/lengths.
                per_occurrence = (torch.special.xlogy(empirical[group], empirical[group]) -
                                  empirical[group] * probability.clamp_min(1e-9).log()).sum(-1)
                scores.append({"dataset": name, "seed": seed, "view": condition, "types": len(observed),
                               "tokens": len(target), "type_mean_kl": float(kl.mean()),
                               "token_weighted_kl": float((kl * support).sum() / support.sum()),
                               "occurrence_kl": float(per_occurrence.mean()),
                               "ece": ece(probability, target, cfg.diagnostics.ece_bins),
                               "brier": float((probability - F.one_hot(target, len(bank["labels"]))).square().sum(-1).mean())})
                for index, word in enumerate(observed):
                    types.append({"dataset": name, "seed": seed, "view": condition, "word": word,
                                  "test_count": int(support[index]),
                                  "train_count": int(counts[stats["vocab"][word]].sum()),
                                  "kl": float(kl[index]), "empirical": empirical[index].tolist(),
                                  "prediction": average[index].tolist()})
    write_csv(output / "scores.csv", scores)
    write_csv(output / "types.csv", types)
    write_csv(output / "illustrations.csv", sorted(types, key=lambda row: (-row["test_count"], row["word"]))[:36])
    write_csv(output / "skips.csv", skips)
    write_csv(output / "summary.csv", seed_summary(scores, ("dataset", "view"), ("type_mean_kl", "occurrence_kl", "ece")))
    plot_table(output / "kl.png", scores, "type_mean_kl")
    return {"conditions": len(scores)}


if __name__ == "__main__":
    cli("e2")
