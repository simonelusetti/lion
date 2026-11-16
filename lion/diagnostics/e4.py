"""Closed-form conditional-domain covariance erasure and fresh readouts."""
import torch
from torch.nn import functional as F

from .common import VIEWS, plot_table, seed_summary, write_csv
from .data import banks
from .readouts import fit, load_y_predictions, measurements, probe_all
from .run import cli


def learn_erasure(train, labels, domains, rtol):
    x = train["features"][:, 0].double()
    mean = x.mean(0)
    centered = x - mean
    y, z = train["y"], train["z"]
    joint = torch.bincount(y * domains + z, minlength=labels * domains).reshape(labels, domains).double()
    conditional = joint / joint.sum(-1, keepdim=True).clamp_min(1)
    residual = F.one_hot(z, domains).double() - conditional[y]
    concept = (F.one_hot(y, labels).double()[:, :, None] * residual[:, None, :]).flatten(1)
    covariance = centered.T @ centered / len(x)
    cross = centered.T @ concept / len(x)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    active = eigenvalues > eigenvalues.max() * rtol
    basis, values = eigenvectors[:, active], eigenvalues[active]
    whitener = (basis * values.rsqrt()) @ basis.T
    unwhitener = (basis * values.sqrt()) @ basis.T
    u, singular, _ = torch.linalg.svd(whitener @ cross, full_matrices=False)
    rank = int((singular > singular.max() * rtol).sum()) if singular.numel() and singular.max() > 0 else 0
    directions = u[:, :rank]
    projection = torch.eye(x.shape[-1], dtype=torch.float64) - unwhitener @ directions @ directions.T @ whitener
    erased_cross = projection @ cross
    return mean, projection, {"removed_rank": rank, "feature_rank": int(active.sum()),
                              "train_cross_covariance_before": float(cross.norm()),
                              "train_cross_covariance_after": float(erased_cross.norm())}


def experiment(cfg, output):
    scores, probes, skips, erasures = [], [], [], []
    for name, bank in banks(cfg, nuisance=True):
        mean, projection, info = learn_erasure(bank["data"]["train"], len(bank["labels"]),
                                                len(bank["z_names"]), cfg.diagnostics.erasure_rtol)
        erasures.append({"dataset": name, **info})
        folder = output / name
        folder.mkdir(parents=True, exist_ok=True)
        torch.save({"mean": mean, "projection": projection, "method": "within-label residual concept LEACE",
                    "uses_test_labels": False}, folder / "erasure.pt")
        erased = {**bank, "data": {}}
        for split, rows in bank["data"].items():
            features = rows["features"].float().clone()
            # A single affine map; inference does not consult Y or Z.
            features[:, 0] = ((features[:, 0].double() - mean) @ projection.T + mean).float()
            erased["data"][split] = {**rows, "features": features}
        for seed in cfg.diagnostics.seeds:
            predictions = fit(erased["data"], bank["labels"], 0, cfg, seed, folder / "y" / f"seed{seed}")
            result, _ = measurements(predictions["test"], bank["data"]["test"]["y"], bank["labels"])
            baseline, _ = measurements(load_y_predictions(cfg, name, "full", seed)["test"],
                                       bank["data"]["test"]["y"], bank["labels"])
            scores.append({"dataset": name, "seed": seed, "view": "erased_full", **result,
                           "original_macro_f1": baseline["macro_f1"],
                           "f1_drop": baseline["macro_f1"] - result["macro_f1"],
                           "accuracy_drop": baseline["accuracy"] - result["accuracy"]})
        post, skipped = probe_all(name, erased, cfg, folder / "post_probes", views=(0,))
        probes.extend({**row, "view": "erased_full"} for row in post)
        skips.extend(skipped)
    write_csv(output / "scores.csv", scores)
    write_csv(output / "z_probes.csv", probes)
    write_csv(output / "erasure.csv", erasures)
    write_csv(output / "skips.csv", skips)
    write_csv(output / "summary.csv", seed_summary(scores, ("dataset", "view"), ("macro_f1", "f1_drop", "accuracy_drop")))
    write_csv(output / "z_summary.csv", seed_summary(probes, ("dataset", "label", "view"),
                                                      ("balanced_accuracy", "above_chance")))
    plot_table(output / "label_drop.png", scores, "f1_drop")
    return {"erased_datasets": len(erasures), "post_erasure_probes": len(probes)}


if __name__ == "__main__":
    cli("e4")
