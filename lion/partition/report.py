"""Compact X0–X4 report and the requested figures, using saved measurements only."""
import math
from pathlib import Path
import shutil

import torch

from ..diagnostics.common import ROOT, read_csv
from .data import load_banks, paths


def number(value, scale=1):
    value = float(value)
    return f"{scale * value:.3f}" if math.isfinite(value) else "undefined"


def estimate(row, prefix="", scale=1):
    return (number(row[prefix + "mean"], scale) + " ± " + number(row[prefix + "seed_sd"], scale) +
            " [" + number(row[prefix + "ci_low"], scale) + ", " + number(row[prefix + "ci_high"], scale) + "]")


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers),
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]]) + "\n"


def figures(cfg, root, destination):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    for name, bank in load_banks(cfg):
        source = root / "frontier" / name / "frontier.pt"
        if source.exists():
            data = torch.load(source, weights_only=False)
            x = data["Y"]["accuracy"][:, 0].mean(0)
            z = data["Z"]["predictor"]
            y = (z - z[:, :, bank["conditions"].index("G"):bank["conditions"].index("G") + 1])[:, 0].mean(0)
            fig, ax = plt.subplots(figsize=(7, 5))
            for family, ids in data["families"].items():
                ax.plot(x[ids].numpy(), y[ids].numpy(), "o-", label=family)
                for i in ids[1:-1]:
                    label = bank["conditions"][i]
                    label = f"m={float(data['counts'][i]):.1f}" if family == "random" else label
                    ax.annotate(label, (float(x[i]), float(y[i])), fontsize=7, xytext=(3, 3), textcoords="offset points")
            ax.axhline(0, color="grey", lw=.7)
            ax.set(xlabel="Y test accuracy", ylabel="Predictor Z balanced accuracy minus G", title=name)
            ax.legend(); fig.tight_layout(); fig.savefig(destination / f"{name}_frontier.png", dpi=150); plt.close(fig)
        curve_path = root / "shift" / "curves.csv"
        if curve_path.exists():
            rows = [r for r in read_csv(curve_path) if r["dataset"] == name]
            families = ["lin"] + (["tree"] if any(r["condition"].startswith("tree") for r in rows) else [])
            fig, axes = plt.subplots(1, len(families), figsize=(6 * len(families), 4), squeeze=False)
            for ax, family in zip(axes[0], families):
                labels = (["lin0"] if family == "tree" else []) + [c for c in bank["conditions"] if c.startswith(family)] + ["full"]
                for kind in ("in_domain", "out_of_domain"):
                    chosen = [next(r for r in rows if r["condition"] == c and r["evaluation"] == kind) for c in labels]
                    xx = list(range(len(labels))); mean = [100 * float(r["gain_mean"]) for r in chosen]
                    ax.plot(xx, mean, "o-", label=kind)
                    ax.fill_between(xx, [100 * float(r["gain_ci_low"]) for r in chosen], [100 * float(r["gain_ci_high"]) for r in chosen], alpha=.15)
                ax.set_xticks(range(len(labels)), labels, rotation=45)
                ax.axhline(0, color="grey", lw=.7); ax.set(title=name + " " + family, ylabel="Window minus word-only accuracy (pp)"); ax.legend()
            fig.tight_layout(); fig.savefig(destination / f"{name}_shift.png", dpi=150); plt.close(fig)
        strata_path = root / "attribution" / "strata.csv"
        if strata_path.exists():
            rows = [r for r in read_csv(strata_path) if r["dataset"] == name]
            distances = ["linear"] + (["tree"] if name != "ner_shared" else [])
            fig, axes = plt.subplots(1, len(distances), figsize=(6 * len(distances), 4), squeeze=False)
            for ax, distance in zip(axes[0], distances):
                chosen = sorted([r for r in rows if r["predictor"] == distance and math.isfinite(float(r["value"]))], key=lambda r: float(r["value"]))
                for response in ("r", "n"):
                    valid = [r for r in chosen if r.get(response + "_mean", "") != ""]
                    xx = [float(r["value"]) for r in valid]
                    ax.plot(xx, [float(r[response + "_mean"]) for r in valid], label=response)
                    ax.fill_between(xx, [float(r[response + "_ci_low"]) for r in valid], [float(r[response + "_ci_high"]) for r in valid], alpha=.15)
                ax.set(xlabel=distance + " distance", ylabel="Drop in true-label log probability", title=name); ax.legend()
            fig.tight_layout(); fig.savefig(destination / f"{name}_distance.png", dpi=150); plt.close(fig)
        attribution = root / "attribution" / name / "attribution.pt"
        if attribution.exists():
            data = torch.load(attribution, weights_only=False)
            valid = torch.tensor([r["kind"] == "word" for r in data["records"]]) & torch.isfinite(data["n"]).all(0)
            x, y = data["r"][:, valid].mean(0), data["n"][:, valid].mean(0)
            ids = torch.randperm(len(x), generator=torch.Generator().manual_seed(1729))[:5000]
            fig, ax = plt.subplots(figsize=(5, 5)); ax.scatter(x[ids], y[ids], s=5, alpha=.2)
            ax.set(xlabel="r: true Y log-probability drop", ylabel="n: true Z log-probability drop", title=name)
            fig.tight_layout(); fig.savefig(destination / f"{name}_scatter.png", dpi=150); plt.close(fig)
        label_path = root / "labels" / "gains.csv"
        if label_path.exists():
            rows = [r for r in read_csv(label_path) if r["dataset"] == name]
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            for ax, metric, scale in zip(axes, ("ba_gain", "ce_gain"), (100, 1)):
                xx = [int(r["k"]) for r in rows]
                ax.plot(xx, [scale * float(r[metric + "_mean"]) for r in rows], "o-")
                ax.fill_between(xx, [scale * float(r[metric + "_ci_low"]) for r in rows], [scale * float(r[metric + "_ci_high"]) for r in rows], alpha=.2)
                ax.set(xlabel="Gold-label window k", ylabel="B minus A BA gain (pp)" if scale == 100 else "A minus B cross-entropy", title=name)
            fig.tight_layout(); fig.savefig(destination / f"{name}_label_context.png", dpi=150); plt.close(fig)


def experiment(cfg, output):
    root, _ = paths(cfg)
    report = ROOT / "report" / "old" / "02_partitions_x0_x4"
    tables = report / "tables"
    destination = report / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    lines = ["# Context partitions: X0–X4 results", "",
             "Implementation and fixed analysis choices: [IMPLEMENTATION_2.md](IMPLEMENTATION_2.md). "
             "Entries show mean ± classifier-seed SD [95% document-block bootstrap interval]. "
             "Seeds are 0/1/2; bootstrap intervals condition on the fitted models. "
             "Only the three X1 comparisons are confirmatory; the other measurements are descriptive. "
             "This is not a test of causal identification.", ""]
    for stage in ("frontier", "shift", "attribution", "labels", "clusters"):
        for source in (root / stage).glob("*.csv"):
            shutil.copy2(source, tables / (stage + "_" + source.name))
    def read(stage, name):
        path = root / stage / name
        return read_csv(path) if path.exists() else []
    mix = read("frontier", "amortized.csv")
    lines += ["## X0 — amortized readout", ""]
    if mix:
        rows = []
        for name in cfg.partition.datasets:
            chosen = [r for r in mix if r["dataset"] == name]
            rows.append([name, sum(r["within_tolerance"] == "True" for r in chosen), len(chosen)])
        lines += [table(["Dataset", "Conditions within all tolerances", "Conditions"], rows),
                  "Operational tolerances are upper bootstrap bounds ≤1 pp for accuracy/F1 loss and ≤0.02 nats KL. "
                  "Passing some conditions does not justify replacing all dedicated readouts. "
                  "All subsequent analyses retain dedicated readouts.", ""]
    else: lines += ["Pending or failed stage; no equivalence result is claimed.", ""]
    matched = read("frontier", "matched.csv")
    lines += ["## X1 — confirmatory comparisons", "", "### Matched label information", ""]
    lines += [table(["Dataset", "Paired comparison", "Nuisance difference (pp)", "Assessment"],
                    [[r["dataset"], r["comparison"], estimate(r, scale=100),
                      "favours " + ("S over L" if r["comparison"] == "tree_minus_linear" else "window locality (L/S)")
                      if float(r["ci_high"]) < 0 else "opposite direction" if float(r["ci_low"]) > 0 else "inconclusive"]
                     for r in matched if r["comparison"] != "matched_90pct"]),
              "Negative differences favour the first family. Undefined values occur when full context has "
              "no positive accuracy gain over word-only, or an interpolation is not defined. Overlapping "
              "intervals do not establish N or equivalence. Interpolation uses the first crossing in the "
              "prespecified family order; random order is mean retained count.", ""]
    knees = read("frontier", "knees.csv")
    lines += ["### Knees", "", table(["Dataset", "Family", "Conditions meeting the knee in all three seeds"],
              [[name, family, ", ".join(r["condition"] for r in knees if r["dataset"] == name and r["family"] == family and r["mean"] == "1.0") or "none / undefined"]
               for name in cfg.partition.datasets for family in ("linear", "tree", "random")
               if any(r["dataset"] == name and r["family"] == family for r in knees)]),
              "The knee requires ≥90% of full-minus-word-only label gain and ≤50% of the positive full-minus-G "
              "predictor-nuisance excess. Seed agreement is a point estimate; stability intervals and undefined "
              "cases remain in the CSV. A window knee is consistent with locality, not sufficient to distinguish L from S.", ""]
    peaks = read("shift", "peaks.csv")
    lines += ["### Domain shift", "", table(["Dataset", "Family", "Empirical finite peak", "Peak minus full accuracy (pp)", "Assessment"],
              [[r["dataset"], r["family"], r["peak_condition"], number(r["peak_minus_full"],100) + " [" + number(r["simultaneous_ci_low"],100) + ", " + number(r["simultaneous_ci_high"],100) + "]",
                "supports finite-window advantage" if r["supports_finite_peak"] == "True" else "inconclusive / no finite-window advantage"] for r in peaks]),
              "Intervals use the maximum bootstrap deviation across finite radii within each family. "
              "They account for choosing the displayed peak within that family, but are not simultaneous "
              "across datasets or all three primary questions. Source/target curves average ordered domain "
              "pairs equally. Small GUM native test document counts still limit Y shift inference.", ""]
    association = read("attribution", "association.csv")
    partial = read("attribution", "partial_r2.csv")
    lines += ["## X2 — token and band attribution (descriptive)", "",
              table(["Dataset", "Quantity", "Estimate"], [[r["dataset"],r["metric"],estimate(r)] for r in association]),
              table(["Dataset", "Word-class minus distance partial R² for nuisance", "Assessment"],
                    [[r["dataset"],estimate(r),"consistent with K" if float(r["ci_low"])>0 else "distance stronger" if float(r["ci_high"])<0 else "inconclusive"]
                     for r in partial if r["response"] == "n" and r["block"] == "word_class_minus_distance"]),
              "The nuisance score uses the conditional representation Z probe, not the predictor-log-probability "
              "probe. Positive r/n means removal reduces true-label confidence; negative scores are retained. "
              "Overlap above the exact random-set Jaccard baseline and positive r–n association are consistent "
              "with shared token channels (N), but cannot establish that no other partition exists. "
              "Bands and matched-count random removals address redundancy; their complete effects are in "
              "the band CSV. All partial-effect coefficients, group partial R², support and missing-n counts "
              "are linked below. Shared NER lacks gold trees/UPOS, so its regression omits those predictors.", ""]
    trend = read("labels", "trend.csv")
    lines += ["## X3 — gold label context (descriptive)", "",
              table(["Dataset", "BA gain at k=2 minus gain at k=0 (pp)", "Assessment"],
                    [[r["dataset"],estimate(r,scale=100),"consistent with A" if float(r["ci_high"])<0 else "opposite trend" if float(r["ci_low"])>0 else "inconclusive"]
                     for r in trend if r["metric"] == "ba_gain_k2_minus_k0"]),
              "A shrinking gain is consistent with label context absorbing information in the word-only proxy. "
              "The probes are linear on position-specific one-hot labels; failure to shrink does not rule out "
              "nonlinear label interactions. The word-only proxy also carries lexical genre information. "
              "Absolute A/B scores and cross-entropy gains are in the table CSV.", ""]
    clusters = read("clusters", "scores.csv")
    lines += ["## X4 — metadata-free clustering (descriptive)", "",
              table(["Dataset", "Visible channel", "K", "NMI", "Matched Z BA (%)"],
                    [[r["dataset"],r["channel"],r["K"],estimate(r,"NMI_"),estimate(r,"balanced_accuracy_",100)]
                     for r in clusters if r["label"] == "all"]),
              "Clustering never uses labels; cluster-to-domain matching uses training-fold Z only. "
              "Scores use held-out documents, and NMI is computed within fold before count-weighted "
              "aggregation because cluster IDs differ between folds. Per-label scores are in the same CSV. "
              "Nonempty inside/outside pooling is required for both channels, preserving paired coverage.", "",
              "## Protocol differences and limits", "",
              "- GUM UPOS reuses precisely the GUM discourse focal positions; it does not enlarge that sample.",
              "- Identical word-only/full endpoints are canonical shared conditions; random controls match each focal's window count.",
              "- Gold UD heads/UPOS are used. No offline parser is installed, so shared NER uses only linear geometry and sentence blocks; its document-level claims are exploratory.",
              "- Z probes use five document folds on pooled native samples, with a disjoint validation fold for checkpoint selection. X4 uses four training folds; Y readouts keep native splits.",
              "- X2 uses three fixed independent sparse-occlusion draws per focal, sampled uniformly during training, rather than re-encoding new masks every epoch.",
              "- The geometry probe uses WordPiece sentence length, first-WordPiece focal position and retained context-word count. Only those scalar inputs are standardized using training-fold statistics.",
              "- X3 uses linear probes; PAD and unlabelled neighbors are separate one-hot categories. Best matching in X4 is training-only balanced Hungarian matching with capacity ceil(clusters/domains) per domain.",
              "- Bootstrap intervals use 200 domain-stratified document-block resamples of predictions, not refits of every classifier; regressions are refitted from resampled document sufficient statistics. Undefined bootstrap cells are omitted and valid counts are recorded.",
              "- X0 tolerances, first-crossing interpolation, nonpositive-gain exclusions, knee thresholds and simultaneous finite-radius shift bands were fixed before the run. These do not constitute a familywise causal test.",
              "", "## Tables and figures", "",
              "All aggregate CSVs: [tables](tables/). Figures: [figures](figures/). "
              "Per-model checkpoints, fold predictions, raw paired attribution tensors and cached representations "
              "remain under results/partition and data/cache/partition; no old E1–E7 result is overwritten.", ""]
    figures(cfg, root, destination)
    (report / "RESULTS_2.md").write_text("\n".join(lines))
    return {"primary_matched_contrasts":len(matched),"shift_peak_comparisons":len(peaks)}
