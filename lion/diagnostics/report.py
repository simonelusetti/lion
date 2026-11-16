"""Write a synthesis with measurements and interpretations explicitly separated."""
from collections import defaultdict
import json

import torch

from .common import locations, read_csv, seed_summary, write_csv
from .data import load_inventory
from .run import cli


DESIGNS = {
    "e1": ("Three views", "Token classification on identical sampled focal positions; full-training lexical statistics define bins.",
           "Differences between separately trained readouts measure the predictive value of these views. "
           "They do not identify spurious context or a causal effect of nuisance Z."),
    "e2": ("Masking and lexical distributions", "KL from empirical training label distributions to held-out mean predictions, occurrence KL, ECE and Brier score.",
           "A lower KL for the mask-trained classifier is consistent with adaptation to masking. "
           "It does not establish P(Y|X_S): finite sampled training, model fit, position and mask length "
           "can all affect the result. Unseen or unsupported ambiguous types are not evaluated."),
    "e3": ("Linear nuisance readability", "Class-balanced linear Z probes trained independently within each label; chance is 1 / eligible domain count.",
           "Above-chance balanced accuracy is evidence that a linear readout can recover source/genre "
           "conditional on the recorded label. It does not establish that the representation encodes a "
           "calibrated posterior over latent state and nuisance. Sparse labels/domains are excluded."),
    "e4": ("Readout erasure", "A training-only affine erasure removes covariance with within-label centered domain indicators; fresh Y MLPs and Z probes evaluate the result.",
           "Y loss is interpretable as a separability diagnostic only together with the post-erasure Z "
           "probe. Remaining readability makes the result inconclusive about complete erasure. A large "
           "Y loss alone cannot distinguish causal entanglement from loss of useful correlated signal, "
           "limited classifier fit, or erasure estimation error."),
    "e5": ("Within-label dependence", "Unbiased RBF MMD of prediction vectors, document-block permutation p values, per-seed BH q values, and lexical JS; stratified by lexical and predictive entropy.",
           "A small q value is evidence of distributional dependence under the block-exchangeability "
           "assumption, not a causal diagnosis of Assumption A. Dataset-identity tests use sentence "
           "blocks and are exploratory. With 199 permutations, resolution is 0.005; sparse cells and "
           "multiple comparisons limit power. Non-rejection is inconclusive, not evidence of invariance. "
           "Predictive entropy strata are properties of each fitted model, not measurements of latent ambiguity."),
    "e6": ("Cross-domain shift", "Each readout sees only one source domain's training/validation examples; test scores and per-label degradation cover source and eligible target domains.",
           "The ordering of degradation across views is the observable diagnostic. Domains differ in "
           "label frequencies, lexical coverage, and (for NER corpora) annotation rules. E5 correlations "
           "are descriptive and use a pooled-data classifier; they do not establish that its measured "
           "dependence causes the source-only model's degradation."),
    "e7": ("Mention substitutions", "Training-only same-type, per-word WordPiece-length-matched replacements of a nearby entity or the focal mention; prediction flips and probability L1 changes are paired.",
           "Neighbor-induced flips show sensitivity to a controlled co-occurring mention. Calling the "
           "change a nuisance-only intervention additionally requires semantic and label preservation, "
           "which this experiment does not prove. Focal substitution changes entity identity; multiword "
           "focal mentions can still change visible context under context-only masking. Neighbor "
           "substitution should leave word-only input identical under the length/alignment controls."),
}


def paired_contrasts(rows, keys, group, metric, a, b, claim):
    cells = defaultdict(dict)
    for row in rows:
        cells[tuple(row[key] for key in keys)][row[group]] = row
    contrasts = []
    for key, cell in cells.items():
        if a in cell and b in cell:
            contrasts.append({"dataset": cell[a]["dataset"], "seed": int(cell[a]["seed"]),
                              "claim": claim, "value": float(cell[a][metric]) - float(cell[b][metric])})
    # Labels/domain pairs are averaged within a seed before reporting seed spread.
    grouped = defaultdict(list)
    for row in contrasts:
        grouped[(row["dataset"], row["seed"], row["claim"])].append(row["value"])
    return [{"dataset": key[0], "seed": key[1], "claim": key[2],
             "value": float(torch.tensor(values).mean()), "cells": len(values)} for key, values in grouped.items()]


def contrasts(experiment_name, rows, folder):
    base = ("dataset", "seed")
    if experiment_name == "e1":
        result = paired_contrasts(rows, base, "view", "accuracy", "full", "word_only", "full_minus_word_accuracy")
        gains = read_csv(folder / "gains.csv")
        entropy = [row for row in gains if row["bin_kind"] == "word_entropy"]
        frequency = [row for row in gains if row["bin_kind"] == "frequency"]
        result += paired_contrasts(entropy, base, "bin", "accuracy_gain", "high", "zero", "ambiguous_minus_unambiguous_context_gain")
        result += paired_contrasts(frequency, base, "bin", "accuracy_gain", "1-5", "21+", "rare_minus_frequent_context_gain")
        return result
    if experiment_name == "e2":
        return paired_contrasts(rows, base, "view", "type_mean_kl", "full_on_word_only", "trained_word_only", "shifted_minus_matched_KL")
    if experiment_name == "e3":
        result = paired_contrasts(rows, (*base, "label", "domains"), "view", "above_chance",
                                  "context_only", "word_only", "context_minus_word_Z_readability")
        grouped = defaultdict(list)
        for row in rows:
            if row["view"] == "context_only":
                grouped[(row["dataset"], int(row["seed"]))].append(float(row["above_chance"]))
        result += [{"dataset": key[0], "seed": key[1], "claim": "context_Z_above_chance",
                    "value": float(torch.tensor(values).mean()), "cells": len(values)} for key, values in grouped.items()]
        return result
    if experiment_name == "e4":
        result = [{"dataset": row["dataset"], "seed": int(row["seed"]), "claim": "Y_F1_drop",
                   "value": float(row["f1_drop"]), "cells": 1} for row in rows]
        grouped = defaultdict(list)
        for row in read_csv(folder / "z_probes.csv"):
            grouped[(row["dataset"], int(row["seed"]))].append(float(row["above_chance"]))
        result += [{"dataset": key[0], "seed": key[1], "claim": "remaining_Z_above_chance",
                    "value": float(torch.tensor(values).mean()), "cells": len(values)} for key, values in grouped.items()]
        return result
    if experiment_name == "e5":
        keys = (*base, "label", "domain_a", "domain_b")
        all_rows = [row for row in rows if row["stratum_kind"] == "all"]
        result = paired_contrasts(all_rows, keys, "view", "mmd_squared", "full", "word_only", "full_minus_word_MMD")
        result += paired_contrasts(all_rows, keys, "view", "mmd_squared", "context_only", "word_only", "context_minus_word_MMD")
        entropy_rows = [row for row in rows if row["stratum_kind"] == "word_entropy"]
        for view in ("full", "word_only", "context_only"):
            result += paired_contrasts([row for row in entropy_rows if row["view"] == view], keys,
                                       "stratum", "mmd_squared", "high", "zero", view + "_ambiguous_minus_unambiguous_MMD")
        return result
    if experiment_name == "e6":
        rows = [row for row in rows if row["source"] != row["target"]]
        keys = (*base, "source", "target")
        return (paired_contrasts(rows, keys, "view", "accuracy_drop", "context_only", "full", "context_minus_full_shift_drop") +
                paired_contrasts(rows, keys, "view", "accuracy_drop", "full", "word_only", "full_minus_word_shift_drop"))
    if experiment_name == "e7":
        rows = [row for row in rows if row["intervention"] == "neighbor" and row["focal_length"] == "all"]
        return paired_contrasts(rows, base, "view", "flip_rate", "full", "word_only", "full_minus_word_neighbor_flip_rate")
    return []


def directional_assessment(name, summary):
    if name == "e4":
        return ("**Claim assessment: inconclusive without successful Z removal.** The table quantifies Y loss "
                "and remaining conditional readability. Small Y loss together with chance-level Z probes "
                "is consistent with separability; larger Y loss is not by itself proof of entanglement.")
    if not summary:
        return "**Claim assessment: inconclusive.** No paired contrast meets the available support."
    positive = sum(row["seeds"] == 3 and row["value_min"] > 0 for row in summary)
    negative = sum(row["seeds"] == 3 and row["value_max"] < 0 for row in summary)
    mixed = len(summary) - positive - negative
    return (f"**Directional claim assessment (descriptive):** {positive} dataset/claim contrasts have "
            f"the predicted positive sign in all three seeds; {negative} have the opposite sign in all "
            f"three seeds; {mixed} are mixed, tied, or incomplete. Consistent positive signs support "
            "the stated directional prediction on this fixed sample; opposite signs contradict that "
            "prediction on this sample. These counts are not significance tests or causal conclusions.")


def table(rows, limit=24):
    if not rows:
        return "No completed measurements or no eligible cells. See the experiment's skip table.\n"
    fields = list(rows[0])
    # Keep the report readable; the CSVs retain SD, min, max and all cells.
    fields = [key for key in fields if not key.endswith(("_min", "_max"))]
    lines = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    for row in rows[:limit]:
        values = []
        for field in fields:
            value = str(row.get(field, ""))
            try:
                number = float(value)
                value = f"{number:.4f}" if not number.is_integer() else str(int(number))
            except ValueError:
                pass
            values.append(value.replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    if len(rows) > limit:
        lines += ["", f"Showing {limit} of {len(rows)} rows; see the linked CSV for the complete table."]
    return "\n".join(lines) + "\n"


def experiment(cfg, output):
    results, _ = locations(cfg)
    inventory = load_inventory(cfg)
    lines = ["# Over-contextualization diagnostics", "", "Protocol and baseline inventory: [plan.md](plan.md). "
             "Sampling caps are a computational budget, not the full-data baseline training population. "
             "Diagnostic readouts are refitted on identical focal samples, with three classifier seeds. "
             "The focal sample is uniform over sentences and changes token weighting relative to the "
             "full corpus. "
             "Seed spread describes optimization variation conditional on one fixed sample, not sampling "
             "uncertainty or confidence intervals.", "", "## Availability in the frozen diagnostic inventory", "",
             "This records availability when this suite was launched. Later baseline repairs do not "
             "expand these diagnostic samples or change their recorded results.", ""]
    for row in inventory["datasets"]:
        if not row["available"]:
            lines.append(f"- {row['name']}: unavailable token cache; baseline statuses: " +
                         ", ".join(run["status"] for run in row["baselines"]) + ". Skipped throughout.")
    lines += ["", "Genre Z is recovered from raw cached UD sentence IDs, rather than invented. "
              "Other native tasks have no retained usable genre/source variable; the shared NER "
              "comparison uses dataset identity only. [inventory.json](inventory.json) records "
              "native document overlap and exact label sets. Metadata alignment is checked against "
              "cached token IDs before encoding.", "", "## Actual sampling", ""]
    sample_path = results / "samples.csv"
    lines.append(table(read_csv(sample_path)) if sample_path.exists() else "Pending representation extraction.\n")
    completed = 0
    for experiment_name, (title, measurement, interpretation) in DESIGNS.items():
        folder = results / experiment_name
        score_path = folder / "scores.csv"
        lines += [f"## {experiment_name.upper()}: {title}", "", "**Measurement.** " + measurement, ""]
        if score_path.exists():
            scores = read_csv(score_path)
            native = inventory["ready"]
            nuisance = [row["name"] for row in inventory["datasets"] if row["available"] and row["z"]]
            shared = ["ner_shared"] if len(inventory["ner_sources"]) >= 2 else []
            expected = {"e1": native + shared, "e2": native, "e7": inventory["ner_sources"]}.get(
                experiment_name, nuisance + shared)
            measured_datasets = sorted({row["dataset"] for row in scores})
            missing_datasets = sorted(set(expected) - set(measured_datasets))
            lines += ["Measured datasets: " + ", ".join(measured_datasets) + ".", ""]
            if missing_datasets:
                lines += ["**No eligible measurements for:** " + ", ".join(missing_datasets) + ". "
                          "These datasets are omitted from this experiment's numerical conclusions.", ""]
                if experiment_name == "e5":
                    manifest = json.loads((results / "banks.json").read_text())
                    for name in missing_datasets:
                        bank = torch.load(manifest[name], map_location="cpu", weights_only=False)
                        test = bank["data"]["test"]
                        document_counts = {domain: len({doc for z, doc in zip(test["z"].tolist(), test["document"])
                                                        if z == index})
                                           for index, domain in enumerate(bank["z_names"])}
                        if max(document_counts.values(), default=0) < cfg.diagnostics.mmd.min_documents:
                            lines += [f"{name}: test domains contain at most {max(document_counts.values(), default=0)} "
                                      f"documents each; the permutation protocol requires {cfg.diagnostics.mmd.min_documents}. "
                                      "Increasing focal tokens cannot increase independent document support.", ""]
            summary = folder / "summary.csv"
            lines += [f"Recorded {len(scores)} seed-level conditions. "
                      f"[Scores]({experiment_name}/scores.csv), "
                      f"[seed summaries]({experiment_name}/summary.csv).", "",
                      table(read_csv(summary)) if summary.exists() else table(scores)]
            paired = contrasts(experiment_name, scores, folder)
            claim_summary = seed_summary(paired, ("dataset", "claim"), ("value",))
            write_csv(folder / "contrasts.csv", paired)
            write_csv(folder / "contrast_summary.csv", claim_summary)
            lines += ["**Measured directional contrasts.** Positive values follow the directional "
                      "prediction named in each contrast (E4 instead records magnitudes without a preferred sign).", "",
                      table(claim_summary), directional_assessment(experiment_name, claim_summary), ""]
            completed += 1
            if experiment_name == "e1":
                lines += ["Frequency/ambiguity concentration: [gains](e1/gain_summary.csv); "
                          "[per-label scores](e1/per_label.csv).", ""]
            if experiment_name == "e4":
                lines += ["[Post-erasure Z probes](e4/z_summary.csv) must be considered with the Y drop. "
                          "[Projection ranks and covariance residuals](e4/erasure.csv).", ""]
            if experiment_name == "e6":
                lines += ["[Per-label degradation](e6/per_label.csv); "
                          "[correlation with E5](e6/e5_correlations.csv).", ""]
            figures = sorted(folder.glob("*.png"))
            for path in figures:
                lines += [f"![{title}]({experiment_name}/{path.name})", ""]
            lines += ["**Interpretation.** " + interpretation, "",
                      "These tables provide diagnostic evidence; support or contradiction of the claimed "
                      "ordering must be assessed across datasets, seed spread, and reported support counts. "
                      "A universal causal conclusion is inconclusive from these observational diagnostics alone.", ""]
        else:
            lines += ["**Status: pending.** No measurements are claimed.", "",
                      "**Interpretation: pending/inconclusive.** " + interpretation, ""]
        if (folder / "skips.csv").exists():
            skipped = read_csv(folder / "skips.csv")
            lines += [f"Skipped cells: {len(skipped)}. [Reasons and supports]({experiment_name}/skips.csv).", ""]
    lines += ["## Scope of the evidence", "", "No selectors or new encoder training are introduced. "
              "Baseline code/checkpoints are unchanged. PNG error bars are sample standard deviations "
              "over three seed-level means (averaging eligible label/domain cells within a seed where "
              "applicable). CSVs preserve the individual cells. Inadequate support is reported as missing "
              "measurement; no missing result is treated as support for the framework.", ""]
    (results / "report.md").write_text("\n".join(lines))
    return {"completed_experiments": completed}


if __name__ == "__main__":
    cli("report")
