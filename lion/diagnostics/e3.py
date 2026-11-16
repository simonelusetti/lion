"""Conditional, balanced linear probes for nuisance Z."""
from .common import VIEWS, plot_table, seed_summary, write_csv
from .data import banks
from .readouts import probe_all
from .run import cli


def experiment(cfg, output):
    rows, skips = [], []
    for name, bank in banks(cfg, nuisance=True):
        scores, skipped = probe_all(name, bank, cfg, output)
        rows.extend({**row, "view": VIEWS[row["view"]]} for row in scores)
        skips.extend(skipped)
    write_csv(output / "scores.csv", rows)
    write_csv(output / "skips.csv", skips)
    write_csv(output / "summary.csv", seed_summary(rows, ("dataset", "label", "view"),
                                                    ("balanced_accuracy", "above_chance")))
    plot_table(output / "readability.png", rows, "above_chance")
    return {"probes": len(rows), "skipped_cells": len(skips)}


if __name__ == "__main__":
    cli("e3")
