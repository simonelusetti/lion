import csv
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]
VIEWS = ("full", "word_only", "context_only")


def absolute(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def locations(cfg):
    return absolute(cfg.diagnostics.results), absolute(cfg.diagnostics.cache)


def dataset_config(cfg, name):
    data = OmegaConf.load(ROOT / "conf" / "data" / f"{name}.yaml")
    return OmegaConf.merge(cfg, {"data": data})


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def subset(data, indices):
    indices = torch.as_tensor(indices, dtype=torch.long)
    return {key: value[indices] if isinstance(value, torch.Tensor) else
            [value[i] for i in indices.tolist()] for key, value in data.items()}


def entropy(probabilities):
    return -torch.special.xlogy(probabilities, probabilities).sum(-1)


def bins(data, kind):
    if kind == "frequency":
        n = data["frequency"]
        return {"unseen": n == 0, "1-5": (n >= 1) & (n <= 5),
                "6-20": (n >= 6) & (n <= 20), "21+": n >= 21}
    h = data["word_entropy"]
    return {"unseen": torch.isnan(h), "zero": h == 0,
            "low": (h > 0) & (h <= 0.6931472), "high": h > 0.6931472}


def plot_table(path, rows, metric, group="view"):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    keys = sorted({(row["dataset"], row[group]) for row in rows if metric in row})
    fig, ax = plt.subplots(figsize=(max(7, len(keys) * 0.45), 4))
    for i, key in enumerate(keys):
        cells = [row for row in rows if (row["dataset"], row[group]) == key and metric in row]
        values = torch.tensor([torch.tensor([float(row[metric]) for row in cells if row["seed"] == seed]).mean()
                               for seed in sorted({row["seed"] for row in cells})])
        ax.bar(i, values.mean().item(), yerr=values.std(unbiased=len(values) > 1).item(), capsize=3)
    ax.set_xticks(range(len(keys)), [" / ".join(key) for key in keys], rotation=70, ha="right")
    ax.set_ylabel(metric)
    if not keys:
        ax.text(0.5, 0.5, "No eligible observations", ha="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def seed_summary(rows, keys, metrics):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    summary = []
    for key, group in groups.items():
        item = dict(zip(keys, key))
        item["seeds"] = len(group)
        for metric in metrics:
            values = torch.tensor([row[metric] for row in group], dtype=torch.float64)
            item.update({f"{metric}_mean": float(values.mean()),
                         f"{metric}_std": float(values.std(unbiased=len(values) > 1)),
                         f"{metric}_min": float(values.min()), f"{metric}_max": float(values.max())})
        summary.append(item)
    return summary
