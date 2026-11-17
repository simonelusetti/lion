"""Reuse the old focal manifests; attach gold UD structure and encode kept sets."""
from collections import defaultdict
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
import pyarrow.parquet as pq
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from ..data import collate, signature, to_device
from ..datasets import UD_REPO
from ..diagnostics.common import ROOT, absolute, dataset_config, write_csv
from ..diagnostics.data import shared_mapping, token_path
from ..encoder import build_encoder
from ..labels import LABELS
from ..runtime import autocast, parallel


def paths(cfg):
    key = signature(OmegaConf.create({"protocol": "partition-v1", "encoder": cfg.encoder,
                                      "linear": cfg.partition.linear, "tree": cfg.partition.tree,
                                      "occlusion": cfg.partition.occlusion_probability,
                                      "draws": cfg.partition.occlusion_draws,
                                      "attribution_focals": cfg.partition.attribution_focals,
                                      "mask_seed": cfg.diagnostics.sample_seed,
                                      "amp": cfg.runtime.amp, "tf32": cfg.runtime.tf32, "dtype": cfg.cache.dtype,
                                      "sources": json.loads(absolute(cfg.partition.sources).read_text())}))
    return absolute(cfg.partition.results), absolute(cfg.partition.cache) / key


def conditions(cfg, tree):
    names = [f"lin{k}" for k in cfg.partition.linear]
    if tree:
        names += [f"tree{k}" for k in cfg.partition.tree]
    return names + ["rand_" + name for name in names if name != "lin0"] + ["full", "G"]


def ud_raw(tree):
    return {split: pq.read_table(hf_hub_download(UD_REPO, f"parquet/{tree}/{source}.parquet",
                                               repo_type="dataset", local_files_only=True),
                                 columns=["tokens", "head", "upos", "sent_id"]).to_pylist()
            for split, source in (("train", "train"), ("validation", "dev"), ("test", "test"))}


def distances(heads, focal, kept):
    n = len(heads)
    parents = torch.tensor(heads)
    if (parents < 0).any() or (parents > n).any():
        raise ValueError("Gold UD heads are outside sentence bounds")
    adjacency = torch.zeros(n, n)
    child = (parents > 0).nonzero().flatten()
    adjacency[child, parents[child] - 1] = 1
    adjacency = adjacency + adjacency.T
    frontier = torch.zeros(len(focal), n, dtype=torch.bool)
    frontier[torch.arange(len(focal)), torch.tensor(focal)] = True
    visited = frontier.clone()
    distance = torch.full((len(focal), n), float("inf"))
    distance[frontier] = 0
    depth = 0
    while frontier.any():
        depth += 1
        frontier = (frontier.float() @ adjacency > 0) & ~visited
        distance[frontier] = depth
        visited |= frontier
    return distance[:, kept]


def folds(data, count, seed):
    names = sorted(set(data["document"]))
    domain = {doc: z for doc, z in zip(data["document"], data["z"].tolist())}
    generator = torch.Generator().manual_seed(seed)
    assignment = {}
    for z in sorted(set(domain.values())):
        group = [name for name in names if domain[name] == z]
        for i, position in enumerate(torch.randperm(len(group), generator=generator).tolist()):
            assignment[group[position]] = i % count
    return torch.tensor([assignment[name] for name in data["document"]])


def eligible_domains(pool, fold, labels, domains, cfg):
    eligible = torch.ones(labels, domains, dtype=torch.bool)
    for test in range(cfg.partition.folds):
        for split, mask in (("test", fold == test), ("validation", fold == (test + 1) % cfg.partition.folds),
                            ("train", (fold != test) & (fold != (test + 1) % cfg.partition.folds))):
            counts = torch.bincount(pool["y"][mask] * domains + pool["z"][mask], minlength=labels * domains).reshape(labels, domains)
            eligible &= counts >= cfg.diagnostics.probe_min[split]
    eligible[eligible.sum(-1) < 2] = False
    return eligible


def attach_row(row, raw, tokenizer, cfg):
    encoded = tokenizer(raw["tokens"], is_split_into_words=True, truncation=True,
                        max_length=cfg.encoder.max_length, add_special_tokens=cfg.encoder.special_tokens)
    if encoded["input_ids"] != row["ids"]:
        raise ValueError("Gold UD structure does not align with frozen focal sample")
    word_ids = encoded.word_ids()
    kept = [word for i, word in enumerate(word_ids) if word is not None and (i == 0 or word != word_ids[i - 1])]
    result = dict(row)
    result["upos"] = [raw["upos"][i] for i in kept]
    result["tree"] = distances(raw["head"], [kept[i] for i in row["focals"]], kept)
    return result


def prepare(cfg, output):
    original = json.loads(absolute(cfg.partition.sources).read_text())
    tokenizer = AutoTokenizer.from_pretrained(cfg.encoder.name, use_fast=True, local_files_only=True)
    results, cache = paths(cfg)
    results.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    manifest, counts = {}, []
    raw_ewt, raw_gum = ud_raw("en_ewt"), ud_raw("en_gum")
    for name in cfg.partition.datasets:
        source_name = "ud_discourse" if name == "gum_upos" else name
        old = torch.load(original[source_name], map_location="cpu", weights_only=False)
        rows = {}
        if name == "ner_shared":
            labels = old["labels"]
            for split in old["data"]:
                rows[split] = []
                for domain, native in enumerate(old["z_names"]):
                    native_bank = torch.load(original[native], weights_only=False)
                    mapping = shared_mapping(native)
                    for row in native_bank["rows"][split]:
                        item = dict(row)
                        item["labels"] = [int(mapping[label]) if label >= 0 else -100 for label in row["labels"]]
                        item["focals"] = [f for f in row["focals"] if item["labels"][f] >= 0]
                        if item["focals"]:
                            item["sentence_id"] += domain * 10**8
                            item["upos"] = [None] * len(row["words"])
                            item["tree"] = None
                            rows[split].append(item)
        else:
            labels = LABELS["ud_upos"] if name == "gum_upos" else old["labels"]
            raw = raw_gum if name in {"ud_discourse", "gum_upos"} else raw_ewt
            for split, selected in old["rows"].items():
                rows[split] = [attach_row(row, raw[split][row["sentence_id"]], tokenizer, cfg) for row in selected]
                if name == "gum_upos":
                    for row in rows[split]:
                        row["labels"] = [labels.index(tag) for tag in row["upos"]]
        parts = {}
        for split, selected in rows.items():
            y = torch.tensor([row["labels"][focal] for row in selected for focal in row["focals"]])
            old_part = old["data"][split]
            ids = [(row["sentence_id"], f) for row in selected for f in row["focals"]]
            prior_ids = list(zip(old_part["sentence"].tolist(), old_part["position"].tolist()))
            if ids != prior_ids:
                raise ValueError(f"Follow-up focal sample changed: {name}/{split}")
            parts[split] = {"y": y, "z": old_part["z"], "sentence": old_part["sentence"],
                            "position": old_part["position"], "document": old_part["document"],
                            "words": old_part["words"], "word_only": old_part["features"][:, 1]}
            counts.append({"dataset": name, "split": split, "sentences": len(selected), "focals": len(y)})
        pool = {key: torch.cat([p[key] for p in parts.values()]) if isinstance(parts["train"][key], torch.Tensor)
                else [v for part in parts.values() for v in part[key]] for key in parts["train"]}
        pool["sentence"] = torch.cat([part["sentence"] + i * 10**9 for i, part in enumerate(parts.values())])
        fold = folds(pool, cfg.partition.folds, cfg.diagnostics.sample_seed)
        eligible = eligible_domains(pool, fold, len(labels), len(old["z_names"]), cfg)
        folder = cache / name
        folder.mkdir(exist_ok=True)
        bank = {"data": parts, "pool": pool, "rows": rows, "fold": fold, "eligible": eligible,
                "labels": labels, "z_names": old["z_names"], "conditions": conditions(cfg, name != "ner_shared"),
                "geometry_unit": "sentence for shared NER; original UD document otherwise",
                "feature_source": {"ud_deprel": "ud_upos", "gum_upos": "ud_discourse"}.get(name, name)}
        torch.save(bank, folder / "bank.pt")
        manifest[name] = str(folder / "bank.pt")
        print(f"Prepared {name}: {len(pool['y'])} pooled focals, {int((eligible.sum(-1)>=2).sum())} probe labels", flush=True)
    (results / "banks.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, path in manifest.items():
        bank = torch.load(path, weights_only=False)
        source = torch.load(manifest[bank["feature_source"]], weights_only=False)
        if not all(torch.equal(bank["pool"][key], source["pool"][key]) for key in ("sentence", "position")):
            raise ValueError(f"Shared encoder feature source has different focal positions: {name}")
    write_csv(results / "samples.csv", counts)
    return {"datasets": len(manifest)}


def load_banks(cfg):
    results, _ = paths(cfg)
    for name, path in json.loads((results / "banks.json").read_text()).items():
        yield name, torch.load(path, map_location="cpu", weights_only=False)


def load_features(cfg, bank, kind="windows"):
    _, cache = paths(cfg)
    return torch.load(cache / bank["feature_source"] / f"{kind}.pt", map_location="cpu", weights_only=False)


def make_masks(row, cfg, names, generator):
    width = len(row["words"])
    focal = torch.tensor(row["focals"])
    linear = (torch.arange(width)[None] - focal[:, None]).abs()
    masks = []
    for name in names:
        if name == "full":
            mask = torch.ones(len(focal), width, dtype=torch.bool)
        elif name == "G":
            mask = torch.zeros(len(focal), width, dtype=torch.bool)
        else:
            base = name.removeprefix("rand_")
            d = row["tree"] if base.startswith("tree") else linear
            radius = int(base.removeprefix("tree").removeprefix("lin"))
            mask = d <= radius
            if name.startswith("rand_"):
                number = mask.sum(-1) - 1
                noise = torch.rand(len(focal), width, generator=generator)
                noise[torch.arange(len(focal)), focal] = 2
                ranks = noise.argsort(-1).argsort(-1)
                mask = ranks < number[:, None]
                mask[torch.arange(len(focal)), focal] = True
        masks.append(mask)
    return torch.stack(masks, dim=1)


class Rows(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index]


def collate_rows(rows):
    batch = collate(rows)
    batch["source"] = torch.tensor([i for i, row in enumerate(rows) for _ in row["focals"]])
    batch["focal"] = torch.tensor([f for row in rows for f in row["focals"]])
    width = batch["positions"].shape[1]
    masks = [F_pad(row["masks"], width) for row in rows]
    batch["masks"] = torch.cat(masks)
    return batch


def F_pad(mask, width):
    return torch.nn.functional.pad(mask, (0, width - mask.shape[-1]))


@torch.no_grad()
def encode(rows, encoder, tokenizer, cfg, pooling=False):
    encode_model = parallel(encoder, cfg)
    outputs = []
    loader = DataLoader(Rows(rows), batch_size=cfg.runtime.batch_size, collate_fn=collate_rows,
                        num_workers=cfg.runtime.workers, shuffle=False)
    for raw in tqdm(loader, desc="Context partitions"):
        batch = to_device(raw, cfg.runtime.device)
        sources, focals, masks = batch["source"], batch["focal"], batch["masks"]
        n, conditions, words = masks.shape
        source = sources[:, None].expand(-1, conditions).flatten()
        focal = focals[:, None].expand(-1, conditions).flatten()
        masks = masks.flatten(0, 1)
        chunks = []
        chunk = min(cfg.runtime.focal_chunk, max(1, cfg.runtime.attention_budget // batch["ids"].shape[1] ** 2))
        for start in range(0, len(source), chunk):
            s, f = source[start:start + chunk], focal[start:start + chunk]
            keep = masks[start:start + chunk]
            word_ids = batch["word_ids"][s]
            visible = keep.gather(1, word_ids.clamp_min(0))
            ids = batch["ids"][s].masked_fill((word_ids >= 0) & ~visible, tokenizer.mask_token_id)
            with autocast(cfg):
                h = encode_model(ids, batch["attention_mask"][s], all_layers=False)["features"][:, :, 0]
            if pooling:
                positions = batch["positions"][s]
                vectors = h.gather(1, positions[..., None].expand(-1, -1, h.shape[-1]))
                result = (vectors.float() * keep[..., None]).sum(1) / keep.sum(-1, keepdim=True).clamp_min(1)
            else:
                result = h[torch.arange(len(s), device=s.device), batch["positions"][s, f]]
            chunks.append(result.cpu())
        outputs.append(torch.cat(chunks).reshape(n, conditions, -1).to(getattr(torch, cfg.cache.dtype)))
    return torch.cat(outputs)


def extract(cfg, output):
    _, cache = paths(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.encoder.name, use_fast=True, local_files_only=True)
    encoder = None
    for name, bank in load_banks(cfg):
        if name != bank["feature_source"]: continue
        for kind in ("windows", "occlusion", "pooling"):
            path = cache / name / f"{kind}.pt"
            if path.exists(): continue
            generator = torch.Generator().manual_seed(cfg.diagnostics.sample_seed + 1000 * ("windows", "occlusion", "pooling").index(kind))
            if encoder is None: encoder = build_encoder(cfg.encoder).to(cfg.runtime.device)
            parts = {}
            for split, selected in bank["rows"].items():
                prepared, geometry, valid = [], [], []
                for row in selected:
                    row = dict(row)
                    focal = torch.tensor(row["focals"])
                    width = len(row["words"])
                    if kind == "windows":
                        mask = make_masks(row, cfg, bank["conditions"], generator)
                    elif kind == "occlusion":
                        mask = torch.rand(len(focal), cfg.partition.occlusion_draws, width, generator=generator) >= cfg.partition.occlusion_probability
                        mask[torch.arange(len(focal))[:, None], torch.arange(cfg.partition.occlusion_draws)[None], focal[:, None]] = True
                    else:
                        distance = (torch.arange(width)[None] - focal[:, None]).abs()
                        mask = torch.stack([distance > 3, (distance <= 3) & (distance > 0)], dim=1)
                        valid.append(mask.any(-1).all(-1))
                    row["masks"] = mask
                    prepared.append(row)
                    count = mask.sum(-1) - mask[torch.arange(len(focal))[:, None], torch.arange(mask.shape[1])[None], focal[:, None]].long()
                    shape = count.shape
                    geometry.append(torch.stack([torch.full(shape, len(row["ids"])),
                                                  torch.tensor([row["positions"][i] for i in row["focals"]])[:, None].expand(shape), count], dim=-1).float())
                parts[split] = {"features": encode(prepared, encoder, tokenizer, cfg, pooling=kind == "pooling"),
                                "geometry": torch.cat(geometry)}
                if valid: parts[split]["valid"] = torch.cat(valid)
            torch.save(parts, path)
    return {"feature_cohorts": 3}


def pooled(parts, key="features"):
    return torch.cat([part[key] for part in parts.values()])


def model_data(bank, feature_parts):
    return {split: {"features": feature_parts[split]["features"], "y": part["y"], "sentence": part["sentence"]}
            for split, part in bank["data"].items()}
