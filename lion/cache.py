"""Sharded frozen word vectors; sentences remain the DataLoader unit."""
from functools import lru_cache
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .data import loader, signature, to_device
from .encoder import word_features
from .runtime import parallel, autocast


def cache_path(cfg):
    key = signature(OmegaConf.create({
        "data": OmegaConf.to_container(cfg.data, resolve=True),
        "encoder": OmegaConf.to_container(cfg.encoder, resolve=True),
        "heads": cfg.method.name in {"head_mix", "layer_head_mix"},
        "dtype": cfg.cache.dtype,
        "precision": cfg.runtime.amp,
        "tf32": cfg.runtime.tf32,
    }))
    return Path(cfg.paths.cache) / "features" / key


@torch.no_grad()
def ensure_cache(cfg, data, tokenizer, encoder):
    root = cache_path(cfg)
    if (root / "manifest.json").exists():
        return root
    root.mkdir(parents=True, exist_ok=True)
    splits = {}
    heads = cfg.method.name in {"head_mix", "layer_head_mix"}
    encode = parallel(encoder, cfg)
    for name, dataset in data.items():
        folder = root / name
        folder.mkdir(exist_ok=True)
        sizes = []
        for shard_id, batch in enumerate(tqdm(loader(dataset, cfg, tokenizer), desc=f"Cache {name}")):
            batch = to_device(batch, cfg.runtime.device)
            with autocast(cfg):
                encoded = word_features(encode(batch["ids"], batch["attention_mask"], heads=heads), batch["positions"])
            valid = batch["word_mask"]
            values = {key: value[valid].to(dtype=getattr(torch, cfg.cache.dtype), device="cpu")
                      for key, value in encoded.items()}
            counts = valid.sum(-1).cpu()
            values.update(labels=batch["labels"][valid].cpu(), counts=counts,
                          offsets=torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)]),
                          sentence_ids=batch["sentence_ids"].cpu())
            torch.save(values, folder / f"{shard_id:05d}.pt")
            sizes.append(len(counts))
        splits[name] = sizes
    manifest = {"splits": splits, "dimension": encoder.dimension,
                "config": OmegaConf.to_container(cfg.encoder, resolve=True)}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return root


class CachedSentences(Dataset):
    def __init__(self, root, split):
        self.root = Path(root) / split
        sizes = json.loads((Path(root) / "manifest.json").read_text())["splits"][split]
        self.shards = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
        self.offsets = torch.tensor([0, *sizes]).cumsum(0)

    def __len__(self):
        return len(self.shards)

    @lru_cache(maxsize=2)
    def shard(self, index):
        return torch.load(self.root / f"{index:05d}.pt", mmap=True, weights_only=True)

    def __getitem__(self, index):
        shard_id = int(self.shards[index])
        data = self.shard(shard_id)
        row = index - int(self.offsets[shard_id])
        start, stop = data["offsets"][row:row + 2].tolist()
        result = {key: data[key][start:stop] for key in ("features", "labels")}
        if "heads" in data:
            result["heads"] = data["heads"][start:stop]
        result["sentence_id"] = data["sentence_ids"][row]
        return result


def collate_cached(rows):
    result = {key: pad_sequence([row[key] for row in rows], batch_first=True,
                                padding_value=-100 if key == "labels" else 0)
              for key in rows[0] if key != "sentence_id"}
    result["sentence_ids"] = torch.stack([row["sentence_id"] for row in rows])
    for key in ("features", "heads"):
        if key in result:
            result[key] = result[key].float()
    counts = torch.tensor([len(row["labels"]) for row in rows])
    result["word_mask"] = torch.arange(result["labels"].shape[1])[None] < counts[:, None]
    return result


def cached_loader(root, split, cfg, train=False):
    return DataLoader(CachedSentences(root, split), collate_fn=collate_cached,
                      batch_size=cfg.train.batch_size if train else cfg.runtime.batch_size,
                      shuffle=train and cfg.train.shuffle, num_workers=cfg.runtime.workers,
                      pin_memory=cfg.runtime.device.startswith("cuda"),
                      persistent_workers=cfg.runtime.workers > 0)
