"""Tokenization, padded word alignment, and sentence DataLoaders."""
from functools import partial
from hashlib import sha1
from pathlib import Path

import torch
from datasets import DatasetDict, load_from_disk
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Sampler
from transformers import AutoTokenizer

from .datasets import BUILDERS
from .labels import LABELS


def signature(value):
    return sha1(OmegaConf.to_yaml(value, resolve=True, sort_keys=True).encode()).hexdigest()[:12]


def prepare_data(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.encoder.name, use_fast=True)
    key = signature(OmegaConf.create({"data": OmegaConf.to_container(cfg.data, resolve=True),
                                     "encoder": OmegaConf.to_container(cfg.encoder, resolve=True)}))
    path = Path(cfg.paths.cache) / "tokens" / key
    if path.exists():
        return load_from_disk(path), tokenizer
    raw = BUILDERS[cfg.data.name]()
    if "validation" not in raw:
        split = raw["train"].train_test_split(test_size=cfg.data.validation_fraction, seed=cfg.data.seed)
        raw = DatasetDict(train=split["train"], validation=split["test"], test=raw["test"])
    labels = LABELS[cfg.data.name]
    index = {str(i): i for i in range(len(labels))} | {name: i for i, name in enumerate(labels)}

    def encode(row, sentence_id):
        tokens = tokenizer(row["tokens"], is_split_into_words=True, truncation=True,
                           max_length=cfg.encoder.max_length,
                           add_special_tokens=cfg.encoder.special_tokens)
        source_words = tokens.word_ids()
        # Pack tokenizer word IDs: source tokens can tokenize to no pieces.
        starts = [i for i, w in enumerate(source_words)
                  if w is not None and (i == 0 or w != source_words[i - 1])]
        kept = [source_words[i] for i in starts]
        slots = {word: slot for slot, word in enumerate(kept)}
        return {"ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"],
                "word_ids": [-1 if w is None else slots[w] for w in source_words],
                "positions": starts,
                "labels": [-100 if row["labels"][w] is None else index[str(row["labels"][w])]
                           for w in kept],
                "words": [row["tokens"][w] for w in kept], "sentence_id": sentence_id}

    data = {}
    for name, split in raw.items():
        cap = cfg.data.cap[name]
        if cap is not None:
            split = split.shuffle(seed=cfg.data.seed).select(range(min(cap, len(split))))
        data[name] = split.map(encode, with_indices=True, remove_columns=split.column_names,
                               load_from_cache_file=False, desc=f"Tokenize {cfg.data.name}/{name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = DatasetDict(data)
    data.save_to_disk(path)
    return data, tokenizer


def collate(rows, pad_id=0):
    padding = {"ids": pad_id, "attention_mask": 0, "word_ids": -1,
               "positions": 0, "labels": -100}
    batch = {key: pad_sequence([torch.tensor(row[key], dtype=torch.long) for row in rows],
                              batch_first=True, padding_value=value)
             for key, value in padding.items()}
    counts = torch.tensor([len(row["positions"]) for row in rows])
    batch["word_mask"] = torch.arange(batch["positions"].shape[1])[None] < counts[:, None]
    batch["sentence_ids"] = torch.tensor([row["sentence_id"] for row in rows])
    return batch


class BucketBatches(Sampler):
    """Sort by length once; shuffle batches, rather than individual sentences."""
    def __init__(self, lengths, batch_size, shuffle, seed):
        order = torch.as_tensor(lengths).argsort().tolist()
        self.batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self), generator=generator).tolist() if self.shuffle else range(len(self))
        self.epoch += 1
        for index in order:
            yield self.batches[index]


def loader(dataset, cfg, tokenizer, train=False):
    options = dict(collate_fn=partial(collate, pad_id=tokenizer.pad_token_id),
                   num_workers=cfg.runtime.workers, pin_memory=cfg.runtime.device.startswith("cuda"),
                   persistent_workers=cfg.runtime.workers > 0)
    if train and cfg.train.bucket:
        sampler = BucketBatches([len(row) for row in dataset["ids"]], cfg.train.batch_size,
                                cfg.train.shuffle, cfg.seed)
        return DataLoader(dataset, batch_sampler=sampler, **options)
    return DataLoader(dataset, batch_size=cfg.train.batch_size if train else cfg.runtime.batch_size,
                      shuffle=train and cfg.train.shuffle, **options)


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}
