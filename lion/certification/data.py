"""Document splits, aligned subwords, and sentence-batched features."""
import torch
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ..data import BucketBatches, prepare_data, signature
from ..datasets import UD_REPO, _gum_discourse_labels
from ..diagnostics.common import absolute, dataset_config, write_csv
from ..diagnostics.data import NER
from ..labels import LABELS
from ..runtime import autocast


def task_names(cfg):
    if cfg.certification.tasks is not None:
        return tuple(cfg.certification.tasks)
    source = cfg.certification.source
    if source in NER:
        return ("ner",)
    return {"en_ewt": ("upos", "deprel"), "ud_discourse": ("discourse",),
            "conll2000": ("chunk",)}[source]


def paths(cfg):
    if cfg.certification.get('run_path') is not None:return absolute(cfg.certification.run_path)
    certification = OmegaConf.to_container(cfg.certification, resolve=True)
    for key in ("train_sentence_cap", "selector_train_focals_per_fold"):
        if certification.get(key) is None:
            certification.pop(key, None)
    runtime = OmegaConf.to_container(cfg.runtime, resolve=True)
    if runtime["device"].startswith("cuda") and runtime["devices"] is not None:
        runtime["device"] = f"cuda:{runtime['devices'][0]}"
    key = signature(OmegaConf.create({
        "protocol": "word-ranked-ltt-v2",
        "encoder": cfg.encoder, "tagger": cfg.tagger, "train": cfg.train,
        "seed": cfg.seed, "runtime": runtime, "certification": certification,
    }))
    return absolute(cfg.certification.results) / key


def ud_rows(cfg, tokenizer, labels):
    source = cfg.certification.source
    discourse = source == "ud_discourse"
    if discourse:
        sources = [("train", "en_gum", "train"), ("validation", "en_gum", "dev"), ("id", "en_gum", "test")]
    else:
        sources = [(name, source, split) for name, split in
                   (("train", "train"), ("validation", "dev"), ("id", "test"))]
        if cfg.certification.get('include_ood', True):sources.append(("ood", cfg.certification.target, "test"))
    lookup = {task: {name: i for i, name in enumerate(names)} for task, names in labels.items()}
    columns = ["tokens", "sent_id", "misc"] if discourse else ["tokens", "sent_id", *lookup]
    held_out = cfg.certification.discourse_ood_genre
    for name, treebank, split in sources:
        path = hf_hub_download(UD_REPO, f"parquet/{treebank}/{split}.parquet", repo_type="dataset",
                               revision=cfg.certification.get('dataset_revision'))
        raw = pq.read_table(path, columns=columns).to_pylist()
        tokenized = tokenizer([row["tokens"] for row in raw], is_split_into_words=True,
                              add_special_tokens=True, truncation=True, max_length=cfg.encoder.max_length)
        for index, row in enumerate(raw):
            domain = treebank
            subset = name
            if discourse:
                held = row["sent_id"].split("_")[1] == held_out
                if held and name != "id":
                    continue
                domain = f"en_gum/{held_out}" if held else "en_gum/train-genres"
                subset = "ood" if held else name
                row["discourse"] = _gum_discourse_labels(row)
            ids = tokenized["input_ids"][index]
            words = tokenized.word_ids(index)
            retained = list(dict.fromkeys(word for word in words if word is not None))
            slots = {word: slot for slot, word in enumerate(retained)}
            word_ids = [slots.get(word, -1) for word in words]
            gold = {}
            for task, mapping in lookup.items():
                names = [row[task][word] for word in retained]
                if task == "deprel":
                    names = [label.split(":")[0] if label else label for label in names]
                gold[task] = [mapping.get(label, -100) for label in names]
            yield subset, domain, row["sent_id"], row["sent_id"].rsplit("-", 1)[0], ids, word_ids, gold


def native_rows(cfg, task, domain):
    source = cfg.certification.source
    local = dataset_config(cfg, domain)
    local.encoder.special_tokens = True
    data, _ = prepare_data(local)
    splits = {"train": "train", "validation": "validation", "id": "test"} if domain == source else {
        f"ood_{domain}": "test"
    }
    for subset, split in splits.items():
        for index, row in enumerate(data[split]):
            key = f"{domain}/{split}/{index}"
            yield subset, domain, key, key, row["ids"], row["word_ids"], {task: row["labels"]}


def ner_transfer_sets(cfg, task, start):
    sets, domains, coverage = {}, {}, []
    for target in cfg.certification.ner_targets:
        if target == cfg.certification.source:
            continue
        split = f"ood_{target}"
        rows = []
        try:
            for _, _, source_id, document, ids, word_ids, gold in native_rows(cfg, task, target):
                if not gold[task]:
                    continue
                rows.append({"ids": torch.tensor(ids), "word_ids": torch.tensor(word_ids),
                             "focals": torch.arange(len(gold[task])),
                             "gold": {task: torch.tensor(gold[task])}, "sentence": start,
                             "source_id": source_id, "document": document})
                start += 1
        except (FileNotFoundError, ConnectionError, HfHubHTTPError) as error:
            coverage.append({"target": target, "status": "unavailable", "reason": str(error)})
            continue
        sets[split], domains[split] = rows, target
        coverage.append({"target": target, "status": "available", "sentences": len(rows)})
    return sets, domains, coverage


def prepare(cfg, output):
    tokenizer = AutoTokenizer.from_pretrained(cfg.encoder.name, use_fast=True)
    source = cfg.certification.source
    tasks = task_names(cfg)
    ud_labels = {"upos": LABELS["ud_upos"],
                 "deprel": sorted({label.split(":")[0] for label in LABELS["ud_deprel"]}),
                 "discourse": LABELS["ud_discourse"]}
    grouped_by_document = source in ("en_ewt", "ud_discourse")
    if grouped_by_document:
        labels = {task: ud_labels[task] for task in tasks}
        examples = ud_rows(cfg, tokenizer, labels)
    else:
        labels = {tasks[0]: LABELS[source]}
        examples = native_rows(cfg, tasks[0], source)
    sets, domains = {}, {}
    for sentence, (subset, domain, source_id, document, ids, word_ids, gold) in enumerate(examples):
        count = len(gold[tasks[0]])
        if not count:
            continue
        targets = {task: torch.tensor(values) for task, values in gold.items()}
        sets.setdefault(subset, []).append({"ids": torch.tensor(ids), "word_ids": torch.tensor(word_ids),
                                            "focals": torch.arange(count),
                                            "gold": targets, "sentence": sentence, "source_id": source_id,
                                            "document": document})
        domains[subset] = domain

    train = sets["train"]
    cap = cfg.certification.train_sentence_cap
    if cap is not None and cap <= 0:
        raise ValueError("certification.train_sentence_cap must be positive.")
    if cap is not None and cap < len(train):
        indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(cfg.seed))[:cap]
        sets["train"] = [train[i] for i in indices.sort().values.tolist()]

    counts = [{"split": name, "dataset": domains[name], "sentences": len(rows),
                       "documents": len({row["document"] for row in rows}),
                       "words": sum(len(row["focals"]) for row in rows),
                       **{f"{task}_labeled": sum(int((row["gold"][task] >= 0).sum()) for row in rows)
                          for task in tasks}} for name, rows in sets.items()]

    docs = sorted({row["document"] for row in sets["train"]})
    order = torch.randperm(len(docs), generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    docs = [docs[i] for i in order]
    scorer_count = round(len(docs) * cfg.certification.scorer_fraction)
    dev_count = round(scorer_count * cfg.certification.scorer_validation_fraction)
    middle = scorer_count + (len(docs) - scorer_count) // 2
    names = ("scorer_validation", "scorer_train", "g1", "g2")
    groups = (docs[:dev_count], docs[dev_count:scorer_count], docs[scorer_count:middle], docs[middle:])
    if any(not group for group in groups):
        raise ValueError("Document split must populate scorer train/validation and both selector halves.")
    assignment = {document: name for name, group in zip(names, groups) for document in group}
    for row in sets["train"]:
        row["fold"] = assignment[row["document"]]
    for name, group in zip(names, groups):
        rows = [row for row in sets["train"] if row["fold"] == name]
        counts.append({"split": name, "documents": len(group), "sentences": len(rows),
                       "words": sum(len(row["focals"]) for row in rows)})
    special = {"mask": tokenizer.mask_token_id, "excluded": tokenizer.all_special_ids}
    torch.save({"sets": sets, "labels": labels, "special": special,
                "domains": domains, "group_unit": "document" if grouped_by_document else "sentence"},
               output / "sentences.pt")
    write_csv(output / "counts.csv", counts)
    return {"train_sentences": len(sets["train"]),
            "ood_sentences": sum(len(rows) for split, rows in sets.items() if split.startswith("ood"))}


def load(cfg):
    return torch.load(paths(cfg) / "prepare" / "sentences.pt", weights_only=False)


def collate(rows):
    lengths = torch.tensor([len(row["ids"]) for row in rows])
    ids = pad_sequence([row["ids"] for row in rows], batch_first=True, padding_value=0)
    counts = torch.tensor([len(row["focals"]) for row in rows])
    batch = {"ids": ids, "lengths": lengths, "attention": torch.arange(ids.shape[1])[None] < lengths[:, None],
             "word_ids": pad_sequence([row["word_ids"] for row in rows], batch_first=True, padding_value=-1),
             "owner": torch.arange(len(rows)).repeat_interleave(counts),
             "focal": torch.cat([row["focals"] for row in rows]),
             "sentence": torch.tensor([row["sentence"] for row in rows]).repeat_interleave(counts),
             "gold": {task: torch.cat([row["gold"][task] for row in rows]) for task in rows[0]["gold"]}}
    if "hidden" in rows[0]:
        batch["hidden"] = pad_sequence([row["hidden"] for row in rows], batch_first=True)
        batch["word_counts"] = torch.tensor([len(row["hidden"]) for row in rows])
    if "group" in rows[0]:
        batch["group"] = torch.cat([row["group"] for row in rows])
    return batch


def loader(rows, cfg, train=False):
    workers = cfg.runtime.workers
    options = {"collate_fn": collate, "num_workers": workers, "persistent_workers": workers > 0,
               "pin_memory": cfg.runtime.device.startswith("cuda")}
    if train and cfg.train.bucket:
        sampler = BucketBatches([len(row["ids"]) for row in rows], cfg.train.batch_size,
                                cfg.train.shuffle, cfg.seed)
        return DataLoader(rows, batch_sampler=sampler, **options)
    return DataLoader(rows, batch_size=cfg.train.batch_size if train else cfg.runtime.batch_size,
                      shuffle=train and cfg.train.shuffle, **options)


def context(counts, focal, width):
    positions = torch.arange(width, device=focal.device)[None]
    return (positions < counts[:, None]) & (positions != focal[:, None])


def word_to_pieces(values, word_ids, special):
    return torch.where(word_ids >= 0, values.gather(1, word_ids.clamp_min(0)), special)


def pool_words(hidden, word_ids, width):
    index = word_ids.clamp_min(0)
    valid = word_ids >= 0
    values = hidden.float() * valid[..., None]
    sums = values.new_zeros(len(values), width, values.shape[-1])
    sums.scatter_add_(1, index[..., None].expand_as(values), values)
    count = values.new_zeros(len(values), width).scatter_add_(1, index, valid.float())
    return sums / count.clamp_min(1)[..., None]


def focal_vectors(encoder, ids, attention, word_ids, focal, replacements=None, gates=None):
    positions = word_ids == focal[:, None]
    values = encoder(ids, attention, positions, replacement_ids=replacements, gates=gates)["features"].float()
    owners = positions.nonzero(as_tuple=True)[0]
    summed = values.new_zeros(len(ids), values.shape[-1]).index_add_(0, owners, values)
    return summed / positions.sum(-1, keepdim=True)


def attach_features(rows, features):
    lengths = [len(row["focals"]) for row in rows]
    return [{**row, "hidden": vectors} for row, vectors in zip(rows, features.split(lengths))]


def unique_queries(ids, attention, positions):
    """Pack identical right-padded BERT queries and map each example back to its query."""
    width = ids.shape[1]
    keys = torch.cat((attention.sum(-1, keepdim=True), positions[:, None], ids), dim=-1)
    unique, inverse = torch.unique(keys, dim=0, return_inverse=True)
    valid = torch.arange(width, device=ids.device)[None] < unique[:, :1]
    return unique[:, 2:].contiguous(), valid, unique[:, 1], inverse


@torch.no_grad()
def full_features(encoder, sets, cfg, dtype=torch.float16):
    result = {}
    device = cfg.runtime.device
    for name, rows in sets.items():
        parts = []
        for batch in loader(rows, cfg):
            with autocast(cfg):
                hidden = encoder(batch["ids"].to(device, non_blocking=True),
                                 batch["attention"].to(device, non_blocking=True))["features"]
            words = batch["word_ids"].to(device, non_blocking=True)
            pooled = pool_words(hidden, words, int(batch["focal"].max()) + 1)
            parts.append(pooled[batch["owner"].to(device, non_blocking=True),
                                batch["focal"].to(device, non_blocking=True)].to(device="cpu", dtype=dtype))
        result[name] = torch.cat(parts)
    return result


def readout_data(rows, features, task):
    y = torch.cat([row["gold"][task] for row in rows])
    sentence = torch.repeat_interleave(torch.tensor([row["sentence"] for row in rows]),
                                       torch.tensor([len(row["focals"]) for row in rows]))
    labeled = y >= 0
    return {"features": features[labeled, None], "y": y[labeled], "sentence": sentence[labeled]}
