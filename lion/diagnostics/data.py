"""Recover metadata, sample paired focal positions, and cache actual MASK views."""
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
from ..encoder import build_encoder
from ..labels import LABELS
from ..runtime import autocast, parallel
from .common import ROOT, dataset_config, entropy, locations, save_json, write_csv


TASKS = {"fewnerd": "coarse NER", "wikiann": "English NER", "conll2003": "NER",
         "wnut": "emerging-entity NER", "ud_upos": "UPOS (EWT)",
         "ud_deprel": "dependency-relation labels (EWT)",
         "ud_discourse": "discourse-relation labels (GUM)", "conll2000": "chunking"}
SHARED = ["O", "PER", "ORG", "LOC"]
NER = ("fewnerd", "wikiann", "conll2003", "wnut")


def sampling_budget(cfg, name):
    overrides = cfg.diagnostics.budgets.get(name, {})
    return {key: overrides.get(key, cfg.diagnostics[key]) for key in ("sentences", "focals")}


def token_path(cfg):
    key = signature(OmegaConf.create({"data": OmegaConf.to_container(cfg.data, resolve=True),
                                     "encoder": OmegaConf.to_container(cfg.encoder, resolve=True)}))
    return ROOT / cfg.paths.cache / "tokens" / key


def metadata(name):
    if not name.startswith("ud_"):
        return None
    treebank = "en_gum" if name == "ud_discourse" else "en_ewt"
    result = {}
    for split, source in (("train", "train"), ("validation", "dev"), ("test", "test")):
        path = hf_hub_download(UD_REPO, f"parquet/{treebank}/{source}.parquet",
                               repo_type="dataset", local_files_only=True)
        rows = pq.read_table(path, columns=["sent_id", "tokens"]).to_pylist()
        result[split] = [{"genre": row["sent_id"].split("_")[1] if treebank == "en_gum"
                         else row["sent_id"].split("-")[0],
                          "document": row["sent_id"].rsplit("-", 1)[0],
                          "source_id": row["sent_id"], "tokens": row["tokens"]} for row in rows]
    return result


def inventory(cfg, output):
    results, _ = locations(cfg)
    results.mkdir(parents=True, exist_ok=True)
    baseline = defaultdict(list)
    for path in sorted((ROOT / "outputs" / "xps").glob("*/config.yaml")):
        saved = OmegaConf.load(path)
        if (saved.stage.name != "train" or saved.method.name != "last" or
                saved.tagger.name != "mlp" or saved.encoder.name != cfg.encoder.name):
            continue
        for meta_path in sorted(path.parent.glob("*/meta.json")):
            meta = json.loads(meta_path.read_text())
            report = meta_path.parent / "report.json"
            scores = json.loads(report.read_text()).get("test", {}) if report.exists() else {}
            baseline[saved.data.name].append({"path": str(meta_path.parent.relative_to(ROOT)),
                                               "status": meta["status"], "seed": saved.seed,
                                               "scores": {key: scores[key] for key in
                                                          ("accuracy", "macro_f1", "span_f1") if key in scores}})
    entries = []
    for name in cfg.diagnostics.datasets:
        local = dataset_config(cfg, name)
        path = token_path(local)
        entry = {"name": name, "task": TASKS[name], "labels": LABELS[name],
                 "available": path.exists(), "token_cache": str(path),
                 "baselines": baseline[name], "splits": {}, "z": None}
        if path.exists():
            data = load_from_disk(path)
            for split, rows in data.items():
                n = sum(len(row) for row in rows["labels"])
                supervised = sum(sum(label >= 0 for label in row) for row in rows["labels"])
                entry["splits"][split] = {"sentences": len(rows), "words": n,
                                           "supervised_words": supervised}
            sampled = sample_sentences(data, cfg, name)
            entry["sample_sizes"] = {split: {"sentences": len(rows), "focals": sum(len(row["focals"]) for row in rows)}
                                     for split, rows in sampled.items()}
            raw = metadata(name)
            if raw is not None:
                for split in data:
                    if len(raw[split]) != len(data[split]):
                        raise ValueError(f"Metadata alignment requires uncapped native splits: {name}/{split}")
                entry["z"] = {"kind": "genre", "values": sorted({row["genre"] for rows in raw.values() for row in rows}),
                              "source": "raw cached UD sent_id, recovered in a diagnostic sidecar",
                              "document_overlap": {split: len({r["document"] for r in raw["train"]} &
                                                              {r["document"] for r in raw[split]})
                                                   for split in ("validation", "test")}}
                domains = entry["z"]["values"]
                supported = torch.ones(len(LABELS[name]), len(domains), dtype=torch.bool)
                for split, rows in sampled.items():
                    pairs = torch.tensor([row["labels"][focal] * len(domains) +
                                          domains.index(raw[split][row["sentence_id"]]["genre"])
                                          for row in rows for focal in row["focals"]])
                    support = torch.bincount(pairs, minlength=len(LABELS[name]) * len(domains)).reshape(len(LABELS[name]), len(domains))
                    supported &= support >= cfg.diagnostics.probe_min[split]
                entry["probe_eligible_labels"] = [LABELS[name][i] for i in (supported.sum(-1) >= 2).nonzero().flatten().tolist()]
        entries.append(entry)
    ready = [row["name"] for row in entries if row["available"]]
    ner = [name for name in NER if name in ready]
    payload = {"datasets": entries, "ready": ready, "ner_sources": ner,
               "encoder": OmegaConf.to_container(cfg.encoder, resolve=True),
               "budget": OmegaConf.to_container(cfg.diagnostics, resolve=True)}
    save_json(results / "inventory.json", payload)
    save_json(output / "inventory.json", payload)
    audit = [{"dataset": row["name"], "split": split, **counts,
              "probe_eligible_labels": len(row.get("probe_eligible_labels", [])) if row["z"] else "n/a"}
             for row in entries for split, counts in row.get("sample_sizes", {}).items()]
    write_csv(results / "sampling_audit.csv", audit)
    maximum_focals = sum(sum(sampling_budget(cfg, name)["focals"].values()) for name in ready)
    lines = ["# Diagnostic experiment plan", "", "## Inventory", "",
             "One artifact snapshot; missing or failed baselines have no fabricated scores. "
             "Baseline code, checkpoints, native splits, and caches are read-only to this study.", "",
             "| Dataset | Task | Labels | Train / validation / test sentences | Baseline test accuracy / macro F1 / span F1 | Z |",
             "|---|---|---:|---|---|---|"]
    for row in entries:
        sizes = " / ".join(str(row["splits"].get(s, {}).get("sentences", "unavailable"))
                           for s in ("train", "validation", "test"))
        completed = [r for r in row["baselines"] if r["status"] == "done"]
        score = "; ".join("seed " + str(r["seed"]) + ": " + " / ".join(
            f"{r['scores'][key]:.4f}" if key in r["scores"] else "n/a"
            for key in ("accuracy", "macro_f1", "span_f1")) for r in completed)
        if not score:
            score = ", ".join(r["status"] for r in row["baselines"]) or "not run"
        z = ", ".join(row["z"]["values"]) if row["z"] else "no retained native Z"
        lines.append(f"| {row['name']} | {row['task']} | {len(row['labels'])} | {sizes} | {score} | {z} |")
    lines += ["", "Encoder: frozen `" + cfg.encoder.name + "`, block 12, first WordPiece per word, "
              "no CLS/SEP, at most 512 WordPieces. Baseline MLP: 768 → 256 → labels, ReLU, dropout 0.1. "
              "AdamW, lr 0.001, weight decay 0.01, clip 1, 64 sentences/step, 10 epochs, "
              "validation macro-F1 selection. Baselines have one seed; diagnostic training uses seeds 0/1/2.",
              "", "### Label sets and actual cached sizes", ""]
    for row in entries:
        lines += [f"- **{row['name']}**: " + ", ".join(row["labels"]) + ".",
                  "  Cached counts: `" + json.dumps(row["splits"]) + "`."]
        for run in row["baselines"]:
            lines.append(f"  Baseline: `{run['path']}` ({run['status']}).")
    lines += ["", "## Assignment", "",
              "- E1/E2: all available native datasets: " + ", ".join(ready) + ".",
              "- E3–E6: EWT UPOS, EWT dependency labels, and GUM discourse labels using recovered genre Z. "
              "Also pooled `ner_shared` using dataset identity from " + ", ".join(ner) + ".",
              "- E7: available native NER datasets: " + ", ".join(ner) + ".",
              "- WikiANN/WNUT (if absent above) are skipped, not silently replaced or repaired. "
              "Unavailable metadata, label/domain cells, and substitutions are written to skip tables.",
              "", "Exact pre-encoding sample sizes and supported probe label counts are in "
              "[sampling_audit.csv](sampling_audit.csv). The GUM support audit at the revised budget "
              "has 11 eligible labels (versus zero at the default budget); UPOS has 9 and dependency "
              "labels have 8. These are conditional cells with enough support, not successful probe results.",
              "", "### Shared NER mapping", "",
              "Keep O→O; person/PER→PER; organization/ORG and WNUT corporation/group→ORG; "
              "location/LOC→LOC. Strip BIO boundaries for this token-level comparison. "
              "Other entity types (including Few-NERD building and CoNLL MISC) are **excluded as targets**, "
              "not relabelled O; they remain visible as context. This is a new four-label diagnostic readout, "
              "not a reinterpretation of a native baseline score. Dataset identity mixes source, "
              "annotation, sampling, and genre differences and is only an observational proxy for Z.",
              "", "## Sampling and protocol", "",
              f"Fixed sampling seed {cfg.diagnostics.sample_seed}; at most {cfg.diagnostics.positions_per_sentence} "
              "uniform valid focal words per uniformly selected sentence. Sentence caps (train/validation/test): "
              f"{cfg.diagnostics.sentences.train}/{cfg.diagnostics.sentences.validation}/{cfg.diagnostics.sentences.test}; "
              "focal caps: "
              f"{cfg.diagnostics.focals.train}/{cfg.diagnostics.focals.validation}/{cfg.diagnostics.focals.test}. "
              "Short/label-free sentences can reduce actual counts, which are saved in samples.csv. "
              "The same positions and feature cache serve all views and all seeds; split membership stays native. "
              "Sentence-uniform focal sampling changes token weighting, so these scores are not unbiased "
              "estimates of full-corpus token performance. "
              "GUM has a larger budget because a sample-support audit found no eligible conditional Z "
              "probe labels at the default budget. GUM sentence caps: 8192/1600/1500; focal caps: "
              "16384/3200/3000. Each focal has two masked forwards, vectorized in runtime-configured chunks; "
              "full sentence representations are computed once. "
              f"The current available-dataset budget bounds masked forwards at {2 * maximum_focals:,}, "
              "plus full-sentence forwards. Only block-12 focal "
              f"vectors are cached (roughly {maximum_focals * 3 * 768 * 2 / 2**20:.0f} MiB before metadata).",
              "",
              "Masking replaces **every WordPiece** of the relevant words by the tokenizer MASK ID. "
              "Attention remains on; this is distinct from Lion's learned attention gates. Word-only "
              "keeps the focal word's pieces; context-only replaces its pieces. Sequence length, focal "
              "position, and masked-piece counts remain observable, so word-only is not literally P(Y|word) "
              "and a calibration mismatch alone cannot refute the conditioning interpretation.",
              "",
              "Train each view from scratch with the baseline MLP/settings and identical seed initialization, "
              "on the paired sample. Score token accuracy, macro F1 and per-label scores; do not compute span "
              "F1 on disconnected focal samples. Frequency/label entropy use all original training words "
              "(case-sensitive), never test labels. Unseen forms have undefined label entropy. E2 uses "
              "frequency≥20 ambiguous types and held-out focal predictions, reporting KL(empirical train || prediction), "
              "type-level distribution error and token ECE, without claiming training frequencies are true posteriors.",
              "",
              "E3 fits balanced linear Z probes separately within each Y, using only domains with at least "
              "32/8/8 train/validation/test tokens. E4 learns one global closed-form LEACE-style projection "
              "from training data, using within-label centered domain indicators as the concept. It removes "
              "linear domain covariance, not guaranteed all classifier-accessible Z. Apply it without gold "
              "labels at inference, retrain Y MLPs, and fit fresh conditional Z probes. A probe above chance "
              "means erasure was incomplete; a Y drop alone does not identify causal entanglement.",
              "",
              "E5 caps each domain/label/stratum at 128 focal tokens (minimum 20 tokens and 3 documents), "
              "uses unbiased RBF MMD and 199 document-block permutations, and reports BH-adjusted q values "
              "within each seed. Native UD document IDs define blocks; shared NER only has sentence blocks "
              "and its p values are exploratory. Word entropy bins: zero, (0,ln2], >ln2, unseen; predictive "
              "entropy cuts are training tertiles. Word-distribution JS uses the same eligible pair. "
              "Native train/heldout document overlaps are in inventory.json; they limit generalization claims.",
              "",
              "E6 trains within each source genre/dataset using only its training and validation samples, "
              "then evaluates native source and other-domain test samples; eligibility is at least 128/16/32 "
              "source tokens and 32 target test tokens. Report per-label support and recall degradation, "
              "including labels absent from source training. Relate these to E5 by domain-pair and label; "
              "correlations are descriptive. E7 uses up to 128 eligible NER focals per dataset; replacements "
              "come only from train, have the same entity type and per-word WordPiece lengths, and change "
              "the nearest neighboring entity within 8 words or the focal mention itself. Keep single- and "
              "multiword focal mentions separate. These controlled substitutions are not proven nuisance-only "
              "interventions; semantic coherence and label preservation are assumptions.",
              "", "## Outputs and execution", "",
              "`python -m lion.diagnostics.run` runs each stage through sibling Forge/Hydra. "
              "`python -m lion.diagnostics.e1` (through `e7`) reproduces one experiment. Config: "
              "`conf/diagnostics/default.yaml`; overrides are ordinary Hydra overrides. Each experiment "
              "writes seed-level and aggregate CSVs and a PNG to results/eN. Final synthesis is "
              "results/report.md. Until execution finishes, measurements remain pending. "
              "Heavy execution is launched in tmux and not monitored by the agent.", ""]
    (results / "plan.md").write_text("\n".join(lines))
    return {"available_datasets": len(ready), "genre_datasets": sum(row["z"] is not None for row in entries)}


def load_inventory(cfg):
    results, _ = locations(cfg)
    return json.loads((results / "inventory.json").read_text())


def train_statistics(data, num_labels, mapping=None):
    # Python builds the string vocabulary; all counting/entropy is tensorized.
    words, labels = [], []
    for row in data["train"]:
        for word, label in zip(row["words"], row["labels"]):
            label = int(mapping[label]) if mapping is not None and label >= 0 else label
            if label >= 0:
                words.append(word)
                labels.append(label)
    vocab = {word: i for i, word in enumerate(sorted(set(words)))}
    word_ids = torch.tensor([vocab[word] for word in words])
    counts = torch.bincount(word_ids * num_labels + torch.tensor(labels),
                            minlength=len(vocab) * num_labels).reshape(-1, num_labels)
    probability = counts.float() / counts.sum(-1, keepdim=True)
    return {"vocab": vocab, "counts": counts, "entropy": entropy(probability)}


def add_statistics(data, stats):
    indices = torch.tensor([stats["vocab"].get(word, -1) for word in data["words"]])
    seen = indices >= 0
    data["frequency"] = torch.zeros(len(indices), dtype=torch.long)
    data["word_entropy"] = torch.full((len(indices),), torch.nan)
    data["frequency"][seen] = stats["counts"].sum(-1)[indices[seen]]
    data["word_entropy"][seen] = stats["entropy"][indices[seen]]


def sample_sentences(data, cfg, name=None):
    generator = torch.Generator().manual_seed(cfg.diagnostics.sample_seed)
    budget = sampling_budget(cfg, name or cfg.data.name)
    selected = {}
    for split, dataset in data.items():
        rows, total = [], 0
        for index in torch.randperm(len(dataset), generator=generator).tolist():
            row = dict(dataset[index])
            valid = (torch.tensor(row["labels"]) >= 0).nonzero().flatten()
            chosen = valid[torch.randperm(len(valid), generator=generator)[:cfg.diagnostics.positions_per_sentence]]
            chosen = chosen[:budget["focals"][split] - total]
            if len(chosen):
                row["focals"] = chosen.sort().values.tolist()
                rows.append(row)
                total += len(chosen)
            if len(rows) >= budget["sentences"][split] or total >= budget["focals"][split]:
                break
        selected[split] = sorted(rows, key=lambda row: row["sentence_id"])
    return selected


class SentenceRows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate_focals(rows):
    batch = collate(rows)
    batch["source"] = torch.tensor([i for i, row in enumerate(rows) for _ in row["focals"]])
    batch["focal"] = torch.tensor([focal for row in rows for focal in row["focals"]])
    return batch


def sentence_loader(rows, cfg):
    return DataLoader(SentenceRows(rows), batch_size=cfg.runtime.batch_size, shuffle=False,
                      num_workers=cfg.runtime.workers, collate_fn=collate_focals,
                      pin_memory=cfg.runtime.device.startswith("cuda"))


@torch.no_grad()
def encode_views(rows, encoder, tokenizer, cfg):
    encode = parallel(encoder, cfg)
    outputs = []
    for raw in tqdm(sentence_loader(rows, cfg), desc="Three views"):
        batch = to_device(raw, cfg.runtime.device)
        source, focal = batch["source"], batch["focal"]
        with autocast(cfg):
            full = encode(batch["ids"], batch["attention_mask"], all_layers=False)["features"]
        full = full[source, batch["positions"][source, focal], 0]
        masked = [[], []]
        chunk = min(cfg.runtime.focal_chunk, max(1, cfg.runtime.attention_budget // batch["ids"].shape[1] ** 2))
        for start in range(0, len(source), chunk):
            s, f = source[start:start + chunk], focal[start:start + chunk]
            ids, word_ids, attention = batch["ids"][s], batch["word_ids"][s], batch["attention_mask"][s]
            focal_pieces = word_ids == f[:, None]
            word_pieces = word_ids >= 0
            for view, replace in enumerate((word_pieces & ~focal_pieces, focal_pieces)):
                changed = ids.masked_fill(replace, tokenizer.mask_token_id)
                with autocast(cfg):
                    encoded = encode(changed, attention, all_layers=False)["features"]
                value = encoded[torch.arange(len(s), device=s.device), batch["positions"][s, f], 0]
                masked[view].append(value.cpu())
        outputs.append(torch.stack([full.cpu(), torch.cat(masked[0]), torch.cat(masked[1])], dim=1)
                       .to(getattr(torch, cfg.cache.dtype)))
    return torch.cat(outputs)


def cache_folder(cfg, name):
    _, cache = locations(cfg)
    key = signature(OmegaConf.create({"encoder": OmegaConf.to_container(cfg.encoder, resolve=True),
                                      "data": OmegaConf.to_container(dataset_config(cfg, name).data, resolve=True),
                                      "sampling": {"seed": cfg.diagnostics.sample_seed,
                                                   "positions_per_sentence": cfg.diagnostics.positions_per_sentence,
                                                   **{key: OmegaConf.to_container(value, resolve=True)
                                                      for key, value in sampling_budget(cfg, name).items()}},
                                      "amp": cfg.runtime.amp, "tf32": cfg.runtime.tf32,
                                      "dtype": cfg.cache.dtype,
                                      "protocol": "mask-token-three-views-v1"}))
    return cache / name / key


def cache_views(cfg, output):
    inventory = load_inventory(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.encoder.name, use_fast=True)
    if tokenizer.mask_token_id is None:
        raise ValueError("Three-view diagnostics require a tokenizer MASK token")
    encoder = None
    manifest = {}
    sample_counts = []
    for name in inventory["ready"]:
        folder = cache_folder(cfg, name)
        print(f"Three-view cache: {name} → {folder}", flush=True)
        if not (folder / "bank.pt").exists():
            local = dataset_config(cfg, name)
            data = load_from_disk(token_path(local))
            stats = train_statistics(data, len(LABELS[name]))
            rows = sample_sentences(data, cfg, name)
            raw_metadata = metadata(name)
            genres = inventory["datasets"][cfg.diagnostics.datasets.index(name)]["z"]
            z_names = genres["values"] if genres else []
            if encoder is None:
                encoder = build_encoder(cfg.encoder).to(cfg.runtime.device)
            bank = {}
            for split, selected in rows.items():
                documents, z, words, sentence, positions, targets = [], [], [], [], [], []
                for row in selected:
                    meta = raw_metadata[split][row["sentence_id"]] if raw_metadata else None
                    if meta:
                        reencoded = tokenizer(meta["tokens"], is_split_into_words=True, truncation=True,
                                              max_length=cfg.encoder.max_length,
                                              add_special_tokens=cfg.encoder.special_tokens)["input_ids"]
                        if reencoded != row["ids"]:
                            raise ValueError(f"Raw metadata does not align with cached sentence: {name}/{split}/{row['sentence_id']}")
                    for focal in row["focals"]:
                        documents.append(meta["document"] if meta else f"{name}:{split}:{row['sentence_id']}")
                        z.append(z_names.index(meta["genre"]) if meta else -1)
                        words.append(row["words"][focal])
                        sentence.append(row["sentence_id"])
                        positions.append(focal)
                        targets.append(row["labels"][focal])
                bank[split] = {"features": encode_views(selected, encoder, tokenizer, cfg),
                               "y": torch.tensor(targets), "z": torch.tensor(z), "words": words,
                               "sentence": torch.tensor(sentence), "position": torch.tensor(positions),
                               "document": documents}
                add_statistics(bank[split], stats)
            folder.mkdir(parents=True, exist_ok=True)
            torch.save({"data": bank, "stats": stats, "rows": rows, "labels": LABELS[name],
                        "z_names": z_names, "config": OmegaConf.to_container(cfg, resolve=True)}, folder / "bank.pt")
        saved = torch.load(folder / "bank.pt", map_location="cpu", weights_only=False)
        manifest[name] = str(folder / "bank.pt")
        sample_counts += [{"dataset": name, "split": split, "sentences": len(saved["rows"][split]),
                           "focals": len(data["y"])} for split, data in saved["data"].items()]
    if len(inventory["ner_sources"]) >= 2:
        manifest["ner_shared"] = shared_bank(cfg, inventory["ner_sources"], manifest)
    results, _ = locations(cfg)
    save_json(results / "banks.json", manifest)
    write_csv(results / "samples.csv", sample_counts)
    return {"datasets": len(manifest), "focals": sum(row["focals"] for row in sample_counts)}


def shared_mapping(name):
    mapping = {"O": 0, "PER": 1, "person": 1, "ORG": 2, "organization": 2,
               "corporation": 2, "group": 2, "LOC": 3, "location": 3}
    return torch.tensor([mapping.get(tag[2:] if tag.startswith(("B-", "I-")) else tag, -1)
                         for tag in LABELS[name]])


def shared_bank(cfg, sources, manifest):
    from .common import subset
    banks = [torch.load(manifest[name], weights_only=False) for name in sources]
    train_words, train_labels = [], []
    for name in sources:
        local = dataset_config(cfg, name)
        data = load_from_disk(token_path(local))
        mapping = shared_mapping(name)
        for row in data["train"]:
            for word, label in zip(row["words"], row["labels"]):
                if label >= 0 and mapping[label] >= 0:
                    train_words.append(word)
                    train_labels.append(int(mapping[label]))
    stats = train_statistics({"train": [{"words": train_words, "labels": train_labels}]}, len(SHARED))
    data = {}
    for split in ("train", "validation", "test"):
        chunks = []
        for domain, (name, bank) in enumerate(zip(sources, banks)):
            old = bank["data"][split]
            mapping = shared_mapping(name)
            selected = subset(old, (mapping[old["y"]] >= 0).nonzero().flatten())
            selected["y"] = mapping[selected["y"]]
            selected["z"] = torch.full_like(selected["y"], domain)
            # Prefix source to keep sentence/document grouping disjoint across corpora.
            selected["document"] = [f"{name}:{doc}" for doc in selected["document"]]
            selected["sentence"] = selected["sentence"] + domain * 10**8
            chunks.append(selected)
        data[split] = {key: torch.cat([chunk[key] for chunk in chunks]) if isinstance(chunks[0][key], torch.Tensor)
                       else [value for chunk in chunks for value in chunk[key]] for key in chunks[0]}
        add_statistics(data[split], stats)
    _, cache = locations(cfg)
    folder = cache / "ner_shared" / signature(OmegaConf.create({"sources": manifest}))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "bank.pt"
    torch.save({"data": data, "stats": stats, "labels": SHARED, "z_names": sources}, path)
    return str(path)


def banks(cfg, nuisance=False):
    results, _ = locations(cfg)
    manifest = json.loads((results / "banks.json").read_text())
    for name, path in manifest.items():
        bank = torch.load(path, map_location="cpu", weights_only=False)
        if not nuisance or bank["z_names"]:
            yield name, bank
