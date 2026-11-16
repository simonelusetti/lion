"""Forge entry point: every stage is selected by configuration."""
import json
from pathlib import Path

import torch
from forge import start_run
from omegaconf import OmegaConf

from .analysis import analyze, word_metadata
from .cache import ensure_cache
from .data import prepare_data, signature
from .encoder import build_encoder
from .labels import LABELS
from .masks import search
from .models import SequenceModel
from .runtime import configure, parallel
from .training import evaluate, fit, make_loaders


ROOT = Path(__file__).resolve().parents[1]


def absolute(path):
    path = Path(path).expanduser()
    return str(path if path.is_absolute() else ROOT / path)


def data_signature(cfg):
    return signature(OmegaConf.create({"data": OmegaConf.to_container(cfg.data, resolve=True),
                                      "encoder": OmegaConf.to_container(cfg.encoder, resolve=True)}))


def mask_records(cfg):
    records = {}
    for source in cfg.masks.sources:
        saved = torch.load(source, map_location="cpu", weights_only=False)
        if saved["signature"] != data_signature(cfg):
            raise ValueError(f"Mask records were built for a different dataset/tokenizer config: {source}")
        for split, rows in saved["records"].items():
            records.setdefault(split, {}).update(rows)
    needs_masks = cfg.method.name == "oracle_mask" or (
        cfg.method.name == "supervised_mask" and cfg.stage.name != "eval")
    if needs_masks and not records:
        raise ValueError("Set masks.sources to the records.pt files from stage=masks")
    if cfg.method.name == "oracle_mask":
        required = {cfg.evaluation.split} if cfg.stage.name == "eval" else {"train", "validation", "test"}
        if required - records.keys():
            raise ValueError(f"Oracle evaluation requires mask files for splits {sorted(required)}")
    return records or None


def main(cfg):
    OmegaConf.resolve(cfg)
    cfg.paths.cache = absolute(cfg.paths.cache)
    cfg.forge.store = absolute(cfg.forge.store)
    cfg.masks.sources = [absolute(path) for path in cfg.masks.sources]
    cfg.analysis.sources = [absolute(path) for path in cfg.analysis.sources]
    for key in ("classifier_source", "selector_source"):
        if cfg.train[key] is not None:
            cfg.train[key] = absolute(cfg.train[key])
    saved = None
    if cfg.stage.name == "eval":
        cfg.evaluation.source = absolute(cfg.evaluation.source)
        saved = torch.load(cfg.evaluation.source, map_location="cpu", weights_only=False)
        for key in ("encoder", "method", "tagger"):
            cfg[key] = OmegaConf.create(saved["config"][key])
    if cfg.runtime.devices is not None and cfg.runtime.device.startswith("cuda"):
        cfg.runtime.device = f"cuda:{cfg.runtime.devices[0]}"
    configure(cfg)
    run = start_run(cfg)
    output = run.path
    if cfg.stage.name == "analyze":
        run.finish(analyze(cfg.analysis.sources, output))
        return 0
    data, tokenizer = prepare_data(cfg)
    if cfg.stage.name == "data":
        run.finish({f"{name}_sentences": len(split) for name, split in data.items()})
        return 0
    encoder = build_encoder(cfg.encoder).to(cfg.runtime.device)
    if cfg.stage.name == "cache":
        path = ensure_cache(cfg, data, tokenizer, encoder)
        (output / "artifacts.json").write_text(json.dumps({"features": str(path)}, indent=2) + "\n")
        run.finish({f"{name}_sentences": len(split) for name, split in data.items()})
        return 0
    if cfg.stage.name == "masks":
        records, summary = search(cfg, data, tokenizer, encoder, output)
        torch.save({"signature": data_signature(cfg), "records": records,
                    "config": OmegaConf.to_container(cfg, resolve=True)}, output / "records.pt")
        run.finish({key: value for key, value in summary.items() if isinstance(value, (int, float))})
        return 0
    labels = saved["labels"] if saved is not None else LABELS[cfg.data.name]
    dataloaders = make_loaders(cfg, data, tokenizer, encoder)
    records = mask_records(cfg)
    if cfg.stage.name == "layerwise":
        torch.save(word_metadata(data, len(labels)), output / "words.pt")
        for layer in cfg.stage.layers:
            local = OmegaConf.merge(cfg, {"method": {"name": "layer", "kind": "layers", "layer": layer}})
            torch.manual_seed(cfg.seed)
            core = SequenceModel(local, encoder, len(labels)).to(cfg.runtime.device)
            fit(core, dataloaders, local, run, output / "layers" / f"layer{layer:02d}", labels)
        run.finish(analyze([output], output))
        return 0
    torch.manual_seed(cfg.seed)
    core = SequenceModel(cfg, encoder, len(labels)).to(cfg.runtime.device)
    if saved is not None:
        core.method.load_state_dict(saved["method"])
        core.tagger.load_state_dict(saved["tagger"])
        shared = cfg.data.name != saved["config"]["data"]["name"]
        split = cfg.evaluation.split
        evaluation_records = records.get(split) if records is not None else None
        result, predictions = evaluate(parallel(core, cfg), core, dataloaders[split], cfg, labels,
                                       evaluation_records, shared=shared)
        torch.save(predictions, output / "predictions.pt")
        (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    else:
        result = fit(core, dataloaders, cfg, run, output, labels, records)
    run.finish({key: value for key, value in result.items() if isinstance(value, (int, float))})
    return 0
