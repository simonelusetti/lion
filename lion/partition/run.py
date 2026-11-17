"""Forge/Hydra stage runner for the X0–X4 follow-up."""
import importlib
import sys

from forge import grid, start_run
from omegaconf import OmegaConf

from ..diagnostics.common import ROOT, absolute, save_json
from ..runtime import configure
from .data import paths


STAGES = ("prepare", "cache", "frontier", "shift", "attribution", "labels", "clusters", "report")


def main(cfg):
    OmegaConf.resolve(cfg)
    cfg.forge.store = str(absolute(cfg.forge.store))
    if cfg.runtime.devices is not None and cfg.runtime.device.startswith("cuda"):
        cfg.runtime.device = f"cuda:{cfg.runtime.devices[0]}"
    configure(cfg)
    stage = cfg.stage.experiment
    tracking = start_run(cfg)
    results, _ = paths(cfg)
    output = results / stage
    output.mkdir(parents=True, exist_ok=True)
    print(f"Partition stage {stage}: {output}", flush=True)
    if stage in {"prepare", "cache"}:
        from .data import prepare, extract
        result = {"prepare": prepare, "cache": extract}[stage](cfg, output)
    else:
        result = importlib.import_module(f"lion.partition.{stage}").experiment(cfg, output)
    save_json(tracking.path / "artifacts.json", {"results": str(output)})
    tracking.finish(result)
    return 0


if __name__ == "__main__":
    grid("lion.partition", sys.argv[1:], [[f"stage.experiment={stage}"] for stage in STAGES], {},
         main_module="run", config_dir=str(ROOT / "conf"), config_name="partition")
