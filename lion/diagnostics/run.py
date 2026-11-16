"""Forge entry point and sequential diagnostic suite."""
import importlib
import sys

from forge import grid, run, start_run
from omegaconf import OmegaConf

from ..runtime import configure
from .common import ROOT, absolute, locations, save_json


EXPERIMENTS = ("inventory", "cache", "e1", "e2", "e3", "e4", "e5", "e6", "e7", "report")


def main(cfg):
    OmegaConf.resolve(cfg)
    results, cache = locations(cfg)
    cfg.diagnostics.results, cfg.diagnostics.cache = str(results), str(cache)
    cfg.forge.store = str(absolute(cfg.forge.store))
    if cfg.runtime.devices is not None and cfg.runtime.device.startswith("cuda"):
        cfg.runtime.device = f"cuda:{cfg.runtime.devices[0]}"
    configure(cfg)
    experiment = cfg.stage.experiment
    tracking = start_run(cfg)
    output = results / experiment
    output.mkdir(parents=True, exist_ok=True)
    print(f"Diagnostic stage {experiment}: {output}", flush=True)
    if experiment in {"inventory", "cache"}:
        from .data import inventory, cache_views
        result = {"inventory": inventory, "cache": cache_views}[experiment](cfg, output)
    else:
        module = importlib.import_module(f"lion.diagnostics.{experiment}")
        result = module.experiment(cfg, output)
    save_json(tracking.path / "artifacts.json", {"results": str(output)})
    tracking.finish(result)
    return 0


def cli(experiment):
    raise SystemExit(run("lion.diagnostics", [f"stage.experiment={experiment}", *sys.argv[1:]],
                         main_module="run", config_dir=str(ROOT / "conf"), config_name="diagnostics"))


if __name__ == "__main__":
    grid("lion.diagnostics", sys.argv[1:], [[f"stage.experiment={name}"] for name in EXPERIMENTS], {},
         main_module="run", config_dir=str(ROOT / "conf"), config_name="diagnostics")
