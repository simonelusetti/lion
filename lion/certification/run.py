"""Forge/Hydra entry point for the selector experiment."""
import os
import sys
import subprocess
import time

from forge import run, start_run
from forge.matching import compose_cfg
from omegaconf import OmegaConf

from ..diagnostics.common import ROOT, absolute, save_json, write_csv
from ..runtime import configure
from . import certify, classify, data, selector


STAGES = {
    "prepare": data.prepare,
    "scorer": selector.scorer_stage,
    "selector": selector.experiment,
    "certify": certify.experiment,
    "clean": classify.clean,
    "classify": classify.experiment,
}

SUITE = (
    ("wnut", "ner"), ("conll2003", "ner"), ("wikiann", "ner"), ("fewnerd", "ner"),
    ("en_ewt", "upos"), ("en_ewt", "deprel"), ("ud_discourse", "discourse"), ("conll2000", "chunk"),
)


def main(cfg):
    OmegaConf.resolve(cfg)
    cfg.forge.store = str(absolute(cfg.forge.store))
    cfg.paths.cache = str(absolute(cfg.paths.cache))
    if cfg.runtime.devices is not None and cfg.runtime.device.startswith("cuda"):
        cfg.runtime.device = f"cuda:{cfg.runtime.devices[0]}"
    stage = cfg.stage.experiment
    threads = cfg.runtime.threads
    if stage in ("selector", "certify", "clean") and cfg.stage.fold is not None and cfg.runtime.selector_cpu_groups is not None:
        threads = min(threads, len(os.sched_getaffinity(0)))
    configure(cfg, threads=threads)
    output = data.paths(cfg) / stage
    output.mkdir(parents=True, exist_ok=True)
    tracking = start_run(cfg)
    save_json(tracking.path / "artifacts.json", {"results": str(output)})
    print(f"Certification {stage}: {output} (CPU threads: {threads})", flush=True)
    clock = output.parent / "prepare" / "started.txt"
    started = time.monotonic()
    if stage == "prepare":
        clock.write_text(str(started))
    if stage == "selector" and cfg.certification.timing_reference_seconds is not None:
        result = selector.experiment(cfg, output, started=float(clock.read_text()))
    else:
        result = STAGES[stage](cfg, output)
    timing = {"stage_seconds": time.monotonic() - started}
    if stage == "classify":
        timing["wall_seconds"] = time.monotonic() - float(clock.read_text())
        print(f"Experiment wall time: {timing['wall_seconds'] / 3600:.2f}h.", flush=True)
    timing_name = "runtime.json" if cfg.stage.fold is None else f"{cfg.stage.fold}_runtime.json"
    save_json(output / timing_name, timing)
    tracking.finish(result)
    return 0


def suite(overrides):
    cfg = compose_cfg("lion.certification", overrides, config_dir=str(ROOT / "conf"), config_name="certification")
    output = absolute(cfg.certification.results)
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for source, task in SUITE:
        started = time.monotonic()
        cmd = [sys.executable, "-B", "-m", "lion.certification.run",
               *overrides, f"certification.source={source}", f"certification.tasks=[{task}]"]
        print(f"Suite {source}/{task}: started", flush=True)
        code = subprocess.run(cmd, cwd=ROOT).returncode
        status = "completed" if code == 0 else "failed"
        summary.append({"source": source, "task": task, "status": status, "exit_code": code,
                        "seconds": round(time.monotonic() - started, 1)})
        write_csv(output / "suite.csv", summary)
        print(f"Suite {source}/{task}: {status} ({summary[-1]['seconds']}s)", flush=True)


def run_stage(args):
    config_dir = str(ROOT / "conf")
    if any(f"stage.experiment={stage}" in args for stage in ("selector", "certify", "clean")):
        cfg = compose_cfg("lion.certification", args, config_dir=config_dir, config_name="certification")
        groups = cfg.runtime.selector_cpu_groups
        if groups is not None and cfg.stage.fold is None:
            if cfg.runtime.device != "cpu":
                raise ValueError("CPU fold groups require runtime=cpu.")
            partitions = list(zip(("g1", "g2"), groups, strict=True))
            workers = [
                subprocess.Popen(["taskset", "-c", str(group), sys.executable, "-B", "-m", "lion.certification.run",
                                  *args, f"stage.fold={fold}"], cwd=ROOT)
                for fold, group in partitions
            ]
            codes = [worker.wait() for worker in workers]
            status = next((code for code in codes if code), 0)
            if not status and cfg.stage.experiment == "certify":
                certify.merge_tables(cfg)
            return status
    return run("lion.certification", args, main_module="run", config_dir=config_dir, config_name="certification")


if __name__ == "__main__":
    overrides = sys.argv[1:]
    if "--suite" in overrides:
        overrides.remove("--suite")
        suite(overrides)
    else:
        stages = [None] if any(arg.startswith("stage.experiment=") for arg in overrides) else STAGES
        for stage in stages:
            args = overrides if stage is None else [*overrides, f"stage.experiment={stage}"]
            code = run_stage(args)
            if code:
                raise SystemExit(code)
