"""Configuration and driver for the end-to-end pipeline test (synthetic task, scripted policy, no network)."""
from __future__ import annotations

import json
from pathlib import Path

from scienceclaw.config import BenchConfig, RunConfig

from scripted_policy import operator_instantiating, scripted_responder
from synthetic_adapter import SYN_CODE, SyntheticAdapter

PIPELINE_SEED = 0       # with the default scripted policy this seed exercises a promotion (A_0 fails some held-out episodes)
PIPELINE_MAX_STEPS = 16  # scripted build (8 edits) + node-replacement repair (6 edits) + finish


def pipeline_adapters() -> dict:
    """Synthetic adapter whose episode budget allows the scripted policy's full build + repair."""
    from scienceclaw.bench.task import Budget

    return {SYN_CODE: SyntheticAdapter(budget=Budget(max_steps=PIPELINE_MAX_STEPS))}


def pipeline_config(out: Path, rounds: int = 2, seed: int = PIPELINE_SEED) -> RunConfig:
    """Small single-discipline configuration (``rounds`` rounds, 1 val, 2 id, 2 ood episodes)."""
    cfg = RunConfig(name="pipeline", runs_root=str(out))
    cfg.bench = BenchConfig(disciplines=[SYN_CODE], items_per_episode=8, rounds=rounds, n_val=1, n_id=2, n_ood=2,
                            seed=seed)
    cfg.solver.max_steps = PIPELINE_MAX_STEPS
    cfg.evolution.min_improved_episodes = 1       # one val episode: the noise guard (>= 2 improved) could never pass
    cfg.evolution.budget_beta = 2.0               # an unrepaired episode stops after 8 edits (~24k tokens), a repaired one needs 14 (~50k)
    cfg.llm.use_cache = False
    cfg.llm.cache_path = ""
    return cfg


def run_pipeline(out: Path, rounds: int = 2, workers: int = 1, seed: int = PIPELINE_SEED,
                 repair: str = "library_or_hash") -> dict:
    """Evolve with the scripted fake policy, evaluate all snapshots and build the report."""
    from scienceclaw.experiments.evaluate import evaluate_snapshots, read_results
    from scienceclaw.experiments.report import build_report
    from scienceclaw.experiments.run_stream import run_stream, snapshot_dirs
    from scienceclaw.llm.fake import FakeLLM

    cfg = pipeline_config(out, rounds, seed)
    llm = FakeLLM(operator_instantiating(scripted_responder(repair)), cfg=cfg.llm)
    adapters = pipeline_adapters()
    run_dir = run_stream(cfg, adapters=adapters, llm=llm)
    res_path = evaluate_snapshots(run_dir, "all", llm=llm, workers=workers, adapters=adapters)
    md = build_report(run_dir, adapters=adapters, n_boot=200)
    results = read_results(res_path)
    rep = json.loads((run_dir / "report" / "report.json").read_text())
    return {"run_dir": str(run_dir), "snapshots": [f"A_{r}" for r in snapshot_dirs(run_dir)],
            "n_results": len(results), "report": str(md), "summary": rep["summary"],
            "promotions": rep["promotions"], "llm_calls": len(llm.calls)}
