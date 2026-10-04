"""Solve episodes with a given program (default A_0) on real adapters and print outcomes.

Usage: python scripts/probe.py --config configs/dev5_api.yaml [--disciplines FoR34,FoR46] [--split src] [--n 1]
       [--mode eval] [--out runs/probe] [--workers 5] [--max-steps 14]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time
from pathlib import Path

from scienceclaw.agent.solver import Solver
from scienceclaw.bench.splits import SplitPlan, load_adapters
from scienceclaw.config import load_config
from scienceclaw.core.program import AgentProgram
from scienceclaw.llm.client import LLMClient


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--disciplines", default="")
    ap.add_argument("--split", default="src")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--mode", default="eval")
    ap.add_argument("--out", default="runs/probe")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--required-tool", action="append", default=[],
                    help="formal required task tool; repeat for multiple tools")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.skip < 0:
        ap.error("--skip must be non-negative")
    if a.max_steps:
        cfg.solver.max_steps = a.max_steps
    if a.disciplines:
        cfg.bench.disciplines = a.disciplines.split(",")
    # Build enough episodes for a prefix skip before selecting the requested
    # suffix. This makes independent follow-up formal batches possible without
    # changing the split seed or reusing the original item IDs.
    count_attr = {"val": "n_val", "id": "n_id", "ood": "n_ood"}.get(a.split)
    if count_attr:
        setattr(cfg.bench, count_attr, max(int(getattr(cfg.bench, count_attr)), a.skip + a.n))
    cfg.bench.rounds = max(cfg.bench.rounds, a.skip + a.n)
    adapters = load_adapters(cfg.bench)
    plan = SplitPlan.build(cfg.bench, adapters)
    for w in plan.warnings:
        print("WARN", w)
    llm = LLMClient(cfg.llm)
    program = AgentProgram()
    out = Path(a.out).resolve()
    jobs = []
    for code in plan.order:
        for ep in plan.episodes[a.split].get(code, [])[a.skip:a.skip + a.n]:
            if a.required_tool:
                refs = ", ".join(a.required_tool)
                ep.objective += (
                    "\n\nFORMAL TOOL REQUIREMENT: this run requires the explicit task tool(s) "
                    f"{refs}. Call the required tool on the visible inputs, wire its output directly "
                    "to submit.y, and finish immediately. Do not substitute another predictor tool, "
                    "a manual heuristic, or a later code node."
                )
            jobs.append(ep)

    def run(ep):
        t0 = time.time()
        res = Solver(cfg.solver, llm, cfg.evolution).solve(ep, program, a.mode, out / ep.id / "work")
        res.save(out / ep.id)
        return ep, res, time.time() - t0

    with cf.ThreadPoolExecutor(a.workers) as pool:
        for ep, res, wall in pool.map(run, jobs):
            ev = res.eval
            print(f"{ep.discipline} {ep.id}: z={res.z} passed={res.passed} primary={getattr(ev, 'primary', None)} "
                  f"norm={ (ev.details or {}).get('norm_score')} completed={ev.completed} hard_ok={ev.hard_ok()} "
                  f"accepted={ev.accepted} repro={ev.reproducible} steps={len(res.steps)} stop={res.stop_reason} "
                  f"wall={wall:.0f}s usage={ {k: v for k, v in res.usage.items() if isinstance(v, (int, float))} }")
            if ev.h:
                print("   h:", {k: v for k, v in ev.h.items()}, {k: v for k, v in (ev.h_msgs or {}).items() if not ev.h.get(k, True)})


if __name__ == "__main__":
    main()
