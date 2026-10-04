"""Reset replay (paper Eq. 8) and isolated Operator execution (boundary replay, Eq. 12).

``replay(graph, episode, program, llm, run_dir)`` re-executes the *whole* workflow from a reset
state: a fresh run directory, an empty checkpoint, no transient state carried over from the
multi-turn session. It returns the scientific output y and the execution trace tau (all node
records in topological order, aggregated LLM usage and wall time). Evaluation (q, h, c) is done by
the caller with ``episode.evaluate(y, trace)``.

``run_operator_isolated(op, inputs, ...)`` executes one OperatorSpec on given boundary input
values in a fresh directory; evolution uses it to check that a candidate Operator reproduces the
recorded boundary outputs of tau+ (boundary replay).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..core.graph import WorkflowGraph
from ..core.operators import OperatorSpec
from ..core.trace import NodeRecord, Trace
from .executor import Executor, add_usage


def fresh_dir(run_dir: str | Path) -> Path:
    """``run_dir`` if it does not exist or is empty, else the first free sibling ``<name>-<i>``.

    Existing content is never removed.
    """
    p = Path(run_dir).resolve()
    if not p.exists() or (p.is_dir() and not any(p.iterdir())):
        p.mkdir(parents=True, exist_ok=True)
        return p
    i = 1
    while True:
        cand = p.with_name(f"{p.name}-{i}")
        if not cand.exists():
            cand.mkdir(parents=True)
            return cand
        i += 1


def replay(graph: WorkflowGraph, episode: Any, program: Any, llm: Any, run_dir: str | Path, *,
           max_parallel_subprocs: int = 4) -> tuple[Any, Trace]:
    """(y, tau) = Replay_{D_E}(G): full execution of ``graph`` from an empty checkpoint in a fresh dir."""
    rd = fresh_dir(run_dir)
    t0 = time.monotonic()
    ex = Executor(episode, program, llm, rd, max_parallel_subprocs=max_parallel_subprocs)
    _, records, y = ex.execute(graph, ex.new_checkpoint())
    usage: dict = {}
    for r in records.values():
        add_usage(usage, r.llm_usage)
    trace = Trace(records=dict(records), order=list(records), llm_usage=usage, wall_s=time.monotonic() - t0,
                  run_dir=str(rd))
    return y, trace


def run_operator_isolated(op: OperatorSpec, inputs: dict, program: Any, llm: Any, run_dir: str | Path,
                          episode: Any = None) -> tuple[dict, NodeRecord]:
    """Execute ``op`` alone on boundary ``inputs`` (fresh dir). Returns (boundary outputs, record).

    ``episode`` supplies tools (if the body contains tool nodes), the budget and the LLM tag; nested
    operator nodes inside the body are resolved from ``program``.
    """
    rd = fresh_dir(run_dir)
    ex = Executor(episode, program, llm, rd, max_parallel_subprocs=1)
    return ex.run_operator(op, inputs)
