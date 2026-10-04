"""Experiment drivers: evolve (run_stream), frozen-snapshot evaluation (evaluate) and reporting (report).

Heavy dependencies (LLM client, solver, evolver) are imported lazily inside the functions.
"""
from .evaluate import evaluate_family_transfer, evaluate_snapshots, read_results
from .report import build_report, compare_runs
from .run_stream import run_stream

__all__ = ["run_stream", "evaluate_snapshots", "evaluate_family_transfer", "read_results", "build_report",
           "compare_runs"]
