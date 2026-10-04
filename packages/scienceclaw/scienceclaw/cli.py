"""Command line interface: ``python -m scienceclaw.cli <command> ...``

Commands
  list-tasks                         adapter availability table (23 ANZSRC disciplines)
  evolve --config X.yaml [--set k=v ...] [--name NAME]
                                     run the evolution stream A_0 -> A_R (writes runs/<name>-<timestamp>/)
  evaluate --run DIR [--snapshots all|A_0,A_7] [--splits id,ood] [--no-rep] [--workers 8]
           [--family-transfer SPLIT]
                                     frozen-snapshot evaluation on D_ID / D_OOD (+ D_rep, + family programs)
  report --run DIR [--pi-reference A_0|method_final] [--normalization ratio|linear]
                                     tables (report/report.md + CSVs) from the run receipts
  compare --runs LABEL=DIR ... [--split ood]
                                     average ranks, Friedman/Nemenyi and sign tests across runs (methods)
  tools search QUERY [--k N] [--kind pretrained|library] [--task FoR37] [--available]
  tools show ID|MODULE [--full]      tool library: retrieve, describe and probe the scilib tools
  tools status [MODULE ...]          which tool modules can run here, and why not
  weights status | plan [ID ...] [--root DIR] | verify [ID ...]
                                     pretrained weights: what is staged, how to stage the rest, hash checks
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from .config import BenchConfig, load_config


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    """``["bench.rounds=3", "llm.policy.model=x"]`` -> {"bench.rounds": 3, ...} (values parsed as YAML scalars)."""
    out: dict[str, Any] = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--set expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = yaml.safe_load(v) if v.strip() != "" else ""
    return out


def _print_table(rows: list[dict], cols: list[str]) -> None:
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


# ------------------------------------------------------------------------------------------------ commands
def cmd_list_tasks(args: argparse.Namespace) -> int:
    from .bench.splits import adapter_status

    cfg = BenchConfig()
    if args.data_root:
        cfg.data_root = args.data_root
    rows = adapter_status(cfg)
    for r in rows:
        r["implemented"] = "yes" if r["implemented"] else "no"
        r["available"] = "yes" if r["available"] else "no"
        r["reason"] = str(r["reason"])[:70]
    _print_table(rows, ["code", "family", "metric", "direction", "implemented", "available", "reason"])
    n_av = sum(r["available"] == "yes" for r in rows)
    print(f"\n{n_av}/{len(rows)} registry disciplines available (data root: {cfg.data_root})")
    return 0


def cmd_evolve(args: argparse.Namespace) -> int:
    from .experiments.run_stream import run_stream

    if args.resume:
        run_dir = run_stream(None, resume_dir=args.resume)
        print(run_dir)
        return 0
    if not args.config:
        raise SystemExit("evolve needs --config (or --resume RUN_DIR)")
    overrides = parse_overrides(args.set)
    if args.name:
        overrides["name"] = args.name
    cfg = load_config(args.config, overrides)
    for w in cfg.gate_warnings():
        print(f"WARNING: {w}", file=sys.stderr)
    run_dir = run_stream(cfg)
    print(run_dir)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .experiments.evaluate import evaluate_family_transfer, evaluate_snapshots

    snaps: Any = "all" if args.snapshots in (None, "all") else [s.strip() for s in args.snapshots.split(",") if s.strip()]
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    path = evaluate_snapshots(args.run, snaps, splits, workers=args.workers, include_rep=not args.no_rep,
                              progress=print)
    if args.family_transfer:
        evaluate_family_transfer(args.run, split=args.family_transfer, workers=args.workers, progress=print)
    print(path)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .experiments.report import build_report

    md = build_report(args.run, pi_reference=args.pi_reference, normalization=args.normalization)
    print(md)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from .experiments.report import compare_runs

    runs = {}
    for it in args.runs:
        label, _, d = it.partition("=")
        runs[label if d else Path(label).name] = d or label
    res = compare_runs(runs, split=args.split)
    print(json.dumps({k: v for k, v in res.items() if k != "scores"}, indent=1, default=str))
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    from . import tools

    if args.action == "search":
        for e in tools.search(args.query, args.k, kind=args.kind, task=args.task, available_only=args.available):
            print(e.card() + "\n")
    elif args.action == "show":
        try:
            print(tools.get(args.target).card(full=True))
        except KeyError:
            print(f"Library `scilib.{args.target}`:\n{tools.module_doc(args.target)}")
            print("\nfunctions: " + ", ".join(e.name for e in tools.catalog() if e.module == args.target))
    else:
        rows = tools.status_table(args.modules or None)
        for r in rows:
            r["available"] = "yes" if r["available"] else "no"
            r["reason"] = str(r["reason"])[:90]
        _print_table(rows, ["module", "kind", "available", "reason"])
    return 0


def cmd_weights(args: argparse.Namespace) -> int:
    from .tools import weights as W

    if args.action == "status":
        rows = [{"id": s["id"], "staged": "yes" if s["present"] else "no", "ready": "yes" if s["ready"] else "no",
                 "missing packages": ",".join(s["missing_packages"]), "used by": ",".join(s["used_by"])} for s in W.status_all()]
        _print_table(rows, ["id", "staged", "ready", "missing packages", "used by"])
        print(f"\nmodel root: {W.model_root()}")
    elif args.action == "plan":
        print("\n".join(W.plan(args.ids or None, args.root)))
    else:
        for a in (args.ids or [x.id for x in W.load()]):
            print(json.dumps(W.verify(a)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m scienceclaw.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("list-tasks", help="adapter availability table")
    s.add_argument("--data-root", default=None)
    s.set_defaults(fn=cmd_list_tasks)

    s = sub.add_parser("evolve", help="evolve A_0 -> A_R over the source stream")
    s.add_argument("--config", default=None)
    s.add_argument("--resume", default=None, metavar="RUN_DIR", help="continue an interrupted run in place")
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="dotted config override")
    s.add_argument("--name", default=None)
    s.set_defaults(fn=cmd_evolve)

    s = sub.add_parser("evaluate", help="evaluate frozen snapshots on held-out splits")
    s.add_argument("--run", required=True)
    s.add_argument("--snapshots", default="all", help="'all' or comma list, e.g. A_0,A_7")
    s.add_argument("--splits", default="id,ood")
    s.add_argument("--no-rep", action="store_true", help="skip D_rep retention evaluation")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--family-transfer", default=None, metavar="SPLIT",
                   help="also evaluate per-source-family programs on SPLIT (e.g. ood)")
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("report", help="build report tables from eval/results.jsonl")
    s.add_argument("--run", required=True)
    s.add_argument("--pi-reference", default="A_0", help="snapshot name or 'method_final'")
    s.add_argument("--normalization", default="ratio", choices=("ratio", "linear"))
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("compare", help="compare runs (methods) on one split")
    s.add_argument("--runs", nargs="+", required=True, metavar="LABEL=DIR")
    s.add_argument("--split", default="ood")
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("tools", help="search, describe and probe the scientific tool library")
    ts = s.add_subparsers(dest="action", required=True)
    a = ts.add_parser("search")
    a.add_argument("query")
    a.add_argument("--k", type=int, default=8)
    a.add_argument("--kind", choices=("pretrained", "library"), default=None)
    a.add_argument("--task", default=None, help="limit to the tools used by a discipline, e.g. FoR37")
    a.add_argument("--available", action="store_true", help="only tools that can run here")
    a = ts.add_parser("show")
    a.add_argument("target", help="tool id (tsfm.forecast) or module (tsfm)")
    a.add_argument("--full", action="store_true")
    a = ts.add_parser("status")
    a.add_argument("modules", nargs="*")
    s.set_defaults(fn=cmd_tools)

    s = sub.add_parser("weights", help="pretrained weights: status, staging plan, hash verification")
    ws = s.add_subparsers(dest="action", required=True)
    ws.add_parser("status")
    a = ws.add_parser("plan")
    a.add_argument("ids", nargs="*")
    a.add_argument("--root", default=None)
    a = ws.add_parser("verify")
    a.add_argument("ids", nargs="*")
    s.set_defaults(fn=cmd_weights)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
