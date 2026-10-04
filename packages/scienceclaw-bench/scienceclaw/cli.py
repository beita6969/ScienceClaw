"""Command line interface: ``python -m scienceclaw.cli <command> ...``

Commands
  list-tasks                         adapter availability table (23 ANZSRC disciplines + TOY)
  smoke [--out DIR] [--rounds N]     offline end-to-end run (TOY adapter + scripted fake policy, no network)
  evolve --config X.yaml [--set k=v ...] [--name NAME]
                                     run the evolution stream A_0 -> A_R (writes runs/<name>-<timestamp>/)
  evaluate --run DIR [--snapshots all|A_0,A_7] [--splits id,ood] [--no-rep] [--workers 8]
           [--family-transfer SPLIT]
                                     frozen-snapshot evaluation on D_ID / D_OOD (+ D_rep, + family programs)
  report --run DIR [--pi-reference A_0|method_final] [--normalization ratio|linear]
                                     tables (report/report.md + CSVs) from the run receipts
  compare --runs LABEL=DIR ... [--split ood]
                                     average ranks, Friedman/Nemenyi and sign tests across runs (methods)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .config import BenchConfig, RunConfig, load_config


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
    n_av = sum(r["available"] == "yes" for r in rows if r["code"] != "TOY")
    print(f"\n{n_av}/{len(rows) - 1} registry disciplines available (data root: {cfg.data_root})")
    return 0


SMOKE_SEED = 1   # with the default scripted policy this seed exercises a promotion (A_0 fails some held-out episodes)
SMOKE_MAX_STEPS = 16   # the scripted build (8 edits) + node-replacement repair (6 edits) + finish


def smoke_adapters() -> dict:
    """TOY adapter whose episode budget allows the scripted policy's full build + repair."""
    from .bench.task import Budget
    from .bench.tasks.toy import ToyAdapter

    return {"TOY": ToyAdapter(budget=Budget(max_steps=SMOKE_MAX_STEPS))}


def smoke_config(out: Path, rounds: int = 2, seed: int = SMOKE_SEED) -> RunConfig:
    """Tiny TOY-only configuration for the offline smoke run (2 rounds, 1 val, 2 id, 2 ood episodes)."""
    cfg = RunConfig(name="smoke", runs_root=str(out))
    cfg.bench = BenchConfig(disciplines=["TOY"], items_per_episode=8, rounds=rounds, n_val=1, n_id=2, n_ood=2,
                            seed=seed)
    cfg.solver.max_steps = SMOKE_MAX_STEPS
    cfg.evolution.min_improved_episodes = 1       # one val episode: the noise guard (>= 2 improved) could never pass
    cfg.evolution.budget_beta = 2.0               # the scripted A_0 gives up after 8 edits (~24k tokens) where a repaired
    #                                               episode needs 14 (~50k): keep the relative cap, but loose enough for that
    cfg.llm.use_cache = False
    cfg.llm.cache_path = ""
    return cfg


_OP_CARD_RE = re.compile(r"\bop:([A-Za-z0-9_\-\.]*[A-Za-z0-9_\-])\(([^\n]*?)\) -> \(([^\n]*?)\)")
_PORT_NAME_RE = re.compile(r"(?:^|, )([A-Za-z_][A-Za-z0-9_]*): ")
_TOY_MODEL_NODE = "fit2"


def _port_by_suffix(ports: list[str], name: str) -> str | None:
    return next((p for p in ports if p == name or p.endswith("__" + name)), None)


def operator_instantiating(base: Any) -> Any:
    """Wrap ``bench.tasks.toy.scripted_toy_responder`` so that a *retrieved* fit-and-predict Operator is instantiated.

    The scripted policy only cites the ``skill:``/``op:`` references it sees in ``uses``; it never places an
    operator node. Since Use(omega) is judged on the passing evidence (an Operator counts as used only if an
    operator node with that ref is in the passing graph and ran ok), a cited-but-unused Operator makes every
    smoke candidate fail R_src. When the prompt carries an Operator card whose ports match the model node
    (``*X_train``, ``*y_train``, ``*X_eval`` -> ``*y_pred``), the model node of the scripted repair
    (``fit2``) becomes an ``operator`` node with that ref and the wiring actions are re-targeted to the
    Operator's port names; the canvas shown to the scripted policy is de-prefixed so its script is unchanged.
    """
    def _text(m: dict) -> str:
        c = m.get("content", "")
        if isinstance(c, list):
            return "\n".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in c)
        return str(c)

    def respond(role: str, messages: list[dict]) -> str:
        if role != "policy":
            return base(role, messages)
        system = _text(messages[0]) if messages and messages[0].get("role") == "system" else ""
        card = _OP_CARD_RE.search(system)
        if card is None:
            return base(role, messages)
        oid = card.group(1)
        ins = _PORT_NAME_RE.findall(card.group(2))
        outs = _PORT_NAME_RE.findall(card.group(3))
        pmap_in = {n: _port_by_suffix(ins, n) for n in ("X_train", "y_train", "X_eval")}
        pmap_out = {"y_pred": _port_by_suffix(outs, "y_pred")}
        if not all(pmap_in.values()) or not all(pmap_out.values()):
            return base(role, messages)
        strip = re.compile(r"\b" + _TOY_MODEL_NODE + r"\.(?:" + "|".join(
            re.escape(v) for v in [*pmap_in.values(), *pmap_out.values()]) + r")\b")
        seen = [dict(m, content=strip.sub(lambda mo: _TOY_MODEL_NODE + "." + mo.group(0).split(".", 1)[1].split("__")[-1], _text(m)))
                if m.get("role") != "system" else m for m in messages]
        reply = json.loads(base(role, seen))
        act = reply.get("action") or {}
        if act.get("type") == "add_node" and (act.get("node") or {}).get("id") == _TOY_MODEL_NODE:
            act["node"] = {"id": _TOY_MODEL_NODE, "kind": "operator", "ref": f"op:{oid}"}
        elif act.get("type") == "add_edge":
            e = act["edge"]
            if e.get("dst") == _TOY_MODEL_NODE and pmap_in.get(e.get("dst_port")):
                e["dst_port"] = pmap_in[e["dst_port"]]
            if e.get("src") == _TOY_MODEL_NODE and pmap_out.get(e.get("src_port")):
                e["src_port"] = pmap_out[e["src_port"]]
        return json.dumps(reply)

    return respond


def run_smoke(out: Path, rounds: int = 2, workers: int = 1, verbose: bool = True, seed: int = SMOKE_SEED,
              repair: str = "library_or_hash") -> dict:
    """Offline end-to-end: evolve on TOY with a scripted fake policy, evaluate all snapshots, build the report.

    No network access: the LLM is ``llm.fake.FakeLLM`` driven by ``bench.tasks.toy.scripted_toy_responder``.
    """
    from .bench.tasks.toy import scripted_toy_responder
    from .experiments.evaluate import evaluate_snapshots, read_results
    from .experiments.report import build_report
    from .experiments.run_stream import run_stream, snapshot_dirs
    from .llm.fake import FakeLLM

    cfg = smoke_config(out, rounds, seed)
    llm = FakeLLM(operator_instantiating(scripted_toy_responder(repair)), cfg=cfg.llm)
    adapters = smoke_adapters()
    run_dir = run_stream(cfg, adapters=adapters, llm=llm)
    res_path = evaluate_snapshots(run_dir, "all", llm=llm, workers=workers, adapters=adapters,
                                  progress=print if verbose else None)
    md = build_report(run_dir, adapters=adapters, n_boot=200)
    results = read_results(res_path)
    rep = json.loads((run_dir / "report" / "report.json").read_text())
    return {"run_dir": str(run_dir), "snapshots": [f"A_{r}" for r in snapshot_dirs(run_dir)],
            "n_results": len(results), "report": str(md), "summary": rep["summary"],
            "promotions": rep["promotions"], "llm_calls": len(llm.calls)}


def cmd_smoke(args: argparse.Namespace) -> int:
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="scienceclaw-smoke-"))
    info = run_smoke(out, rounds=args.rounds, workers=args.workers, seed=args.seed)
    print(f"\nrun dir : {info['run_dir']}\nsnapshots: {info['snapshots']}\nresults : {info['n_results']}"
          f"\nreport  : {info['report']}\nfake LLM calls: {info['llm_calls']}")
    for r in info["summary"]:
        pi = r.get("pi")
        pi_s = f"{pi:.1f}" if isinstance(pi, (int, float)) and pi == pi else "-"
        print(f"  {r['snapshot']:>5} {r['split']:>4}  MacroSR={r['macro_sr']:.3f}  PI(vs A_0)={pi_s}")
    tot = next((p for p in info["promotions"] if p["round"] == "total"), None)
    if tot:
        print(f"  promotions: {tot['promoted']}/{tot['candidates']} candidates")
    ok = bool(info["snapshots"]) and info["n_results"] > 0
    return 0 if ok else 1


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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m scienceclaw.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("list-tasks", help="adapter availability table")
    s.add_argument("--data-root", default=None)
    s.set_defaults(fn=cmd_list_tasks)

    s = sub.add_parser("smoke", help="offline end-to-end run with the TOY adapter and a scripted fake policy")
    s.add_argument("--out", default=None, help="runs root (default: a new temporary directory)")
    s.add_argument("--rounds", type=int, default=2)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--seed", type=int, default=SMOKE_SEED)
    s.set_defaults(fn=cmd_smoke)

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
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
