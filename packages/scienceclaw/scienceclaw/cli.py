"""Command line interface: ``python -m scienceclaw.cli <command> ...``

Commands
  tools search QUERY [--k N] [--kind pretrained|library] [--available]
  tools show ID|MODULE [--full]      tool library: retrieve, describe and probe the scilib tools
  tools status [MODULE ...]          which tool modules can run here, and why not
  weights status | plan [ID ...] [--root DIR] | verify [ID ...]
                                     pretrained weights: what is staged, how to stage the rest, hash checks
  setup [--profile full|light] [--check]
                                     install and verify every tool (Python packages, weights, upstream sources)
  doctor                             setup state and availability of every library module
  live candidates | show ID | promote ID | rollback VERSION | history
                                     review, promote and roll back the program changes learned from live sessions
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def _print_table(rows: list[dict], cols: list[str]) -> None:
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


# ------------------------------------------------------------------------------------------------ commands
def cmd_tools(args: argparse.Namespace) -> int:
    from . import tools

    if args.action == "search":
        for e in tools.search(args.query, args.k, kind=args.kind, available_only=args.available):
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


def cmd_live(args: argparse.Namespace) -> int:
    """Review, promote and roll back the program changes learned from live sessions (the user's side of the gate)."""
    from .config import RunConfig
    from .program.store import ProgramStore
    from .evolution.live import LiveEvolution

    home = Path(args.home or os.environ.get("SCIENCECLAW_HOME") or Path.home() / ".scienceclaw").expanduser()
    store = ProgramStore(home / "program")
    evo = LiveEvolution(store, home, None, RunConfig(), [])
    if args.action == "candidates":
        rows = [{"id": c["id"], "status": c["status"], "variant": c["variant"], "decision": str(c["decision"])[:70]}
                for c in evo.candidates()]
        _print_table(rows, ["id", "status", "variant", "decision"])
        print(f"\nactive program: {store.head()}")
    elif args.action == "show":
        print(json.dumps(evo.candidate(args.target), indent=1, default=str))
    elif args.action == "promote":
        res = evo.promote(args.target)
        print(json.dumps({k: res[k] for k in ("candidate", "status", "reason")}, indent=1))
    elif args.action == "rollback":
        print(f"active program: {store.rollback(args.target)}")
    else:
        for h in store.history():
            print(json.dumps({k: h.get(k) for k in ("time", "version", "event", "candidate", "parent", "from", "to")}, default=str))
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    """Install and verify every tool of the library (Python packages, pretrained weights, upstream sources)."""
    from . import bootstrap

    def progress(p: dict) -> None:
        extra = f" ({p['index']}/{p['total']}, ~{p['approx_gb']} GB)" if "index" in p else ""
        print(f"[setup] {p.get('asset') or p['step']}{extra}: {p.get('message', '')}".rstrip(": "), flush=True)

    st = bootstrap.run(args.profile, args.only or None, args.skip or None, args.with_optional, check_only=args.check,
                       progress=None if args.check else progress)
    if args.check:
        print(json.dumps({k: st[k] for k in ("profile", "model_root", "to_stage", "download_gb", "free_gb", "complete")}, indent=1))
        return 0 if st["complete"] else 1
    if st.get("error"):
        print(f"[setup] {st['error']}", file=sys.stderr)
        return 2
    for r in st["results"]:
        print(f"  {r['id']:28s} {r['status']:7s} {r.get('seconds', '')}s" + ("" if r["status"] == "ok" else f"  see {r.get('log')}"))
    gaps = st.get("modules") or []
    if gaps:
        print("\nmodules that still cannot run here:")
        _print_table([{"module": m["module"], "reason": str(m["reason"])[:100]} for m in gaps], ["module", "reason"])
    print("\nsetup complete" if st["complete"] else f"\nsetup INCOMPLETE: assets {st.get('missing_assets')}, "
          f"packages {st['python']['missing_modules']}")
    return 0 if st["complete"] else 1


def cmd_doctor(args: argparse.Namespace) -> int:
    from . import bootstrap, tools

    st = bootstrap.status()
    print(f"setup: {st['state']}" + (f" - {st['detail']}" if st.get("detail") else ""))
    rows = [{"module": r["module"], "kind": r["kind"], "available": "yes" if r["available"] else "no", "reason": str(r["reason"])[:90]}
            for r in tools.status_table()]
    _print_table(rows, ["module", "kind", "available", "reason"])
    return 0 if st["complete"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m scienceclaw.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("tools", help="search, describe and probe the scientific tool library")
    ts = s.add_subparsers(dest="action", required=True)
    a = ts.add_parser("search")
    a.add_argument("query")
    a.add_argument("--k", type=int, default=8)
    a.add_argument("--kind", choices=("pretrained", "library"), default=None)
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

    s = sub.add_parser("setup", help="install and verify every tool: Python packages, pretrained weights, upstream sources")
    s.add_argument("--profile", choices=("full", "light"), default=os.environ.get("SCIENCECLAW_SETUP_PROFILE", "full"),
                   help="light leaves out assets larger than 1.5 GB")
    s.add_argument("--only", nargs="*", default=None, help="stage only these weight assets")
    s.add_argument("--skip", nargs="*", default=None, help="leave these weight assets out")
    s.add_argument("--with-optional", action="store_true", help="also stage assets no wrapper needs")
    s.add_argument("--check", action="store_true", help="only report what is missing (exit 1 if anything is)")
    s.set_defaults(fn=cmd_setup)

    s = sub.add_parser("doctor", help="setup state and availability of every library module")
    s.set_defaults(fn=cmd_doctor)

    s = sub.add_parser("live", help="candidates learned from live sessions: review, promote, roll back")
    s.add_argument("--home", default=None, help="engine state directory (default $SCIENCECLAW_HOME or ~/.scienceclaw)")
    ls = s.add_subparsers(dest="action", required=True)
    ls.add_parser("candidates")
    ls.add_parser("history")
    for name in ("show", "promote", "rollback"):
        a = ls.add_parser(name)
        a.add_argument("target", help="candidate id (show, promote) or program version (rollback)")
    s.set_defaults(fn=cmd_live)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
