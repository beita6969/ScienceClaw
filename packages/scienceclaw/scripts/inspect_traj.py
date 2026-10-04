"""Print a compact step-by-step view of solver trajectories.

Usage: python scripts/inspect_traj.py <trajectory.jsonl | solve dir | run dir> [--grep FoR34] [--full]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def describe(action: dict | None) -> str:
    if not action:
        return "(no action)"
    t, p = action.get("type"), action.get("payload") or {}
    if t == "add_node":
        n = p.get("node", {})
        return f"add_node {n.get('id')}[{n.get('kind')}{':' + str(n.get('ref')) if n.get('ref') else ''}]"
    if t in ("add_edge", "remove_edge"):
        e = p.get("edge", {})
        return f"{t} {e.get('src')}.{e.get('src_port')} -> {e.get('dst')}.{e.get('dst_port')}"
    if t == "modify_node":
        return f"modify_node {p.get('id')} {sorted((p.get('patch') or {}).keys())}"
    if t == "remove_node":
        return f"remove_node {p.get('id')}"
    if t == "batch":
        return "batch[" + "; ".join(describe(a) for a in p.get("actions", [])) + "]"
    return str(t)


def show(path: Path, full: bool) -> None:
    print(f"===== {path}")
    for line in path.read_text().splitlines():
        r = json.loads(line)
        fb = r.get("feedback") or {}
        flag = "OK " if fb.get("action_ok") else "ERR"
        print(f" k={r.get('step'):>2} {flag} {describe(r.get('action'))[:110]}")
        if r.get("parse_error"):
            print(f"      parse_error: {r['parse_error'][:300]}")
        if fb.get("action_error"):
            print(f"      action_error: {fb['action_error'][:400]}")
        for nid, rec in (fb.get("records") or {}).items():
            if rec.get("status") not in ("ok",) and not rec.get("cached"):
                print(f"      node {nid}: {rec.get('status')} {(rec.get('error') or '')[:400 if full else 200]}")
        if fb.get("integrity_violations"):
            print(f"      integrity: {fb['integrity_violations']}")
        if fb.get("submit_ready"):
            print(f"      SUBMIT ready; visible constraints: {fb.get('visible_constraints')}; dev: {str(fb.get('dev'))[:200]}")
        if r.get("evidence_idx") is not None:
            print(f"      evidence #{r['evidence_idx']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--grep", default="")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    p = Path(a.path)
    files = [p] if p.is_file() else sorted(p.rglob("trajectory.jsonl"))
    for f in files:
        if a.grep in str(f):
            show(f, a.full)


if __name__ == "__main__":
    main()
