"""Per-discipline before/after table over many probe runs (all episodes, not one row per discipline).

Usage: python scripts/dev/adaptation_table.py stock=runs/probe_local27b_a0,runs/probe_pass10_n3 adapted='runs/probe_for*_scilib*' [--md]
Each argument is LABEL=comma-separated dirs/globs; every episode dir with a result.json counts once (infra errors skipped).
Columns per label: passed/n, mean norm score, mean steps, mean total tokens.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path


def episodes(spec: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    seen = set()
    for pat in spec.split(","):
        for run in sorted(glob.glob(pat)):
            for res in sorted(Path(run).glob("*/result.json")):
                d = res.parent
                if d.resolve() in seen:
                    continue
                seen.add(d.resolve())
                r = json.loads(res.read_text())
                if r.get("infra_error"):
                    continue
                ev = json.loads((d / "eval.json").read_text()) if (d / "eval.json").exists() else {}
                us = json.loads((d / "usage.json").read_text()) if (d / "usage.json").exists() else {}
                out[d.name.split("-")[0]].append({
                    "pass": bool(r.get("passed")), "norm": (ev.get("details") or {}).get("norm_score"),
                    "steps": r.get("n_steps"), "tok": us.get("total_tokens"), "stop": r.get("stop_reason")})
    return out


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def cell(eps: list[dict] | None) -> list[str]:
    if not eps:
        return ["-", "-", "-", "-"]
    n = mean([e["norm"] for e in eps]); s = mean([e["steps"] for e in eps]); t = mean([e["tok"] for e in eps])
    return [f"{sum(e['pass'] for e in eps)}/{len(eps)}", "-" if n is None else f"{n:.2f}",
            "-" if s is None else f"{s:.0f}", "-" if t is None else f"{t/1000:.0f}k"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="LABEL=dir[,dir...] (globs allowed)")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    data = []
    for s in a.specs:
        label, _, spec = s.partition("=")
        data.append((label, episodes(spec)))
    codes = sorted({c for _, d in data for c in d})
    head = ["disc"] + [f"{l}:{c}" for l, _ in data for c in ("pass", "norm", "steps", "tok")]
    sep = " | " if a.md else "  "
    if a.md:
        print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    else:
        print(sep.join(head))
    for c in codes:
        row = [c] + [x for _, d in data for x in cell(d.get(c))]
        print(("| " + sep.join(row) + " |") if a.md else sep.join(row))


if __name__ == "__main__":
    main()
