"""Side-by-side table of probe runs (one episode dir per discipline: result.json, eval.json, usage.json).

Usage: python scripts/summarize_probe.py runs/probe_local27b_a0 [runs/probe_all_low ...] [--md]
Columns per run: pass, primary, norm, steps, stop reason, total tokens.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(run: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for res in sorted(run.glob("*/result.json")):
        d = res.parent
        r = json.loads(res.read_text())
        ev = json.loads((d / "eval.json").read_text()) if (d / "eval.json").exists() else {}
        us = json.loads((d / "usage.json").read_text()) if (d / "usage.json").exists() else {}
        out[d.name.split("-")[0]] = {
            "pass": bool(r.get("passed")),
            "primary": r.get("primary"),
            "norm": (ev.get("details") or {}).get("norm_score"),
            "steps": r.get("n_steps"),
            "stop": r.get("stop_reason"),
            "tokens": us.get("total_tokens"),
            "within_budget": r.get("within_budget"),
            "infra": r.get("infra_error"),
        }
    return out


def fmt(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    data = [(Path(r).name, load(Path(r))) for r in a.runs]
    codes = sorted({c for _, d in data for c in d})
    sep = " | " if a.md else "  "
    head = ["disc"] + [f"{n}:{c}" for n, _ in data for c in ("pass", "primary", "norm", "steps", "stop", "tok")]
    if a.md:
        print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    else:
        print(sep.join(head))
    for c in codes:
        row = [c]
        for _, d in data:
            v = d.get(c)
            row += ["-"] * 6 if v is None else [
                "PASS" if v["pass"] else "fail", fmt(v["primary"]), fmt(v["norm"]), fmt(v["steps"]), fmt(v["stop"]),
                fmt(v["tokens"]),
            ]
        print(("| " + sep.join(row) + " |") if a.md else sep.join(row))
    for n, d in data:
        print(f"# {n}: {sum(v['pass'] for v in d.values())}/{len(d)} passed")


if __name__ == "__main__":
    main()
