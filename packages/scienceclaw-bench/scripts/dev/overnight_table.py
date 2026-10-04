"""Merge the probe evidence (stock vs adapted A_0) with a full run's report into one per-discipline markdown table.

Usage: python scripts/dev/overnight_table.py RUN_DIR [--stock DIRS] [--adapted GLOBS]
RUN_DIR: a finished (or partly evaluated) run with report/per_discipline.csv.
Columns: stock probe pass, adapted probe pass, then for id and ood the success rate / mean normalised score of the first
and the last evaluated snapshot (n episodes in brackets).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from adaptation_table import episodes  # noqa: E402


def frac(rows: list[dict]) -> str:
    return f"{sum(r['pass'] for r in rows)}/{len(rows)}" if rows else "-"


def load_report(run: Path) -> dict[tuple[str, str, str], dict]:
    path = run / "report" / "per_discipline.csv"
    if not path.exists():
        return {}
    return {(r["snapshot"], r["split"], r["discipline"]): r for r in csv.DictReader(path.open())}


def cell(r: dict | None) -> str:
    if not r:
        return "-"
    nm = r.get("mean_norm_score")
    nm = f"{float(nm):.2f}" if nm not in (None, "") else "?"
    return f"{float(r['success_rate']):.2f} / {nm} (n={r['n']})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--stock", default="runs/probe_local27b_a0,runs/probe_pass10_n3")
    ap.add_argument("--adapted", default="runs/probe_for*_scilib*,runs/probe_for48_fix")
    a = ap.parse_args()
    run = Path(a.run)
    rep = load_report(run)
    snaps = sorted({k[0] for k in rep}, key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else -1)
    first, last = (snaps[0], snaps[-1]) if snaps else ("A_0", "A_0")
    stock, adapted = episodes(a.stock), episodes(a.adapted)
    discs = sorted(set(stock) | set(adapted) | {k[2] for k in rep})
    print(f"| disc | stock A_0 probe (src) | adapted A_0 probe (src) | id {first} sr/norm | id {last} sr/norm | "
          f"ood {first} sr/norm | ood {last} sr/norm |")
    print("|---|---|---|---|---|---|---|")
    for d in discs:
        print(f"| {d} | {frac(stock.get(d, []))} | {frac(adapted.get(d, []))} | {cell(rep.get((first, 'id', d)))} | "
              f"{cell(rep.get((last, 'id', d)))} | {cell(rep.get((first, 'ood', d)))} | {cell(rep.get((last, 'ood', d)))} |")


if __name__ == "__main__":
    main()
