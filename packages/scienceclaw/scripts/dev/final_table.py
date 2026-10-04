"""One per-discipline table: stock A_0 probe, frozen-run A_0 (val/id/ood) and a re-probe of A_0 with the final code.

Usage: python scripts/dev/final_table.py RUN_DIR [--stock DIRS] [--final TAG,TAG,...]
RUN_DIR: the full run whose A_0 rows (eval/results.jsonl, val/A0) are the "frozen" column.
--final: probe tags written by final_probe.sh (runs/final_a0_<tag>_{id,ood,val}); a discipline is read from the first
tag that has it. Cells are passed/n; the last column lists every failing episode of the final probe with its primary,
reference, stop reason and wall time.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from adaptation_table import episodes  # noqa: E402


def frac(n_pass: int, n: int) -> str:
    return f"{n_pass}/{n}" if n else "-"


def frozen(run: Path) -> dict[tuple[str, str], list[bool]]:
    out: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for line in (run / "eval" / "results.jsonl").open():
        r = json.loads(line)
        if r["snapshot"] == "A_0" and r["split"] in ("id", "ood"):
            out[(r["discipline"], r["split"])].append(bool(r["accepted"]) and not r.get("failed"))
    for d in sorted((run / "val" / "A0").glob("*/result.json")):
        r = json.loads(d.read_text())
        out[(d.parent.name.split("-")[0], "val")].append(bool(r.get("passed")))
    return out


def final_probe(tags: list[str]) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for tag in tags:
        taken = set(out)
        for split in ("val", "id", "ood"):
            for res in sorted(Path("runs").glob(f"final_a0_{tag}_{split}/*/result.json")):
                d = res.parent
                disc = d.name.split("-")[0]
                if (disc, split) in taken:
                    continue
                r = json.loads(res.read_text())
                if r.get("infra_error"):
                    continue
                ev = json.loads((d / "eval.json").read_text()) if (d / "eval.json").exists() else {}
                use = json.loads((d / "usage.json").read_text()) if (d / "usage.json").exists() else {}
                det = ev.get("details") or {}
                out[(disc, split)].append({
                    "dir": d.name, "pass": bool(r.get("passed")), "primary": r.get("primary"),
                    "ref": det.get("reference") if det.get("reference") is not None else (ev.get("metrics") or {}).get("reference_pq_plus"),
                    "norm": det.get("norm_score"), "stop": r.get("stop_reason"), "wall": use.get("wall_s"),
                    "tok": use.get("total_tokens")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--stock", default="runs/probe_local27b_a0,runs/probe_pass10_n3")
    ap.add_argument("--final", default="main,late")
    a = ap.parse_args()
    fr = frozen(Path(a.run))
    fin = final_probe(a.final.split(","))
    stock = episodes(a.stock)
    discs = sorted({k[0] for k in fr} | {k[0] for k in fin} | set(stock))
    print("| disc | stock A_0 (src) | frozen val | frozen id | frozen ood | final val | final id | final ood | final failures |")
    print("|---|---|---|---|---|---|---|---|---|")
    for d in discs:
        s = stock.get(d, [])
        row = [d, frac(sum(e["pass"] for e in s), len(s))]
        for split in ("val", "id", "ood"):
            v = fr.get((d, split), [])
            row.append(frac(sum(v), len(v)))
        fails = []
        for split in ("val", "id", "ood"):
            v = fin.get((d, split), [])
            row.append(frac(sum(e["pass"] for e in v), len(v)))
            for e in v:
                if not e["pass"]:
                    p = "-" if e["primary"] is None else f"{e['primary']:.4g}"
                    rf = "-" if e["ref"] is None else f"{e['ref']:.4g}"
                    w = "-" if e["wall"] is None else f"{e['wall']:.0f}s"
                    fails.append(f"{split}:{e['dir'].split('-')[-1]} {p} vs {rf} ({e['stop']}, {w})")
        row.append("; ".join(fails) or "-")
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
