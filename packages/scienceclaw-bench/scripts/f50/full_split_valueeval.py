#!/usr/bin/env python3
"""Trusted-side full-split ValueEval diagnostic.

This driver runs the adapter's fixed visible-data ``fit_predict`` route over one complete split and prints aggregate
official F1 diagnostics.  It is deliberately outside the agent runtime: labels are read only by the trusted adapter
to score the aggregate and are never passed to the agent-facing ToolSpec or emitted in the JSON result.

Examples::

    PYTHONPATH=. python scripts/f50/full_split_valueeval.py --split id ood
    PYTHONPATH=. python scripts/f50/full_split_valueeval.py --split id --data-root /path/to/datasets
"""
from __future__ import annotations

import argparse
import json
import sys

from scienceclaw.bench.tasks.for50_valueeval import ValueEvalAdapter


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", nargs="+", choices=("src", "val", "id", "ood"), required=True,
                    help="complete split(s) to audit")
    ap.add_argument("--data-root", default=None, help="ScienceClaw dataset root (defaults to environment/registry)")
    args = ap.parse_args(argv)

    adapter = ValueEvalAdapter(data_root=args.data_root)
    ok, why = adapter.available()
    if not ok:
        print(f"FoR50 unavailable: {why}", file=sys.stderr)
        return 2
    rows = [adapter.full_split_diagnostics(split) for split in args.split]
    print(json.dumps(rows[0] if len(rows) == 1 else rows, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
