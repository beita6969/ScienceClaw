#!/usr/bin/env python3
"""Trusted-side full-pool FoR45 reference diagnostic.

This command scores the existing visible-caption medoid and image-kNN references over every image-backed item in
one or more split pools.  It never emits captions or target labels and does not alter formal episode scoring.

Example::

    PYTHONPATH=. python scripts/f45_full_split_reference.py --split src val id ood
"""
from __future__ import annotations

import argparse
import json
import sys

from scienceclaw.bench.tasks.for45_americasnlp import Adapter


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", nargs="+", choices=("src", "val", "id", "ood"), required=True)
    ap.add_argument("--data-root", default=None, help="ScienceClaw dataset root (defaults to registry)")
    args = ap.parse_args(argv)
    adapter = Adapter(data_root=args.data_root)
    ok, why = adapter.available()
    if not ok:
        print(f"FoR45 unavailable: {why}", file=sys.stderr)
        return 2
    out = [adapter.full_split_reference(split) for split in args.split]
    print(json.dumps(out[0] if len(out) == 1 else out, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
