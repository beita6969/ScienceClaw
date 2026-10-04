#!/usr/bin/env python3
"""Audit the content-addressed FoR51 MLIP feature cache without computing features.

This is a read-only trusted-side check.  It enumerates the adapter's structures, derives the
same cache key used by ``scilib.matphonon_mlip``, and reports valid, missing, and malformed rows
for one model/parameter tuple.  It never reads or prints targets and never starts SevenNet/CHGNet.
Run with ``SCIENCECLAW_DATA_ROOT`` and ``SCIENCECLAW_MLIP_CACHE`` set in the server environment.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from scienceclaw.bench.tasks.for51_matbench import Adapter
from scilib.matphonon_mlip import NAMES, _cache_key


def audit_cache(structures: dict[str, dict], ids: list[str], cache: Path, model: str, mesh: int = 8,
                min_len: float = 7.0, disp: float = 0.01) -> dict[str, Any]:
    """Return coverage counts for one exact cache parameter tuple."""
    valid = missing = malformed = 0
    examples: list[str] = []
    for iid in ids:
        path = cache / f"{_cache_key(structures[iid], mesh, min_len, disp, model)}.json"
        if not path.is_file():
            missing += 1
            continue
        try:
            row = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=float)
            good = row.shape == (len(NAMES),) and bool(np.isfinite(row).all())
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            good = False
        if good:
            valid += 1
        else:
            malformed += 1
            if len(examples) < 3:
                examples.append(path.name)
    return {"model": model, "mesh": mesh, "min_len": float(min_len), "disp": float(disp),
            "n_structures": len(ids), "valid": valid, "missing": missing, "malformed": malformed,
            "examples": examples, "complete": valid == len(ids) and not missing and not malformed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("sevennet", "chgnet"), default="sevennet")
    ap.add_argument("--mesh", type=int, default=8)
    ap.add_argument("--min-len", type=float, default=7.0)
    ap.add_argument("--disp", type=float, default=0.01)
    ap.add_argument("--cache", type=Path, default=None, help="default: SCIENCECLAW_MLIP_CACHE")
    ap.add_argument("--require-complete", action="store_true", help="return 2 unless every structure is valid")
    args = ap.parse_args()
    raw_cache = str(args.cache) if args.cache is not None else os.environ.get("SCIENCECLAW_MLIP_CACHE", "")
    cache = Path(raw_cache) if raw_cache else None
    if cache is None or not cache.is_dir():
        print(f"cache directory is missing: {raw_cache or '<SCIENCECLAW_MLIP_CACHE unset>'}", flush=True)
        return 2
    data = Adapter()._data.get()
    out = audit_cache(data.structures, data.ids, cache, args.model, args.mesh, args.min_len, args.disp)
    print(json.dumps(out, sort_keys=True), flush=True)
    return 0 if out["complete"] or not args.require_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
