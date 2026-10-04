#!/usr/bin/env python3
"""Precompute the frozen SevenNet phonon-feature cache for FoR51.

Run this inside a Slurm GPU allocation, for example with ``PYTHONPATH`` containing both the
ScienceClaw checkout and its ``sv_pkgs`` directory.  The script never fits a model and never writes
targets: it only reads crystal structures from the FoR51 adapter and lets ``matphonon_mlip`` write
content-addressed feature rows to ``SCIENCECLAW_MLIP_CACHE``.  Existing rows are skipped by the
scilib cache, so an interrupted allocation can be resumed safely.  ``--start``/``--stop`` select a
zero-based half-open range of dataset rows, allowing several allocations to prewarm disjoint ranges
without changing cache keys or touching the held-out targets.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from scienceclaw.bench.tasks.for51_matbench import Adapter
from scilib import matphonon_mlip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("sevennet", "sevennet_mf0_pbe", "sevennet_mf0_r2scan", "chgnet"), default="sevennet")
    ap.add_argument("--batch", type=int, default=16, help="structures per local GPU call")
    ap.add_argument("--mesh", type=int, default=8)
    ap.add_argument("--min-len", type=float, default=7.0)
    ap.add_argument("--disp", type=float, default=0.01)
    ap.add_argument("--start", type=int, default=0, help="zero-based first dataset row (inclusive)")
    ap.add_argument("--stop", type=int, default=None, help="zero-based dataset row (exclusive); default: end")
    args = ap.parse_args()
    if args.batch < 1:
        raise SystemExit("--batch must be positive")
    if args.start < 0:
        raise SystemExit("--start must be non-negative")
    if not matphonon_mlip._local_ok(args.model):
        raise SystemExit(f"{args.model} packages are unavailable; run this in the configured GPU environment")
    # Ensure a fresh model-specific cache path is usable before _local writes rows.
    # _cache_dir intentionally only accepts existing directories so callers can
    # disable caching by omission; the prewarm CLI is the explicit opt-in writer.
    cache_root = os.environ.get("SCIENCECLAW_MLIP_CACHE")
    if cache_root:
        Path(cache_root).mkdir(parents=True, exist_ok=True)

    data = Adapter()._data.get()
    if args.start > len(data.ids):
        raise SystemExit(f"--start={args.start} exceeds dataset size {len(data.ids)}")
    stop = len(data.ids) if args.stop is None else int(args.stop)
    if stop < args.start or stop > len(data.ids):
        raise SystemExit(f"--stop must satisfy {args.start} <= stop <= {len(data.ids)}")
    structures = [data.structures[i] for i in data.ids[args.start:stop]]
    t0 = time.time()
    done = 0
    failed = 0
    for start in range(0, len(structures), args.batch):
        batch = structures[start:start + args.batch]
        rows = np.asarray(matphonon_mlip._local(batch, args.mesh, args.min_len, args.disp, args.model), dtype=float)
        valid = int(np.isfinite(rows).all(axis=1).sum())
        failed += len(batch) - valid
        done += len(batch)
        print(f"{args.start + done}/{stop} structures; valid_batch={valid}/{len(batch)} failed_total={failed}; "
              f"elapsed={time.time() - t0:.1f}s", flush=True)
    if failed:
        raise SystemExit(f"prewarm incomplete: {failed} rows returned non-finite features; rerun the same range")
    print(f"completed model={args.model} rows=[{args.start},{stop}) n={done} valid={done} "
          f"elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
