#!/usr/bin/env python3
"""Run the explicit FoR51 CHGNet ToolSpec on a cache-complete source episode.

This is an engineering-only smoke, not a formal scorer.  It uses the adapter's public
ToolSpecs to load visible train structures/targets and unlabeled source structures, then calls
``fit_chgnet_mlip`` directly.  The hidden evaluation target is not passed to the tool and no
``Episode.evaluate``/scorer is run.  Run in the configured CHGNet environment with the exact
cache tuple expected by the formal route (mesh=8, min_len=7, disp=0.01).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from scienceclaw.bench.tasks.for51_matbench import Adapter


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-index", type=int, default=2, help="zero-based source episode to probe")
    ap.add_argument("--seed", type=int, default=20260928)
    args = ap.parse_args()
    if args.src_index < 0:
        raise SystemExit("--src-index must be non-negative")
    root = os.environ.get("SCIENCECLAW_DATA_ROOT")
    if not root:
        raise SystemExit("SCIENCECLAW_DATA_ROOT is required")
    ad = Adapter(data_root=root)
    # Building the episode supplies the adapter-owned closures.  We intentionally never read
    # ep._evaluate, ep.lineage targets, or run a scorer below.
    eps = ad.build_episodes("src", args.src_index + 1, seed=args.seed, items_per_episode=16)
    ep = eps[args.src_index]
    tools = {t.name: t for t in ep.tools}
    train = tools["load_train"].fn({}, {})
    evaluation = tools["load_eval_inputs"].fn({}, {})
    result = tools["fit_chgnet_mlip"].fn(
        {"train_structures": train["structures"], "train_targets": train["targets"],
         "eval_structures": evaluation["structures"]}, {}
    )
    pred = np.asarray(result["pred"], dtype=float)
    if pred.shape != (16,) or not np.isfinite(pred).all():
        raise RuntimeError("fit_chgnet_mlip returned a non-finite or wrong-shaped prediction")
    print(json.dumps({"episode_id": ep.id, "tool": "fit_chgnet_mlip", "provenance": result["provenance"],
                      "shape": list(pred.shape), "finite": True, "min": float(pred.min()),
                      "max": float(pred.max()), "mean": float(pred.mean())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
