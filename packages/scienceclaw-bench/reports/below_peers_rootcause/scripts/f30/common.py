"""Shared helpers for the FoR30 analysis (read-only use of the repo; own cache dir)."""
import os, sys, time, json
import numpy as np
REPO = "/Users/admin/Library/Application Support/Claude/scratch-workspaces/b286c85c-fb50-4909-a7a4-692fbcc9e2e6/67544b4d-5c42-4571-a504-64db11cd447d/scratch-2026-09-28-749df2/scienceclaw"
WORK = "/private/tmp/claude-501/sc-scratch/fix/f30_work"
sys.path.insert(0, REPO)
from scienceclaw.bench.tasks import for30_phenobench as F30   # noqa: E402
from scilib import phenoseg as ps                              # noqa: E402

def adapter(scale=2):
    return F30.Adapter(cache_dir=WORK + "/cache", scale=scale)

def load_imgs(a, ids, keys=("images",)):
    arrs = [a._load(i) for i in ids]
    return {k: np.stack([x[k] for x in arrs]) for k in keys}

def score(a, ids, pred):
    """Official-port metrics (percent) of a prediction dict for image ids (cohort PQ+)."""
    m = F30.aggregate(a._stats(ids, pred))
    return {k: (None if m[k] is None else float(m[k])) for k in ("pq_plus", "iou_soil", "iou_crop", "iou_weed", "pq_crop", "pq_leaf")}

def groups(a):
    roles = a._roles()
    return {s: {g: sorted(r.id for r in rs if r.group == g) for g in sorted({r.group for r in rs})} for s, rs in roles.items()}
