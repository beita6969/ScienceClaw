"""Shared loading for the FoR40 development scripts: cohorts (probe clips with labels, 24 normal support clips per machine),
AST embeddings by clip, and the pooled official score."""
import json
from pathlib import Path

import numpy as np

from scienceclaw.bench.tasks import for40_dcase as D

ROOT = Path("/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets/for40-dcase2024-task2")
MANIFEST = "/private/tmp/claude-501/sc-scratch/f40_manifest.json"
EMB = "/private/tmp/claude-501/sc-scratch/f40/ast_emb.npz"
EMB_CLAP = "/private/tmp/claude-501/sc-scratch/f40/clap_emb.npz"


def cohorts(splits=("src", "val", "id")):
    """{split: {machine: (probe clips, support clips)}}; probe Clip objects carry label / domain / official_id."""
    design = D.load_design(ROOT)
    return {s: {m: (list(co.clips.values()), design.support[m]) for m, co in design.cohorts[s].items()} for s in splits}


def public_ids():
    man = json.load(open(MANIFEST))
    return {m["id"]: m["public_id"] for m in man}


def embeddings():
    z, c = np.load(EMB), np.load(EMB_CLAP)
    assert list(z["ids"]) == list(c["ids"])
    idx = {p: i for i, p in enumerate(z["ids"])}
    out = {k: z[k] for k in z.files if k != "ids"}
    out.update({"clap_" + k: c[k] for k in c.files if k != "ids"})
    return out, idx


def pooled(per_machine):
    """per_machine: {machine: (y, domain, score)} -> pooled official score (harmonic mean over machines of the three values)."""
    from scipy.stats import hmean
    vals = []
    for y, d, s in per_machine.values():
        o = D.official_scores(np.asarray(y), np.asarray(d), np.asarray(s))
        vals += [o["auc_source"], o["auc_target"], o["pauc"]]
    return float(hmean(np.maximum(vals, D.EPS)))
