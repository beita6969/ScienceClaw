"""Numeric parity: official PRBonn/phenobench hierarchical eval (torch) vs the scienceclaw numpy port (for30_phenobench).

GT = real PhenoBench labels from the smoke subset (16 images). Predictions = seeded perturbations of the GT
(boundary jitter, merges, splits, drops, class flips, scrambled instance ids).
"""
import sys, os, glob, json, shutil, tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

REPO = "/Users/admin/Library/Application Support/Claude/scratch-workspaces/b286c85c-fb50-4909-a7a4-692fbcc9e2e6/67544b4d-5c42-4571-a504-64db11cd447d/scratch-2026-09-28-749df2/scienceclaw"
OFF = "/Users/admin/Datasets/ScienceClaw-rebuild-20260928/reference_code/FoR30/phenobench/src"
sys.path.insert(0, REPO)
sys.path.insert(0, OFF)
import scienceclaw.bench.tasks.for30_phenobench as ours
from phenobench.evaluation.evaluate_semantics import evaluate_semantics
from phenobench.evaluation.evaluate_plant_instance_masks_panoptic import evaluate_plant_instances
from phenobench.evaluation.evaluate_leaf_instance_masks_panoptic import evaluate_leaf_instances

ROOT = Path("/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets/for30-phenobench/smoke_data/PhenoBench")
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "val"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
STRENGTH = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
rng = np.random.default_rng(SEED)


def perturb_instances(inst, sem_mask_fn=None):
    """Return a perturbed instance map with scrambled ids."""
    out = np.zeros_like(inst, dtype=np.int32)
    ids = [i for i in np.unique(inst) if i > 0]
    new_id = 1
    perm = rng.permutation(len(ids) + 2 * len(ids) + 5) + 1  # scrambled id pool
    k = 0
    for i in ids:
        m = inst == i
        r = rng.random()
        if r < 0.10 * STRENGTH:          # drop
            continue
        if r < 0.25 * STRENGTH:          # shift a few px
            dy, dx = rng.integers(-6, 7, 2)
            m = np.roll(np.roll(m, dy, 0), dx, 1)
        elif r < 0.35 * STRENGTH:        # erode / dilate
            m = ndimage.binary_dilation(m, iterations=int(rng.integers(1, 5))) if rng.random() < .5 else ndimage.binary_erosion(m, iterations=int(rng.integers(1, 5)))
        elif r < 0.45 * STRENGTH:        # split in two halves
            ys, xs = np.nonzero(m)
            if len(ys) > 10:
                cut = np.median(xs)
                out[m & (np.arange(m.shape[1])[None, :] <= cut) & (out == 0)] = perm[k]; k += 1
                out[m & (np.arange(m.shape[1])[None, :] > cut) & (out == 0)] = perm[k]; k += 1
                continue
        out[m & (out == 0)] = perm[k]; k += 1
    # merge: relabel some adjacent pairs
    if rng.random() < 0.7 * STRENGTH:
        ids2 = [i for i in np.unique(out) if i > 0]
        for _ in range(min(3, len(ids2) // 2)):
            a, b = rng.choice(ids2, 2, replace=False)
            out[out == b] = a
    return out


def make_pred(sem, pinst, linst):
    # semantics: start from GT, flip a few connected regions and jitter boundary
    ps = sem.copy().astype(np.int32)
    ps[ps == 3] = 1; ps[ps == 4] = 2
    lab, n = ndimage.label(ps > 0)
    for j in range(1, n + 1):
        if rng.random() < 0.15 * STRENGTH:
            reg = lab == j
            ps[reg] = 3 - ps[reg]   # crop <-> weed
    noise = ndimage.binary_dilation(rng.random(ps.shape) < 0.0005, iterations=6)
    ps[noise & (rng.random() < .5)] = 1
    ps[noise & (rng.random() >= .5)] = 2
    ps = ndimage.median_filter(ps, size=5)
    pl = perturb_instances(pinst)
    ll = perturb_instances(linst)
    # keep plant instances consistent with predicted semantics being vegetation (as an agent would): nothing forced
    return ps.astype(np.uint16), pl.astype(np.uint16), ll.astype(np.uint16)


names = sorted(os.path.basename(p) for p in glob.glob(str(ROOT / SPLIT / "semantics" / "*.png")))
tmp = Path(tempfile.mkdtemp(prefix="pb_parity_"))
for d in ("semantics", "plant_instances", "leaf_instances"):
    (tmp / d).mkdir()
gts = {}
for nme in names:
    g = {k: np.asarray(Image.open(ROOT / SPLIT / k / nme)) for k in
         ("semantics", "plant_instances", "leaf_instances", "plant_visibility", "leaf_visibility")}
    gts[nme] = g
    ps, pl, ll = make_pred(g["semantics"].astype(np.int32), g["plant_instances"].astype(np.int32), g["leaf_instances"].astype(np.int32))
    for k, a in (("semantics", ps), ("plant_instances", pl), ("leaf_instances", ll)):
        Image.fromarray(a.astype(np.uint16)).save(tmp / k / nme)
    gts[nme]["pred"] = {"semantics": ps, "plant_instances": pl, "leaf_instances": ll}

args = {"phenobench_dir": ROOT, "prediction_dir": tmp, "split": SPLIT, "export": tmp / "export"}
args["export"].mkdir()
sem = evaluate_semantics(args)
pl = evaluate_plant_instances(args)
lf = evaluate_leaf_instances(args)
off = {"iou_soil": sem["soil"], "iou_crop": sem["crop"], "iou_weed": sem["weed"], "pq_crop": pl["plants_cls"][1]["pq"],
       "pq_weed_plants": pl["plants_cls"].get(2, {}).get("pq"), "pq_leaf": lf["leaves_pq"]}
off["pq_plus"] = (off["iou_soil"] + off["iou_weed"] + off["pq_crop"] + off["pq_leaf"]) / 4


def run_ours(scale):
    stats = []
    for nme, g in gts.items():
        gg = {k: (np.asarray(g[k]) if "vis" not in k else np.asarray(g[k], dtype=float) / 255.0) for k in
              ("semantics", "plant_instances", "leaf_instances", "plant_visibility", "leaf_visibility")}
        pp = dict(g["pred"])
        if scale > 1:
            gg = {k: v[::scale, ::scale] for k, v in gg.items()}
            pp = {k: v[::scale, ::scale] for k, v in pp.items()}
        pred = {"semantics": pp["semantics"], "plant_instances": pp["plant_instances"], "leaf_instances": pp["leaf_instances"]}
        gt = gg
        stats.append(ours.hierarchical_image_stats(pred, gt))
    return ours.aggregate(stats)


m1 = run_ours(1)
m2 = run_ours(2)
keys = ["iou_soil", "iou_crop", "iou_weed", "pq_crop", "pq_weed_plants", "pq_leaf", "pq_plus"]
print(f"split={SPLIT} seed={SEED} strength={STRENGTH} n_img={len(names)}")
print(f"{'metric':16s} {'official':>10s} {'ours@1024':>10s} {'ours@512':>10s}")
for k in keys:
    o = off.get(k); a = m1.get(k); b = m2.get(k)
    f = lambda v: "   None" if v is None else f"{v:10.4f}"
    print(f"{k:16s} {f(o)} {f(a)} {f(b)}")
shutil.rmtree(tmp)
