import sys, numpy as np
sys.path.insert(0, ".")
sys.path.insert(0, "/Users/admin/Datasets/ScienceClaw-rebuild-20260928/reference_code/FoR32/nnUNet")
from scienceclaw.bench.tasks.for32_msd_hippocampus import case_scores, dice
# official nnU-Net formulas (evaluate_predictions.py): dice = 2tp/(2tp+fp+fn) or nan if denominator 0 ; loaded from the source text
src = open("/Users/admin/Datasets/ScienceClaw-rebuild-20260928/reference_code/FoR32/nnUNet/nnunetv2/evaluation/evaluate_predictions.py").read()
import re
print([l.strip() for l in src.splitlines() if "Dice" in l or "2 * tp" in l or "dice" in l][:6])
rng = np.random.default_rng(0)
mx = 0
for t in range(200):
    gt = rng.integers(0, 3, (30, 40, 30)).astype(np.uint8)
    pr = gt.copy(); m = rng.random(gt.shape) < rng.uniform(0, .6); pr[m] = rng.integers(0, 3, m.sum())
    ours = case_scores(pr, gt, (1., 1., 1.))["dsc"]
    ref = []
    for c in (1, 2):
        tp = ((gt == c) & (pr == c)).sum(); fp = ((gt != c) & (pr == c)).sum(); fn = ((gt == c) & (pr != c)).sum()
        ref.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan)
    mx = max(mx, np.max(np.abs(np.array(ours) - np.array(ref))))
print("max abs diff vs 2tp/(2tp+fp+fn):", mx)
# empty-empty convention
z = np.zeros((5, 5, 5), np.uint8)
print("both empty: ours", dice(z == 1, z == 1), "(nnU-Net: nan -> excluded from mean)")
