"""Shrink the cached features: keep X/y/crop_scale + the stride-4 lattice (lat[::2,::2]) for src, val and 12+12 test images; drop the rest."""
import sys, os, glob; sys.path.insert(0, '/private/tmp/claude-501/sc-scratch/fix/f30_work')
from common import *
a = adapter(2); p = a._pools(); rng = np.random.default_rng(0)
test = {"id": sorted(rng.choice(p["id"], 12, replace=False).tolist()), "ood": sorted(rng.choice(p["ood"], 12, replace=False).tolist())}
json.dump(test, open(WORK + "/out/test_ids.json", "w"), indent=1)
os.makedirs(WORK + "/feats_c", exist_ok=True)
keep = p["src"] + p["val"] + test["id"] + test["ood"]
for i in keep:
    z = np.load(WORK + "/feats/" + i.replace("/", "__") + ".npz")
    np.savez(WORK + "/feats_c/" + i.replace("/", "__") + ".npz", lat4=z["lat"][::2, ::2], X=z["X"], y=z["y"], crop_scale=z["crop_scale"])
print(len(keep), "kept")
