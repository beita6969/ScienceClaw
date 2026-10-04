"""FoR34: inputs for the GPU-host experiment (no episode data, no per-episode labels beyond the pools' own).
Writes smiles.txt (one SMILES per line, graph-index order), inputs.npz (labels, pool indices for src/val/id/ood, 4 deterministic training draws
from the adapter's own sampler, the official split indices)."""
import numpy as np
from pathlib import Path
from scienceclaw.bench.tasks import for34_molhiv as M

OUT = Path("/private/tmp/claude-501/sc-scratch/f34"); OUT.mkdir(parents=True, exist_ok=True)
ad = M.MolhivAdapter()
data = ad._load_data() if hasattr(ad, "_load_data") else M._load(ad.root)
pools = ad._get_pools(data)
arrs = {"labels": data.labels.astype(np.int8)}
for s, d in pools.items():
    idx = np.array(sorted(M._idx(i) for c in (0, 1) for i in d[c]), dtype=np.int64)
    arrs[f"pool_{s}"] = idx
    print(s, idx.size, int(data.labels[idx].sum()))
for k in range(4):
    tr, dv = ad._train_sample(data, f"f34-exp-{k}")
    arrs[f"train_{k}"], arrs[f"dev_{k}"] = tr, dv
    print("draw", k, tr.size, int(data.labels[tr].sum()), dv.size)
for s in ("train", "valid", "test"):
    arrs[f"split_{s}"] = np.asarray(data.split[s], dtype=np.int64)
np.savez(OUT / "inputs.npz", **arrs)
open(OUT / "smiles.txt", "w").write("\n".join(str(x) for x in data.smiles) + "\n")
print("smiles", len(data.smiles))
