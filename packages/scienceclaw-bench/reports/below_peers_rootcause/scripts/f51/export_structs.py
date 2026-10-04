import sys, json, numpy as np
REPO = "/Users/admin/Library/Application Support/Claude/scratch-workspaces/b286c85c-fb50-4909-a7a4-692fbcc9e2e6/67544b4d-5c42-4571-a504-64db11cd447d/scratch-2026-09-28-749df2/scienceclaw"
sys.path.insert(0, REPO)
from scienceclaw.bench.tasks.for51_matbench import Adapter
a = Adapter(); d = a._data.get()
out = []
for i in d.ids:
    st = d.structures[i]
    out.append({"lattice": np.asarray(st["lattice"], float).tolist(), "species": [str(s) for s in st["species"]],
                "frac_coords": np.asarray(st["frac_coords"], float).tolist()})
json.dump(out, open("/private/tmp/claude-501/sc-scratch/f51/structs.json", "w"))
print(len(out))
