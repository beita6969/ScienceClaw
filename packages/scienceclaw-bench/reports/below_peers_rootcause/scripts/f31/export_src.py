"""Export train/dev/query tables and wild types of one pool's assays (argv[2], default src) for the PLM tool work."""
import pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scienceclaw.bench.tasks.for31_proteingym import Adapter

ad = Adapter()
pools = ad._pools()
print({k: len(v) for k, v in pools.items()})
out = {}
pool = sys.argv[2] if len(sys.argv) > 2 else "src"
for a in pools[pool]:
    it = ad._item(a)
    out[a] = {k: it[k] for k in ("train", "dev", "query", "wild_type")}
idx = ad._index()
print("n", pool, len(out), "lengths min/med/max", min(len(v["wild_type"]) for v in out.values()),
      sorted(len(v["wild_type"]) for v in out.values())[len(out) // 2], max(len(v["wild_type"]) for v in out.values()))
print("train sizes", sorted(len(v["train"]) for v in out.values())[::8])
pickle.dump(out, open(sys.argv[1], "wb"))
