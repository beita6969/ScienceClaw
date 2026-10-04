"""ESM-2 features of the src-pool assays through the file-spool bridge (needs SCIENCECLAW_REMOTE_SPOOL, broker running)."""
import pickle, sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scilib import proteinplm

items = pickle.load(open(sys.argv[1], "rb"))
out_path, chunk = Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 6
feats = pickle.load(open(out_path, "rb")) if out_path.exists() else {}
todo = [a for a in items if a not in feats]
print("available:", proteinplm.available(), "todo", len(todo), flush=True)
for i in range(0, len(todo), chunk):
    names = todo[i:i + chunk]
    tabs = []
    for a in names:
        it = items[a]
        t = pd.concat([it["train"].assign(part="train"), it["dev"].assign(part="dev"), it["query"].assign(part="query")], ignore_index=True)
        t["assay_id"] = a
        tabs.append(t)
    tab = pd.concat(tabs, ignore_index=True)
    t0 = time.time()
    F = proteinplm.plm_features(tab[["assay_id", "position", "wt_aa", "mut_aa"]], {a: items[a]["wild_type"] for a in names})
    for a in names:
        m = (tab["assay_id"] == a).to_numpy()
        feats[a] = F[m].reset_index(drop=True)
    pickle.dump(feats, open(out_path, "wb"))
    print(f"{names[0]}..: {len(names)} assays, {len(tab)} rows, {time.time() - t0:.0f}s", flush=True)
