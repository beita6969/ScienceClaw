"""One-look pool evaluation on the adapter's real episodes: ContractModel(plm=False) vs ContractModel(plm=True), same visible data and seeds.

usage: python pool_eval.py <npz> <split: id|ood> <n_episodes> <out.json> [seed]
PLM scores come from the precomputed table (same code path and values as the GPU calls; see bridge_check.py for the real-bridge equality check).
"""
import json
import sys
import time
from types import SimpleNamespace

import numpy as np

import common as K
import lookup
from scienceclaw.bench.splits import load_adapters

npz, split, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 20260928
lookup.install(npz)
ad = load_adapters(SimpleNamespace(disciplines=["FoR48"], data_root=None))["FoR48"]
res = {"split": split, "seed": seed, "episodes": []}
for ep in ad.build_episodes(split, n, seed, 16):
    tr = ep.tool("load_train").fn({}, {})
    hy = ep.tool("load_hypotheses").fn({}, {})["hypotheses"]
    ev = ep.tool("load_eval_inputs").fn({}, {})
    row = {"id": ep.id, "n_items": len(ev["items"])}
    for name, plm in (("old", False), ("plm", True)):
        t0 = time.time()
        m = K.C.ContractModel(seed=0, n_jobs=2, plm=plm).fit(tr["train_documents"], tr["train_annotations"], hy)
        y = m.predict(ev["items"], ev["documents"])
        r = ep._evaluate(y, None)
        row[name] = {"map": r.primary, "ap": r.details["pooled_payload"]["ap_pred"], "accepted": bool(r.accepted),
                     "p_at_r80": r.metrics["p_at_r80"], "nli_acc": r.metrics["nli_binary_accuracy"], "sec": round(time.time() - t0, 1)}
    row["ref_ap"] = r.details["pooled_payload"]["ap_ref"]
    row["docs"] = [it["doc_id"] for it in ev["items"]]
    res["episodes"].append(row)
    print(row["id"], f"old {row['old']['map']:.4f} plm {row['plm']['map']:.4f} ref {r.details['reference']:.4f} "
          f"({row['old']['sec']:.0f}s/{row['plm']['sec']:.0f}s)", flush=True)
    json.dump(res, open(out, "w"))
old = [a for e in res["episodes"] for a in e["old"]["ap"]]
new = [a for e in res["episodes"] for a in e["plm"]["ap"]]
ref = [a for e in res["episodes"] for a in e["ref_ap"]]
grp = [d for e in res["episodes"] for d in e["docs"]]
d = K.paired_boot(old, new, grp)
print(f"--- {split}: {len(old)} pairs in {len(res['episodes'])} episodes, clustered by contract")
print(f"reference {np.mean(ref):.4f}   plm=False {np.mean(old):.4f}   plm=True {np.mean(new):.4f}   diff {d[0]:+.4f} ({d[1]:+.4f}, {d[2]:+.4f})")
print("accepted old/plm:", sum(e["old"]["accepted"] for e in res["episodes"]), sum(e["plm"]["accepted"] for e in res["episodes"]), "of", len(res["episodes"]))
for k in ("p_at_r80", "nli_acc"):
    print(k, f"old {np.mean([e['old'][k] for e in res['episodes']]):.4f} plm {np.mean([e['plm'][k] for e in res['episodes']]):.4f}")
