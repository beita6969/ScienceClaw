"""The first N episodes of a pool with ContractModel(plm=True) through the real remote bridge (no table), next to the table values saved by pool_eval.py.

usage: python pool_bridge.py <pool_eval.json> <n_episodes> <out.json>     (needs SCIENCECLAW_REMOTE_SPOOL and a running broker)
"""
import json
import sys
import time
from types import SimpleNamespace

import common as K
from scienceclaw.bench.splits import load_adapters

src, n, out = json.load(open(sys.argv[1])), int(sys.argv[2]), sys.argv[3]
ad = load_adapters(SimpleNamespace(disciplines=["FoR48"], data_root=None))["FoR48"]
rows = []
for ep, ref in zip(ad.build_episodes(src["split"], n, src["seed"], 16), src["episodes"]):
    assert ep.id == ref["id"]
    tr = ep.tool("load_train").fn({}, {})
    hy = ep.tool("load_hypotheses").fn({}, {})["hypotheses"]
    ev = ep.tool("load_eval_inputs").fn({}, {})
    t0 = time.time()
    y = K.C.ContractModel(seed=0, n_jobs=2, plm=True).fit(tr["train_documents"], tr["train_annotations"], hy).predict(ev["items"], ev["documents"])
    r = ep._evaluate(y, None)
    rows.append({"id": ep.id, "bridge_map": r.primary, "table_map": ref["plm"]["map"], "old_map": ref["old"]["map"], "sec": round(time.time() - t0)})
    print(rows[-1], flush=True)
    json.dump(rows, open(out, "w"))
