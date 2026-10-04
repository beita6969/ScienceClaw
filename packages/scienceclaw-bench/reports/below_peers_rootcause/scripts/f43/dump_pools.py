import os, sys, pickle, statistics as st
from scienceclaw.bench.tasks import for43_hipe as H
ad = H.HipeOCRepairAdapter()
units = ad._units.get(); pools = ad._pools.get()
def desc(name, uids):
    L = [len(units[u].ocr) for u in uids]
    print(f"{name}: n={len(uids)} chars_total={sum(L)} median={st.median(L) if L else 0} max={max(L) if L else 0}")
for pool in ("iid", "ood"):
    for ts, uids in sorted(pools.train[pool].items()):
        desc(f"train/{pool}/{ts}", uids)
for sp, d in pools.eval_split.items():
    for ts, uids in sorted(d.items()):
        desc(f"eval/{sp}/{ts}", uids)
print("excluded", pools.excluded_train_groups)
pickle.dump({"units": {u: units[u].__dict__ for u in units}, "pools_train": pools.train, "pools_eval": pools.eval_split},
            open("/private/tmp/claude-501/sc-scratch/f43/pool.pkl", "wb"))
