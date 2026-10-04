import sys, json, glob, os
sys.path.insert(0, ".")
import pandas as pd, numpy as np
from scienceclaw.bench.tasks.for31_proteingym import Adapter
a = Adapter(cache_dir="/private/tmp/claude-501/sc-scratch/sota/g1_work/cache31")
ok, why = a.available(); print(ok, why)
S="/Users/admin/Datasets/ScienceClaw-rebuild-20260928/reference_code/FoR31/ProteinGym/benchmarks/DMS_supervised/substitutions/Spearman/DMS_substitutions_Spearman_DMS_level_fold_random_5.csv"
off=pd.read_csv(S).set_index("DMS_id")
R="runs"
cases=[("id",3740910319,"final_a0_main_id","FoR31-id-s3740910319"),("ood",1946474706,"final_a0_main_ood","FoR31-ood-s1946474706"),("val",1389687156,"final_a0_v_val","FoR31-val-s1389687156")]
for split,seed,run,pre in cases:
    eps=a.build_episodes(split,2 if split!="val" else 1,seed)
    for j,e in enumerate(eps):
        ev=f"{R}/{run}/{pre}-e{j:02d}/eval.json"
        if not os.path.exists(ev): continue
        d=json.load(open(ev))["details"]
        assays=e.lineage.get("item_ids")
        if assays is None: print(list(e.lineage.keys())); continue
        ours=np.array(d["per_item_spearman"]); ref=np.array(d["reference_per_item"])
        k=off.loc[assays,"Kermut"].to_numpy(); p=off.loc[assays,"ProteinNPT"].to_numpy(); o=off.loc[assays,"One-Hot Encodings"].to_numpy()
        print(split,j,"n_assay",len(assays),"agent %.3f ref(sitemean) %.3f | official random-fold Kermut %.3f PNPT %.3f OHE %.3f"%(ours.mean(),ref.mean(),k.mean(),p.mean(),o.mean()))
