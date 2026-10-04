import sys, glob, os
import numpy as np
from PIL import Image
REPO="/Users/admin/Library/Application Support/Claude/scratch-workspaces/b286c85c-fb50-4909-a7a4-692fbcc9e2e6/67544b4d-5c42-4571-a504-64db11cd447d/scratch-2026-09-28-749df2/scienceclaw"
sys.path.insert(0,REPO)
import scienceclaw.bench.tasks.for30_phenobench as ours
R="/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets/for30-phenobench/smoke_data/PhenoBench"
for mode in ["allsoil","oracle_sem_no_inst"]:
  for sp in ["train","val"]:
    stats=[]
    for n in sorted(os.listdir(f"{R}/{sp}/semantics")):
        g={k:np.asarray(Image.open(f"{R}/{sp}/{k}/{n}")) for k in ("semantics","plant_instances","leaf_instances","plant_visibility","leaf_visibility")}
        g["plant_visibility"]=g["plant_visibility"]/255.; g["leaf_visibility"]=g["leaf_visibility"]/255.
        g={k:(v[::2,::2]) for k,v in g.items()}
        z=np.zeros_like(g["semantics"])
        if mode=="allsoil": pred={"semantics":z,"plant_instances":z,"leaf_instances":z}
        else: pred={"semantics":g["semantics"],"plant_instances":z,"leaf_instances":z}
        stats.append(ours.hierarchical_image_stats(pred,g))
    m=ours.aggregate(stats)
    print(mode,sp,{k:(None if m[k] is None else round(m[k],2)) for k in ["pq_plus","iou_soil","iou_crop","iou_weed","pq_crop","pq_leaf"]})
