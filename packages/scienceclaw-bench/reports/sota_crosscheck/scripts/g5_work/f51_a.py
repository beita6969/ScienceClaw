import sys, numpy as np, json, time
sys.path.insert(0, ".")
from pathlib import Path
from scienceclaw.bench.tasks import for51_matbench as m
t=time.time()
ad = m.MatbenchPhononsAdapter()
data = ad._data.get(); parts = ad._parts.get()
print("load s", time.time()-t)
print({k: len(v) for k,v in parts.items()}, "total", len(data.ids), "ood", len(data.ood))
y = np.array([data.targets[i] for i in data.ids])
print("target stats: min %.1f med %.1f mean %.1f max %.1f std %.1f" % (y.min(), np.median(y), y.mean(), y.max(), y.std()))
print("full-data MAD-baseline (predict mean) MAE: %.2f; predict median MAE %.2f" % (np.abs(y-y.mean()).mean(), np.abs(y-np.median(y)).mean()))
for k,v in parts.items():
    yy=np.array([data.targets[i] for i in v]); print(k,len(v),"mean %.1f med %.1f std %.1f max %.1f meanAD-from-mean %.1f"%(yy.mean(),np.median(yy),yy.std(),yy.max(),np.abs(yy-yy.mean()).mean()))
import pickle
pickle.dump({"ids":data.ids,"targets":data.targets,"features":data.features,"structures":data.structures,"ood":data.ood,"parts":parts}, open("/private/tmp/claude-501/sc-scratch/sota/g5_work/f51_data.pkl","wb"))
