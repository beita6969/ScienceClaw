import sys, json, time, numpy as np
sys.path.insert(0, "/home/bedicloud/sharestore2/zxc/scienceclaw/code")
sys.path.append("/home/bedicloud/sharestore2/zxc/scienceclaw/sv_pkgs")
from scilib import matphonon_mlip as mm
S = json.load(open("/home/bedicloud/sharestore2/zxc/scienceclaw/f51/structs.json"))
print("local_ok", mm._local_ok("sevennet"), flush=True)
t0 = time.time(); rows = []
for i in range(0, len(S), 50):
    rows.append(mm.phonon_features(S[i:i + 50], model="sevennet")["X"])
    print(i, f"{time.time() - t0:.0f}s", flush=True)
X = np.vstack(rows); np.save("/home/bedicloud/sharestore2/zxc/scienceclaw/f51/X_l3i5_module.npy", X); print("done", X.shape, flush=True)
