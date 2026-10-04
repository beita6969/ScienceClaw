"""Cache phenoseg pixel features once per image (scale-2 = 512 px): stride-2 lattice (test-time input of predict_probs) and a fixed
class-balanced sample of 4000 px/class (the training rows fit_pixel_classifier would draw). Read-only use of the repo code."""
import sys, os, time; sys.path.insert(0,'/private/tmp/claude-501/sc-scratch/fix/f30_work')
from multiprocessing import Pool
from common import *
SCALE = int(sys.argv[1]) if len(sys.argv) > 1 else 2
OUT = f"{WORK}/feats{'' if SCALE==2 else SCALE}"; os.makedirs(OUT, exist_ok=True)
def work(item):
    fn = OUT + "/" + item.replace("/", "__") + ".npz"
    if os.path.exists(fn): return item
    a = adapter(SCALE)
    d = a._load(item)
    img, sem = d["images"], d["semantics"]
    F = ps._features_one(img)
    y = ps._merge_partial(sem)
    rng = np.random.default_rng(abs(hash(item)) % (2**31) if False else int.from_bytes(item.encode()[-6:], "little") % (2**31))
    Xs, Ys = [], []
    Fr = F.reshape(-1, F.shape[-1]); yr = y.ravel()
    for c in range(3):
        w = np.flatnonzero(yr == c)
        if len(w):
            i = rng.choice(w, min(4000 * (2 if SCALE==1 else 1), len(w)), replace=False); Xs.append(Fr[i]); Ys.append(yr[i])
    np.savez(fn, lat=F[::2, ::2].astype(np.float32), X=np.concatenate(Xs).astype(np.float32), y=np.concatenate(Ys).astype(np.int8),
             crop_scale=np.float32(ps._crop_scale(y == 1) or -1))
    return item
if __name__ == "__main__":
    a = adapter(SCALE); pools = a._pools()
    items = [i for k in ("src", "val", "id", "ood") for i in pools[k]]
    if SCALE == 1: items = [i for k in ("src","val") for i in pools[k]][:1] + [] if False else items
    t = time.time()
    with Pool(3) as p:
        for k, it in enumerate(p.imap_unordered(work, items)):
            if k % 16 == 0: print(k, round(time.time() - t), flush=True)
    print("done", time.time() - t)
