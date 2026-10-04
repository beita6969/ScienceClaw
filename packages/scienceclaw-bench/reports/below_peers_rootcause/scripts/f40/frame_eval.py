"""Frame-level (token-level) nearest-neighbour scores from AST hidden states, on the src / val cohorts only (GPU host).
usage: python frame_eval.py <manifest.json> <clips dir> <ast dir>
Per clip: 10.24 s windows (start and end window when longer); layer-L hidden states, patch tokens averaged over the frequency axis ->
101 time frames x 768 per window. Score of a probe clip = aggregate over its frames of the distance to the nearest frame of the
normal support clips (optionally also of the other probe clips)."""
import json
import sys
import time

import numpy as np
import torch
from scipy.io import wavfile
from scipy.stats import hmean
from sklearn.metrics import roc_auc_score
from transformers import ASTFeatureExtractor, ASTForAudioClassification

man = json.load(open(sys.argv[1]))
cdir, mdir = sys.argv[2], sys.argv[3]
fe = ASTFeatureExtractor.from_pretrained(mdir)
net = ASTForAudioClassification.from_pretrained(mdir).eval().cuda().half()
WIN = int(10.24 * 16000)
LAYERS = (6, 8, 10, 12)
feat = {}
t0 = time.time()
with torch.inference_mode():
    for m in man:
        if m["role"] == "id":
            continue
        sr, x = wavfile.read(f"{cdir}/{m['id']}")
        x = x.astype(np.float32) / 32768.0
        wins = [x] if len(x) <= WIN else [x[:WIN], x[-WIN:]]
        inp = fe(wins, sampling_rate=16000, return_tensors="pt")["input_values"].cuda().half()
        hs = net(inp, output_hidden_states=True).hidden_states
        d = {}
        for l in LAYERS:
            tok = hs[l][:, 2:].float().reshape(len(wins), 12, 101, -1).mean(1)       # (win, 101, D)
            d[l] = tok.reshape(-1, tok.shape[-1])                                      # frames of all windows
        feat[m["public_id"]] = d
print("features", len(feat), round(time.time() - t0), "s", flush=True)

ROLE = {"source": "src", "val": "val", "support": "support"}
by = {}
for m in man:
    if m["role"] == "id":
        continue
    by.setdefault((ROLE[m["role"]], m["machine"]), []).append(m)


def official(y, dom, s):
    y, dom, s = np.asarray(y), np.asarray(dom), np.asarray(s, float)
    v = []
    for d in ("source", "target"):
        sel = (dom == d) | (y != 0)
        v.append(roc_auc_score(y[sel], s[sel]))
    v.append(roc_auc_score(y, s, max_fpr=0.1))
    return v


def prep(F, S, how):
    if how == "l2":
        return torch.nn.functional.normalize(F, dim=-1), torch.nn.functional.normalize(S, dim=-1)
    mu, sd = S.mean(0), S.std(0) + 1e-3
    Fz, Sz = (F - mu) / sd, (S - mu) / sd
    return (Fz, Sz) if how == "zs" else (torch.nn.functional.normalize(Fz, dim=-1), torch.nn.functional.normalize(Sz, dim=-1))


def clip_scores(layer, how, agg, pool):
    out = {}
    for split in ("src", "val"):
        pm = {}
        for mach in sorted({k[1] for k in by}):
            probes, sup = by[(split, mach)], by[("support", mach)]
            Sf = torch.cat([feat[p["public_id"]][layer] for p in sup])
            allp = [feat[p["public_id"]][layer] for p in probes]
            Pf = torch.cat(allp)
            Pp, Sp = prep(Pf, Sf, how)
            n_frames = [a.shape[0] for a in allp]
            owner = torch.repeat_interleave(torch.arange(len(allp), device=Pf.device), torch.tensor(n_frames, device=Pf.device))
            dS = torch.cdist(Pp, Sp).min(1).values
            if pool:
                dP = torch.cdist(Pp, Pp)
                dP[owner[:, None] == owner[None, :]] = float("inf")
                dS = torch.minimum(dS, dP.min(1).values)
            sc = []
            for i in range(len(allp)):
                v = dS[owner == i]
                if agg == "mean":
                    sc.append(v.mean().item())
                elif agg == "max":
                    sc.append(v.max().item())
                else:
                    k = max(1, int(round(len(v) * float(agg[3:]) / 100)))
                    sc.append(v.topk(k).values.mean().item())
            pm[mach] = ([p["label"] for p in probes], [p["domain"] for p in probes], sc)
        out[split] = pm
    return out


def pooled(pm):
    vals = []
    for y, d, s in pm.values():
        vals += official(y, d, s)
    return float(hmean(np.maximum(vals, 2.2e-16))), float(np.mean(vals))


rows = []
for layer in LAYERS:
    for how in ("l2", "zs"):
        for agg in ("mean", "top10", "top25", "max"):
            for pool in (False, True):
                r = clip_scores(layer, how, agg, pool)
                hs, ar = pooled(r["src"]), pooled(r["val"])
                rows.append((f"L{layer}/{how}/{agg}/{'pool' if pool else 'sup'}", hs[0], ar[0], hs[1], ar[1]))
for name, a, b, c, d in sorted(rows, key=lambda r: -(r[1] + r[2])):
    print(f"{name:24s} src {a:.4f} val {b:.4f} mean {(a + b) / 2:.4f}   (arith src {c:.4f} val {d:.4f})")
