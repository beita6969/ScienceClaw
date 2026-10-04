"""Server-side experiment: mask-aware amortised student model vs the 1-D IRT toolkit on a pseudo-validation carved from the visible training students.

Uses only the visible training matrix (no id/ood pool).  python exp_nn.py train.npz [epochs]
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/bedicloud/sharestore2/zxc/scienceclaw/code")
from scilib import adaptive  # noqa: E402

Z = np.load(sys.argv[1])
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
A = Z["answers"].astype(np.int64)
META = Z["meta"]
n, Q = A.shape
rng = np.random.default_rng(7)
perm = rng.permutation(n)
V, F = np.sort(perm[:800]), np.sort(perm[800:])
dev = torch.device("cuda")


def meta_feats(m):
    g = m[:, 0]
    age = m[:, 1]
    prem = m[:, 2]
    ok = lambda x: (~np.isnan(x)).astype(np.float32)
    f = np.stack([np.nan_to_num(g == 1), np.nan_to_num(g == 2), ok(g), np.nan_to_num((np.nan_to_num(age, nan=2007) - 2007) / 3.0),
                  ok(age), np.nan_to_num(prem), ok(prem)], axis=1).astype(np.float32)
    return f


MF = meta_feats(META)


def protocol(rows, seed):
    r = np.random.default_rng(seed)
    a = A[rows]
    tg = np.zeros_like(a, dtype=bool)
    for i in range(len(rows)):
        obs = np.flatnonzero(a[i] >= 0)
        tg[i, r.permutation(obs)[:max(1, int(round(0.2 * obs.size)))]] = True
    return a, tg, (a >= 0) & ~tg


class Net(nn.Module):
    def __init__(self, d=64, h=1024, p=0.3):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(3 * Q + 7, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, d))
        self.E = nn.Parameter(torch.randn(Q, d) * 0.05)
        self.b = nn.Parameter(torch.zeros(Q))

    def forward(self, ans, rev, mf):
        x = torch.cat([ans.float(), (rev == 1).float(), (rev == 0).float(), mf], dim=1)
        return self.enc(x) @ self.E.T + self.b


def to_t(x, dtype=None):
    t = torch.as_tensor(x, device=dev)
    return t if dtype is None else t.to(dtype)


def random_reveal(can, r):
    sel = np.full((can.shape[0], 10), -1)
    for i in range(can.shape[0]):
        idx = np.flatnonzero(can[i])
        pick = r.permutation(idx)[:10]
        sel[i, :pick.size] = pick
    return sel


def reveal_from(a, sel):
    rev = np.full(a.shape, -1, dtype=np.int64)
    for i in range(a.shape[0]):
        for q in sel[i]:
            if q >= 0:
                rev[i, q] = a[i, q]
    return rev


def predict_nn(net, a, rev, mf):
    net.eval()
    with torch.no_grad():
        out = []
        for s in range(0, a.shape[0], 256):
            out.append(torch.sigmoid(net(to_t(a[s:s + 256] >= 0), to_t(rev[s:s + 256]), to_t(mf[s:s + 256]))).cpu().numpy())
    return np.concatenate(out)


def acc(p, a, tg):
    return float(((p > 0.5).astype(np.int64) == a)[tg].mean())


# ---- pseudo-validation
av, tgv, canv = protocol(V, 123)
model = adaptive.fit_item_curves(A[F])
t0 = time.time()
sel_rand = random_reveal(canv, np.random.default_rng(5))
sel_batch = adaptive.select_queries(model, canv, [], 10, 10, "batch")
print(f"selection {time.time()-t0:.1f}s", flush=True)
rev_rand, rev_batch = reveal_from(av, sel_rand), reveal_from(av, sel_batch)
maj = adaptive.fit_item_curves(A[F])["majority"]
print("majority", acc(np.tile(maj, (len(V), 1)).astype(float), av, tgv))
for name, rev in (("rand", rev_rand), ("batch", rev_batch)):
    print("IRT", name, acc(adaptive.predict_proba(model, rev), av, tgv), flush=True)
print("IRT none", acc(adaptive.predict_proba(model, np.full_like(av, -1)), av, tgv), flush=True)

# ---- NN
net = Net().to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
AF, MFF = A[F], MF[F]
r = np.random.default_rng(99)
mfv = MF[V]
best = 0
for ep in range(EPOCHS):
    net.train()
    order = r.permutation(len(F))
    tot = 0.0
    for s in range(0, len(F), 128):
        rows = order[s:s + 128]
        a = AF[rows]
        tg = np.zeros_like(a, dtype=bool)
        for i in range(len(rows)):
            obs = np.flatnonzero(a[i] >= 0)
            tg[i, r.permutation(obs)[:max(1, int(round(0.2 * obs.size)))]] = True
        can = (a >= 0) & ~tg
        sel = random_reveal(can, r)
        rev = reveal_from(a, sel)
        # the model predicts every answered cell that was not revealed
        lossmask = (a >= 0) & (rev < 0)
        logit = net(to_t(a >= 0), to_t(rev), to_t(MFF[rows]))
        y = to_t((a == 1).astype(np.float32))
        m = to_t(lossmask)
        loss = (nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none") * m).sum() / m.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += float(loss)
    sched.step()
    if ep % 5 == 4 or ep == EPOCHS - 1:
        pr, pb = predict_nn(net, av, rev_rand, mfv), predict_nn(net, av, rev_batch, mfv)
        print(f"ep {ep+1} loss {tot/ (len(F)/128):.4f} NN rand {acc(pr, av, tgv):.4f} batch {acc(pb, av, tgv):.4f}", flush=True)
