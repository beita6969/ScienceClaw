"""Crop-plant / crop-leaf instance proposals from the frozen SAM 2.1 mask decoder (weights stored locally, no network) with a
learned proposal selector, on top of the DINOv2 pixel classifier of ``scilib.phenoseg_deep``. Array layout as in
``scilib.phenoseg``: ``images`` uint8 (n,H,W,3), ``semantics`` int (n,H,W) with 0 soil, 1 crop, 2 weed, 3 partial crop,
4 partial weed, ``plant_instances`` / ``leaf_instances`` int (n,H,W), 0 = none; a prediction is a dict with the keys
``semantics``, ``plant_instances``, ``leaf_instances`` (the layout of the task output ``y``).

available() -> bool
    True when torch, transformers (Sam2Model), lightgbm, the stored SAM 2.1 (HF format) and DINOv2 weights can be loaded in this
    interpreter, or when a remote GPU worker is configured. With the worker, ``fit_sam_selector`` runs on its GPU host and returns a
    handle (``RemoteSelector``) that names the fitted selector stored there; ``predict_sam_panoptic`` accepts that handle and
    returns the arrays; ``save_selector`` / ``load_selector`` store and read the handle. Answers are stored, so an identical call
    returns the same result; arrays the host has not received before are transferred at roughly 1 MB per 5-10 s (the
    label arrays compress well, the RGB images do not).
fit_sam_selector(images, semantics, plant_instances, leaf_instances, plant_visibility=None, leaf_visibility=None,
                 seed=0, grid=64, oof_folds=3, device="auto", params=None) -> SamSelector
    1. ``phenoseg_deep.fit_deep_classifier`` on the labelled images (model.plant_size as there) and its out-of-fold class
       probabilities (``oof_folds`` consecutive blocks of images).
    2. For every labelled image, a proposal pool of binary masks: the SAM 2.1 (hiera-large) decoder is prompted with a
       ``grid`` x ``grid`` lattice of single points that fall on ground-truth vegetation (semantics > 0), with the bounding
       box of every connected vegetation component (>= 30 px), and with 1, 2, 3, 4 and 6 k-means centres inside every
       component; all 3 multimask outputs with predicted IoU >= 0.3 are kept, masks < 12 px are dropped and near-duplicates
       are removed by mask-IoU non-maximum suppression (threshold 0.9).
    3. 16 features per proposal (area, area / s^2 with s = ``phenoseg.growth_scale`` of the image, SAM predicted IoU,
       multimask level, prompt type, mean out-of-fold soil / crop / weed probability inside the mask, crop share =
       crop / (crop + weed), bounding-box height / width / fill fraction / aspect, s, crop and weed fraction of the image).
       Target 1: best IoU of the proposal with a ground-truth crop plant; target 2: best IoU with a ground-truth crop leaf
       (instances with visibility <= 0.5 or < 16 px are not used). Two LightGBM regressors (300 rounds, 31 leaves, lr 0.05,
       4 threads) predict the targets from the features.
    ``plant_visibility`` / ``leaf_visibility`` are the visible fractions of the instance covering each pixel (0..1; 0..255
    is also accepted); missing visibility is read as fully visible.
    ``params`` overrides the defaults of ``model.params``: plant_thr 0.3, leaf_thr 0.3 (minimum predicted IoU for a proposal to
    be used), plant_ov 0.25, leaf_ov 0.25 (a proposal is dropped when more than this fraction of it overlaps proposals
    already taken, taken in order of decreasing predicted IoU), share_thr 0.6 (crop share above which a plant proposal is a
    crop plant; the others are weed regions).
predict_sam_panoptic(model, images, device="auto") -> dict of arrays
    per image: class probabilities from the DINOv2 model, vegetation = crop + weed probability > 0.5, proposal pool on that
    vegetation, features, predicted IoUs, then greedy selection: plant proposals (score >= plant_thr) give the plant instances
    and set the semantics inside them to crop or weed by crop share; leaf proposals (score >= leaf_thr, crop share >
    share_thr) are clipped to predicted crop pixels and give the leaf instances. An image without vegetation, or one whose
    pool is empty, falls back to ``phenoseg.panoptic_from_probs``.
save_selector(model, path) / load_selector(path) -> SamSelector
    pickle of the fitted selector, so that fitting and prediction can run in different processes / code nodes.

Model: SAM 2.1 hiera-large (Ravi et al., 2024; Apache-2.0), promptable segmentation trained on SA-1B and SA-V; no PhenoBench
annotations were used. The selector and the DINOv2 head are fitted only on the images passed to ``fit_sam_selector``.
Cost with a GPU (shared A100/H800-class, 24 labelled 512 x 512 images): fitting takes several minutes, prediction about
5-15 s per image (the pool has hundreds to a few thousand masks per image); the sandbox limit per code node should be kept in
mind, so fit and save in one node and predict chunks of images in later nodes. Repeated runs of the same process give the
same masks on the same device; different GPUs can differ in a few pixels.
"""
from __future__ import annotations

import hashlib
import pickle
import time

import numpy as np

from . import phenoseg as ps
from . import phenoseg_deep as pd_
from . import _remote
from ._pretrained import have_module, model_path, switched_off, torch_device

__all__ = ["available", "fit_sam_selector", "predict_sam_panoptic", "save_selector", "load_selector", "SamSelector"]

DEFAULT_PARAMS = dict(plant_thr=0.3, leaf_thr=0.3, plant_ov=0.25, leaf_ov=0.25, share_thr=0.6)
FEATS = ["area", "area_s2", "sam", "level", "ptype", "mc", "mw", "ms", "share", "bh", "bw", "fill", "aspect", "s",
         "crop_frac", "weed_frac"]
_LGB = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=20, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, num_threads=4, verbose=-1)
_sam: dict = {}


def _local_ok() -> bool:
    if switched_off() or model_path("sam2hf") is None or not pd_.available():
        return False
    return have_module("torch") and have_module("transformers") and have_module("lightgbm")


def available() -> bool:
    return _local_ok() or _remote.enabled()


class SamSelector:
    def __init__(self, pixel_model, plant_scorer, leaf_scorer, params: dict, n_train: int, fit_s: float):
        self.pixel_model, self.plant_scorer, self.leaf_scorer = pixel_model, plant_scorer, leaf_scorer
        self.params, self.n_train, self.fit_s = dict(params), n_train, fit_s


class RemoteSelector:
    """Handle of a selector that was fitted on the GPU host: ``path`` names the stored SamSelector there."""

    def __init__(self, path: str, params: dict, n_train: int, fit_s: float):
        self.path, self.params, self.n_train, self.fit_s = path, dict(params), n_train, fit_s


def save_selector(model: SamSelector, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=4)


def load_selector(path: str) -> SamSelector:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- SAM 2.1 proposals

def _load_sam(dev: str):
    if dev not in _sam:
        import torch
        from transformers import Sam2Model, Sam2Processor
        p = str(model_path("sam2hf"))
        _sam[dev] = (Sam2Model.from_pretrained(p, dtype=torch.float32).to(dev).eval(), Sam2Processor.from_pretrained(p))
    return _sam[dev]


class _Session:
    """SAM 2.1 image embedding of one image and prompt batches on it."""

    def __init__(self, img: np.ndarray, dev: str):
        import torch
        self.model, self.proc = _load_sam(dev)
        self.dev = dev
        self.inp = self.proc(images=img, return_tensors="pt").to(dev)
        with torch.no_grad():
            self.emb = self.model.get_image_embeddings(self.inp["pixel_values"])

    def _run(self, points=None, labels=None, boxes=None, min_iou=0.3):
        import torch
        kw = dict(original_sizes=self.inp["original_sizes"], return_tensors="pt")
        with torch.no_grad():
            if boxes is not None:
                pin = self.proc(images=None, input_boxes=[boxes], **kw).to(self.dev)
                o = self.model(input_boxes=pin["input_boxes"], image_embeddings=self.emb, multimask_output=True)
            else:
                pin = self.proc(images=None, input_points=points, input_labels=labels, **kw).to(self.dev)
                o = self.model(input_points=pin["input_points"], input_labels=pin["input_labels"],
                               image_embeddings=self.emb, multimask_output=True)
            m = self.proc.post_process_masks(o.pred_masks.cpu(), self.inp["original_sizes"].cpu(), binarize=True)[0]
        s = o.iou_scores.cpu()[0]
        sel = (s >= min_iou).numpy()
        return m.numpy().astype(bool)[sel], s.numpy()[sel], np.nonzero(sel)

    def _collect(self, parts):
        if not parts:
            return None
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts])

    def points(self, pts, chunk=128):
        parts = []
        for a in range(0, len(pts), chunk):
            c = pts[a:a + chunk]
            m, s, (_, lv) = self._run(points=[[[list(p)] for p in c]], labels=[[[1] for _ in c]])
            parts.append((m, s, lv))
        return self._collect(parts)

    def boxes(self, bxs, chunk=64):
        parts = []
        for a in range(0, len(bxs), chunk):
            c = bxs[a:a + chunk]
            m, s, (_, lv) = self._run(boxes=[list(map(float, b)) for b in c])
            parts.append((m, s, lv))
        return self._collect(parts)

    def multipoints(self, sets, chunk=64):
        parts, by = [], {}
        for k, s_ in enumerate(sets):
            by.setdefault(len(s_), []).append(k)
        for n, ks in by.items():
            for a in range(0, len(ks), chunk):
                c = ks[a:a + chunk]
                m, s, (_, lv) = self._run(points=[[[list(p) for p in sets[k]] for k in c]], labels=[[[1] * n for _ in c]])
                parts.append((m, s, lv))
        return self._collect(parts)


def _kmeans_points(mask: np.ndarray, k: int, rng) -> list | None:
    ys, xs = np.nonzero(mask)
    if len(ys) < k:
        return None
    xy = np.stack([xs, ys], 1).astype(np.float32)
    c = xy[rng.choice(len(xy), k, replace=False)]
    for _ in range(8):
        a = ((xy[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if (a == j).any():
                c[j] = xy[a == j].mean(0)
    return [tuple(xy[((xy - c[j]) ** 2).sum(-1).argmin()].tolist()) for j in range(k)]


def _nms(masks: np.ndarray, scores: np.ndarray, thr: float, dev: str = "cpu") -> np.ndarray:
    """Greedy mask-IoU non-maximum suppression (highest score first); pairwise IoU by one matrix product per block of rows."""
    import torch
    K = len(masks)
    A = torch.from_numpy(masks.reshape(K, -1)).to(dev).float()
    areas = A.sum(1)
    iou = torch.empty(K, K, device=dev)
    for a in range(0, K, 256):
        inter = A[a:a + 256] @ A.T
        iou[a:a + 256] = inter / (areas[a:a + 256, None] + areas[None] - inter + 1e-9)
    over = (iou > thr).cpu().numpy()
    gone = np.zeros(K, bool)
    keep: list[int] = []
    for i in np.argsort(-scores):
        if gone[i]:
            continue
        keep.append(int(i))
        gone |= over[i]
    return np.array(keep, int)


def _build_pool(img: np.ndarray, veg: np.ndarray, seed: int, grid: int, dev: str, nms_thr: float = 0.9):
    """Proposal masks (K,H,W) bool plus SAM score / multimask level / prompt type, prompted on the vegetation mask ``veg``."""
    from scipy import ndimage as ndi
    if not veg.any():
        return None
    ses = _Session(img, dev)
    H, W = veg.shape
    rng = np.random.default_rng(seed)
    ys = (np.arange(grid) + 0.5) * H / grid
    xs = (np.arange(grid) + 0.5) * W / grid
    pts = [(float(x), float(y)) for y in ys for x in xs if veg[int(y), int(x)]]
    parts = []
    r = ses.points(pts) if pts else None
    if r is not None:
        parts.append((*r, 0))
    lab, n = ndi.label(ndi.binary_dilation(veg, iterations=2))
    bxs, sets = [], []
    for c in range(1, n + 1):
        comp = (lab == c) & veg
        if comp.sum() < 30:
            continue
        yy, xx = np.nonzero(comp)
        bxs.append((xx.min(), yy.min(), xx.max() + 1, yy.max() + 1))
        for k in (1, 2, 3, 4, 6):
            if comp.sum() > 60 * k:
                s_ = _kmeans_points(comp, k, rng)
                if s_:
                    sets.append(s_)
    r = ses.boxes(bxs) if bxs else None
    if r is not None:
        parts.append((*r, 1))
    r = ses.multipoints(sets) if sets else None
    if r is not None:
        parts.append((*r, 2))
    if not parts:
        return None
    M = np.concatenate([p[0] for p in parts])
    S = np.concatenate([p[1] for p in parts])
    L = np.concatenate([p[2] for p in parts])
    T = np.concatenate([np.full(len(p[0]), p[3]) for p in parts])
    ok = M.reshape(len(M), -1).sum(1) >= 12
    M, S, L, T = M[ok], S[ok], L[ok], T[ok]
    if not len(M):
        return None
    keep = _nms(M, S, nms_thr, dev)
    return dict(masks=M[keep], sam=S[keep], level=L[keep], ptype=T[keep])


# ---------------------------------------------------------------- features, targets, selection

def _mask_feats(pool: dict, probs: np.ndarray, dev: str) -> np.ndarray:
    import torch
    M = torch.from_numpy(pool["masks"]).to(dev)
    K, H, W = M.shape
    probs = np.asarray(probs, np.float32)
    Pt = torch.from_numpy(probs).to(dev).reshape(-1, 3)
    crop = probs[..., 1] > np.maximum(probs[..., 0], probs[..., 2])
    s = float(ps.growth_scale(crop[None], ps.DEFAULT_PARAMS["plant_size"])[0])
    out = np.zeros((K, len(FEATS)), np.float32)
    for a in range(0, K, 128):
        m = M[a:a + 128]
        Af = m.reshape(len(m), -1).float()
        area = Af.sum(1)
        mp = (Af @ Pt) / area[:, None]
        rows, cols = m.any(2), m.any(1)
        y0 = rows.float().argmax(1)
        y1 = H - rows.flip(1).float().argmax(1)
        x0 = cols.float().argmax(1)
        x1 = W - cols.flip(1).float().argmax(1)
        bh, bw = (y1 - y0).float(), (x1 - x0).float()
        fill = area / (bh * bw).clamp(min=1)
        asp = torch.maximum(bh, bw) / torch.minimum(bh, bw).clamp(min=1)
        blk = torch.stack([area, area / (s * s), mp[:, 1], mp[:, 2], mp[:, 0], mp[:, 1] / (mp[:, 1] + mp[:, 2] + 1e-6),
                           bh, bw, fill, asp], 1).cpu().numpy()
        out[a:a + 128, [0, 1, 5, 6, 7, 8, 9, 10, 11, 12]] = blk
    out[:, 2], out[:, 3], out[:, 4] = pool["sam"], pool["level"], pool["ptype"]
    out[:, 13] = s
    out[:, 14] = float(crop.mean())
    out[:, 15] = float((probs[..., 2] > np.maximum(probs[..., 0], probs[..., 1])).mean())
    return out


def _targets(pool: dict, plant_inst, leaf_inst, sem, pvis, lvis, dev: str):
    """Best IoU of every proposal with a visible ground-truth crop plant / crop leaf."""
    import torch
    M = torch.from_numpy(pool["masks"]).to(dev)
    K = len(M)
    Af = M.reshape(K, -1)
    semm = np.where(sem == 3, 1, np.where(sem == 4, 2, sem))
    res = []
    for inst, vis, only_crop in ((plant_inst, pvis, True), (leaf_inst, lvis, False)):
        gts = []
        for g in np.unique(inst):
            if g == 0:
                continue
            gm = inst == g
            if gm.sum() < 16 or (vis is not None and vis[gm].max() <= 0.5):
                continue
            if only_crop and (semm[gm] == 1).mean() < 0.5:
                continue
            gts.append(gm)
        if not gts:
            res.append(np.zeros(K, np.float32))
            continue
        G = torch.from_numpy(np.stack(gts)).to(dev).reshape(len(gts), -1).float()
        best = torch.zeros(K, device=dev)
        for a in range(0, K, 128):
            A = Af[a:a + 128].float()
            inter = A @ G.T
            best[a:a + 128] = (inter / (A.sum(1, keepdim=True) + G.sum(1)[None] - inter + 1e-9)).max(1).values
        res.append(best.cpu().numpy())
    return res[0], res[1]


def _select(masks: np.ndarray, score: np.ndarray, thr: float, max_overlap: float, allowed=None) -> list[int]:
    taken = np.zeros(masks.shape[1:], bool)
    chosen = []
    for k in np.argsort(-score):
        if score[k] < thr:
            break
        if allowed is not None and not allowed[k]:
            continue
        m = masks[k]
        a = m.sum()
        if a == 0 or (m & taken).sum() > max_overlap * a:
            continue
        chosen.append(int(k))
        taken |= m
    return chosen


def _compose(pool: dict, F: np.ndarray, sp: np.ndarray, sl: np.ndarray, probs: np.ndarray, prm: dict) -> dict:
    M = pool["masks"]
    H, W = probs.shape[:2]
    sem = probs.argmax(-1).astype(np.int16)
    veg = sem > 0
    share = F[:, FEATS.index("share")]
    pl = np.zeros((H, W), np.int32)
    taken = np.zeros((H, W), bool)
    nid = 0
    for k in _select(M, sp, prm["plant_thr"], prm["plant_ov"]):
        m = M[k] & ~taken
        taken |= M[k]
        cls = 1 if share[k] > prm["share_thr"] else 2
        sem[m & veg] = cls
        if cls == 1:
            nid += 1
            pl[m] = nid
    lf = np.zeros((H, W), np.int32)
    crop = sem == 1
    taken = np.zeros((H, W), bool)
    nid = 0
    for k in _select(M, sl, prm["leaf_thr"], prm["leaf_ov"], share > prm["share_thr"]):
        m = M[k] & crop & ~taken
        if m.sum() < 8:
            continue
        taken |= M[k]
        nid += 1
        lf[m] = nid
    return dict(semantics=sem, plant_instances=pl, leaf_instances=lf)


# ---------------------------------------------------------------- fit / predict

def _norm_vis(v, n_expected):
    if v is None:
        return [None] * n_expected
    v = np.asarray(v, np.float32)
    return v / 255.0 if v.max() > 1.0 else v


def fit_sam_selector(images, semantics, plant_instances, leaf_instances, plant_visibility=None, leaf_visibility=None,
                     seed: int = 0, grid: int = 64, oof_folds: int = 3, device: str = "auto", params: dict | None = None,
                     **deep_kwargs) -> SamSelector:
    if not _local_ok() and _remote.enabled():
        r = _remote.call("phenoseg_sam", "fit_sam_selector",
                         dict(images=np.asarray(images), semantics=np.asarray(semantics),
                              plant_instances=np.asarray(plant_instances), leaf_instances=np.asarray(leaf_instances),
                              plant_visibility=None if plant_visibility is None else np.asarray(plant_visibility),
                              leaf_visibility=None if leaf_visibility is None else np.asarray(leaf_visibility),
                              seed=seed, grid=grid, oof_folds=oof_folds, device=device, params=params, **deep_kwargs))
        return RemoteSelector(r["path"], r["params"], r["n_train"], r["fit_s"])
    if not _local_ok():
        raise RuntimeError("fit_sam_selector: torch, transformers, lightgbm or the stored SAM 2.1 / DINOv2 weights are not available here")
    import lightgbm as lgb
    t0 = time.time()
    dev = torch_device(device)
    X = np.asarray(images)
    sem = np.asarray(semantics)
    n = len(X)
    pv, lv = _norm_vis(plant_visibility, n), _norm_vis(leaf_visibility, n)
    pix = pd_.fit_deep_classifier(X, sem, seed=seed, device=dev, **deep_kwargs)
    idx = np.arange(n)
    oof = np.zeros((n,) + X.shape[1:3] + (3,), np.float32)
    for f in np.array_split(idx, max(2, min(oof_folds, n))):
        tr = np.setdiff1d(idx, f)
        m = pd_.fit_deep_classifier(X[tr], sem[tr], seed=seed, device=dev, **deep_kwargs)
        oof[f] = pd_.predict_deep_probs(m, X[f], device=dev).astype(np.float32)
    feats, tp, tl = [], [], []
    for i in range(n):
        veg = (sem[i] > 0)
        pool = _build_pool(X[i], veg, seed + i, grid, dev)
        if pool is None:
            continue
        feats.append(_mask_feats(pool, oof[i], dev))
        a, b = _targets(pool, np.asarray(plant_instances[i]), np.asarray(leaf_instances[i]), sem[i],
                        None if pv[i] is None else pv[i], None if lv[i] is None else lv[i], dev)
        tp.append(a)
        tl.append(b)
    F = np.concatenate(feats)
    mp = lgb.train(_LGB, lgb.Dataset(F, np.concatenate(tp)), 300)
    ml = lgb.train(_LGB, lgb.Dataset(F, np.concatenate(tl)), 300)
    return SamSelector(pix, mp, ml, dict(DEFAULT_PARAMS, **(params or {})), n, time.time() - t0)


def predict_sam_panoptic(model: SamSelector, images, device: str = "auto") -> dict:
    if isinstance(model, RemoteSelector):
        return _remote.call("phenoseg_sam", "predict_sam_panoptic",
                            {"model": model.path, "images": np.asarray(images), "device": device})
    dev = torch_device(device)
    X = np.asarray(images)
    P = pd_.predict_deep_probs(model.pixel_model, X, device=dev).astype(np.float32)
    prm = dict(model.params)
    out = {k: [] for k in ("semantics", "plant_instances", "leaf_instances")}
    for i in range(len(X)):
        veg = P[i][..., 1:].sum(-1) > 0.5
        pool = _build_pool(X[i], veg, int(hashlib.md5(X[i].tobytes()).hexdigest()[:8], 16), 64, dev)
        if pool is None:
            fb = ps.panoptic_from_probs(P[i:i + 1], dict(ps.DEFAULT_PARAMS, plant_size=float(model.pixel_model.plant_size)))
            r = {k: fb[k][0] for k in out}
        else:
            F = _mask_feats(pool, P[i], dev)
            r = _compose(pool, F, model.plant_scorer.predict(F), model.leaf_scorer.predict(F), P[i], prm)
        for k in out:
            out[k].append(r[k])
    return {k: np.stack(v) for k, v in out.items()}
