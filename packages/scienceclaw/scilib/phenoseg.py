"""Tools for hierarchical panoptic segmentation of top-down field images: soil / crop / weed semantics, crop-plant
instances and crop-leaf instances (PhenoBench PQ+). numpy / scipy / lightgbm only; every function works on the arrays it
is given.

Layout: ``images`` uint8 (n,H,W,3); ``semantics`` int (n,H,W): 0 soil, 1 crop, 2 weed, 3 partial crop, 4 partial weed;
``plant_instances`` / ``leaf_instances`` int (n,H,W), 0 = none; probabilities float (n,H,W,3), channels (soil, crop, weed).
A prediction is a dict with keys ``semantics``, ``plant_instances``, ``leaf_instances`` (each (n,H,W)): the layout of the
task output ``y`` and of the ``prediction`` input of ``score_dev``.

fit_predict(train_images, train_semantics, targets, params=None, seed=0, stride=2, pixels_per_image=12000,
            n_estimators=200, budget_s=None) -> list of prediction dicts
    fit_pixel_classifier, then predict_probs and panoptic_from_probs for each image array in the list ``targets`` (a single
    array returns one dict). E.g. ``dev_pred, y = fit_predict(tr_images, tr_semantics, [dev_images, eval_images])``.
    About 30 s to fit on 24 images and 2-3 s per predicted image on an idle core pair.
pixel_features(images) -> float32 (...,35)
    per-pixel features: RGB, ExG / ExR / VARI / saturation, Gaussian-smoothed channels (sigma 1.5/4/8), local ExG std and
    gradient, vegetation-density maps (sigma 6-48), the image's Otsu ExG threshold, difference-of-Gaussian, Laplacian and
    Hessian-trace responses of ExG, coarse ExG / contrast context.
fit_pixel_classifier(images, semantics, seed=0, pixels_per_image=12000, n_estimators=200, budget_s=None) -> PixelModel
    LightGBM (2 threads) on a class-balanced pixel sample of each image; labels 3/4 are merged into 1/2. ``budget_s`` stops
    adding trees after that many seconds. ``model.plant_size`` = median over images of the 95th percentile of the crop-mask
    distance transform (the labelled plant size, in pixels).
predict_probs(model, images, stride=2) -> float16 (n,H,W,3)
    probabilities on a stride-``stride`` pixel lattice, bilinearly resized.
oof_probs(images, semantics, n_folds=3, groups=None, seed=0, ...) -> float16 (n,H,W,3)
    out-of-fold probabilities; a group (e.g. one capture) is never split across folds (default groups: consecutive blocks
    of images); the default fit is smaller (100 trees, 6000 px per image).
growth_scale(x, plant_size=10.0) -> float (n,)
    95th percentile of the crop-mask distance transform divided by ``plant_size``, clipped to [0.7, 3]; ``x`` = probabilities,
    a semantics map (crop = 1 or 3) or boolean crop masks.
panoptic_from_probs(probs, params=None) -> prediction dict
    per image: (1) p = Gaussian(probs, ``smooth``); vegetation = p_crop + p_weed > ``veg_thr``; crop = vegetation and
    ``crop_weight`` * p_crop > p_weed (then a binary opening); other vegetation = weed. (2) s = growth_scale of the crop
    mask; the lengths and areas below are multiplied by s (areas by s^2). (3) plants: if s >= ``overlap_scale`` the 8-connected
    components of the crop mask, otherwise split_instances(crop, ``plant_sigma``, ``plant_dist``, ``plant_area``,
    ``plant_close``, ``rel_floor``). (4) leaves: split_instances(crop, ``leaf_sigma``, ``leaf_dist``, ``leaf_area``).
    ``params`` overrides DEFAULT_PARAMS.
split_instances(mask, sigma, min_dist, min_area, close=0, rel_floor=0.2) -> int32 (H,W)
    boolean mask -> instance ids: optional binary closing, Euclidean distance transform smoothed with Gaussian ``sigma``,
    markers = its local maxima in a window of radius ``min_dist``, ``scipy.ndimage.watershed_ift`` on the mask, unmarked
    components keep their own id, ids smaller than min(``min_area``, ``rel_floor`` x largest id) pixels are removed.
pq_plus(pred, gt) -> dict
    PQ+, IoU soil / crop / weed and PQ crop / leaf in percent, defined as in the task objective (visibility filter included);
    ``gt`` = dict with ``semantics``, ``plant_instances``, ``leaf_instances`` and optionally the two ``*_visibility`` arrays,
    e.g. the annotation arrays of labelled images, so a prediction for them can be scored without ``score_dev``.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import ndimage as ndi

__all__ = ["DEFAULT_PARAMS", "PixelModel", "pixel_features", "fit_pixel_classifier", "predict_probs", "oof_probs",
           "growth_scale", "panoptic_from_probs", "split_instances", "pq_plus", "fit_predict"]

DEFAULT_PARAMS = dict(smooth=1.0, veg_thr=0.6, crop_weight=0.6, plant_sigma=8.0, plant_dist=32, plant_area=300,
                      plant_close=4, rel_floor=0.2, leaf_sigma=1.0, leaf_dist=4, leaf_area=12, overlap_scale=1.7,
                      plant_size=10.0)
_LGB = dict(objective="multiclass", num_class=3, learning_rate=0.08, num_leaves=63, min_data_in_leaf=40,
            bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.7, num_threads=2, verbose=-1, force_row_wise=True,
            deterministic=True)
_g = ndi.gaussian_filter


# ------------------------------------------------------------------------------------------------ features
def _otsu(x: np.ndarray, bins: int = 256) -> float:
    h, e = np.histogram(x, bins=bins)
    h = h.astype(float)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def _features_one(img: np.ndarray) -> np.ndarray:
    f = np.asarray(img).astype(np.float32) / 255.0
    R, G, B = f[..., 0], f[..., 1], f[..., 2]
    s = R + G + B + 1e-6
    r, gg, b = R / s, G / s, B / s
    exg = 2 * gg - r - b
    exr = 1.4 * r - gg
    vari = (G - R) / (G + R - B + 1e-2)
    mx, mn = f.max(-1), f.min(-1)
    sat = (mx - mn) / (mx + 1e-6)
    t = _otsu(exg)
    veg = (exg > t).astype(np.float32)
    F = [R, G, B, exg, exr, vari, sat, mx, gg - r]
    for sg in (1.5, 4, 8):
        F += [_g(G, sg), _g(exg, sg), _g(sat, sg)]
    m = _g(exg, 3)
    F.append(np.sqrt(np.maximum(_g(exg ** 2, 3) - m ** 2, 0)))
    F.append(np.hypot(_g(exg, 1.5, order=(0, 1)), _g(exg, 1.5, order=(1, 0))))
    F += [_g(veg, sg) for sg in (6, 12, 24, 48)]
    F.append(np.full_like(exg, t))
    # multi-scale contrast, ridge / blob responses and coarse context
    F += [_g(exg, 2) - _g(exg, 6), _g(exg, 4) - _g(exg, 12), _g(G, 3) - _g(G, 10), _g(exg, 16), _g(exg, 32),
          np.sqrt(np.maximum(_g(exg ** 2, 8) - _g(exg, 8) ** 2, 0)), _g(mx - mn, 16),
          ndi.laplace(_g(exg, 2)),
          _g(exg, 2, order=(0, 2)) + _g(exg, 2, order=(2, 0)), _g(exg, 4, order=(0, 2)) + _g(exg, 4, order=(2, 0))]
    return np.stack(F, -1).astype(np.float32)


def pixel_features(images) -> np.ndarray:
    """Per-pixel features (35 float32 channels, computed from the image itself) of one image (H,W,3) or a stack
    (n,H,W,3); the result has shape (H,W,35) or (n,H,W,35) (about 100 bytes per pixel)."""
    a = np.asarray(images)
    if a.ndim == 3:
        return _features_one(a)
    return np.stack([_features_one(im) for im in a])


# ------------------------------------------------------------------------------------------------ classifier
class PixelModel:
    """Fitted pixel classifier (picklable): ``booster`` (lightgbm.Booster, 3 classes soil/crop/weed), ``plant_size``
    (see fit_pixel_classifier), ``n_trees`` (boosting rounds) and ``fit_s`` (seconds spent fitting)."""

    def __init__(self, booster, plant_size: float, n_trees: int, fit_s: float):
        self.booster, self.plant_size, self.n_trees, self.fit_s = booster, float(plant_size), int(n_trees), float(fit_s)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """(m,35) feature rows -> (m,3) probabilities (soil, crop, weed)."""
        return np.asarray(self.booster.predict(np.asarray(X, dtype=np.float32), num_threads=2)).reshape(len(X), 3)


def _merge_partial(sem: np.ndarray) -> np.ndarray:
    y = np.asarray(sem).astype(np.int64).copy()
    y[y == 3] = 1
    y[y == 4] = 2
    return y


def _crop_scale(crop: np.ndarray) -> float | None:
    if int(crop.sum()) < 500:
        return None
    return float(np.percentile(ndi.distance_transform_edt(crop)[crop], 95))


def fit_pixel_classifier(images, semantics, seed: int = 0, pixels_per_image: int = 12000, n_estimators: int = 200,
                         budget_s: float | None = None) -> PixelModel:
    """LightGBM multiclass (soil / crop / weed; labels 3 and 4 are merged into 1 and 2) on ``pixels_per_image`` pixels per
    image, drawn without replacement and split evenly over the classes present (all pixels of a rarer class are kept), with
    ``n_estimators`` rounds (learning rate 0.08, 63 leaves, 2 threads, seeded). If ``budget_s`` is given, boosting stops after
    that many seconds. ``PixelModel.plant_size`` is the median over images with >= 500 crop pixels of the 95th percentile of
    the distance transform of the labelled crop mask (10.0 if there is none)."""
    import lightgbm as lgb

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    X, Y, sizes = [], [], []
    for im, sem in zip(images, semantics):
        y = _merge_partial(sem)
        sc = _crop_scale(y == 1)
        if sc is not None:
            sizes.append(sc)
        F = _features_one(im)
        F = F.reshape(-1, F.shape[-1])
        yr = y.ravel()
        for c in range(3):
            w = np.flatnonzero(yr == c)
            if len(w):
                i = rng.choice(w, min(pixels_per_image // 3, len(w)), replace=False)
                X.append(F[i])
                Y.append(yr[i])
    X, Y = np.concatenate(X), np.concatenate(Y)
    if len(np.unique(Y)) < 3:
        raise ValueError("fit_pixel_classifier: the training labels must contain soil, crop and weed pixels")
    callbacks = []
    if budget_s is not None:
        def _stop(env):
            if time.perf_counter() - t0 > budget_s and env.iteration >= 5:
                raise lgb.callback.EarlyStopException(env.iteration, [])
        callbacks.append(_stop)
    booster = lgb.train(dict(_LGB, seed=int(seed)), lgb.Dataset(X, Y, free_raw_data=True), num_boost_round=int(n_estimators),
                        callbacks=callbacks)
    n_trees = booster.current_iteration()
    return PixelModel(booster, float(np.median(sizes)) if sizes else 10.0, n_trees, time.perf_counter() - t0)


def predict_probs(model: PixelModel, images, stride: int = 2) -> np.ndarray:
    """Class probabilities (n,H,W,3) float16, channels (soil, crop, weed). The classifier is applied to every
    ``stride``-th pixel in both directions and the probability maps are bilinearly resized to (H,W)."""
    out = []
    for im in np.asarray(images):
        H, W = im.shape[:2]
        F = _features_one(im)[::stride, ::stride]
        h, w, d = F.shape
        p = model.predict_proba(F.reshape(-1, d)).reshape(h, w, 3)
        if (h, w) != (H, W):
            p = ndi.zoom(p, (H / h, W / w, 1), order=1)
        out.append(np.clip(p, 0, 1).astype(np.float16))
    return np.stack(out)


def oof_probs(images, semantics, n_folds: int = 3, groups=None, seed: int = 0, stride: int = 2,
              pixels_per_image: int = 6000, n_estimators: int = 100, budget_s: float | None = None) -> np.ndarray:
    """Out-of-fold class probabilities (n,H,W,3) float16: image i is predicted by a classifier fitted on the images of the
    other folds. ``groups`` (length n) gives each image a group label and a group is never split across folds; the default
    groups are ``n_folds`` consecutive blocks of images. ``budget_s`` bounds each fold's fit."""
    images, semantics = np.asarray(images), np.asarray(semantics)
    n = len(images)
    g = np.asarray(groups) if groups is not None else np.repeat(np.arange(n_folds), np.diff(np.linspace(0, n, n_folds + 1).astype(int)))
    labels = list(dict.fromkeys(g.tolist()))
    if len(labels) < 2:
        raise ValueError("oof_probs: need at least two groups")
    fold_of = {lab: k % min(n_folds, len(labels)) for k, lab in enumerate(labels)}
    fold = np.array([fold_of[v] for v in g.tolist()])
    out = np.zeros(images.shape[:3] + (3,), dtype=np.float16)
    for k in sorted(set(fold.tolist())):
        te, tr = np.flatnonzero(fold == k), np.flatnonzero(fold != k)
        m = fit_pixel_classifier(images[tr], semantics[tr], seed=seed, pixels_per_image=pixels_per_image,
                                 n_estimators=n_estimators, budget_s=budget_s)
        out[te] = predict_probs(m, images[te], stride=stride)
    return out


# ------------------------------------------------------------------------------------------------ instances
def _crop_mask_from_probs(p: np.ndarray, prm: dict) -> tuple[np.ndarray, np.ndarray]:
    pc, pw = _g(p[..., 1].astype(np.float32), prm["smooth"]), _g(p[..., 2].astype(np.float32), prm["smooth"])
    veg = (pc + pw) > prm["veg_thr"]
    crop = ndi.binary_opening(veg & (pc * prm["crop_weight"] > pw))
    return crop, veg


def _as_crop_masks(x, prm: dict) -> list[np.ndarray]:
    a = np.asarray(x)
    if a.ndim == 4 and a.shape[-1] == 3 and a.dtype.kind == "f":
        return [_crop_mask_from_probs(p, prm)[0] for p in a]
    if a.ndim == 2:
        a = a[None]
    if a.dtype == bool:
        return list(a)
    return [(m == 1) | (m == 3) for m in a]


def growth_scale(x, plant_size: float = DEFAULT_PARAMS["plant_size"]) -> np.ndarray:
    """Per-image plant size relative to ``plant_size`` pixels: 95th percentile of the Euclidean distance transform of the
    crop mask divided by ``plant_size``, clipped to [0.7, 3] (1.0 when the mask has < 500 pixels). ``x`` is class
    probabilities (n,H,W,3) (crop mask as in panoptic_from_probs stage 1 with the default parameters), an integer
    semantics map (n,H,W) (crop = label 1 or 3) or boolean crop masks (n,H,W)."""
    out = []
    for m in _as_crop_masks(x, DEFAULT_PARAMS):
        sc = _crop_scale(m)
        out.append(1.0 if sc is None else float(np.clip(sc / plant_size, 0.7, 3.0)))
    return np.asarray(out)


def split_instances(mask, sigma: float, min_dist: float, min_area: float, close: int = 0, rel_floor: float = 0.2) -> np.ndarray:
    """Instance labels (int32, 0 = background) of a boolean mask (H,W): optional binary closing with a disc-like element of
    radius ``close`` (labels are restored to the un-closed mask), Euclidean distance transform smoothed with Gaussian
    ``sigma`` (0 = unsmoothed), markers = local maxima of the smoothed transform (> 1) in a window of radius ``min_dist``,
    ``scipy.ndimage.watershed_ift`` on the closed mask, unmarked components keep their own label, and labels with fewer than
    min(``min_area``, ``rel_floor`` x the largest label's area) pixels are removed."""
    mask = np.asarray(mask, dtype=bool)
    m2 = mask
    if close > 0:
        st = ndi.iterate_structure(ndi.generate_binary_structure(2, 1), int(close))
        m2 = mask | ndi.binary_closing(np.pad(mask, close + 1), structure=st)[close + 1:-close - 1, close + 1:-close - 1]
    lab, n = ndi.label(m2, structure=np.ones((3, 3)))
    if n == 0:
        return lab.astype(np.int32)
    dt = ndi.distance_transform_edt(m2)
    ds = _g(dt, sigma) if sigma > 0 else dt
    mk = (ds == ndi.maximum_filter(ds, size=int(min_dist) * 2 + 1)) & m2 & (ds > 1.0)
    mlab, nm = ndi.label(mk, structure=np.ones((3, 3)))
    out = lab
    if nm:
        surf = (255 - np.clip(ds / max(ds.max(), 1e-6) * 254, 0, 254)).astype(np.uint8)
        surf[~m2] = 255
        out = ndi.watershed_ift(surf, mlab.astype(np.int32))
        out[~m2] = 0
        miss = m2 & (out <= 0)
        out[miss] = lab[miss] + out.max() + 1
    out = out * mask
    ids, cnt = np.unique(out[out > 0], return_counts=True)
    if len(ids):
        out[np.isin(out, ids[cnt < min(min_area, rel_floor * cnt.max())])] = 0
    return out.astype(np.int32)


def panoptic_from_probs(probs, params: dict | None = None) -> dict:
    """Prediction dict (semantics int16, plant_instances int32, leaf_instances int32; each (n,H,W)) from class probabilities
    (n,H,W,3) by the four stages described in the module docstring. ``params`` overrides entries of DEFAULT_PARAMS."""
    prm = dict(DEFAULT_PARAMS, **(params or {}))
    P = np.asarray(probs)
    sem = np.zeros(P.shape[:3], np.int16)
    pl = np.zeros(P.shape[:3], np.int32)
    lf = np.zeros(P.shape[:3], np.int32)
    for i in range(P.shape[0]):
        crop, veg = _crop_mask_from_probs(P[i], prm)
        sem[i][crop] = 1
        sem[i][veg & ~crop] = 2
        s = float(growth_scale(crop[None], prm["plant_size"])[0])
        if s >= prm["overlap_scale"]:
            lab, _ = ndi.label(crop, structure=np.ones((3, 3)))
            a = np.bincount(lab.ravel())
            a[0] = 0
            lab[(a < 16)[lab]] = 0
            pl[i] = lab
        else:
            pl[i] = split_instances(crop, prm["plant_sigma"] * s, round(prm["plant_dist"] * s),
                                    prm["plant_area"] * s * s, round(prm["plant_close"] * s), prm["rel_floor"])
        lf[i] = split_instances(crop, prm["leaf_sigma"] * s, round(prm["leaf_dist"] * s), prm["leaf_area"] * s * s)
    return dict(semantics=sem, plant_instances=pl, leaf_instances=lf)


def fit_predict(train_images, train_semantics, targets, params: dict | None = None, seed: int = 0, stride: int = 2,
                pixels_per_image: int = 12000, n_estimators: int = 200, budget_s: float | None = None):
    """fit_pixel_classifier on the training arrays, then predict_probs and panoptic_from_probs for every image array in the
    list ``targets`` (the plant size of ``params`` defaults to the fitted ``model.plant_size``). Returns a list of prediction
    dicts in the order of ``targets``; if ``targets`` is a single (n,H,W,3) array, returns one dict."""
    single = isinstance(targets, np.ndarray)
    model = fit_pixel_classifier(train_images, train_semantics, seed=seed, pixels_per_image=pixels_per_image,
                                 n_estimators=n_estimators, budget_s=budget_s)
    prm = dict(DEFAULT_PARAMS, plant_size=model.plant_size, **(params or {}))
    outs = [panoptic_from_probs(predict_probs(model, t, stride=stride), prm) for t in ([targets] if single else targets)]
    return outs[0] if single else outs


# ------------------------------------------------------------------------------------------------ metric
def _confusion(pred_sem: np.ndarray, gt_sem: np.ndarray) -> np.ndarray:
    p, g = _merge_partial(pred_sem).ravel(), _merge_partial(gt_sem).ravel()
    ok = (p >= 0) & (p < 3) & (g >= 0) & (g < 3)
    return np.bincount(g[ok] * 3 + p[ok], minlength=9).reshape(3, 3).astype(np.int64)


def _pq_class(pred_inst: np.ndarray, gt_inst: np.ndarray) -> float:
    """PQ of one class in one image from instance maps that are already masked to the class (0 = none): matches are the
    prediction/ground-truth pairs with IoU > 0.5; PQ = sum IoU / (TP + FP/2 + FN/2)."""
    pl, pa = np.unique(pred_inst[pred_inst > 0], return_counts=True)
    gl, ga = np.unique(gt_inst[gt_inst > 0], return_counts=True)
    tp, num = 0, 0.0
    both = (pred_inst > 0) & (gt_inst > 0)
    if both.any():
        pi = np.searchsorted(pl, pred_inst[both])
        gi = np.searchsorted(gl, gt_inst[both])
        pairs, cnt = np.unique(pi.astype(np.int64) * len(gl) + gi, return_counts=True)
        p_idx, g_idx = pairs // len(gl), pairs % len(gl)
        iou = cnt / (pa[p_idx] + ga[g_idx] - cnt)
        hit = iou > 0.5
        tp, num = int(hit.sum()), float(iou[hit].sum())
    den = tp + 0.5 * (len(pl) - tp) + 0.5 * (len(gl) - tp)
    return num / den if den > 0 else float("nan")


def _drop_partial(pred_inst, pred_sem, gt_inst, gt_sem, gt_vis) -> None:
    """Ground-truth instances whose maximum visibility is <= 0.5 are removed from ``gt_sem``; predicted instances with more
    than half of their pixels inside one such ground-truth instance are removed from ``pred_sem`` (in place)."""
    ids, inv = np.unique(gt_inst.ravel(), return_inverse=True)
    vmax = np.zeros(len(ids))
    np.maximum.at(vmax, inv, np.asarray(gt_vis, dtype=float).ravel())
    partial = ids[(ids > 0) & (vmax <= 0.5)]
    if len(partial) == 0:
        return
    in_partial = np.isin(gt_inst, partial)
    gt_sem[in_partial] = 0
    sel = in_partial & (pred_inst > 0)
    if not sel.any():
        return
    pl, pa = np.unique(pred_inst[pred_inst > 0], return_counts=True)
    area = dict(zip(pl.tolist(), pa.tolist()))
    pairs, cnt = np.unique(np.stack([pred_inst[sel], gt_inst[sel]], 1), axis=0, return_counts=True)
    drop = [int(a) for (a, _), c in zip(pairs, cnt) if c / (area[int(a)] + 1e-12) > 0.5]
    if drop:
        pred_sem[np.isin(pred_inst, drop)] = 0


def pq_plus(pred: dict, gt: dict) -> dict:
    """PQ+ = mean(IoU_soil, IoU_weed, PQ_crop, PQ_leaf) in percent, with the definitions of the task objective. IoU: pixel
    confusion matrix pooled over the images after mapping labels 3 -> 1 and 4 -> 2. PQ_crop: per image, ground-truth plant
    instances with maximum ``plant_visibility`` <= 0.5 are removed from the ground-truth semantics and predicted plants lying
    > 50 % inside one of them from the prediction; PQ of class 1 with IoU > 0.5 matches; averaged over the images that contain
    ground-truth crop. PQ_leaf: the same on ``leaf_instances`` (class = instance > 0) with ``leaf_visibility``. ``pred``:
    prediction dict; ``gt``: dict with ``semantics``, ``plant_instances``, ``leaf_instances`` and optionally
    ``plant_visibility`` / ``leaf_visibility`` (missing = fully visible). Returns pq_plus, iou_soil, iou_crop, iou_weed,
    pq_crop, pq_leaf (None when no image has that class)."""
    ps, gs = np.asarray(pred["semantics"]), np.asarray(gt["semantics"])
    n = len(gs)
    cm = np.zeros((3, 3), np.int64)
    crop_pq, leaf_pq = [], []
    for i in range(n):
        cm += _confusion(ps[i], gs[i])
        p_sem, g_sem = ps[i].astype(np.int64).copy(), gs[i].astype(np.int64).copy()
        p_ins, g_ins = np.asarray(pred["plant_instances"][i]).astype(np.int64), np.asarray(gt["plant_instances"][i]).astype(np.int64)
        vis = gt.get("plant_visibility")
        _drop_partial(p_ins, p_sem, g_ins, g_sem, np.ones(g_ins.shape) if vis is None else np.asarray(vis[i]))
        if (g_sem == 1).any():
            crop_pq.append(_pq_class(p_ins * (p_sem == 1), g_ins * (g_sem == 1)))
        pl_ins, gl_ins = np.asarray(pred["leaf_instances"][i]).astype(np.int64), np.asarray(gt["leaf_instances"][i]).astype(np.int64)
        pl_sem, gl_sem = (pl_ins > 0).astype(np.int64), (gl_ins > 0).astype(np.int64)
        vis = gt.get("leaf_visibility")
        _drop_partial(pl_ins, pl_sem, gl_ins, gl_sem, np.ones(gl_ins.shape) if vis is None else np.asarray(vis[i]))
        if (gl_sem == 1).any():
            leaf_pq.append(_pq_class(pl_ins * (pl_sem == 1), gl_ins * (gl_sem == 1)))
    den = cm.sum(0) + cm.sum(1) - np.diag(cm)
    iou = np.where(den > 0, np.diag(cm) / np.maximum(den, 1), 0.0) * 100.0
    crop_pq = [v for v in crop_pq if np.isfinite(v)]
    leaf_pq = [v for v in leaf_pq if np.isfinite(v)]
    pc = 100.0 * float(np.mean(crop_pq)) if crop_pq else None
    pf = 100.0 * float(np.mean(leaf_pq)) if leaf_pq else None
    parts = [v for v in (float(iou[0]), float(iou[2]), pc, pf) if v is not None]
    return {"pq_plus": float(np.mean(parts)), "iou_soil": float(iou[0]), "iou_crop": float(iou[1]), "iou_weed": float(iou[2]),
            "pq_crop": pc, "pq_leaf": pf}
