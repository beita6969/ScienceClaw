"""Pixel classification of field images from frozen DINOv2 features (weights stored locally, no network), for use with the
panoptic stage of ``scilib.phenoseg``. Same array layout as ``scilib.phenoseg``: ``images`` uint8 (n,H,W,3), ``semantics``
int (n,H,W) with 0 soil, 1 crop, 2 weed, 3 partial crop, 4 partial weed; probabilities float16 (n,H,W,3) in the channel
order (soil, crop, weed), so they go straight into ``phenoseg.panoptic_from_probs``.

available() -> bool
    True when torch, transformers, lightgbm and the stored DINOv2 weights can be loaded in this interpreter.
fit_deep_classifier(images, semantics, backbone="dinov2_large", size=728, pca_dim=64, use_hand=True, seed=0,
                    pixels_per_image=12000, n_estimators=200, budget_s=None, device="auto") -> DeepPixelModel
    Runs the frozen DINOv2 encoder (ViT-L/14 'dinov2_large' or ViT-B/14 'dinov2_base'; the image is resized so that its
    longer side is about ``size`` pixels, a multiple of 14 per side, and normalised with the ImageNet statistics) and takes
    the last-layer patch features (one per 14x14 patch of the resized image). PCA (fitted on the training patches) reduces
    them to ``pca_dim``; the features of a pixel are the bilinear interpolation of the reduced patch grid at that pixel,
    plus, if ``use_hand``, the 35 hand features of ``phenoseg.pixel_features``. A LightGBM multiclass model (the settings of
    ``phenoseg.fit_pixel_classifier``: class-balanced sample of ``pixels_per_image`` pixels per image, ``n_estimators``
    rounds, 2 threads) is fitted on the labelled pixels; labels 3 and 4 are merged into 1 and 2. ``model.plant_size`` is
    defined as for ``phenoseg.PixelModel``. The encoder runs in float32; a run is repeatable on the same device.
predict_deep_probs(model, images, stride=2, device="auto") -> float16 (n,H,W,3)
    class probabilities on a stride-``stride`` pixel lattice, bilinearly resized to (H,W), as ``phenoseg.predict_probs``.
fit_predict_deep(train_images, train_semantics, targets, params=None, seed=0, stride=2, **fit_kwargs) -> list of prediction dicts
    fit_deep_classifier, predict_deep_probs and ``phenoseg.panoptic_from_probs`` for every image array in ``targets`` (a
    single array returns one dict); ``params`` overrides ``phenoseg.DEFAULT_PARAMS``, whose ``plant_size`` defaults to the
    fitted ``model.plant_size``. ``fit_kwargs`` are the keyword arguments of fit_deep_classifier.
patch_features(images, backbone="dinov2_large", size=728, device="auto") -> float32 (n, gh, gw, C)
    the encoder's patch feature grids (C = 1024 for ViT-L, 768 for ViT-B); repeated calls on the same images are cached in
    the process.

Model: DINOv2 (Oquab et al., TMLR 2024; Apache-2.0), self-supervised on the LVD-142M image collection without labels; no
PhenoBench annotations were used. Cost with a GPU: about 0.1-0.3 s per image for ViT-L at size 728, 20-40 s to fit the
classifier on 24 labelled images, 1-3 s to load the weights in every process; on CPU ViT-L takes 6-15 s per image at
size 728 (ViT-B roughly a third of that; a smaller ``size`` such as 476 is proportionally cheaper).
"""
from __future__ import annotations

import hashlib
import time

import numpy as np

from . import phenoseg as ps
from ._pretrained import have_module, model_path, set_cpu_threads, switched_off, torch_device

__all__ = ["available", "patch_features", "fit_deep_classifier", "predict_deep_probs", "fit_predict_deep", "DeepPixelModel"]

_BACKBONES = {"dinov2_large": "dinov2-large", "dinov2_base": "dinov2-base"}
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_models: dict = {}
_cache: dict = {}
_CACHE_MAX = 96


def _weights(backbone: str):
    d = _BACKBONES.get(backbone)
    if d is None:
        raise ValueError(f"backbone must be one of {sorted(_BACKBONES)}")
    p = model_path(d)
    return p if p is not None and (p / "model.safetensors").is_file() and (p / "config.json").is_file() else None


def available() -> bool:
    return (not switched_off() and have_module("torch") and have_module("transformers") and have_module("lightgbm")
            and any(_weights(b) is not None for b in _BACKBONES))


def _encoder(backbone: str, device: str):
    key = (backbone, device)
    if key not in _models:
        import torch
        from transformers import AutoModel
        w = _weights(backbone)
        if w is None:
            raise RuntimeError(f"phenoseg_deep: the stored weights of {backbone} are not available here")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        m = AutoModel.from_pretrained(str(w), local_files_only=True, torch_dtype=torch.float32)
        _models[key] = m.to(device).eval()
    return _models[key]


def _grid_size(H: int, W: int, size: int) -> tuple[int, int]:
    s = float(size) / max(H, W)
    return max(1, round(H * s / 14)), max(1, round(W * s / 14))


def patch_features(images, backbone: str = "dinov2_large", size: int = 728, device: str = "auto") -> np.ndarray:
    import torch
    import torch.nn.functional as Fn
    X = np.asarray(images)
    if X.ndim != 4 or X.shape[-1] != 3:
        raise ValueError(f"images must have shape (n,H,W,3), got {X.shape}")
    dev = torch_device(device)
    if dev == "cpu":
        set_cpu_threads(4)
    net = _encoder(backbone, dev)
    n, H, W, _ = X.shape
    gh, gw = _grid_size(H, W, size)
    out: list = [None] * n
    todo = []
    for i in range(n):
        key = (backbone, int(size), hashlib.blake2b(np.ascontiguousarray(X[i]).tobytes(), digest_size=16).hexdigest())
        if key in _cache:
            out[i] = _cache[key]
        else:
            todo.append((i, key))
    mean = torch.tensor(_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=dev).view(1, 3, 1, 1)
    step = 4 if dev != "cpu" else 1
    with torch.no_grad():
        for a in range(0, len(todo), step):
            chunk = todo[a:a + step]
            x = torch.from_numpy(np.stack([X[i] for i, _ in chunk])).to(dev).permute(0, 3, 1, 2).float() / 255.0
            x = Fn.interpolate(x, size=(gh * 14, gw * 14), mode="bicubic", align_corners=False, antialias=True)
            x = ((x.clamp(0, 1) - mean) / std).contiguous()
            h = net(pixel_values=x).last_hidden_state[:, 1:, :]
            h = h.reshape(len(chunk), gh, gw, -1).float().cpu().numpy()
            for (i, key), f in zip(chunk, h):
                out[i] = f
                if len(_cache) < _CACHE_MAX:
                    _cache[key] = f
    return np.stack(out)


class DeepPixelModel:
    """Fitted classifier (picklable): ``booster`` (LightGBM, classes soil / crop / weed), ``plant_size`` (see
    ``phenoseg.PixelModel``), ``backbone``, ``size``, PCA ``mean`` and ``components`` (pca_dim, C), ``use_hand``, ``n_trees``,
    ``fit_s``."""

    def __init__(self, booster, plant_size, backbone, size, mean, components, use_hand, n_trees, fit_s):
        self.booster, self.plant_size, self.backbone, self.size = booster, float(plant_size), backbone, int(size)
        self.mean, self.components, self.use_hand = mean, components, bool(use_hand)
        self.n_trees, self.fit_s = int(n_trees), float(fit_s)


def _reduce(grids: np.ndarray, mean: np.ndarray, comps: np.ndarray) -> np.ndarray:
    n, gh, gw, C = grids.shape
    return ((grids.reshape(-1, C) - mean) @ comps.T).reshape(n, gh, gw, -1).astype(np.float32)


def _interp(grid: np.ndarray, H: int, W: int, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a (gh,gw,D) grid, laid over the (H,W) image (cell centres at (i + 0.5) * H / gh), at the
    pixel centres (ys, xs); positions outside the outermost cell centres take the border value."""
    gh, gw, _ = grid.shape
    gy = np.clip((np.asarray(ys, np.float64) + 0.5) * gh / H - 0.5, 0, gh - 1)
    gx = np.clip((np.asarray(xs, np.float64) + 0.5) * gw / W - 0.5, 0, gw - 1)
    y0, x0 = np.floor(gy).astype(int), np.floor(gx).astype(int)
    y1, x1 = np.minimum(y0 + 1, gh - 1), np.minimum(x0 + 1, gw - 1)
    wy, wx = (gy - y0)[:, None].astype(np.float32), (gx - x0)[:, None].astype(np.float32)
    return (grid[y0, x0] * (1 - wy) * (1 - wx) + grid[y0, x1] * (1 - wy) * wx
            + grid[y1, x0] * wy * (1 - wx) + grid[y1, x1] * wy * wx)


def _pixel_features(grid_red: np.ndarray, im: np.ndarray, use_hand: bool, ys: np.ndarray, xs: np.ndarray,
                    hand: np.ndarray | None = None) -> np.ndarray:
    H, W = im.shape[:2]
    parts = [_interp(grid_red, H, W, ys, xs)]
    if use_hand:
        parts.append(hand[ys, xs] if hand is not None else ps._features_one(im)[ys, xs])
    return np.concatenate(parts, axis=1).astype(np.float32)


def fit_deep_classifier(images, semantics, backbone: str = "dinov2_large", size: int = 728, pca_dim: int = 64,
                        use_hand: bool = True, seed: int = 0, pixels_per_image: int = 12000, n_estimators: int = 200,
                        budget_s: float | None = None, device: str = "auto") -> DeepPixelModel:
    import lightgbm as lgb

    t0 = time.perf_counter()
    X_img = np.asarray(images)
    grids = patch_features(X_img, backbone, size, device)
    C = grids.shape[-1]
    flat = grids.reshape(-1, C).astype(np.float64)
    mean = flat.mean(0)
    k = int(min(pca_dim, C, len(flat)))
    evals, evecs = np.linalg.eigh(np.cov(flat - mean, rowvar=False))
    comps = evecs[:, np.argsort(evals)[::-1][:k]].T.astype(np.float32)
    mean = mean.astype(np.float32)
    red = _reduce(grids, mean, comps)
    rng = np.random.default_rng(seed)
    Xs, Ys, sizes = [], [], []
    for im, sem, g in zip(X_img, semantics, red):
        y = ps._merge_partial(sem)
        sc = ps._crop_scale(y == 1)
        if sc is not None:
            sizes.append(sc)
        H, W = y.shape
        hand = ps._features_one(im) if use_hand else None
        yr = y.ravel()
        for c in range(3):
            w = np.flatnonzero(yr == c)
            if len(w):
                i = rng.choice(w, min(pixels_per_image // 3, len(w)), replace=False)
                Xs.append(_pixel_features(g, im, use_hand, i // W, i % W, hand))
                Ys.append(yr[i])
    X, Y = np.concatenate(Xs), np.concatenate(Ys)
    if len(np.unique(Y)) < 3:
        raise ValueError("fit_deep_classifier: the training labels must contain soil, crop and weed pixels")
    callbacks = []
    if budget_s is not None:
        def _stop(env):
            if time.perf_counter() - t0 > budget_s and env.iteration >= 5:
                raise lgb.callback.EarlyStopException(env.iteration, [])
        callbacks.append(_stop)
    booster = lgb.train(dict(ps._LGB, seed=int(seed)), lgb.Dataset(X, Y, free_raw_data=True), num_boost_round=int(n_estimators),
                        callbacks=callbacks)
    return DeepPixelModel(booster, float(np.median(sizes)) if sizes else 10.0, backbone, size, mean, comps, use_hand,
                          booster.current_iteration(), time.perf_counter() - t0)


def predict_deep_probs(model: DeepPixelModel, images, stride: int = 2, device: str = "auto") -> np.ndarray:
    from scipy import ndimage as ndi
    X_img = np.asarray(images)
    red = _reduce(patch_features(X_img, model.backbone, model.size, device), model.mean, model.components)
    out = []
    for im, g in zip(X_img, red):
        H, W = im.shape[:2]
        ys0, xs0 = np.arange(0, H, stride), np.arange(0, W, stride)
        yy, xx = np.meshgrid(ys0, xs0, indexing="ij")
        hand = ps._features_one(im) if model.use_hand else None
        F = _pixel_features(g, im, model.use_hand, yy.ravel(), xx.ravel(), hand)
        h, w = len(ys0), len(xs0)
        p = np.asarray(model.booster.predict(F, num_threads=2)).reshape(h, w, 3)
        if (h, w) != (H, W):
            p = ndi.zoom(p, (H / h, W / w, 1), order=1)
        out.append(np.clip(p, 0, 1).astype(np.float16))
    return np.stack(out)


def fit_predict_deep(train_images, train_semantics, targets, params: dict | None = None, seed: int = 0, stride: int = 2,
                     **fit_kwargs):
    single = isinstance(targets, np.ndarray)
    device = fit_kwargs.get("device", "auto")
    model = fit_deep_classifier(train_images, train_semantics, seed=seed, **fit_kwargs)
    prm = dict(ps.DEFAULT_PARAMS, plant_size=model.plant_size, **(params or {}))
    outs = [ps.panoptic_from_probs(predict_deep_probs(model, t, stride=stride, device=device), prm)
            for t in ([targets] if single else targets)]
    return outs[0] if single else outs
