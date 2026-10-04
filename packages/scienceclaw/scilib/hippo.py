"""Tools for voxel-wise segmentation of the hippocampus (0 background, 1 anterior, 2 posterior) in small 3-D T1-weighted
MRI crops. Volumes are 3-D numpy arrays of any intensity scale and any shape (the ``images`` / ``labels`` / ``case_ids``
outputs of the load_* tools are passed as they are); nothing is read from disk, everything is computed from the arrays
that are passed in. Runs on CPU (LightGBM with 2 threads, numpy/scipy). All functions are deterministic given ``seed``.

Entry point
    fit_predict(train_images, train_labels, eval_images, train_ids=None, seed=0, **kw) -> list of uint8 arrays
        Fits HippocampusSegmenter on the labelled training volumes and returns one label volume per evaluation image (same
        shape as the image, values 0/1/2), in the order of ``eval_images``; any list of images can be passed (for example
        the dev images followed by the evaluation images). ``train_ids`` (optional case ids such as 'hippocampus_123')
        makes the fusion features of the training volumes exclude the volumes of the same subject.

Metric (identical to the task metric)
    dsc(pred, gt) -> float                            Dice 2|P&G| / (|P| + |G|) of two masks (1.0 if both are empty)
    case_dsc(pred, gt) -> (dsc_1, dsc_2)              per-label Dice of one label volume
    mean_dsc(preds, gts) -> float                     mean over cases of the mean of the label-1 and label-2 Dice

Building blocks
    normalize_intensity(img, method="robust") -> float32 array
        Per-volume normalisation without absolute-scale dependence: "robust" = (x - median) / (p95 - p5) clipped to
        [-3, 3] (invariant to affine rescaling); "rank" = mid-rank in [0, 1] (invariant to any monotone rescaling).
    subject_groups(case_ids) -> list[int]             subject index (id + 1) // 2 of ids such as 'hippocampus_123'
                                                      (volumes 2k-1 and 2k are the two crops of one scan)
    LocationAtlas().fit(labels)                       class frequencies of the training masks on a normalised
        .prob(shape) -> (3, *shape)                   48x64x48 grid; prior probability of every class at every voxel
        .support(shape, margin) -> bool array         voxels within ``margin`` voxels of the union of the two labels' support
    label_fusion(img, atlas_images, atlas_labels, k=10, search=1, patch=3, max_shift=4, margin=4) -> (3, *shape)
        Patch-based multi-atlas label fusion: every atlas volume is translated (integer shift up to ``max_shift`` voxels
        around centre alignment, masked squared difference of the rank-normalised intensities inside the atlas-support
        region) onto the target; the ``k`` best-matching atlases vote at every voxel with weights
        exp(-D / h2), D = mean squared difference of ``patch``^3 rank-normalised-intensity patches between the target voxel
        and the atlas voxels within +-``search`` voxels, h2 = the voxel's smallest D. Returns class probabilities.
    voxel_features(img, atlas=None, fusion=None) -> (X, names)
        (n_voxels, n_features) float32 matrix, voxels in C order: normalised intensities, Gaussian-smoothed intensities,
        difference of Gaussians, gradient magnitudes, local mean/std, coordinates (normalised and centred), atlas
        probabilities and distances to the atlas centroids, and (if ``fusion`` = label_fusion output is given) the
        fusion probabilities, their smoothed foreground mass and the signed distance to the fused foreground.
    HippocampusSegmenter(...).fit(images, labels, groups=None)
        LightGBM voxel classifier on voxel_features restricted to the neighbourhood of the atlas support (all other voxels
        are background); fusion features of a training volume come from the other training volumes (leave-one-out, or
        leave-subject-out with ``groups``).
        .predict_proba(img) -> (3, *shape); .predict(img) -> uint8 labels (probabilities smoothed with a Gaussian of
        ``smooth`` voxels, background probability multiplied by ``background_weight``, then postprocess).
    HippocampusSegmenter parameters (defaults; ``fit_predict(..., **kw)`` forwards kw to the constructor): LightGBM
        n_estimators=100, learning_rate=0.12, num_leaves=31, min_child_samples=40; training-voxel sampling fg_fraction=0.5
        (share of the foreground voxels of every training volume) and bg_fraction=0.15 (share of the background voxels
        inside the candidate region); margin=4 (candidate region, voxels); fusion_k=10, fusion_search=1, fusion_patch=3,
        fusion_max_shift=4 (label_fusion arguments k, search, patch, max_shift); smooth=0.7, background_weight=1.4
        (predict-time post-processing); seed=0; n_jobs=2.
    postprocess(prob, smooth=0.0, background_weight=1.0) -> uint8
        argmax; largest connected foreground component; holes filled; the largest connected component of each label
        (other components take the other label).
    cross_validate(images, labels, case_ids=None, n_folds=4, folds=None, **kw) -> dict
        Subject-grouped K-fold of fit_predict (cost: one fit_predict per evaluated fold); ``folds`` = list of fold indices
        to evaluate (default all). Returns mean_dsc, dsc_anterior, dsc_posterior, per_case (n, 2), seconds.

Approximate CPU cost of fit_predict with 28 training volumes: ~1.5 s per training volume and ~1 s per evaluation volume.
"""
from __future__ import annotations

import re
import time

import numpy as np

__all__ = ["dsc", "case_dsc", "mean_dsc", "normalize_intensity", "subject_groups", "LocationAtlas", "label_fusion",
           "voxel_features", "postprocess", "HippocampusSegmenter", "fit_predict", "cross_validate"]

ATLAS_GRID = (48, 64, 48)
CLASSES = (1, 2)


# ----------------------------------------------------------------------------------------------------------------
# metric
# ----------------------------------------------------------------------------------------------------------------
def dsc(pred, gt) -> float:
    """Dice 2|P&G| / (|P| + |G|) of two masks (nonzero = inside); 1.0 when both are empty."""
    p, g = np.asarray(pred).astype(bool), np.asarray(gt).astype(bool)
    den = int(p.sum()) + int(g.sum())
    if den == 0:
        return 1.0
    return float(2.0 * np.logical_and(p, g).sum() / den)


def case_dsc(pred, gt) -> tuple[float, float]:
    """(Dice of label 1, Dice of label 2) of one predicted label volume against its ground truth."""
    pred, gt = np.asarray(pred), np.asarray(gt)
    return tuple(dsc(pred == c, gt == c) for c in CLASSES)


def mean_dsc(preds, gts) -> float:
    """Task metric: mean over cases of the mean over labels 1 and 2 of the Dice coefficient."""
    return float(np.mean([np.mean(case_dsc(p, g)) for p, g in zip(preds, gts)]))


# ----------------------------------------------------------------------------------------------------------------
# intensity normalisation, groups
# ----------------------------------------------------------------------------------------------------------------
def _as_volume(img) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim != 3:
        raise ValueError(f"expected a 3-D volume, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        a = np.nan_to_num(a.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return a


def normalize_intensity(img, method: str = "robust") -> np.ndarray:
    """Per-volume intensity normalisation (float32) that does not depend on the scale/encoding of the input.
    "robust": (x - median) / (p95 - p5), clipped to [-3, 3]. "rank": mid-rank of every voxel among all voxels of the volume
    in [0, 1] (ties share their mean rank)."""
    a = _as_volume(img).astype(np.float64)
    if method == "robust":
        p5, p50, p95 = np.percentile(a, [5, 50, 95])
        return np.clip((a - p50) / max(p95 - p5, 1e-6 * max(abs(p50), 1.0)), -3, 3).astype(np.float32)
    if method == "rank":
        flat = a.ravel()
        srt = np.sort(flat)
        lo = np.searchsorted(srt, flat, side="left")
        hi = np.searchsorted(srt, flat, side="right")
        return ((lo + hi - 1) / 2.0 / max(1, flat.size - 1)).reshape(a.shape).astype(np.float32)
    raise ValueError(f"unknown method {method!r}; use 'robust' or 'rank'")


def subject_groups(case_ids) -> list[int]:
    """Subject index (id + 1) // 2 from the trailing integer of every id (e.g. 'hippocampus_123' -> 62)."""
    out = []
    for c in case_ids:
        m = re.search(r"(\d+)\D*$", str(c))
        if m is None:
            raise ValueError(f"no trailing integer in case id {c!r}")
        out.append((int(m.group(1)) + 1) // 2)
    return out


# ----------------------------------------------------------------------------------------------------------------
# location atlas
# ----------------------------------------------------------------------------------------------------------------
def _grid_index(shape_from, shape_to):
    axes = [np.minimum(((np.arange(t) + 0.5) * f / t).astype(int), f - 1) for f, t in zip(shape_from, shape_to)]
    return np.ix_(*axes)


class LocationAtlas:
    """Class frequencies (background, label 1, label 2) of the training masks on a normalised grid (each mask is resampled
    to ``grid`` by nearest neighbour and the frequencies are averaged). ``prob(shape)`` interpolates them linearly at the
    voxel centres of a volume of the given shape (normalised coordinates)."""

    def __init__(self, grid=ATLAS_GRID):
        self.grid = tuple(int(g) for g in grid)

    def fit(self, labels) -> "LocationAtlas":
        labels = [np.asarray(lab).astype(int) for lab in labels]
        if not labels or not any((lab > 0).any() for lab in labels):
            raise ValueError("LocationAtlas.fit needs at least one label volume with foreground voxels")
        acc = np.zeros((3, *self.grid), dtype=np.float64)
        for lab in labels:
            g = lab[_grid_index(lab.shape, self.grid)]
            for c in range(3):
                acc[c] += (g == c)
        self.freq_ = (acc / len(labels)).astype(np.float32)
        gs = np.stack(np.meshgrid(*[(np.arange(n) + 0.5) / n for n in self.grid], indexing="ij"))
        self.centroid_ = [((self.freq_[c] * gs).sum((1, 2, 3)) / max(float(self.freq_[c].sum()), 1e-9)) for c in CLASSES]
        self._cache: dict = {}
        return self

    def prob(self, shape) -> np.ndarray:
        """(3, *shape) float32 prior probabilities (background, label 1, label 2)."""
        from scipy import ndimage

        shape = tuple(int(s) for s in shape)
        if ("prob", shape) not in self._cache:
            axes = [(np.arange(s) + 0.5) / s * n - 0.5 for s, n in zip(shape, self.grid)]
            coords = np.stack(np.meshgrid(*axes, indexing="ij"))
            self._cache[("prob", shape)] = np.stack(
                [ndimage.map_coordinates(self.freq_[c], coords, order=1, mode="nearest") for c in range(3)]
            ).astype(np.float32)
        return self._cache[("prob", shape)]

    def support(self, shape, margin: int = 4) -> np.ndarray:
        """Boolean mask of the voxels within ``margin`` voxels of the union of the two labels' support (frequency > 0)."""
        from scipy import ndimage

        sup = self.prob(shape)[1:].sum(0) > 0
        return ndimage.binary_dilation(sup, iterations=int(margin)) if margin > 0 else sup


# ----------------------------------------------------------------------------------------------------------------
# patch-based multi-atlas label fusion (translation alignment)
# ----------------------------------------------------------------------------------------------------------------
def _window(a: np.ndarray, start, size, mode: str) -> np.ndarray:
    """a[start : start + size] per axis, with coordinates outside the array filled by ``mode`` ('edge' or 'constant')."""
    start = np.asarray(start, dtype=int)
    size = np.asarray(size, dtype=int)
    lo = np.clip(start, 0, np.array(a.shape))
    hi = np.clip(start + size, 0, np.array(a.shape))
    core = a[tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))]
    if core.size == 0:
        return np.zeros(tuple(size), dtype=a.dtype)
    pad_before = np.maximum(lo - start, 0)
    pad_after = np.maximum(start + size - hi, 0)
    if pad_before.any() or pad_after.any():
        core = np.pad(core, [(int(b), int(c)) for b, c in zip(pad_before, pad_after)], mode=mode)
    return core


def _fuse(T: np.ndarray, atlases: list, M: np.ndarray, k: int, search: int, patch: int, max_shift: int) -> np.ndarray:
    """Core of label_fusion on normalised intensities: T target, atlases = [(normalised image, int labels)], M ROI mask."""
    from scipy import ndimage, signal

    shape = np.array(T.shape)
    idx = np.argwhere(M)
    lo, hi = idx.min(0), idx.max(0) + 1
    box = hi - lo
    R, S, pe = int(max_shift), int(search), int(patch) // 2
    MT = (M * T)[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))].astype(np.float32)
    Mb = M[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))].astype(np.float32)
    const = float((MT * MT).sum())
    n_roi = float(Mb.sum())
    scored = []
    for A, L in atlases:
        c = (np.array(A.shape) - shape) // 2
        Aw = _window(A, lo + c - R, box + 2 * R, "edge").astype(np.float32)
        cross = signal.correlate(Aw, MT, mode="valid", method="fft")
        sq = signal.correlate(Aw * Aw, Mb, mode="valid", method="fft")
        ssd = const - 2.0 * cross + sq
        o = np.array(np.unravel_index(int(np.argmin(ssd)), ssd.shape)) - R
        scored.append((float(ssd.min()) / n_roi, c + o, A, L))
    scored.sort(key=lambda t: t[0])
    scored = scored[: max(1, int(k))]
    ext = pe + S
    tb_lo = lo - pe
    Tb = _window(T, tb_lo, box + 2 * pe, "edge").astype(np.float32)
    offs = [(i, j, l) for i in range(-S, S + 1) for j in range(-S, S + 1) for l in range(-S, S + 1)]
    if len(scored) * len(offs) * int(box.prod()) * 4 > 6e8:
        raise ValueError("label_fusion: k * (2*search+1)^3 * region size is too large; reduce k or search")
    cen = tuple(slice(pe, pe + int(b)) for b in box)
    Ds = np.empty((len(scored) * len(offs), *box), dtype=np.float32)
    Ls = np.empty((len(scored) * len(offs), *box), dtype=np.uint8)
    n = 0
    for _, shift, A, L in scored:
        start = tb_lo + shift - S
        Aw = _window(A, start, box + 2 * pe + 2 * S, "edge").astype(np.float32)
        Lw = _window(L, start, box + 2 * pe + 2 * S, "constant").astype(np.uint8)
        for (i, j, l) in offs:
            sl = tuple(slice(S + d, S + d + int(b) + 2 * pe) for d, b in zip((i, j, l), box))
            diff = Tb - Aw[sl]
            D = ndimage.uniform_filter(diff * diff, patch, mode="nearest")
            Ds[n] = D[cen]
            Ls[n] = Lw[tuple(slice(pe + S + d, pe + S + d + int(b)) for d, b in zip((i, j, l), box))]
            n += 1
    h2 = Ds.min(0) + 1e-4
    W = np.exp(-Ds / h2)
    tot = W.sum(0) + 1e-12
    prob = np.zeros((3, *T.shape), dtype=np.float32)
    prob[0] = 1.0
    sl_box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    for cl in range(3):
        prob[cl][sl_box] = ((W * (Ls == cl)).sum(0) / tot).astype(np.float32)
    return prob


def label_fusion(img, atlas_images, atlas_labels, k: int = 10, search: int = 1, patch: int = 3, max_shift: int = 4,
                 margin: int = 4, atlas: LocationAtlas | None = None) -> np.ndarray:
    """Patch-based multi-atlas label fusion of ``img`` from labelled atlas volumes; returns (3, *img.shape) float32 class
    probabilities (background outside the region within ``margin`` voxels of the location-atlas support; ``atlas`` defaults
    to LocationAtlas fitted on ``atlas_labels``). See the module description for the algorithm."""
    atlas = atlas if atlas is not None else LocationAtlas().fit(atlas_labels)
    T = normalize_intensity(img, "rank")
    ats = [(normalize_intensity(a, "rank"), np.asarray(lb).astype(np.uint8)) for a, lb in zip(atlas_images, atlas_labels)]
    return _fuse(T, ats, atlas.support(T.shape, margin), k, search, patch, max_shift)


# ----------------------------------------------------------------------------------------------------------------
# features
# ----------------------------------------------------------------------------------------------------------------
def voxel_features(img, atlas: LocationAtlas | None = None, fusion: np.ndarray | None = None, sigmas=(1.0, 2.0, 3.0)):
    """(n_voxels, n_features) float32 feature matrix of one volume (voxels in C order) and the feature names; see the module
    description. Distances are in voxels (1 mm)."""
    from scipy import ndimage

    a = _as_volume(img)
    shape = a.shape
    rank = normalize_intensity(a, "rank")
    rob = normalize_intensity(a, "robust")
    feats, names = [rank, rob], ["int_rank", "int_robust"]
    sm = {}
    for s in sigmas:
        sm[s] = ndimage.gaussian_filter(rob, s, mode="nearest")
        feats.append(sm[s])
        names.append(f"gauss_s{s:g}")
    feats.append(sm[sigmas[0]] - sm[sigmas[-1]])
    names.append("dog")
    for s in (1.0, 2.0):
        feats.append(ndimage.gaussian_gradient_magnitude(rob, s, mode="nearest"))
        names.append(f"gradmag_s{s:g}")
    m1 = ndimage.uniform_filter(rob, 5, mode="nearest")
    v1 = np.maximum(ndimage.uniform_filter(rob * rob, 5, mode="nearest") - m1 * m1, 0)
    feats += [m1, np.sqrt(v1)]
    names += ["locmean5", "locstd5"]
    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    feats += [(g + 0.5) / shape[i] for i, g in enumerate(grids)]
    names += [f"pos_norm{i}" for i in range(3)]
    feats += [g - (shape[i] - 1) / 2.0 for i, g in enumerate(grids)]
    names += [f"pos_centre{i}" for i in range(3)]
    if atlas is not None:
        p = atlas.prob(shape)
        feats += [p[0], p[1], p[2]]
        names += ["prior_bg", "prior_1", "prior_2"]
        for j, c in enumerate(atlas.centroid_):
            feats.append(np.sqrt(sum(((grids[i] + 0.5) - c[i] * shape[i]) ** 2 for i in range(3))))
            names.append(f"dist_centroid_{j + 1}")
    if fusion is not None:
        f = np.asarray(fusion, dtype=np.float32)
        fg = f[1] + f[2]
        feats += [f[1], f[2], ndimage.gaussian_filter(fg, 1.5), ndimage.gaussian_filter(fg, 3.0),
                  ndimage.gaussian_filter(f[1] / (fg + 1e-3), 1.5)]
        names += ["fus_1", "fus_2", "fus_fg_s1.5", "fus_fg_s3", "fus_ratio_s1.5"]
        m = fg > 0.5
        if m.any() and not m.all():
            sd = ndimage.distance_transform_edt(m) - ndimage.distance_transform_edt(~m)
        else:
            sd = np.full(shape, -10.0 if not m.any() else 10.0)
        feats.append(sd)
        names.append("fus_signed_dist")
    X = np.stack([np.asarray(f, dtype=np.float32).reshape(-1) for f in feats], axis=1)
    return X, names


# ----------------------------------------------------------------------------------------------------------------
# post-processing
# ----------------------------------------------------------------------------------------------------------------
def _largest_component(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    lab, n = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def postprocess(prob, smooth: float = 0.0, background_weight: float = 1.0) -> np.ndarray:
    """(3, X, Y, Z) class probabilities -> uint8 label volume. Optional Gaussian smoothing of each probability map
    (``smooth`` voxels) and multiplication of the background probability by ``background_weight``; then argmax, the largest
    connected foreground component with holes filled, and only the largest connected component of each label (other
    components take the other label)."""
    from scipy import ndimage

    prob = np.asarray(prob, dtype=np.float32)
    if smooth > 0:
        prob = np.stack([ndimage.gaussian_filter(p, smooth) for p in prob])
    if background_weight != 1.0:
        prob = prob.copy()
        prob[0] *= background_weight
    lab = np.argmax(prob, axis=0)
    fg = lab > 0
    if not fg.any():
        return np.zeros(lab.shape, dtype=np.uint8)
    fg = ndimage.binary_fill_holes(_largest_component(fg))
    lab = np.where(fg, np.argmax(prob[1:], axis=0) + 1, 0)
    for c in CLASSES:
        m = lab == c
        if m.any():
            lab[m & ~_largest_component(m)] = 3 - c
    return lab.astype(np.uint8)


# ----------------------------------------------------------------------------------------------------------------
# classifier
# ----------------------------------------------------------------------------------------------------------------
class HippocampusSegmenter:
    """Location atlas + patch-based label fusion + LightGBM voxel classifier (see the module description).

    Candidate voxels are those within ``margin`` voxels of the atlas support; everything else is background. Training uses
    a random ``fg_fraction`` of the foreground voxels and ``bg_fraction`` of the background candidates of every volume."""

    def __init__(self, n_estimators=100, learning_rate=0.12, num_leaves=31, min_child_samples=40, fg_fraction=0.5,
                 bg_fraction=0.15, margin=4, fusion_k=10, fusion_search=1, fusion_patch=3, fusion_max_shift=4,
                 smooth=0.7, background_weight=1.4, seed=0, n_jobs=2):
        self.n_estimators, self.learning_rate, self.num_leaves = n_estimators, learning_rate, num_leaves
        self.min_child_samples, self.fg_fraction, self.bg_fraction = min_child_samples, fg_fraction, bg_fraction
        self.margin, self.fusion_k, self.fusion_search = margin, fusion_k, fusion_search
        self.fusion_patch, self.fusion_max_shift = fusion_patch, fusion_max_shift
        self.smooth, self.background_weight, self.seed, self.n_jobs = smooth, background_weight, seed, n_jobs

    def _fusion(self, T, atlases, shape):
        return _fuse(T, atlases, self.atlas_.support(shape, self.margin), self.fusion_k, self.fusion_search,
                     self.fusion_patch, self.fusion_max_shift)

    def fit(self, images, labels, groups=None):
        import lightgbm as lgb

        images = [_as_volume(im) for im in images]
        labels = [np.asarray(lb).astype(np.uint8) for lb in labels]
        if len(images) != len(labels) or not images:
            raise ValueError("images and labels must be non-empty lists of the same length")
        for i, (im, lb) in enumerate(zip(images, labels)):
            if im.shape != lb.shape:
                raise ValueError(f"training volume {i}: image shape {im.shape} != label shape {lb.shape}")
            if lb.max() > 2:
                raise ValueError(f"training volume {i}: labels must be in {{0, 1, 2}}")
        if len(images) < 3:
            raise ValueError("at least 3 labelled training volumes are needed")
        self.atlas_ = LocationAtlas().fit(labels)
        self.atlases_ = [(normalize_intensity(im, "rank"), lb) for im, lb in zip(images, labels)]
        grp = list(range(len(images))) if groups is None else list(groups)
        rng = np.random.default_rng(self.seed)
        Xs, ys = [], []
        for i, (im, lb) in enumerate(zip(images, labels)):
            others = [self.atlases_[j] for j in range(len(images)) if grp[j] != grp[i]] or self.atlases_
            fus = self._fusion(self.atlases_[i][0], others, im.shape)
            X, self.names_ = voxel_features(im, self.atlas_, fus)
            y = lb.reshape(-1).astype(int)
            cand = self.atlas_.support(im.shape, self.margin).reshape(-1)
            keep = ((y > 0) & (rng.random(y.size) < self.fg_fraction)) | (cand & (y == 0) & (rng.random(y.size) < self.bg_fraction))
            Xs.append(X[keep])
            ys.append(y[keep])
        X, y = np.concatenate(Xs), np.concatenate(ys)
        self.model_ = lgb.LGBMClassifier(
            objective="multiclass", n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            num_leaves=self.num_leaves, min_child_samples=self.min_child_samples, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, max_bin=63, random_state=self.seed, n_jobs=self.n_jobs,
            deterministic=True, force_row_wise=True, verbose=-1)
        self.model_.fit(X, y)
        return self

    def predict_proba(self, img) -> np.ndarray:
        """(3, *shape) float32 class probabilities (background, anterior, posterior); background outside the candidates."""
        if not hasattr(self, "model_"):
            raise RuntimeError("call fit(images, labels) first")
        a = _as_volume(img)
        shape = a.shape
        fus = self._fusion(normalize_intensity(a, "rank"), self.atlases_, shape)
        X, _ = voxel_features(a, self.atlas_, fus)
        cand = self.atlas_.support(shape, self.margin).reshape(-1)
        prob = np.zeros((3, X.shape[0]), dtype=np.float32)
        prob[0] = 1.0
        if cand.any():
            prob[:, cand] = self.model_.predict_proba(X[cand]).T
        return prob.reshape(3, *shape)

    def predict(self, img) -> np.ndarray:
        """uint8 label volume (0/1/2) with the shape of ``img``."""
        return postprocess(self.predict_proba(img), self.smooth, self.background_weight)


def fit_predict(train_images, train_labels, eval_images, train_ids=None, seed: int = 0, **kw):
    """Fit ``HippocampusSegmenter(seed=seed, **kw)`` on the labelled training volumes and return one uint8 label volume
    (values 0, 1, 2; shape identical to the image) per evaluation image, in order. ``train_ids`` (optional) are case ids
    used to exclude same-subject volumes when the fusion features of the training volumes are built."""
    groups = subject_groups(train_ids) if train_ids is not None else None
    seg = HippocampusSegmenter(seed=seed, **kw).fit(train_images, train_labels, groups=groups)
    return [seg.predict(im) for im in eval_images]


def cross_validate(images, labels, case_ids=None, n_folds: int = 4, folds=None, seed: int = 0, **kw) -> dict:
    """Subject-grouped K-fold cross-validation of ``fit_predict`` on labelled volumes (groups = ``subject_groups(case_ids)``
    when ids are given, else consecutive blocks). ``folds``: fold indices to evaluate (default all). Returns mean_dsc,
    dsc_anterior, dsc_posterior (over the evaluated volumes), per_case (n, 2) list (NaN for volumes not evaluated), seconds."""
    t0 = time.time()
    n = len(images)
    if case_ids is not None:
        g = np.asarray(subject_groups(case_ids))
        ug = np.unique(g)
        np.random.default_rng(seed).shuffle(ug)
        fold_of = {u: i % n_folds for i, u in enumerate(ug)}
        fold = np.array([fold_of[u] for u in g])
    else:
        fold = np.arange(n) * n_folds // n
    per = np.full((n, 2), np.nan)
    for f in (range(n_folds) if folds is None else folds):
        te = np.flatnonzero(fold == f)
        tr = np.flatnonzero(fold != f)
        if len(te) == 0 or len(tr) == 0:
            continue
        preds = fit_predict([images[i] for i in tr], [labels[i] for i in tr], [images[i] for i in te],
                            train_ids=None if case_ids is None else [case_ids[i] for i in tr], seed=seed, **kw)
        for i, p in zip(te, preds):
            per[i] = case_dsc(p, labels[i])
    ok = ~np.isnan(per[:, 0])
    return {"mean_dsc": float(per[ok].mean()), "dsc_anterior": float(per[ok, 0].mean()),
            "dsc_posterior": float(per[ok, 1].mean()), "per_case": per.tolist(), "seconds": time.time() - t0}
