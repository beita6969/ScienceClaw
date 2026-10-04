"""Tools for stereo music source separation into vocals / drums / bass / other (BSSEval-v4 SDR, 1-second windows).

Array layout used everywhere: mixtures ``(k, n, 2)`` (item, sample, channel); stems and estimates ``(k, 4, n, 2)`` (item,
target, sample, channel) with the targets in the order ``TARGETS = (vocals, drums, bass, other)``; linear PCM amplitude,
``sample_rate`` Hz (default ``SR = 22050``). The stems of an excerpt sum approximately to its mixture.

sdr_scores(references, estimates, sample_rate) -> dict
    The task metric. Keys ``sdr`` (mean over the 4 targets of the median over items), ``target_median`` (4,), ``item``
    (k, 4). Per 1-s window (hop 1 s, the last partial window is dropped) SDR = 10 log10(sum ref^2 / sum (est - ref)^2) over
    both channels; a window is skipped (NaN) when ANY of the 4 references or ANY of the 4 estimates has channel-sum exactly
    0 at every sample of it; item score of a target = median over its valid windows (NaN if none, +inf for a perfect
    estimate); NaN item scores are ignored by the median over items; ``sdr`` is NaN if a target has no valid item. The
    estimate is compared with the reference sample by sample, so SDR depends on the amplitude scale of the estimate (a
    constant factor on an estimate changes its SDR; an estimate of (near) zero amplitude has SDR 0 dB in every window).
gain_only(train_mixtures, train_stems, mixtures=None, sample_rate) -> dict
    Constant gain per target: ``gains`` (4,) maximising the train SDR of ``gain * mixture``. Returns ``gains``,
    ``train_sdr`` (SDR of the scaled training mixtures against the training stems), ``mixture_sdr`` (same for the unscaled
    mixture), ``null_sdr`` (same for an estimate that holds only the 1e-7 square-wave offset, i.e. no signal) and, when
    ``mixtures`` is given, ``estimates`` (k, 4, n, 2) = gain_j * mixture (no separation, only a level change).
SoftMaskSeparator(power=1.5, n_estimators=60, learning_rate=0.2, weight_power=1.0, ...)
    STFT ratio-mask separator (2048 Hann window, hop 1024, mixture phase kept). ``.fit(train_mixtures, train_stems)``
    learns, per target, gradient-boosted trees that map 16 per-bin features of the MIXTURE (log frequency and bin, log
    magnitude of the mid channel and its deviation from the frame mean, harmonic/percussive log ratio of time- vs
    frequency-median-filtered magnitude, left/right pan, side/mid ratio, inter-channel cosine, three box-mean context ratios,
    frame loudness, four neighbour differences) to the ideal ratio mask of the training stems (target power / total power of
    the 4 stems, channel mean); a training bin is weighted by its mid-channel magnitude ** ``weight_power``
    (default 1, in RMS-normalised units). ``.separate(mixtures)`` predicts the 4 masks, clips them to [1e-4, 1], raises them to
    ``power``, renormalises them over the 4 targets, applies them to both channels and inverts the STFT; a 1e-7 square-wave
    offset keeps every 1-s window non-zero. Mixtures are divided by their RMS before the STFT and the result is multiplied
    back. Trains on ``rows_per_excerpt`` random time-frequency bins per training excerpt; deterministic (``seed``).
separate(train_mixtures, train_stems, mixtures, **kwargs) -> float32 estimates
    ``SoftMaskSeparator(**kwargs).fit(...).separate(mixtures)``; ``mixtures`` may also be a list of arrays (e.g.
    ``[dev_mixtures, eval_mixtures]``), in which case a list of estimate arrays is returned. Shapes are validated before
    fitting.
cross_validate(train_mixtures, train_stems, groups=None, n_folds=4, **kwargs) -> dict
    Grouped k-fold over the training excerpts (``groups``: one label per excerpt, e.g. the track; the excerpts of one group
    are never split between fit and test; default: every excerpt its own group). Per fold the separator is fitted on the
    other groups and evaluated on the held-out excerpts; the constant-gain estimate (gains fitted on the same other groups),
    the unscaled mixture and the no-signal estimate are scored on the same held-out excerpts. Returns ``sdr``,
    ``target_median``, ``item`` for the separator, ``gain_only_sdr`` / ``gain_only_target_median``, ``mixture_sdr`` /
    ``mixture_target_median``, ``null_sdr`` / ``null_target_median``, ``folds``.
oracle_mask_sdr(mixtures, stems, power=1.0)
    SDR of the ideal ratio masks computed from the true stems (same STFT, mixture phase): the highest SDR a mask on this
    STFT can reach on these excerpts.
stft(x) / istft(Z, n)
    The transform used above: (n, 2) <-> (2, 1025, frames) complex.

Example (``track_ids``: one label per training excerpt, e.g. the ``track_ids`` output of ``load_train``):
    cv = cross_validate(train_mixtures, train_stems, groups=track_ids)
    dev_est, eval_est = separate(train_mixtures, train_stems, [dev_mixtures, eval_mixtures])   # float32, (n_dev|n_eval, 4, n, 2)

Cost on CPU (2 threads): ``separate`` fits in about 5 s and needs about 2 s per 6-s stereo excerpt; ``cross_validate`` with 8
excerpts and 4 folds takes about a minute. All functions are deterministic for fixed arguments and validate shapes.
"""
from __future__ import annotations

import numpy as np

__all__ = ["SR", "TARGETS", "sdr_scores", "gain_only", "SoftMaskSeparator", "separate", "cross_validate",
           "oracle_mask_sdr", "stft", "istft"]

SR = 22_050
TARGETS = ("vocals", "drums", "bass", "other")
NPERSEG, HOP = 2048, 1024
_EPS = 1e-8
_LOGMAG = 2                                                                    # column of ``_features``: log |mid|


# ================================================================================================ validation
def _check_mix(x, name: str) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim == 4 and a.shape[1] == 4:
        raise ValueError(f"{name} has shape {a.shape}, which looks like stems (k, 4, n, 2); {name} must be mixtures of "
                         "shape (k, n, 2)")
    if a.ndim != 3 or a.shape[2] != 2:
        raise ValueError(f"{name} must have shape (k, n, 2) (item, sample, stereo channel), got {a.shape}")
    if a.shape[1] < NPERSEG:
        raise ValueError(f"{name}: {a.shape[1]} samples per item, at least {NPERSEG} needed")
    if not np.issubdtype(a.dtype, np.number) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite numeric array")
    return a


def _check_stems(x, name: str) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != 4 or a.shape[1] != 4 or a.shape[3] != 2:
        raise ValueError(f"{name} must have shape (k, 4, n, 2) with targets in the order {TARGETS}, got {a.shape}")
    if not np.issubdtype(a.dtype, np.number) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite numeric array")
    return a


def _check_train(train_mixtures, train_stems) -> tuple[np.ndarray, np.ndarray]:
    mix = _check_mix(train_mixtures, "train_mixtures")
    stems = _check_stems(train_stems, "train_stems")
    if stems.shape[0] != mix.shape[0] or stems.shape[2] != mix.shape[1]:
        raise ValueError(f"train_stems {stems.shape} does not match train_mixtures {mix.shape}: expected "
                         f"({mix.shape[0]}, 4, {mix.shape[1]}, 2)")
    if mix.shape[0] < 2:
        raise ValueError("at least 2 training excerpts are needed")
    return mix, stems


# ================================================================================================ metric
def _window_stats(x: np.ndarray, win: int) -> np.ndarray:
    """(S, n, C) -> (S, nwin, win, C) float64 view of the whole 1-s windows (hop = win)."""
    nw = x.shape[1] // win
    return np.asarray(x[:, : nw * win], dtype=np.float64).reshape(x.shape[0], nw, win, x.shape[2])


def _silent(x: np.ndarray) -> np.ndarray:
    """(S, nwin, win, C) -> (S, nwin): channel-sum exactly 0 at every sample of the window."""
    return np.all(x.sum(axis=3) == 0, axis=2)


def _item_sdr(ref: np.ndarray, est: np.ndarray, win: int) -> np.ndarray:
    """(S, n, C) references and estimates -> (S,) median over valid 1-s windows of SDR (NaN if no valid window)."""
    r, e = _window_stats(ref, win), _window_stats(est, win)
    out = np.full(r.shape[0], np.nan)
    if r.shape[1] == 0:
        return out
    skip = np.any(_silent(r), axis=0) | np.any(_silent(e), axis=0)              # (nwin,)
    num = np.sum(r ** 2, axis=(2, 3))
    den = np.sum((e - r) ** 2, axis=(2, 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(den == 0, np.inf, 10.0 * np.log10(num / np.where(den == 0, 1.0, den)))
    for j in range(r.shape[0]):
        v = f[j][~skip]
        v = v[~np.isnan(v)]
        if v.size:
            out[j] = float(np.median(v))
    return out


def _aggregate(item: np.ndarray) -> tuple[float, np.ndarray]:
    """(k, S) item scores -> (mean over targets of the median over items, per-target medians); NaN if a target has no
    valid item."""
    med = np.full(item.shape[1], np.nan)
    for j in range(item.shape[1]):
        v = item[:, j][~np.isnan(item[:, j])]
        if v.size:
            med[j] = float(np.median(v))
    return (float(np.mean(med)) if not np.any(np.isnan(med)) else float("nan")), med


def sdr_scores(references, estimates, sample_rate: int = SR) -> dict:
    """The task metric on any references/estimates of shape (k, 4, n, 2).

    Returns ``{"sdr": float, "target_median": (4,) array, "item": (k, 4) array}`` (definition in the module docstring)."""
    ref = _check_stems(references, "references")
    est = _check_stems(estimates, "estimates")
    if ref.shape != est.shape:
        raise ValueError(f"references {ref.shape} and estimates {est.shape} must have the same shape")
    win = int(sample_rate)
    item = np.stack([_item_sdr(ref[i], est[i], win) for i in range(ref.shape[0])]) if ref.shape[0] else np.zeros((0, 4))
    sdr, med = _aggregate(item)
    return {"sdr": sdr, "target_median": med, "item": item}


# ================================================================================================ constant gain
_GAIN_GRID = np.geomspace(0.01, 3.0, 400)


def _gain_stats(mix: np.ndarray, stems: np.ndarray, win: int):
    """Per (item, target, window): a = sum ref^2, b = sum mix^2, c = sum mix*ref, and the validity of every window."""
    A, B, C, V = [], [], [], []
    for i in range(mix.shape[0]):
        r = _window_stats(stems[i], win)                                        # (4, nw, win, 2)
        m = _window_stats(mix[i][None], win)[0]                                 # (nw, win, 2)
        A.append(np.sum(r ** 2, axis=(2, 3)))
        B.append(np.sum(m ** 2, axis=(1, 2)))
        C.append(np.einsum("wsc,jwsc->jw", m, r))
        V.append(~(np.any(_silent(r), axis=0) | _silent(m[None])[0]))
    return A, B, C, V


def _fit_gains(mix: np.ndarray, stems: np.ndarray, win: int) -> np.ndarray:
    """Per target: the gain on a log grid maximising the median over items of the median over windows of the SDR of
    ``gain * mixture`` (ties resolved to the middle of the maximising plateau, in log gain)."""
    A, B, C, V = _gain_stats(mix, stems, win)
    gains = np.ones(4)
    for j in range(4):
        obj = np.full(_GAIN_GRID.size, -np.inf)
        rows = []
        for a, b, c, v in zip(A, B, C, V):
            if v.any():
                aw, bw, cw = a[j][v], b[v], c[j][v]
                den = _GAIN_GRID[:, None] ** 2 * bw[None] - 2 * _GAIN_GRID[:, None] * cw[None] + aw[None]
                with np.errstate(divide="ignore", invalid="ignore"):
                    sdr = 10 * np.log10(aw[None] / np.maximum(den, 1e-30))
                sdr = np.where(np.isnan(sdr), -np.inf, sdr)
                rows.append(np.median(sdr, axis=1))
        if rows:
            obj = np.median(np.stack(rows), axis=0)
            best = np.flatnonzero(obj >= obj.max() - 1e-9)
            gains[j] = float(np.exp(np.mean(np.log(_GAIN_GRID[best]))))
    return gains


def _square(n: int) -> np.ndarray:
    """(1, n, 1) float32 square wave of +-1 (the offset pattern, scaled by ``dither``, added to every estimate)."""
    return np.sign(np.sin(np.arange(n, dtype=np.float64))).astype(np.float32)[None, :, None]


def _null_estimates(shape, dither: float = 1e-7) -> np.ndarray:
    """(k, 4, n, 2) estimates that hold nothing but the square-wave offset."""
    return np.broadcast_to(dither * _square(shape[2]), shape).astype(np.float32)


def gain_only(train_mixtures, train_stems, mixtures=None, sample_rate: int = SR) -> dict:
    """Constant gain per target, fitted on the training excerpts (see the module docstring for the returned keys)."""
    mix, stems = _check_train(train_mixtures, train_stems)
    g = _fit_gains(mix, stems, int(sample_rate))
    scaled = g[None, :, None, None] * mix[:, None].astype(np.float64)
    out = {"gains": g,
           "train_sdr": sdr_scores(stems, scaled, sample_rate)["sdr"],
           "mixture_sdr": sdr_scores(stems, np.repeat(mix[:, None], 4, axis=1), sample_rate)["sdr"],
           "null_sdr": sdr_scores(stems, _null_estimates(stems.shape), sample_rate)["sdr"]}
    if mixtures is not None:
        m = _check_mix(mixtures, "mixtures")
        out["estimates"] = (g[None, :, None, None] * m[:, None].astype(np.float64)).astype(np.float32)
    return out


# ================================================================================================ STFT + features
def stft(x) -> np.ndarray:
    """(n, 2) signal -> (2, 1025, frames) complex STFT (Hann, 2048 window, hop 1024, zero-padded borders)."""
    from scipy.signal import stft as _stft
    return _stft(np.asarray(x, dtype=np.float64).T, fs=SR, nperseg=NPERSEG, noverlap=NPERSEG - HOP,
                 boundary="zeros", padded=True)[2]


def istft(Z, n: int) -> np.ndarray:
    """(2, 1025, frames) complex STFT -> (n, 2) float32 signal."""
    from scipy.signal import istft as _istft
    x = _istft(Z, fs=SR, nperseg=NPERSEG, noverlap=NPERSEG - HOP)[1][:, :n]
    return np.pad(x, ((0, 0), (0, max(0, n - x.shape[1])))).T.astype(np.float32)


def _features(Z: np.ndarray) -> np.ndarray:
    """(F * T, 16) float32 per-bin features of one stereo STFT (rows in (frequency, frame) C order)."""
    from scipy.ndimage import median_filter, uniform_filter
    L, R = Z[0], Z[1]
    aL, aR, aM, aS = np.abs(L), np.abs(R), np.abs(0.5 * (L + R)), np.abs(0.5 * (L - R))
    F, T = aM.shape
    lm = np.log(aM + _EPS)
    frame = np.repeat(lm.mean(0, keepdims=True), F, 0)
    hp = np.log(median_filter(aM, size=(1, 9)) + _EPS) - np.log(median_filter(aM, size=(9, 1)) + _EPS)
    ctx = lambda s: np.log(uniform_filter(aM, size=s) + _EPS) - lm             # noqa: E731
    fl = np.repeat(np.arange(F)[:, None], T, 1)
    cols = [np.log1p(fl), fl, lm, lm - frame, hp, (aL - aR) / (aL + aR + _EPS), np.log(aS + _EPS) - lm,
            np.real(L * np.conj(R)) / (aL * aR + _EPS), ctx((1, 9)), ctx((9, 1)), ctx((33, 1)), frame,
            np.roll(lm, 1, 0) - lm, np.roll(lm, -1, 0) - lm, np.roll(lm, 1, 1) - lm, np.roll(lm, -1, 1) - lm]
    return np.stack([c.reshape(-1) for c in cols], 1).astype(np.float32)


def _ratio_masks(stems: np.ndarray) -> np.ndarray:
    """(4, n, 2) stems -> ideal ratio masks (4, F, T): channel-mean power of each target / total power."""
    P = np.stack([np.mean(np.abs(stft(s)) ** 2, axis=0) for s in stems])
    return P / (P.sum(0, keepdims=True) + 1e-12)


def _rms(x: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))
    return r if r > 1e-9 else 1.0


def _regressor(n_estimators: int, learning_rate: float, num_leaves: int, min_samples_leaf: int, n_jobs: int, seed: int):
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
                             min_child_samples=min_samples_leaf, n_jobs=n_jobs, random_state=seed, verbosity=-1,
                             deterministic=True, force_row_wise=True)
    except ImportError:                                                         # pragma: no cover
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=n_estimators, learning_rate=learning_rate, max_leaf_nodes=num_leaves,
                                             min_samples_leaf=min_samples_leaf, random_state=seed)


class SoftMaskSeparator:
    """STFT ratio-mask separator with gradient-boosted trees (module docstring). Parameters: ``power`` of the masks before
    renormalisation, ``n_estimators`` / ``learning_rate`` / ``num_leaves`` / ``min_samples_leaf`` of the trees,
    ``rows_per_excerpt`` random time-frequency bins per training excerpt, ``weight_power`` (each training bin is weighted by
    its mid-channel magnitude ** weight_power), ``n_jobs`` threads, ``seed``, ``dither`` amplitude of the square-wave offset
    added to every estimate."""

    def __init__(self, power: float = 1.5, n_estimators: int = 60, learning_rate: float = 0.2, num_leaves: int = 31,
                 min_samples_leaf: int = 50, rows_per_excerpt: int = 24_000, weight_power: float = 1.0,
                 n_jobs: int = 2, seed: int = 0, dither: float = 1e-7) -> None:
        if power <= 0:
            raise ValueError("power must be positive")
        if n_estimators < 1 or learning_rate <= 0 or num_leaves < 2 or rows_per_excerpt < 1:
            raise ValueError("n_estimators, rows_per_excerpt >= 1, learning_rate > 0 and num_leaves >= 2 are required")
        self.power, self.n_estimators, self.num_leaves = float(power), int(n_estimators), int(num_leaves)
        self.learning_rate, self.weight_power = float(learning_rate), float(weight_power)
        self.min_samples_leaf, self.rows_per_excerpt = int(min_samples_leaf), int(rows_per_excerpt)
        self.n_jobs, self.seed, self.dither = int(n_jobs), int(seed), float(dither)
        self.models_: list | None = None

    def fit(self, train_mixtures, train_stems) -> "SoftMaskSeparator":
        mix, stems = _check_train(train_mixtures, train_stems)
        rng = np.random.default_rng(self.seed)
        X, Y = [], []
        for x, s in zip(mix, stems):
            g = _rms(x)
            f = _features(stft(x / g))
            m = _ratio_masks(s / g).reshape(4, -1).T
            w = rng.choice(len(f), min(self.rows_per_excerpt, len(f)), replace=False)
            X.append(f[w])
            Y.append(m[w])
        X, Y = np.vstack(X), np.vstack(Y)
        sw = np.exp(self.weight_power * X[:, _LOGMAG]) if self.weight_power else None    # weight ~ magnitude ** weight_power
        if sw is not None:
            sw = sw / sw.mean()
        self.models_ = [_regressor(self.n_estimators, self.learning_rate, self.num_leaves, self.min_samples_leaf,
                                   self.n_jobs, self.seed).fit(X, Y[:, j], sample_weight=sw) for j in range(4)]
        return self

    def masks(self, mixture) -> np.ndarray:
        """(n, 2) mixture -> the 4 normalised masks (4, F, T) used by ``separate``."""
        if self.models_ is None:
            raise RuntimeError("call fit(train_mixtures, train_stems) first")
        Z = stft(np.asarray(mixture, dtype=np.float64) / _rms(mixture))
        F, T = Z.shape[1:]
        f = _features(Z)
        m = np.stack([np.clip(md.predict(f), 1e-4, 1.0).reshape(F, T) for md in self.models_]) ** self.power
        return m / m.sum(0, keepdims=True)

    def separate(self, mixtures) -> np.ndarray:
        """(k, n, 2) mixtures -> (k, 4, n, 2) float32 estimates in the order vocals, drums, bass, other."""
        mix = _check_mix(mixtures, "mixtures")
        n = mix.shape[1]
        sq = _square(n)
        out = np.empty((mix.shape[0], 4, n, 2), dtype=np.float32)
        for i, x in enumerate(mix):
            g = _rms(x)
            Z = stft(x / g)
            m = self.masks(x)
            for j in range(4):
                out[i, j] = istft(Z * m[j][None], n) * g
            out[i] += self.dither * sq
        return out


def separate(train_mixtures, train_stems, mixtures, **kwargs):
    """Fit a ``SoftMaskSeparator(**kwargs)`` on the training excerpts and separate ``mixtures`` (an array (k, n, 2), or a
    list/tuple of such arrays -> list of estimate arrays). Returns float32 estimates (k, 4, n, 2)."""
    many = isinstance(mixtures, (list, tuple))
    for m in (mixtures if many else [mixtures]):                                # fail before the (slow) fit
        _check_mix(m, "mixtures")
    sep = SoftMaskSeparator(**kwargs).fit(train_mixtures, train_stems)
    return [sep.separate(m) for m in mixtures] if many else sep.separate(mixtures)


# ================================================================================================ validation / diagnostics
def _folds(groups, m: int, n_folds: int) -> list[np.ndarray]:
    """Held-out index sets: groups (sorted by label) dealt round-robin to ``n_folds`` folds; a group never straddles."""
    labels = np.asarray([str(i) for i in range(m)] if groups is None else [str(g) for g in groups])
    if labels.shape != (m,):
        raise ValueError(f"groups must have one label per training excerpt ({m}), got shape {labels.shape}")
    uniq = sorted(set(labels.tolist()))
    if len(uniq) < 2:
        raise ValueError("at least 2 distinct groups are needed for cross-validation")
    k = max(2, min(int(n_folds), len(uniq)))
    fold_of = {u: i % k for i, u in enumerate(uniq)}
    return [np.flatnonzero([fold_of[l] == f for l in labels]) for f in range(k)]


def cross_validate(train_mixtures, train_stems, groups=None, n_folds: int = 4, sample_rate: int = SR, **kwargs) -> dict:
    """Grouped k-fold estimate of the separator, the constant-gain estimate and the unscaled mixture (docstring)."""
    mix, stems = _check_train(train_mixtures, train_stems)
    est = np.zeros(stems.shape, dtype=np.float32)
    gest = np.zeros(stems.shape, dtype=np.float32)
    held = _folds(groups, mix.shape[0], n_folds)
    for te in held:
        tr = np.setdiff1d(np.arange(mix.shape[0]), te)
        sep = SoftMaskSeparator(**kwargs).fit(mix[tr], stems[tr])
        est[te] = sep.separate(mix[te])
        g = _fit_gains(mix[tr], stems[tr], int(sample_rate))
        gest[te] = (g[None, :, None, None] * mix[te][:, None].astype(np.float64)).astype(np.float32)
    raw = np.repeat(mix[:, None], 4, axis=1)
    a, b, c, d = (sdr_scores(stems, e, sample_rate) for e in (est, gest, raw, _null_estimates(stems.shape)))
    return {"sdr": a["sdr"], "target_median": a["target_median"], "item": a["item"],
            "gain_only_sdr": b["sdr"], "gain_only_target_median": b["target_median"],
            "mixture_sdr": c["sdr"], "mixture_target_median": c["target_median"],
            "null_sdr": d["sdr"], "null_target_median": d["target_median"], "folds": len(held)}


def oracle_mask_sdr(mixtures, stems, power: float = 1.0, sample_rate: int = SR) -> float:
    """SDR of ratio masks computed from the true stems (mixture phase, ``power`` as in the separator)."""
    mix, stems = _check_train(mixtures, stems)
    n = mix.shape[1]
    est = np.empty(stems.shape, dtype=np.float32)
    for i in range(mix.shape[0]):
        g = _rms(mix[i])
        m = _ratio_masks(stems[i] / g) ** power
        m /= m.sum(0, keepdims=True) + 1e-12
        Z = stft(mix[i] / g)
        for j in range(4):
            est[i, j] = istft(Z * m[j][None], n) * g
        est[i] += 1e-7 * _square(n)
    return sdr_scores(stems, est, sample_rate)["sdr"]
