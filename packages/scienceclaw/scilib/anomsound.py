"""Tools for anomalous sound detection on ONE machine type: normal training clips + unlabeled test clips.

Conventions. Waveform rows returned by load_train / load_eval_inputs are zero-padded to the longest clip; ``lengths``
holds each clip's true number of samples. A *mel list* is a list of (frames, n_mels) float arrays in dB, one per clip
(the arrays may differ in frame count). Anomaly scores: larger = more anomalous; only their ordering matters.

score_clips(train, evalset, members=None, embed=None, embed_members=("nn2_pool",)) -> (n_eval,) float
                                                                 one call from the load_train / load_eval_inputs outputs
                                                                 (dicts with waveforms, lengths, sample_rate)
    ``embed`` (default None): a model name of ``scilib.audioenc`` ("ast_audioset", "clap_htsat") or a list of them (needs a GPU
    worker). The score is then the rank average (as ``rank_average``) of the names "<model>:<component>" for every model in ``embed``
    and every component in ``embed_members`` (names of EMBEDDING_COMPONENTS; computed by ``embedding_scores`` from the "pooled"
    ``audioenc.embed`` vectors of the train and eval clips), plus the ``members`` of this module when ``members`` is given. With
    ``embed`` and ``members=None`` no log-mel member is used. ``embed=None`` is the call described before.
    Example: ``score_clips(train, evalset, embed=["ast_audioset", "clap_htsat"])``.
fit_predict(train_mels, eval_mels, members=None) -> (n_eval,)    the same from two mel lists
log_mel_list(waveforms, lengths, sample_rate=16000, n_fft=1024, hop=512, n_mels=128) -> mel list
                                                                 log-mel (Hann, centred frames, dB) of every row cut to its length
trim_mels(log_mel, lengths, hop=512) -> mel list                 cuts a (n, frames, n_mels) array computed from padded rows to
                                                                 each clip's frame count 1 + length // hop
clip_descriptors(mels, kind) -> (n, d)                           kind 'ms': per-mel mean and std over frames (2 * n_mels);
                                                                 'm': mean; 'mx': maximum over frames; 'q95': 95th percentile
component_scores(train_mels, eval_mels, n_bands=8, lof_neighbors=5) -> {name: (n_eval,)}   raw anomaly scores, keys COMPONENTS:
    nn_train : distance of the 'ms' descriptor (z-scored with the training mean/std) to the nearest training clip
    nn2_pool : distance to the second-nearest clip of the pool {training clips} + {the other eval clips}
    nn_pool  : min(nn_train, distance to the nearest other eval clip)
    lof      : Local Outlier Factor of the eval clips inside that pool ('m' descriptor, z-scored)
    band_max : 'mx' descriptor z-scored, mel axis cut into n_bands equal bands, nearest training clip searched per band,
               mean of the band distances
    maha     : Mahalanobis distance of the 'm' descriptor to the training mean, Ledoit-Wolf shrunk covariance of the
               training clips
    nn2_pool, nn_pool and lof place the unlabeled eval clips in the point cloud (no labels are involved); nn_train,
    band_max and maha use the training clips only.
embedding_scores(train_emb, eval_emb, lof_neighbors=5) -> {name: (n_eval,)}   raw anomaly scores from per-clip embedding vectors (rows
                                                                 of any encoder, e.g. ``scilib.audioenc.embed``); rows are L2-normalised, keys
                                                                 EMBEDDING_COMPONENTS:
    nn_train : Euclidean distance to the nearest training row
    nn2_pool : distance to the second-nearest row of the pool {training rows} + {the other eval rows}
    nn_pool  : min(nn_train, distance to the nearest other eval row)
    lof      : Local Outlier Factor of the eval rows inside that pool (negated, larger = more anomalous)
rank_average(scores, members) -> (n_eval,) in (0, 1]              mean over members of rank / n (ties averaged)
DEFAULT_MEMBERS                                                   members used when ``members`` is None
probe_report(train_mels, eval_mels) -> dict                       Spearman matrix between the components and, per eval clip,
                                                                 whether its nearest neighbour in the 'ms' pool is a training
                                                                 clip or another eval clip (diagnostic, not a score)
official_score(y_true, domain, score) -> dict                     task metric for arrays with known labels (1 = anomaly) and
                                                                 domains ('source' | 'target'): auc_source, auc_target, pauc,
                                                                 official_score = harmonic mean of the three

Every function is deterministic and needs well under a second per episode on CPU. Trailing frames of a mel array that
sit at the -100 dB floor (zero padding) are dropped; non-finite values raise ValueError.
"""
from __future__ import annotations

import numpy as np

__all__ = ["DEFAULT_MEMBERS", "COMPONENTS", "score_clips", "fit_predict", "log_mel_list", "trim_mels", "clip_descriptors",
           "component_scores", "embedding_scores", "EMBEDDING_COMPONENTS", "rank_average", "probe_report", "official_score"]

COMPONENTS = ("nn_train", "nn2_pool", "nn_pool", "lof", "band_max", "maha")
EMBEDDING_COMPONENTS = ("nn_train", "nn2_pool", "nn_pool", "lof")
DEFAULT_MEMBERS = ("nn2_pool", "lof", "nn_pool", "band_max", "maha")
_FLOOR_DB = -100.0            # 10 * log10(1e-10): value of a mel bin without energy


# ----------------------------------------------------------------------------------------------- log-mel front end
def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)   # noqa: E731
    mel2hz = lambda m: 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)  # noqa: E731
    pts = mel2hz(np.linspace(hz2mel(0.0), hz2mel(sr / 2.0), n_mels + 2))
    freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    fb = np.zeros((n_mels, freqs.size))
    for i in range(n_mels):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        up = (freqs - lo) / max(c - lo, 1e-9)
        down = (hi - freqs) / max(hi - c, 1e-9)
        fb[i] = np.maximum(0.0, np.minimum(up, down)) * (2.0 / max(hi - lo, 1e-9))
    return fb


def log_mel_list(waveforms, lengths, sample_rate: float = 16000.0, n_fft: int = 1024, hop: int = 512,
                 n_mels: int = 128) -> list:
    """Log-mel power spectrogram in dB (frames, n_mels) of every waveform row, cut to its true length first."""
    W = np.asarray(waveforms, dtype=np.float64)
    if W.ndim == 1:
        W = W[None]
    if W.ndim != 2:
        raise ValueError("waveforms must be a 2-D array (n_clips, n_samples)")
    L = np.asarray(lengths, dtype=np.int64).reshape(-1)
    if L.size != W.shape[0]:
        raise ValueError(f"lengths has {L.size} entries for {W.shape[0]} waveform rows")
    if not (64 <= n_fft <= 8192 and 16 <= hop <= n_fft and 8 <= n_mels <= 256):
        raise ValueError("need 64 <= n_fft <= 8192, 16 <= hop <= n_fft, 8 <= n_mels <= 256")
    sr = int(round(float(sample_rate)))
    fb, win, pad = _mel_filterbank(sr, n_fft, n_mels), np.hanning(n_fft), n_fft // 2
    out = []
    for w, n in zip(W, L):
        x = w[: max(int(n), 1)]
        x = np.pad(x, pad, mode="reflect") if x.size > pad else np.pad(x, pad)
        n_frames = 1 + (x.size - n_fft) // hop
        idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
        spec = np.abs(np.fft.rfft(x[idx] * win[None, :], axis=1)) ** 2
        out.append((10.0 * np.log10(np.maximum(spec @ fb.T, 1e-10))).astype(np.float32))
    return out


def trim_mels(log_mel, lengths, hop: int = 512) -> list:
    """Mel list from a (n, frames, n_mels) array made of zero-padded rows: clip i keeps its first 1 + lengths[i] // hop frames."""
    A = np.asarray(log_mel)
    if A.ndim != 3:
        raise ValueError("log_mel must be a 3-D array (n_clips, frames, n_mels)")
    L = np.asarray(lengths, dtype=np.int64).reshape(-1)
    if L.size != A.shape[0]:
        raise ValueError(f"lengths has {L.size} entries for {A.shape[0]} clips")
    return [A[i, : min(A.shape[1], 1 + int(n) // int(hop))] for i, n in enumerate(L)]


def _mel_list(mels, name: str) -> list:
    """Validated list of float64 (frames, n_mels) arrays; zero-padding frames at the end are dropped."""
    if isinstance(mels, np.ndarray) and mels.ndim == 2:
        raise ValueError(f"{name}: expected one (frames, n_mels) array per clip (a list or a 3-D array), got a 2-D array")
    items = list(mels)
    if not items:
        raise ValueError(f"{name}: no clips")
    out = []
    for i, S in enumerate(items):
        S = np.asarray(S, dtype=np.float64)
        if S.ndim != 2 or S.shape[0] < 1 or S.shape[1] < 1:
            raise ValueError(f"{name}[{i}]: expected a (frames, n_mels) array, got shape {S.shape}")
        if not np.all(np.isfinite(S)):
            raise ValueError(f"{name}[{i}]: non-finite values in the log-mel array")
        live = np.flatnonzero(S.max(axis=1) > _FLOOR_DB + 1e-3)
        out.append(S[: int(live[-1]) + 1] if live.size else S)
    widths = {S.shape[1] for S in out}
    if len(widths) != 1:
        raise ValueError(f"{name}: clips have different numbers of mel bins {sorted(widths)}")
    return out


# ----------------------------------------------------------------------------------------------- descriptors / distances
_DESC = {
    "ms": lambda S: np.concatenate([S.mean(axis=0), S.std(axis=0)]),
    "m": lambda S: S.mean(axis=0),
    "mx": lambda S: S.max(axis=0),
    "q95": lambda S: np.percentile(S, 95, axis=0),
}


def clip_descriptors(mels, kind: str = "ms") -> np.ndarray:
    """(n_clips, d) per-clip descriptor of a mel list: 'ms' mean and std over frames, 'm' mean, 'mx' max, 'q95' 95th percentile."""
    if kind not in _DESC:
        raise ValueError(f"unknown descriptor {kind!r}; choose from {sorted(_DESC)}")
    return np.array([_DESC[kind](S) for S in _mel_list(mels, "mels")])


def _zscore(Ft: np.ndarray, Fe: np.ndarray):
    mu, sd = Ft.mean(axis=0), Ft.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (Ft - mu) / sd, (Fe - mu) / sd


def _dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T, 0.0))


def _lof_factor(pool: np.ndarray, lof_neighbors: int) -> np.ndarray:
    """``negative_outlier_factor_`` of a Local Outlier Factor fitted on the rows of ``pool``. Inside a sandboxed code node the
    default neighbour search is refused (its thread-pool control reads /proc/self/maps); the tree search finds the same neighbours."""
    from sklearn.neighbors import LocalOutlierFactor
    k = int(max(1, min(lof_neighbors, len(pool) - 1)))
    try:
        return LocalOutlierFactor(n_neighbors=k).fit(pool).negative_outlier_factor_
    except PermissionError:
        return LocalOutlierFactor(n_neighbors=k, algorithm="kd_tree").fit(pool).negative_outlier_factor_


def _finite(v: np.ndarray) -> np.ndarray:
    """Replace non-finite scores by the largest finite one (or 0), so ranks and metrics stay defined."""
    v = np.asarray(v, dtype=float)
    ok = np.isfinite(v)
    if ok.all():
        return v
    return np.where(ok, v, v[ok].max() if ok.any() else 0.0)


# ----------------------------------------------------------------------------------------------- component scores
def component_scores(train_mels, eval_mels, n_bands: int = 8, lof_neighbors: int = 5) -> dict:
    """Raw anomaly scores of the eval clips, one (n_eval,) array per name in ``COMPONENTS`` (see the module docstring)."""
    from sklearn.covariance import LedoitWolf
    tr, ev = _mel_list(train_mels, "train_mels"), _mel_list(eval_mels, "eval_mels")
    if tr[0].shape[1] != ev[0].shape[1]:
        raise ValueError(f"train and eval clips have different numbers of mel bins ({tr[0].shape[1]} vs {ev[0].shape[1]})")
    n_t, n_e = len(tr), len(ev)
    if n_t < 3:
        raise ValueError(f"need at least 3 training clips, got {n_t}")
    idx = np.arange(n_e)
    # 'ms' cloud: nearest neighbours
    Zt, Ze = _zscore(clip_descriptors(tr, "ms"), clip_descriptors(ev, "ms"))
    nn_train = _dist(Ze, Zt).min(axis=1)
    d_all = _dist(Ze, np.vstack([Zt, Ze]))
    d_all[idx, n_t + idx] = np.inf                                  # an eval clip is not its own neighbour
    nn2_pool = np.sort(d_all, axis=1)[:, min(1, d_all.shape[1] - 1)]
    d_ee = _dist(Ze, Ze)
    np.fill_diagonal(d_ee, np.inf)
    nn_pool = np.minimum(nn_train, d_ee.min(axis=1)) if n_e > 1 else nn_train.copy()
    # 'm' descriptor: LOF in the pool, Mahalanobis to the training mean
    Mt, Me = _zscore(clip_descriptors(tr, "m"), clip_descriptors(ev, "m"))
    pool = np.vstack([Mt, Me])
    lof = -_lof_factor(pool, lof_neighbors)[n_t:]
    prec = LedoitWolf().fit(Mt).precision_
    dm = Me - Mt.mean(axis=0)
    maha = np.einsum("ij,jk,ik->i", dm, prec, dm)
    # 'mx' descriptor: per-band nearest training clip
    Xt, Xe = _zscore(clip_descriptors(tr, "mx"), clip_descriptors(ev, "mx"))
    nb = int(max(1, min(n_bands, Xt.shape[1])))
    edges = np.linspace(0, Xt.shape[1], nb + 1).astype(int)
    band_max = np.mean([_dist(Xe[:, a:b], Xt[:, a:b]).min(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=0)
    return {k: _finite(v) for k, v in {"nn_train": nn_train, "nn2_pool": nn2_pool, "nn_pool": nn_pool, "lof": lof,
                                       "band_max": band_max, "maha": maha}.items()}


def _unit_rows(X, name: str) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] < 1:
        raise ValueError(f"{name}: expected a (n_clips, d) array, got shape {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name}: non-finite values")
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(nrm < 1e-12, 1.0, nrm)


def embedding_scores(train_emb, eval_emb, lof_neighbors: int = 5) -> dict:
    """Raw anomaly scores of the eval rows from embedding vectors, one (n_eval,) array per name in ``EMBEDDING_COMPONENTS``."""
    Et, Ee = _unit_rows(train_emb, "train_emb"), _unit_rows(eval_emb, "eval_emb")
    if Et.shape[1] != Ee.shape[1]:
        raise ValueError(f"train and eval embeddings have different dimensions ({Et.shape[1]} vs {Ee.shape[1]})")
    n_t, n_e = len(Et), len(Ee)
    if n_t < 3:
        raise ValueError(f"need at least 3 training clips, got {n_t}")
    idx = np.arange(n_e)
    nn_train = _dist(Ee, Et).min(axis=1)
    d_all = _dist(Ee, np.vstack([Et, Ee]))
    d_all[idx, n_t + idx] = np.inf                                  # an eval clip is not its own neighbour
    nn2_pool = np.sort(d_all, axis=1)[:, min(1, d_all.shape[1] - 1)]
    d_ee = _dist(Ee, Ee)
    np.fill_diagonal(d_ee, np.inf)
    nn_pool = np.minimum(nn_train, d_ee.min(axis=1)) if n_e > 1 else nn_train.copy()
    pool = np.vstack([Et, Ee])
    lof = -_lof_factor(pool, lof_neighbors)[n_t:]
    return {k: _finite(v) for k, v in {"nn_train": nn_train, "nn2_pool": nn2_pool, "nn_pool": nn_pool, "lof": lof}.items()}


def rank_average(scores: dict, members=None) -> np.ndarray:
    """Mean over ``members`` (default DEFAULT_MEMBERS) of rank / n of each score vector; values in (0, 1]."""
    from scipy.stats import rankdata
    members = tuple(DEFAULT_MEMBERS if members is None else members)
    if not members:
        raise ValueError("members is empty")
    missing = [m for m in members if m not in scores]
    if missing:
        raise ValueError(f"unknown members {missing}; available: {sorted(scores)}")
    return np.mean([rankdata(_finite(scores[m])) / len(scores[m]) for m in members], axis=0)


def fit_predict(train_mels, eval_mels, members=None, n_bands: int = 8, lof_neighbors: int = 5) -> np.ndarray:
    """Anomaly score (larger = more anomalous) of every eval clip: rank average of ``component_scores`` over ``members``."""
    if members is not None and isinstance(members, str):
        members = (members,)
    unknown = [m for m in (members or ()) if m not in COMPONENTS]
    if unknown:
        raise ValueError(f"unknown members {unknown}; available: {list(COMPONENTS)}")
    return rank_average(component_scores(train_mels, eval_mels, n_bands, lof_neighbors), members)


def score_clips(train: dict, evalset: dict, members=None, n_fft: int = 1024, hop: int = 512, n_mels: int = 128,
                embed=None, embed_members=("nn2_pool",)) -> np.ndarray:
    """``fit_predict`` from the outputs of load_train and load_eval_inputs (dicts with waveforms, lengths, sample_rate); ``embed`` /
    ``embed_members``: see the module docstring."""
    for name, d in (("train", train), ("evalset", evalset)):
        if not isinstance(d, dict) or "waveforms" not in d or "lengths" not in d:
            raise ValueError(f"{name} must be the dict returned by the load tool (keys waveforms, lengths, sample_rate)")
    sr = float(train.get("sample_rate", evalset.get("sample_rate", 16000.0)))
    if embed is not None:
        return _embedding_clip_scores(train, evalset, sr, members, n_fft, hop, n_mels, embed, embed_members)
    tm = log_mel_list(train["waveforms"], train["lengths"], sr, n_fft, hop, n_mels)
    em = log_mel_list(evalset["waveforms"], evalset["lengths"], float(evalset.get("sample_rate", sr)), n_fft, hop, n_mels)
    return fit_predict(tm, em, members)


def _embedding_clip_scores(train, evalset, sr, members, n_fft, hop, n_mels, embed, embed_members) -> np.ndarray:
    from . import audioenc
    models = (embed,) if isinstance(embed, str) else tuple(embed)
    comps = (embed_members,) if isinstance(embed_members, str) else tuple(embed_members)
    if not models or not comps:
        raise ValueError("embed and embed_members must name at least one model and one component")
    bad = [c for c in comps if c not in EMBEDDING_COMPONENTS]
    if bad:
        raise ValueError(f"unknown embed_members {bad}; available: {list(EMBEDDING_COMPONENTS)}")
    scores, names = {}, []
    for m in models:
        te = audioenc.embed(train["waveforms"], train["lengths"], sr, model=m, kind="pooled")
        ee = audioenc.embed(evalset["waveforms"], evalset["lengths"], float(evalset.get("sample_rate", sr)), model=m, kind="pooled")
        raw = embedding_scores(te, ee)
        for c in comps:
            scores[f"{m}:{c}"] = raw[c]
            names.append(f"{m}:{c}")
    if members is not None:
        mem = (members,) if isinstance(members, str) else tuple(members)
        unknown = [x for x in mem if x not in COMPONENTS]
        if unknown:
            raise ValueError(f"unknown members {unknown}; available: {list(COMPONENTS)}")
        tm = log_mel_list(train["waveforms"], train["lengths"], sr, n_fft, hop, n_mels)
        em = log_mel_list(evalset["waveforms"], evalset["lengths"], float(evalset.get("sample_rate", sr)), n_fft, hop, n_mels)
        scores.update(component_scores(tm, em))
        names += list(mem)
    return rank_average(scores, names)


# ----------------------------------------------------------------------------------------------- diagnostics / metric
def probe_report(train_mels, eval_mels) -> dict:
    """Diagnostics (no labels): Spearman correlation between the components and the nearest-neighbour kind per eval clip."""
    from scipy.stats import rankdata
    tr, ev = _mel_list(train_mels, "train_mels"), _mel_list(eval_mels, "eval_mels")
    comps = component_scores(tr, ev)
    names = list(COMPONENTS)
    R = np.array([rankdata(comps[k]) for k in names])
    corr = np.corrcoef(R) if R.shape[1] > 1 else np.eye(len(names))
    Zt, Ze = _zscore(clip_descriptors(tr, "ms"), clip_descriptors(ev, "ms"))
    d_t = _dist(Ze, Zt).min(axis=1)
    d_e = _dist(Ze, Ze)
    np.fill_diagonal(d_e, np.inf)
    d_e = d_e.min(axis=1) if len(ev) > 1 else np.full(len(ev), np.inf)
    return {"components": names, "spearman": np.nan_to_num(corr).round(3).tolist(),
            "nearest_kind": ["train" if a <= b else "eval" for a, b in zip(d_t, d_e)],
            "dist_to_nearest_train": d_t.round(4).tolist(), "dist_to_nearest_other_eval": d_e.round(4).tolist()}


def official_score(y_true, domain, score, max_fpr: float = 0.1) -> dict:
    """auc_source (source normals + all anomalies), auc_target (target normals + all anomalies), pauc (all clips, FPR <= max_fpr)
    and official_score = their harmonic mean. y_true: 1 = anomaly; domain: 'source' | 'target' per clip."""
    from scipy.stats import hmean
    from sklearn.metrics import roc_auc_score
    y, d, s = np.asarray(y_true), np.asarray(domain), np.asarray(score, dtype=float)
    out = {}
    for name, dom in (("auc_source", "source"), ("auc_target", "target")):
        sel = (d == dom) | (y != 0)
        out[name] = float(roc_auc_score(y[sel], s[sel]))
    out["pauc"] = float(roc_auc_score(y, s, max_fpr=max_fpr))
    out["official_score"] = float(hmean(np.maximum([out["auc_source"], out["auc_target"], out["pauc"]], np.finfo(float).eps)))
    return out
