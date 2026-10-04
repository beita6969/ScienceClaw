"""Supervised variant-effect prediction for deep-mutational-scanning (DMS) single substitutions (ProteinGym-style assays).

A variant ``table`` is a pandas DataFrame with one row per single substitution and the columns ``position`` (1-based index
into the wild-type sequence), ``wt_aa`` and ``mut_aa`` (one-letter codes); a labelled table also has ``DMS_score``
(higher = fitter). ``assay_id`` (or ``item``) names the assay of a row when a table holds several assays; the wild type is
one sequence string per assay (``wild_type``: str for one assay, dict assay_id -> str for several, as returned by
load_train / load_eval_inputs). Everything is computed from the arrays given: the labelled variants of the assay
(training rows), the wild-type sequence and the constant amino-acid tables below; no sequence database or pretrained
model is used unless extra feature columns are passed in (``scilib.proteinplm`` computes protein-language-model columns).

Metric
spearman(pred, truth) -> float                       Spearman rank correlation, 0.0 if either side is constant
mean_spearman(pred, truth, groups) -> dict           per-group Spearman ({"mean", "per_group"}); ``groups`` = assay id per row

Amino-acid and sequence tables
AA                                                   the 20 residues, in the row/column order of BLOSUM62 and ``aa_properties``
BLOSUM62                                             20 x 20 integer substitution matrix
aa_properties() -> DataFrame                         per residue (index AA): Kyte-Doolittle hydropathy, side-chain volume (A^3),
                                                     formal charge, polarity, Chou-Fasman helix / sheet / turn propensity,
                                                     aromatic flag, Grantham composition / polarity / volume
sequence_context(wild_type) -> DataFrame             one row per position (index 1..L) computed from the wild-type sequence only:
                                                     wild-type residue properties, windowed mean hydropathy / volume / charge /
                                                     aromatic, G, P fractions (+-2, +-5, +-10), helical (100 deg) and strand
                                                     (180 deg) hydropathy moments and neighbour projections, relative position,
                                                     distance to the termini

Position statistics of labelled variants
SiteMatrix(train, wild_type)                         positions x 20 matrix of the (standardised) training scores of one assay,
                                                     NaN where the variant is not in ``train``; ``.site_mean(shrink=..., window=...)``
                                                     per-position mean with a hierarchical prior (the mean of neighbouring positions
                                                     along the sequence, itself shrunk to the assay mean), ``.counts()``,
                                                     ``.features(table, exclude_self=...)`` -> DataFrame (below)
variant_features(table, wild_type, train, exclude_self=False) -> DataFrame
                                                     model inputs of each ``table`` row: substitution descriptors (BLOSUM62,
                                                     property values / differences, residue-class indicators), wild-type context,
                                                     and statistics of the training variants at the same position (count, mean,
                                                     median, std, min, max, shrunk mean, mean of each residue class, mean of the
                                                     mutant's own class, similarity-weighted means over BLOSUM62 / property
                                                     kernels) and at neighbouring positions (site means at offsets +-1..4, windowed
                                                     means). With ``exclude_self=True`` the row's own entry is removed from every
                                                     statistic (use it when ``table`` is ``train``, so a row never sees its own score)

Models (each takes the training table and the query table of ONE assay and returns one score per query row)
MODELS                                               the model names accepted by ``models=``: "site", "ridge", "lgbm", "et",
                                                     "factor", "knn"
fit_predict(train, wild_type, query, models=("ridge","lgbm","et"), ...) -> np.ndarray
                                                     features from ``variant_features`` (leave-one-out for the training rows) ->
                                                     the listed models -> rank-average of their predictions. Models: "site" (shrunk
                                                     site mean), "ridge" (ridge regression), "lgbm" (LightGBM), "et" (extra trees),
                                                     "factor" (site bias + residue bias + low-rank position x residue factors,
                                                     alternating least squares), "knn" (kNN in feature space). Optional
                                                     ``extra_train`` / ``extra_query``: further feature columns (DataFrames
                                                     row-aligned with the training / query tables, same columns) appended to the
                                                     features of "ridge", "lgbm" and "et"
fit_predict_assays(train, wild_type, query, models=..., pool=False, ...) -> np.ndarray
                                                     ``fit_predict`` per assay for multi-assay tables; output rows follow ``query``.
                                                     ``pool=True`` adds one LightGBM trained on the standardised rows of all assays;
                                                     ``plm=True`` passes the six ``scilib.proteinplm.plm_features`` columns of the
                                                     train and query rows as extra columns (needs ``proteinplm.available()``)
cross_validate(train, wild_type, models=..., n_folds=5, by="variant", seed=0, ...) -> dict
                                                     K-fold estimate of ``fit_predict`` on the training variants of one assay
                                                     (``by="variant"`` holds out random variants, ``by="position"`` holds out whole
                                                     positions); returns {"spearman": mean over folds, "per_fold": [...]}
rank_average(preds, weights=None) -> np.ndarray      weighted mean of the rank-normalised prediction vectors

Example (a code node with inputs train_table, dev_table, eval_table, wild_type; the dev table has no scores, its
predictions are scored by the score_dev tool):
    from scilib.proteinfit import fit_predict_assays
    dev_pred, eval_pred = fit_predict_assays(train_table, wild_type, [dev_table, eval_table])
    return {"dev_pred": dev_pred, "predictions": eval_pred}   # one float per row, in each table's row order

Notes: all functions are deterministic (``seed``); with the defaults ``fit_predict_assays`` on 16 assays x 1,024 training
variants runs on CPU in about a minute with 2 threads. ``fit_predict`` raises ValueError with an explanation when columns are
missing, a position lies outside the sequence, a ``wt_aa`` disagrees with the sequence, or the training table is too small.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

__all__ = ["AA", "BLOSUM62", "spearman", "mean_spearman", "aa_properties", "sequence_context", "SiteMatrix",
           "variant_features", "fit_predict", "fit_predict_assays", "cross_validate", "rank_average", "MODELS"]

AA = "ARNDCQEGHILKMFPSTWYV"
AAI = {a: i for i, a in enumerate(AA)}
MODELS = ("site", "ridge", "lgbm", "et", "factor", "knn")

_B62 = """
 4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
-1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
-2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
-2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
 0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
-1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
 0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
-2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
-1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
-1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
-1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
-2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
 1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
 0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
-2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
 0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""
BLOSUM62 = np.array([[int(v) for v in line.split()] for line in _B62.strip().splitlines()], dtype=int)

# per-residue constants, listed in AA order
_PROPS = {
    "hydropathy": [1.8, -4.5, -3.5, -3.5, 2.5, -3.5, -3.5, -0.4, -3.2, 4.5, 3.8, -3.9, 1.9, 2.8, -1.6, -0.8, -0.7,
                   -0.9, -1.3, 4.2],                                                    # Kyte & Doolittle 1982
    "volume": [88.6, 173.4, 114.1, 111.1, 108.5, 143.8, 138.4, 60.1, 153.2, 166.7, 166.7, 168.6, 162.9, 189.9, 112.7,
               89.0, 116.1, 227.8, 193.6, 140.0],                                        # Zamyatnin 1972, A^3
    "charge": [0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "polar": [0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0],             # R N D Q E H K S T Y
    "helix": [1.42, 0.98, 0.67, 1.01, 0.70, 1.11, 1.51, 0.57, 1.00, 1.08, 1.21, 1.16, 1.45, 1.13, 0.57, 0.77, 0.83,
              1.08, 0.69, 1.06],                                                         # Chou-Fasman Pa
    "sheet": [0.83, 0.93, 0.89, 0.54, 1.19, 1.10, 0.37, 0.75, 0.87, 1.60, 1.30, 0.74, 1.05, 1.38, 0.55, 0.75, 1.19,
              1.37, 1.47, 1.70],                                                         # Chou-Fasman Pb
    "turn": [0.66, 0.95, 1.56, 1.46, 1.19, 0.98, 0.74, 1.56, 0.95, 0.47, 0.59, 1.01, 0.60, 0.60, 1.52, 1.43, 0.96,
             0.96, 1.14, 0.50],                                                          # Chou-Fasman Pt
    "aromatic": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],          # H F W Y
    "grantham_c": [0, 0.65, 1.33, 1.38, 2.75, 0.89, 0.92, 0.74, 0.58, 0, 0, 0.33, 0, 0, 0.39, 1.42, 0.71, 0.13, 0.20, 0],
    "grantham_p": [8.1, 10.5, 11.6, 13.0, 5.5, 10.5, 12.3, 9.0, 10.4, 5.2, 4.9, 11.3, 5.7, 5.2, 8.0, 9.2, 8.6, 5.4,
                   6.2, 5.9],
    "grantham_v": [31, 124, 56, 54, 55, 85, 83, 3, 96, 111, 111, 119, 105, 132, 32.5, 32, 61, 170, 136, 84],
}
# residue classes (partition of the 20 residues) used for class-wise position statistics
_CLASSES = ("AVILMC", "FWY", "STNQ", "KRH", "DE", "G", "P")
_CLASS_OF = np.array([next(i for i, c in enumerate(_CLASSES) if a in c) for a in AA])
_CLASS_MASK = np.array([[a in c for a in AA] for c in _CLASSES])                       # (7, 20)
_KD = np.array(_PROPS["hydropathy"], dtype=float)
_VOL = np.array(_PROPS["volume"], dtype=float)


def aa_properties() -> pd.DataFrame:
    """Constant per-residue table (index = AA): hydropathy, volume, charge, polar, helix, sheet, turn, aromatic,
    grantham_c, grantham_p, grantham_v."""
    return pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in _PROPS.items()}, index=list(AA))


_PROP_Z = None


def _prop_matrix() -> np.ndarray:
    """(20, n_props) standardised property matrix."""
    global _PROP_Z
    if _PROP_Z is None:
        P = aa_properties().to_numpy()
        _PROP_Z = (P - P.mean(0)) / P.std(0)
    return _PROP_Z


# --------------------------------------------------------------------------------------------------------- metric
def spearman(pred, truth) -> float:
    """Spearman rank correlation with average ranks for ties; 0.0 if either vector is constant or shorter than 3."""
    from scipy.stats import spearmanr
    p, t = np.asarray(pred, dtype=float).ravel(), np.asarray(truth, dtype=float).ravel()
    if len(p) != len(t):
        raise ValueError(f"spearman: pred has {len(p)} values, truth {len(t)}")
    if len(p) < 3 or np.all(p == p[0]) or np.all(t == t[0]):
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = spearmanr(p, t).statistic
    return float(r) if np.isfinite(r) else 0.0


def mean_spearman(pred, truth, groups) -> dict:
    """Spearman per group (assay) and their mean: {"mean": float, "per_group": {group: rho}} (groups in order of first
    appearance)."""
    pred, truth, groups = np.asarray(pred, dtype=float), np.asarray(truth, dtype=float), np.asarray(groups)
    if not (len(pred) == len(truth) == len(groups)):
        raise ValueError("mean_spearman: pred, truth and groups must have the same length")
    per = {}
    for g in pd.unique(groups):
        m = groups == g
        per[g] = spearman(pred[m], truth[m])
    return {"mean": float(np.mean(list(per.values()))) if per else 0.0, "per_group": per}


def _rank01(x) -> np.ndarray:
    from scipy.stats import rankdata
    x = np.asarray(x, dtype=float)
    return (rankdata(x) - 0.5) / len(x) if len(x) else x


def rank_average(preds, weights=None) -> np.ndarray:
    """Weighted mean of the rank-normalised (0..1, average ranks) prediction vectors; ``preds``: list of 1-D arrays."""
    preds = [np.asarray(p, dtype=float).ravel() for p in preds]
    if not preds:
        raise ValueError("rank_average: no predictions given")
    w = np.ones(len(preds)) if weights is None else np.asarray(weights, dtype=float)
    if len(w) != len(preds):
        raise ValueError("rank_average: one weight per prediction vector is required")
    return sum(wi * _rank01(p) for wi, p in zip(w, preds)) / w.sum()


# --------------------------------------------------------------------------------------------------- sequence context
def _shift(x: np.ndarray, d: int, fill=np.nan) -> np.ndarray:
    """out[i] = x[i + d] (``fill`` outside the array)."""
    n = len(x)
    out = np.full(n, fill, dtype=float)
    if d >= 0:
        out[:n - d] = x[d:]
    else:
        out[-d:] = x[:n + d]
    return out


def _window_mean(x: np.ndarray, w: int, include_center: bool = True) -> np.ndarray:
    tot, cnt = np.zeros(len(x)), np.zeros(len(x))
    for d in range(-w, w + 1):
        if d == 0 and not include_center:
            continue
        s = _shift(x, d)
        ok = ~np.isnan(s)
        tot += np.where(ok, s, 0.0)
        cnt += ok
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def _moment(h: np.ndarray, angle_deg: float, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Hydropathy moment magnitude over +-w and the projection of the neighbours onto the periodicity phase of the centre."""
    re, im, n = np.zeros(len(h)), np.zeros(len(h)), np.zeros(len(h))
    proj = np.zeros(len(h))
    th = np.deg2rad(angle_deg)
    for d in range(-w, w + 1):
        s = _shift(h, d)
        ok = ~np.isnan(s)
        v = np.where(ok, s, 0.0)
        re += v * np.cos(th * d)
        im += v * np.sin(th * d)
        n += ok
        if d != 0:
            proj += v * np.cos(th * d)
    n = np.maximum(n, 1)
    return np.sqrt(re ** 2 + im ** 2) / n, proj / n


def _encode(seq: str) -> np.ndarray:
    return np.array([AAI.get(c, -1) for c in seq], dtype=int)


def _prop_of(idx: np.ndarray, values) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    out = np.where(idx >= 0, v[np.clip(idx, 0, 19)], np.nan)
    return out


def sequence_context(wild_type: str) -> pd.DataFrame:
    """Position-level features of the wild-type sequence (index = 1-based position): wt residue properties, windowed
    composition (+-2, +-5, +-10), hydropathy moments for a 100 degree (helix) and 180 degree (strand) periodicity,
    neighbour projection of hydropathy on those periodicities, relative position and distance to the termini."""
    seq = str(wild_type).strip().upper()
    if not seq:
        raise ValueError("sequence_context: empty wild-type sequence")
    idx = _encode(seq)
    L = len(seq)
    h = _prop_of(idx, _KD)
    cols: dict[str, np.ndarray] = {"wt_hydropathy": h, "wt_volume": _prop_of(idx, _VOL)}
    for name in ("charge", "helix", "sheet", "turn", "aromatic"):
        cols[f"wt_{name}"] = _prop_of(idx, _PROPS[name])
    gly = np.where(idx >= 0, (idx == AAI["G"]).astype(float), np.nan)
    pro = np.where(idx >= 0, (idx == AAI["P"]).astype(float), np.nan)
    for w in (2, 5, 10):
        cols[f"win{w}_hydropathy"] = _window_mean(h, w)
        cols[f"win{w}_volume"] = _window_mean(cols["wt_volume"], w)
        cols[f"win{w}_charge"] = _window_mean(cols["wt_charge"], w)
        cols[f"win{w}_aromatic"] = _window_mean(cols["wt_aromatic"], w)
        cols[f"win{w}_gly"] = _window_mean(gly, w)
        cols[f"win{w}_pro"] = _window_mean(pro, w)
    for name, ang in (("helix", 100.0), ("strand", 180.0)):
        mag, proj = _moment(h, ang, 5)
        cols[f"moment_{name}"] = mag
        cols[f"proj_{name}"] = proj
    pos = np.arange(1, L + 1, dtype=float)
    cols["rel_pos"] = pos / L
    cols["dist_term"] = np.minimum(pos - 1, L - pos)
    cols["seq_len"] = np.full(L, float(L))
    df = pd.DataFrame(cols, index=np.arange(1, L + 1))
    return df.fillna(df.mean())


# ------------------------------------------------------------------------------------------------ position statistics
def _check_table(t, need, what) -> None:
    if not isinstance(t, pd.DataFrame):
        raise ValueError(f"{what} must be a pandas DataFrame, got {type(t).__name__}")
    miss = [c for c in need if c not in t.columns]
    if miss:
        raise ValueError(f"{what} lacks column(s) {miss}; it has {list(t.columns)}")


def _wt_str(wild_type, what="wild_type") -> str:
    if isinstance(wild_type, dict):
        raise ValueError(f"{what}: a dict of sequences was given; select one assay's sequence (or use fit_predict_assays)")
    s = str(wild_type).strip().upper()
    if not s:
        raise ValueError(f"{what}: empty wild-type sequence")
    return s


def _positions_aas(t: pd.DataFrame, seq: str, what: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _check_table(t, ["position", "wt_aa", "mut_aa"], what)
    pos = t["position"].to_numpy()
    try:
        pos = pos.astype(int)
    except (TypeError, ValueError):
        raise ValueError(f"{what}: column position must be integer (1-based), e.g. 24 for 'A24G'")
    L = len(seq)
    if len(pos) and (pos.min() < 1 or pos.max() > L):
        raise ValueError(f"{what}: position outside the wild-type sequence (1..{L}): {int(pos.min())}..{int(pos.max())}")
    wa = np.array([AAI.get(str(a), -1) for a in t["wt_aa"]], dtype=int)
    ma = np.array([AAI.get(str(a), -1) for a in t["mut_aa"]], dtype=int)
    if (ma < 0).any():
        bad = sorted({str(a) for a in t["mut_aa"] if str(a) not in AAI})
        raise ValueError(f"{what}: mut_aa must be one of the 20 standard residues {AA}; found {bad[:5]}")
    ref = _encode(seq)[pos - 1]
    if (wa != ref).any():
        i = int(np.flatnonzero(wa != ref)[0])
        raise ValueError(f"{what}: wt_aa {t['wt_aa'].iloc[i]!r} at position {int(pos[i])} disagrees with the wild-type "
                         f"sequence ({seq[pos[i] - 1]!r}); is the sequence of another assay being used?")
    return pos, wa, ma


def _shrunk_mean(total, count, prior, k):
    return (total + k * prior) / (count + k)


def _kernels() -> dict[str, np.ndarray]:
    """Residue-similarity kernels (20 x 20, zero diagonal) used for similarity-weighted position means."""
    B = BLOSUM62.astype(float)
    K1 = np.exp(B / 2.0)
    Z = _prop_matrix()
    D2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    K2 = np.exp(-D2 / (2 * 2.0 ** 2 * Z.shape[1] / 4.0))
    hyd = np.abs(_KD[:, None] - _KD[None, :])
    K3 = np.exp(-hyd / 2.0) * np.exp(-np.abs(_VOL[:, None] - _VOL[None, :]) / 40.0)
    for K in (K1, K2, K3):
        np.fill_diagonal(K, 0.0)
    return {"blosum": K1, "props": K2, "hydvol": K3}


_KERNELS = _kernels()


class SiteMatrix:
    """Standardised training scores of one assay as a (positions x 20) matrix (NaN = variant not in the training rows).

    ``train`` needs the columns position, wt_aa, mut_aa, DMS_score. Scores are standardised (mean 0, sd 1 over the training
    rows) unless ``standardize=False``; ``.mu_`` and ``.sd_`` are the constants used."""

    def __init__(self, train: pd.DataFrame, wild_type: str, standardize: bool = True):
        seq = _wt_str(wild_type)
        _check_table(train, ["DMS_score"], "train")
        pos, _, aa = _positions_aas(train, seq, "train")
        y = train["DMS_score"].to_numpy(dtype=float)
        if len(y) < 5:
            raise ValueError(f"train has {len(y)} labelled variants; at least 5 are needed")
        if not np.isfinite(y).all():
            raise ValueError("train: DMS_score contains non-finite values")
        self.L = len(seq)
        self.seq = seq
        self.mu_ = float(y.mean()) if standardize else 0.0
        self.sd_ = float(y.std()) or 1.0 if standardize else 1.0
        z = (y - self.mu_) / self.sd_
        tot, cnt = np.zeros((self.L, 20)), np.zeros((self.L, 20))
        np.add.at(tot, (pos - 1, aa), z)
        np.add.at(cnt, (pos - 1, aa), 1.0)
        self.M = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
        self.n_train = len(y)

    def counts(self) -> np.ndarray:
        """Number of training variants per position (index 0 = position 1)."""
        return (~np.isnan(self.M)).sum(1)

    def _sums(self) -> tuple[np.ndarray, np.ndarray]:
        obs = ~np.isnan(self.M)
        return np.where(obs, self.M, 0.0).sum(1), obs.sum(1).astype(float)

    def neighbour_prior(self, window: int = 3, sigma: float = 2.0, shrink: float = 2.0) -> np.ndarray:
        """Per position: Gaussian-weighted mean of the training scores at the OTHER positions within +-window (positions with
        no training variant contribute nothing), shrunk towards 0 (the assay mean) with pseudo-count ``shrink``."""
        s, c = self._sums()
        num, den = np.zeros(self.L), np.zeros(self.L)
        for d in range(-window, window + 1):
            if d == 0:
                continue
            w = np.exp(-0.5 * (d / sigma) ** 2)
            ss, cc = _shift(s, d, 0.0), _shift(c, d, 0.0)
            num += w * ss
            den += w * cc
        return _shrunk_mean(num, den, 0.0, shrink)

    def site_mean(self, shrink: float = 1.0, window: int = 3, sigma: float = 2.0, prior_shrink: float = 2.0,
                  standardized: bool = True) -> np.ndarray:
        """(L,) hierarchical per-position mean: (sum of the position's scores + shrink * prior) / (count + shrink), where
        the prior is ``neighbour_prior`` (mean of neighbouring positions, itself shrunk to the assay mean). Scores are in
        standardised units unless ``standardized=False``."""
        s, c = self._sums()
        prior = self.neighbour_prior(window, sigma, prior_shrink)
        m = _shrunk_mean(s, c, prior, shrink)
        return m if standardized else m * self.sd_ + self.mu_

    def features(self, table: pd.DataFrame, exclude_self: bool = False, shrink: float = 1.0, window: int = 3) -> pd.DataFrame:
        """Position statistics for each ``table`` row (see ``variant_features``); ``exclude_self`` removes the row's own
        variant from the statistics of its position."""
        pos, _, aa = _positions_aas(table, self.seq, "table")
        return self._features(pos - 1, aa, exclude_self, shrink, window)

    def _features(self, p: np.ndarray, a: np.ndarray, exclude_self: bool, shrink: float, window: int) -> pd.DataFrame:
        n = len(p)
        R = self.M[p].copy()
        if exclude_self:
            R[np.arange(n), a] = np.nan
        obs = ~np.isnan(R)
        Rz = np.where(obs, R, 0.0)
        cnt = obs.sum(1).astype(float)
        tot = Rz.sum(1)
        prior_nb = self.neighbour_prior(window)[p]                    # excludes the position itself: no self-leak
        out: dict[str, np.ndarray] = {}
        out["site_n"] = cnt
        out["site_mean"] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
        out["site_shrunk"] = _shrunk_mean(tot, cnt, prior_nb, shrink)
        out["site_shrunk_hard"] = _shrunk_mean(tot, cnt, prior_nb, 4.0)
        out["nb_prior"] = prior_nb
        srt = np.sort(np.where(obs, R, np.inf), axis=1)
        lo = np.clip((cnt - 1) // 2, 0, 19).astype(int)
        hi = np.clip(cnt // 2, 0, 19).astype(int)
        r = np.arange(n)
        med = 0.5 * (srt[r, lo] + srt[r, hi])
        out["site_median"] = np.where(cnt > 0, med, np.nan)
        out["site_min"] = np.where(cnt > 0, srt[:, 0], np.nan)
        mx = np.sort(np.where(obs, R, -np.inf), axis=1)[:, -1]
        out["site_max"] = np.where(cnt > 0, mx, np.nan)
        var = np.where(cnt > 1, (np.where(obs, (R - out["site_mean"][:, None]) ** 2, 0.0)).sum(1) / np.maximum(cnt - 1, 1), np.nan)
        out["site_std"] = np.sqrt(var)
        # class-wise means over the observed variants of the position
        for ci, mask in enumerate(_CLASS_MASK):
            o = obs & mask[None, :]
            c = o.sum(1)
            out[f"cls{ci}_mean"] = np.where(c > 0, (Rz * o).sum(1) / np.maximum(c, 1), np.nan)
        own = _CLASS_MASK[_CLASS_OF[a]]
        o = obs & own
        c = o.sum(1)
        out["own_class_mean"] = np.where(c > 0, (Rz * o).sum(1) / np.maximum(c, 1), np.nan)
        out["own_class_n"] = c.astype(float)
        # similarity-weighted means
        for name, K in _KERNELS.items():
            W = K[a] * obs
            den = W.sum(1)
            out[f"sim_{name}"] = np.where(den > 0, (W * Rz).sum(1) / np.where(den > 0, den, 1.0), np.nan)
        # nearest observed substitutions by hydropathy: means of the two most similar observed residues
        d = np.abs(_KD[a][:, None] - _KD[None, :]) + np.where(obs, 0.0, np.inf)
        order = np.argsort(d, axis=1, kind="stable")[:, :2]
        vals = np.take_along_axis(R, order, 1)
        okk = np.isfinite(np.take_along_axis(d, order, 1))
        out["near_hyd2"] = np.where(okk.sum(1) > 0, np.where(okk, vals, 0).sum(1) / np.maximum(okk.sum(1), 1), np.nan)
        # neighbouring positions
        s, c_all = self._sums()
        sm = np.where(c_all > 0, s / np.maximum(c_all, 1), np.nan)
        for dd in (-4, -3, -2, -1, 1, 2, 3, 4):
            out[f"nb{dd:+d}"] = _shift(sm, dd)[p]
        for name, offs in (("nb_win1", (-1, 1)), ("nb_win2", (-2, -1, 1, 2)), ("nb_helix", (-4, -3, 3, 4)),
                           ("nb_strand", (-2, 2)), ("nb_far", (-7, -6, -5, 5, 6, 7))):
            stack = np.stack([_shift(sm, o) for o in offs], 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[name] = np.nanmean(stack, 1)[p]
        return pd.DataFrame(out)


def _substitution_block(wa: np.ndarray, ma: np.ndarray) -> pd.DataFrame:
    P = aa_properties().to_numpy()
    names = list(aa_properties().columns)
    out: dict[str, np.ndarray] = {"blosum62": BLOSUM62[np.clip(wa, 0, 19), ma].astype(float)}
    for j, nm in enumerate(names):
        out[f"mut_{nm}"] = P[ma, j]
    for j, nm in enumerate(names[:8]):
        wv = np.where(wa >= 0, P[np.clip(wa, 0, 19), j], np.nan)
        if nm in ("hydropathy", "volume", "charge", "helix", "sheet", "aromatic"):
            out[f"delta_{nm}"] = P[ma, j] - wv
        if nm in ("polar", "turn"):
            out[f"delta_{nm}"] = P[ma, j] - wv
    mc = _CLASS_OF[ma]
    for ci in range(len(_CLASSES)):
        out[f"mut_cls{ci}"] = (mc == ci).astype(float)
    out["mut_idx"] = ma.astype(float)
    out["wt_idx"] = wa.astype(float)
    out["same_class"] = (_CLASS_OF[np.clip(wa, 0, 19)] == mc).astype(float)
    out["to_pro"] = (ma == AAI["P"]).astype(float)
    out["from_gly"] = (wa == AAI["G"]).astype(float)
    out["to_gly"] = (ma == AAI["G"]).astype(float)
    out["from_pro"] = (wa == AAI["P"]).astype(float)
    return pd.DataFrame(out)


def variant_features(table: pd.DataFrame, wild_type: str, train: pd.DataFrame, exclude_self: bool = False,
                     shrink: float = 1.0, window: int = 3) -> pd.DataFrame:
    """Feature table (same row order as ``table``, all float, NaN = not defined) of substitution descriptors, wild-type
    context and position statistics of the ``train`` variants (columns: see the module docstring). Position statistics
    are in standardised units of ``train`` (mean 0, sd 1). Pass ``exclude_self=True`` when ``table`` is ``train`` itself."""
    seq = _wt_str(wild_type)
    sm = SiteMatrix(train, seq)
    pos, wa, ma = _positions_aas(table, seq, "table")
    sub = _substitution_block(wa, ma)
    ctx = sequence_context(seq).iloc[pos - 1].reset_index(drop=True)
    st = sm._features(pos - 1, ma, exclude_self, shrink, window)
    X = pd.concat([sub, ctx, st], axis=1)
    X.index = table.index
    return X.astype(float)


# ------------------------------------------------------------------------------------------------------------ models
def _gauss_rank(y: np.ndarray) -> np.ndarray:
    from scipy.special import ndtri
    return ndtri(_rank01(y))


def _impute(Xtr: np.ndarray, Xte: np.ndarray, extra: list[np.ndarray] | None = None) -> tuple[np.ndarray, ...]:
    med = np.nanmedian(np.where(np.isfinite(Xtr), Xtr, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return tuple(np.where(np.isfinite(A), A, med[None, :]) for A in (Xtr, Xte, *(extra or [])))


def _ridge_fit_predict(Xtr, ytr, Xte, seed) -> np.ndarray:
    from sklearn.linear_model import RidgeCV
    Xtr_i, Xte_i = _impute(Xtr, Xte)
    mu, sd = Xtr_i.mean(0), Xtr_i.std(0)
    keep = sd > 1e-9
    A = (Xtr_i[:, keep] - mu[keep]) / sd[keep]
    B = (Xte_i[:, keep] - mu[keep]) / sd[keep]
    B = np.clip(B, -6, 6)
    m = RidgeCV(alphas=np.logspace(0, 3.5, 12)).fit(A, ytr)
    return m.predict(B)


def _lgbm_fit_predict(Xtr, ytr, Xte, seed, n_estimators=300, learning_rate=0.03, num_leaves=6, min_child_samples=10,
                      colsample=0.5, subsample=0.8, reg_lambda=5.0, n_jobs=2) -> np.ndarray:
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
                          min_child_samples=min_child_samples, colsample_bytree=colsample, subsample=subsample,
                          subsample_freq=1, reg_lambda=reg_lambda, random_state=seed, n_jobs=n_jobs, verbose=-1,
                          deterministic=True, force_row_wise=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(Xtr, ytr)
        return m.predict(Xte)


def _et_fit_predict(Xtr, ytr, Xte, seed) -> np.ndarray:
    from sklearn.ensemble import ExtraTreesRegressor
    A, B = _impute(Xtr, Xte)
    m = ExtraTreesRegressor(n_estimators=250, min_samples_leaf=3, max_features=0.4, random_state=seed, n_jobs=2)
    return m.fit(A, ytr).predict(B)


def _knn_fit_predict(Xtr, ytr, Xte, seed, cols=None, k=15) -> np.ndarray:
    from sklearn.neighbors import KNeighborsRegressor
    A, B = _impute(Xtr, Xte)
    mu, sd = A.mean(0), A.std(0)
    keep = sd > 1e-9
    A, B = (A[:, keep] - mu[keep]) / sd[keep], np.clip((B[:, keep] - mu[keep]) / sd[keep], -6, 6)
    return KNeighborsRegressor(n_neighbors=min(k, len(A)), weights="distance").fit(A, ytr).predict(B)


def _factor_fit_predict(sm: SiteMatrix, p_te: np.ndarray, a_te: np.ndarray, rank: int = 2, lam: float = 3.0,
                        kappa: float = 1.0, n_iter: int = 25, seed: int = 0, window: int = 3) -> np.ndarray:
    """mu + b_p + c_a + u_p . v_a fitted by alternating ridge regressions on the observed cells of ``sm`` (b_p is shrunk
    towards the neighbouring-position prior)."""
    M = sm.M
    obs = ~np.isnan(M)
    Z = np.where(obs, M, 0.0)
    L = M.shape[0]
    prior = sm.neighbour_prior(window)
    rng = np.random.default_rng(seed)
    V = rng.normal(0, 0.3, (20, rank))
    c = np.zeros(20)
    U = np.zeros((L, rank))
    b = prior.copy()
    obsf = obs.astype(float)
    for _ in range(n_iter):
        # positions: r = M - c_a; unknowns [b_p - prior_p, u_p]
        r = np.where(obs, Z - c[None, :] - prior[:, None], 0.0)
        Va = np.concatenate([np.ones((20, 1)), V], 1)
        A = np.einsum("pa,ai,aj->pij", obsf, Va, Va)
        pen = np.diag([kappa] + [lam] * rank)
        A = A + pen[None]
        rhs = np.einsum("pa,ai->pi", r, Va)
        sol = np.linalg.solve(A, rhs[..., None])[..., 0]
        b, U = sol[:, 0] + prior, sol[:, 1:]
        # residues: r = M - b_p; unknowns [c_a, v_a]
        r2 = np.where(obs, Z - b[:, None], 0.0)
        Ua = np.concatenate([np.ones((L, 1)), U], 1)
        A2 = np.einsum("pa,pi,pj->aij", obsf, Ua, Ua) + np.diag([1e-3] + [lam] * rank)[None]
        rhs2 = np.einsum("pa,pi->ai", r2, Ua)
        sol2 = np.linalg.solve(A2, rhs2[..., None])[..., 0]
        c, V = sol2[:, 0], sol2[:, 1:]
    return b[p_te] + c[a_te] + (U[p_te] * V[a_te]).sum(1)


def _check_models(models) -> tuple[str, ...]:
    models = (models,) if isinstance(models, str) else tuple(models)
    bad = [m for m in models if m not in MODELS]
    if bad or not models:
        raise ValueError(f"unknown model(s) {bad}; choose from {list(MODELS)}")
    return models


def _as_list(query):
    """(list of query tables, was_list)."""
    if isinstance(query, pd.DataFrame):
        return [query], False
    if isinstance(query, (list, tuple)) and query and all(isinstance(q, pd.DataFrame) for q in query):
        return list(query), True
    raise ValueError("query must be a DataFrame or a list of DataFrames")


def _split(x: np.ndarray, sizes: list[int]) -> list[np.ndarray]:
    return np.split(x, np.cumsum(sizes)[:-1])


def _extra_block(extra, n: int, what: str) -> pd.DataFrame | None:
    """Validated extra feature columns (a DataFrame with ``n`` rows, row-aligned with the table), positionally reset."""
    if extra is None:
        return None
    if not isinstance(extra, pd.DataFrame) or len(extra) != n:
        raise ValueError(f"{what} must be a DataFrame with one row per row of the table ({n} rows)")
    return extra.reset_index(drop=True).astype(float)


def _member_predictions(train, seq, queries, models, target, shrink, window, seed, extra_train=None, extra_query=None):
    """model -> array over the concatenated ``queries``, plus the feature tables (train, concatenated query)."""
    _check_table(train, ["position", "wt_aa", "mut_aa", "DMS_score"], "train")
    qall = pd.concat(queries, ignore_index=True)
    Xtr = variant_features(train, seq, train, exclude_self=True, shrink=shrink, window=window)
    Xte = variant_features(qall, seq, train, exclude_self=False, shrink=shrink, window=window)
    etr = _extra_block(extra_train, len(train), "extra_train")
    if etr is not None:
        eq = extra_query if isinstance(extra_query, (list, tuple)) else [extra_query]
        if extra_query is None or len(eq) != len(queries):
            raise ValueError("extra_query must be given with extra_train: one DataFrame per query table")
        ete = pd.concat([_extra_block(e, len(q), "extra_query") for e, q in zip(eq, queries)], ignore_index=True)
        if list(ete.columns) != list(etr.columns):
            raise ValueError("extra_train and extra_query must have the same columns")
        Xtr = pd.concat([Xtr.reset_index(drop=True), etr], axis=1)
        Xte = pd.concat([Xte.reset_index(drop=True), ete], axis=1)
    elif extra_query is not None:
        raise ValueError("extra_query was given without extra_train")
    y = train["DMS_score"].to_numpy(dtype=float)
    yt = _gauss_rank(y) if target == "rank" else (y - y.mean()) / (y.std() or 1.0)
    A, B = Xtr.to_numpy(), Xte.to_numpy()
    members: dict[str, np.ndarray] = {}
    for m in models:
        if m == "site":
            members[m] = Xte["site_shrunk"].fillna(0.0).to_numpy() + 1e-6 * Xte["blosum62"].to_numpy()
        elif m == "ridge":
            members[m] = _ridge_fit_predict(A, yt, B, seed)
        elif m == "lgbm":
            members[m] = _lgbm_fit_predict(A, yt, B, seed)
        elif m == "et":
            members[m] = _et_fit_predict(A, yt, B, seed)
        elif m == "knn":
            members[m] = _knn_fit_predict(A, yt, B, seed)
        elif m == "factor":
            pos, _, aa = _positions_aas(qall, seq, "query")
            members[m] = _factor_fit_predict(SiteMatrix(train, seq), pos - 1, aa, seed=seed, window=window)
    return members, Xtr, Xte, yt


def fit_predict(train: pd.DataFrame, wild_type: str, query, models=("ridge", "lgbm", "et"), weights=None,
                target: str = "rank", shrink: float = 1.0, window: int = 3, seed: int = 0, return_members: bool = False,
                extra_train: pd.DataFrame | None = None, extra_query=None):
    """Scores for the ``query`` variants of ONE assay from its labelled ``train`` variants (see module docstring).

    Position statistics of every training row are computed without that row (``variant_features(..., exclude_self=True)``);
    the query rows use all training rows. ``query``: a table, or a list of tables scored by the same fitted models (each
    table is rank-averaged on its own). ``models``: names from ``MODELS``; ``weights``: one weight per model for the
    rank-average; ``target``: "rank" (fit normal scores of the training ranks) or "raw" (fit standardised scores);
    ``shrink``/``window``: position-mean prior pseudo-count and neighbour window. Returns a float array with one score per
    ``query`` row (higher = fitter), or a list of arrays for a list of tables; with ``return_members=True`` the result is
    followed by a dict model -> its raw score array (per table when ``query`` is a list). ``extra_train`` / ``extra_query``:
    optional additional feature columns (a DataFrame row-aligned with ``train``; a DataFrame, or a list of DataFrames, row-aligned
    with ``query``; same columns) that are appended to the features of "ridge", "lgbm" and "et" (the other models ignore them)."""
    models = _check_models(models)
    if target not in ("rank", "raw"):
        raise ValueError("target must be 'rank' or 'raw'")
    seq = _wt_str(wild_type)
    queries, was_list = _as_list(query)
    members = _member_predictions(train, seq, queries, models, target, shrink, window, seed, extra_train, extra_query)[0]
    sizes = [len(q) for q in queries]
    per_model = {m: _split(v, sizes) for m, v in members.items()}
    preds = [rank_average([per_model[m][k] for m in models], weights) for k in range(len(queries))]
    if not was_list:
        preds = preds[0]
        per_model = {m: v[0] for m, v in per_model.items()}
    return (preds, per_model) if return_members else preds


def _group_column(t: pd.DataFrame, group: str | None, what: str) -> np.ndarray:
    if group is None:
        for g in ("assay_id", "item"):
            if g in t.columns:
                return t[g].to_numpy()
        return np.zeros(len(t), dtype=int)
    if group not in t.columns:
        raise ValueError(f"{what} has no column {group!r}; columns: {list(t.columns)}")
    return t[group].to_numpy()


def _assay_wt(wild_type, key, gkey_alt=None) -> str:
    if isinstance(wild_type, dict):
        for k in (key, str(key), gkey_alt):
            if k is not None and k in wild_type:
                return wild_type[k]
        raise ValueError(f"wild_type has no sequence for assay {key!r}; keys: {list(wild_type)[:5]}")
    return wild_type


def fit_predict_assays(train: pd.DataFrame, wild_type, query, models=("ridge", "lgbm", "et"), pool: bool = False,
                       group: str | None = None, weights=None, target: str = "rank", shrink: float = 1.0, window: int = 3,
                       seed: int = 0, extra_train: pd.DataFrame | None = None, extra_query=None, plm: bool = False):
    """``fit_predict`` for every assay of a multi-assay table (assays identified by the ``group`` column, default
    ``assay_id`` else ``item``; ``wild_type``: dict assay_id -> sequence, or one string). ``query``: a table or a list of
    tables (e.g. ``[dev_table, eval_table]``, fitted once). Returns one score per query row in the row order of each table
    (a list of arrays for a list of tables); scores are comparable within an assay only. ``pool=True`` adds to every
    assay's rank-average one LightGBM model trained on the position-statistic features of ALL assays' training rows
    (targets = normal scores of the within-assay ranks). ``extra_train`` / ``extra_query``: optional additional feature columns
    (DataFrames row-aligned with ``train`` and with each query table, same columns), appended to the features of "ridge",
    "lgbm" and "et" of every assay. ``plm=True`` passes the six ``scilib.proteinplm.plm_features`` columns of the train and
    query rows as ``extra_train`` / ``extra_query`` (needs ``scilib.proteinplm.available()``; not combinable with
    ``extra_train``)."""
    models = _check_models(models)
    if target not in ("rank", "raw"):
        raise ValueError("target must be 'rank' or 'raw'")
    queries, was_list = _as_list(query)
    for q in queries:
        _check_table(q, ["position", "wt_aa", "mut_aa"], "query")
    gtr = _group_column(train, group, "train")
    gqs = [_group_column(q, group, "query") for q in queries]
    if plm:
        if extra_train is not None or extra_query is not None:
            raise ValueError("plm=True cannot be combined with extra_train / extra_query")
        from . import proteinplm
        if not proteinplm.available():
            raise RuntimeError("plm=True: scilib.proteinplm is not available here")
        parts = [(train, gtr)] + list(zip(queries, gqs))
        allt = pd.concat([pd.DataFrame({"position": t["position"].to_numpy(), "wt_aa": t["wt_aa"].to_numpy(),
                                        "mut_aa": t["mut_aa"].to_numpy(), "assay_id": g}) for t, g in parts], ignore_index=True)
        feats = proteinplm.plm_features(allt, wild_type)
        cuts = np.cumsum([0] + [len(t) for t, _ in parts])
        blocks = [feats.iloc[cuts[i]:cuts[i + 1]].reset_index(drop=True) for i in range(len(parts))]
        extra_train, extra_query = blocks[0], blocks[1:]
    ids = list(pd.unique(np.concatenate(gqs)))
    have = set(gtr.tolist())
    missing = [a for a in ids if a not in have]
    if missing:
        raise ValueError(f"train has no rows for query assay(s) {missing[:5]}")
    if (extra_train is None) != (extra_query is None):
        raise ValueError("extra_train and extra_query must be given together")
    eqs = None
    if extra_train is not None:
        eqs = extra_query if isinstance(extra_query, (list, tuple)) else [extra_query]
        if len(eqs) != len(queries):
            raise ValueError("extra_query must have one DataFrame per query table")
        _extra_block(extra_train, len(train), "extra_train")
        for e, q in zip(eqs, queries):
            _extra_block(e, len(q), "extra_query")
    fitted = {}
    for a in ids:
        qs = [q[g == a] for q, g in zip(queries, gqs)]
        et = extra_train[gtr == a] if extra_train is not None else None
        eq = [e[g == a] for e, g in zip(eqs, gqs)] if eqs is not None else None
        try:
            fitted[a] = (qs,) + _member_predictions(train[gtr == a], _assay_wt(wild_type, a), qs, models, target, shrink,
                                                    window, seed, et, eq)
        except ValueError as ex:
            raise ValueError(f"assay {a!r}: {ex}") from None
    pooled = {}
    if pool:
        Xall = np.concatenate([fitted[a][2].to_numpy() for a in ids])
        yall = np.concatenate([fitted[a][4] for a in ids])
        Xq = np.concatenate([fitted[a][3].to_numpy() for a in ids])
        pp = _lgbm_fit_predict(Xall, yall, Xq, seed, n_estimators=400, num_leaves=12, min_child_samples=20)
        pooled = dict(zip(ids, _split(pp, [len(fitted[a][3]) for a in ids])))
    outs = [np.zeros(len(q)) for q in queries]
    for a in ids:
        qs, members, _, _, _ = fitted[a]
        sizes = [len(q) for q in qs]
        parts = {m: _split(v, sizes) for m, v in members.items()}
        if a in pooled:
            parts["pooled"] = _split(pooled[a], sizes)
        names = list(models) + (["pooled"] if a in pooled else [])
        w = None if weights is None else list(weights) + ([1.0] if a in pooled else [])
        for k, g in enumerate(gqs):
            outs[k][g == a] = rank_average([parts[m][k] for m in names], w)
    return outs if was_list else outs[0]


def cross_validate(train: pd.DataFrame, wild_type: str, models=("ridge", "lgbm", "et"), n_folds: int = 5,
                   by: str = "variant", seed: int = 0, **kw) -> dict:
    """K-fold estimate of ``fit_predict`` on the labelled variants of ONE assay: each fold trains on the other folds and
    scores the held-out rows with Spearman. ``by="variant"``: random variant folds; ``by="position"``: whole positions are
    held out together. Returns {"spearman": mean over folds, "per_fold": [rho, ...]}."""
    if by not in ("variant", "position"):
        raise ValueError("by must be 'variant' or 'position'")
    _check_table(train, ["position", "wt_aa", "mut_aa", "DMS_score"], "train")
    n = len(train)
    if n < 5 * 2:
        raise ValueError(f"train has {n} rows; too few for cross-validation")
    rng = np.random.default_rng(seed)
    if by == "variant":
        fold = rng.permutation(n) % n_folds
    else:
        pos = train["position"].to_numpy(dtype=int)
        up = np.unique(pos)
        pf = dict(zip(rng.permutation(up), np.arange(len(up)) % n_folds))
        fold = np.array([pf[p] for p in pos])
    y = train["DMS_score"].to_numpy(dtype=float)
    rhos = []
    for k in range(n_folds):
        te = fold == k
        if te.sum() < 3 or (~te).sum() < 10:
            continue
        pred = fit_predict(train[~te], wild_type, train[te], models=models, seed=seed, **kw)
        rhos.append(spearman(pred, y[te]))
    return {"spearman": float(np.mean(rhos)) if rhos else 0.0, "per_fold": [float(r) for r in rhos]}
