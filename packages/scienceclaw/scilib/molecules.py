"""Fingerprint / descriptor features and tree-ensemble classifiers for molecules given as SMILES (binary labels, ROC-AUC).

featurize(smiles, kinds="concat", ...) -> (X, valid, names)   RDKit features, X float32 (n, d), finite; valid (n,) bool
scaffold_groups(smiles) -> ndarray[int] (n,)                  Bemis-Murcko scaffold id (same id = same scaffold)
fit_predict(train, y, *queries, ...) -> scores                fit on labelled molecules, score each query set
grouped_cv_auc(train, y, groups="scaffold", ...) -> dict      out-of-fold ROC-AUC with folds that never split a group
TreeEnsemble(...).fit(X, y).predict(X)                        the classifier used by the two functions above

``train`` and each query set are lists of SMILES strings (``fit_predict`` featurises them itself with ``kinds``) or
ready 2-D feature matrices; ``y`` is 0/1 with one entry per ``train`` row (rows RDKit cannot parse are dropped inside).
``fit_predict`` returns one float array in [0, 1] per query set (a single array if one set is given, a list
otherwise); higher = more likely label 1.

Feature kinds (``kinds`` = one name or a list; several kinds are concatenated in the order given):
  morgan_counts    Morgan (ECFP-like) atom-environment counts, radius ``radius`` (default 2), hashed to ``n_bits`` (2048)
  atompair_counts  atom-pair counts hashed to ``n_bits`` (2048)
  maccs            the 167 MACCS keys
  descriptors      every RDKit 2D descriptor (rdkit.Chem.Descriptors.descList); nan / inf -> 0, values clipped to +-1e6
  concat           morgan_counts + atompair_counts + maccs + descriptors (default; 4,480 columns for the defaults)
A SMILES that fails RDKit sanitisation is parsed without sanitisation; one that cannot be parsed at all gets an all-zero
row and valid False (it is left out of training and receives the median score of its query set).

TreeEnsemble: a class-balanced entropy RandomForest and ExtraTrees (n_trees each, default 250, min_samples_leaf=2,
seeded); predict returns the mean of the two predicted probabilities. Measured CPU time on one thread: featurising
1,000 molecules with ``concat`` takes about 11 s, fitting on 4,000 molecules x 4,480 columns takes about 22 s
(250 trees per model), scoring 1,000 molecules takes under 1 s; ``n_jobs`` (at most 2) splits the work over two workers.
Everything is deterministic given ``seed``.
"""
from __future__ import annotations

import functools
import warnings

import numpy as np

__all__ = ["KINDS", "featurize", "feature_names", "scaffold_groups", "TreeEnsemble", "fit_predict", "grouped_cv_auc"]

KINDS = ("morgan_counts", "atompair_counts", "maccs", "descriptors")
_CONCAT = KINDS
_MAX_JOBS = 2
_CLIP = 1e6
_PARALLEL_MIN = 200            # fewer molecules than this are featurised in-process (no worker start-up cost)


# ------------------------------------------------------------------------------------------------ RDKit plumbing
def _rdkit():
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")


def _parse(smiles):
    """RDKit molecule for a SMILES; unsanitisable structures (e.g. hypervalent metal complexes) are kept unsanitised."""
    from rdkit import Chem
    _rdkit()
    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        m = Chem.MolFromSmiles(str(smiles), sanitize=False)
        if m is not None:
            try:
                m.UpdatePropertyCache(strict=False)
                Chem.FastFindRings(m)
            except Exception:
                return None
    return m if m is not None and m.GetNumAtoms() > 0 else None


@functools.lru_cache(maxsize=8)
def _generators(radius: int, n_bits: int):
    from rdkit.Chem import rdFingerprintGenerator as G
    return G.GetMorganGenerator(radius=radius, fpSize=n_bits), G.GetAtomPairGenerator(fpSize=n_bits)


@functools.lru_cache(maxsize=1)
def _descriptor_functions():
    from rdkit.Chem import Descriptors
    return tuple(Descriptors.descList)


def _kinds_tuple(kinds) -> tuple[str, ...]:
    ks = [kinds] if isinstance(kinds, str) else list(kinds)
    out: list[str] = []
    for k in ks:
        if k == "concat":
            out.extend(_CONCAT)
        elif k in KINDS:
            out.append(k)
        else:
            raise ValueError(f"unknown feature kind {k!r}; use one or a list of {list(KINDS) + ['concat']}")
    if not out:
        raise ValueError("kinds is empty")
    return tuple(out)


def _check_fp_args(radius: int, n_bits: int) -> tuple[int, int]:
    radius, n_bits = int(radius), int(n_bits)
    if not (0 <= radius <= 6 and 16 <= n_bits <= 16384):
        raise ValueError("need 0 <= radius <= 6 and 16 <= n_bits <= 16384")
    return radius, n_bits


def feature_names(kinds="concat", radius: int = 2, n_bits: int = 2048) -> list[str]:
    """Column names of ``featurize(..., kinds, radius, n_bits)``."""
    radius, n_bits = _check_fp_args(radius, n_bits)
    names: list[str] = []
    for k in _kinds_tuple(kinds):
        if k == "morgan_counts":
            names += [f"morgan{radius}_{j}" for j in range(n_bits)]
        elif k == "atompair_counts":
            names += [f"atompair_{j}" for j in range(n_bits)]
        elif k == "maccs":
            names += [f"maccs_{j}" for j in range(167)]
        else:
            names += [n for n, _ in _descriptor_functions()]
    return names


def _row(m, kinds: tuple[str, ...], radius: int, n_bits: int) -> np.ndarray:
    morgan, apair = _generators(radius, n_bits)
    parts = []
    for k in kinds:
        if k == "morgan_counts":
            parts.append(morgan.GetCountFingerprintAsNumPy(m))
        elif k == "atompair_counts":
            parts.append(apair.GetCountFingerprintAsNumPy(m))
        elif k == "maccs":
            from rdkit.Chem import MACCSkeys
            parts.append(np.array(list(MACCSkeys.GenMACCSKeys(m))))
        else:
            d = []
            for _, f in _descriptor_functions():
                try:
                    d.append(float(f(m)))
                except Exception:
                    d.append(0.0)
            parts.append(np.array(d))
    return np.nan_to_num(np.concatenate(parts).astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def _featurize_chunk(smiles: list, kinds: tuple[str, ...], radius: int, n_bits: int, width: int):
    X = np.zeros((len(smiles), width), dtype=np.float32)
    valid = np.zeros(len(smiles), dtype=bool)
    for i, s in enumerate(smiles):
        m = _parse(s)
        if m is None:
            continue
        try:
            X[i] = np.clip(_row(m, kinds, radius, n_bits), -_CLIP, _CLIP)
            valid[i] = True
        except Exception:               # RDKit raises assorted C++ errors on odd structures: zero row, valid False
            X[i] = 0.0
    return X, valid


def _jobs(n_jobs) -> int:
    return max(1, min(int(n_jobs), _MAX_JOBS))


def featurize(smiles, kinds="concat", radius: int = 2, n_bits: int = 2048, n_jobs: int = 2):
    """RDKit features of a list of SMILES -> ``(X, valid, names)``: X float32 (n, d) with only finite values, valid[i] False
    (row all zeros) if RDKit cannot build molecule i, names = column names. See the module docstring for ``kinds``."""
    smiles = [str(s) for s in smiles]
    kinds = _kinds_tuple(kinds)
    radius, n_bits = _check_fp_args(radius, n_bits)
    names = feature_names(kinds, radius, n_bits)
    width, n_jobs = len(names), _jobs(n_jobs)
    if n_jobs == 1 or len(smiles) < _PARALLEL_MIN:
        X, valid = _featurize_chunk(smiles, kinds, radius, n_bits, width)
        return X, valid, names
    from joblib import Parallel, delayed
    size = max(50, -(-len(smiles) // (4 * n_jobs)))
    chunks = [smiles[i:i + size] for i in range(0, len(smiles), size)]
    res = Parallel(n_jobs=n_jobs)(delayed(_featurize_chunk)(c, kinds, radius, n_bits, width) for c in chunks)
    return np.vstack([r[0] for r in res]), np.concatenate([r[1] for r in res]), names


def scaffold_groups(smiles) -> np.ndarray:
    """Integer id of the Bemis-Murcko scaffold (RDKit ``MurckoScaffoldSmiles``) of every SMILES, numbered in order of first
    appearance; molecules without a ring share the id of the empty scaffold; every unparsable SMILES gets its own id."""
    from rdkit.Chem.Scaffolds import MurckoScaffold
    _rdkit()
    ids: dict[str, int] = {}
    out = np.zeros(len(smiles), dtype=np.int64)
    for i, s in enumerate(smiles):
        m = _parse(s)
        try:
            key = MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m is not None else None
        except Exception:
            key = None
        if key is None:
            key = f"<unparsable {i}>"
        out[i] = ids.setdefault(key, len(ids))
    return out


# ------------------------------------------------------------------------------------------------ models
def _is_smiles(x) -> bool:
    return not isinstance(x, np.ndarray) and len(x) > 0 and isinstance(x[0], (str, np.str_))


def _as_matrix(x, kinds, radius, n_bits, n_jobs) -> tuple[np.ndarray, np.ndarray]:
    """(X float32 (n, d), valid (n,)) from a SMILES list or a 2-D feature matrix (non-finite entries -> 0)."""
    if _is_smiles(x):
        X, valid, _ = featurize(x, kinds, radius, n_bits, n_jobs)
        return X, valid
    X = np.asarray(x, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("expected a list of SMILES strings or a 2-D feature matrix")
    return np.clip(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), -_CLIP, _CLIP).astype(np.float32), np.ones(len(X), bool)


class TreeEnsemble:
    """Entropy RandomForest + ExtraTrees (class_weight="balanced_subsample", min_samples_leaf) on a feature matrix.

    ``fit(X, y)`` needs both classes in y. ``predict(X)`` -> float64 (n,) in [0, 1]: mean of the two models' predicted
    probabilities of label 1 (a molecule's score does not depend on which other molecules are scored with it)."""

    def __init__(self, n_trees: int = 250, min_samples_leaf: int = 2, seed: int = 0, n_jobs: int = 2):
        self.n_trees, self.min_samples_leaf, self.seed, self.n_jobs = int(n_trees), int(min_samples_leaf), int(seed), _jobs(n_jobs)

    def fit(self, X, y) -> "TreeEnsemble":
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        X, y = np.asarray(X), np.asarray(y).astype(int)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2-D with one row per label")
        if len(set(y.tolist())) < 2:
            raise ValueError("y must contain both classes 0 and 1")
        kw = dict(n_estimators=self.n_trees, min_samples_leaf=self.min_samples_leaf, class_weight="balanced_subsample",
                  n_jobs=self.n_jobs, random_state=self.seed)
        self.models_ = [RandomForestClassifier(criterion="entropy", **kw).fit(X, y), ExtraTreesClassifier(**kw).fit(X, y)]
        self.classes_ = self.models_[0].classes_
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X)
        col = list(self.classes_).index(1)
        p = np.mean([m.predict_proba(X)[:, col] for m in self.models_], axis=0)
        return np.clip(np.nan_to_num(p, nan=0.5), 0.0, 1.0)


def _fill_invalid(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Unparsable molecules get the median score of the parsable ones (0.5 if there are none)."""
    out = np.array(scores, dtype=np.float64)
    out[~valid] = float(np.median(out[valid])) if valid.any() else 0.5
    return out


def fit_predict(train, y, *queries, kinds="concat", radius: int = 2, n_bits: int = 2048, n_trees: int = 250,
                min_samples_leaf: int = 2, seed: int = 0, n_jobs: int = 2):
    """Fit ``TreeEnsemble`` on the labelled molecules (``train``: SMILES list or feature matrix, ``y``: 0/1) and score
    each query set (SMILES list or feature matrix with the same columns) -> array in [0, 1] per set (list if several).
    Training rows that RDKit cannot parse are dropped; unparsable query SMILES get their set's median score."""
    if not queries:
        raise ValueError("give at least one query set to score")
    y = np.asarray(y).astype(int)
    Xtr, vtr = _as_matrix(train, kinds, radius, n_bits, n_jobs)
    if len(Xtr) != len(y):
        raise ValueError(f"train has {len(Xtr)} rows but y has {len(y)}")
    model = TreeEnsemble(n_trees, min_samples_leaf, seed, n_jobs).fit(Xtr[vtr], y[vtr])
    outs = []
    for q in queries:
        Xq, vq = _as_matrix(q, kinds, radius, n_bits, n_jobs)
        if Xq.shape[1] != Xtr.shape[1]:
            raise ValueError(f"query has {Xq.shape[1]} columns, train has {Xtr.shape[1]}")
        s = model.predict(Xq) if len(Xq) else np.zeros(0)
        outs.append(_fill_invalid(s, vq) if len(Xq) else s)
    return outs[0] if len(outs) == 1 else outs


def grouped_cv_auc(train, y, groups="scaffold", n_splits: int = 5, kinds="concat", radius: int = 2, n_bits: int = 2048,
                   n_trees: int = 100, min_samples_leaf: int = 2, seed: int = 0, n_jobs: int = 2) -> dict:
    """Out-of-fold ROC-AUC of ``TreeEnsemble`` on the labelled molecules. Folds are stratified by label and never split a
    group; ``groups`` = "scaffold" (needs SMILES), None (plain stratified folds) or one group id per row.
    Returns {"auc": pooled out-of-fold ROC-AUC, "fold_aucs": per-fold AUCs, "oof": (n,) out-of-fold scores,
    "fold": (n,) fold index of every row, "groups": group ids or None}. Rows RDKit cannot parse are left out of the
    folds and the AUC (oof NaN, fold -1)."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    y = np.asarray(y).astype(int)
    X, valid = _as_matrix(train, kinds, radius, n_bits, n_jobs)
    if len(X) != len(y):
        raise ValueError(f"train has {len(X)} rows but y has {len(y)}")
    if isinstance(groups, str):
        if groups != "scaffold":
            raise ValueError('groups must be "scaffold", None or one group id per row')
        if not _is_smiles(train):
            raise ValueError('groups="scaffold" needs SMILES strings in train; pass explicit group ids for a feature matrix')
        g = scaffold_groups(train)
    elif groups is None:
        g = None
    else:
        g = np.asarray(groups)
        if len(g) != len(y):
            raise ValueError("groups needs one id per training row")
    keep = np.flatnonzero(valid)
    Xk, yk = X[keep], y[keep]
    oof = np.full(len(y), np.nan)
    fold = np.full(len(y), -1, dtype=int)
    fold_aucs: list[float] = []
    if g is None:
        splits = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(Xk, yk)
    else:
        splits = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(Xk, yk, g[keep])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k, (a, b) in enumerate(splits):
            fold[keep[b]] = k
            oof[keep[b]] = TreeEnsemble(n_trees, min_samples_leaf, seed, n_jobs).fit(Xk[a], yk[a]).predict(Xk[b])
            if len(set(yk[b].tolist())) == 2:
                fold_aucs.append(float(roc_auc_score(yk[b], oof[keep[b]])))
    return {"auc": float(roc_auc_score(yk, oof[keep])), "fold_aucs": fold_aucs, "oof": oof, "fold": fold, "groups": g}
