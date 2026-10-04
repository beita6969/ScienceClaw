"""Label-free structure-feature regression for the Matbench phonon target.

``fit_predict(X_train, y_train, X_eval)`` fits an ensemble of scikit-learn
``ExtraTreesRegressor`` models to the arrays supplied by the caller.  The
positive frequency target is fitted in ``log`` space and transformed back to
cm^-1 for predictions.  Features are treated as structure-only inputs: this
module never reads a dataset, target column, or network resource.

``cross_validate(X, y)`` performs shuffled K-fold evaluation using only the
training arrays and reports fold MAE in cm^-1.  Both functions use the same
fixed defaults and are deterministic for a fixed ``seeds`` tuple.  The
returned diagnostics record model and input provenance; dataset overlap is
unknown to this array-only module and must be reported by the caller.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = ["fit_predict", "fit_hist_predict", "cross_validate", "PROVENANCE"]

PROVENANCE = {
    "model": "sklearn.ensemble.ExtraTreesRegressor",
    "target_transform": "log",
    "feature_source": "caller supplied structure-only descriptors",
    "dataset_access": False,
    "training_overlap": "unknown; inspect caller dataset provenance",
}


def _xy(X: Any, y: Any | None = None, *, xname: str = "X") -> tuple[np.ndarray, np.ndarray | None]:
    a = np.asarray(X, dtype=float)
    if a.ndim != 2 or not a.shape[0] or not a.shape[1]:
        raise ValueError(f"{xname} must be a non-empty 2-D array")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{xname} contains non-finite values")
    if y is None:
        return a, None
    b = np.asarray(y, dtype=float).reshape(-1)
    if b.shape[0] != a.shape[0]:
        raise ValueError(f"X and y have different row counts ({a.shape[0]} vs {b.shape[0]})")
    if not np.all(np.isfinite(b)) or np.any(b <= 0):
        raise ValueError("y must contain finite positive frequencies")
    return a, b


def _impute(train: np.ndarray, *queries: np.ndarray) -> tuple[np.ndarray, ...]:
    """Replace non-finite descriptor values using training-column medians.

    ``_xy`` rejects non-finite values for direct calls.  This helper is kept
    separate so CV and callers can explicitly opt into a reproducible imputer
    by passing finite arrays after their own preprocessing.
    """
    med = np.nanmedian(train, axis=0)
    med[~np.isfinite(med)] = 0.0
    out = [np.where(np.isfinite(z), z, med) for z in (train,) + queries]
    return tuple(np.asarray(z, dtype=float) for z in out)


def _fit_one(X: np.ndarray, y: np.ndarray, Q: np.ndarray, *, seed: int, n_estimators: int,
             max_features: float | int | str, min_samples_leaf: int, n_jobs: int) -> np.ndarray:
    from sklearn.ensemble import ExtraTreesRegressor

    model = ExtraTreesRegressor(
        n_estimators=int(n_estimators),
        max_features=max_features,
        min_samples_leaf=int(min_samples_leaf),
        random_state=int(seed),
        n_jobs=int(n_jobs),
    )
    model.fit(X, np.log(y))
    return np.exp(model.predict(Q))


def _predict(X: np.ndarray, y: np.ndarray, Q: np.ndarray, *, seeds: Sequence[int], n_estimators: int,
             max_features: float | int | str, min_samples_leaf: int, n_jobs: int,
             clip: tuple[float, float] | None) -> np.ndarray:
    if not seeds:
        raise ValueError("seeds must contain at least one integer")
    preds = np.mean([_fit_one(X, y, Q, seed=int(s), n_estimators=n_estimators,
                              max_features=max_features, min_samples_leaf=min_samples_leaf,
                              n_jobs=n_jobs) for s in seeds], axis=0)
    if clip is not None:
        lo, hi = (float(clip[0]), float(clip[1]))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo < 0 or hi <= lo:
            raise ValueError("clip must be a finite increasing non-negative (lo, hi) pair")
        preds = np.clip(preds, lo, hi)
    return np.asarray(preds, dtype=float)


def fit_predict(X_train: Any, y_train: Any, X_eval: Any, *, n_estimators: int = 300,
                max_features: float | int | str = 0.5, min_samples_leaf: int = 1,
                seeds: Sequence[int] = (0, 1, 2), n_jobs: int = 1,
                clip: tuple[float, float] | None = (0.0, 5000.0),
                return_info: bool = False) -> np.ndarray | dict[str, Any]:
    """Fit on labelled source rows and predict evaluation rows.

    ``X_eval`` is never used to fit preprocessing or targets.  Set
    ``return_info=True`` to obtain predictions plus the reproducibility and
    overlap fields in :data:`PROVENANCE`.
    """
    if int(n_estimators) < 1 or int(min_samples_leaf) < 1 or int(n_jobs) == 0:
        raise ValueError("n_estimators/min_samples_leaf must be >= 1 and n_jobs must be nonzero")
    X, y = _xy(X_train, y_train)
    Q, _ = _xy(X_eval, xname="X_eval")
    if Q.shape[1] != X.shape[1]:
        raise ValueError(f"X_eval has {Q.shape[1]} features; X_train has {X.shape[1]}")
    pred = _predict(X, y, Q, seeds=seeds, n_estimators=int(n_estimators), max_features=max_features,
                    min_samples_leaf=int(min_samples_leaf), n_jobs=int(n_jobs), clip=clip)
    if not return_info:
        return pred
    return {
        "predictions": pred,
        "provenance": {**PROVENANCE, "n_train": int(X.shape[0]), "n_features": int(X.shape[1]),
                        "seeds": [int(s) for s in seeds], "n_estimators": int(n_estimators),
                        "min_samples_leaf": int(min_samples_leaf), "clip": None if clip is None else list(clip)},
    }


def fit_hist_predict(X_train: Any, y_train: Any, X_eval: Any, *, max_iter: int = 250,
                     learning_rate: float = 0.03, max_leaf_nodes: int = 31,
                     l2_regularization: float = 2.0, seeds: Sequence[int] = (0, 1),
                     return_info: bool = False) -> np.ndarray | dict[str, Any]:
    """Fit a deterministic log-target HistGradientBoosting ensemble.

    This is a second CPU-only structure regressor, useful when ExtraTrees over-smooths the phonon target.  It fits
    only on the labelled source arrays; the evaluation descriptors are used for prediction only.  The defaults were
    selected on training-only cross-validation and are intentionally modest enough for one episode's 676 structures.
    """
    if int(max_iter) < 1 or int(max_leaf_nodes) < 2 or float(learning_rate) <= 0 or float(l2_regularization) < 0:
        raise ValueError("max_iter/max_leaf_nodes must be positive and learning_rate/l2_regularization valid")
    if not seeds:
        raise ValueError("seeds must contain at least one integer")
    X, y = _xy(X_train, y_train)
    Q, _ = _xy(X_eval, xname="X_eval")
    if Q.shape[1] != X.shape[1]:
        raise ValueError(f"X_eval has {Q.shape[1]} features; X_train has {X.shape[1]}")
    from sklearn.ensemble import HistGradientBoostingRegressor

    pred = []
    for seed in seeds:
        m = HistGradientBoostingRegressor(max_iter=int(max_iter), learning_rate=float(learning_rate),
                                          max_leaf_nodes=int(max_leaf_nodes),
                                          l2_regularization=float(l2_regularization), random_state=int(seed))
        m.fit(X, np.log(y))
        pred.append(np.exp(m.predict(Q)))
    out = np.mean(pred, axis=0).astype(float)
    if not return_info:
        return out
    return {"predictions": out,
            "provenance": {**PROVENANCE, "model": "sklearn.ensemble.HistGradientBoostingRegressor",
                           "target_transform": "log", "n_train": int(X.shape[0]),
                           "n_features": int(X.shape[1]), "seeds": [int(s) for s in seeds],
                           "max_iter": int(max_iter), "learning_rate": float(learning_rate),
                           "max_leaf_nodes": int(max_leaf_nodes), "l2_regularization": float(l2_regularization)}}


def cross_validate(X: Any, y: Any, *, folds: int = 5, seed: int = 0, n_estimators: int = 300,
                   max_features: float | int | str = 0.5, min_samples_leaf: int = 1,
                   seeds: Sequence[int] = (0, 1, 2), n_jobs: int = 1,
                   clip: tuple[float, float] | None = (0.0, 5000.0)) -> dict[str, Any]:
    """Shuffled K-fold MAE, with every fold fit on its training rows only."""
    X, y = _xy(X, y)
    if int(folds) < 2 or int(folds) > X.shape[0]:
        raise ValueError(f"folds must be between 2 and n_train ({X.shape[0]})")
    from sklearn.model_selection import KFold

    splitter = KFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    fold_mae: list[float] = []
    counts: list[int] = []
    for tr, te in splitter.split(X):
        pred = _predict(X[tr], y[tr], X[te], seeds=seeds, n_estimators=int(n_estimators),
                        max_features=max_features, min_samples_leaf=int(min_samples_leaf),
                        n_jobs=int(n_jobs), clip=clip)
        fold_mae.append(float(np.mean(np.abs(pred - y[te]))))
        counts.append(int(te.size))
    return {
        "mae": float(np.average(fold_mae, weights=counts)),
        "fold_mae": fold_mae,
        "fold_sizes": counts,
        "provenance": {**PROVENANCE, "n_train": int(X.shape[0]), "n_features": int(X.shape[1]),
                        "folds": int(folds), "split_seed": int(seed), "seeds": [int(s) for s in seeds],
                        "n_estimators": int(n_estimators), "min_samples_leaf": int(min_samples_leaf),
                        "clip": None if clip is None else list(clip)},
    }
