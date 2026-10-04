"""Deterministic, label-isolated regression helper tests for FoR51."""
from __future__ import annotations

import numpy as np
import pytest

from scilib import matphonon_regression as mr


def _data(seed: int = 4, n: int = 48, d: int = 6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    # Positive nonlinear target; the caller supplies labels and the module has
    # no access to this construction or to an evaluation target.
    y = np.exp(2.0 + 0.45 * X[:, 0] - 0.25 * X[:, 1] + 0.05 * rng.normal(size=n))
    return X, y


def test_fit_predict_is_deterministic_and_reports_provenance():
    X, y = _data()
    a = mr.fit_predict(X[:36], y[:36], X[36:], n_estimators=32, seeds=(0, 3), return_info=True)
    b = mr.fit_predict(X[:36], y[:36], X[36:], n_estimators=32, seeds=(0, 3), return_info=True)
    assert np.array_equal(a["predictions"], b["predictions"])
    assert a["predictions"].shape == (12,) and np.isfinite(a["predictions"]).all()
    assert a["provenance"]["dataset_access"] is False
    assert a["provenance"]["target_transform"] == "log"
    assert a["provenance"]["training_overlap"].startswith("unknown")


def test_cross_validate_is_training_only_and_quantitative():
    X, y = _data()
    out = mr.cross_validate(X, y, folds=4, seed=9, n_estimators=32, seeds=(0, 3))
    assert len(out["fold_mae"]) == 4 and len(out["fold_sizes"]) == 4
    assert out["mae"] == pytest.approx(np.average(out["fold_mae"], weights=out["fold_sizes"]))
    assert np.isfinite(out["mae"]) and out["mae"] > 0
    assert out["provenance"]["folds"] == 4 and out["provenance"]["dataset_access"] is False


def test_hist_gradient_boosting_is_deterministic_and_label_isolated():
    X, y = _data(n=32)
    a = mr.fit_hist_predict(X[:24], y[:24], X[24:], max_iter=25, seeds=(0, 1), return_info=True)
    b = mr.fit_hist_predict(X[:24], y[:24], X[24:], max_iter=25, seeds=(0, 1), return_info=True)
    assert np.array_equal(a["predictions"], b["predictions"])
    assert a["provenance"]["model"].endswith("HistGradientBoostingRegressor")
    assert a["provenance"]["dataset_access"] is False


def test_shape_target_and_fold_checks_are_strict():
    X, y = _data(n=12)
    with pytest.raises(ValueError, match="different row counts"):
        mr.fit_predict(X, y[:-1], X[:2], n_estimators=4, seeds=(0,))
    with pytest.raises(ValueError, match="positive frequencies"):
        mr.fit_predict(X, np.r_[y[:-1], 0.0], X[:2], n_estimators=4, seeds=(0,))
    with pytest.raises(ValueError, match="folds"):
        mr.cross_validate(X, y, folds=13, n_estimators=4, seeds=(0,))
