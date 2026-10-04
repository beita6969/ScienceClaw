"""scilib.causal: interface text, sandbox import, estimator correctness on synthetic confounded data, determinism."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scilib
from scilib import causal as C
from scienceclaw.runtime.integrity import scan_code


@pytest.fixture(autouse=True, scope="module")
def _two_threads():
    """The code-node sandbox limits BLAS/OpenMP to 2 threads; do the same here (also keeps the tests fast on a loaded machine)."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        yield


def _data(n: int = 700, seed: int = 0, effect: float = 2.0, hetero: float = 0.0, confound: float = 1.0):
    """Confounded observational data with linear surfaces: y = 1 + 2 x0 - x1 + z (effect + hetero x0) + noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    z = (rng.random(n) < 1 / (1 + np.exp(-confound * X[:, 0]))).astype(int)
    tau = effect + hetero * X[:, 0]
    y = 1 + 2 * X[:, 0] - X[:, 1] + z * tau + rng.normal(0, 0.5, n)
    return X, z, y, tau


def _cov_table(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x_1": rng.integers(20, 40, n), "x_2": rng.choice(["A", "B", "C"], n), "x_3": rng.normal(size=n),
                         "x_4": rng.integers(0, 2, n).astype(bool), "x_5": np.ones(n), "x_6": rng.choice(["u", "v"], n)})


def test_describe_lists_every_public_name():
    text = scilib.describe("causal")
    for name in C.__all__:
        assert name in text


def test_interface_text_is_factual():
    """F5: the policy-visible text names no reference baseline, margin or recommendation."""
    low = scilib.describe("causal").lower()
    for word in ("reference", "baseline", "margin", "accept", "best", "recommend", "should", "worst", "state of the art"):
        assert word not in low, word


def test_code_node_may_import_scilib():
    assert scan_code("from scilib.causal import estimate_effects\n\ndef run(inputs, config):\n    return {}\n") == []


def test_design_matrix_handles_mixed_types_and_is_standardised():
    cov = _cov_table()
    cov.loc[3, "x_3"] = np.nan
    cov["x_7"] = cov["x_6"].astype("category")
    X = C.design_matrix(cov)
    assert X.shape[0] == len(cov) and np.isfinite(X).all()
    assert np.allclose(X.mean(axis=0), 0, atol=1e-9) and np.allclose(X.std(axis=0), 1, atol=1e-9)
    # x_1 (int), x_2 -> 2 dummies, x_3, x_4 (bool), x_6 -> 1 dummy, x_7 -> 1 dummy; x_5 constant is dropped
    assert X.shape[1] == 1 + 2 + 1 + 1 + 1 + 1
    assert np.array_equal(C.design_matrix(cov), X)
    raw = C.design_matrix(cov, standardize=False)
    assert raw.shape == X.shape and raw[:, 0].min() >= 20


def test_design_matrix_accepts_numeric_matrix_and_numeric_strings():
    A = np.random.default_rng(0).normal(size=(50, 4))
    assert C.design_matrix(A).shape == (50, 4)
    df = pd.DataFrame({"a": ["1", "2", "3", "4"], "b": ["x", "y", "x", "y"]})
    assert C.design_matrix(df).shape == (4, 2)


def test_estimators_recover_a_confounded_constant_effect():
    X, z, y, tau = _data()
    assert abs(C.diff_means(z, y).tau - 2.0) > 1.0                                    # confounded
    for name in ("regression_adjustment", "impute_ridge", "tlearner_ridge", "xlearner_ridge", "aipw_ridge", "ipw_logit"):
        r = C.get_method(name)(X, z, y)
        tol = 0.6 if name == "ipw_logit" else 0.2
        assert isinstance(r, C.Effect) and r.tau == pytest.approx(2.0, abs=tol), (name, r)
        assert float(r) == r.tau and np.isfinite(r.se) and r.se > 0


def test_gradient_boosting_learners_run_and_are_deterministic():
    X, z, y, _ = _data(n=300)
    for lr in ("hgb", "lgbm", "rf", "extra", "ridge_hgb"):
        a = C.outcome_imputation(X, z, y, "att", lr)
        b = C.outcome_imputation(X, z, y, "att", lr)
        assert a == b and np.isfinite(a.tau) and a.se > 0, (lr, a)
        if lr in ("hgb", "lgbm", "ridge_hgb"):            # forests do not extrapolate a linear surface into the treated region
            assert abs(a.tau - 2.0) < 0.6, (lr, a)


def test_att_and_ate_differ_under_effect_heterogeneity():
    X, z, y, tau = _data(n=3000, hetero=1.0, effect=1.0, seed=3)
    att_true, ate_true = tau[z == 1].mean(), tau.mean()
    assert att_true - ate_true > 0.3
    for fam in ("impute", "tlearner", "xlearner", "aipw"):
        att = C.get_method(f"{fam}_ridge", "att")(X, z, y).tau
        ate = C.get_method(f"{fam}_ridge", "ate")(X, z, y).tau
        assert att == pytest.approx(att_true, abs=0.25), (fam, att, att_true)
        assert ate == pytest.approx(ate_true, abs=0.25), (fam, ate, ate_true)


def test_aipw_standard_error_is_calibrated_roughly():
    est = []
    for s in range(12):
        X, z, y, tau = _data(n=500, seed=100 + s)
        r = C.aipw_effect(X, z, y, "att", "ridge", n_folds=3)
        est.append((r.tau - tau[z == 1].mean(), r.se))
    err, se = np.array(est).T
    assert 0.3 < np.sqrt(np.mean(err ** 2)) / np.mean(se) < 3.0


def test_propensity_scores_are_clipped_and_crossfitted():
    X, z, y, _ = _data(confound=6.0)
    p = C.propensity_scores(X, z, "logit", clip=(0.05, 0.95))
    assert p.min() >= 0.05 and p.max() <= 0.95 and p.shape == z.shape
    assert np.array_equal(p, C.propensity_scores(X, z, "logit", clip=(0.05, 0.95)))
    assert C.propensity_scores(X[:300], z[:300], "hgb", n_folds=2).shape == (300,)


def test_overlap_report_flags_poor_overlap():
    good = C.overlap_report(*_data(confound=0.2)[:2])
    bad = C.overlap_report(*_data(confound=6.0)[:2])
    assert bad["frac_treated_above_control_q99"] > good["frac_treated_above_control_q99"]
    assert bad["ess_att_weights"] < good["ess_att_weights"] and bad["max_abs_smd"] > good["max_abs_smd"]
    for k in ("n", "n_treated", "treated_share", "propensity_q05_q50_q95_treated"):
        assert k in good


def test_estimate_effects_entry_point():
    cov = _cov_table(300, 1)
    X = C.design_matrix(cov)
    rng = np.random.default_rng(5)
    Z = (rng.random((3, 300)) < 0.4).astype(int)
    Y = np.stack([X[:, 0] + 1.5 * z + rng.normal(0, 0.3, 300) for z in Z])
    est = C.estimate_effects(cov, Z, Y, methods="impute_ridge")
    assert est.shape == (3,) and np.isfinite(est).all() and np.allclose(est, 1.5, atol=0.3)
    assert np.array_equal(est, C.estimate_effects(cov, Z, Y, methods="impute_ridge"))
    d = C.estimate_effects(cov, Z, Y, methods=["impute_ridge", "regression_adjustment", "diff_means"], combine="median", details=True)
    assert set(d["per_method"]) == {"impute_ridge", "regression_adjustment", "diff_means"} and d["estimate"].shape == (3,)
    assert np.allclose(d["estimate"], np.median(np.vstack(list(d["per_method"].values())), axis=0))
    assert C.estimate_effects(cov, Z[0], Y[0], methods="impute_ridge").shape == (1,)          # a single dataset


def test_estimate_effects_clips_and_falls_back():
    cov = _cov_table(200, 2)
    rng = np.random.default_rng(6)
    Z = (rng.random((2, 200)) < 0.5).astype(int)
    Y = rng.normal(size=(2, 200))

    def broken(X, z, y):
        raise RuntimeError("boom")

    est = C.estimate_effects(cov, Z, Y, methods=broken)                       # falls back to regression_adjustment
    X = C.design_matrix(cov)
    assert np.allclose(est, [C.regression_adjustment(X, z, y).tau for z, y in zip(Z, Y)], atol=1e-9)
    huge = C.estimate_effects(cov, Z, Y, methods=lambda X, z, y: 1e6, cap=2.0)
    assert np.allclose(huge, 2.0 * Y.std(axis=1, ddof=1))
    with pytest.raises(ValueError):
        C.estimate_effects(cov, Z, Y[:, :50])


def test_errors_are_informative():
    X, z, y, _ = _data(n=100)
    with pytest.raises(ValueError, match="single arm"):
        C.regression_adjustment(X, np.ones(100, int), y)
    with pytest.raises(ValueError, match="unknown method"):
        C.get_method("nope")
    with pytest.raises(ValueError, match="estimand"):
        C.outcome_imputation(X, z, y, "atc")
    with pytest.raises(ValueError, match="unknown learner"):
        C.fit_predict_outcome("svm", X, y, X)
    assert set(C.METHODS) >= {"diff_means", "regression_adjustment", "impute_hgb3", "aipw_hgb", "xlearner_lgbm", "ipw_logit"}


def test_combine_estimates():
    A = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 4.0], [10.0, 4.0]])
    assert np.allclose(C.combine_estimates(A, "mean"), [3.25, 2.5])
    assert np.allclose(C.combine_estimates(A, "median"), [1.5, 2.5])
    assert np.allclose(C.combine_estimates(A, "trimmed"), [1.5, 2.5])
    assert np.allclose(C.combine_estimates({"a": A[0], "b": A[1]}, "mean"), [0.5, 1.0])


def test_semi_synthetic_check_scores_methods_against_a_known_effect():
    X, z, y, _ = _data(n=500, seed=8)
    df = C.semi_synthetic_check(X, z, y, ["diff_means", "impute_ridge"], n_rep=2, surface="ridge")
    assert list(df.index) == ["diff_means", "impute_ridge"] and {"rmse_sd", "bias_sd", "seconds"} <= set(df.columns)
    assert df.loc["impute_ridge", "rmse_sd"] < df.loc["diff_means", "rmse_sd"]
    again = C.semi_synthetic_check(X, z, y, ["diff_means", "impute_ridge"], n_rep=2, surface="ridge")
    assert df[["rmse_sd", "bias_sd"]].equals(again[["rmse_sd", "bias_sd"]])


def test_validate_methods_reports_placebo_and_sorts():
    cov = _cov_table(400, 4)
    Xd = C.design_matrix(cov)
    rng = np.random.default_rng(9)
    Z = (rng.random((2, 400)) < 1 / (1 + np.exp(-Xd[:, 0]))).astype(int)
    Y = np.stack([2 * Xd[:, 0] + Xd[:, 1] + z + rng.normal(0, 0.4, 400) for z in Z])
    df = C.validate_methods(cov, Z, Y, ["diff_means", "impute_ridge", "regression_adjustment"], n_datasets=2, surface="ridge")
    assert list(df.columns) == ["rmse_sd", "bias_sd", "placebo_abs_sd", "seconds"]
    assert df["rmse_sd"].is_monotonic_increasing and df.index[0] != "diff_means"
    assert np.isfinite(df.to_numpy()).all()


needs_stochtree = pytest.mark.skipif(not C.have_module("stochtree"), reason="stochtree not installed")


@needs_stochtree
def test_bart_method_is_registered_and_recovers_the_effect():
    assert "bart" in C.METHODS
    X, z, y, tau = _data(n=400, hetero=1.0, effect=1.0, seed=5)
    att_true, ate_true = tau[z == 1].mean(), tau.mean()
    a = C.bart_effect(X, z, y, "att", n_trees=50, burn_in=100, draws=200)
    b = C.bart_effect(X, z, y, "ate", n_trees=50, burn_in=100, draws=200)
    assert isinstance(a, C.Effect) and np.isfinite(a.se) and a.se > 0
    assert a.tau == pytest.approx(att_true, abs=0.35) and b.tau == pytest.approx(ate_true, abs=0.35)


@needs_stochtree
def test_bart_is_deterministic_for_a_seed_and_differs_across_seeds():
    X, z, y, _ = _data(n=300, seed=6)
    kw = dict(n_trees=30, burn_in=50, draws=100)
    a, b = C.bart_effect(X, z, y, "att", seed=1, **kw), C.bart_effect(X, z, y, "att", seed=1, **kw)
    c = C.bart_effect(X, z, y, "att", seed=2, **kw)
    assert a == b and a.tau != c.tau


@needs_stochtree
def test_bart_inside_estimate_effects():
    X, z, y, _ = _data(n=300, seed=7)
    Z, Y = np.vstack([z, z]), np.vstack([y, y + 1.0])
    out = C.estimate_effects(X, Z, Y, methods=["bart"], details=True)
    assert out["estimate"].shape == (2,) and set(out["per_method"]) == {"bart"}
    assert out["estimate"][1] == pytest.approx(out["estimate"][0], abs=0.6)


@needs_stochtree
def test_bcf_recovers_att_and_ate_and_is_deterministic():
    assert "bcf" in C.METHODS
    X, z, y, tau = _data(n=400, hetero=1.0, effect=1.0, seed=8)
    kw = dict(burn_in=100, draws=200)
    a = C.bcf_effect(X, z, y, "att", **kw)
    b = C.bcf_effect(X, z, y, "ate", **kw)
    assert a == C.bcf_effect(X, z, y, "att", **kw) and np.isfinite(a.se) and a.se > 0
    assert a.tau == pytest.approx(tau[z == 1].mean(), abs=0.35) and b.tau == pytest.approx(tau.mean(), abs=0.35)
    assert float(C.get_method("bcf", "att", burn_in=50, draws=100)(X, z, y)) == pytest.approx(a.tau, abs=0.5)
