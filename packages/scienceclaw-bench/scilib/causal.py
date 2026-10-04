"""Tools for average-treatment-effect estimation from observational data with a binary treatment (ACIC-2016 style).

Conventions: X (n, p) numeric covariates, z (n,) 0/1 treatment, y (n,) outcome; estimand "att" = mean effect over the
treated units, "ate" = mean effect over all units. Every estimator returns Effect(tau, se) (float(effect) = tau).

design_matrix(cov) -> X                          mixed-type table (numbers, strings, categories, bools; NaN filled) -> float matrix:
                                                 one-hot of non-numeric columns, constant columns dropped, z-scored
overlap_report(X, z) -> dict                     treated share, cross-fitted logistic propensity quantiles per arm, share of treated
                                                 above the 99th percentile of the controls' propensity, effective sample size of ATT
                                                 weights, largest standardised covariate mean difference
fit_predict_outcome(learner, X_tr, y_tr, X_te)   regression fit + prediction; learner in ridge | hgb | hgb3 | lgbm | rf | extra |
                                                 ridge_hgb or a callable (X_tr, y_tr, X_te) -> predictions
                                                 (hgb3 = mean of three HistGradientBoosting configurations; ridge_hgb = ridge plus
                                                 boosted trees on its residuals)
propensity_scores(X, z, model, n_folds, clip)    cross-fitted P(z = 1 | x), model in logit | hgb, clipped to `clip`
diff_means(z, y)                                 mean(y | z=1) - mean(y | z=0), Welch standard error
regression_adjustment(X, z, y)                   coefficient of z in the least-squares fit y ~ 1 + z + X (HC1 standard error)
outcome_imputation(X, z, y, estimand, learner)   control (and, for ATE, treated) outcome model per arm; the missing potential outcome of
                                                 every unit is predicted and the observed outcome is kept for the factual arm:
                                                 ATT = mean over treated of (y - mu0_hat(x)); se = sd/sqrt(n_treated) of those terms
t_learner(X, z, y, estimand, learner)            mean of mu1_hat(x) - mu0_hat(x) over the target units (arm models fitted on their arm)
x_learner(X, z, y, estimand, learner)            Kuenzel et al. X-learner: imputed effects regressed on x per arm, mixed with the propensity
bart_effect(X, z, y, estimand, n_trees, burn_in, draws, seed)
                                                 Hill (2011) BART imputation, sampler from the `stochtree` package (present only when that
                                                 package is importable; also listed in METHODS as "bart"): one Bayesian additive regression
                                                 tree model of y on (x, z) sampled by MCMC (n_trees 200, burn_in 300, draws 1000, one
                                                 chain, from the root, no warm start); the outcome of each unit under the other arm is the
                                                 posterior-mean prediction with z flipped and the observed outcome is kept for the factual
                                                 arm: ATT = mean over treated of (y - mu0_hat(x, 0)); ATE also averages (mu1_hat(x, 1) - y)
                                                 over controls; se = standard deviation of that mean over the posterior draws.
                                                 One fit on ~4,800 units x 80 columns takes about 12 s on one thread
bcf_effect(X, z, y, estimand, burn_in, draws, seed)
                                                 Bayesian causal forests (Hahn, Murray & Carvalho 2020), sampler from the `stochtree` package
                                                 (same availability as bart_effect; listed in METHODS as "bcf"): a prognostic forest (250
                                                 trees) and a treatment-effect forest (50 trees) with the cross-fitted logistic propensity
                                                 as an extra covariate of the prognostic forest, MCMC from the root (burn_in 300, draws
                                                 1000, one chain); the effect is the posterior mean of tau(x) averaged over the target
                                                 units (treated for ATT, all for ATE); se = standard deviation of that mean over the draws.
                                                 One fit on ~4,800 units x 80 columns takes about 30 s on one thread (extra threads gave no
                                                 speed-up in a test); burn_in 150 / draws 500 takes about 16 s
ipw_effect(X, z, y, estimand, propensity, clip)  Hajek inverse-propensity weighting with cross-fitted propensities
aipw_effect(X, z, y, estimand, learner, ...)     cross-fitted augmented IPW (outcome model + propensity), influence-function standard error
METHODS                                          valid method names for get_method / estimate_effects / validate_methods:
                                                 diff_means, regression_adjustment, and <family>_<learner> with family in impute | tlearner |
                                                 xlearner | aipw and learner as above (e.g. impute_hgb3, aipw_lgbm), plus ipw_logit, ipw_hgb
get_method(name, **kw) -> f(X, z, y) -> Effect   look up (and parametrise) a method
estimate_effects(cov, treatment, outcome, methods, estimand, cap, combine, details)
                                                 one entry point: cov table + (n_datasets, n) treatment and outcome matrices ->
                                                 one estimate per dataset (np.ndarray); methods = name | list of names/callables;
                                                 several methods are combined by `combine` (mean | median); non-finite values fall
                                                 back to regression_adjustment; estimates are clipped to +-cap * sd(y)
                                                 (defaults: methods=("impute_lgbm", "xlearner_lgbm"), estimand="att", cap=4.0, combine="mean")
combine_estimates(est, how)                      mean | median | trimmed over a (n_methods, n_datasets) array or dict of arrays
semi_synthetic_check(X, z, y, methods, ...)      plug-in simulation from ONE visible dataset: outcome surfaces are fitted to the
                                                 observed data, noise is resampled from residuals, the treatment vector is kept, so the
                                                 true effect of the simulated data is known; returns per-method RMSE/bias in sd(y) units
                                                 (the fitted-surface family is `surface`; results favour methods close to that family)
validate_methods(cov, treatment, outcome, methods, n_datasets, n_rep, ...)
                                                 semi_synthetic_check over the first datasets plus a permuted-treatment placebo
                                                 (true effect 0); returns a DataFrame indexed by method, sorted by semi-synthetic RMSE
"""
from __future__ import annotations

import time
import warnings
from functools import partial
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd

from ._pretrained import have_module

__all__ = ["Effect", "design_matrix", "overlap_report", "fit_predict_outcome", "propensity_scores", "diff_means",
           "regression_adjustment", "outcome_imputation", "t_learner", "x_learner", "ipw_effect", "aipw_effect", "bart_effect", "bcf_effect",
           "METHODS", "get_method", "estimate_effects", "combine_estimates", "semi_synthetic_check", "validate_methods"]

_SEED = 0
_HGB_CONFIGS = (
    dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0),
    dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=10, l2_regularization=1.0),
    dict(max_iter=300, learning_rate=0.10, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0),
)
LEARNERS = ("ridge", "hgb", "hgb3", "lgbm", "rf", "extra", "ridge_hgb")


class Effect(NamedTuple):
    """A point estimate and its standard error (nan when the method has none)."""
    tau: float
    se: float = float("nan")

    def __float__(self) -> float:
        return float(self.tau)


# ----------------------------------------------------------------------------------------------- preprocessing
def design_matrix(cov, standardize: bool = True) -> np.ndarray:
    """Float design matrix of a covariate table (DataFrame, 2-D array or dict of columns).

    Numeric and bool columns are kept (NaN -> column median); every other column (strings, categories, mixed) is one-hot coded
    with its first level dropped; constant columns are removed; with ``standardize`` each column is centred and scaled to unit SD.
    """
    df = pd.DataFrame(cov).reset_index(drop=True)
    cols: list[np.ndarray] = []
    for c in df.columns:
        s = df[c]
        v = None
        if pd.api.types.is_bool_dtype(s):
            v = s.to_numpy(dtype=float)
        elif pd.api.types.is_numeric_dtype(s):
            v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        else:
            try:
                v = pd.to_numeric(s, errors="raise").to_numpy(dtype=float)
            except (ValueError, TypeError):
                v = None
        if v is not None:
            if np.isnan(v).any():
                med = np.nanmedian(v) if not np.isnan(v).all() else 0.0
                v = np.where(np.isnan(v), med, v)
            cols.append(v)
        else:
            t = s.astype(str).fillna("nan") if hasattr(s, "fillna") else s.astype(str)
            for lev in sorted(t.unique())[1:]:
                cols.append((t == lev).to_numpy(dtype=float))
    if not cols:
        raise ValueError("design_matrix: the table has no columns")
    X = np.column_stack(cols).astype(float)
    sd = X.std(axis=0)
    X = X[:, sd > 1e-12]
    if standardize and X.shape[1]:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X


def _as_arrays(X, z, y=None):
    X = np.asarray(X, dtype=float)
    z = np.asarray(z).astype(int).ravel()
    if X.ndim != 2 or len(z) != len(X):
        raise ValueError(f"X must be (n, p) and z of length n, got X {X.shape}, z {z.shape}")
    if not np.isin(z, (0, 1)).all():
        raise ValueError("z must contain only 0 and 1")
    if z.min() == z.max():
        raise ValueError("z has a single arm: the effect is not identified")
    if y is None:
        return X, z
    y = np.asarray(y, dtype=float).ravel()
    if len(y) != len(z):
        raise ValueError("y and z differ in length")
    return X, z, y


def _folds(n: int, k: int, seed: int) -> np.ndarray:
    """Deterministic fold labels 0..k-1 (a seeded permutation)."""
    f = np.arange(n) % k
    return f[np.random.RandomState(seed).permutation(n)]


def _stratified_folds(z: np.ndarray, k: int, seed: int) -> np.ndarray:
    f = np.empty(len(z), dtype=int)
    for a in (0, 1):
        idx = np.flatnonzero(z == a)
        f[idx] = _folds(len(idx), k, seed + a)
    return f


# ----------------------------------------------------------------------------------------------------- models
def _single_thread():
    """Context manager: BLAS and OpenMP pools limited to one thread. The models here are fitted on a few thousand rows, where extra
    threads gain little, and on a shared or oversubscribed CPU their barriers can make one fit take seconds instead of
    milliseconds. Predictions are identical."""
    from threadpoolctl import threadpool_limits
    return threadpool_limits(limits=1)


def fit_predict_outcome(learner, X_train, y_train, X_test, seed: int = _SEED) -> np.ndarray:
    """Regress y_train on X_train with ``learner`` and predict X_test (1-D array); fits run on one thread.

    Learners: "ridge" (RidgeCV over 9 alphas); "hgb" (HistGradientBoosting, 200 iterations, lr 0.05, 15 leaves, l2 1); "hgb3" (mean of
    three HistGradientBoosting configurations); "lgbm" (LightGBM, 300 trees, lr 0.05, 15 leaves); "rf" / "extra" (200-tree
    random forest / extra-trees, min leaf 5, one third of the features per split); "ridge_hgb" (ridge, then "hgb" on its residuals);
    or a callable ``learner(X_train, y_train, X_test) -> predictions``.
    """
    if callable(learner):
        return _fit_predict_outcome(learner, X_train, y_train, X_test, seed)
    with _single_thread():
        return _fit_predict_outcome(learner, X_train, y_train, X_test, seed)


def _fit_predict_outcome(learner, X_train, y_train, X_test, seed: int = _SEED) -> np.ndarray:
    X_train, X_test = np.asarray(X_train, float), np.asarray(X_test, float)
    y_train = np.asarray(y_train, float).ravel()
    if callable(learner):
        return np.asarray(learner(X_train, y_train, X_test), dtype=float).ravel()
    if len(X_test) == 0:
        return np.zeros(0)
    if learner == "ridge":
        from sklearn.linear_model import RidgeCV
        return RidgeCV(alphas=np.logspace(-2, 4, 9)).fit(X_train, y_train).predict(X_test)
    if learner in ("hgb", "hgb3"):
        from sklearn.ensemble import HistGradientBoostingRegressor as H
        cfgs = _HGB_CONFIGS[:1] if learner == "hgb" else _HGB_CONFIGS
        return np.mean([H(random_state=seed, **c).fit(X_train, y_train).predict(X_test) for c in cfgs], axis=0)
    if learner == "lgbm":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20, reg_lambda=1.0,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, n_jobs=1, random_state=seed, verbose=-1)
        return m.fit(X_train, y_train).predict(X_test)
    if learner in ("rf", "extra"):
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
        cls = RandomForestRegressor if learner == "rf" else ExtraTreesRegressor
        return cls(n_estimators=200, min_samples_leaf=5, max_features=0.33, n_jobs=1, random_state=seed).fit(
            X_train, y_train).predict(X_test)
    if learner == "ridge_hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor as H
        from sklearn.linear_model import RidgeCV
        r = RidgeCV(alphas=np.logspace(-2, 4, 9)).fit(X_train, y_train)
        h = H(random_state=seed, **_HGB_CONFIGS[0]).fit(X_train, y_train - r.predict(X_train))
        return r.predict(X_test) + h.predict(X_test)
    raise ValueError(f"unknown learner {learner!r}; choose from {LEARNERS} or pass a callable")


def _fit_classifier_proba(model: str, X_tr, z_tr, X_te, seed: int) -> np.ndarray:
    if model in ("logit", "logistic"):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=0.05, max_iter=500).fit(X_tr, z_tr).predict_proba(X_te)[:, 1]
    if model == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier as H
        return H(max_iter=100, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=40, l2_regularization=1.0,
                 random_state=seed).fit(X_tr, z_tr).predict_proba(X_te)[:, 1]
    raise ValueError(f"unknown propensity model {model!r}; choose logit | hgb")


def propensity_scores(X, z, model: str = "logit", n_folds: int = 5, clip: tuple[float, float] = (0.02, 0.98),
                      seed: int = _SEED) -> np.ndarray:
    """Cross-fitted propensity P(z = 1 | x) (each unit predicted by a model fitted on the other folds), clipped to ``clip``.
    model: "logit" (L2-penalised logistic regression, C = 0.05, expects standardised X) or "hgb" (small boosted trees)."""
    X, z = _as_arrays(X, z)
    p = np.zeros(len(z))
    with _single_thread():
        if n_folds <= 1:
            p[:] = _fit_classifier_proba(model, X, z, X, seed)
        else:
            f = _stratified_folds(z, n_folds, seed)
            for k in range(n_folds):
                te = f == k
                p[te] = _fit_classifier_proba(model, X[~te], z[~te], X[te], seed)
    return np.clip(p, clip[0], clip[1])


def overlap_report(X, z, C: float = 0.05, n_folds: int = 5, seed: int = _SEED) -> dict:
    """Overlap diagnostics from a cross-fitted logistic propensity (visible data only): n, n_treated, treated_share,
    propensity_q05/q50/q95 for treated and controls, frac_treated_above_control_q99 (share of treated units whose propensity exceeds the
    99th percentile of the controls'), frac_controls_below_treated_q01, ess_att_weights (effective sample size of the controls under
    ATT weights e/(1-e)), max_abs_smd (largest absolute standardised mean difference between arms over the columns of X)."""
    from sklearn.linear_model import LogisticRegression
    X, z = _as_arrays(X, z)
    e = np.zeros(len(z))
    f = _stratified_folds(z, n_folds, seed)
    for k in range(n_folds):
        te = f == k
        e[te] = LogisticRegression(C=C, max_iter=500).fit(X[~te], z[~te]).predict_proba(X[te])[:, 1]
    et, ec = e[z == 1], e[z == 0]
    w = np.clip(ec, 1e-3, 1 - 1e-3) / (1 - np.clip(ec, 1e-3, 1 - 1e-3))
    sd = np.sqrt((X[z == 1].var(axis=0) + X[z == 0].var(axis=0)) / 2) + 1e-12
    smd = np.abs(X[z == 1].mean(axis=0) - X[z == 0].mean(axis=0)) / sd
    q = lambda a, p: float(np.quantile(a, p))
    return {"n": int(len(z)), "n_treated": int(z.sum()), "treated_share": float(z.mean()),
            "propensity_q05_q50_q95_treated": [q(et, .05), q(et, .5), q(et, .95)],
            "propensity_q05_q50_q95_control": [q(ec, .05), q(ec, .5), q(ec, .95)],
            "frac_treated_above_control_q99": float(np.mean(et > q(ec, .99))),
            "frac_controls_below_treated_q01": float(np.mean(ec < q(et, .01))),
            "ess_att_weights": float(w.sum() ** 2 / (w ** 2).sum()), "max_abs_smd": float(smd.max())}


# ----------------------------------------------------------------------------------------------- estimators
def diff_means(z, y) -> Effect:
    """mean(y | z=1) - mean(y | z=0) with the Welch standard error (ignores covariates; ATT and ATE coincide in expectation only
    under random assignment)."""
    z = np.asarray(z).astype(int).ravel()
    y = np.asarray(y, float).ravel()
    a, b = y[z == 1], y[z == 0]
    return Effect(float(a.mean() - b.mean()), float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))))


def regression_adjustment(X, z, y, estimand: str = "att") -> Effect:
    """Coefficient of z in the minimum-norm least-squares fit y ~ 1 + z + X, with an HC1 standard error. (One common coefficient for
    both estimands.)"""
    X, z, y = _as_arrays(X, z, y)
    A = np.column_stack([np.ones(len(y)), z.astype(float), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    try:
        bread = np.linalg.pinv(A.T @ A)
        meat = (A * res[:, None] ** 2).T @ A
        cov = bread @ meat @ bread * len(y) / max(len(y) - A.shape[1], 1)
        se = float(np.sqrt(max(cov[1, 1], 0.0)))
    except np.linalg.LinAlgError:
        se = float("nan")
    return Effect(float(coef[1]), se)


def _check_estimand(estimand: str) -> str:
    if estimand not in ("att", "ate"):
        raise ValueError("estimand must be 'att' or 'ate'")
    return estimand


def _arm_predictions(X, z, y, learner, seed, need_mu1: bool):
    """(mu0_hat, mu1_hat) over all units, each from the model fitted on its own arm (in-sample for that arm's units)."""
    mu0 = fit_predict_outcome(learner, X[z == 0], y[z == 0], X, seed)
    mu1 = fit_predict_outcome(learner, X[z == 1], y[z == 1], X, seed) if need_mu1 else None
    return mu0, mu1


def outcome_imputation(X, z, y, estimand: str = "att", learner="lgbm", seed: int = _SEED) -> Effect:
    """Imputation estimator: the missing potential outcome of a unit is predicted by the outcome model of the opposite arm and the
    observed outcome is kept for the factual arm. ATT = mean over treated of (y_i - mu0_hat(x_i)), mu0_hat fitted on controls only.
    ATE additionally fits mu1_hat on the treated and averages (mu1_hat(x_i) - y_i) over controls too. se = sd / sqrt(n) of the
    per-unit terms (a plain sample-mean error that ignores model uncertainty)."""
    _check_estimand(estimand)
    X, z, y = _as_arrays(X, z, y)
    mu0 = fit_predict_outcome(learner, X[z == 0], y[z == 0], X[z == 1], seed)
    d = y[z == 1] - mu0
    if estimand == "ate":
        mu1 = fit_predict_outcome(learner, X[z == 1], y[z == 1], X[z == 0], seed)
        d = np.concatenate([d, mu1 - y[z == 0]])
    return Effect(float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))))


def t_learner(X, z, y, estimand: str = "att", learner="lgbm", seed: int = _SEED) -> Effect:
    """T-learner: mu1_hat fitted on treated units, mu0_hat on controls; effect = mean of mu1_hat(x) - mu0_hat(x) over the target
    units (treated for ATT, all for ATE; each arm's own units are predicted in-sample by its model)."""
    _check_estimand(estimand)
    X, z, y = _as_arrays(X, z, y)
    mu0, mu1 = _arm_predictions(X, z, y, learner, seed, True)
    tgt = z == 1 if estimand == "att" else np.ones(len(z), bool)
    d = (mu1 - mu0)[tgt]
    return Effect(float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))))


def x_learner(X, z, y, estimand: str = "att", learner="lgbm", propensity: str = "logit", seed: int = _SEED) -> Effect:
    """X-learner (Kuenzel et al. 2019): imputed effects D1 = y - mu0_hat(x) on treated and D0 = mu1_hat(x) - y on controls are each
    regressed on x with ``learner`` (tau1_hat, tau0_hat); tau_hat(x) = g(x) tau0_hat(x) + (1 - g(x)) tau1_hat(x) with g the cross-fitted
    propensity; effect = mean of tau_hat over the target units."""
    _check_estimand(estimand)
    X, z, y = _as_arrays(X, z, y)
    t, c = z == 1, z == 0
    mu0_t = fit_predict_outcome(learner, X[c], y[c], X[t], seed)
    mu1_c = fit_predict_outcome(learner, X[t], y[t], X[c], seed)
    d1, d0 = y[t] - mu0_t, mu1_c - y[c]
    tau1 = fit_predict_outcome(learner, X[t], d1, X, seed)
    tau0 = fit_predict_outcome(learner, X[c], d0, X, seed)
    g = propensity_scores(X, z, propensity, seed=seed)
    tau = g * tau0 + (1 - g) * tau1
    d = tau[t] if estimand == "att" else tau
    return Effect(float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))))


def ipw_effect(X, z, y, estimand: str = "att", propensity: str = "logit", clip: tuple[float, float] = (0.02, 0.98),
               n_folds: int = 5, seed: int = _SEED) -> Effect:
    """Hajek (normalised) inverse-propensity weighting with cross-fitted, clipped propensities. ATT: treated weight 1, control weight
    e/(1-e); ATE: 1/e and 1/(1-e). se from the influence function with the weighted means plugged in."""
    _check_estimand(estimand)
    X, z, y = _as_arrays(X, z, y)
    e = propensity_scores(X, z, propensity, n_folds, clip, seed)
    t, c = z == 1, z == 0
    if estimand == "att":
        w1, w0 = np.ones(t.sum()), e[c] / (1 - e[c])
    else:
        w1, w0 = 1 / e[t], 1 / (1 - e[c])
    m1, m0 = np.sum(w1 * y[t]) / w1.sum(), np.sum(w0 * y[c]) / w0.sum()
    v1 = np.sum(w1 ** 2 * (y[t] - m1) ** 2) / w1.sum() ** 2
    v0 = np.sum(w0 ** 2 * (y[c] - m0) ** 2) / w0.sum() ** 2
    se = float(np.sqrt(v1 + v0))
    return Effect(float(m1 - m0), se)


def aipw_effect(X, z, y, estimand: str = "att", learner="lgbm", propensity: str = "logit",
                clip: tuple[float, float] = (0.02, 0.98), n_folds: int = 5, seed: int = _SEED) -> Effect:
    """Cross-fitted augmented IPW (doubly robust). Outcome models and propensities are fitted on K-1 folds and evaluated on the held-out
    fold. ATT: tau = mean_T(y - mu0) - (1/n_T) sum_{i in C} e_i/(1-e_i) (y_i - mu0_i); ATE: mean(mu1 - mu0 + z(y - mu1)/e -
    (1-z)(y - mu0)/(1-e)). se = sd of the estimated influence function / sqrt(n)."""
    _check_estimand(estimand)
    X, z, y = _as_arrays(X, z, y)
    n = len(y)
    e = propensity_scores(X, z, propensity, n_folds, clip, seed)
    mu0, mu1 = np.zeros(n), np.zeros(n)
    f = _stratified_folds(z, n_folds, seed)
    for k in range(n_folds):
        te, tr = f == k, f != k
        mu0[te] = fit_predict_outcome(learner, X[tr & (z == 0)], y[tr & (z == 0)], X[te], seed)
        if estimand == "ate":
            mu1[te] = fit_predict_outcome(learner, X[tr & (z == 1)], y[tr & (z == 1)], X[te], seed)
    if estimand == "att":
        p1 = z.mean()
        core = (z * (y - mu0) - (1 - z) * e / (1 - e) * (y - mu0)) / p1
        tau = core.mean()
        psi = core - z * tau / p1
    else:
        core = mu1 - mu0 + z * (y - mu1) / e - (1 - z) * (y - mu0) / (1 - e)
        tau = core.mean()
        psi = core - tau
    return Effect(float(tau), float(psi.std(ddof=1) / np.sqrt(n)))


def bart_effect(X, z, y, estimand: str = "att", n_trees: int = 200, burn_in: int = 300, draws: int = 1000,
                seed: int = _SEED) -> Effect:
    """BART imputation estimator (Hill 2011) with the stochtree sampler: one BART model of y on (x, z), MCMC from the root (``burn_in``
    discarded iterations, ``draws`` retained, one chain, one thread). The missing potential outcome of a unit is the posterior mean
    of the model with z set to the other arm, the observed outcome is kept for the factual arm. ATT = mean over treated of
    (y - mu0_hat(x)); ATE additionally averages (mu1_hat(x) - y) over the controls. se = standard deviation over the posterior draws
    of that mean."""
    _check_estimand(estimand)
    X, z, y = _as_arrays(X, z, y)
    try:
        from stochtree import BARTModel
    except ImportError as ex:
        raise RuntimeError("bart_effect needs the stochtree package") from ex
    t, c = z == 1, z == 0
    Xa = np.column_stack([X, z])
    parts = [np.column_stack([X[t], np.zeros(int(t.sum()))])]
    if estimand == "ate":
        parts.append(np.column_stack([X[c], np.ones(int(c.sum()))]))
    m = BARTModel()
    with _single_thread():
        m.sample(X_train=Xa, y_train=y, X_test=np.vstack(parts), num_gfr=0, num_burnin=int(burn_in), num_mcmc=int(draws),
                 general_params={"random_seed": int(seed), "num_threads": 1}, mean_forest_params={"num_trees": int(n_trees)})
    pred = np.asarray(m.y_hat_test, float)                     # (n_test, draws)
    nt = int(t.sum())
    contrast = y[t][:, None] - pred[:nt]
    if estimand == "ate":
        contrast = np.vstack([contrast, pred[nt:] - y[c][:, None]])
    per_draw = contrast.mean(axis=0)
    return Effect(float(contrast.mean()), float(per_draw.std(ddof=1)))


def bcf_effect(X, z, y, estimand: str = "att", burn_in: int = 300, draws: int = 1000, seed: int = _SEED,
               propensity: str = "logit") -> Effect:
    """Bayesian causal forests (Hahn, Murray & Carvalho 2020) with the stochtree sampler: y = mu(x, e_hat) + tau(x) z + noise with a
    prognostic forest mu (250 trees, covariates x and the cross-fitted propensity e_hat) and a treatment-effect forest tau (50 trees);
    MCMC from the root (``burn_in`` discarded iterations, ``draws`` retained, one chain, one thread). The effect is the posterior mean of
    tau(x) averaged over the target units (treated for ATT, all units for ATE); se = standard deviation over the posterior draws of
    that mean."""
    _check_estimand(estimand)
    X, z, y = _as_arrays(X, z, y)
    try:
        from stochtree import BCFModel
    except ImportError as ex:
        raise RuntimeError("bcf_effect needs the stochtree package") from ex
    e = propensity_scores(X, z, propensity, seed=seed)
    tgt = z == 1 if estimand == "att" else np.ones(len(z), bool)
    m = BCFModel()
    with _single_thread():
        m.sample(X_train=X, Z_train=z.astype(float), y_train=y, propensity_train=e, X_test=X[tgt], Z_test=np.ones(int(tgt.sum())),
                 propensity_test=e[tgt], num_gfr=0, num_burnin=int(burn_in), num_mcmc=int(draws),
                 general_params={"random_seed": int(seed), "num_threads": 1})
    tau = np.asarray(m.tau_hat_test, float)                    # (n_target, draws)
    return Effect(float(tau.mean()), float(tau.mean(axis=0).std(ddof=1)))


# ------------------------------------------------------------------------------------------------ registry
def _build_registry() -> dict[str, Callable]:
    reg: dict[str, Callable] = {"diff_means": lambda X, z, y, estimand="att", **kw: diff_means(z, y),
                                "regression_adjustment": regression_adjustment,
                                "ipw_logit": partial(ipw_effect, propensity="logit"),
                                "ipw_hgb": partial(ipw_effect, propensity="hgb")}
    for lr in LEARNERS:
        reg[f"impute_{lr}"] = partial(outcome_imputation, learner=lr)
        reg[f"tlearner_{lr}"] = partial(t_learner, learner=lr)
        reg[f"xlearner_{lr}"] = partial(x_learner, learner=lr)
        reg[f"aipw_{lr}"] = partial(aipw_effect, learner=lr)
    if have_module("stochtree"):
        reg["bart"] = bart_effect
        reg["bcf"] = bcf_effect
    return reg


_REGISTRY = _build_registry()
METHODS = tuple(_REGISTRY)


def get_method(name, estimand: str = "att", **kw) -> Callable:
    """``f(X, z, y) -> Effect`` for a method name in METHODS (extra keyword arguments override the method's parameters, e.g.
    ``get_method("aipw_hgb", n_folds=3, clip=(0.05, 0.95))``); a callable ``f(X, z, y)`` is returned unchanged."""
    if callable(name):
        return name
    if name not in _REGISTRY:
        raise ValueError(f"unknown method {name!r}; available: {', '.join(METHODS)}")
    base = _REGISTRY[name]
    return lambda X, z, y: base(X, z, y, estimand=estimand, **kw)


def combine_estimates(est, how: str = "mean") -> np.ndarray:
    """Combine estimates of several methods: ``est`` = dict name -> (n_datasets,) array or an (n_methods, n_datasets) array;
    how = "mean" | "median" | "trimmed" (mean without the smallest and largest method when at least 4 methods)."""
    A = np.vstack([np.asarray(v, float) for v in est.values()]) if isinstance(est, dict) else np.atleast_2d(np.asarray(est, float))
    if how == "mean":
        return A.mean(axis=0)
    if how == "median":
        return np.median(A, axis=0)
    if how == "trimmed":
        if A.shape[0] < 4:
            return A.mean(axis=0)
        return np.sort(A, axis=0)[1:-1].mean(axis=0)
    raise ValueError("how must be mean | median | trimmed")


def _method_list(methods) -> list:
    return [methods] if isinstance(methods, str) or callable(methods) else list(methods)


def _name(m) -> str:
    return m if isinstance(m, str) else getattr(m, "__name__", repr(m))


def estimate_effects(cov, treatment, outcome, methods=("impute_lgbm", "xlearner_lgbm"), estimand: str = "att", cap: float = 4.0,
                     combine: str = "mean", details: bool = False):
    """Effect estimate for every dataset. ``cov``: covariate table shared by the datasets (raw table or a numeric matrix); ``treatment``,
    ``outcome``: (n_datasets, n) arrays. ``methods``: a name from METHODS, a callable ``f(X, z, y) -> float | Effect`` or a list of them
    (combined across methods with ``combine``). Each estimate that is not finite (or a method that raises) is replaced by
    regression_adjustment; estimates are clipped to +-cap * sd(y) of their dataset. Returns a float array (n_datasets,); with
    ``details=True`` a dict {"estimate", "per_method" (name -> array), "se" (name -> array), "seconds"}."""
    _check_estimand(estimand)
    X = design_matrix(cov)
    Z, Y = np.atleast_2d(np.asarray(treatment)), np.atleast_2d(np.asarray(outcome, float))
    if Z.shape != Y.shape or Z.shape[1] != len(X):
        raise ValueError(f"treatment {Z.shape} and outcome {Y.shape} must be (n_datasets, {len(X)})")
    ms = _method_list(methods)
    fns = [get_method(m, estimand) for m in ms]
    names = [_name(m) for m in ms]
    est = np.full((len(fns), len(Z)), np.nan)
    se = np.full_like(est, np.nan)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j, (z, y) in enumerate(zip(Z, Y)):
            z = z.astype(int)
            for i, f in enumerate(fns):
                try:
                    r = f(X, z, y)
                    est[i, j], se[i, j] = (float(r.tau), float(r.se)) if isinstance(r, Effect) else (float(r), np.nan)
                except Exception:
                    est[i, j] = np.nan
    bad = ~np.isfinite(est)
    if bad.any():
        for i, j in zip(*np.nonzero(bad)):
            try:
                est[i, j] = regression_adjustment(X, Z[j].astype(int), Y[j]).tau
            except Exception:
                est[i, j] = 0.0
    lim = cap * Y.std(axis=1, ddof=1)
    est = np.clip(est, -lim, lim)
    out = combine_estimates(est, combine)
    if details:
        return {"estimate": out, "per_method": dict(zip(names, est)), "se": dict(zip(names, se)),
                "seconds": time.time() - t0}
    return out


# --------------------------------------------------------------------------------- validation on visible data
def _surface(X, z, y, surface, seed):
    """Fitted arm surfaces mu0_hat, mu1_hat over all units and resampled-noise pools (residuals of each arm)."""
    mu = []
    res = []
    for a in (0, 1):
        m = z == a
        # out-of-fold predictions of the arm's own units keep the noise pool honest (residuals are not shrunk by in-sample fit)
        f = _folds(int(m.sum()), 4, seed + a)
        oof = np.zeros(int(m.sum()))
        Xa, ya = X[m], y[m]
        for k in range(4):
            oof[f == k] = fit_predict_outcome(surface, Xa[f != k], ya[f != k], Xa[f == k], seed)
        res.append(ya - oof)
        mu.append(fit_predict_outcome(surface, Xa, ya, X, seed))
    return mu[0], mu[1], res[0], res[1]


def semi_synthetic_check(X, z, y, methods=("impute_lgbm",), n_rep: int = 2, estimand: str = "att", surface="extra",
                         seed: int = _SEED) -> pd.DataFrame:
    """Plug-in simulation from ONE visible dataset (no hidden information is used). Outcome surfaces mu0_hat / mu1_hat are fitted to the
    observed arms with ``surface`` (a learner name), the observed treatment vector and covariates are kept, and n_rep synthetic outcomes
    y_syn = z mu1_hat + (1 - z) mu0_hat + noise (noise resampled from out-of-fold residuals of the respective arm) are drawn. The true
    effect of the simulated data (mean of mu1_hat - mu0_hat over the target units) is known, so each method is scored on it.
    Returns a DataFrame indexed by method: rmse_sd, bias_sd (estimate - truth in units of sd(y_syn)), seconds (mean per run).
    The simulated surfaces are smoother/simpler than an unknown real surface; a method from the same family as ``surface`` is
    advantaged."""
    _check_estimand(estimand)
    X, z, y = _as_arrays(X, z, y)
    rng = np.random.RandomState(seed)
    mu0, mu1, r0, r1 = _surface(X, z, y, surface, seed)
    tgt = z == 1 if estimand == "att" else np.ones(len(z), bool)
    truth = float(np.mean((mu1 - mu0)[tgt]))
    ms = _method_list(methods)
    rows: dict[str, list] = {_name(m): [] for m in ms}
    secs: dict[str, list] = {_name(m): [] for m in ms}
    fns = {_name(m): get_method(m, estimand) for m in ms}
    for _ in range(n_rep):
        eps = np.where(z == 1, rng.choice(r1, len(z)), rng.choice(r0, len(z)))
        ys = np.where(z == 1, mu1, mu0) + eps
        sd = ys.std(ddof=1)
        for nm, f in fns.items():
            t0 = time.time()
            try:
                v = float(f(X, z, ys))
            except Exception:
                v = np.nan
            secs[nm].append(time.time() - t0)
            rows[nm].append((v - truth) / sd)
    out = pd.DataFrame({nm: {"rmse_sd": float(np.sqrt(np.nanmean(np.square(v)))) if np.isfinite(v).any() else np.nan,
                             "bias_sd": float(np.nanmean(v)) if np.isfinite(v).any() else np.nan,
                             "seconds": float(np.mean(secs[nm]))} for nm, v in ((k, np.array(vv)) for k, vv in rows.items())}).T
    return out


def validate_methods(cov, treatment, outcome, methods=("regression_adjustment", "impute_lgbm"), n_datasets: int = 3,
                     n_rep: int = 1, estimand: str = "att", surface="extra", seed: int = _SEED) -> pd.DataFrame:
    """Score ``methods`` on the visible data only: semi_synthetic_check on the first ``n_datasets`` datasets (n_rep draws each; rmse_sd is
    pooled over datasets and draws) plus a placebo on the same datasets (treatment vector randomly permuted, so the true effect is 0;
    placebo_abs_sd = mean |estimate| / sd(y)). Returns a DataFrame indexed by method with columns rmse_sd, bias_sd, placebo_abs_sd,
    seconds (mean per dataset), sorted by rmse_sd."""
    _check_estimand(estimand)
    X = design_matrix(cov)
    Z, Y = np.atleast_2d(np.asarray(treatment)), np.atleast_2d(np.asarray(outcome, float))
    ms = _method_list(methods)
    names = [_name(m) for m in ms]
    fns = {nm: get_method(m, estimand) for nm, m in zip(names, ms)}
    mse = {nm: [] for nm in names}
    bias = {nm: [] for nm in names}
    sec = {nm: [] for nm in names}
    plc = {nm: [] for nm in names}
    rng = np.random.RandomState(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j in range(min(n_datasets, len(Z))):
            z, y = Z[j].astype(int), Y[j]
            ss = semi_synthetic_check(X, z, y, ms, n_rep, estimand, surface, seed + j)
            zp = rng.permutation(z)
            for nm in names:
                mse[nm].append(ss.loc[nm, "rmse_sd"] ** 2)
                bias[nm].append(ss.loc[nm, "bias_sd"])
                sec[nm].append(ss.loc[nm, "seconds"])
                try:
                    plc[nm].append(abs(float(fns[nm](X, zp, y))) / y.std(ddof=1))
                except Exception:
                    plc[nm].append(np.nan)
    out = pd.DataFrame({nm: {"rmse_sd": float(np.sqrt(np.nanmean(mse[nm]))), "bias_sd": float(np.nanmean(bias[nm])),
                             "placebo_abs_sd": float(np.nanmean(plc[nm])), "seconds": float(np.mean(sec[nm]))}
                        for nm in names}).T
    return out.sort_values("rmse_sd")
