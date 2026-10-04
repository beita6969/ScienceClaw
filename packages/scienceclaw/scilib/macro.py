"""Annual macro-panel forecasting: a panel of economies x series kinds, several forecast steps ahead, scored by sMAPE.

Data layout (the ``load_train`` / ``load_eval_inputs`` / ``load_dev`` tool outputs): ``train_panel`` is a table with columns
``economy_id, region, income_level, indicator`` and one column per period ``t-31 .. t0`` (annual values, NaN = missing; one row per
economy and series kind); ``history`` is (n_items, L) with the item's series (oldest first), ``indicator`` the kind of each item,
``covariates`` (n_items, n_kinds, L) the item's economy over all kinds in the order of ``covariate_indicators``. A kind has a *role*:
'level' (strictly positive), 'percent' (in (0, 100]) or 'change' (annual percentage change, may be negative). Roles come from the
``indicator_kinds`` descriptions (the ``load_train`` port) and otherwise from the value range (any value <= 0 -> 'change'; all values
<= 100 -> 'percent'; else 'level'). Forecast steps are 1..horizon after the last period.

fit_predict(train_panel, history, indicator, indicator_kinds=None, covariates=None, covariate_indicators=None, economy_id=None,
            region=None, income_level=None, time_offsets=None, target_offsets=None, horizon=None, members=None, weights="equal",
            n_backtest=8, n_periods=None, seed=0, n_jobs=1, return_info=False) -> (n_items, horizon) array
    Whole pipeline: forecasts the log-ratio ln(y_{t+h} / y_t) of every item with the members below, combines them per kind with
    ``weights`` and returns level forecasts (positive; percent-role rows clipped to <= 100). Keyword names equal the tool port names,
    so ``fit_predict(panel, **eval_inputs)`` works (``time_offsets`` / ``target_offsets`` are only read for horizon = len(target_offsets)).
    The pool = every row of ``train_panel`` plus the items' own economies (history and covariates, also for economies absent from the
    panel), restricted to the first ``n_periods`` (default: ``history.shape[1]``) period columns of the panel: panel and history both start
    at the same period, so calling with the ``load_dev`` history uses only periods up to the dev origin. ``members``: names from MEMBERS
    (default DEFAULT_MEMBERS, a fixed set of four). ``weights``: 'equal' (1/len(members) each; default), 'auto' (fit_weights on the
    rolling-origin backtest below, sum <= 1, weight left over stays on the flat member, fitted weights mixed 50:50 with equal weights)
    or {member: w}. Rolling-origin backtest (``n_backtest`` origins spaced one period apart; origin j has column index
    L - 1 - horizon - j, so its ``horizon`` targets are all observed; every member is refitted on the periods up to the origin).
    ``n_jobs``: LightGBM threads (max 2). ``return_info=True`` returns (forecast, info) with info = {'roles', 'members', 'weights' {kind:
    {member: w}}, 'backtest_origins' (origins as periods relative to t0, newest first), 'backtest_smape' {kind: {'flat',
    'combination', member...}} in % over the backtest rows of the pool at all backtest origins pooled (log-ratio sMAPE; 'combination' =
    the forecast under ``weights``), 'backtest_by_origin' {kind: {'flat', 'combination', member...: [sMAPE per origin]}}}.
    ~5-30 s per call with return_info.
Blocks
  build_pool(...) -> Pool                 arrays per kind (n_economies, L) with the economies' ids; Pool.X[kind], Pool.roles, Pool.econ
  member_forecasts(pool, kind, origin, members, horizon, ...) -> {member: (n_economies, horizon)}
                                          log-ratio forecasts for every economy of the pool from periods <= ``origin`` (column index)
  rolling_backtest(pool, kind, members, horizon, n_origins=8, seed=0, n_jobs=1, step=1) -> dict
                                          {'origins', 'pred' {member: [(n_o, horizon)]}, 'actual' [(n_o, horizon)]} log-ratios; origins
                                          L - 1 - horizon - step * j (j = 0..n_origins-1, column >= 12)
  fit_weights(pred, actual, cap=1.0, shrink=0.0) -> {member: w}     non-negative weights, sum <= cap, minimising the mean sMAPE
                                          of sum_m w_m * pred_m against actual (both as lists of arrays); ``shrink`` mixes in equal weights
  smape_logratio(actual, pred) -> float   sMAPE in % of two log-ratio arrays: 200 * mean tanh(|actual - pred| / 2)
  period_profile(train_panel, n_periods=None, last=12, indicator_kinds=None) -> {kind: {'periods', 'median', 'mad'}}
                                          per period (the last ``last`` ones): median and median absolute deviation across economies of the
                                          annual log-change (level / percent roles) or of the value (change role)
  local_forecast(x, method, horizon, **params) -> (horizon,)   one positive series -> level forecast ('flat', 'drift', 'revert',
                                          'theta', 'holt')
  infer_roles(panel_or_arrays, indicator_kinds=None) -> {kind: role}
  panel_to_arrays(train_panel, n_periods=None) -> (econ_ids, {kind: (n, L)}, {'region': {id: str}, 'income_level': {id: str}})
sMAPE facts. sMAPE(y, p) = 200 * mean_h tanh(|ln y_h - ln p_h| / 2) for positive y, p: for positive series the score is a
concave function of the absolute error in log space (~ 100 * mean |ln y - ln p| for small errors), so log-ratio forecasts scored with
absolute error are the matching loss.
Backtest facts. The dev split (load_dev / score_dev) is one origin, t-4, with the targets t-3..t0 of the evaluation items' own economies:
a few dozen values per kind, which come from the last observed periods. Any shock shared by many economies in those periods is in the
dev targets and (for windows ending there) in the last training windows of the pooled members. The rolling backtest covers several
origins; 'backtest_by_origin' gives the scores per origin and ``period_profile`` gives the cross-economy median / spread of the annual
change per period, so windows with and without a common shock can be told apart. Rows scored: the dev split scores the evaluation items
only (one row per item, about half of them per kind), the rolling backtest scores every economy of the pool at each of its origins
(n_economies x n_backtest rows per kind; rows at neighbouring origins overlap in time), so a dev score is the mean of far fewer rows.
MEMBERS (log-ratio forecasts r_h = ln y_{t+h} - ln y_t from one economy's own past and the pool); DEFAULT_MEMBERS = ('ridge', 'huber',
'lgbm_core', 'robdrift'), combined with equal weights. The four defaults were chosen from rolling-origin backtests over the periods
of the source-region panel (all origins with a fully observed horizon, including windows with a common shock), scored per kind.
  flat      r_h = 0 (the series stays at its value at the origin)
  drift     g = mean of the last ``k=10`` annual log-changes clipped to +-0.2; r_h = g * sum_{j<=h} phi^j, phi = 0.8
  robdrift  g = median of the last ``k=10`` annual log-changes of the economy, mixed 60:40 with the median g of all economies of the
            pool (``shrink=0.4``); r_h = g * sum_{j<=h} phi^j, phi = 0.9
  ridge     pooled Ridge(alpha=1000) on standardised features, one model per (kind, h), fitted on windows (economy, origin) from all pool
            series up to the forecast origin (targets must lie at or before it); all features below, including the other kinds' rates
  huber     pooled HuberRegressor(alpha=1, epsilon=1.35) on standardised core features (below), same windows
  lgbm      pooled LightGBM (L1 loss, 100 trees, 4 leaves, min 100 rows per leaf, learning rate 0.05, ``n_jobs`` threads (default 1: the data are small and results do not depend on it)), same windows; all features
  lgbm_core pooled LightGBM (L1 loss, 150 trees, 8 leaves, min 40 rows per leaf, learning rate 0.05), same windows; core features
  revert    r_h = (1 - phi^h) * (mean of the last ``k=5`` log values - log value at the origin), phi = 0.85
  theta     scilib.forecast.theta on the log series -> r_h
  Features of a window at origin o for a series of kind K (log values interpolated over gaps): mean log-change over the last 1, 2, 3, 5, 8
  periods; log value at the origin minus the mean of its last 3 / 5 / 10 / all values; std of the last 10 log-changes; ln y_o; share of
  observed periods; for every other kind: its latest, 3-period mean and 8-period mean rate (log-change for level / percent roles,
  value / 100 clipped to +-0.5 for 'change'). Core features = the series' own features g1, g3, g8, dev3, dev10, devall, vol, ln y_o (no
  other kind's rates, no observation share). Missing features are set to the training mean. Region and income are not features.
"""
from __future__ import annotations

import re
import warnings

import numpy as np

from . import forecast as _fc

__all__ = ["MEMBERS", "DEFAULT_MEMBERS", "Pool", "infer_roles", "panel_to_arrays", "build_pool", "member_forecasts",
           "rolling_backtest", "fit_weights", "smape_logratio", "period_profile", "local_forecast", "fit_predict"]

MEMBERS = ("flat", "drift", "robdrift", "revert", "theta", "ridge", "huber", "lgbm", "lgbm_core")
BLOCK_MEMBERS = MEMBERS                                # member_forecasts / rolling_backtest / fit_predict accept the same names
DEFAULT_MEMBERS = ("ridge", "huber", "lgbm_core", "robdrift")
_CORE = ("dev10", "dev3", "devall", "g1", "g3", "g8", "lvl", "vol")     # the series' own features used by 'huber' / 'lgbm_core'
_PERIOD = re.compile(r"t(0|[+-]\d+)$")
_MIN_OBS = 10          # a series needs this many observed periods to be a training / backtest row
_O_LO = 6              # first window origin (column index) of the pooled models
_BT_MIN = 12           # earliest backtest origin (column index)


# ------------------------------------------------------------------------------------------------ roles / panel
def _period_columns(panel) -> list[str]:
    cols = [c for c in panel.columns if _PERIOD.match(str(c))]
    return sorted(cols, key=lambda c: int(_PERIOD.match(str(c)).group(1)))


def infer_roles(arrays, indicator_kinds: dict | None = None) -> dict:
    """{kind: role}. ``arrays``: {kind: array of values} (or a train_panel table, whose rows are grouped by ``indicator``).
    A recognised ``indicator_kinds`` description decides ('percentage change' -> change, 'percentage of' -> percent,
    'level' -> level); otherwise any value <= 0 -> 'change', all values <= 100 -> 'percent', else 'level'."""
    if hasattr(arrays, "columns"):
        pc = _period_columns(arrays)
        arrays = {k: g[pc].to_numpy(float) for k, g in arrays.groupby("indicator")}
    roles = {}
    for k, v in arrays.items():
        d = str((indicator_kinds or {}).get(k, "")).lower()
        if "change" in d:
            roles[k] = "change"
        elif "percentage of" in d or "percent of" in d:
            roles[k] = "percent"
        elif "level" in d:
            roles[k] = "level"
        else:
            x = np.asarray(v, float)
            x = x[np.isfinite(x)]
            roles[k] = "change" if x.size and x.min() <= 0 else "percent" if x.size and x.max() <= 100 else "level"
    return roles


def panel_to_arrays(train_panel, n_periods: int | None = None):
    """(economy ids sorted, {kind: (n_economies, L) values, NaN = missing}, {'region': {id: str}, 'income_level': {id: str}}) from the
    ``load_train`` table; L = ``n_periods`` (first columns) or all period columns."""
    pc = _period_columns(train_panel)
    if n_periods is not None:
        pc = pc[:int(n_periods)]
    ids = sorted(set(train_panel["economy_id"]))
    pos = {e: i for i, e in enumerate(ids)}
    arrays = {k: np.full((len(ids), len(pc)), np.nan) for k in sorted(set(train_panel["indicator"]))}
    for e, k, row in zip(train_panel["economy_id"], train_panel["indicator"], train_panel[pc].to_numpy(float)):
        arrays[k][pos[e]] = row
    meta = {}
    for col in ("region", "income_level"):
        meta[col] = dict(zip(train_panel["economy_id"], train_panel[col])) if col in train_panel.columns else {}
    return ids, arrays, meta


class Pool:
    """Series of a set of economies: ``econ`` ids, ``X[kind]`` (n, L) raw values, ``roles[kind]``, ``items`` = list of (row, kind) of the
    items to forecast, ``L`` periods. Log arrays of the level / percent kinds: ``lraw[kind]`` (NaN where missing or <= 0) and ``lint[kind]``
    (gaps interpolated, trailing gaps repeat the final observed value)."""

    def __init__(self, econ, X, roles, items, L):
        self.econ, self.X, self.roles, self.items, self.L = list(econ), X, roles, list(items), int(L)
        self.kinds = sorted(X)
        self.targets = [k for k in self.kinds if roles[k] in ("level", "percent")]
        self.lraw, self.lint, self.rate = {}, {}, {}
        for k in self.targets:
            x = np.where(X[k] > 0, X[k], np.nan)
            lr = np.log(np.where(np.isfinite(x), x, 1.0))
            lr[~np.isfinite(x)] = np.nan
            self.lraw[k] = lr
            self.lint[k] = _interp_rows(lr)
            r = np.full_like(lr, np.nan)
            r[:, 1:] = np.diff(self.lint[k], axis=1)
            r[:, 1:][~np.isfinite(lr[:, 1:]) | ~np.isfinite(lr[:, :-1])] = np.nan
            self.rate[k] = np.clip(r, -0.5, 0.5)
        for k in self.kinds:
            if k not in self.targets:
                self.rate[k] = np.clip(np.asarray(X[k], float) / 100.0, -0.5, 0.5)
        self._fcache: dict = {}


def _interp_rows(lr: np.ndarray) -> np.ndarray:
    out = np.full_like(lr, np.nan)
    idx = np.arange(lr.shape[1])
    for i in range(lr.shape[0]):
        ok = np.isfinite(lr[i])
        if ok.sum() == 0:
            continue
        first = idx[ok][0]
        out[i, first:] = np.interp(idx[first:], idx[ok], lr[i, ok])       # np.interp repeats the final observed value after `last`
        out[i, :first] = np.nan
    return out


def build_pool(train_panel, history, indicator, indicator_kinds: dict | None = None, covariates=None, covariate_indicators=None,
               economy_id=None, n_periods: int | None = None) -> Pool:
    """Pool of the panel economies (first ``n_periods`` period columns; default history.shape[1]) plus the items' economies. Item i
    contributes ``history[i]`` as its economy's series of kind ``indicator[i]`` and ``covariates[i]`` as the economy's series of the kinds in
    ``covariate_indicators`` (only where the panel has no value); items without ``economy_id`` are their own economy.
    ``train_panel`` may be None (pool = the items' economies)."""
    hist = np.atleast_2d(np.asarray(history, float))
    n, L = hist.shape
    L = int(n_periods) if n_periods is not None else L
    ind = [str(k) for k in indicator]
    if len(ind) != n:
        raise ValueError(f"indicator has {len(ind)} entries for {n} histories")
    econ, X, meta_kinds = [], {}, {}
    if train_panel is not None:
        econ, X, _ = panel_to_arrays(train_panel, L)
        if next(iter(X.values())).shape[1] < L:
            raise ValueError(f"train_panel has fewer than {L} period columns")
    pos = {e: i for i, e in enumerate(econ)}
    cov = None if covariates is None else np.asarray(covariates, float)
    ckinds = [str(k) for k in (covariate_indicators or [])]
    if cov is not None and cov.shape[1] != len(ckinds):
        raise ValueError("covariates and covariate_indicators disagree")
    eids = [str(e) for e in economy_id] if economy_id is not None else [f"__item{i}" for i in range(n)]
    for i in range(n):
        if eids[i] not in pos:
            pos[eids[i]] = len(econ)
            econ.append(eids[i])
            for k in X:
                X[k] = np.vstack([X[k], np.full((1, X[k].shape[1]), np.nan)])
    m = len(econ)
    for k in set(ind) | set(ckinds):
        if k not in X:
            X[k] = np.full((m, L), np.nan)
    for i in range(n):
        e = pos[eids[i]]
        h = hist[i, :L]
        cur = X[ind[i]][e, :h.size]
        X[ind[i]][e, :h.size] = np.where(np.isfinite(h), h, cur)
        if cov is not None:
            for c, k in enumerate(ckinds):
                v = cov[i, c, :L]
                cur = X[k][e, :v.size]
                X[k][e, :v.size] = np.where(np.isfinite(cur), cur, v)
    roles = infer_roles(X, indicator_kinds)
    bad = [k for k in ind if roles[k] == "change"]
    if bad:
        raise ValueError(f"items of kind(s) {sorted(set(bad))} have role 'change' (values <= 0); only 'level' / 'percent' kinds can be forecast")
    return Pool(econ, X, roles, [(pos[eids[i]], ind[i]) for i in range(n)], L)


# ------------------------------------------------------------------------------------------------ features / members
def smape_logratio(actual, pred) -> float:
    """sMAPE in % between two arrays of log-ratios ln(y_{t+h}/y_t): 200 * mean tanh(|actual - pred| / 2)."""
    d = np.abs(np.asarray(actual, float) - np.asarray(pred, float))
    return float(200.0 * np.mean(np.tanh(d / 2.0)))


def _nm(a: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(a, axis=1)


def _features(pool: Pool, kind: str, o: int):
    """(names, (n_economies, F)) of the pooled-model features at origin column ``o`` for series of ``kind`` (cached)."""
    key = (kind, o)
    if key in pool._fcache:
        return pool._fcache[key]
    ly = pool.lint[kind][:, :o + 1]
    n = ly.shape[0]

    def lag(j):
        return ly[:, o - j] if o - j >= 0 else np.full(n, np.nan)

    f = {}
    for j in (1, 2, 3, 5, 8):
        f[f"g{j}"] = np.clip((lag(0) - lag(j)) / j, -0.5, 0.5)
    for j in (3, 5, 10):
        f[f"dev{j}"] = np.clip(lag(0) - _nm(ly[:, -j:]), -1.5, 1.5)
    f["devall"] = np.clip(lag(0) - _nm(ly), -2.0, 2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f["vol"] = np.nanstd(np.diff(ly, axis=1)[:, -10:], axis=1) if o >= 2 else np.full(n, np.nan)
    f["lvl"] = lag(0)
    f["nobs"] = np.isfinite(pool.lraw[kind][:, :o + 1]).sum(1) / float(o + 1)
    for c in pool.kinds:
        if c != kind:
            r = pool.rate[c][:, :o + 1]
            f[f"{c}_1"] = r[:, -1]
            f[f"{c}_3"] = _nm(r[:, -3:])
            f[f"{c}_8"] = _nm(r[:, -8:])
    names = sorted(f)
    out = (names, np.column_stack([f[k] for k in names]))
    pool._fcache[key] = out
    return out


def _window_data(pool: Pool, kind: str, o_end: int, h: int):
    """Training windows (features, target ln y_{o+h} - ln y_o) for all origins o in [_O_LO, o_end - h] with y_o and y_{o+h} observed."""
    xs, ys = [], []
    lr = pool.lraw[kind]
    for o in range(_O_LO, o_end - h + 1):
        _, F = _features(pool, kind, o)
        y = lr[:, o + h] - lr[:, o]
        ok = np.isfinite(y) & (np.isfinite(pool.lraw[kind][:, :o + 1]).sum(1) >= _MIN_OBS // 2)
        xs.append(F[ok])
        ys.append(y[ok])
    if not xs:
        raise ValueError("no training windows: too few periods before the origin")
    return np.vstack(xs), np.concatenate(ys)


def _pooled(pool: Pool, kind: str, origin: int, horizon: int, model: str, seed: int, n_jobs: int, **prm) -> np.ndarray:
    names, Fte = _features(pool, kind, origin)
    if model in ("huber", "lgbm_core"):
        keep = [i for i, n in enumerate(names) if n in _CORE]
        Fte = Fte[:, keep]
    out = np.zeros((Fte.shape[0], horizon))
    for h in range(1, horizon + 1):
        Xtr, ytr = _window_data(pool, kind, origin, h)
        if model in ("huber", "lgbm_core"):
            Xtr = Xtr[:, keep]
        mu = np.array([np.mean(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.0 for c in Xtr.T])
        A = np.where(np.isfinite(Xtr), Xtr, mu)
        B = np.where(np.isfinite(Fte), Fte, mu)
        sd = A.std(axis=0)
        sd[sd < 1e-9] = 1.0
        A, B = (A - mu) / sd, (B - mu) / sd
        if model == "ridge":
            from sklearn.linear_model import Ridge
            out[:, h - 1] = Ridge(alpha=prm.get("alpha", 1000.0)).fit(A, ytr).predict(B)
        elif model == "huber":
            from sklearn.linear_model import HuberRegressor
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[:, h - 1] = HuberRegressor(alpha=prm.get("alpha", 1.0), epsilon=prm.get("epsilon", 1.35), max_iter=300).fit(A, ytr).predict(B)
        else:
            import lightgbm as lgb
            core = model == "lgbm_core"
            mdl = lgb.LGBMRegressor(objective="l1", n_estimators=prm.get("n_estimators", 150 if core else 100),
                                    learning_rate=prm.get("learning_rate", 0.05), num_leaves=prm.get("num_leaves", 8 if core else 4),
                                    min_child_samples=prm.get("min_child_samples", 40 if core else 100),
                                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, n_jobs=max(1, min(int(n_jobs), 2)),
                                    verbose=-1, random_state=seed, deterministic=True, force_row_wise=True)
            out[:, h - 1] = mdl.fit(A, ytr).predict(B)
    return out


def _drift(ly: np.ndarray, horizon: int, k: int = 10, phi: float = 0.8, clip: float = 0.2) -> np.ndarray:
    d = np.diff(ly, axis=1)[:, -k:]
    g = np.clip(np.nan_to_num(_nm(d)), -clip, clip)
    return g[:, None] * np.cumsum(phi ** np.arange(1, horizon + 1))[None, :]


def _robdrift(ly: np.ndarray, horizon: int, k: int = 10, shrink: float = 0.4, phi: float = 0.9) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = np.nanmedian(np.diff(ly, axis=1)[:, -k:], axis=1)
        gp = np.nanmedian(g) if np.isfinite(g).any() else 0.0
    g = (1.0 - shrink) * np.where(np.isfinite(g), g, gp) + shrink * gp
    return g[:, None] * np.cumsum(phi ** np.arange(1, horizon + 1))[None, :]


def _revert(ly: np.ndarray, horizon: int, k: int = 5, phi: float = 0.85) -> np.ndarray:
    gap = np.nan_to_num(_nm(ly[:, -k:]) - ly[:, -1])
    return gap[:, None] * (1.0 - phi ** np.arange(1, horizon + 1))[None, :]


def _theta_rows(ly: np.ndarray, horizon: int) -> np.ndarray:
    out = np.zeros((ly.shape[0], horizon))
    for i in range(ly.shape[0]):
        x = ly[i][np.isfinite(ly[i])]
        if x.size >= 8:
            out[i] = _fc.theta(x, horizon, 1) - x[-1]
    return out


def member_forecasts(pool: Pool, kind: str, origin: int, members, horizon: int, seed: int = 0, n_jobs: int = 1, params: dict | None = None) -> dict:
    """{member: (n_economies, horizon)} log-ratio forecasts ln y_{origin+h} - ln y_origin for every economy of the pool and series of
    ``kind`` (rows without any observation give 0), using only the periods <= ``origin`` (column index)."""
    if kind not in pool.targets:
        raise ValueError(f"kind {kind!r} has role {pool.roles.get(kind)!r}; forecastable kinds: {pool.targets}")
    ly = pool.lint[kind][:, :origin + 1]
    out = {}
    params = params or {}
    for m in members:
        prm = params.get(m, {})
        if m == "flat":
            out[m] = np.zeros((ly.shape[0], horizon))
        elif m == "drift":
            out[m] = _drift(ly, horizon, **prm)
        elif m == "robdrift":
            out[m] = _robdrift(ly, horizon, **prm)
        elif m == "revert":
            out[m] = _revert(ly, horizon, **prm)
        elif m == "theta":
            out[m] = _theta_rows(ly, horizon)
        elif m in ("ridge", "huber", "lgbm", "lgbm_core"):
            out[m] = _pooled(pool, kind, origin, horizon, m, seed, n_jobs, **prm)
        else:
            raise ValueError(f"unknown member {m!r}; available: {BLOCK_MEMBERS}")
        out[m] = np.where(np.isfinite(out[m]), out[m], 0.0)
    return out


def rolling_backtest(pool: Pool, kind: str, members, horizon: int, n_origins: int = 8, seed: int = 0, n_jobs: int = 1, step: int = 1) -> dict:
    """Backtest origins ``L - 1 - horizon - step * j`` (j = 0..n_origins-1, column index >= 12; the origin ``L - 1 - horizon`` is the newest, its
    targets end at the last period) with every member fitted on periods <= origin. Rows = pool series with the origin value and all
    ``horizon`` targets observed and >= 10 observed periods. Returns {'origins', 'pred' {member: [arrays (rows, horizon) per origin]},
    'actual' [arrays per origin]} in log-ratios (origins newest first)."""
    origins = [pool.L - 1 - horizon - int(step) * j for j in range(int(n_origins)) if pool.L - 1 - horizon - int(step) * j >= _BT_MIN]
    lr = pool.lraw[kind]
    res = {"origins": origins, "pred": {m: [] for m in members}, "actual": []}
    for ob in origins:
        y = lr[:, ob + 1:ob + 1 + horizon] - lr[:, [ob]]
        ok = np.isfinite(y).all(axis=1) & (np.isfinite(lr[:, :ob + 1]).sum(1) >= _MIN_OBS)
        mf = member_forecasts(pool, kind, ob, members, horizon, seed, n_jobs)
        for m in members:
            res["pred"][m].append(mf[m][ok])
        res["actual"].append(y[ok])
    return res


def fit_weights(pred, actual, cap: float = 1.0, shrink: float = 0.0) -> dict:
    """Weights w_m >= 0 with sum <= ``cap`` minimising the mean sMAPE (200 * tanh(|d| / 2)) of sum_m w_m * pred_m against ``actual``.
    ``pred`` = {member: array or list of arrays}, ``actual`` = array or list of arrays (same shapes, log-ratios). ``shrink`` in [0, 1] mixes
    the fitted weights with equal weights summing to ``cap``. Deterministic (SLSQP from three starts)."""
    from scipy.optimize import minimize
    names = list(pred)
    A = np.concatenate([np.asarray(a, float).ravel() for a in _as_list(actual)])
    P = np.stack([np.concatenate([np.asarray(a, float).ravel() for a in _as_list(pred[m])]) for m in names])
    if A.size == 0:
        return {m: cap / len(names) for m in names}

    def loss(w):
        return float(np.mean(np.tanh(np.sqrt((A - w @ P) ** 2 + 1e-6) / 2.0)))

    k = len(names)
    best_w, best = None, np.inf
    for w0 in (np.full(k, cap / k), np.zeros(k) + 1e-3, np.eye(k)[0] * cap):
        r = minimize(loss, w0, method="SLSQP", bounds=[(0.0, cap)] * k, constraints=[{"type": "ineq", "fun": lambda w: cap - w.sum()}],
                     options={"maxiter": 100, "ftol": 1e-9})
        w = np.clip(r.x, 0.0, cap)
        if w.sum() > cap:
            w = w * cap / w.sum()
        v = loss(w)
        if v < best - 1e-12:
            best, best_w = v, w
    best_w = (1.0 - shrink) * best_w + shrink * np.full(k, cap / k)
    return {m: float(w) for m, w in zip(names, best_w)}


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def period_profile(train_panel, n_periods: int | None = None, last: int = 12, indicator_kinds: dict | None = None) -> dict:
    """{kind: {'periods' [labels], 'median' (last,), 'mad' (last,)}}: for each of the last ``last`` period columns (of the first ``n_periods``
    columns), the median and the median absolute deviation across economies of the annual log-change ln x_t - ln x_{t-1} ('level' /
    'percent' roles; economies with both values > 0) or of the value itself ('change' role). A period where the median is large
    relative to the MAD is one where many economies moved together."""
    pc = _period_columns(train_panel)
    if n_periods is not None:
        pc = pc[:int(n_periods)]
    _, arrays, _ = panel_to_arrays(train_panel, len(pc))
    roles = infer_roles(arrays, indicator_kinds)
    k_last = min(int(last), len(pc) - 1)
    out = {}
    for k, x in arrays.items():
        if roles[k] == "change":
            v = x[:, -k_last:]
        else:
            lx = np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), np.nan)
            v = np.diff(lx, axis=1)[:, -k_last:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med = np.nanmedian(v, axis=0)
            mad = np.nanmedian(np.abs(v - med[None, :]), axis=0)
        out[k] = {"periods": list(pc[-k_last:]), "median": med, "mad": mad}
    return out


# ------------------------------------------------------------------------------------------------ single-series helper
def local_forecast(x, method: str, horizon: int = 4, **params) -> np.ndarray:
    """Level forecast (horizon,) of one positive series (NaN gaps interpolated, trailing NaN ignored). method: 'flat' (the value at the end of x repeated),
    'drift' (params k=10, phi=0.8, clip=0.2), 'revert' (k=5, phi=0.85), 'theta' (scilib.forecast.theta on ln x), 'holt'
    (scilib.forecast.holt_damped on ln x)."""
    x = np.asarray(x, float)
    lx = np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), np.nan)
    ok = np.isfinite(lx)
    if ok.sum() == 0:
        raise ValueError("series has no positive observation")
    idx = np.arange(lx.size)
    first, last = idx[ok][0], idx[ok][-1]
    y = np.interp(idx[first:last + 1], idx[ok], lx[ok])
    if method == "flat":
        r = np.zeros(horizon)
    elif method == "drift":
        r = _drift(y[None, :], horizon, **params)[0]
    elif method == "revert":
        r = _revert(y[None, :], horizon, **params)[0]
    elif method == "theta":
        r = _fc.theta(y, horizon, 1) - y[-1]
    elif method == "holt":
        r = _fc.holt_damped(y, horizon, 1) - y[-1]
    else:
        raise ValueError(f"unknown method {method!r}; available: flat, drift, revert, theta, holt")
    return np.exp(y[-1] + np.asarray(r, float))


# ------------------------------------------------------------------------------------------------ entry point
def fit_predict(train_panel, history, indicator, indicator_kinds: dict | None = None, covariates=None, covariate_indicators=None,
                economy_id=None, region=None, income_level=None, time_offsets=None, target_offsets=None, horizon: int | None = None,
                members=None, weights="equal", n_backtest: int = 8, n_periods: int | None = None, seed: int = 0, n_jobs: int = 1,
                return_info: bool = False):
    """Forecast every item of ``history`` (n, L) ``horizon`` steps ahead; see the module docstring. Returns (n, horizon) level forecasts
    (and an info dict with ``return_info=True``)."""
    hist = np.atleast_2d(np.asarray(history, float))
    if horizon is None:
        horizon = len(target_offsets) if target_offsets is not None else 4
    horizon = int(horizon)
    members = list(DEFAULT_MEMBERS if members is None else members)
    for m in members:
        if m not in MEMBERS:
            raise ValueError(f"unknown member {m!r}; available: {MEMBERS}")
    if not members:
        raise ValueError("members must not be empty")
    pool = build_pool(train_panel, hist, indicator, indicator_kinds, covariates, covariate_indicators, economy_id, n_periods)
    o_last = pool.L - 1
    out = np.zeros((hist.shape[0], horizon))
    info = {"roles": dict(pool.roles), "members": members, "weights": {}, "backtest_origins": [], "backtest_smape": {}, "backtest_by_origin": {}}
    for kind in sorted({k for _, k in pool.items}):
        rows = [i for i, (_, k) in enumerate(pool.items) if k == kind]
        erow = np.array([pool.items[i][0] for i in rows])
        mf = member_forecasts(pool, kind, o_last, members, horizon, seed, n_jobs)
        if isinstance(weights, dict):
            w = {m: float(weights.get(m, 0.0)) for m in members}
        elif weights in ("auto", "equal"):
            w = {m: 1.0 / len(members) for m in members}
        else:
            raise ValueError("weights must be 'auto', 'equal' or a {member: weight} dict")
        if weights == "auto" or (return_info and n_backtest > 0):
            bt = rolling_backtest(pool, kind, members, horizon, n_backtest, seed, n_jobs)
            if bt["origins"]:
                if weights == "auto":
                    w = fit_weights(bt["pred"], bt["actual"], shrink=0.5)
                act = np.concatenate([a.ravel() for a in bt["actual"]])
                cat = {m: np.concatenate([a.ravel() for a in bt["pred"][m]]) for m in members}
                info["backtest_smape"][kind] = {"flat": smape_logratio(act, 0.0), "combination": smape_logratio(act, sum(w[m] * cat[m] for m in members)),
                                                **{m: smape_logratio(act, cat[m]) for m in members}}
                per = {"flat": [smape_logratio(a, 0.0) for a in bt["actual"]],
                       "combination": [smape_logratio(a, sum(w[m] * bt["pred"][m][j] for m in members)) for j, a in enumerate(bt["actual"])]}
                for m in members:
                    per[m] = [smape_logratio(a, bt["pred"][m][j]) for j, a in enumerate(bt["actual"])]
                info["backtest_by_origin"][kind] = per
                info["backtest_origins"] = [int(o - o_last) for o in bt["origins"]]
        info["weights"][kind] = w
        r = sum(w[m] * mf[m] for m in members)[erow]
        base = pool.lint[kind][erow, o_last]
        if not np.isfinite(base).all():
            raise ValueError(f"items {[rows[j] for j in np.flatnonzero(~np.isfinite(base))]} of kind {kind!r} have no positive observation in their history")
        pred = np.exp(base[:, None] + r)
        if pool.roles[kind] == "percent":
            pred = np.minimum(pred, 100.0)
        for j, i in enumerate(rows):
            out[i] = pred[j]
    return (out, info) if return_info else out
