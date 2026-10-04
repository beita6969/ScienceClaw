"""Day-ahead hourly building-load forecasting from 168-hour contexts (kWh; CVRMSE %).

Window convention: a context is 168 consecutive hourly loads (7 days) ending right before the first target hour; the
forecast covers the next 24 hours. For target hour h (0..23) the value ``k`` days earlier is ``context[:, 168 - 24*k + h]``
(k = 1..7). Timestamps are ISO strings such as ``2016-11-26 01:00:00`` or ``2016-11-26T01:00:00`` (no time zone, hourly).
Every function works only on the arrays it is given; a fitted model uses only the history passed to ``fit``.

Constants: CONTEXT_H = 168, HORIZON_H = 24, CANDIDATES = names of the forecasts returned by ``candidates``.

Metric (local checks; lower is better)
cvrmse(y_true, y_pred) -> float                      100 * RMSE / mean(y_true) over every given hour
cvrmse_by_building(y_true, y_pred, building_id) -> dict      CVRMSE of the rows of each building (rows pooled)
balanced_cvrmse(y_true, y_pred, building_id, category) -> float     per-building CVRMSE, median within each category
                                                     ('residential' / 'commercial'), mean of the category medians

Context-only forecasts (no fitting); context (n, 168) -> (n, 24)
lag_matrix(context, days=7) -> (n, 24, days)         [i, h, k-1] = load at target hour h, k days earlier
yesterday(context)                                   the 24 hours immediately before the target day, repeated
day_mean(context, days=7) / day_median(context, days=7)     per-hour mean / median over the last ``days`` days
blend_weight(context, backtest_days=2) -> (n,)       w in [0, 1]: each of the last ``backtest_days`` context days is
                                                     forecast from the days before it by (a) the day just before it and
                                                     (b) the per-hour median of all days before it; w = err_b / (err_a + err_b)
backtest_blend(context, backtest_days=2)             w * yesterday + (1 - w) * day_median
core_forecast(context, category)                     day_median for 'residential' rows, backtest_blend for 'commercial' rows

Data preparation
calendar_features(start, hours=24) -> dict           hour, dayofweek (Monday = 0), weekend (0/1), hour_of_week (0..167);
                                                     each an int array (n, hours) for the hours following ``start``
history_windows(load, history_start, building_id=None, category=None, stride=6, first_hour=0, last_hour=None) -> dict
                                                     sliding (context, target) windows inside the given histories:
                                                     context (N, 168), target (N, 24), building_index (N,), building_id,
                                                     category, context_start, target_start (lists of N ISO strings)
hour_of_week_profile(x, start) -> (168,)             mean load of a 1-D history per hour of the week / its overall mean

Learned model (pooled over the buildings passed to ``fit``)
fit(load, history_start, building_id, category, ...) -> LoadForecaster
    load (n_buildings, T) kWh, hourly from history_start[b]; training rows are 168 h windows slid (every ``stride`` hours)
    over each building's own load; features are scale-free (context divided by its mean), target = next 24 h / context mean;
    models: one LightGBM regressor and one Ridge regression on standardised features (n_jobs <= 2, seeded).
    'profile' feature = hour_of_week_profile of the building; for a training row it is computed without the time block
    (1 of ``profile_folds`` = 4 equal blocks of that history) that holds the row's target hours, for forecasts from the whole history.
LoadForecaster.candidates(context, target_start, building_id, category, w_core=0.5, pretrained=False) -> dict of (n, 24) float arrays >= 0
    'yesterday'  yesterday(context)                  'mean7'   day_mean(context)
    'median7'    day_median(context)                 'core'    core_forecast(context, category)
    'ml'         mean of the LightGBM and Ridge predictions (in kWh)
    'ens'        w_core * core + (1 - w_core) * ml
    building_id[i] must be a building passed to ``fit`` (its hour-of-week profile is a feature); category[i] is
    'residential' or 'commercial'.
    ``pretrained`` (default False): True adds two more keys, PRETRAINED_CANDIDATES (needs ``scilib.tsfm``, a GPU worker; one call
    per ``candidates`` call, all rows at once):
    'chronos2'      median forecast of Chronos-2 (``tsfm.forecast``, model "chronos_2", the whole 168-h context), clipped at 0
    'ens_chronos2'  0.5 * 'ens' + 0.5 * 'chronos2'
LoadForecaster.ml_forecast(context, target_start, building_id, category)      the 'ml' array alone
LoadForecaster.predict(...)                          same arguments as ``candidates``; returns candidates(...)['ens']
forecast_candidates(load, history_start, history_building_id, history_category, context, target_start, building_id,
                    category, model=None, w_core=0.5, pretrained=False, ...) -> dict
                                                     fit (or reuse ``model``) and return ``candidates`` for one set of windows
backtest_history(load, history_start, building_id, category, holdout_hours=720, stride=24, pretrained=False, ...) -> dict
                                                     fit on the history without its last ``holdout_hours``, forecast day-ahead
                                                     windows whose targets lie in those hours: {'scores': {name: balanced
                                                     CVRMSE}, 'n_windows': int}; it needs only the histories, so it can run
                                                     inside the node (the targets of the dev windows are not returned to code:
                                                     a dev array is scored by the score_dev tool)
pick_lowest(candidates, scores, prefer='ens', rel_tol=0.03) -> (name, array)
                                                     candidate with the lowest score; if ``prefer`` scores within ``rel_tol``
                                                     (relative) of the lowest, ``prefer`` is returned instead

Call pattern (inside one code node; every input arrives as a tool output)::

    from scilib import loadforecast as lf
    m = lf.fit(load, history_start, hist_building_id, hist_category)        # tool load_history
    dev = m.candidates(dev_context, dev_target_start, dev_building_id, dev_category)      # tool load_dev
    ev = m.candidates(context, target_start, building_id, category)                       # tool load_eval_inputs
    # dev['ens'] / ev['ens'] are (n_dev, 24) / (n, 24) kWh arrays; the other keys are the alternatives listed above
    # with the pretrained candidates added (one extra GPU-worker call per line):
    dev = m.candidates(dev_context, dev_target_start, dev_building_id, dev_category, pretrained=True)
    ev = m.candidates(context, target_start, building_id, category, pretrained=True)      # keys also 'chronos2', 'ens_chronos2'
"""
from __future__ import annotations

import numpy as np

__all__ = ["CONTEXT_H", "HORIZON_H", "CANDIDATES", "cvrmse", "cvrmse_by_building", "balanced_cvrmse",
           "lag_matrix", "yesterday", "day_mean", "day_median", "blend_weight", "backtest_blend", "core_forecast",
           "calendar_features", "history_windows", "hour_of_week_profile", "LoadForecaster", "fit",
           "forecast_candidates", "backtest_history", "pick_lowest", "PRETRAINED_CANDIDATES"]

CONTEXT_H = 168
HORIZON_H = 24
CATEGORIES = ("residential", "commercial")
CANDIDATES = ("yesterday", "mean7", "median7", "core", "ml", "ens")
PRETRAINED_CANDIDATES = ("chronos2", "ens_chronos2")
FEATURE_NAMES = (["hour", "dayofweek", "weekend"] + [f"lag{k}" for k in range(1, 8)]
                 + ["lag_median", "lag_mean", "lag_min", "lag_max", "lag_std", "profile", "last_hour", "last_day_mean",
                    "context_std", "residential"])


# ------------------------------------------------------------------------------------------------ input checks
def _context(context) -> np.ndarray:
    a = np.asarray(context, dtype=float)
    if a.ndim == 1 and a.size == CONTEXT_H:
        a = a[None]
    if a.ndim != 2 or a.shape[1] != CONTEXT_H:
        raise ValueError(f"context must have shape (n, {CONTEXT_H}), got {tuple(a.shape)}")
    if not np.all(np.isfinite(a)):
        raise ValueError("context contains non-finite values")
    return a


def _is_residential(category, n: int) -> np.ndarray:
    cat = np.asarray(category).astype(str)
    if cat.shape != (n,):
        raise ValueError(f"category must hold one label per row ({n}), got shape {cat.shape}")
    bad = sorted(set(cat.tolist()) - set(CATEGORIES))
    if bad:
        raise ValueError(f"unknown category {bad}; expected one of {list(CATEGORIES)}")
    return cat == "residential"


def _hours_since_epoch(ts) -> np.ndarray:
    """int64 hours since 1970-01-01T00 of ISO strings / datetime64 / pandas timestamps (one per element, flattened)."""
    import pandas as pd
    if isinstance(ts, (str, np.datetime64)) or np.ndim(ts) == 0:
        ts = [ts]
    arr = np.asarray(list(ts) if not isinstance(ts, np.ndarray) else ts)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[h]").astype(np.int64).ravel()
    idx = pd.to_datetime([str(t) for t in arr.ravel()], format="ISO8601")
    return np.asarray(idx.values).astype("datetime64[h]").astype(np.int64)


def _iso(hours: np.ndarray) -> list[str]:
    return [str(t) for t in np.asarray(hours, dtype=np.int64).astype("datetime64[h]").astype("datetime64[s]")]


# ------------------------------------------------------------------------------------------------ metric
def cvrmse(y_true, y_pred) -> float:
    """100 * sqrt(mean((y_true - y_pred)^2)) / mean(y_true) over every given hour (percent; lower is better)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch {y.shape} vs {p.shape}")
    m = float(np.mean(y))
    if not m > 0:
        raise ValueError("mean of y_true is not positive (CVRMSE undefined)")
    return float(100.0 * np.sqrt(np.mean((y - p) ** 2)) / m)


def cvrmse_by_building(y_true, y_pred, building_id) -> dict:
    """{building: cvrmse(...)} where the rows of one building (y_true[i] for building_id[i] == b) are pooled."""
    y, p, b = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float), np.asarray(building_id).astype(str)
    if y.shape != p.shape or len(b) != len(y):
        raise ValueError("y_true, y_pred and building_id must agree on the number of rows")
    return {u: cvrmse(y[b == u], p[b == u]) for u in sorted(set(b.tolist()))}


def balanced_cvrmse(y_true, y_pred, building_id, category) -> float:
    """Per-building CVRMSE (rows of a building pooled), median within each category present, mean of those medians."""
    per = cvrmse_by_building(y_true, y_pred, building_id)
    b, c = np.asarray(building_id).astype(str), np.asarray(category).astype(str)
    cat_of = dict(zip(b.tolist(), c.tolist()))
    meds = [float(np.median([v for u, v in per.items() if cat_of[u] == k])) for k in CATEGORIES
            if any(cat_of[u] == k for u in per)]
    if not meds:
        raise ValueError("no buildings of a known category")
    return float(np.mean(meds))


# ------------------------------------------------------------------------------------------------ context-only forecasts
def lag_matrix(context, days: int = 7) -> np.ndarray:
    """(n, 24, days): entry [i, h, k-1] is the load of context row i at target hour h, k days earlier (k = 1..days)."""
    if not 1 <= int(days) <= 7:
        raise ValueError("days must be in 1..7")
    c = _context(context)
    return np.stack([c[:, CONTEXT_H - 24 * k: CONTEXT_H - 24 * k + HORIZON_H] for k in range(1, int(days) + 1)], axis=2)


def yesterday(context) -> np.ndarray:
    """(n, 24): the last 24 context hours, i.e. each target hour equals the load 24 h earlier."""
    return lag_matrix(context, 1)[:, :, 0]


def day_mean(context, days: int = 7) -> np.ndarray:
    """(n, 24): per target hour, the mean of the same hour over the last ``days`` days."""
    return lag_matrix(context, days).mean(axis=2)


def day_median(context, days: int = 7) -> np.ndarray:
    """(n, 24): per target hour, the median of the same hour over the last ``days`` days."""
    return np.median(lag_matrix(context, days), axis=2)


def blend_weight(context, backtest_days: int = 2) -> np.ndarray:
    """(n,) weight w in [0, 1] of ``yesterday`` in ``backtest_blend``.

    For each of the last ``backtest_days`` days d of the context: err_a += mean((day d-1 - day d)^2) and
    err_b += mean((per-hour median of days 0..d-1 - day d)^2); w = err_b / (err_a + err_b + 1e-9)."""
    c = _context(context)
    if not 1 <= int(backtest_days) <= 5:
        raise ValueError("backtest_days must be in 1..5")
    d = c.reshape(len(c), 7, 24)
    err_a = np.zeros(len(c))
    err_b = np.zeros(len(c))
    for t in range(7 - int(backtest_days), 7):
        err_a += ((d[:, t - 1] - d[:, t]) ** 2).mean(axis=1)
        err_b += ((np.median(d[:, :t], axis=1) - d[:, t]) ** 2).mean(axis=1)
    return err_b / (err_a + err_b + 1e-9)


def backtest_blend(context, backtest_days: int = 2) -> np.ndarray:
    """(n, 24): w * yesterday + (1 - w) * day_median with w = ``blend_weight``."""
    w = blend_weight(context, backtest_days)[:, None]
    return w * yesterday(context) + (1.0 - w) * day_median(context)


def core_forecast(context, category) -> np.ndarray:
    """(n, 24), >= 0: ``day_median`` for 'residential' rows, ``backtest_blend`` for 'commercial' rows."""
    c = _context(context)
    res = _is_residential(category, len(c))[:, None]
    return np.maximum(np.where(res, day_median(c), backtest_blend(c)), 0.0)


# ------------------------------------------------------------------------------------------------ calendar / windows
def calendar_features(start, hours: int = 24) -> dict:
    """Calendar features of the ``hours`` hours starting at each timestamp of ``start`` (ISO strings or datetime64).

    Returns int arrays of shape (n, hours): 'hour' (0..23), 'dayofweek' (Monday = 0), 'weekend' (Saturday / Sunday = 1),
    'hour_of_week' (dayofweek * 24 + hour)."""
    h0 = _hours_since_epoch(start)
    hrs = h0[:, None] + np.arange(int(hours), dtype=np.int64)[None, :]
    hod = hrs % 24
    dow = (hrs // 24 + 3) % 7                       # 1970-01-01 was a Thursday (Monday = 0)
    return {"hour": hod, "dayofweek": dow, "weekend": (dow >= 5).astype(np.int64), "hour_of_week": dow * 24 + hod}


def _load_matrix(load) -> np.ndarray:
    a = np.asarray(load, dtype=float)
    if a.ndim == 1:
        a = a[None]
    if a.ndim != 2:
        raise ValueError(f"load must have shape (n_buildings, T), got {tuple(a.shape)}")
    if not np.all(np.isfinite(a)):
        raise ValueError("load contains non-finite values")
    return a


def _window_starts(T: int, stride: int, first_hour: int, last_hour: int | None) -> np.ndarray:
    last = T if last_hour is None else min(int(last_hour), T)
    return np.arange(int(first_hour) + CONTEXT_H, last - HORIZON_H + 1, max(1, int(stride)), dtype=np.int64)


def history_windows(load, history_start, building_id=None, category=None, stride: int = 6, first_hour: int = 0,
                    last_hour: int | None = None) -> dict:
    """Sliding windows over hourly histories ``load`` (n_buildings, T), hour t of building b starting at history_start[b].

    A window with target start s (an index into the history, s = first_hour + 168, +stride, ... <= last_hour - 24) has
    context load[b, s-168:s] and target load[b, s:s+24]; only hours in [first_hour, last_hour) are used
    (last_hour defaults to T). Returns a dict: context (N, 168), target (N, 24), building_index (N,) (row of ``load``),
    building_id / category (lists of N; category entries are None when not given), context_start / target_start (lists of N ISO strings); buildings are in row order."""
    x = _load_matrix(load)
    nb, T = x.shape
    h0 = _starts(history_start, nb)
    bid = [str(i) for i in range(nb)] if building_id is None else [str(b) for b in building_id]
    cat = [None] * nb if category is None else [str(c) for c in category]
    if len(bid) != nb or len(cat) != nb:
        raise ValueError("building_id and category need one entry per building")
    st = _window_starts(T, stride, first_hour, last_hour)
    if len(st) == 0:
        raise ValueError(f"history of {T} h is too short for windows of {CONTEXT_H} + {HORIZON_H} h in the requested range")
    ci = st[:, None] + np.arange(-CONTEXT_H, 0)[None, :]
    ti = st[:, None] + np.arange(HORIZON_H)[None, :]
    ctx = np.concatenate([x[b][ci] for b in range(nb)])
    tgt = np.concatenate([x[b][ti] for b in range(nb)])
    bidx = np.repeat(np.arange(nb), len(st))
    tstart = np.concatenate([h0[b] + st for b in range(nb)])
    return {"context": ctx, "target": tgt, "building_index": bidx, "building_id": [bid[b] for b in bidx],
            "category": [cat[b] for b in bidx], "context_start": _iso(tstart - CONTEXT_H), "target_start": _iso(tstart)}


def _starts(history_start, nb: int) -> np.ndarray:
    h0 = _hours_since_epoch(history_start)
    if len(h0) == 1 and nb > 1:
        h0 = np.repeat(h0, nb)
    if len(h0) != nb:
        raise ValueError(f"history_start needs one timestamp per building ({nb}), got {len(h0)}")
    return h0


def hour_of_week_profile(x, start) -> np.ndarray:
    """(168,): mean of the 1-D hourly history ``x`` (first hour at ``start``) per hour of the week (Monday 00h = 0),
    divided by the overall mean of ``x`` (hours of the week that do not occur give 0)."""
    x = np.asarray(x, dtype=float).ravel()
    hrs = _starts(start, 1)[0] + np.arange(len(x), dtype=np.int64)
    how = ((hrs // 24 + 3) % 7) * 24 + hrs % 24
    s = np.bincount(how, weights=x, minlength=168) / np.maximum(np.bincount(how, minlength=168), 1)
    return s / max(float(x.mean()), 1e-6)


def _block_profiles(x: np.ndarray, h0: int, folds: int) -> tuple[np.ndarray, np.ndarray]:
    """(folds, 168) hour-of-week profiles of the 1-D history ``x`` that leave out one of ``folds`` equal time blocks each,
    and the (folds + 1,) block edges (indices into ``x``). ``folds`` = 1 gives the profile of the whole history."""
    T = len(x)
    edges = np.linspace(0, T, folds + 1).astype(np.int64)
    hrs = h0 + np.arange(T, dtype=np.int64)
    how = ((hrs // 24 + 3) % 7) * 24 + hrs % 24
    tot_s, tot_c = np.bincount(how, weights=x, minlength=168), np.bincount(how, minlength=168).astype(float)
    out = np.zeros((folds, 168))
    for j in range(folds):
        lo, hi = edges[j], edges[j + 1]
        if folds == 1:
            s, c, n, m = tot_s, tot_c, T, x.sum()
        else:
            s = tot_s - np.bincount(how[lo:hi], weights=x[lo:hi], minlength=168)
            c = tot_c - np.bincount(how[lo:hi], minlength=168)
            n, m = T - (hi - lo), x.sum() - x[lo:hi].sum()
        out[j] = s / np.maximum(c, 1.0) / max(m / max(n, 1), 1e-6)
    return out, edges


# ------------------------------------------------------------------------------------------------ features
def _design(context, target_start, residential: np.ndarray, profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale-free feature rows (n * 24, len(FEATURE_NAMES)) ordered (window, target hour), and the (n,) context means."""
    ctx = _context(context)
    n = len(ctx)
    scale = np.maximum(ctx.mean(axis=1), 1e-6)
    c = ctx / scale[:, None]
    lags = lag_matrix(c).reshape(n * 24, 7)
    cal = calendar_features(target_start)
    if cal["hour"].shape[0] != n:
        raise ValueError(f"target_start has {cal['hour'].shape[0]} entries for {n} context rows")
    prof = np.asarray(profile, dtype=float).reshape(n, 168)
    how = np.take_along_axis(prof, cal["hour_of_week"], axis=1).ravel()
    rep = lambda a: np.repeat(a, 24)                                  # noqa: E731  (per window -> per target hour)
    X = np.column_stack([cal["hour"].ravel(), cal["dayofweek"].ravel(), cal["weekend"].ravel(), lags,
                         np.median(lags, 1), lags.mean(1), lags.min(1), lags.max(1), lags.std(1), how,
                         rep(c[:, -1]), rep(c[:, -24:].mean(1)), rep(c.std(1)), rep(residential.astype(float))])
    return X.astype(float), scale


# ------------------------------------------------------------------------------------------------ learned model
class LoadForecaster:
    """Pooled LightGBM + Ridge day-ahead model trained on sliding windows of the buildings' own histories.

    ``stride`` hours between training windows, LightGBM ``n_estimators`` / ``num_leaves`` / ``min_child_samples`` /
    ``learning_rate``, Ridge ``ridge_alpha``, ``n_jobs`` (capped at 2), ``seed``, ``profile_folds``: the 'profile' feature of a
    training row is the building's hour-of-week profile computed without the time block (one of ``profile_folds`` equal
    blocks of the history) that holds the row's target hours; 1 = profile of the whole history. Forecast rows always use the
    whole-history profile. Call :meth:`fit`, then :meth:`candidates` / :meth:`predict`."""

    def __init__(self, stride: int = 6, n_estimators: int = 120, num_leaves: int = 15, min_child_samples: int = 200,
                 learning_rate: float = 0.05, ridge_alpha: float = 30.0, n_jobs: int = 2, seed: int = 0,
                 profile_folds: int = 4):
        self.stride, self.n_estimators, self.num_leaves = int(stride), int(n_estimators), int(num_leaves)
        self.min_child_samples, self.learning_rate = int(min_child_samples), float(learning_rate)
        self.ridge_alpha, self.n_jobs, self.seed = float(ridge_alpha), max(1, min(int(n_jobs), 2)), int(seed)
        self.profile_folds = max(1, int(profile_folds))

    def fit(self, load, history_start, building_id, category) -> "LoadForecaster":
        import lightgbm as lgb
        from sklearn.linear_model import Ridge
        x = _load_matrix(load)
        nb = len(x)
        bid = [str(b) for b in building_id]
        if len(bid) != nb or len(set(bid)) != nb:
            raise ValueError("building_id needs one distinct entry per row of load")
        res = _is_residential(category, nb)
        h0 = _starts(history_start, nb)
        w = history_windows(x, h0.astype("datetime64[h]"), bid, category, stride=self.stride)
        blocks = [_block_profiles(x[b], int(h0[b]), self.profile_folds) for b in range(nb)]
        prof = np.stack([hour_of_week_profile(x[b], h0[b:b + 1].astype("datetime64[h]")) for b in range(nb)])
        bi = w["building_index"]
        st = _window_starts(x.shape[1], self.stride, 0, None)
        train_prof = np.empty((len(bi), 168))
        for b in range(nb):
            edges = blocks[b][1]
            fold = np.clip(np.searchsorted(edges, st, side="right") - 1, 0, self.profile_folds - 1)
            train_prof[b * len(st):(b + 1) * len(st)] = blocks[b][0][fold]
        X, scale = _design(w["context"], w["target_start"], res[bi], train_prof)
        y = (w["target"] / scale[:, None]).ravel()
        mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-6
        self.lgb_ = lgb.LGBMRegressor(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate, num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=5.0, n_jobs=self.n_jobs, random_state=self.seed, deterministic=True, force_row_wise=True,
            verbose=-1).fit(X, y)
        self.ridge_ = Ridge(alpha=self.ridge_alpha).fit((X - mu) / sd, y)
        self.mu_, self.sd_ = mu, sd
        self.building_id_ = bid
        self.profile_ = {b: prof[i] for i, b in enumerate(bid)}
        self.n_train_windows_ = len(w["context"])
        return self

    def _profiles(self, building_id, n: int) -> np.ndarray:
        if not hasattr(self, "lgb_"):
            raise RuntimeError("call fit(...) first")
        ids = [str(b) for b in building_id]
        if len(ids) != n:
            raise ValueError(f"building_id has {len(ids)} entries for {n} context rows")
        unknown = sorted(set(ids) - set(self.profile_))
        if unknown:
            raise ValueError(f"buildings {unknown[:5]} were not passed to fit (fitted: {self.building_id_[:8]} ...)")
        return np.stack([self.profile_[b] for b in ids])

    def ml_forecast(self, context, target_start, building_id, category) -> np.ndarray:
        """(n, 24) kWh, >= 0: mean of the LightGBM and Ridge predictions (rescaled by each context mean)."""
        ctx = _context(context)
        X, scale = _design(ctx, target_start, _is_residential(category, len(ctx)), self._profiles(building_id, len(ctx)))
        pred = (self.lgb_.predict(X) + self.ridge_.predict((X - self.mu_) / self.sd_)).reshape(len(ctx), 24) / 2.0
        return np.maximum(pred * scale[:, None], 0.0)

    def candidates(self, context, target_start, building_id, category, w_core: float = 0.5, pretrained: bool = False) -> dict:
        """Dict name -> (n, 24) kWh arrays (>= 0) for the names in ``CANDIDATES`` (and ``PRETRAINED_CANDIDATES`` when
        ``pretrained``); see the module docstring."""
        ctx = _context(context)
        core = core_forecast(ctx, category)
        ml = self.ml_forecast(ctx, target_start, building_id, category)
        w = float(w_core)
        out = {"yesterday": yesterday(ctx), "mean7": day_mean(ctx), "median7": day_median(ctx), "core": core,
               "ml": ml, "ens": np.maximum(w * core + (1.0 - w) * ml, 0.0)}
        if pretrained:
            from . import tsfm
            c2 = np.maximum(tsfm.forecast(list(ctx), HORIZON_H, quantiles=(0.5,), model="chronos_2")[:, :, 0].astype(float), 0.0)
            out["chronos2"] = c2
            out["ens_chronos2"] = 0.5 * out["ens"] + 0.5 * c2
        return out

    def predict(self, context, target_start, building_id, category, w_core: float = 0.5) -> np.ndarray:
        """``candidates(...)['ens']``."""
        return self.candidates(context, target_start, building_id, category, w_core)["ens"]


def fit(load, history_start, building_id, category, **kwargs) -> LoadForecaster:
    """``LoadForecaster(**kwargs).fit(load, history_start, building_id, category)``."""
    return LoadForecaster(**kwargs).fit(load, history_start, building_id, category)


def forecast_candidates(load, history_start, history_building_id, history_category, context, target_start, building_id,
                        category, model: LoadForecaster | None = None, w_core: float = 0.5, pretrained: bool = False, **kwargs) -> dict:
    """Fit on the histories (or reuse ``model``) and return ``candidates`` for the windows (context, target_start, ...)."""
    m = model if model is not None else fit(load, history_start, history_building_id, history_category, **kwargs)
    return m.candidates(context, target_start, building_id, category, w_core, pretrained)


def backtest_history(load, history_start, building_id, category, holdout_hours: int = 720, stride: int = 24,
                     w_core: float = 0.5, pretrained: bool = False, **kwargs) -> dict:
    """Day-ahead check inside the histories: fit on load[:, :T - holdout_hours], forecast every ``stride``-th window whose
    24 target hours lie in the last ``holdout_hours`` and return {'scores': {name: balanced CVRMSE (%)}, 'n_windows': int,
    'per_building': {name: {building: CVRMSE (%)}}}. The contexts of those windows may reach into the fitted part.
    Buildings whose holdout windows have no positive load (CVRMSE undefined) are left out."""
    x = _load_matrix(load)
    T = x.shape[1]
    cut = T - int(holdout_hours)
    if cut < CONTEXT_H + HORIZON_H + 1:
        raise ValueError(f"holdout_hours={holdout_hours} leaves too little of the {T}-h history to fit")
    m = fit(x[:, :cut], history_start, building_id, category, **kwargs)
    w = history_windows(x, history_start, building_id, category, stride=stride, first_hour=cut - CONTEXT_H)
    cands = m.candidates(w["context"], w["target_start"], w["building_id"], w["category"], w_core, pretrained)
    # a building without any load in the holdout has an undefined CVRMSE: leave it out of the scores
    bid = np.asarray(w["building_id"]).astype(str)
    tgt = np.asarray(w["target"], dtype=float)
    keep = np.array([tgt[bid == u].mean() > 0 for u in bid])
    if not keep.any():
        raise ValueError("no building has positive load in the holdout windows")
    cands = {k: v[keep] for k, v in cands.items()}
    w = {k: (np.asarray(v)[keep] if len(v) == len(keep) else v) for k, v in w.items()}
    per = {k: cvrmse_by_building(w["target"], v, w["building_id"]) for k, v in cands.items()}
    scores = {k: balanced_cvrmse(w["target"], v, w["building_id"], w["category"]) for k, v in cands.items()}
    return {"scores": scores, "n_windows": int(keep.sum()), "per_building": per}


def pick_lowest(candidates: dict, scores: dict, prefer: str = "ens", rel_tol: float = 0.03):
    """(name, array) of the candidate with the lowest ``scores[name]`` (names missing from ``scores`` are ignored);
    if ``prefer`` scores within ``rel_tol`` (relative) of the lowest, ``prefer`` is returned instead."""
    names = [k for k in candidates if k in scores]
    if not names:
        raise ValueError("no candidate has a score")
    best = min(names, key=lambda k: scores[k])
    if prefer in scores and prefer in candidates and scores[prefer] <= scores[best] * (1.0 + float(rel_tol)):
        best = prefer
    return best, candidates[best]
