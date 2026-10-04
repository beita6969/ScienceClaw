"""Univariate / panel time-series forecasting: metrics, local models, rolling-origin backtest, combination, global window models.

A series is a 1-D float array (oldest value first, no NaN); ``horizon`` = steps to forecast; ``period`` = observations per season
(12 monthly, 4 quarterly, 1 none). A panel is a list of series (lengths may differ). Forecasts: (horizon,) per series, (n, horizon)
per panel.

fit_predict(histories, horizon, period, methods=None, how=None, weights=None, nonneg=True, n_jobs=2, train=None,
            global_models=None, phase=None, train_phase=None, train_cut="auto", pretrained=0.0, pretrained_model="chronos_2")
            -> (n, horizon)
    forecast_panel(local methods) + global_window(global models) -> combine -> clip at 0 if ``nonneg``. ``methods=None`` ->
    DEFAULT_METHODS, ``global_models=None`` -> GLOBAL_MODELS (``()`` switches either group off), ``how=None`` -> 'median'. The global
    models are fitted on ``train`` (a list of series; default: ``histories`` themselves). ``phase`` / ``train_phase``: season position
    of the first observation of each history / train series. ``train_cut``: "auto" (default) truncates every train series that extends a
    history (the history equals its first observations, e.g. a dev history whose full series is in ``train``) to the length of that
    history, so no value after the end of a history is used to forecast it; an int n drops the trailing n observations of every train
    series (0 = train as given). Example: ``fit_predict(hist, 24, 12,
    train=train_series)``; with the pretrained forecast mixed in at weight 0.5: ``fit_predict(hist, 24, 12,
    train=train_series, pretrained=0.5)``.
    ``pretrained`` (default 0.0; a weight w in [0, 1]) mixes the result with ``pretrained_forecast``: (1 - w) * (the combined
    forecast above, after the clip at 0) + w * pretrained_forecast(histories, horizon, pretrained_model). Needs ``scilib.tsfm``
    (a GPU worker); w = 0 makes no call to it.
pretrained_forecast(histories, horizon, model="chronos_2", context=120) -> (n, horizon)
    Median forecast of a pretrained time-series model (``scilib.tsfm``; ``model`` = "chronos_2" or "chronos_bolt") from the last
    ``context`` observations of each series, one call for the whole panel. A series whose values are all >= 0 is passed through
    log1p and the forecast back through expm1; other series are passed as they are. No season length is given to the model.
forecast_panel(histories, horizon, period, methods=None, n_jobs=2) -> {method: (n, horizon)}    (names from METHODS or callables)
combine(forecasts, how="mean"|"median"|"trimmed", weights=None) -> (n, horizon)    element-wise over {name: (n, horizon)}
backtest_panel(histories, horizon, period, methods=None, n_origins=2, step=None, n_jobs=2, global_models=None, train=None,
               phase=None, train_phase=None, max_series=48, seed=0) -> dict
    rolling origins: origin j = 0..n_origins-1 drops the last horizon + j*step values (step = horizon by default) of every series
    (of ``train`` too, default: the histories are the pool), forecasts the dropped values of up to ``max_series`` (seeded subset)
    histories from the rest with the local ``methods`` and with the global models fitted on the cut pool, scores MASE. Members =
    local methods + global models, named as in fit_predict. Keys: mase {member: mean over series and origins}, per_origin,
    forecasts {member: [(n_used, horizon) per origin]}, actuals, insamples, index, ranked (members by mase), period. Members of
    one origin share their rows; the row count can differ between origins (short series are skipped). Backtesting a panel
    of many series (e.g. every training series) against itself needs no ``train`` argument.
combination_mase(bt, members=None, how=None, weights=None) -> float    MASE of a combination of backtested members (None = all)
inverse_error_weights({method: error}, power=1.0) -> {method: weight}    weights proportional to error ** -power, sum 1
global_window(train, targets, horizon, period, models=GLOBAL_MODELS, n_lags=None, max_windows=60, train_phase=None,
              target_phase=None, n_estimators=60, seed=0, n_jobs=2) -> {model: (len(targets), horizon)}
    GLOBAL_MODELS = ("ridge", "extra_trees"): sklearn Ridge(alpha=1) / ExtraTreesRegressor(``n_estimators`` trees,
    min_samples_leaf=5, max_features=0.5), multi-output over the horizon, fitted on windows cut from every series of ``train``
    (all its observed values; the last ``max_windows`` window ends per series; a window needs ``horizon`` values after it). Window =
    last ``n_lags`` values (default max(3*period, 12)) divided by the mean of their last season, one-hot of the season position of
    the first forecast step, log1p(level); target = next ``horizon`` values divided by the same level; sample weight = level /
    seasonal scale of its series, divided by the median and clipped to [0.1, 10]. ``train_phase`` / ``target_phase``: season
    position (0..period-1, e.g. month - 1) of each series' first observation (default 0). Targets shorter than ``n_lags`` are wrap-padded.
    Forecasts clipped at 0. Every observed value of ``train`` is used, so to score a held-out tail pass train series cut at
    that origin. About 3-6 CPU-s for ~270 series of ~300 values, horizon 24, period 12 (ridge < 1 s, extra_trees the rest).
global_lgbm(train, targets, horizon, period, n_lags=None, stride=None, n_estimators=150, seed=0, n_jobs=2) -> (len(targets), horizon)
    one LightGBM (L1 loss) per horizon step on windows cut from every series of ``train``; inputs: last ``n_lags`` values minus the
    mean of the last season, divided by the series' seasonal scale, plus level/scale and last-season std/scale; ``stride`` = spacing
    of window ends. About one CPU-minute for ~250 series of ~300 values, horizon 24.

Metrics (``insample`` = the series the scale is taken from)
  seasonal_scale(insample, period) = mean_{t>period} |x_t - x_{t-period}|;  mase(y_true, y_pred, insample, period) = mean_h |y_h - p_h|
  / seasonal_scale;  smape(y_true, y_pred) = 200/H * sum_h |y_h - p_h| / (|y_h| + |p_h|) in percent (0/0 term = 0);
  rmsse(y_true, y_pred, insample, period) = sqrt(mean_h (y_h - p_h)^2 / mean_{t>period} (x_t - x_{t-period})^2);
  panel_mase(y_true, y_pred, insamples, period) = mean of mase over the series of a panel.

Single-series methods ``f(x, horizon, period) -> (horizon,)`` (METHODS maps name -> function, method_help() the descriptions); a
method falls back to repeat_season when the series is too short, a fit fails or its forecast is implausible:
  repeat_last, repeat_season                final observation / final season repeated
  seasonal_average             per-season mean of the last 3 seasons (seasonal_average_k: another count)
  repeat_season_drift  repeat_season plus the damped mean season-over-season change of the last 2 seasons
  ses, holt_damped, theta      simple exp. smoothing / damped-trend Holt / classical Theta (SES + half the OLS slope) on the seasonally
                               adjusted series (classical decomposition, multiplicative if all values > 0, applied when the lag-period
                               autocorrelation test at 90 % is significant)
  ets                          statsmodels ETSModel: candidates (error, trend, seasonal) = (A,N,A), (A,Ad,A), (M,N,M), (M,Ad,M), lowest AICc
  ets_damped_add               ETSModel (A,Ad,A): additive error, damped additive trend, additive seasonality
  ets_damped_mul               ETSModel (A,Ad,M), heuristic initial states; additive seasonality if a value is <= 0
  stl_ets                      robust STL; seasonally adjusted part by damped-trend ETS, last seasonal cycle repeated (log scale if all > 0)
  sarima_airline               SARIMAX (0,1,1)(0,1,1)_period on log values if all > 0, else raw values
DEFAULT_METHODS = ("ets_damped_add", "theta", "stl_ets", "sarima_airline"). Cost per series (CPU): repeat_season/theta/seasonal_average ~0 s, stl_ets ~0.05 s,
ets_damped_add/ets_damped_mul ~0.05 s, ets and sarima_airline ~0.3 s.
"""
from __future__ import annotations

import warnings

import numpy as np

__all__ = ["seasonal_scale", "mase", "smape", "rmsse", "panel_mase", "repeat_last", "repeat_season", "seasonal_average", "seasonal_average_k",
           "repeat_season_drift", "ses", "holt_damped", "theta", "ets", "ets_damped_add", "ets_damped_mul", "stl_ets", "sarima_airline",
           "METHODS", "DEFAULT_METHODS", "method_help", "forecast_panel", "combine", "backtest_panel",
           "combination_mase", "inverse_error_weights", "fit_predict", "global_lgbm", "global_window", "GLOBAL_MODELS", "pretrained_forecast"]


# ------------------------------------------------------------------------------------------------ metrics
def _arr(x) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    return a


def seasonal_scale(insample, period: int = 1) -> float:
    """mean_{t>period} |x_t - x_{t-period}| of the finite values of ``insample`` (``period`` = 1: lag-1 difference scale)."""
    x = _arr(insample)
    x = x[np.isfinite(x)]
    m = max(int(period), 1)
    if x.size <= m:
        raise ValueError(f"in-sample length {x.size} <= period {m}")
    return float(np.mean(np.abs(x[m:] - x[:-m])))


def mase(y_true, y_pred, insample, period: int = 1) -> float:
    """Mean absolute scaled error (Hyndman & Koehler 2006): mean_h |y_h - p_h| / seasonal_scale(insample, period).
    Raises ValueError if the scale is 0."""
    s = seasonal_scale(insample, period)
    if s <= 0:
        raise ValueError("MASE scale is zero (constant seasonal differences)")
    a, p = _arr(y_true), _arr(y_pred)
    if a.shape != p.shape:
        raise ValueError(f"mase: y_true has {a.size} values, y_pred has {p.size}")
    return float(np.mean(np.abs(a - p)) / s)


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in percent (M4 definition): 200/H * sum_h |y_h - p_h| / (|y_h| + |p_h|); a 0/0 term counts as 0."""
    y, p = _arr(y_true), _arr(y_pred)
    if y.shape != p.shape:
        raise ValueError(f"smape: y_true has {y.size} values, y_pred has {p.size}")
    den = np.abs(y) + np.abs(p)
    terms = np.where(den > 0, np.abs(y - p) / np.where(den > 0, den, 1.0), 0.0)
    return float(200.0 * np.mean(terms))


def rmsse(y_true, y_pred, insample, period: int = 1) -> float:
    """Root mean squared scaled error (M5 definition): sqrt(mean_h (y_h - p_h)^2 / mean_{t>period} (x_t - x_{t-period})^2)."""
    x = _arr(insample)
    x = x[np.isfinite(x)]
    m = max(int(period), 1)
    if x.size <= m:
        raise ValueError(f"in-sample length {x.size} <= period {m}")
    den = float(np.mean((x[m:] - x[:-m]) ** 2))
    if den <= 0:
        raise ValueError("RMSSE scale is zero (constant seasonal differences)")
    return float(np.sqrt(np.mean((_arr(y_true) - _arr(y_pred)) ** 2) / den))


def panel_mase(y_true, y_pred, insamples, period: int = 1) -> float:
    """Mean over series of ``mase`` (``y_true`` / ``y_pred``: (n, horizon) arrays or lists, ``insamples``: list of n series)."""
    n = len(insamples)
    if len(y_true) != n or len(y_pred) != n:
        raise ValueError(f"panel_mase: {len(y_true)} y_true rows, {len(y_pred)} y_pred rows, {n} insample series (must be equal)")
    return float(np.mean([mase(a, b, c, period) for a, b, c in zip(y_true, y_pred, insamples)]))


# ------------------------------------------------------------------------------------------------ helpers
def _hz(horizon) -> int:
    h = int(horizon)
    if h < 1:
        raise ValueError("horizon must be >= 1")
    return h


def _pd(period) -> int:
    return max(int(period), 1)


def _check_series(x, name: str = "series") -> np.ndarray:
    """1-D float copy of ``x``; ValueError naming ``name`` when it is empty or has non-finite values."""
    a = _arr(x)
    if a.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} has {int(np.sum(~np.isfinite(a)))} non-finite value(s)")
    return a


def _sane(x: np.ndarray, f: np.ndarray, horizon: int) -> bool:
    """A forecast is usable when it has the right length, is finite and stays within 5 x the historical range around the history."""
    if f is None or np.shape(f) != (horizon,) or not np.all(np.isfinite(f)):
        return False
    hi, lo = float(np.max(x)), float(np.min(x))
    span = max(hi - lo, abs(hi), 1.0)
    return bool(np.max(f) <= hi + 5.0 * span and np.min(f) >= lo - 5.0 * span)


def _guarded(fn):
    """Wrap a method: bad input or a failed / implausible fit yields ``repeat_season`` instead of an exception."""
    def wrapped(x, horizon, period=1, **kw):
        x = _arr(x)
        horizon, period = _hz(horizon), _pd(period)
        if x.size < 2 or not np.all(np.isfinite(x)):
            raise ValueError("series must have >= 2 finite values")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f = np.asarray(fn(x, horizon, period, **kw), dtype=float)
        except Exception:
            f = None
        return f if f is not None and _sane(x, f, horizon) else repeat_season(x, horizon, period)
    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    return wrapped


# ------------------------------------------------------------------------------------------------ simple methods
def repeat_last(x, horizon: int, period: int = 1) -> np.ndarray:
    """Repeat the last observation."""
    x = _arr(x)
    return np.full(_hz(horizon), x[-1])


def repeat_season(x, horizon: int, period: int = 1) -> np.ndarray:
    """Repeat the last ``period`` observations (repeat_last when the series is shorter than one season)."""
    x = _arr(x)
    horizon, m = _hz(horizon), _pd(period)
    if x.size < m:
        return np.full(horizon, x[-1])
    return np.array([x[x.size - m + (h % m)] for h in range(horizon)])


def seasonal_average_k(x, horizon: int, period: int = 1, k: int = 3) -> np.ndarray:
    """Forecast for step h = mean of the same-season values in the last ``k`` complete seasons."""
    x = _arr(x)
    horizon, m = _hz(horizon), _pd(period)
    k = max(1, min(int(k), x.size // m))
    if x.size < m:
        return np.full(horizon, x[-1])
    last = x[x.size - k * m:].reshape(k, m).mean(axis=0)
    return np.array([last[h % m] for h in range(horizon)])


def seasonal_average(x, horizon: int, period: int = 1) -> np.ndarray:
    """Per-season mean over the last 3 seasons, repeated (see ``seasonal_average_k``)."""
    return seasonal_average_k(x, horizon, period, 3)


def repeat_season_drift(x, horizon: int, period: int = 1) -> np.ndarray:
    """repeat_season + d_j where d_j = (mean over the last 2 seasons of the season-over-season change) * (season index j), the drift
    phi-damped (phi = 0.8 per season) so that it flattens out; only for series with at least 3 seasons."""
    x = _arr(x)
    horizon, m = _hz(horizon), _pd(period)
    base = repeat_season(x, horizon, m)
    if x.size < 3 * m:
        return base
    d = float(np.mean(x[-2 * m:] - x[-3 * m:-m]))
    steps = np.arange(horizon) // m + 1
    phi = 0.8
    return base + d * np.cumsum(phi ** np.arange(1, steps.max() + 1))[steps - 1]


# ------------------------------------------------------------------------------------------------ seasonal adjustment
def _acf(x: np.ndarray, nlags: int) -> np.ndarray:
    xc = x - x.mean()
    den = float(np.sum(xc * xc))
    if den <= 0:
        return np.zeros(nlags + 1)
    return np.array([np.sum(xc[k:] * xc[:x.size - k]) / den for k in range(nlags + 1)])


def _is_seasonal(x: np.ndarray, m: int) -> bool:
    """90 % autocorrelation test at lag m (M4 benchmark convention); needs at least 3 seasons."""
    if m <= 1 or x.size < 3 * m:
        return False
    r = _acf(x, m)
    lim = 1.645 * np.sqrt((1.0 + 2.0 * np.sum(r[1:m] ** 2)) / x.size)
    return bool(abs(r[m]) > lim)


def _seasonal_indices(x: np.ndarray, m: int, mult: bool) -> np.ndarray:
    """Classical decomposition indices (centred moving average); index p applies to time t with t % m == p."""
    w = np.r_[0.5, np.ones(m - 1), 0.5] / m if m % 2 == 0 else np.ones(m) / m
    ma = np.convolve(x, w, mode="valid")
    off = (w.size - 1) // 2
    seg = x[off:off + ma.size]
    ratio = seg / ma if mult else seg - ma
    pos = (np.arange(ma.size) + off) % m
    idx = np.array([np.mean(ratio[pos == p]) if np.any(pos == p) else (1.0 if mult else 0.0) for p in range(m)])
    return idx / idx.mean() if mult else idx - idx.mean()


def _deseasonalise(x: np.ndarray, m: int, force: bool | None = None):
    """(adjusted series, function h-array -> seasonal factor/offset for steps after the end, mult flag) or the series itself
    when no seasonality is detected (``force=True/False`` overrides the test)."""
    seasonal = _is_seasonal(x, m) if force is None else bool(force and m > 1 and x.size >= 2 * m)
    if not seasonal:
        return x, (lambda h: np.ones(np.size(h))), True
    mult = bool(np.all(x > 0))
    idx = _seasonal_indices(x, m, mult)
    n = x.size
    if mult:
        return x / idx[np.arange(n) % m], (lambda h: idx[(n + np.asarray(h)) % m]), True
    return x - idx[np.arange(n) % m], (lambda h: idx[(n + np.asarray(h)) % m]), False


def _ses_fit(y: np.ndarray):
    """(alpha, last level) of simple exponential smoothing with the initial level at y[0] and alpha minimising the one-step SSE."""
    from scipy.optimize import minimize_scalar
    from scipy.signal import lfilter

    def levels(a):
        return lfilter([a], [1.0, -(1.0 - a)], y, zi=[(1.0 - a) * y[0]])[0]

    def sse(a):
        lv = levels(a)
        e = y[1:] - lv[:-1]
        return float(np.sum(e * e))

    res = minimize_scalar(sse, bounds=(0.01, 0.99), method="bounded", options={"xatol": 1e-4})
    a = float(res.x)
    return a, float(levels(a)[-1])


def _adjusted_call(x, horizon, m, core):
    """Deseasonalise, forecast the adjusted series with ``core(adj, horizon)`` and re-apply the seasonal factors."""
    seasonal = _is_seasonal(x, m)
    adj, sfun, mult = _deseasonalise(x, m)
    fc = core(adj, horizon)
    if not seasonal:
        return fc
    s = sfun(np.arange(horizon))
    return fc * s if mult else fc + s


@_guarded
def ses(x, horizon: int, period: int = 1) -> np.ndarray:
    """Simple exponential smoothing (alpha by one-step SSE) on the seasonally adjusted series, flat forecast, re-seasonalised."""
    def core(adj, h):
        return np.full(h, _ses_fit(adj)[1])
    return _adjusted_call(x, horizon, period, core)


@_guarded
def theta(x, horizon: int, period: int = 1) -> np.ndarray:
    """Classical Theta (Assimakopoulos & Nikolopoulos 2000; Hyndman & Billah 2003): SES forecast plus half the OLS trend slope
    b/2 * (h - 1 + 1/alpha - (1-alpha)^n / alpha), on the seasonally adjusted series, then re-seasonalised."""
    def core(adj, h):
        n = adj.size
        a, level = _ses_fit(adj)
        t = np.arange(n, dtype=float)
        b = float(np.polyfit(t, adj, 1)[0])
        steps = np.arange(h)
        return level + 0.5 * b * (steps + 1.0 / a - (1.0 - a) ** n / a)
    return _adjusted_call(x, horizon, period, core)


@_guarded
def holt_damped(x, horizon: int, period: int = 1) -> np.ndarray:
    """Damped-trend Holt (statsmodels ExponentialSmoothing, additive trend, damped) on the seasonally adjusted series."""
    def core(adj, h):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        r = ExponentialSmoothing(adj, trend="add", damped_trend=True, initialization_method="estimated").fit()
        return np.asarray(r.forecast(h))
    return _adjusted_call(x, horizon, period, core)


# ------------------------------------------------------------------------------------------------ ETS family
_ETS_CANDIDATES = (  # (error, trend, damped, seasonal)
    ("add", None, False, "add"), ("add", "add", True, "add"),
    ("mul", None, False, "mul"), ("mul", "add", True, "mul"),
)


def _ets_fit(x: np.ndarray, m: int, cfg, maxiter: int = 200, init: str = "estimated"):
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    err, trend, damped, seas = cfg
    if m <= 1:
        seas = None
    elif x.size < 2 * m + 2:
        return None
    if (err == "mul" or seas == "mul") and not np.all(x > 0):
        return None
    mod = ETSModel(x, error=err, trend=trend, damped_trend=damped, seasonal=seas, seasonal_periods=m if seas else None,
                   initialization_method=init)
    return mod.fit(disp=False, maxiter=maxiter)


def _ets_best(x, horizon, m, cands, init: str = "estimated"):
    best, best_f = np.inf, None
    for cfg in cands:
        try:
            r = _ets_fit(x, m, cfg, init=init)
            if r is None:
                continue
            f = np.asarray(r.forecast(horizon), dtype=float)
            a = float(r.aicc)
        except Exception:
            continue
        if np.isfinite(a) and _sane(x, f, horizon) and a < best:
            best, best_f = a, f
    return best_f


@_guarded
def ets(x, horizon: int, period: int = 1) -> np.ndarray:
    """statsmodels ETSModel; candidates (error, trend, seasonal) = (A,N,A), (A,Ad,A), (M,N,M), (M,Ad,M) (multiplicative ones only for
    strictly positive series; without seasonality the seasonal term is dropped); the lowest AICc gives the forecast."""
    return _ets_best(x, horizon, period, _ETS_CANDIDATES)


@_guarded
def ets_damped_add(x, horizon: int, period: int = 1) -> np.ndarray:
    """ETS(A, Ad, A): additive error, damped additive trend, additive seasonality (no seasonal term when period = 1)."""
    return _ets_best(x, horizon, period, (("add", "add", True, "add"),))


@_guarded
def ets_damped_mul(x, horizon: int, period: int = 1) -> np.ndarray:
    """ETS(A, Ad, M) with heuristic initial states: additive error, damped additive trend, multiplicative seasonality (additive
    seasonality when the series contains a value <= 0; no seasonal term when period = 1)."""
    seas = "mul" if np.all(x > 0) else "add"
    return _ets_best(x, horizon, period, (("add", "add", True, seas),), init="heuristic")


@_guarded
def stl_ets(x, horizon: int, period: int = 1) -> np.ndarray:
    """Robust STL (statsmodels) into trend+remainder and seasonal parts; the seasonally adjusted series is forecast by ETS(A,Ad,N)
    (SES if that fails) and the last seasonal cycle is repeated; fitted on log values when the series is strictly positive."""
    from statsmodels.tsa.seasonal import STL
    if period <= 1 or x.size < 2 * period + 2:
        return _ets_best(x, horizon, 1, (("add", "add", True, None),))
    use_log = bool(np.all(x > 0))
    y = np.log(x) if use_log else x
    comp = STL(y, period=period, robust=True).fit()
    seas = np.asarray(comp.seasonal)
    adj = y - seas
    f = _ets_best(adj, horizon, 1, (("add", "add", True, None), ("add", None, False, None)))
    if f is None:
        f = np.full(horizon, _ses_fit(adj)[1])
    cyc = seas[-period:]
    s = np.array([cyc[h % period] for h in range(horizon)])
    out = f + s
    return np.exp(out) if use_log else out


@_guarded
def sarima_airline(x, horizon: int, period: int = 1) -> np.ndarray:
    """SARIMAX(0,1,1)(0,1,1)_period (no constant) by maximum likelihood on log values when strictly positive (else raw values)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    use_log = bool(np.all(x > 0))
    y = np.log(x) if use_log else x
    if period <= 1 or x.size < 3 * period:
        mod = SARIMAX(y, order=(0, 1, 1), trend="n")
    else:
        mod = SARIMAX(y, order=(0, 1, 1), seasonal_order=(0, 1, 1, period), trend="n")
    r = mod.fit(disp=False, maxiter=100)
    f = np.asarray(r.forecast(horizon))
    return np.exp(f) if use_log else f


METHODS = {
    "repeat_last": repeat_last, "repeat_season": repeat_season, "seasonal_average": seasonal_average, "repeat_season_drift": repeat_season_drift, "ses": ses,
    "holt_damped": holt_damped, "theta": theta, "ets": ets, "ets_damped_add": ets_damped_add, "ets_damped_mul": ets_damped_mul, "stl_ets": stl_ets,
    "sarima_airline": sarima_airline,
}
DEFAULT_METHODS = ("ets_damped_add", "theta", "stl_ets", "sarima_airline")


def method_help() -> dict:
    """{name: first docstring sentence} of every entry of ``METHODS``."""
    return {k: (f.__doc__ or "").strip().split("\n")[0] for k, f in METHODS.items()}


# ------------------------------------------------------------------------------------------------ panel level
def _resolve(methods) -> dict:
    methods = DEFAULT_METHODS if methods is None else methods
    if isinstance(methods, dict):
        return {str(k): (METHODS[v] if isinstance(v, str) else v) for k, v in methods.items()}
    if isinstance(methods, str):
        methods = (methods,)
    out = {}
    for mth in methods:
        if callable(mth):
            out[getattr(mth, "__name__", str(mth))] = mth
        elif mth in METHODS:
            out[mth] = METHODS[mth]
        else:
            raise ValueError(f"unknown method {mth!r}; available: {sorted(METHODS)}")
    return out


def _run_one(fn, x, horizon, period):
    return np.asarray(fn(x, horizon, period), dtype=float)


def forecast_panel(histories, horizon: int, period: int, methods=None, n_jobs: int = 2) -> dict:
    """Forecast every series of ``histories`` with every method: ``{method: (n, horizon) array}``. ``methods``: names of
    ``METHODS`` (or callables ``f(x, horizon, period)``); ``None`` = ``DEFAULT_METHODS``. ``n_jobs`` <= 2 processes."""
    from joblib import Parallel, delayed
    fns = _resolve(methods)
    hs = [_arr(h) for h in histories]
    horizon, period = _hz(horizon), _pd(period)
    jobs = [(k, i) for k in fns for i in range(len(hs))]
    n_jobs = max(1, min(int(n_jobs), 2))
    res = None
    if n_jobs > 1 and len(jobs) > 3:
        try:
            res = Parallel(n_jobs=n_jobs)(delayed(_run_one)(fns[k], hs[i], horizon, period) for k, i in jobs)
        except Exception:                      # worker processes unavailable -> serial
            res = None
    if res is None:
        res = [_run_one(fns[k], hs[i], horizon, period) for k, i in jobs]
    out = {k: np.zeros((len(hs), horizon)) for k in fns}
    for (k, i), f in zip(jobs, res):
        out[k][i] = f
    return out


def combine(forecasts: dict, how: str = "mean", weights: dict | None = None) -> np.ndarray:
    """Combine ``{name: (n, horizon)}`` forecasts element-wise. how = 'mean' (or weighted mean with ``weights`` {name: w}, normalised
    to sum 1), 'median', or 'trimmed' (mean after dropping the largest and smallest member; falls back to the median with < 3 members)."""
    names = list(forecasts)
    if not names:
        raise ValueError("no forecasts to combine")
    arrs = {k: np.asarray(forecasts[k], dtype=float) for k in names}
    if len({a.shape for a in arrs.values()}) > 1:
        raise ValueError("combine: members must have one common shape, got " + ", ".join(f"{k}: {a.shape}" for k, a in arrs.items())
                         + " (backtest forecasts of different origins have different row counts)")
    stack = np.stack([arrs[k] for k in names])
    if weights is not None:
        w = np.array([float(weights.get(k, 0.0)) for k in names])
        if w.sum() <= 0:
            raise ValueError("weights sum to zero")
        return np.tensordot(w / w.sum(), stack, axes=1)
    if how == "mean":
        return stack.mean(axis=0)
    if how == "median":
        return np.median(stack, axis=0)
    if how == "trimmed":
        if len(names) < 3:
            return np.median(stack, axis=0)
        return np.sort(stack, axis=0)[1:-1].mean(axis=0)
    raise ValueError("how must be 'mean', 'median' or 'trimmed'")


def backtest_panel(histories, horizon: int, period: int, methods=None, n_origins: int = 2, step: int | None = None,
                   n_jobs: int = 2, global_models=None, train=None, phase=None, train_phase=None, max_series: int | None = 48,
                   seed: int = 0) -> dict:
    """Rolling-origin evaluation using only ``histories`` (and ``train``, cut at the same origin). Origin j (j = 0..n_origins-1) drops
    the last ``horizon + j * step`` observations (``step`` defaults to ``horizon``) of every series, forecasts the next ``horizon``
    values of the targets from what is left and scores them (MASE, scale taken from the part used for fitting). Members: the local
    ``methods`` (``None`` = ``DEFAULT_METHODS``) and the ``global_window`` models ``global_models`` (``None`` = ``GLOBAL_MODELS``,
    ``()`` = none), the latter fitted at each origin on the cut series of ``train`` (default: the cut ``histories``); a ``train``
    series that extends a history is first truncated to the length of that history (as with ``train_cut="auto"`` in ``fit_predict``),
    the others must end at the same time as ``histories``. ``max_series``: at most that many targets (seeded random subset of
    ``histories``, ``None`` = all) are forecast; the global models still see the whole pool. A target with fewer than
    ``2 * period + 2`` values left, or a constant seasonal difference, is skipped at that origin. Returns a dict: ``mase`` {member:
    mean over series and origins}, ``per_origin`` {member: [mean MASE per origin]}, ``forecasts`` {member: [(n_used, horizon) per
    origin]}, ``actuals`` and ``insamples`` (lists over origins), ``index`` (indices into ``histories`` used per origin; all members
    of one origin share it), ``ranked`` (members by ``mase``), ``period``."""
    fns = _resolve(methods)
    gm = _resolve_global(global_models)
    horizon, period = _hz(horizon), _pd(period)
    step = horizon if step is None else max(int(step), 1)
    hs = [_check_series(h, f"histories[{i}]") for i, h in enumerate(histories)]
    ph = None if phase is None else _phase_list(phase, len(hs), period)
    pool = hs if train is None else _limit_to_histories([_check_series(t, f"train[{i}]") for i, t in enumerate(train)], hs)
    tph = ph if train is None and train_phase is None else (None if train_phase is None else _phase_list(train_phase, len(pool), period))
    cand = list(range(len(hs)))
    if max_series is not None and len(cand) > int(max_series):
        cand = sorted(int(i) for i in np.random.RandomState(int(seed)).choice(len(hs), int(max_series), replace=False))
    minlen = max(2 * period + 2, 4)
    names = list(fns) + list(gm)
    fc, act, ins, idx, per = {k: [] for k in names}, [], [], [], {k: [] for k in names}
    for j in range(int(n_origins)):
        cut = horizon + j * step
        use = [i for i in cand if hs[i].size - cut >= minlen and _scale_ok(hs[i][:hs[i].size - cut], period)]
        if not use:
            break
        tr = [hs[i][:hs[i].size - cut] for i in use]
        te = np.stack([hs[i][hs[i].size - cut:hs[i].size - cut + horizon] for i in use])
        f = forecast_panel(tr, horizon, period, fns, n_jobs) if fns else {}
        if gm:
            f.update(global_window([p[:max(p.size - cut, 0)] for p in pool], tr, horizon, period, gm, train_phase=tph,
                                   target_phase=None if ph is None else [ph[i] for i in use], n_jobs=n_jobs))
        for k in names:
            fc[k].append(f[k])
            per[k].append(panel_mase(te, f[k], tr, period))
        act.append(te)
        ins.append(tr)
        idx.append(use)
    if not act:
        raise ValueError("series too short for a single backtest origin")
    tot = {k: float(np.mean([mase(a, p, t, period) for o in range(len(act))
                             for a, p, t in zip(act[o], fc[k][o], ins[o])])) for k in names}
    return {"mase": tot, "per_origin": per, "forecasts": fc, "actuals": act, "insamples": ins, "index": idx,
            "ranked": sorted(tot, key=tot.get), "period": period}


def _scale_ok(x, period) -> bool:
    try:
        return seasonal_scale(x, period) > 0
    except ValueError:
        return False


def combination_mase(bt: dict, members=None, how: str | None = None, weights: dict | None = None, period: int | None = None) -> float:
    """MASE of ``combine`` over the ``members`` (names in ``bt['forecasts']``, ``None`` = all) of a ``backtest_panel`` result ``bt``
    (mean over series and origins); ``how=None`` = 'median' as in ``fit_predict``. ``period`` defaults to the value stored by
    ``backtest_panel`` in ``bt['period']``."""
    period = bt.get("period", 1) if period is None else period
    members = list(bt["forecasts"]) if members is None else members
    unknown = [k for k in members if k not in bt["forecasts"]]
    if unknown:
        raise ValueError(f"unknown backtest members {unknown}; available: {list(bt['forecasts'])}")
    errs = []
    for o in range(len(bt["actuals"])):
        f = combine({k: bt["forecasts"][k][o] for k in members}, "median" if how is None else how, weights)
        errs.extend(mase(a, p, t, period) for a, p, t in zip(bt["actuals"][o], f, bt["insamples"][o]))
    return float(np.mean(errs))


def inverse_error_weights(mase_by_method: dict, power: float = 1.0) -> dict:
    """Weights proportional to ``mase ** -power`` (summing to 1) from a {method: error} dict."""
    w = {k: max(float(v), 1e-12) ** -float(power) for k, v in mase_by_method.items()}
    z = sum(w.values())
    return {k: v / z for k, v in w.items()}


def _limit_to_histories(pool, hs) -> list:
    """Truncate every series of ``pool`` that extends a history of ``hs`` (the history equals its first observations and is shorter)
    to the length of the shortest such history; other series are returned unchanged."""
    if not hs:
        return list(pool)
    key = {}
    for h in hs:
        if h.size >= 4:
            key.setdefault(h[:4].tobytes(), []).append(h)
    out = []
    for t in pool:
        lim = t.size
        if t.size > 4:
            for h in key.get(t[:4].tobytes(), ()):
                if h.size < lim and np.array_equal(t[:h.size], h):
                    lim = h.size
        out.append(t[:lim])
    return out


def _resolve_global(global_models) -> tuple:
    """``None`` -> GLOBAL_MODELS, a name -> (name,), a sequence -> tuple (empty = no global model); unknown names raise."""
    gm = GLOBAL_MODELS if global_models is None else ((global_models,) if isinstance(global_models, str) else tuple(global_models))
    bad = [k for k in gm if k not in GLOBAL_MODELS]
    if bad:
        raise ValueError(f"unknown global model {bad}; available: {list(GLOBAL_MODELS)}")
    return gm


def fit_predict(histories, horizon: int, period: int, methods=None, how: str | None = None, weights: dict | None = None,
                nonneg: bool = True, n_jobs: int = 2, train=None, global_models=None, phase=None, train_phase=None,
                train_cut: int | str = "auto", pretrained: float = 0.0, pretrained_model: str = "chronos_2") -> np.ndarray:
    """Forecast every series of ``histories``: local ``methods`` (``None`` = ``DEFAULT_METHODS``, ``()`` = none) via
    ``forecast_panel`` plus the ``global_window`` models ``global_models`` (``None`` = ``GLOBAL_MODELS``, ``()`` = none) fitted on
    ``train`` (a list of series; default: ``histories`` themselves), combined element-wise by ``combine(how)`` (``None`` = 'median'; with
    ``weights`` a weighted mean) and clipped at 0 when ``nonneg``. Returns ``(n, horizon)``. ``phase`` / ``train_phase``: season
    position of the first observation of each history / train series (default 0). ``train_cut="auto"`` truncates each ``train`` series
    that extends a history (the history equals its first observations) to the length of that history, so that the global models see no
    value later than the end of the history they forecast; an int n drops the trailing n observations of every ``train`` series
    instead (0 = ``train`` as given). Without an explicit ``train``, a pool too small to fit a global model is skipped with a warning.
    ``pretrained`` = w in [0, 1]: the result is (1 - w) * (the result above) + w * ``pretrained_forecast(histories, horizon,
    pretrained_model)``; w = 0 (default) does not call it."""
    hs = [_check_series(h, f"histories[{i}]") for i, h in enumerate(histories)]
    if not hs:
        raise ValueError("histories is empty")
    horizon, period = _hz(horizon), _pd(period)
    fns = _resolve(methods)
    gm = _resolve_global(global_models)
    fcs = forecast_panel(hs, horizon, period, fns, n_jobs) if fns else {}
    if gm:
        pool = hs if train is None else [_check_series(t, f"train[{i}]") for i, t in enumerate(train)]
        tph = phase if train is None and train_phase is None else train_phase
        if isinstance(train_cut, str):
            if train_cut != "auto":
                raise ValueError(f"train_cut must be 'auto' or an int, got {train_cut!r}")
            if train is not None:
                pool = _limit_to_histories(pool, hs)
        elif int(train_cut) > 0:
            pool = [t[:max(t.size - int(train_cut), 0)] for t in pool]
        try:
            fcs.update(global_window(pool, hs, horizon, period, gm, train_phase=tph, target_phase=phase, n_jobs=n_jobs))
        except ValueError as ex:
            if train is not None or not fns or "training windows" not in str(ex):
                raise
            warnings.warn(f"global models skipped: {ex}", RuntimeWarning, stacklevel=2)
    out = combine(fcs, "median" if how is None else how, weights)
    out = np.maximum(out, 0.0) if nonneg else out
    w = float(pretrained)
    if not 0.0 <= w <= 1.0:
        raise ValueError("pretrained must be a weight between 0 and 1")
    if w > 0.0:
        out = (1.0 - w) * out + w * pretrained_forecast(hs, horizon, pretrained_model)
    return out


def pretrained_forecast(histories, horizon: int, model: str = "chronos_2", context: int = 120) -> np.ndarray:
    """Median forecast of a pretrained time-series model (``scilib.tsfm``) from the last ``context`` observations of every series:
    ``(n, horizon)``. Series with all values >= 0 go through log1p / expm1, the others are used as they are."""
    from . import tsfm
    hs = [_check_series(h, f"histories[{i}]") for i, h in enumerate(histories)]
    if not hs:
        raise ValueError("histories is empty")
    horizon = _hz(horizon)
    ctx = int(context)
    pos = [bool(h.min() >= 0.0) for h in hs]
    xs = [np.log1p(h[-ctx:]) if ok else h[-ctx:] for h, ok in zip(hs, pos)]
    q = tsfm.forecast(xs, horizon, quantiles=(0.5,), model=model, context_length=ctx)[:, :, 0].astype(float)
    return np.stack([np.expm1(q[i]) if ok else q[i] for i, ok in enumerate(pos)])


# ------------------------------------------------------------------------------------------------ global model
def _norm_stats(x: np.ndarray, m: int):
    """(level, scale) of a window: mean of its last season and the seasonal-difference scale of the window (never 0)."""
    lvl = float(np.mean(x[-m:]))
    try:
        sc = seasonal_scale(x, m)
    except ValueError:
        sc = 0.0
    if not sc > 0:
        sc = float(np.mean(np.abs(x))) or 1.0
    return lvl, sc


def _features(win: np.ndarray, m: int):
    lvl, sc = _norm_stats(win, m)
    return np.r_[(win - lvl) / sc, lvl / sc, np.std(win[-m:]) / sc], lvl, sc


def global_lgbm(train, targets, horizon: int, period: int, n_lags: int | None = None, stride: int | None = None,
                n_estimators: int = 150, seed: int = 0, n_jobs: int = 2) -> np.ndarray:
    """One LightGBM regressor per horizon step trained on normalised windows cut from all series of ``train``; see the module
    docstring. Returns ``(len(targets), horizon)`` (series with fewer than ``n_lags`` values are left-padded with their first value)."""
    import lightgbm as lgb
    horizon, m = _hz(horizon), _pd(period)
    L = int(n_lags) if n_lags else max(3 * m, 12)
    stride = int(stride) if stride else max(1, m // 4)
    X, Y = [], []
    for s in train:
        s = _arr(s)
        for o in range(L, s.size - horizon + 1, stride):
            f, lvl, sc = _features(s[o - L:o], m)
            X.append(f)
            Y.append((s[o:o + horizon] - lvl) / sc)
    if len(X) < 20:
        raise ValueError("too few training windows; lower n_lags or supply longer series")
    X, Y = np.asarray(X), np.asarray(Y)
    T, base = [], []
    for s in targets:
        s = _arr(s)
        win = s[-L:] if s.size >= L else np.r_[np.full(L - s.size, s[0]), s]
        f, lvl, sc = _features(win, m)
        T.append(f)
        base.append((lvl, sc))
    T = np.asarray(T)
    out = np.zeros((len(T), horizon))
    for h in range(horizon):
        mdl = lgb.LGBMRegressor(objective="l1", n_estimators=int(n_estimators), learning_rate=0.05, num_leaves=15,
                                min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                random_state=int(seed), n_jobs=max(1, min(int(n_jobs), 2)), verbose=-1)
        mdl.fit(X, Y[:, h])
        out[:, h] = mdl.predict(T)
    lv = np.array([b[0] for b in base])[:, None]
    sc = np.array([b[1] for b in base])[:, None]
    return out * sc + lv


# ------------------------------------------------------------------------------------------------ global window models
GLOBAL_MODELS = ("ridge", "extra_trees")


def _phase_list(phase, n: int, m: int) -> list:
    if phase is None:
        return [0] * n
    if np.isscalar(phase):
        return [int(phase) % m] * n
    ph = [int(p) % m for p in phase]
    if len(ph) != n:
        raise ValueError(f"phase has {len(ph)} entries for {n} series")
    return ph


def _gw_row(win: np.ndarray, phase_next: int, m: int):
    lvl = max(float(np.mean(win[-m:])), 1e-6)
    hot = np.zeros(m)
    hot[int(phase_next) % m] = 1.0
    return np.r_[win / lvl, hot, np.log1p(lvl)], lvl


def global_window(train, targets, horizon: int, period: int, models=GLOBAL_MODELS, n_lags: int | None = None,
                  max_windows: int = 60, train_phase=None, target_phase=None, n_estimators: int = 60, seed: int = 0,
                  n_jobs: int = 2) -> dict:
    """Direct multi-output window regressors shared by all series: ``{model: (len(targets), horizon) array}``; see the module docstring.
    Series shorter than ``n_lags`` are wrap-padded on the left. Forecasts are clipped at 0."""
    horizon, m = _hz(horizon), _pd(period)
    L = int(n_lags) if n_lags else max(3 * m, 12)
    models = _resolve_global(models)
    tr = [_arr(s) for s in train]
    tg = [_check_series(s, f"targets[{i}]") for i, s in enumerate(targets)]
    if not tg:
        raise ValueError("targets is empty")
    for i, s in enumerate(tr):
        if not np.all(np.isfinite(s)):
            raise ValueError(f"train[{i}] has non-finite values")
    ph_tr, ph_tg = _phase_list(train_phase, len(tr), m), _phase_list(target_phase, len(tg), m)
    X, Y, W = [], [], []
    for s, p0 in zip(tr, ph_tr):
        if s.size < L + horizon:
            continue
        sc = float(np.mean(np.abs(s[m:] - s[:-m]))) if s.size > m else 0.0
        sc = sc if sc > 0 else (float(np.mean(np.abs(s))) or 1.0)
        for e in list(range(L, s.size - horizon + 1))[-int(max_windows):]:
            w = s[e - L:e]
            if np.mean(w[-m:]) <= 0:
                continue
            f, lvl = _gw_row(w, p0 + e, m)
            X.append(f)
            Y.append(s[e:e + horizon] / lvl)
            W.append(lvl / sc)
    if len(X) < 20:
        raise ValueError(f"too few training windows ({len(X)} from {len(tr)} train series; a series needs n_lags + horizon = "
                         f"{L + horizon} values and a positive last-season mean); lower n_lags or supply longer series")
    X, Y, W = np.asarray(X), np.asarray(Y), np.asarray(W)
    W = np.clip(W / np.median(W), 0.1, 10.0)
    T, lv = [], []
    for s, p0 in zip(tg, ph_tg):
        win = s[-L:] if s.size >= L else np.pad(s, (L - s.size, 0), mode="wrap")
        f, lvl = _gw_row(win, p0 + s.size, m)
        T.append(f)
        lv.append(lvl)
    T, lv = np.asarray(T), np.asarray(lv)[:, None]
    out = {}
    for k in models:
        if k == "ridge":
            from sklearn.linear_model import Ridge
            mod = Ridge(alpha=1.0).fit(X, Y, sample_weight=W)
        else:
            from sklearn.ensemble import ExtraTreesRegressor
            mod = ExtraTreesRegressor(int(n_estimators), min_samples_leaf=5, max_features=0.5, random_state=int(seed),
                                      n_jobs=max(1, min(int(n_jobs), 2))).fit(X, Y, sample_weight=W)
        out[k] = np.maximum(mod.predict(T) * lv, 0.0)
    return out
