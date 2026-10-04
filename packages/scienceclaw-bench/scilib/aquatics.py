"""Probabilistic 30-day daily forecasts of dissolved oxygen (mg/L) and water temperature (degC) at NEON aquatic sites, scored by CRPS.

Data layout (``load_eval_inputs`` / ``load_dev`` outputs): ``history`` (n, L, 3) daily means, channel 0 = oxygen, 1 = temperature,
2 = chlorophyll-a (not used here); NaN = not observed or withheld. ``history[i, t]`` is the day ``reference_date[i] - (L - 1 - t)``
days, so the last column is the reference date and the forecast covers the days reference_date + 1 .. + 30. Deliverable: a dict of
four (n, 30) float arrays ``oxygen_mu, oxygen_sigma, temperature_mu, temperature_sigma`` (normal predictive distributions).
Every function works only on the arrays it is given. ``load_dev`` and ``load_eval_inputs`` share one layout (``load_dev``'s
reference_date is 30 days earlier and the 30 days after it are inside the eval history), so the same call serves both.

fit_predict(history, reference_date, lam=0.75, smult=1.0, variables=None, horizon_days=30, pretrained=0.0, **ports) -> dict
    Whole forecast for all items at once (about 1-3 s for 16 items); keyword names equal the tool port names, so
    ``fit_predict(**inputs)`` works (``units``, ``site_id``, ``site_type`` are accepted and ignored).
    Per item and variable: (1) a smooth day-of-year profile fitted to the item's history (seasonal_profile); (2) the anomaly
    a_t = observation - profile; (3) two predictors at each origin day: fast = mean anomaly of the last 3 days (>= 1 observation),
    slow = mean anomaly of the last 30 days (>= 5 observations); (4) for each lead h = 1..30 a ridge regression (penalty 1.0) of
    the anomaly h days later on [fast, slow] over all origin days of the history that have both predictors and an observed
    target, fitted once per item ("site" fit) and once pooled over all items passed in ("pooled" fit). Forecast:
    mu = profile(day) + w_h . [fast, slow] at the last day, with w_h = lam * site + (1 - lam) * pooled (pooled only when the item
    has < 200 pairs at that lead); sigma_h = sqrt(lam * s_site^2 + (1 - lam) * s_pooled^2) * smult, s = standard deviation of the
    regression residuals at lead h, floored at 0.05. Predictors that are NaN at the last day are treated as anomaly 0. An item
    whose variable has < 60 observed days uses doy_window_forecast for that variable. mu is clipped to [0, 30] (oxygen) /
    [-5, 45] (temperature), sigma to <= 50.
    lam in [0, 1]: weight of the item's own fit against the pooled fit (0 = pooled only, 1 = item only). smult > 0: factor on sigma.
    ``pretrained`` (default 0.0; a weight w in [0, 1]): every mu and sigma array of the result becomes (1 - w) * (the value above) +
    w * (the same quantity from Chronos-2, ``scilib.tsfm`` model "chronos_2", which needs a GPU worker; w = 0 makes no call). Chronos-2
    is given the whole history of the variable (NaN = not observed) and returns the 0.25 / 0.5 / 0.75 quantiles of the 30 days:
    mu = the median clipped to [0, 45], sigma = (q75 - q25) / 1.349 clipped to [0.05, 50]. An item with < 3 observed days of a
    variable keeps the value above for that variable. Example: ``fit_predict(**inputs, pretrained=0.5)``.
    Values tried in backtests: lam 0.25 / 0.5 / 0.75 / 1.0, smult 0.9 / 1.0 / 1.1 / 1.2; the mean CRPS changed by a few percent at most
    over these ranges (see Sensitivity below). The pooled fit uses the items given in one call: with few items it approaches the item fit.

data_report(history, reference_date=None, variables=None) -> pandas.DataFrame
    One row per (item, variable): ``item``, ``variable``, ``n_obs`` (observed days in the whole history), ``n_obs_365`` (last 365
    days), ``n_obs_30`` (last 30 days), ``age_days`` (days from the last observation to the reference date; NaN if none),
    ``first_obs_age_days`` (age of the oldest observation) and ``fallback`` (True when fit_predict uses doy_window_forecast).

doy_window_forecast(history, reference_date, half_window=7, min_n=5, min_sd=0.1, variables=None) -> dict
    Deliverable dict from day-of-year windows: mu / sigma = mean / sample sd of the item's observations within +-``half_window``
    days of the target day of the year (circular, all years of the history); fewer than ``min_n`` observations -> mean / sd of the
    last 365 days (all observations if fewer than 2); sd floored at ``min_sd``. Raises ValueError if a variable has no observation.

backtest(history, reference_date, n_origins=3, offset=30, forecaster=None, **params) -> dict
    Local rolling-origin check that needs only ``history``: for origin shift s = offset, 2*offset, ... (``n_origins`` of them) the
    history is cut ``s`` days before the reference date (later days set to NaN, reference dates moved back), ``forecaster``
    (default fit_predict; ``params`` are passed to it, e.g. lam=0.5) forecasts the next 30 days and the observed days of
    ``history`` are the targets. Returns {"score": mean of the two per-variable CRPS pooled over origins, "per_variable":
    {"oxygen", "temperature"}, "per_origin": [score per shift], "shifts": [...], "n_obs": {variable: count}}. Origin shift 30 is the
    window that ``score_dev`` uses. Days that are NaN in ``history`` (not observed, or withheld) are not scored.

score(pred, obs) -> (mean_crps, {"oxygen": ..., "temperature": ...})
    pred = deliverable dict (n, 30) arrays; obs (n, 30, >=2) observed oxygen / temperature (NaN skipped); mean CRPS of each variable
    over observed (item, day) pairs, and their equal-weight mean.
crps_normal(mu, sigma, y) -> array
    closed-form CRPS of N(mu, sigma^2) at y (elementwise; Gneiting & Raftery 2007).
seasonal_profile(y, dates, target_dates=None, harmonics=2, half_window=10, shrink=0.3) -> (profile at dates, profile at target_dates)
    y (L,) daily values (NaN allowed), dates (L,) datetime64[D]. Ridge (1e-3) least-squares constant + ``harmonics`` annual
    sin / cos pairs of the day-of-year fraction (period 365.25) on the observed days, plus a correction table by day of the year:
    the mean residual of the observed days within +-``half_window`` days (circular), multiplied by ``shrink`` * n / (n + 3) (0 when
    n < 5). Returns the fitted values at ``dates`` and the values at ``target_dates`` (None -> second output None).
history_dates(reference_date, length) -> (n, length) datetime64[D]     the day of every history column
Constants: H = 30 (lead days), KEYS = the four deliverable key names.

Call pattern (one code node; every input arrives as a tool output)::

    from scilib import aquatics as aq
    y = aq.fit_predict(**load_eval_inputs)        # dict of four (n, 30) arrays -> the deliverable
    dev = aq.fit_predict(**load_dev)              # same call on the dev inputs -> score_dev.pred
    # equivalently aq.fit_predict(history, reference_date) with the two arrays / lists
    check = aq.backtest(load_eval_inputs["history"], load_eval_inputs["reference_date"])   # local rolling-origin scores
"""
from __future__ import annotations

import numpy as np

__all__ = ["H", "KEYS", "fit_predict", "data_report", "doy_window_forecast", "backtest", "score", "crps_normal",
           "seasonal_profile", "history_dates"]

H = 30
VARIABLES = ("oxygen", "temperature")
KEYS = tuple(f"{v}_{s}" for v in VARIABLES for s in ("mu", "sigma"))
MU_RANGE = {"oxygen": (0.0, 30.0), "temperature": (-5.0, 45.0)}
SIGMA_MAX = 50.0
SIGMA_FLOOR = 0.05
FAST_DAYS, SLOW_DAYS, SLOW_MIN = 3, 30, 5
RIDGE = 1.0
MIN_PAIRS = 200            # pairs at one lead needed for the item's own fit to enter
MIN_OBS = 60               # observed days of a variable below which doy_window_forecast is used


# ------------------------------------------------------------------------------------------------ dates / inputs
def history_dates(reference_date, length: int) -> np.ndarray:
    """(n, length) datetime64[D]: column t of row i is reference_date[i] - (length - 1 - t) days."""
    t0 = _as_days(reference_date)
    return t0[:, None] - np.arange(length - 1, -1, -1).astype("timedelta64[D]")


def _as_days(reference_date) -> np.ndarray:
    if isinstance(reference_date, (str, np.datetime64)) or np.ndim(reference_date) == 0:
        reference_date = [reference_date]
    arr = np.asarray(list(reference_date) if not isinstance(reference_date, np.ndarray) else reference_date)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[D]").ravel()
    return np.array([np.datetime64(str(s)[:10], "D") for s in arr.ravel()], dtype="datetime64[D]")


def _channels(variables, n_channels: int) -> tuple[int, int]:
    if variables is None:
        idx = (0, 1)
    else:
        names = [str(v) for v in variables]
        missing = [v for v in VARIABLES if v not in names]
        if missing:
            raise ValueError(f"variables {names} lack {missing}")
        idx = tuple(names.index(v) for v in VARIABLES)
    if max(idx) >= n_channels:
        raise ValueError(f"history has {n_channels} channels, oxygen / temperature need indices {idx}")
    return idx


def _prepare(history, reference_date, variables):
    hist = np.asarray(history, dtype=float)
    if hist.ndim != 3 or hist.shape[1] < 2:
        raise ValueError(f"history must have shape (n_items, n_days, n_channels), got {tuple(hist.shape)}")
    days = _as_days(reference_date)
    if days.shape[0] != hist.shape[0]:
        raise ValueError(f"reference_date has {days.shape[0]} entries for {hist.shape[0]} items")
    chan = _channels(variables, hist.shape[2])
    hist = np.where(np.isfinite(hist), hist, np.nan)                # +-inf -> missing
    dates = history_dates(days, hist.shape[1])
    tdays = days[:, None] + np.arange(1, H + 1).astype("timedelta64[D]")
    return hist, dates, tdays, chan


def _check_ports(horizon_days, ports) -> None:
    if int(horizon_days) != H:
        raise ValueError(f"horizon_days must be {H}, got {horizon_days}")
    unknown = sorted(set(ports) - {"units", "site_id", "site_type"})
    if unknown:
        raise TypeError(f"unexpected keyword arguments {unknown}")


# ------------------------------------------------------------------------------------------------ seasonal profile
def _frac(d: np.ndarray) -> np.ndarray:
    return (d - d.astype("datetime64[Y]")).astype(int) / 365.25


def _design(d: np.ndarray, harmonics: int) -> np.ndarray:
    f = _frac(d)
    cols = [np.ones(len(d))]
    for k in range(1, harmonics + 1):
        cols += [np.sin(2 * np.pi * k * f), np.cos(2 * np.pi * k * f)]
    return np.stack(cols, 1)


def _doy(d: np.ndarray) -> np.ndarray:
    return np.minimum((d - d.astype("datetime64[Y]")).astype(int), 365)


def seasonal_profile(y, dates, target_dates=None, harmonics: int = 2, half_window: int = 10, shrink: float = 0.3):
    """Smooth day-of-year profile of one daily series: (values at ``dates``, values at ``target_dates`` or None)."""
    y = np.asarray(y, dtype=float)
    dates = np.asarray(dates).astype("datetime64[D]")
    if y.shape != dates.shape or y.ndim != 1:
        raise ValueError(f"y and dates must be 1-D of equal length, got {y.shape} and {dates.shape}")
    obs = np.isfinite(y)
    if not obs.any():
        raise ValueError("seasonal_profile: the series has no observation")
    X = _design(dates, harmonics)
    w = np.linalg.solve(X[obs].T @ X[obs] + 1e-3 * np.eye(X.shape[1]), X[obs].T @ y[obs])
    res = (y - X @ w)[obs]
    doy_obs = (dates - dates.astype("datetime64[Y]")).astype(int)[obs]
    table = np.zeros(366)
    for d in range(366):
        dist = np.abs(doy_obs - d)
        dist = np.minimum(dist, 365 - dist)
        s = res[dist <= half_window]
        if len(s) >= 5:
            table[d] = shrink * s.mean() * len(s) / (len(s) + 3)
    fit_h = X @ w + table[_doy(dates)]
    if target_dates is None:
        return fit_h, None
    td = np.asarray(target_dates).astype("datetime64[D]")
    return fit_h, _design(td, harmonics) @ w + table[_doy(td)]


# ------------------------------------------------------------------------------------------------ anomaly regression
def _trailing_mean(a: np.ndarray, width: int, min_n: int) -> np.ndarray:
    fin = np.isfinite(a)
    csum = np.r_[0.0, np.cumsum(np.where(fin, a, 0.0))]
    cnt_c = np.r_[0, np.cumsum(fin)]
    i = np.arange(len(a))
    lo = np.maximum(i - width + 1, 0)
    cnt = cnt_c[i + 1] - cnt_c[lo]
    out = np.full(len(a), np.nan)
    ok = cnt >= min_n
    out[ok] = (csum[i + 1] - csum[lo])[ok] / cnt[ok]
    return out


def _features(a: np.ndarray) -> np.ndarray:
    return np.stack([_trailing_mean(a, FAST_DAYS, 1), _trailing_mean(a, SLOW_DAYS, SLOW_MIN)], 1)


def _lead_stats(a: np.ndarray) -> dict:
    """Sufficient statistics of the (features at day t, anomaly at t + h) pairs of one series, for h = 1..H."""
    F = _features(a)
    fin_f = np.isfinite(F).all(1)
    st = {"n": np.zeros(H), "sF": np.zeros((H, 2)), "FtF": np.zeros((H, 2, 2)), "sY": np.zeros(H),
          "FtY": np.zeros((H, 2)), "YtY": np.zeros(H)}
    L = len(a)
    for h in range(1, H + 1):
        if h >= L:
            break
        y = a[h:]
        f = F[:L - h]
        ok = np.isfinite(y) & fin_f[:L - h]
        f, y = f[ok], y[ok]
        st["n"][h - 1] = len(y)
        st["sF"][h - 1] = f.sum(0)
        st["FtF"][h - 1] = f.T @ f
        st["sY"][h - 1] = y.sum()
        st["FtY"][h - 1] = f.T @ y
        st["YtY"][h - 1] = y @ y
    return st


def _add_stats(stats: list[dict]) -> dict:
    return {k: sum(s[k] for s in stats) for k in stats[0]}


def _solve(st: dict) -> tuple[np.ndarray, np.ndarray]:
    """Ridge coefficients (H, 2) and residual standard deviation (H,) per lead; NaN sd where there is no pair."""
    A = st["FtF"] + RIDGE * np.eye(2)
    w = np.linalg.solve(A, st["FtY"][:, :, None])[:, :, 0]
    n = np.maximum(st["n"], 1)
    r_sum = st["sY"] - (w * st["sF"]).sum(1)
    r_sq = st["YtY"] - 2 * (w * st["FtY"]).sum(1) + np.einsum("hi,hij,hj->h", w, st["FtF"], w)
    var = np.maximum(r_sq / n - (r_sum / n) ** 2, 0.0)
    sd = np.where(st["n"] > 0, np.sqrt(var), np.nan)
    return w, sd


def _forecast_variable(Y: np.ndarray, dates: np.ndarray, tdays: np.ndarray, lam: float, smult: float):
    """(mu, sigma), each (n, H), for one variable of n items; Y (n, L)."""
    n = Y.shape[0]
    fut, stats, last = [], [], []
    for i in range(n):
        prof_h, prof_f = seasonal_profile(Y[i], dates[i], tdays[i])
        a = Y[i] - prof_h
        fut.append(prof_f)
        stats.append(_lead_stats(a))
        f = _features(a)[-1]
        last.append(np.where(np.isfinite(f), f, 0.0))
    cg, sg = _solve(_add_stats(stats))
    if not np.isfinite(sg).all():                      # no pair at some lead in any item: use the spread of all anomalies
        allsd = np.sqrt(max(sum(s["YtY"][0] for s in stats), 0.0) / max(sum(s["n"][0] for s in stats), 1.0)) or 1.0
        sg = np.where(np.isfinite(sg), sg, allsd)
    mu, sig = np.empty((n, H)), np.empty((n, H))
    for i in range(n):
        cs, ss = _solve(stats[i])
        ok = (stats[i]["n"] >= MIN_PAIRS) & np.isfinite(ss)
        c = np.where(ok[:, None], lam * cs + (1 - lam) * cg, cg)
        s = np.where(ok, np.sqrt(lam * np.nan_to_num(ss) ** 2 + (1 - lam) * sg ** 2), sg)
        mu[i] = fut[i] + c @ last[i]
        sig[i] = np.maximum(s * smult, SIGMA_FLOOR)
    return mu, sig


# ------------------------------------------------------------------------------------------------ day-of-year windows
def _window_variable(Y: np.ndarray, dates: np.ndarray, tdays: np.ndarray, half_window: int, min_n: int, min_sd: float):
    """(mu, sd), each (n, H): mean / sample sd of the observations within +-half_window days of each target day of the year.
    Rows without any observation are NaN."""
    n = Y.shape[0]
    mu, sd = np.full((n, H), np.nan), np.full((n, H), np.nan)
    for i in range(n):
        y = Y[i]
        fin = np.isfinite(y)
        if not fin.any():
            continue
        hv, hd = y[fin], dates[i][fin]
        doy_h = (hd - hd.astype("datetime64[Y]")).astype(int)
        last = y[-365:]
        last = last[np.isfinite(last)]
        fb_mu = float(last.mean()) if last.size >= 2 else float(hv.mean())
        fb_sd = max(float(last.std(ddof=1)) if last.size >= 2 else min_sd, min_sd)
        doy_t = (tdays[i] - tdays[i].astype("datetime64[Y]")).astype(int)
        dist = np.abs(doy_h[None, :] - doy_t[:, None])
        dist = np.minimum(dist, 365 - dist)
        sel = dist <= half_window
        for k in range(H):
            if sel[k].sum() >= min_n:
                s = hv[sel[k]]
                mu[i, k], sd[i, k] = s.mean(), max(float(s.std(ddof=1)), min_sd)
            else:
                mu[i, k], sd[i, k] = fb_mu, fb_sd
    return mu, sd


def doy_window_forecast(history, reference_date, half_window: int = 7, min_n: int = 5, min_sd: float = 0.1,
                        variables=None, horizon_days: int = H, **ports) -> dict:
    """Deliverable dict from day-of-year windows of each item's history (see module docstring)."""
    _check_ports(horizon_days, ports)
    hist, dates, tdays, chan = _prepare(history, reference_date, variables)
    out = {}
    for v, c in zip(VARIABLES, chan):
        mu, sd = _window_variable(hist[:, :, c], dates, tdays, half_window, min_n, min_sd)
        bad = np.flatnonzero(np.isnan(mu[:, 0]))
        if bad.size:
            raise ValueError(f"no observed {v} in the history of item(s) {bad.tolist()}")
        lo, hi = MU_RANGE[v]
        out[f"{v}_mu"], out[f"{v}_sigma"] = np.clip(mu, lo, hi), np.clip(sd, 1e-3, SIGMA_MAX)
    return out


# ------------------------------------------------------------------------------------------------ anomaly forecast
def fit_predict(history, reference_date, lam: float = 0.75, smult: float = 1.0, variables=None, horizon_days: int = H,
                pretrained: float = 0.0, **ports) -> dict:
    """Deliverable dict of four (n, 30) arrays for the items of ``history`` / ``reference_date`` (see module docstring)."""
    _check_ports(horizon_days, ports)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    if not smult > 0:
        raise ValueError(f"smult must be > 0, got {smult}")
    if not 0.0 <= float(pretrained) <= 1.0:
        raise ValueError(f"pretrained must be a weight in [0, 1], got {pretrained}")
    hist, dates, tdays, chan = _prepare(history, reference_date, variables)
    n = hist.shape[0]
    out = {}
    for v, c in zip(VARIABLES, chan):
        y = hist[:, :, c]
        rich = np.isfinite(y).sum(1) >= MIN_OBS
        mu, sg = np.empty((n, H)), np.empty((n, H))
        if rich.any():
            mu[rich], sg[rich] = _forecast_variable(y[rich], dates[rich], tdays[rich], lam, smult)
        if (~rich).any():
            m, s = _window_variable(y[~rich], dates[~rich], tdays[~rich], 7, 5, 0.1)
            bad = np.flatnonzero(np.isnan(m[:, 0]))
            if bad.size:
                raise ValueError(f"no observed {v} in the history of item(s) {np.flatnonzero(~rich)[bad].tolist()}")
            mu[~rich], sg[~rich] = m, s
        lo, hi = MU_RANGE[v]
        out[f"{v}_mu"], out[f"{v}_sigma"] = np.clip(mu, lo, hi), np.clip(sg, 1e-3, SIGMA_MAX)
        if float(pretrained) > 0.0:
            _mix_pretrained(out, v, y, float(pretrained))
    return out


def _mix_pretrained(out: dict, v: str, y: np.ndarray, w: float) -> None:
    """Replace out[v_mu], out[v_sigma] of the items with >= 3 observed days by (1 - w) * (value) + w * Chronos-2 (see fit_predict)."""
    from . import tsfm
    ok = np.flatnonzero(np.isfinite(y).sum(1) >= 3)
    if ok.size == 0:
        return
    q = tsfm.forecast([y[i] for i in ok], H, quantiles=(0.25, 0.5, 0.75), model="chronos_2").astype(float)
    mu_c = np.clip(q[:, :, 1], 0.0, 45.0)
    sd_c = np.clip((q[:, :, 2] - q[:, :, 0]) / 1.349, 0.05, 50.0)
    out[f"{v}_mu"][ok] = (1.0 - w) * out[f"{v}_mu"][ok] + w * mu_c
    out[f"{v}_sigma"][ok] = (1.0 - w) * out[f"{v}_sigma"][ok] + w * sd_c


# ------------------------------------------------------------------------------------------------ report / scoring
def data_report(history, reference_date=None, variables=None):
    """pandas DataFrame with one row per (item, variable): observation counts and age of the last observation."""
    import pandas as pd

    hist = np.asarray(history, dtype=float)
    if hist.ndim != 3:
        raise ValueError(f"history must have shape (n_items, n_days, n_channels), got {tuple(hist.shape)}")
    chan = _channels(variables, hist.shape[2])
    rows = []
    L = hist.shape[1]
    for i in range(hist.shape[0]):
        for v, c in zip(VARIABLES, chan):
            fin = np.isfinite(hist[i, :, c])
            idx = np.flatnonzero(fin)
            rows.append({"item": i, "variable": v, "n_obs": int(fin.sum()), "n_obs_365": int(fin[-365:].sum()),
                         "n_obs_30": int(fin[-30:].sum()),
                         "age_days": float(L - 1 - idx[-1]) if idx.size else np.nan,
                         "first_obs_age_days": float(L - 1 - idx[0]) if idx.size else np.nan,
                         "fallback": bool(fin.sum() < MIN_OBS)})
    return pd.DataFrame(rows)


def crps_normal(mu, sigma, y) -> np.ndarray:
    """Closed-form CRPS of N(mu, sigma^2) at y, elementwise (sigma > 0)."""
    from scipy.special import ndtr

    mu, sigma, y = (np.asarray(a, dtype=float) for a in (mu, sigma, y))
    if np.any(sigma <= 0):
        raise ValueError("sigma must be > 0")
    z = (y - mu) / sigma
    return sigma * (z * (2 * ndtr(z) - 1) + 2 * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi))


def _sums(pred: dict, obs: np.ndarray) -> dict:
    obs = np.asarray(obs, dtype=float)
    if obs.ndim != 3 or obs.shape[1] != H or obs.shape[2] < 2:
        raise ValueError(f"obs must have shape (n, {H}, >=2), got {tuple(obs.shape)}")
    out = {}
    for j, v in enumerate(VARIABLES):
        mu, sg = np.asarray(pred[f"{v}_mu"], dtype=float), np.asarray(pred[f"{v}_sigma"], dtype=float)
        if mu.shape != obs.shape[:2] or sg.shape != mu.shape:
            raise ValueError(f"pred[{v}_mu / {v}_sigma] must have shape {tuple(obs.shape[:2])}, got {mu.shape} / {sg.shape}")
        o = obs[:, :, j]
        m = np.isfinite(o)
        out[v] = (float(crps_normal(mu[m], sg[m], o[m]).sum()), int(m.sum()))
    return out


def score(pred: dict, obs) -> tuple[float, dict]:
    """(equal-weight mean of the two per-variable mean CRPS, {variable: mean CRPS over observed item-days})."""
    per = {v: (s / n if n else float("nan")) for v, (s, n) in _sums(pred, obs).items()}
    return float(np.mean(list(per.values()))), per


def backtest(history, reference_date, n_origins: int = 3, offset: int = 30, forecaster=None, **params) -> dict:
    """Rolling-origin CRPS check inside ``history`` (origins ``offset``, 2 * offset, ... days before the reference date)."""
    hist = np.asarray(history, dtype=float)
    days = _as_days(reference_date)
    if hist.ndim != 3 or days.shape[0] != hist.shape[0]:
        raise ValueError("history must be (n_items, n_days, n_channels) with one reference_date per item")
    if n_origins < 1 or offset < 1:
        raise ValueError("n_origins and offset must be >= 1")
    forecaster = forecaster or fit_predict
    chan = _channels(params.get("variables"), hist.shape[2])
    L = hist.shape[1]
    tot = {v: [0.0, 0] for v in VARIABLES}
    per_origin, shifts = [], []
    for k in range(1, n_origins + 1):
        s = offset * k
        if L - s < 400:
            break
        cut = hist[:, :L - s, :]
        obs = hist[:, L - s:L - s + H, :][:, :, list(chan)]
        if obs.shape[1] < H:                                       # targets would run past the reference date
            obs = np.concatenate([obs, np.full((obs.shape[0], H - obs.shape[1], 2), np.nan)], axis=1)
        pred = forecaster(cut, days - np.timedelta64(s, "D"), **params)
        sums = _sums(pred, obs)
        for v in VARIABLES:
            tot[v][0] += sums[v][0]
            tot[v][1] += sums[v][1]
        per_origin.append(float(np.mean([sums[v][0] / sums[v][1] for v in VARIABLES if sums[v][1]] or [np.nan])))
        shifts.append(s)
    if not shifts:
        raise ValueError("history is too short for a backtest")
    per = {v: (a / b if b else float("nan")) for v, (a, b) in tot.items()}
    return {"score": float(np.mean(list(per.values()))), "per_variable": per, "per_origin": per_origin, "shifts": shifts,
            "n_obs": {v: b for v, (a, b) in tot.items()}}
