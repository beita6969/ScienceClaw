"""Tools for forecasting global gridded weather fields (2 m temperature on the WeatherBench 2 longitude x latitude grid).

Layout. Field arrays have (longitude, latitude) as their LAST two axes: load_train ``t2m`` (T, 64, 32) K with ``time`` (T ISO UTC
strings, 6-hourly); ``context`` of load_dev / load_eval_inputs (n, 4, 64, 32) K, axis 1 in the order of ``context_offsets_h``
(-18, -12, -6, 0 h relative to ``init_time[i]``; index 3 is the field at ``init_time``), ``init_time`` (n ISO strings); forecast
lead 24 h. Times may be ISO strings or numpy datetime64. Longitude is periodic (index 63 neighbours 0); latitude is not.
load_dev and load_eval_inputs have the same layout, so one forecast function (context, init_time) -> (n, lon, lat) applies to both.
Wiring: score_dev.pred takes the forecasts of load_dev's ``context`` / ``init_time`` (the December-2018 initialisations);
the submitted y takes the forecasts of load_eval_inputs' ``context`` / ``init_time``. A forecast computed from load_eval_inputs
and wired to score_dev is compared with the dev targets at other times. Both forecasts can come from one fitted model.

parse_times(times) -> datetime64[m] array
    ISO-8601 UTC strings / datetime64 values as a 1-D array (used by every ``times`` / ``init_time`` argument below).
lat_weights(latitude) -> (n_lat,)
    WeatherBench 2 cell-area weights ~ sin(lat + d/2) - sin(lat - d/2) (d = grid spacing), mean 1.
lat_weighted_rmse(pred, truth, weights=None, per_item=False) -> float | (n,)
    task metric: per item sqrt(mean over (lon, lat) of w(lat) * (pred - truth)^2), then the mean over items (``per_item=True``:
    the (n,) array). ``weights=None``: lat_weights of an equiangular grid with n_lat rows.
fit_seasonal_cycle(fields, times, n_harmonics=3, hour_bins=4) -> dict
    per grid point and UTC-hour bin (``hour_bins`` equal bins of the day): least-squares constant + ``n_harmonics`` annual
    harmonics of the day of year (period 365.25 d) of a (T, lon, lat) record. Plain dict of arrays.
seasonal_cycle_at(cycle, times) -> (len(times), lon, lat)
    values of the fitted cycle at arbitrary UTC times (any year).
anomalies(fields, times, cycle) -> (n, lon, lat)
    ``fields - seasonal_cycle_at(cycle, times)`` as float64.
fit_patch_ridge(fields, times, radius=2, lam=30.0, n_harmonics=3, hour_bins=4, context_offsets_h=(-18, -12, -6, 0), lead_h=24,
                cycle=None) -> dict
    model predicting the anomaly at init + ``lead_h`` at every grid cell from the anomalies of all context fields inside the
    (2*radius+1) x (2*radius+1) window centred on that cell (longitude wraps, rows beyond the poles repeat the edge row): one
    ridge regression per latitude row, pooled over longitudes and over every (context, target) group of steps in ``times``;
    penalty = lam * n_samples / 1000. ``cycle``: an existing fit_seasonal_cycle result (default: fitted on ``fields`` with
    n_harmonics / hour_bins). Plain dict (may be passed between nodes); fits the 1336-step load_train in seconds, < 1 GB.
predict_patch_ridge(model, context, init_time) -> (n, lon, lat)
    seasonal cycle at init + lead_h plus the predicted anomaly; row i depends only on context[i] and init_time[i].
fit_predict(fields, times, context, init_time, **fit_kwargs) -> (n, lon, lat)
    fit_patch_ridge(fields, times, **fit_kwargs) followed by predict_patch_ridge on (context, init_time).
blocked_cv(fields, times, fit=None, predict=None, block_days=30.0, n_blocks=None, gap_days=6.0, init_hours=(0, 12),
           context_offsets_h=(-18, -12, -6, 0), lead_h=24, weights=None, **fit_kwargs) -> dict
    temporal cross-validation inside one record: contiguous blocks of about ``block_days`` days (``n_blocks`` overrides); per block
    ``fit(train_fields, train_times) -> model`` is called on the steps more than ``gap_days`` from the block (both sides) and
    ``predict(model, context (1, 4, lon, lat), init_time (1,))`` once per held-out initialisation (UTC hour in ``init_hours``;
    context fields and target inside the block). Defaults: fit_patch_ridge(..., **fit_kwargs) / predict_patch_ridge; any callables
    with these signatures work. Returns {"rmse": mean of the per-initialisation lat_weighted_rmse, "n_items", "blocks": [{"start",
    "end", "n_items", "n_train_steps", "rmse"}], "init_time": list, "rmse_items": (n_items,), "block": (n_items,)}. Folds are
    deterministic, so runs with different fit / predict / settings compare item by item. About 15 CPU-s for the default model.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

__all__ = ["parse_times", "lat_weights", "lat_weighted_rmse", "fit_seasonal_cycle", "seasonal_cycle_at", "anomalies",
           "fit_patch_ridge", "predict_patch_ridge", "fit_predict", "blocked_cv"]

_PERIOD_D = 365.25          # annual period (days) of the harmonics
_MIN_PER_DAY = 1440
_DEFAULT_OFFSETS_H = (-18, -12, -6, 0)


# ------------------------------------------------------------------------------------------------ small helpers
def parse_times(times) -> np.ndarray:
    """ISO-8601 UTC strings / datetimes -> 1-D ``datetime64[m]`` array."""
    try:
        return np.atleast_1d(np.asarray(times, dtype="datetime64[m]"))
    except (ValueError, TypeError) as ex:
        raise ValueError(f"times must be ISO-8601 UTC strings such as '2018-01-01T06:00' or datetime64 values: {ex}") from None


def _minutes(times) -> np.ndarray:
    return parse_times(times).astype(np.int64)


def _field_stack(x, name: str, ndim: int) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != ndim or a.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a numeric array with {ndim} axes (..., longitude, latitude), got shape {a.shape}")
    return a


def _default_latitude(n_lat: int) -> np.ndarray:
    d = 180.0 / n_lat
    return -90.0 + d / 2 + d * np.arange(n_lat)


def lat_weights(latitude) -> np.ndarray:
    """WeatherBench 2 area weights of an equiangular latitude grid (degrees, ascending): sin(lat + d/2) - sin(lat - d/2), mean 1."""
    lat = np.asarray(latitude, dtype=float)
    d = float(np.abs(np.diff(lat)).mean())
    w = np.sin(np.deg2rad(np.clip(lat + d / 2, -90.0, 90.0))) - np.sin(np.deg2rad(np.clip(lat - d / 2, -90.0, 90.0)))
    return w / w.mean()


def lat_weighted_rmse(pred, truth, weights=None, per_item: bool = False):
    """sqrt(mean over (lon, lat) of w(lat) (pred - truth)^2) per item, averaged over items (WeatherBench 2 convention).

    ``pred`` / ``truth``: (n, lon, lat) (or a single (lon, lat) field); ``weights``: (lat,) mean-1 weights (default: equiangular
    grid weights for the number of latitude rows). Returns a float, or the (n,) per-item array if ``per_item``."""
    p, t = np.asarray(pred, dtype=float), np.asarray(truth, dtype=float)
    if p.shape != t.shape or p.ndim < 2:
        raise ValueError(f"pred and truth must have the same shape (..., longitude, latitude), got {p.shape} and {t.shape}")
    w = lat_weights(_default_latitude(p.shape[-1])) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (p.shape[-1],):
        raise ValueError(f"weights must have shape ({p.shape[-1]},) (one per latitude row), got {w.shape}")
    per = np.sqrt(np.mean((p - t) ** 2 * w, axis=(-2, -1)))
    return per if per_item or per.ndim == 0 else float(per.mean())


# ------------------------------------------------------------------------------------------------ seasonal cycle
def _harmonic_design(minutes: np.ndarray, n_harmonics: int) -> np.ndarray:
    t = minutes.astype("datetime64[m]")
    doy = (t - t.astype("datetime64[Y]").astype("datetime64[m]")) / np.timedelta64(_MIN_PER_DAY, "m")
    w = 2 * np.pi * doy / _PERIOD_D
    cols = [np.ones_like(w)] + [f(k * w) for k in range(1, n_harmonics + 1) for f in (np.cos, np.sin)]
    return np.stack(cols, 1)


def _hour_bin(minutes: np.ndarray, hour_bins: int) -> np.ndarray:
    return ((minutes % _MIN_PER_DAY) * hour_bins) // _MIN_PER_DAY


def fit_seasonal_cycle(fields, times, n_harmonics: int = 3, hour_bins: int = 4) -> dict:
    """Least-squares mean seasonal cycle per grid point and UTC-hour bin from a (T, lon, lat) record (``times``: T UTC times).

    Model per hour bin: c0 + sum_{k=1..n_harmonics} a_k cos(2 pi k d / 365.25) + b_k sin(2 pi k d / 365.25), d = fractional
    day of year. Every hour bin needs at least ``1 + 2 * n_harmonics`` records. Returns {"coefs": (hour_bins, 1 + 2 n_harmonics,
    lon, lat), "n_harmonics", "hour_bins"}."""
    F = _field_stack(fields, "fields", 3)
    m = _minutes(times)
    if len(m) != len(F):
        raise ValueError(f"fields has {len(F)} time steps but times has {len(m)}")
    if n_harmonics < 0 or hour_bins < 1:
        raise ValueError("n_harmonics must be >= 0 and hour_bins >= 1")
    D, hb = _harmonic_design(m, n_harmonics), _hour_bin(m, hour_bins)
    nlon, nlat = F.shape[1:]
    coefs = np.zeros((hour_bins, D.shape[1], nlon * nlat))
    for h in range(hour_bins):
        sel = hb == h
        if sel.sum() < D.shape[1]:
            raise ValueError(f"hour bin {h} has {int(sel.sum())} records, fewer than the {D.shape[1]} coefficients "
                             f"(n_harmonics={n_harmonics}); use fewer harmonics / hour bins or a longer record")
        coefs[h] = np.linalg.lstsq(D[sel], F[sel].reshape(int(sel.sum()), -1).astype(np.float64), rcond=None)[0]
    return {"kind": "seasonal_cycle", "coefs": coefs.reshape(hour_bins, D.shape[1], nlon, nlat),
            "n_harmonics": int(n_harmonics), "hour_bins": int(hour_bins)}


def seasonal_cycle_at(cycle: dict, times) -> np.ndarray:
    """(len(times), lon, lat) float64 values of a ``fit_seasonal_cycle`` result at the given UTC times."""
    m = _minutes(times)
    coefs = cycle["coefs"]
    D, hb = _harmonic_design(m, cycle["n_harmonics"]), _hour_bin(m, cycle["hour_bins"])
    out = np.empty((len(m),) + coefs.shape[2:])
    for h in np.unique(hb):
        sel = hb == h
        out[sel] = np.tensordot(D[sel], coefs[h], axes=(1, 0))
    return out


def anomalies(fields, times, cycle: dict) -> np.ndarray:
    """``fields - seasonal_cycle_at(cycle, times)`` for (T, lon, lat) fields, float64."""
    F = _field_stack(fields, "fields", 3)
    if len(F) != len(np.atleast_1d(times)):
        raise ValueError(f"fields has {len(F)} time steps but times has {len(np.atleast_1d(times))}")
    return F.astype(np.float64) - seasonal_cycle_at(cycle, times)


# ------------------------------------------------------------------------------------------------ patch ridge
def _pad(a: np.ndarray, r: int) -> np.ndarray:
    """Pad the last two axes (lon, lat) by r cells: longitude wraps, latitude repeats the edge row."""
    if r == 0:
        return a
    lead = [(0, 0)] * (a.ndim - 2)
    return np.pad(np.pad(a, lead + [(r, r), (0, 0)], mode="wrap"), lead + [(0, 0), (r, r)], mode="edge")


def _row_features(P: np.ndarray, j: int, r: int) -> np.ndarray:
    """P: padded anomalies (m, L, lon + 2r, lat + 2r) -> (m, lon, L * (2r+1)^2) window features of latitude row j
    (feature order: context index, latitude offset, longitude offset)."""
    k = 2 * r + 1
    W = sliding_window_view(P[..., j:j + k], k, axis=2)          # (m, L, lon, k lat, k lon)
    return W.transpose(0, 2, 1, 3, 4).reshape(P.shape[0], W.shape[2], -1)


def _offsets(context_offsets_h) -> tuple[int, ...]:
    off = tuple(int(o) for o in context_offsets_h)
    if len(off) == 0 or list(off) != sorted(set(off)):
        raise ValueError(f"context_offsets_h must be strictly increasing hours, got {off}")
    return off


def fit_patch_ridge(fields, times, radius: int = 2, lam: float = 30.0, n_harmonics: int = 3, hour_bins: int = 4,
                    context_offsets_h=_DEFAULT_OFFSETS_H, lead_h: int = 24, cycle: dict | None = None,
                    chunk: int = 128) -> dict:
    """Fit the per-latitude-row window ridge model on a (T, lon, lat) record with ``times`` (T UTC times, strictly increasing,
    not necessarily regular). Training groups = every step t0 of the record whose fields at t0 + context_offsets_h and at
    t0 + lead_h are all present. Returns a plain dict (arrays and numbers) for ``predict_patch_ridge``."""
    F = _field_stack(fields, "fields", 3)
    m = _minutes(times)
    if len(m) != len(F):
        raise ValueError(f"fields has {len(F)} time steps but times has {len(m)}")
    if len(m) > 1 and not np.all(np.diff(m) > 0):
        raise ValueError("times must be strictly increasing")
    r, off = int(radius), _offsets(context_offsets_h)
    if r < 0 or r > F.shape[1] // 2:
        raise ValueError(f"radius must be in [0, n_lon // 2] = [0, {F.shape[1] // 2}], got {radius}")
    nlon, nlat = F.shape[1:]
    if cycle is None:
        cycle = fit_seasonal_cycle(F, m, n_harmonics, hour_bins)
    A = anomalies(F, m, cycle)                                                   # (T, lon, lat)
    want = m[:, None] + np.array(off + (int(lead_h),), dtype=np.int64)[None] * 60
    pos = np.minimum(np.searchsorted(m, want), len(m) - 1)
    ok = np.all(m[pos] == want, axis=1)
    if ok.sum() < 20:
        raise ValueError(f"only {int(ok.sum())} usable (context, target) groups: the record needs steps at t0 + "
                         f"{list(off)} h and t0 + {lead_h} h for many t0 (times must contain the required regular steps)")
    pos = pos[ok]
    L, d = len(off), len(off) * (2 * r + 1) ** 2
    P = _pad(A, r)                                                               # (T, lon + 2r, lat + 2r)
    G, B = np.zeros((nlat, d, d)), np.zeros((nlat, d))
    for s in range(0, len(pos), max(int(chunk), 1)):
        Pc, Yc = P[pos[s:s + chunk, :L]], A[pos[s:s + chunk, L]]                 # (m, L, lon + 2r, lat + 2r), (m, lon, lat)
        for j in range(nlat):
            X = _row_features(Pc, j, r).reshape(-1, d)
            G[j] += X.T @ X
            B[j] += X.T @ Yc[:, :, j].reshape(-1)
    n_samples = len(pos) * nlon
    ridge = float(lam) * n_samples / 1000.0
    eye = np.eye(d)
    betas = np.stack([np.linalg.solve(G[j] + ridge * eye, B[j]) for j in range(nlat)])
    return {"kind": "patch_ridge", "cycle": cycle, "betas": betas, "radius": r, "lam": float(lam),
            "context_offsets_h": list(off), "lead_h": int(lead_h), "n_groups": int(len(pos)), "grid": [int(nlon), int(nlat)]}


def predict_patch_ridge(model: dict, context, init_time, context_offsets_h=None) -> np.ndarray:
    """Forecast at ``init_time + model["lead_h"]`` for context (n, L, lon, lat) fields at ``init_time + context_offsets_h``
    (L = len(model["context_offsets_h"])); returns (n, lon, lat) float64 in the units of the fitted record."""
    if model.get("kind") != "patch_ridge":
        raise ValueError("model must be a dict returned by fit_patch_ridge")
    off = model["context_offsets_h"]
    if context_offsets_h is not None and [int(o) for o in context_offsets_h] != list(off):
        raise ValueError(f"context_offsets_h {list(context_offsets_h)} differs from the model's {off}")
    C = _field_stack(context, "context", 4).astype(np.float64)
    n, L, nlon, nlat = C.shape
    if L != len(off) or [nlon, nlat] != model["grid"]:
        raise ValueError(f"context must have shape (n, {len(off)}, {model['grid'][0]}, {model['grid'][1]}), got {C.shape}")
    t0 = parse_times(init_time).astype(np.int64)
    if len(t0) != n:
        raise ValueError(f"context has {n} items but init_time has {len(t0)}")
    cyc = model["cycle"]
    ctx_t = (t0[:, None] + np.array(off, dtype=np.int64)[None] * 60).reshape(-1)
    A = C - seasonal_cycle_at(cyc, ctx_t).reshape(n, L, nlon, nlat)
    r = model["radius"]
    P = _pad(A, r)
    out = seasonal_cycle_at(cyc, t0 + model["lead_h"] * 60)
    for j in range(nlat):
        out[:, :, j] += _row_features(P, j, r) @ model["betas"][j]
    return out


def fit_predict(fields, times, context, init_time, **fit_kwargs) -> np.ndarray:
    """``predict_patch_ridge(fit_patch_ridge(fields, times, **fit_kwargs), context, init_time)``."""
    return predict_patch_ridge(fit_patch_ridge(fields, times, **fit_kwargs), context, init_time)


# ------------------------------------------------------------------------------------------------ blocked CV
def blocked_cv(fields, times, fit=None, predict=None, block_days: float = 30.0, n_blocks: int | None = None,
               gap_days: float = 6.0, init_hours=(0, 12),
               context_offsets_h=_DEFAULT_OFFSETS_H, lead_h: int = 24, weights=None, **fit_kwargs) -> dict:
    """Blocked temporal cross-validation inside a (T, lon, lat) record (see the module docstring for the returned keys).

    Training steps of a fold are those farther than ``gap_days`` from the first/last step of the held-out block, so no
    training group (context or target) touches the neighbourhood of a held-out initialisation. ``fit_kwargs`` go to
    ``fit_patch_ridge`` (with the given ``context_offsets_h`` / ``lead_h``) when ``fit`` is None."""
    F = _field_stack(fields, "fields", 3)
    tt = parse_times(times)
    m = tt.astype(np.int64)
    if len(m) != len(F) or (len(m) > 1 and not np.all(np.diff(m) > 0)):
        raise ValueError("times must be strictly increasing with one entry per field")
    off = _offsets(context_offsets_h)
    if 0 not in off:
        raise ValueError("context_offsets_h must contain 0 (the field at init_time)")
    if n_blocks is None:
        n_blocks = max(2, int(round((m[-1] - m[0]) / _MIN_PER_DAY / float(block_days))))
    if n_blocks < 2:
        raise ValueError("n_blocks must be >= 2")
    if fit is None:
        def fit(f, t):                                                   # noqa: E306
            return fit_patch_ridge(f, t, context_offsets_h=off, lead_h=lead_h, **fit_kwargs)
        if predict is None:
            predict = predict_patch_ridge
    elif fit_kwargs:
        raise ValueError(f"unexpected arguments for a custom fit: {sorted(fit_kwargs)}")
    if predict is None:
        raise ValueError("a custom fit needs a custom predict(model, context, init_time)")
    w = None if weights is None else np.asarray(weights, dtype=float)
    edges = np.round(np.linspace(0, len(m), n_blocks + 1)).astype(int)
    gap = int(round(gap_days * _MIN_PER_DAY))
    rel = np.array(off + (int(lead_h),), dtype=np.int64) * 60
    hour_ok = np.isin((m % _MIN_PER_DAY) // 60, list(init_hours)) & (m % 60 == 0)
    rmse_items, init_all, block_all, blocks = [], [], [], []
    for b in range(n_blocks):
        lo, hi = edges[b], edges[b + 1]
        t_lo, t_hi = m[lo], m[hi - 1]
        train = (m < t_lo - gap) | (m > t_hi + gap)
        want = m[:, None] + rel[None]
        pos = np.minimum(np.searchsorted(m, want), len(m) - 1)
        inside = np.all((want >= t_lo) & (want <= t_hi) & (m[pos] == want), axis=1) & hour_ok
        idx = np.flatnonzero(inside)
        if len(idx) == 0 or train.sum() == 0:
            raise ValueError(f"block {b} has no held-out initialisations or no training steps (n_blocks={n_blocks}, gap_days={gap_days}); use fewer, longer blocks or a shorter gap")
        model = fit(F[train], tt[train])
        per = []
        for i in idx:
            ctx = F[pos[i, :len(off)]][None].astype(np.float64)                       # (1, L, lon, lat)
            pred = np.asarray(predict(model, ctx, tt[i:i + 1]), dtype=np.float64)
            if pred.shape != (1,) + F.shape[1:]:
                raise ValueError(f"predict must return shape (1, {F.shape[1]}, {F.shape[2]}) for one initialisation, got {pred.shape}")
            per.append(float(lat_weighted_rmse(pred, F[pos[i, len(off)]][None], w, per_item=True)[0]))
        rmse_items += per
        init_all += [str(tt[i]) for i in idx]
        block_all += [b] * len(idx)
        blocks.append({"start": str(tt[lo]), "end": str(tt[hi - 1]), "n_items": int(len(idx)),
                       "n_train_steps": int(train.sum()), "rmse": float(np.mean(per))})
    arr = np.array(rmse_items)
    return {"rmse": float(arr.mean()), "n_items": int(len(arr)), "blocks": blocks, "init_time": init_all,
            "rmse_items": arr, "block": np.array(block_all, dtype=int)}
