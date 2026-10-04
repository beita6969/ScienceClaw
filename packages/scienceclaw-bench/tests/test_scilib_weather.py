"""scilib.weather: metric parity with the adapter, seasonal cycle, patch ridge vs a brute-force loop, leak-free blocked CV."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scilib
from scilib import weather as W
from scienceclaw.bench.tasks._forecast_common import wb2_lat_weights
from scienceclaw.bench.tasks.for37_weatherbench import weighted_rmse
from scienceclaw.runtime.integrity import scan_code

NLON, NLAT = 16, 8
OFFSETS = (-18, -12, -6, 0)


def _times(n_days: int, start: str = "2018-01-01T00") -> np.ndarray:
    t0 = np.datetime64(start, "m")
    return t0 + np.arange(4 * n_days) * np.timedelta64(360, "m")


def _seasonal_truth(times: np.ndarray) -> np.ndarray:
    """(T, lon, lat) seasonal + diurnal signal built independently of the library (pandas day-of-year)."""
    ts = pd.DatetimeIndex(times.astype("datetime64[ns]"))
    doy = ts.dayofyear.values - 1 + ts.hour.values / 24.0 + ts.minute.values / 1440.0
    w = 2 * np.pi * doy / 365.25
    lat = np.linspace(-1, 1, NLAT)[None, None, :]
    lon = np.cos(2 * np.pi * np.arange(NLON) / NLON)[None, :, None]
    diurnal = (ts.hour.values // 6)[:, None, None] * 1.5
    return (280.0 + 20.0 * lat + 2.0 * lon + (12.0 * np.cos(w) - 5.0 * np.sin(w))[:, None, None] * (0.5 + 0.5 * lat)
            + (2.0 * np.cos(2 * w))[:, None, None] + diurnal)


def _synthetic(n_days: int = 365, seed: int = 0, start: str = "2018-01-01T00"):
    """Seasonal cycle + an eastward-drifting damped anomaly (one cell per 24 h): (fields float32, times)."""
    rng = np.random.default_rng(seed)
    times = _times(n_days, start)
    T = len(times)
    kern = np.exp(-0.5 * (np.arange(-3, 4) / 1.2) ** 2)
    kern /= kern.sum()
    a = np.zeros((T, NLON, NLAT))
    a[0] = rng.normal(size=(NLON, NLAT))
    for t in range(1, T):
        shifted = np.roll(a[t - 1], 1, axis=0) if t % 4 == 0 else a[t - 1]
        noise = rng.normal(size=(NLON, NLAT))
        noise = sum(k * np.roll(noise, s, axis=0) for k, s in zip(kern, range(-3, 4)))
        a[t] = 0.85 * shifted + 0.5 * noise
    return (_seasonal_truth(times) + 2.0 * a).astype(np.float32), times


@pytest.fixture(scope="module")
def data():
    return _synthetic(730)


# ---------------------------------------------------------------------------------------------- interface text
def test_describe_lists_every_public_name_and_layout():
    text = scilib.describe("weather")
    for name in W.__all__:
        assert name in text
    assert "(longitude, latitude)" in text and "-18, -12, -6, 0" in text and "context" in text


def test_policy_visible_text_has_no_reference_or_acceptance_wording():
    text = scilib.describe("weather").lower()
    for bad in ("persistence", "climatolog", "naive", "no-change", "random walk", "reference", "accept", "margin",
                "you should", "best recipe", "0.96", "beat"):
        assert bad not in text, bad


def test_code_node_may_import_scilib():
    code = "from scilib.weather import fit_predict, blocked_cv\n\ndef run(inputs, config):\n    return {}\n"
    assert scan_code(code) == []


# ---------------------------------------------------------------------------------------------- metric
def test_lat_weights_and_rmse_match_the_adapter():
    lat = -90 + 5.625 / 2 + 5.625 * np.arange(32)
    assert np.allclose(W.lat_weights(lat), wb2_lat_weights(lat))
    w = W.lat_weights(lat)
    assert w.mean() == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    p, t = rng.normal(size=(5, 64, 32)) + 280, rng.normal(size=(5, 64, 32)) + 280
    assert np.allclose(W.lat_weighted_rmse(p, t, per_item=True), weighted_rmse(p, t, w))
    assert W.lat_weighted_rmse(p, t, w) == pytest.approx(float(weighted_rmse(p, t, w).mean()))
    assert W.lat_weighted_rmse(p, t) == pytest.approx(W.lat_weighted_rmse(p, t, w))     # default weights = equiangular grid
    assert W.lat_weighted_rmse(p[0], t[0]) == pytest.approx(float(weighted_rmse(p[:1], t[:1], w)[0]))
    with pytest.raises(ValueError):
        W.lat_weighted_rmse(p, t[:4])
    with pytest.raises(ValueError):
        W.lat_weighted_rmse(p, t, w[:5])


def test_parse_times_accepts_iso_strings_and_datetime64():
    a = W.parse_times(["2018-01-01T06:00", "2018-01-01T12:00:00"])
    b = W.parse_times(np.array(["2018-01-01T06", "2018-01-01T12"], dtype="datetime64[h]"))
    assert a.dtype == np.dtype("datetime64[m]") and np.array_equal(a, b)
    with pytest.raises(ValueError):
        W.parse_times(["not a time"])


# ---------------------------------------------------------------------------------------------- seasonal cycle
def test_seasonal_cycle_recovers_a_planted_cycle_in_another_year():
    tr = _times(730)
    cyc = W.fit_seasonal_cycle(_seasonal_truth(tr), tr, n_harmonics=2, hour_bins=4)
    assert cyc["coefs"].shape == (4, 5, NLON, NLAT)
    other = _times(200, "2020-03-05T00")                     # a leap year, never seen in the fit
    assert np.allclose(W.seasonal_cycle_at(cyc, other), _seasonal_truth(other), atol=1e-6)
    an = W.anomalies(_seasonal_truth(other), other, cyc)
    assert an.dtype == np.float64 and np.abs(an).max() < 1e-6
    iso = [str(t)[:16] for t in other[:8]]                   # ISO strings are accepted as well
    assert np.allclose(W.seasonal_cycle_at(cyc, iso), W.seasonal_cycle_at(cyc, other[:8]))


def test_seasonal_cycle_needs_enough_records():
    tr = _times(1)
    with pytest.raises(ValueError):
        W.fit_seasonal_cycle(np.zeros((len(tr), 2, 2)), tr)
    with pytest.raises(ValueError):
        W.fit_seasonal_cycle(np.zeros((5, 2, 2)), tr)         # times/fields length mismatch


# ---------------------------------------------------------------------------------------------- patch ridge
def _brute_force_predict(F, times, ctx, init, radius, lam, cycle):
    """Independent loop implementation of the documented model (one ridge per latitude row)."""
    A = W.anomalies(F, times, cycle)
    m = W.parse_times(times).astype(np.int64)
    index = {int(v): i for i, v in enumerate(m)}
    nlon, nlat = A.shape[1:]
    rows = [[] for _ in range(nlat)]
    tgts = [[] for _ in range(nlat)]

    def feats(fields, j):          # fields (L, lon, lat) anomalies -> (lon, L * (2r+1)^2)
        out = np.zeros((nlon, len(fields) * (2 * radius + 1) ** 2))
        for i in range(nlon):
            col = 0
            for k in range(len(fields)):
                for dj in range(-radius, radius + 1):
                    for di in range(-radius, radius + 1):
                        out[i, col] = fields[k][(i + di) % nlon, min(max(j + dj, 0), nlat - 1)]
                        col += 1
        return out

    for s, t in enumerate(m):
        need = [t + o * 60 for o in OFFSETS] + [t + 24 * 60]
        if not all(int(x) in index for x in need):
            continue
        ids = [index[int(x)] for x in need]
        for j in range(nlat):
            rows[j].append(feats(A[ids[:4]], j))
            tgts[j].append(A[ids[4], :, j])
    n_groups = len(rows[0])
    ridge = lam * n_groups * nlon / 1000.0
    out = np.zeros((len(ctx), nlon, nlat))
    t0 = W.parse_times(init).astype(np.int64)
    for n in range(len(ctx)):
        an = ctx[n] - W.seasonal_cycle_at(cycle, t0[n] + np.array(OFFSETS) * 60)
        base = W.seasonal_cycle_at(cycle, t0[n:n + 1] + 24 * 60)[0]
        for j in range(nlat):
            X, y = np.concatenate(rows[j]), np.concatenate(tgts[j])
            beta = np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ y)
            out[n, :, j] = base[:, j] + feats(an, j) @ beta
    return out


def test_patch_ridge_matches_a_brute_force_loop_implementation():
    F, times = _synthetic(60, seed=3)
    F, ctx_f = F[:200], F[200:208]
    tr_t = times[:200]
    cyc = W.fit_seasonal_cycle(F, tr_t, n_harmonics=1, hour_bins=2)
    model = W.fit_patch_ridge(F, tr_t, radius=1, lam=5.0, cycle=cyc)
    ctx = np.stack([ctx_f[i:i + 4] for i in range(3)]).astype(np.float64)
    init = [str(times[203 + i])[:16] for i in range(3)]
    got = W.predict_patch_ridge(model, ctx, init)
    want = _brute_force_predict(F, tr_t, ctx, init, 1, 5.0, cyc)
    assert got.shape == (3, NLON, NLAT) and np.allclose(got, want, atol=1e-7)
    assert model["n_groups"] == 193                                                  # steps 3..195 have all fields


def test_patch_ridge_is_per_item_deterministic_and_plain_data(data):
    F, times = data
    model = W.fit_patch_ridge(F[:1500], times[:1500], radius=1)
    assert isinstance(model, dict) and all(isinstance(v, (np.ndarray, dict, list, int, float, str)) for v in model.values())
    idx = np.array([1520, 1900, 2500])
    ctx = np.stack([F[i - 3:i + 1] for i in idx]).astype(np.float64)
    init = [str(times[i])[:16] for i in idx]
    batch = W.predict_patch_ridge(model, ctx, init)
    for k in range(3):                                         # row i depends only on context[i], init_time[i]
        assert np.allclose(batch[k], W.predict_patch_ridge(model, ctx[k:k + 1], init[k:k + 1])[0], rtol=0, atol=1e-8)
    again = W.fit_predict(F[:1500], times[:1500], ctx, init, radius=1)
    assert np.allclose(again, batch, rtol=0, atol=1e-8)
    assert np.isfinite(batch).all() and batch.dtype == np.float64


def test_patch_ridge_beats_last_field_on_a_drifting_signal(data):
    F, times = data
    model = W.fit_patch_ridge(F[:2300], times[:2300], radius=1)
    idx = np.arange(2304, 2900, 8)
    ctx = np.stack([F[i - 3:i + 1] for i in idx]).astype(np.float64)
    pred = W.predict_patch_ridge(model, ctx, [str(times[i])[:16] for i in idx])
    truth = F[idx + 4].astype(np.float64)
    assert W.lat_weighted_rmse(pred, truth) < 0.9 * W.lat_weighted_rmse(ctx[:, -1], truth)


def test_patch_ridge_input_checks(data):
    F, times = data
    model = W.fit_patch_ridge(F[:400], times[:400], radius=1)
    ctx = F[:4][None].astype(np.float64)
    with pytest.raises(ValueError):
        W.predict_patch_ridge(model, ctx[:, :3], [str(times[3])[:16]])                # wrong number of context fields
    with pytest.raises(ValueError):
        W.predict_patch_ridge(model, ctx, [str(times[3])[:16]] * 2)                   # init_time length
    with pytest.raises(ValueError):
        W.predict_patch_ridge(model, ctx, [str(times[3])[:16]], context_offsets_h=(-12, -6, 0, 6))
    with pytest.raises(ValueError):
        W.predict_patch_ridge({"kind": "other"}, ctx, [str(times[3])[:16]])
    with pytest.raises(ValueError):
        W.fit_patch_ridge(F[:400], times[:400][::-1])                                # not increasing
    with pytest.raises(ValueError):
        W.fit_patch_ridge(F[:400], times[:400], radius=NLON)                         # window wider than the grid
    with pytest.raises(ValueError):
        W.fit_patch_ridge(F[:400:2], times[:400:2], radius=1)                        # 12-hourly record: no 6-hourly groups


def test_patch_ridge_handles_gaps_in_the_training_times(data):
    F, times = data
    keep = np.ones(len(times), bool)
    keep[300:340] = False                                                             # a hole of 10 days
    model = W.fit_patch_ridge(F[keep][:900], times[keep][:900], radius=1)
    full = W.fit_patch_ridge(F[:900], times[:900], radius=1)
    assert 0 < model["n_groups"] < full["n_groups"]


# ---------------------------------------------------------------------------------------------- blocked CV
def test_blocked_cv_folds_are_leak_free_and_one_item_at_a_time(data):
    F, times = data
    train_seen, calls = [], []

    def fit(f, t):
        train_seen.append(np.asarray(t, dtype="datetime64[m]").copy())
        assert len(f) == len(t)
        return {"n": len(t)}

    def predict(model, context, init_time):
        calls.append((context.shape, len(init_time), str(np.asarray(init_time, dtype="datetime64[m]")[0])))
        return np.asarray(context)[:, -1]

    res = W.blocked_cv(F, times, fit=fit, predict=predict, block_days=60, gap_days=6)
    nb = len(res["blocks"])
    assert nb == len(train_seen) == 12 and nb == len(set(res["block"]))
    gap = np.timedelta64(6 * 24 * 60, "m")
    for b, blk in enumerate(res["blocks"]):
        lo, hi = np.datetime64(blk["start"], "m"), np.datetime64(blk["end"], "m")
        tr = train_seen[b]
        assert not np.any((tr >= lo - gap) & (tr <= hi + gap))                        # gap on both sides of the block
        assert blk["n_train_steps"] == len(tr)
    assert len(calls) == res["n_items"] == len(res["init_time"]) == len(res["rmse_items"])
    assert all(shape == (1, 4, NLON, NLAT) and n == 1 for shape, n, _ in calls)
    for iso, b in zip(res["init_time"], res["block"]):
        t0 = np.datetime64(iso, "m")
        blk = res["blocks"][b]
        assert t0.astype("datetime64[h]").astype(int) % 24 in (0, 12)
        assert t0 - np.timedelta64(18, "h") >= np.datetime64(blk["start"], "m")
        assert t0 + np.timedelta64(24, "h") <= np.datetime64(blk["end"], "m")
    # the score is the documented per-initialisation lat-weighted RMSE of the callable's forecasts
    m = W.parse_times(times)
    pos = {str(t)[:16]: i for i, t in enumerate(m)}
    want = [W.lat_weighted_rmse(F[pos[s]][None], F[pos[s] + 4][None]) for s in res["init_time"]]
    assert np.allclose(res["rmse_items"], want) and res["rmse"] == pytest.approx(float(np.mean(want)))


def test_blocked_cv_default_model_is_deterministic_and_informative(data):
    F, times = data
    a = W.blocked_cv(F, times, block_days=90, radius=1)
    b = W.blocked_cv(F, times, block_days=90, radius=1)
    assert a["rmse"] == b["rmse"] and np.array_equal(a["rmse_items"], b["rmse_items"])
    assert a["n_items"] > 100 and all(np.isfinite(blk["rmse"]) for blk in a["blocks"])
    last = W.blocked_cv(F, times, block_days=90, fit=lambda f, t: None, predict=lambda mdl, c, t: np.asarray(c)[:, -1])
    assert np.array_equal(last["block"], a["block"]) and last["init_time"] == a["init_time"]   # same folds and items
    assert a["rmse"] < 0.9 * last["rmse"]


def test_blocked_cv_argument_errors(data):
    F, times = data
    with pytest.raises(ValueError):
        W.blocked_cv(F, times, fit=lambda f, t: None)                                # custom fit needs a predict
    with pytest.raises(ValueError):
        W.blocked_cv(F, times, fit=lambda f, t: None, predict=lambda m, c, t: c[:, -1], radius=2)
    with pytest.raises(ValueError):
        W.blocked_cv(F, times, n_blocks=1)
    with pytest.raises(ValueError):
        W.blocked_cv(F, times, context_offsets_h=(-18, -12, -6), radius=1)          # no field at init_time
    with pytest.raises(ValueError):
        W.blocked_cv(F, times, block_days=60, gap_days=400, radius=1)               # gap leaves nothing to train on
    with pytest.raises(ValueError):
        W.blocked_cv(F, times[::-1])
