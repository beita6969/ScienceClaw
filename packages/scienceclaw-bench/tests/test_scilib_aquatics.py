"""scilib.aquatics: interface text, sandbox import, deliverable format, layout/date handling, fallbacks, determinism,
dev truncation, agreement with the adapter's day-of-year window forecast and CRPS (synthetic NEON-like series)."""
from __future__ import annotations

import numpy as np
import pytest

import scilib
from scilib import aquatics as aq
from scienceclaw.runtime.integrity import scan_code

L, H = 1461, 30


def _series(n: int = 6, seed: int = 0, phi: float = 0.85):
    """n sites x (oxygen, temperature, chla): annual cycle (site-specific phase / amplitude) + AR(1) anomalies; NaN gaps."""
    rng = np.random.default_rng(seed)
    t0 = np.array(["2022-07-15", "2022-08-20", "2023-01-10", "2023-05-05", "2023-09-30", "2024-03-03"][:n], dtype="datetime64[D]")
    days = t0[:, None] - np.arange(L - 1, -1, -1).astype("timedelta64[D]")
    hist = np.full((n, L, 3), np.nan)
    for i in range(n):
        f = (days[i] - days[i].astype("datetime64[Y]")).astype(int) / 365.25
        for c, (mean, amp, sd) in enumerate([(9.0, 2.5, 0.6), (11.0, 8.0, 1.2), (3.0, 1.0, 0.5)]):
            a = np.zeros(L)
            e = rng.normal(0, sd, L)
            for t in range(1, L):
                a[t] = phi * a[t - 1] + e[t]
            y = mean + amp * np.cos(2 * np.pi * (f - 0.55 - 0.03 * i)) * (1 if c else -1) + a
            y[rng.random(L) < 0.1] = np.nan
            hist[i, :, c] = y
    hist[:, -1, :2] = np.where(np.isfinite(hist[:, -1, :2]), hist[:, -1, :2], 5.0)     # an observation on the last day
    return hist, [str(d) for d in t0]


@pytest.fixture(scope="module")
def data():
    return _series()


def test_describe_lists_every_public_name_and_avoids_reference_words():
    text = scilib.describe("aquatics")
    for name in aq.__all__:
        if name != "H":
            assert name in text
    low = text.lower()
    for word in ("naive", "no-change", "no change", "random walk", "persistence", "last value", "last-value", "climatolog",
                 "previous-day", "accepted iff", "success criterion"):
        assert word not in low
    import re

    assert not re.search(r"reference(?![_ ]date)", text, re.I)


def test_code_node_may_import_scilib():
    assert scan_code("from scilib.aquatics import fit_predict\n\ndef run(inputs, config):\n    return {}\n") == []


def test_fit_predict_format_ranges_and_determinism(data):
    hist, ref = data
    y = aq.fit_predict(hist, ref)
    assert set(y) == set(aq.KEYS) == {"oxygen_mu", "oxygen_sigma", "temperature_mu", "temperature_sigma"}
    for k, a in y.items():
        assert a.shape == (6, H) and np.isfinite(a).all()
    assert (y["oxygen_sigma"] >= aq.SIGMA_FLOOR).all() and (y["temperature_sigma"] >= aq.SIGMA_FLOOR).all()
    assert (y["oxygen_sigma"] <= 50).all() and (y["temperature_sigma"] <= 50).all()
    assert (0 <= y["oxygen_mu"]).all() and (y["oxygen_mu"] <= 30).all()
    assert (-5 <= y["temperature_mu"]).all() and (y["temperature_mu"] <= 45).all()
    y2 = aq.fit_predict(hist, ref)
    for k in y:
        np.testing.assert_array_equal(y[k], y2[k])


def test_ports_keywords_and_argument_checks(data):
    hist, ref = data
    inputs = dict(history=hist, variables=["oxygen", "temperature", "chla"], units=["mg/L", "degC", "ug/L"], horizon_days=30,
                  site_id=list("ABCDEF"), site_type=["lake"] * 6, reference_date=ref)
    y = aq.fit_predict(**inputs)
    z = aq.fit_predict(hist, ref)
    for k in y:
        np.testing.assert_array_equal(y[k], z[k])
    assert set(aq.doy_window_forecast(**inputs)) == set(aq.KEYS)
    assert aq.backtest(**inputs, n_origins=1)["shifts"] == [30]
    with pytest.raises(TypeError):
        aq.fit_predict(hist, ref, sigma_mult=2.0)
    with pytest.raises(ValueError):
        aq.fit_predict(hist, ref, lam=1.5)
    with pytest.raises(ValueError):
        aq.fit_predict(hist, ref, smult=0.0)
    with pytest.raises(ValueError):
        aq.fit_predict(hist, ref, horizon_days=10)
    with pytest.raises(ValueError):
        aq.fit_predict(hist, ref[:3])
    with pytest.raises(ValueError):
        aq.fit_predict(hist[:, :, 0], ref)


def test_variable_order_follows_the_variables_argument(data):
    hist, ref = data
    swapped = hist[:, :, [2, 1, 0]]
    y = aq.fit_predict(hist, ref)
    z = aq.fit_predict(swapped, ref, variables=["chla", "temperature", "oxygen"])
    for k in y:
        np.testing.assert_allclose(y[k], z[k])
    with pytest.raises(ValueError):
        aq.fit_predict(hist, ref, variables=["chla", "temperature", "x"])


def test_dates_follow_the_reference_date(data):
    hist, ref = data
    d = aq.history_dates(ref, L)
    assert d.shape == (6, L) and d[0, -1] == np.datetime64(ref[0]) and d[0, 0] == np.datetime64(ref[0]) - np.timedelta64(L - 1, "D")
    # a series that is a pure function of the calendar day is forecast from the calendar, whatever the history length
    t0 = np.datetime64("2023-06-01")
    days = t0 - np.arange(999, -1, -1).astype("timedelta64[D]")
    f = (days - days.astype("datetime64[Y]")).astype(int) / 365.25
    y = 8 + 3 * np.sin(2 * np.pi * f)
    h = np.stack([y, y + 5, y], 1)[None]
    out = aq.fit_predict(h, [str(t0)])
    tf = (t0 + np.arange(1, H + 1).astype("timedelta64[D]"))
    ff = (tf - tf.astype("datetime64[Y]")).astype(int) / 365.25
    np.testing.assert_allclose(out["oxygen_mu"][0], 8 + 3 * np.sin(2 * np.pi * ff), atol=0.1)
    np.testing.assert_allclose(out["temperature_mu"][0], 13 + 3 * np.sin(2 * np.pi * ff), atol=0.1)
    assert (out["oxygen_sigma"] <= 0.5).all()


def test_beats_day_of_year_windows_on_autocorrelated_series(data):
    hist, ref = data
    n = 6
    # hold out the last 30 days of the histories as targets
    cut, cut_ref = hist[:, :L - H], [str(np.datetime64(r) - np.timedelta64(H, "D")) for r in ref]
    obs = hist[:, L - H:, :2]
    s_new, per_new = aq.score(aq.fit_predict(cut, cut_ref), obs)
    s_win, _ = aq.score(aq.doy_window_forecast(cut, cut_ref), obs)
    assert s_new < s_win and set(per_new) == {"oxygen", "temperature"}
    bt = aq.backtest(hist, ref, n_origins=2)
    assert bt["shifts"] == [30, 60] and len(bt["per_origin"]) == 2
    assert bt["per_origin"][0] == pytest.approx(s_new)                        # shift 30 = the hold-out above
    bw = aq.backtest(hist, ref, n_origins=2, forecaster=aq.doy_window_forecast)
    assert bt["score"] < bw["score"]
    assert bt["score"] == pytest.approx(np.mean(list(bt["per_variable"].values()))) and n == 6


def test_uses_only_the_given_arrays_and_items_are_order_free(data):
    hist, ref = data
    cut = hist[:, :L - 40].copy()
    cut_ref = [str(np.datetime64(r) - np.timedelta64(40, "D")) for r in ref]
    # a history whose later days are set to NaN forecasts exactly like the same history cut at that day
    masked = hist.copy()
    masked[:, L - 40:] = np.nan
    a = aq.fit_predict(cut, cut_ref)
    b = aq.fit_predict(masked[:, :L - 40], cut_ref)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    # permuting the items permutes the rows (the pooled fit does not depend on the order)
    perm = [3, 0, 5, 1, 4, 2]
    p = aq.fit_predict(hist[perm], [ref[i] for i in perm])
    y = aq.fit_predict(hist, ref)
    for k in y:
        np.testing.assert_allclose(p[k], y[k][perm], atol=1e-9)


def test_lam_extremes_and_smult_scale_sigma(data):
    hist, ref = data
    y1 = aq.fit_predict(hist, ref, smult=1.0)
    y2 = aq.fit_predict(hist, ref, smult=2.0)
    big = y1["oxygen_sigma"] > aq.SIGMA_FLOOR
    np.testing.assert_allclose(y2["oxygen_sigma"][big], 2 * y1["oxygen_sigma"][big])
    np.testing.assert_allclose(y2["oxygen_mu"], y1["oxygen_mu"])
    pooled = aq.fit_predict(hist, ref, lam=0.0)
    single = aq.fit_predict(hist[:1], ref[:1], lam=0.0)
    own = aq.fit_predict(hist[:1], ref[:1], lam=1.0)
    # one item: its pooled fit is its own fit
    np.testing.assert_allclose(single["oxygen_mu"], own["oxygen_mu"])
    assert pooled["oxygen_mu"].shape == (6, H)


def test_sparse_and_empty_items(data):
    hist, ref = data
    h = hist.copy()
    h[2, :, 0] = np.nan
    h[2, -20:, 0] = 8.0 + np.arange(20) * 0.01                # 20 observed oxygen days only -> day-of-year window fallback
    rep = aq.data_report(h)
    row = rep[(rep["item"] == 2) & (rep["variable"] == "oxygen")].iloc[0]
    assert row["fallback"] and row["n_obs"] == 20 and row["age_days"] == 0 and row["n_obs_30"] == 20
    assert not rep[(rep["item"] == 0)]["fallback"].any() and list(rep.columns[:3]) == ["item", "variable", "n_obs"]
    y = aq.fit_predict(h, ref)
    w = aq.doy_window_forecast(h, ref)
    np.testing.assert_allclose(y["oxygen_mu"][2], w["oxygen_mu"][2])
    np.testing.assert_allclose(y["oxygen_sigma"][2], w["oxygen_sigma"][2])
    assert all(np.isfinite(a).all() for a in y.values())
    h[4, :, 1] = np.nan
    with pytest.raises(ValueError, match=r"no observed temperature.*\[4\]"):
        aq.fit_predict(h, ref)
    with pytest.raises(ValueError, match="no observed temperature"):
        aq.doy_window_forecast(h, ref)
    h2 = np.full_like(hist, np.nan)                            # nothing observed in any item: informative error
    with pytest.raises(ValueError):
        aq.fit_predict(h2, ref)


def test_window_forecast_equals_the_adapter_window_statistics(data):
    hist, ref = data
    from scienceclaw.bench.tasks.for41_neon import climatology

    w = aq.doy_window_forecast(hist, ref)
    for i in (0, 3):
        t0 = np.datetime64(ref[i])
        hd = t0 - np.arange(L - 1, -1, -1).astype("timedelta64[D]")
        td = t0 + np.arange(1, H + 1).astype("timedelta64[D]")
        for j, v in enumerate(("oxygen", "temperature")):
            mu, sd = climatology(hist[i, :, j], hd, td)
            np.testing.assert_allclose(w[f"{v}_mu"][i], mu)
            np.testing.assert_allclose(w[f"{v}_sigma"][i], sd)


def test_crps_and_score_match_the_adapter_metric(data):
    from scienceclaw.bench.tasks._forecast_common import crps_normal as adapter_crps

    rng = np.random.default_rng(1)
    mu, sg, y = rng.normal(size=200), rng.uniform(0.1, 3, 200), rng.normal(size=200) * 2
    np.testing.assert_allclose(aq.crps_normal(mu, sg, y), adapter_crps(mu, sg, y))
    with pytest.raises(ValueError):
        aq.crps_normal(0.0, 0.0, 1.0)
    hist, ref = data
    pred = aq.fit_predict(hist, ref)
    obs = np.full((6, H, 2), np.nan)
    obs[:, :10, 0] = 9.0
    obs[:, 5:15, 1] = 12.0
    s, per = aq.score(pred, obs)
    manual = {"oxygen": aq.crps_normal(pred["oxygen_mu"][:, :10], pred["oxygen_sigma"][:, :10], 9.0).mean(),
              "temperature": aq.crps_normal(pred["temperature_mu"][:, 5:15], pred["temperature_sigma"][:, 5:15], 12.0).mean()}
    assert per["oxygen"] == pytest.approx(manual["oxygen"]) and per["temperature"] == pytest.approx(manual["temperature"])
    assert s == pytest.approx((manual["oxygen"] + manual["temperature"]) / 2)
    with pytest.raises(ValueError):
        aq.score(pred, obs[:, :5])


def test_seasonal_profile_block():
    d = np.datetime64("2021-01-01") + np.arange(1461).astype("timedelta64[D]")
    f = (d - d.astype("datetime64[Y]")).astype(int) / 365.25
    y = 10 + 4 * np.sin(2 * np.pi * f) + 1.0 * np.cos(4 * np.pi * f)
    y[::7] = np.nan
    fit, fut = aq.seasonal_profile(y, d, d[-10:] + np.timedelta64(10, "D"))
    assert np.nanmax(np.abs(fit - y)) < 0.3 and fut.shape == (10,)
    assert aq.seasonal_profile(y, d)[1] is None
    with pytest.raises(ValueError):
        aq.seasonal_profile(np.full(len(d), np.nan), d)


def test_speed_and_memory_light(data):
    import time

    hist, ref = data
    t = time.time()
    aq.fit_predict(np.repeat(hist, 3, axis=0), ref * 3)
    assert time.time() - t < 15
