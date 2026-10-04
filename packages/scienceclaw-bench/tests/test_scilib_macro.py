"""scilib.macro: interface text, sandbox import, roles, pooled forecasts, weights, determinism, dev truncation (synthetic panel)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scilib
from scilib import macro
from scienceclaw.runtime.integrity import scan_code

L = 24
KINDS = {"T1": "share of a total, percentage of a total (0-100)", "T2": "a level in constant units",
         "C1": "annual percentage change of a level", "C2": "annual percentage change of another level"}


def _panel(n: int = 30, seed: int = 0, length: int = L):
    """n economies x 4 kinds: T2 = level with a drift shared by the economies (random-walk noise), T1 = share, C1 / C2 = changes."""
    rng = np.random.default_rng(seed)
    cols = [f"t{-(length - 1 - j)}" if j < length - 1 else "t0" for j in range(length)]
    rows = []
    vals = {}
    for i in range(n):
        g = rng.normal(0.02, 0.02)
        lv = np.exp(np.log(rng.uniform(500, 5000)) + np.cumsum(g + rng.normal(0, 0.04, length)))
        sh = np.clip(rng.uniform(3, 20) * np.exp(np.cumsum(rng.normal(0, 0.06, length))), 0.2, 60)
        c1 = np.r_[np.nan, np.diff(np.log(lv))] * 100
        c2 = rng.normal(3, 4, length)
        vals[i] = {"T2": lv, "T1": sh, "C1": c1, "C2": c2}
        for k in KINDS:
            rows.append([f"E{i:03d}", "R" + str(i % 3), "L" + str(i % 2), k, *vals[i][k]])
    panel = pd.DataFrame(rows, columns=["economy_id", "region", "income_level", "indicator", *cols])
    return panel, vals


def _items(vals, ids, length: int = L):
    ind = ["T2" if j % 2 == 0 else "T1" for j in range(len(ids))]
    hist = np.stack([vals[i][k][:length] for i, k in zip(ids, ind)])
    cov = np.stack([[vals[i][k][:length] for k in ("T1", "T2", "C1", "C2")] for i in ids])
    return dict(history=hist, indicator=ind, covariates=cov, covariate_indicators=["T1", "T2", "C1", "C2"],
                economy_id=[f"E{i:03d}" for i in ids], region=["R0"] * len(ids), income_level=["L0"] * len(ids),
                time_offsets=list(range(-(length - 1), 1)), target_offsets=[1, 2, 3, 4])


@pytest.fixture(scope="module")
def data():
    panel, vals = _panel()
    return panel, vals, _items(vals, [1, 2, 3, 4, 5, 6])


def test_describe_lists_every_public_name():
    text = scilib.describe("macro")
    for name in macro.__all__:
        assert name in text
    for name in macro.MEMBERS:
        assert name in text
    low = text.lower()
    for word in ("naive", "no-change", "no change", "random walk", "reference", "persistence", "last value"):
        assert word not in low


def test_code_node_may_import_scilib():
    assert scan_code("from scilib.macro import fit_predict\n\ndef run(inputs, config):\n    return {}\n") == []


def test_roles_from_descriptions_and_from_values(data):
    panel, _, _ = data
    assert macro.infer_roles(panel, KINDS) == {"T1": "percent", "T2": "level", "C1": "change", "C2": "change"}
    assert macro.infer_roles(panel) == {"T1": "percent", "T2": "level", "C1": "change", "C2": "change"}
    ids, arrays, meta = macro.panel_to_arrays(panel, 10)
    assert len(ids) == 30 and arrays["T2"].shape == (30, 10) and meta["region"]["E001"] == "R1"


def test_fit_predict_shape_positive_percent_bound_and_determinism(data):
    panel, _, ev = data
    p1 = macro.fit_predict(panel, indicator_kinds=KINDS, **ev)
    p2 = macro.fit_predict(panel, indicator_kinds=KINDS, **ev)
    assert p1.shape == (6, 4) and np.isfinite(p1).all() and (p1 > 0).all() and np.array_equal(p1, p2)
    pct = np.array([k == "T1" for k in ev["indicator"]])
    assert (p1[pct] <= 100).all()
    assert np.abs(np.log(p1 / ev["history"][:, [-1]])).max() < 1.0            # forecasts stay near the origin value


def test_flat_member_and_local_forecast_return_the_origin_value(data):
    panel, _, ev = data
    p = macro.fit_predict(panel, indicator_kinds=KINDS, members=["flat"], **ev)
    assert np.allclose(p, ev["history"][:, [-1]] * np.ones((1, 4)))
    x = np.array([10.0, 11.0, np.nan, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0])
    assert np.allclose(macro.local_forecast(x, "flat", 4), 21.0)
    d = macro.local_forecast(x, "drift", 4)
    assert d.shape == (4,) and (np.diff(np.r_[21.0, d]) > 0).all()
    for method in ("revert", "theta", "holt"):
        assert np.isfinite(macro.local_forecast(x, method, 3)).all()
    with pytest.raises(ValueError):
        macro.local_forecast(x, "arima")


def test_pooled_models_beat_flat_on_a_planted_pattern():
    """Level series with a common drift: the pooled models (which see all economies' windows) should beat the flat forecast."""
    panel, vals = _panel(60, seed=3, length=28)
    ids = list(range(50, 60))
    ev = _items(vals, ids, length=24)
    y = np.stack([vals[i]["T2"][24:28] for i in ids])
    lv = np.array([k == "T2" for k in ev["indicator"]])
    p, info = macro.fit_predict(panel, indicator_kinds=KINDS, return_info=True, **ev)
    flat = np.repeat(ev["history"][:, [-1]], 4, axis=1)

    def sm(a, b):
        return 200 * np.mean(np.abs(a - b) / (np.abs(a) + np.abs(b)))
    assert sm(y[lv], p[lv]) < sm(y[lv], flat[lv])
    assert set(info["backtest_smape"]) == {"T1", "T2"} and {"flat", "combination", *macro.DEFAULT_MEMBERS} <= set(info["backtest_smape"]["T2"])
    assert info["roles"]["T2"] == "level" and info["weights"]["T2"] == {m: 0.25 for m in macro.DEFAULT_MEMBERS}
    # rolling backtest: 8 origins spaced one period apart, newest = horizon periods before the last one
    assert info["backtest_origins"] == [-4 - j for j in range(8)]
    by = info["backtest_by_origin"]["T2"]
    assert len(by["flat"]) == 8 and len(by["ridge"]) == 8 and np.isclose(np.mean(by["flat"]), info["backtest_smape"]["T2"]["flat"], rtol=0.3)


def test_dev_history_truncates_the_panel(data):
    """A shorter history (the dev split) uses only the first len(history) period columns: appending later columns changes nothing."""
    panel, vals, _ = data
    dev = _items(vals, [1, 2, 3, 4], length=L - 4)
    p1 = macro.fit_predict(panel, indicator_kinds=KINDS, **dev)
    cols = [c for c in panel.columns if c.startswith("t")]
    panel2 = panel.copy()
    panel2[cols[-4:]] = panel2[cols[-4:]] * 7.0          # later periods must not matter
    p2 = macro.fit_predict(panel2, indicator_kinds=KINDS, **dev)
    assert np.array_equal(p1, p2)
    pool = macro.build_pool(panel, dev["history"], dev["indicator"], KINDS, dev["covariates"], dev["covariate_indicators"], dev["economy_id"])
    assert pool.L == L - 4 and pool.X["T2"].shape[1] == L - 4


def test_members_have_no_look_ahead(data):
    """member_forecasts(origin) reads only columns <= origin: overwriting later columns leaves the forecasts unchanged."""
    panel, _, ev = data
    pool = macro.build_pool(panel, ev["history"], ev["indicator"], KINDS, ev["covariates"], ev["covariate_indicators"], ev["economy_id"])
    names = [m for m in macro.MEMBERS if m != "flat"]
    a = macro.member_forecasts(pool, "T2", 15, names, 4)
    for k in pool.X:
        pool.X[k][:, 16:] *= 3.0
    pool2 = macro.Pool(pool.econ, pool.X, pool.roles, pool.items, pool.L)
    b = macro.member_forecasts(pool2, "T2", 15, names, 4)
    for m in a:
        assert a[m].shape == (len(pool.econ), 4) and np.allclose(a[m], b[m]), m


def test_rolling_backtest_and_fit_weights():
    rng = np.random.default_rng(0)
    truth = rng.normal(0, 0.2, (200, 4))
    good, bad = truth + rng.normal(0, 0.03, truth.shape), rng.normal(0, 0.2, truth.shape)
    w = macro.fit_weights({"good": [good[:100], good[100:]], "bad": [bad[:100], bad[100:]]}, [truth[:100], truth[100:]])
    assert w["good"] > 0.8 and w["bad"] < 0.2 and sum(w.values()) <= 1.0 + 1e-9
    w0 = macro.fit_weights({"bad": bad}, truth, cap=1.0)                       # a useless member is shrunk towards the flat forecast
    assert w0["bad"] < 0.3
    assert macro.smape_logratio(np.zeros(3), np.zeros(3)) == 0.0
    assert macro.smape_logratio(np.array([0.1]), np.array([0.0])) == pytest.approx(200 * np.tanh(0.05))
    panel, _ = _panel(30)
    ev = _items(_panel(30)[1], [1, 2], L)
    pool = macro.build_pool(panel, ev["history"], ev["indicator"], KINDS, ev["covariates"], ev["covariate_indicators"], ev["economy_id"])
    bt = macro.rolling_backtest(pool, "T2", ["drift", "ridge"], 4, n_origins=2)
    assert bt["origins"] == [L - 5, L - 6] and len(bt["pred"]["ridge"]) == 2 and bt["pred"]["ridge"][0].shape == bt["actual"][0].shape
    bt4 = macro.rolling_backtest(pool, "T2", ["drift"], 4, n_origins=2, step=4)
    assert bt4["origins"] == [L - 5, L - 9]
    assert macro.rolling_backtest(pool, "T2", ["drift"], 4, n_origins=50)["origins"] == list(range(L - 5, 11, -1))       # never before column 12


def test_auto_weights_and_errors(data):
    panel, _, ev = data
    p, info = macro.fit_predict(panel, indicator_kinds=KINDS, weights="auto", members=["drift", "ridge"], n_backtest=2, return_info=True, **ev)
    assert p.shape == (6, 4) and all(sum(w.values()) <= 1.0 + 1e-9 for w in info["weights"].values())
    with pytest.raises(ValueError):
        macro.fit_predict(panel, indicator_kinds=KINDS, members=["arima"], **ev)
    with pytest.raises(ValueError):
        macro.fit_predict(panel, indicator_kinds=KINDS, weights="best", **ev)
    bad = dict(ev)
    bad["history"] = ev["history"].copy()
    bad["history"][0] = np.nan
    bad["covariates"] = None
    with pytest.raises(ValueError, match="no positive observation"):
        macro.fit_predict(panel, indicator_kinds=KINDS, **{**bad, "economy_id": None})


def test_every_member_is_accepted_by_fit_predict_and_the_default_is_the_documented_set(data):
    panel, _, ev = data
    assert macro.DEFAULT_MEMBERS == ("ridge", "huber", "lgbm_core", "robdrift") and set(macro.DEFAULT_MEMBERS) <= set(macro.MEMBERS)
    for m in macro.MEMBERS:
        p = macro.fit_predict(panel, indicator_kinds=KINDS, members=[m], **ev)
        assert p.shape == (6, 4) and np.isfinite(p).all() and (p > 0).all(), m
    with pytest.raises(ValueError):
        macro.fit_predict(panel, indicator_kinds=KINDS, members=[], **ev)
    d = macro.fit_predict(panel, indicator_kinds=KINDS, **ev)
    w = macro.fit_predict(panel, indicator_kinds=KINDS, members=list(macro.DEFAULT_MEMBERS), weights={m: 0.25 for m in macro.DEFAULT_MEMBERS}, **ev)
    assert np.allclose(d, w)


def test_core_members_ignore_the_other_kinds_but_ridge_uses_them(data):
    panel, _, ev = data
    pool = macro.build_pool(panel, ev["history"], ev["indicator"], KINDS, ev["covariates"], ev["covariate_indicators"], ev["economy_id"])
    a = macro.member_forecasts(pool, "T2", 18, ["ridge", "huber", "lgbm_core"], 4)
    X2 = {k: v.copy() for k, v in pool.X.items()}
    X2["C1"] = X2["C1"] * 5.0 + 20.0
    X2["C2"] = -X2["C2"]
    b = macro.member_forecasts(macro.Pool(pool.econ, X2, pool.roles, pool.items, pool.L), "T2", 18, ["ridge", "huber", "lgbm_core"], 4)
    assert np.allclose(a["huber"], b["huber"]) and np.allclose(a["lgbm_core"], b["lgbm_core"])
    assert not np.allclose(a["ridge"], b["ridge"])


def test_robust_drift_uses_the_median_and_shrinks_towards_the_pool():
    ly = np.log(np.array([[100.0 * 1.05 ** j for j in range(20)],                       # steady 5 % growth
                          [100.0 * 1.05 ** j for j in range(19)] + [1000.0],            # same, with one outlier change at the end
                          [100.0] * 20]))
    r = macro._robdrift(ly, 3, shrink=0.0)
    assert np.allclose(r[0], r[1]) and np.allclose(r[2], 0.0)                          # an outlier change does not move the median
    r4 = macro._robdrift(ly, 3, shrink=1.0)
    assert np.allclose(r4[0], r4[2])                                                    # full shrink: every row gets the pool median
    assert (np.diff(r[0]) > 0).all() and r[0, 0] == pytest.approx(np.log(1.05) * 0.9)


def test_period_profile_shows_a_common_shock():
    panel, _ = _panel(40, seed=5, length=20)
    cols = [c for c in panel.columns if c.startswith("t")]
    t2 = panel["indicator"] == "T2"
    panel.loc[t2, cols[-1]] = panel.loc[t2, cols[-2]] * np.exp(-0.3)                    # every economy falls by 30 % in the last period
    prof = macro.period_profile(panel, last=6, indicator_kinds=KINDS)
    assert prof["T2"]["periods"] == cols[-6:] and prof["T2"]["median"].shape == (6,)
    assert prof["T2"]["median"][-1] == pytest.approx(-0.3, abs=1e-9) and prof["T2"]["mad"][-1] < 1e-9
    assert prof["T2"]["mad"][0] > 0.005 and abs(prof["T2"]["median"][0]) < 0.1
    assert "C1" in prof and prof["C1"]["median"].shape == (6,)
    short = macro.period_profile(panel, n_periods=10, last=50, indicator_kinds=KINDS)
    assert short["T2"]["median"].shape == (9,)


def test_shrunk_fit_weights_keep_half_of_the_equal_weight():
    """fit_weights(shrink=0.5) mixes the fitted weights 50:50 with equal weights (the 'auto' setting of fit_predict), so every weight >= 0.5 / n_members."""
    rng = np.random.default_rng(1)
    truth = rng.normal(0, 0.2, (300, 4))
    preds = {m: truth * 0.3 + rng.normal(0, 0.2, truth.shape) for m in ("a", "b", "c")}
    w = macro.fit_weights(preds, truth, shrink=0.5)
    assert min(w.values()) >= 0.5 * (1.0 / 3) - 1e-9 and sum(w.values()) <= 1.0 + 1e-9
