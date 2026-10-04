"""scilib.loadforecast: lag/calendar alignment, metric parity with the FoR33 adapter, context-only forecasts, window
extraction from the history only, the pooled LightGBM + Ridge model on synthetic loads, interface text, sandbox import."""
from __future__ import annotations

import re
import tempfile

import numpy as np
import pandas as pd
import pytest

import scilib
from scilib import loadforecast as lf
from scienceclaw.bench.tasks import for33_buildingsbench as m
from scienceclaw.core.graph import Node
from scienceclaw.core.schema import PortSchema
from scienceclaw.runtime.integrity import scan_code
from scienceclaw.runtime.sandbox import run_code_node

START = "2016-03-01T00:00:00"                     # a Tuesday


def _synthetic(nb: int = 3, T: int = 1500, seed: int = 0, start: str = START):
    """Hourly loads with a weekly shape (evening peak, quiet weekends), per-building level and multiplicative noise."""
    rng = np.random.default_rng(seed)
    hrs = pd.date_range(start, periods=T, freq="h")
    hod, dow = hrs.hour.values, hrs.dayofweek.values
    shape = 1.0 + 0.8 * np.exp(-((hod - 19) / 3.0) ** 2) - 0.4 * (dow >= 5) * (hod > 8)
    load = np.stack([(5.0 + 20 * b) * shape * rng.lognormal(0.0, 0.35, T) for b in range(nb)])
    cats = ["residential", "commercial", "residential"][:nb]
    return load, [f"b{b}" for b in range(nb)], cats, [start] * nb


# ------------------------------------------------------------------------------------------------ metric / alignment
def test_metric_matches_the_adapter():
    rng = np.random.default_rng(1)
    y = rng.gamma(2.0, 5.0, (12, 24))
    p = y * rng.lognormal(0, 0.3, y.shape)
    ids = ["a", "b", "c", "d"] * 3
    cats = {"a": "residential", "b": "commercial", "c": "residential", "d": "commercial"}
    assert lf.cvrmse(y, p) == pytest.approx(m.nrmse_pct(y, p))
    per = m.per_building_nrmse(y, p, ids)
    assert lf.cvrmse_by_building(y, p, ids) == pytest.approx(per)
    assert lf.balanced_cvrmse(y, p, ids, [cats[i] for i in ids]) == pytest.approx(m.balanced_nrmse(y, p, ids, cats)[0])
    with pytest.raises(ValueError):
        lf.cvrmse(np.zeros((2, 24)), np.ones((2, 24)))


def test_lag_matrix_alignment_and_simple_forecasts():
    ctx = np.arange(2 * 168, dtype=float).reshape(2, 168)
    L = lf.lag_matrix(ctx)
    assert L.shape == (2, 24, 7)
    for k in range(1, 8):
        for h in (0, 5, 23):
            assert L[1, h, k - 1] == ctx[1, 168 - 24 * k + h]
    assert np.array_equal(lf.yesterday(ctx), ctx[:, 144:]) and np.array_equal(lf.yesterday(ctx), m.persistence(ctx))
    assert np.allclose(lf.day_mean(ctx), L.mean(2)) and np.allclose(lf.day_median(ctx, 3), np.median(L[:, :, :3], 2))
    assert lf.lag_matrix(ctx[0]).shape == (1, 24, 7)                       # a single context row is accepted
    for bad in (np.zeros((2, 100)), np.full((2, 168), np.nan)):
        with pytest.raises(ValueError):
            lf.lag_matrix(bad)


def test_blend_weight_backtest_and_core_forecast():
    A, B = 10.0, 30.0
    d = np.full((1, 7, 24), A)
    d[:, 5:] = B                                                    # level shift on the last two context days
    ctx = d.reshape(1, 168)
    assert lf.blend_weight(ctx)[0] == pytest.approx(2 / 3, abs=1e-6)  # yesterday errs (B-A)^2 once, the median twice
    assert lf.blend_weight(np.full((1, 168), 4.0))[0] == 0.0          # zero backtest error on both: median
    rng = np.random.default_rng(2)
    c = rng.gamma(2.0, 3.0, (9, 168))
    w = lf.blend_weight(c)
    assert w.shape == (9,) and np.all((w >= 0) & (w <= 1))
    cats = ["residential", "commercial", "commercial"] * 3
    core = lf.core_forecast(c, cats)
    res = np.array(cats) == "residential"
    assert np.allclose(core[res], lf.day_median(c)[res]) and np.allclose(core[~res], lf.backtest_blend(c)[~res])
    assert np.allclose(lf.backtest_blend(c), w[:, None] * lf.yesterday(c) + (1 - w[:, None]) * lf.day_median(c))
    with pytest.raises(ValueError, match="category"):
        lf.core_forecast(c, ["home"] * 9)


def test_calendar_features_match_pandas_and_accept_both_timestamp_styles():
    starts = ["2016-11-26 01:00:00", "2012-10-28T14:00:00", "2018-05-21 11:00:00"]
    cal = lf.calendar_features(starts)
    for i, s in enumerate(starts):
        idx = pd.date_range(pd.Timestamp(s), periods=24, freq="h")
        assert np.array_equal(cal["hour"][i], idx.hour) and np.array_equal(cal["dayofweek"][i], idx.dayofweek)
        assert np.array_equal(cal["weekend"][i], (idx.dayofweek >= 5).astype(int))
        assert np.array_equal(cal["hour_of_week"][i], idx.dayofweek * 24 + idx.hour)
    assert lf.calendar_features(np.array(["2016-03-01T00"], dtype="datetime64[h]"), hours=48)["hour"].shape == (1, 48)


def test_history_windows_use_only_the_given_history():
    load, ids, cats, starts = _synthetic(2, T=700)
    w = lf.history_windows(load, starts, ids, cats, stride=24)
    per_b = len(range(168, 700 - 24 + 1, 24))
    assert w["context"].shape == (2 * per_b, 168) and w["target"].shape == (2 * per_b, 24)
    assert w["building_id"][:per_b] == [ids[0]] * per_b and w["category"][-1] == cats[1]
    j = per_b + 3                                                    # window j: building 1, target start s = 168 + 3 * 24
    s = 168 + 3 * 24
    assert np.array_equal(w["context"][j], load[1, s - 168:s]) and np.array_equal(w["target"][j], load[1, s:s + 24])
    assert w["target_start"][j] == str(pd.Timestamp(starts[1]) + pd.Timedelta(hours=s)).replace(" ", "T")
    assert pd.Timestamp(w["target_start"][j]) - pd.Timestamp(w["context_start"][j]) == pd.Timedelta(hours=168)
    sub = lf.history_windows(load, starts, ids, cats, stride=24, first_hour=100, last_hour=500)
    assert sub["context"].shape[0] == 2 * len(range(268, 500 - 24 + 1, 24))
    assert np.array_equal(sub["target"][0], load[0, 268:292])
    with pytest.raises(ValueError, match="too short"):
        lf.history_windows(load[:, :150], starts, ids, cats)


def test_hour_of_week_profile():
    x = np.tile(np.arange(168, dtype=float) + 1.0, 6)
    p = lf.hour_of_week_profile(x, "2016-02-29T00:00:00")            # a Monday: hour i of the history is hour-of-week i
    assert p.shape == (168,) and np.allclose(p, (np.arange(168) + 1.0) / x.mean())
    shifted = lf.hour_of_week_profile(x, "2016-03-01T00:00:00")      # start on Tuesday: the profile is rolled by 24 h
    assert np.allclose(shifted, np.roll(p, 24))


def test_block_profiles_leave_out_the_target_block():
    rng = np.random.default_rng(3)
    x = rng.uniform(1.0, 9.0, 1200)
    h0 = int(lf._hours_since_epoch("2016-03-01T00:00:00")[0])
    whole, _ = lf._block_profiles(x, h0, 1)
    assert np.allclose(whole[0], lf.hour_of_week_profile(x, "2016-03-01T00:00:00"))
    blocks, edges = lf._block_profiles(x, h0, 4)
    assert blocks.shape == (4, 168) and list(edges) == [0, 300, 600, 900, 1200]
    for j in range(4):
        keep = np.ones(1200, bool)
        keep[edges[j]:edges[j + 1]] = False
        hrs = h0 + np.arange(1200)[keep]
        how = ((hrs // 24 + 3) % 7) * 24 + hrs % 24
        want = np.bincount(how, weights=x[keep], minlength=168) / np.maximum(np.bincount(how, minlength=168), 1) / x[keep].mean()
        assert np.allclose(blocks[j], want)


# ------------------------------------------------------------------------------------------------ learned model
@pytest.fixture(scope="module")
def fitted():
    load, ids, cats, starts = _synthetic(3, T=1500, seed=3)
    hist = load[:, :1200]
    model = lf.fit(hist, starts, ids, cats, stride=12, n_estimators=40)
    # evaluation windows from the later block of the same process (targets not in the fitted history)
    w = lf.history_windows(load, starts, ids, cats, stride=24, first_hour=1200 - 168)
    return model, load, ids, cats, starts, w


def test_candidates_shapes_nonnegative_and_learned_model_uses_the_weekly_shape(fitted):
    model, load, ids, cats, starts, w = fitted
    c = model.candidates(w["context"], w["target_start"], w["building_id"], w["category"])
    assert list(c) == list(lf.CANDIDATES)
    n = len(w["context"])
    for k, v in c.items():
        assert v.shape == (n, 24) and np.all(np.isfinite(v)) and v.min() >= 0, k
    assert np.allclose(c["ens"], 0.5 * c["core"] + 0.5 * c["ml"])
    y = w["target"]
    err = {k: lf.cvrmse(y, v) for k, v in c.items()}
    assert err["ml"] < err["yesterday"] and err["median7"] < err["yesterday"]     # noise makes yesterday a poor forecast
    assert model.n_train_windows_ == 3 * len(range(168, 1200 - 24 + 1, 12))
    assert np.allclose(model.predict(w["context"], w["target_start"], w["building_id"], w["category"], w_core=0.25),
                       np.maximum(0.25 * c["core"] + 0.75 * c["ml"], 0))
    assert np.array_equal(model.ml_forecast(w["context"], w["target_start"], w["building_id"], w["category"]), c["ml"])


def test_window_forecasts_do_not_depend_on_other_windows(fitted):
    """Row i of every candidate is a function of window i's own context (no pooling across windows of a building)."""
    model, load, ids, cats, starts, w = fitted
    args = (w["context"], w["target_start"], w["building_id"], w["category"])
    full = model.candidates(*args)
    rng = np.random.default_rng(0)
    other = rng.permutation(len(w["context"]))[:5]
    for i in (0, len(w["context"]) // 2, len(w["context"]) - 1):
        single = model.candidates(w["context"][i:i + 1], w["target_start"][i:i + 1], w["building_id"][i:i + 1],
                                  w["category"][i:i + 1])
        for k in full:
            assert np.allclose(single[k][0], full[k][i], rtol=1e-9, atol=1e-9), k
    perturbed = w["context"].copy()
    perturbed[other] *= 3.0                                        # change other windows' contexts
    keep = np.setdiff1d(np.arange(len(perturbed)), other)
    again = model.candidates(perturbed, *args[1:])
    assert all(np.allclose(again[k][keep], full[k][keep]) for k in full)


def test_fit_is_deterministic_and_reusable_through_forecast_candidates(fitted):
    model, load, ids, cats, starts, w = fitted
    args = (w["context"][:6], w["target_start"][:6], w["building_id"][:6], w["category"][:6])
    a = lf.forecast_candidates(load[:, :1200], starts, ids, cats, *args, stride=12, n_estimators=40)   # an independent fit
    b = model.candidates(*args)
    c = lf.forecast_candidates(None, None, None, None, *args, model=model)
    for k in a:
        assert np.array_equal(a[k], b[k]), k
        assert np.array_equal(b[k], c[k]), k


def test_informative_errors(fitted):
    model, load, ids, cats, starts, w = fitted
    ctx, ts = w["context"][:2], w["target_start"][:2]
    with pytest.raises(ValueError, match="not passed to fit"):
        model.candidates(ctx, ts, ["zzz", ids[0]], ["residential"] * 2)
    with pytest.raises(ValueError, match="category"):
        model.candidates(ctx, ts, ids[:2], ["office", "home"])
    with pytest.raises(ValueError, match="shape"):
        model.candidates(ctx[:, :100], ts, ids[:2], cats[:2])
    with pytest.raises(ValueError, match="one distinct"):
        lf.fit(load[:, :400], starts, ["x", "x", "y"], cats)
    with pytest.raises(RuntimeError, match="fit"):
        lf.LoadForecaster().candidates(ctx, ts, ids[:2], cats[:2])


def test_thread_cap():
    assert lf.LoadForecaster(n_jobs=16).n_jobs == 2 and lf.LoadForecaster(n_jobs=0).n_jobs == 1


def test_backtest_history_fits_without_the_holdout(monkeypatch):
    load, ids, cats, starts = _synthetic(2, T=1300, seed=5)
    seen = {}
    real = lf.fit

    def spy(x, *a, **k):
        seen["shape"] = np.asarray(x).shape
        return real(x, *a, **k)

    monkeypatch.setattr(lf, "fit", spy)
    r = lf.backtest_history(load, starts, ids, cats[:2], holdout_hours=600, stride=24, n_estimators=30)
    assert seen["shape"] == (2, 700)                                     # the last 600 hours never reach fit()
    assert r["n_windows"] == 2 * len(range(0, 600 - 24 + 1, 24))
    assert set(r["scores"]) == set(lf.CANDIDATES) == set(r["per_building"])
    assert all(np.isfinite(v) and v > 0 for v in r["scores"].values())
    with pytest.raises(ValueError, match="too little"):
        lf.backtest_history(load, starts, ids, cats[:2], holdout_hours=1250)


def test_backtest_history_leaves_out_buildings_without_holdout_load():
    load, ids, cats, starts = _synthetic(3, T=1300, seed=6)
    load = np.array(load, dtype=float)
    load[2, 700:] = 0.0                                                  # no load at all in the holdout
    r = lf.backtest_history(load, starts, ids, cats[:3], holdout_hours=600, stride=24, n_estimators=30)
    n_win = len(range(0, 600 - 24 + 1, 24))
    assert r["n_windows"] == 2 * n_win
    assert set(r["per_building"]["ens"]) == set(ids[:2])
    assert all(np.isfinite(v) and v > 0 for v in r["scores"].values())
    load[:, 700:] = 0.0
    with pytest.raises(ValueError, match="positive load"):
        lf.backtest_history(load, starts, ids, cats[:3], holdout_hours=600, stride=24, n_estimators=30)


def test_pick_lowest():
    cand = {"a": np.zeros((1, 24)), "b": np.ones((1, 24)), "ens": np.full((1, 24), 2.0)}
    assert lf.pick_lowest(cand, {"a": 5.0, "b": 4.0, "ens": 4.5})[0] == "b"
    assert lf.pick_lowest(cand, {"a": 5.0, "b": 4.0, "ens": 4.1})[0] == "ens"                   # default rel_tol = 3 %
    assert lf.pick_lowest(cand, {"a": 5.0, "b": 4.0, "ens": 4.1}, rel_tol=0.0)[0] == "b"
    assert lf.pick_lowest(cand, {"a": 5.0, "b": 4.0, "ens": 4.5}, rel_tol=0.2)[0] == "ens"      # within 20 % of the lowest
    assert lf.pick_lowest(cand, {"a": 5.0, "b": 4.0})[0] == "b"                                # 'ens' unscored: ignored
    with pytest.raises(ValueError):
        lf.pick_lowest(cand, {})


# ------------------------------------------------------------------------------------------------ interface text
def test_describe_lists_every_public_name_and_is_factual():
    text = scilib.describe("loadforecast")
    for name in lf.__all__:
        assert name in text, name
    for cand in lf.CANDIDATES:
        assert f"'{cand}'" in text
    low = text.lower()
    # the policy-visible text must stay method-neutral about the scoring baseline (tests/test_adapter_visible_text.py)
    for word in ("persistence", "naive", "previous-day", "previous day", "reference", "accept", "margin", "should", "best"):
        assert word not in low, word
    assert not re.search(r"\b0\.[89]\d?\s*(x|×)", text)


def test_code_node_may_import_scilib():
    assert scan_code("from scilib import loadforecast as lf\nfrom scilib.loadforecast import fit\n\n"
                     "def run(inputs, config):\n    return {}\n") == []


def test_library_runs_inside_the_code_node_sandbox():
    load, ids, cats, starts = _synthetic(2, T=700, seed=7)
    w = lf.history_windows(load, starts, ids, cats, stride=24, first_hour=400)
    code = (
        "import numpy as np\n"
        "from scilib import loadforecast as lf\n\n"
        "def run(inputs, config):\n"
        "    m = lf.fit(inputs['load'], inputs['start'], inputs['ids'], inputs['cats'], stride=24, n_estimators=20)\n"
        "    c = m.candidates(inputs['ctx'], inputs['ts'], inputs['bid'], inputs['cat'])\n"
        "    return {'y': c['ens'], 'names': list(c)}\n")
    vals = {"load": load, "start": starts, "ids": ids, "cats": cats, "ctx": w["context"][:4], "ts": w["target_start"][:4],
            "bid": w["building_id"][:4], "cat": w["category"][:4]}
    node = Node(id="solve", kind="code", code=code, inputs={k: PortSchema("any") for k in vals},
                outputs={"y": PortSchema("any"), "names": PortSchema("any")})
    with tempfile.TemporaryDirectory() as td:
        out, meta = run_code_node(node, vals, td, 120)
    assert out is not None, meta.get("error")
    assert np.asarray(out["y"]).shape == (4, 24) and out["names"] == list(lf.CANDIDATES)
