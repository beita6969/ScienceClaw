"""scilib.forecast: metric parity with the adapter, methods on synthetic seasonal panels, fallbacks, backtest, combination,
global model, interface text, sandbox import."""
from __future__ import annotations

import numpy as np
import pytest

import scilib
from scilib import forecast as fc
from scienceclaw.bench.tasks import _forecast_common as common
from scienceclaw.bench.tasks.for35_tourism import snaive as adapter_snaive
from scienceclaw.runtime.integrity import scan_code

H, M = 12, 4


def _panel(n: int = 8, length: int = 60, seed: int = 0, noise: float = 1.0):
    """Trend + seasonal + noise series with different levels / amplitudes (period 4)."""
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    out = []
    for i in range(n):
        lvl, slope, amp = 50 + 20 * i, 0.3 + 0.1 * i, 8 + 2 * i
        seas = amp * np.array([1.0, 0.2, -0.8, -0.4])[t % M]
        out.append(lvl + slope * t + seas + rng.normal(0, noise * (1 + 0.2 * i), length))
    return out


def test_metrics_match_adapter_definitions():
    rng = np.random.default_rng(1)
    for _ in range(10):
        x = np.abs(rng.normal(50, 10, 80))
        y, p = np.abs(rng.normal(50, 10, 24)), np.abs(rng.normal(50, 10, 24))
        assert fc.mase(y, p, x, 12) == pytest.approx(common.mase(y, p, x, 12))
        assert fc.smape(y, p) == pytest.approx(common.smape(y, p))
    assert fc.smape([0, 1], [0, 1]) == 0.0                                  # 0/0 term counts as 0
    x = np.array([1.0, 3, 2, 5, 4])
    assert fc.seasonal_scale(x, 1) == pytest.approx(np.mean([2, 1, 3, 1]))
    assert fc.rmsse([1, 2], [2, 4], x, 1) == pytest.approx(np.sqrt(np.mean([1, 4]) / np.mean([4, 1, 9, 1])))
    with pytest.raises(ValueError):
        fc.mase([1.0], [1.0], [2.0, 2.0, 2.0], 1)                           # zero scale


def test_snaive_matches_adapter_reference():
    x = np.random.default_rng(2).normal(size=50) + 10
    np.testing.assert_allclose(fc.repeat_season(x, 24, 12), adapter_snaive(x, 24, 12))
    assert fc.repeat_season(x[:5], 7, 12).tolist() == [x[4]] * 7                   # shorter than one season -> repeat_last


CHEAP = ("repeat_season", "seasonal_average", "repeat_season_drift", "ses", "theta", "holt_damped", "ets_damped_add", "ets_damped_mul", "stl_ets")


def test_every_cheap_method_is_finite_deterministic_and_sane():
    hist = _panel(3)
    a = fc.forecast_panel(hist, H, M, CHEAP, n_jobs=1)
    b = fc.forecast_panel(hist, H, M, CHEAP, n_jobs=1)
    for k in CHEAP:
        assert a[k].shape == (3, H) and np.all(np.isfinite(a[k]))
        np.testing.assert_array_equal(a[k], b[k])
        assert np.all(a[k] > 0) and np.all(a[k] < 3 * max(h.max() for h in hist))
    assert set(fc.METHODS) >= set(CHEAP) | {"ets", "sarima_airline", "repeat_last"}
    assert set(fc.method_help()) == set(fc.METHODS)


def test_ets_and_sarima_forecast_a_seasonal_series():
    x = _panel(1, 80)[0]
    for name in ("ets", "sarima_airline"):
        f = fc.METHODS[name](x[:-H], H, M)
        assert f.shape == (H,) and np.all(np.isfinite(f))
        assert fc.mase(x[-H:], f, x[:-H], M) < fc.mase(x[-H:], fc.repeat_season(x[:-H], H, M), x[:-H], M) * 1.5


def test_combination_of_local_models_beats_seasonal_naive_on_a_noisy_trending_panel():
    hist = _panel(8, 72, seed=3, noise=2.0)
    tr, te = [h[:-H] for h in hist], [h[-H:] for h in hist]
    base = fc.panel_mase(te, fc.forecast_panel(tr, H, M, "repeat_season")["repeat_season"], tr, M)
    y = fc.fit_predict(tr, H, M, methods=("theta", "stl_ets", "repeat_season_drift", "ets_damped_add"), how="median",
                       global_models=(), n_jobs=1)
    assert y.shape == (8, H) and np.all(y >= 0)
    assert fc.panel_mase(te, y, tr, M) < base


def test_degenerate_inputs_fall_back_instead_of_failing():
    short = np.array([5.0, 6.0, 7.0])
    const = np.full(30, 4.0)
    zeros = np.r_[np.zeros(20), np.arange(20.0)]
    for x in (short, const, zeros):
        for k in CHEAP + ("ets", "sarima_airline"):
            f = fc.METHODS[k](x, 8, M)
            assert f.shape == (8,) and np.all(np.isfinite(f)), (k, x[:3])
    with pytest.raises(ValueError):
        fc.METHODS["theta"]([1.0], 3, 1)                                    # a single observation
    with pytest.raises(ValueError):
        fc.forecast_panel([const], 3, 4, ["no_such_method"])
    y = fc.fit_predict([const], 3, 4, methods=("theta", "repeat_season"), global_models=(), n_jobs=1)
    assert np.all(y >= 0)
    with pytest.warns(RuntimeWarning, match="global models skipped"):        # implicit pool too small for a global model
        y = fc.fit_predict([const], 3, 4, methods=("theta",), n_jobs=1)
    np.testing.assert_allclose(y, fc.fit_predict([const], 3, 4, methods=("theta",), global_models=(), n_jobs=1))


def test_seasonal_detection_and_reseasonalisation():
    t = np.arange(96)
    x = 100 + 40 * np.sin(2 * np.pi * t / 12) + np.random.default_rng(0).normal(0, 1, 96)
    assert fc._is_seasonal(x, 12)
    assert not fc._is_seasonal(np.random.default_rng(1).normal(size=96), 12)
    f = fc.theta(x, 12, 12)
    assert fc.mase(100 + 40 * np.sin(2 * np.pi * (t[-1] + 1 + np.arange(12)) / 12), f, x, 12) < 1.0


def test_backtest_panel_scores_are_consistent():
    hist = _panel(4, 60, seed=5)
    bt = fc.backtest_panel(hist, H, M, ("repeat_season", "theta", "seasonal_average"), n_origins=2, n_jobs=1, global_models=())
    assert bt["ranked"][0] == min(bt["mase"], key=bt["mase"].get) and len(bt["actuals"]) == 2
    # origin 1 holds out the last H points, origin 2 the H before them
    np.testing.assert_allclose(bt["actuals"][0], np.stack([h[-H:] for h in hist]))
    np.testing.assert_allclose(bt["actuals"][1], np.stack([h[-2 * H:-H] for h in hist]))
    manual = np.mean([fc.mase(bt["actuals"][o][i], bt["forecasts"]["theta"][o][i], bt["insamples"][o][i], M)
                      for o in range(2) for i in range(4)])
    assert bt["mase"]["theta"] == pytest.approx(manual)
    assert fc.combination_mase(bt, ["theta"]) == pytest.approx(manual)
    both = fc.combination_mase(bt, ["theta", "repeat_season"], how="mean")
    assert both == pytest.approx(fc.combination_mase(bt, ["theta", "repeat_season"], weights={"theta": 1, "repeat_season": 1}))
    with pytest.raises(ValueError):
        fc.backtest_panel([np.arange(5.0)], H, M, "repeat_season", global_models=())   # too short for one origin
    # a series too short for the second origin is skipped there only
    mixed = fc.backtest_panel([hist[0], hist[1][:30]], H, M, "repeat_season", n_origins=2, n_jobs=1, global_models=())
    assert [len(i) for i in mixed["index"]] == [2, 1]


def test_combine_and_weights():
    f = {"a": np.array([[1.0, 2.0]]), "b": np.array([[3.0, 4.0]]), "c": np.array([[10.0, 0.0]])}
    np.testing.assert_allclose(fc.combine(f, "mean"), [[14 / 3, 2.0]])
    np.testing.assert_allclose(fc.combine(f, "median"), [[3.0, 2.0]])
    np.testing.assert_allclose(fc.combine(f, "trimmed"), [[3.0, 2.0]])
    np.testing.assert_allclose(fc.combine(f, weights={"a": 3, "b": 1, "c": 0}), [[1.5, 2.5]])
    w = fc.inverse_error_weights({"a": 1.0, "b": 2.0})
    assert sum(w.values()) == pytest.approx(1.0) and w["a"] == pytest.approx(2 * w["b"])
    with pytest.raises(ValueError):
        fc.combine(f, "mode")


def test_global_lgbm_learns_a_shared_seasonal_shape_and_is_deterministic():
    hist = _panel(10, 90, seed=7, noise=0.5)
    tr, te = [h[:-H] for h in hist], [h[-H:] for h in hist]
    p1 = fc.global_lgbm(tr, tr, H, M, n_estimators=40, stride=1, n_jobs=1)
    p2 = fc.global_lgbm(tr, tr, H, M, n_estimators=40, stride=1, n_jobs=1)
    assert p1.shape == (10, H) and np.all(np.isfinite(p1))
    np.testing.assert_array_equal(p1, p2)
    assert fc.panel_mase(te, p1, tr, M) < fc.panel_mase(te, fc.forecast_panel(tr, H, M, "repeat_last")["repeat_last"], tr, M)
    with pytest.raises(ValueError):
        fc.global_lgbm([np.arange(10.0)], tr, H, M)


def test_global_window_models_share_seasonal_shape_and_respect_phase():
    hist = _panel(10, 90, seed=8, noise=0.5)
    tr, te = [h[:-H] for h in hist], [h[-H:] for h in hist]
    g1 = fc.global_window(tr, tr, H, M, n_estimators=30, n_jobs=1)
    g2 = fc.global_window(tr, tr, H, M, n_estimators=30, n_jobs=1)
    assert list(g1) == list(fc.GLOBAL_MODELS)
    repeat_last = fc.panel_mase(te, fc.forecast_panel(tr, H, M, "repeat_last")["repeat_last"], tr, M)
    for k, v in g1.items():
        assert v.shape == (10, H) and np.all(np.isfinite(v)) and np.all(v >= 0)
        np.testing.assert_array_equal(v, g2[k])
        assert fc.panel_mase(te, v, tr, M) < repeat_last, k
    # the season position of the first observation changes the features (one-hot of the first forecast step's phase)
    g3 = fc.global_window(tr, tr, H, M, models="ridge", target_phase=1, n_jobs=1)
    assert not np.allclose(g3["ridge"], g1["ridge"])
    # short target series are wrap-padded; a single-value series still gets a forecast
    short = fc.global_window(tr, [np.array([5.0]), tr[0][:7]], H, M, models="ridge", n_jobs=1)["ridge"]
    assert short.shape == (2, H) and np.all(np.isfinite(short))
    with pytest.raises(ValueError):
        fc.global_window(tr, tr, H, M, models=["svm"])
    with pytest.raises(ValueError):
        fc.global_window([np.arange(10.0)], tr, H, M)


def test_fit_predict_combines_local_and_global_members():
    hist = _panel(10, 90, seed=9, noise=0.5)
    tr, te = [h[:-H] for h in hist], [h[-H:] for h in hist]
    y = fc.fit_predict(tr[:3], H, M, methods=("theta", "ets_damped_add"), train=tr, n_jobs=1)
    assert y.shape == (3, H) and np.all(y >= 0)
    only_local = fc.fit_predict(tr[:3], H, M, methods=("theta", "ets_damped_add"), global_models=(), n_jobs=1)
    assert not np.allclose(y, only_local)
    one = fc.fit_predict(tr[:3], H, M, methods=("theta",), train=tr, global_models=("ridge",), how="mean", n_jobs=1)
    g = fc.global_window(tr, tr[:3], H, M, models="ridge", n_jobs=1)["ridge"]
    th = fc.forecast_panel(tr[:3], H, M, "theta", n_jobs=1)["theta"]
    np.testing.assert_allclose(one, np.maximum((th + g) / 2, 0))
    # defaults: median of DEFAULT_METHODS and GLOBAL_MODELS members; the global pool is the histories themselves without ``train``
    d = fc.fit_predict(tr, H, M, n_jobs=1)
    mem = fc.forecast_panel(tr, H, M, fc.DEFAULT_METHODS, n_jobs=1)
    mem.update(fc.global_window(tr, tr, H, M, n_jobs=1))
    assert set(mem) == set(fc.DEFAULT_METHODS) | set(fc.GLOBAL_MODELS)
    np.testing.assert_allclose(d, np.maximum(fc.combine(mem, "median"), 0))
    # only global members
    g_only = fc.fit_predict(tr[:2], H, M, methods=(), train=tr, global_models="ridge", n_jobs=1)
    np.testing.assert_allclose(g_only, fc.global_window(tr, tr[:2], H, M, models="ridge", n_jobs=1)["ridge"])
    # an explicit train that cannot fit a global model is an error, not a silent skip
    with pytest.raises(ValueError, match="training windows"):
        fc.fit_predict(tr[:2], H, M, train=[np.arange(10.0)], n_jobs=1)


def test_fit_predict_train_cut_keeps_later_values_out_of_the_global_fit():
    hist = _panel(10, 90, seed=10, noise=0.5)
    dev = [h[:-H] for h in hist[:3]]
    a = fc.fit_predict(dev, H, M, methods=("theta",), train=hist, train_cut=H, n_jobs=1)
    b = fc.fit_predict(dev, H, M, methods=("theta",), train=[h[:-H] for h in hist], n_jobs=1)
    np.testing.assert_allclose(a, b)
    c = fc.fit_predict(dev, H, M, methods=("theta",), train=hist, train_cut=0, n_jobs=1)     # train as given: sees the tail
    assert not np.allclose(a, c)
    d = fc.fit_predict(dev, H, M, methods=("theta",), train=hist, n_jobs=1)                # "auto": train series extending a history are cut to it
    e = fc.fit_predict(dev, H, M, methods=("theta",), train=[h[:-H] for h in hist[:3]] + hist[3:], n_jobs=1)
    np.testing.assert_allclose(d, e)                       # only the three series that extend a history are cut
    with pytest.raises(ValueError, match="train_cut"):
        fc.fit_predict(dev, H, M, methods=("theta",), train=hist, train_cut="none", n_jobs=1)
    tampered = [h.copy() for h in hist]
    for h in tampered:
        h[-H:] += 1000.0                                       # values beyond the cut must not matter
    np.testing.assert_allclose(a, fc.fit_predict(dev, H, M, methods=("theta",), train=tampered, train_cut=H, n_jobs=1))


def test_train_series_extending_a_history_is_cut_to_it_by_default():
    hist = _panel(10, 90, seed=12, noise=0.5)
    dev = [h[:-H] for h in hist[:4]]                      # dev histories: prefixes of the first four train series
    same = [h.copy() for h in hist[:4]]                   # eval-like histories: equal to their train series -> nothing is cut
    cut = fc._limit_to_histories([h.copy() for h in hist], [np.asarray(d, float) for d in dev])
    assert [c.size for c in cut] == [h.size - H for h in hist[:4]] + [h.size for h in hist[4:]]
    assert [c.size for c in fc._limit_to_histories(hist, same)] == [h.size for h in hist]
    # the global members of a backtest do not see the tail of the train series either
    kw = dict(n_origins=1, n_jobs=1, max_series=4, seed=1)
    a = fc.backtest_panel(dev, H, M, ("theta",), train=hist, **kw)
    tampered = [h.copy() for h in hist]
    for h in tampered:
        h[-H:] += 1000.0
    b = fc.backtest_panel(dev, H, M, ("theta",), train=tampered, **kw)
    assert a["mase"] == b["mase"]
    c = fc.backtest_panel(dev, H, M, ("theta",), train=[h[:-H] for h in hist[:4]] + hist[4:], **kw)
    assert a["mase"] == c["mase"]


def test_backtest_panel_with_global_members():
    hist = _panel(12, 90, seed=11, noise=0.5)
    bt = fc.backtest_panel(hist, H, M, ("theta", "ets_damped_add"), n_origins=2, n_jobs=1, max_series=6, seed=3)
    assert set(bt["mase"]) == {"theta", "ets_damped_add", "ridge", "extra_trees"} and set(bt["ranked"]) == set(bt["mase"])
    assert [len(i) for i in bt["index"]] == [6, 6] and bt["index"][0] == bt["index"][1] == sorted(bt["index"][0])
    for o in range(2):
        assert {v[o].shape for v in bt["forecasts"].values()} == {(6, H)}          # all members of one origin share their rows
    again = fc.backtest_panel(hist, H, M, ("theta", "ets_damped_add"), n_origins=2, n_jobs=1, max_series=6, seed=3)
    assert again["index"] == bt["index"] and again["mase"] == bt["mase"]
    # origin 0: the global model is fitted on the pool cut by H, so it never sees the held-out values
    tampered = [h.copy() for h in hist]
    for h in tampered:
        h[-H:] += 500.0
    ridge_only = lambda hh: fc.backtest_panel(hh, H, M, (), n_origins=1, n_jobs=1, global_models="ridge", max_series=None)
    r0, r1 = ridge_only(hist), ridge_only(tampered)
    np.testing.assert_allclose(r0["forecasts"]["ridge"][0], r1["forecasts"]["ridge"][0])
    # a combination of all members = combination_mase(bt) with the default members
    assert fc.combination_mase(bt) == pytest.approx(fc.combination_mase(bt, list(bt["forecasts"])))
    with pytest.raises(ValueError, match="unknown backtest members"):
        fc.combination_mase(bt, ["nope"])
    # a separate ``train`` pool ending at the same time as the histories
    bt2 = fc.backtest_panel(hist[:3], H, M, (), n_origins=1, n_jobs=1, global_models="ridge", train=hist)
    assert bt2["index"] == [[0, 1, 2]] and bt2["forecasts"]["ridge"][0].shape == (3, H)


def test_shape_errors_name_the_offending_arguments():
    with pytest.raises(ValueError, match="combine: members must have one common shape"):
        fc.combine({"a": np.zeros((3, 4)), "b": np.zeros((2, 4))})
    with pytest.raises(ValueError, match="y_true has 4 values, y_pred has 3"):
        fc.mase(np.zeros(4), np.zeros(3), np.arange(20.0), 1)
    with pytest.raises(ValueError, match="2 y_true rows, 3 y_pred rows, 3 insample"):
        fc.panel_mase(np.zeros((2, 4)), np.zeros((3, 4)), [np.arange(20.0)] * 3, 1)
    tr = _panel(6, 60)
    with pytest.raises(ValueError, match=r"targets\[1\] is empty"):
        fc.global_window(tr, [tr[0], np.array([])], H, M)
    with pytest.raises(ValueError, match=r"histories\[0\] has 1 non-finite"):
        fc.fit_predict([np.array([1.0, np.nan, 3.0])], 3, 1)


def test_describe_lists_every_public_name():
    text = scilib.describe("forecast")
    assert text.startswith("Library `scilib.forecast`")
    for name in fc.__all__:
        if name in ("seasonal_average_k", "repeat_last"):
            continue
        assert name in text, name
    for name in fc.DEFAULT_METHODS:
        assert name in text


def test_code_node_may_import_scilib():
    assert scan_code("from scilib.forecast import fit_predict\n\ndef run(inputs, config):\n    return {}\n") == []
