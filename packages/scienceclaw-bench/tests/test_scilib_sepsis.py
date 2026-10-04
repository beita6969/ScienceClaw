"""scilib.sepsis: utility parity with the adapter, causality of the features, folds, determinism, end-to-end on synthetic stays."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scilib
from scilib import sepsis as sp
from scienceclaw.bench.tasks import for42_sepsis as m
from scienceclaw.runtime.integrity import scan_code

FAST = {"n_estimators": 40}


def _labels(n: int, onset: int | None) -> np.ndarray:
    y = np.zeros(n, dtype=int)
    if onset is not None:
        y[onset:] = 1
    return y


def _stays(n_stays: int, seed: int, n_septic: int) -> pd.DataFrame:
    """Synthetic long table: septic stays get rising HR/Resp/Temp before the first label hour."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_stays):
        n = int(rng.integers(24, 44))
        septic = i < n_septic
        onset = int(rng.integers(12, n - 8)) if septic else None
        d = {"patient_id": np.full(n, f"p{seed}_{i}"), "hour": np.arange(n)}
        drift = np.zeros(n)
        if septic:
            drift = np.clip(np.arange(n) - (onset - 8), 0, None) / 6.0
        for c in sp.FEATURES:
            d[c] = np.full(n, np.nan)
        d["HR"] = 80 + 4 * drift + rng.normal(0, 3, n)
        d["Resp"] = 16 + 1.5 * drift + rng.normal(0, 1.5, n)
        d["Temp"] = 36.8 + 0.25 * drift + rng.normal(0, 0.2, n)
        d["SBP"] = 120 - 5 * drift + rng.normal(0, 6, n)
        d["MAP"] = 85 - 3 * drift + rng.normal(0, 4, n)
        d["O2Sat"] = 97 - 0.5 * drift + rng.normal(0, 1, n)
        d["Lactate"] = np.where(rng.random(n) < 0.15, 1.2 + 0.4 * drift + rng.normal(0, 0.2, n), np.nan)
        d["Age"] = np.full(n, float(rng.integers(30, 85)))
        d["Gender"] = np.full(n, float(rng.integers(0, 2)))
        d["HospAdmTime"] = np.full(n, -float(rng.integers(0, 40)))
        d["ICULOS"] = np.arange(1, n + 1, dtype=float)
        d["SepsisLabel"] = _labels(n, onset)
        frames.append(pd.DataFrame(d))
    df = pd.concat(frames, ignore_index=True)
    miss = rng.random((len(df), 3)) < 0.1                      # sporadic missing vitals
    for j, c in enumerate(("HR", "Resp", "Temp")):
        df.loc[miss[:, j], c] = np.nan
    return df


@pytest.fixture(scope="module")
def train() -> pd.DataFrame:
    return _stays(60, 1, 12)


@pytest.fixture(scope="module")
def test_table() -> pd.DataFrame:
    return _stays(20, 2, 6)


@pytest.fixture(scope="module")
def fitted(train) -> sp.SepsisModel:
    return sp.SepsisModel(n_folds=3, params=FAST).fit(train)


# ---------------------------------------------------------------------------------------------------- interface
def test_describe_lists_every_public_name():
    text = scilib.describe("sepsis")
    for name in sp.__all__:
        assert name in text


def test_code_node_may_import_scilib():
    assert scan_code("from scilib.sepsis import fit_predict\n\ndef run(inputs, config):\n    return {}\n") == []


# ---------------------------------------------------------------------------------------------------- utility
def test_utility_matches_the_official_loop_and_the_adapter():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 70))
        y = _labels(n, int(rng.integers(0, n)) if rng.random() < 0.5 else None)
        p = (rng.random(n) < rng.random()).astype(int)
        assert sp.utility_of_stay(y, p) == pytest.approx(m.compute_prediction_utility_official(y, p))
        assert sp.utility_of_stay(y, p) == pytest.approx(m.prediction_utility(y, p))
        assert sp.utility_of_stay(y, sp._best_predictions(y)) == pytest.approx(
            m.prediction_utility(y, m.best_predictions(y)))


def test_normalized_utility_matches_the_adapter_aggregation():
    rng = np.random.default_rng(1)
    ys = [_labels(int(n), int(rng.integers(0, n)) if i % 3 == 0 else None) for i, n in enumerate(rng.integers(15, 50, 9))]
    ps = [(rng.random(len(y)) < 0.4).astype(int) for y in ys]
    tri = [m.utility_triplet(y, p) for y, p in zip(ys, ps)]
    expect = m.normalized_utility(sum(t[0] for t in tri), sum(t[1] for t in tri), sum(t[2] for t in tri))
    assert sp.normalized_utility(ys, ps) == pytest.approx(expect)
    assert sp.normalized_utility(ys, [sp._best_predictions(y) for y in ys]) == pytest.approx(1.0)
    assert sp.normalized_utility(ys, [np.zeros(len(y), int) for y in ys]) == pytest.approx(0.0)
    assert np.isnan(sp.normalized_utility([np.zeros(5, int)], [np.ones(5, int)]))      # U_best == U_inaction
    with pytest.raises(ValueError):
        sp.normalized_utility(ys, ps[:-1])
    with pytest.raises(ValueError):
        sp.utility_of_stay(np.zeros(5), np.zeros(4))


def test_row_gains_make_the_utility_linear_in_the_alarms():
    rng = np.random.default_rng(2)
    for onset in (None, 3, 20, 35):
        y = _labels(40, onset)
        g = sp.row_gains(y)
        assert g.shape == (40,)
        for _ in range(20):
            p = (rng.random(40) < 0.3).astype(int)
            assert sp.utility_of_stay(y, p) == pytest.approx(sp.utility_of_stay(y, np.zeros(40, int)) + float(np.sum(p * g)))
    assert np.all(sp.row_gains(_labels(10, None)) == pytest.approx(-0.05))


def test_stay_targets(train):
    lab, win, gain = (sp.stay_targets(train, k) for k in ("label", "window", "gain"))
    assert np.array_equal(lab, train["SepsisLabel"].to_numpy())
    for pid, g in train.groupby("patient_id", sort=False):
        y = g["SepsisLabel"].to_numpy()
        rows = g.index.to_numpy()
        assert np.allclose(gain[rows], sp.row_gains(y))
        if y.any():
            ts = int(np.argmax(y)) + 6
            expect = np.zeros(len(y), int)
            expect[max(0, ts - 12): ts + 4] = 1
            assert np.array_equal(win[rows], expect)
        else:
            assert win[rows].sum() == 0
    with pytest.raises(ValueError):
        sp.stay_targets(train, "nope")


# ---------------------------------------------------------------------------------------------------- features
def test_features_are_causal(test_table):
    feats = sp.build_features(test_table)
    assert len(feats) == len(test_table) and list(feats.index) == list(test_table.index)
    assert not any(c.endswith("_cnt") for c in feats.columns)
    # cutting every stay after hour 15, or scrambling everything after it, leaves the earlier rows unchanged
    cut = test_table[test_table["hour"] <= 15]
    f_cut = sp.build_features(cut)
    assert np.allclose(feats.loc[cut.index].to_numpy(dtype=float), f_cut.to_numpy(dtype=float), equal_nan=True)
    scr = test_table.copy()
    late = scr["hour"] > 15
    rng = np.random.default_rng(0)
    for c in ("HR", "Resp", "Temp", "SBP", "MAP", "Lactate"):
        scr.loc[late, c] = rng.normal(50, 30, int(late.sum()))
    f_scr = sp.build_features(scr)
    assert np.allclose(feats.loc[~late].to_numpy(dtype=float), f_scr.loc[~late].to_numpy(dtype=float), equal_nan=True)
    assert not np.allclose(feats.loc[late, "HR"].to_numpy(), f_scr.loc[late, "HR"].to_numpy())


def test_features_locf_and_since():
    df = pd.DataFrame({"patient_id": ["a"] * 5 + ["b"] * 3, "hour": [0, 1, 2, 3, 4, 0, 1, 2]})
    for c in sp.FEATURES:
        df[c] = np.nan
    df["ICULOS"] = [1, 2, 3, 4, 5, 1, 2, 3]
    df["HR"] = [np.nan, 90, np.nan, np.nan, 100, 60, np.nan, 61]
    f = sp.build_features(df)
    assert np.isnan(f["HR"].iloc[0]) and f["HR"].iloc[1:5].tolist() == [90, 90, 90, 100]
    assert f["HR"].iloc[5:].tolist() == [60, 60, 61]                 # nothing carried across stays
    assert f["HR_since"].iloc[[1, 2, 3, 4]].tolist() == [0, 1, 2, 0]
    with pytest.raises(ValueError):
        sp.build_features(df.drop(columns=["HR"]))


def test_measurement_columns_are_optional(test_table):
    with_m = sp.build_features(test_table)
    without = sp.build_features(test_table, measurement=False)
    dropped = set(with_m.columns) - set(without.columns)
    assert dropped and all(c.endswith("_since") or c.startswith("n_meas") for c in dropped)
    assert set(without.columns) <= set(with_m.columns)
    assert np.allclose(with_m[list(without.columns)].to_numpy(dtype=float), without.to_numpy(dtype=float), equal_nan=True)


def test_smooth_and_hold_are_trailing_and_per_stay():
    ids = np.array(["a"] * 4 + ["b"] * 3)
    s = np.array([1.0, 3.0, 5.0, 7.0, 10.0, 0.0, 20.0])
    out = sp.causal_smooth(s, ids, 2)
    assert out.tolist() == [1.0, 2.0, 4.0, 6.0, 10.0, 5.0, 10.0]
    assert sp.causal_smooth(s, ids, 1).tolist() == s.tolist()
    p = np.array([0, 1, 0, 0, 0, 0, 1])
    assert sp.hold_alarms(p, ids, 1).tolist() == [0, 1, 1, 0, 0, 0, 1]
    assert sp.hold_alarms(p, ids, 0).tolist() == p.tolist()
    assert sp.hold_alarms(np.array([0, 0, 0, 1, 0, 0, 0]), ids, 5).tolist() == [0, 0, 0, 1, 0, 0, 0]  # not across stays


def test_stay_folds_keep_stays_whole_and_spread_septic_stays():
    ids = [f"p{i}" for i in range(50)]
    septic = [i < 10 for i in range(50)]
    fold = sp.stay_folds(ids, septic, 5, seed=3)
    assert set(fold) == set(ids) and sorted(set(fold.values())) == [0, 1, 2, 3, 4]
    for k in range(5):
        assert sum(1 for i in range(10) if fold[ids[i]] == k) == 2
    assert fold == sp.stay_folds(ids, septic, 5, seed=3)
    with pytest.raises(ValueError):
        sp.stay_folds(ids, septic[:-1], 5)


# ---------------------------------------------------------------------------------------------------- model
def test_model_learns_the_synthetic_signal(fitted, test_table):
    assert fitted.oof_utility_ > 0.2
    pred = fitted.predict(test_table)
    y = [g["SepsisLabel"].to_numpy() for _, g in test_table.groupby("patient_id", sort=False)]
    assert [len(a) for a in pred] == [len(a) for a in y]
    assert all(set(np.unique(a)) <= {0, 1} for a in pred) and all(a.dtype.kind == "i" for a in pred)
    nu = sp.normalized_utility(y, pred)
    assert nu > 0.2
    assert nu > sp.normalized_utility(y, [np.ones(len(a), int) for a in y])


def test_predictions_are_causal_and_deterministic(train, fitted, test_table):
    full = fitted.score(test_table)
    cut = test_table[test_table["hour"] <= 17]
    assert np.allclose(fitted.score(cut), full[cut.index.to_numpy()])
    again = sp.SepsisModel(n_folds=3, params=FAST).fit(train)
    assert again.threshold_ == fitted.threshold_
    assert np.array_equal(again.oof_scores_, fitted.oof_scores_)
    assert all(np.array_equal(a, b) for a, b in zip(again.predict(test_table), fitted.predict(test_table)))
    # smoothing and holding stay causal
    sm = sp.SepsisModel(n_folds=3, smooth=3, hold=2, params=FAST).fit(train)
    p_full = sm.predict(test_table)
    p_cut = sm.predict(cut)
    for a, b in zip(p_full, p_cut):
        assert np.array_equal(a[:len(b)], b)


def test_fit_predict_entry_point(train, test_table):
    dev = test_table[test_table["patient_id"].isin(pd.unique(test_table["patient_id"])[:8])]
    (p_dev, p_test), info = sp.fit_predict(train, [dev, test_table], n_folds=3, seed=1)
    assert len(p_dev) == 8 and len(p_test) == 20
    assert set(info) == {"threshold", "oof_utility", "oof_alarm_rate", "n_positive_hours"}
    assert 0.0 <= info["oof_alarm_rate"] <= 1.0
    assert info["n_positive_hours"] == [int(sum(a.sum() for a in p_dev)), int(sum(a.sum() for a in p_test))]
    with pytest.raises(ValueError):
        sp.fit_predict(train, test_table)                    # a single table instead of a list


def test_predict_follows_the_requested_patient_order(fitted, test_table):
    ids = list(pd.unique(test_table["patient_id"]))
    order = ids[::-1][:5]
    got = fitted.predict(test_table, patient_ids=order)
    base = fitted.predict(test_table)
    for pid, a in zip(order, got):
        assert np.array_equal(a, base[ids.index(pid)])
    with pytest.raises(ValueError):
        fitted.predict(test_table, patient_ids=["missing"])


def test_prevalence_moves_the_cutoff_down(train):
    low = sp.SepsisModel(n_folds=3, prevalence=None, params=FAST).fit(train)
    high = sp.SepsisModel(n_folds=3, prevalence=0.5, params=FAST).fit(train)
    assert np.array_equal(low.oof_scores_, high.oof_scores_)
    assert high.threshold_ <= low.threshold_


def test_defaults(fitted):
    m = sp.SepsisModel()
    assert (m.smooth, m.hold, m.prevalence, m.measurement, m.linear_weight) == (3, 0, 0.073, False, 0.25)
    assert not any(c.endswith("_since") or c.startswith("n_meas") for c in fitted.columns_)
    assert 0.0 < fitted.oof_alarm_rate_ < 1.0


def test_linear_component_changes_the_scores_and_stays_deterministic(train, fitted, test_table):
    gbm_only = sp.SepsisModel(n_folds=3, linear_weight=0.0, params=FAST).fit(train)
    assert not np.array_equal(gbm_only.oof_scores_, fitted.oof_scores_)
    assert np.all((fitted.oof_scores_ >= 0) & (fitted.oof_scores_ <= 1))
    s1 = fitted.score(test_table)
    s2 = sp.SepsisModel(n_folds=3, params=FAST).fit(train).score(test_table)
    assert np.array_equal(s1, s2)
    cut = test_table[test_table["hour"] <= 17]
    assert np.allclose(fitted.score(cut), s1[cut.index.to_numpy()])
    y = [g["SepsisLabel"].to_numpy() for _, g in test_table.groupby("patient_id", sort=False)]
    assert sp.normalized_utility(y, gbm_only.predict(test_table)) > 0.2


def test_informative_errors(train, test_table):
    with pytest.raises(RuntimeError):
        sp.SepsisModel().predict(test_table)
    with pytest.raises(ValueError):
        sp.SepsisModel(targets=("nope",))
    with pytest.raises(ValueError):
        sp.SepsisModel(prevalence=1.5)
    with pytest.raises(ValueError):
        sp.SepsisModel(linear_weight=1.0)
    with pytest.raises(ValueError):
        sp.SepsisModel(n_folds=3, params=FAST).fit(train.drop(columns=["SepsisLabel"]))
    no_sepsis = train.assign(SepsisLabel=0)
    with pytest.raises(ValueError):
        sp.SepsisModel(n_folds=3, params=FAST).fit(no_sepsis)
