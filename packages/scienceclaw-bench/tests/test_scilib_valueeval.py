"""scilib.valueeval: metric parity with the adapter, interface text, sandbox import, end-to-end on synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scilib
from scilib import valueeval as ve
from scienceclaw.bench.tasks import for50_valueeval as m
from scienceclaw.runtime.integrity import scan_code


def test_official_f1_matches_adapter_metric():
    rng = np.random.default_rng(3)
    for _ in range(20):
        t = (rng.random((16, 20)) < 0.15).astype(int)
        p = (rng.random((16, 20)) < 0.4).astype(int)
        a, b = m.official_f1(t, p), ve.official_f1_report(t, p)
        assert a["f1"] == pytest.approx(b["f1"], nan_ok=True)
        assert a["mean_category_f1"] == pytest.approx(b["mean_category_f1"], nan_ok=True)
        assert a["n_categories"] == b["n_categories"]
    assert ve.official_f1(np.zeros((4, 3), int), np.ones((4, 3), int)) != ve.official_f1(np.zeros((4, 3), int), np.ones((4, 3), int))  # NaN


def test_describe_lists_every_public_name():
    text = scilib.describe("valueeval")
    for name in ve.__all__:
        assert name in text


def test_code_node_may_import_scilib():
    assert scan_code("from scilib.valueeval import fit_predict\n\ndef run(inputs, config):\n    return {}\n") == []


def _synthetic(n: int, seed: int):
    rng = np.random.default_rng(seed)
    words = [["cats", "dogs", "tax", "war"], ["school", "money", "trees", "rivers"]]
    Y = (rng.random((n, 4)) < 0.3).astype(int)
    prem = [" ".join(rng.choice(words[0] + words[1], 3).tolist() + [words[0][c] for c in np.flatnonzero(y[:2])] +
                     [words[1][c] for c in np.flatnonzero(y[2:])]) for y in Y]
    df = pd.DataFrame({"conclusion": [f"c{i % 25}" for i in range(n)], "stance": ["in favor of", "against"] * (n // 2),
                       "premise": prem})
    return df, Y


def test_fit_predict_learns_a_planted_signal_and_is_deterministic():
    tr, Y = _synthetic(400, 0)
    te, Yt = _synthetic(100, 1)
    for decision in ("expected_f1", "threshold"):
        r1 = ve.fit_predict(tr, Y, [te], n_folds=3, k=16, n_jobs=1, decision=decision)
        r2 = ve.fit_predict(tr, Y, [te], n_folds=3, k=16, n_jobs=1, decision=decision)
        (p1,), (p2,) = r1, r2
        assert p1.shape == (100, 4) and set(np.unique(p1)) <= {0, 1}
        assert np.array_equal(p1, p2) and r1.oof_f1 == r2.oof_f1
        assert r1.oof_f1 > 0.5 and ve.official_f1(Yt, p1) > ve.official_f1(Yt, np.ones_like(Yt)) + 0.1
        assert r1.scores[0].shape == r1.probs[0].shape == (100, 4)
    assert ve.fit_predict(tr, Y, [te], n_folds=3, n_jobs=1).threshold is None
    assert ve.fit_predict(tr, Y, [te], n_folds=3, n_jobs=1, decision="threshold").counts is None
    with pytest.raises(ValueError):
        ve.fit_predict(tr, Y, [te], decision="nope")


def test_fit_predict_unpacks_one_matrix_per_target_and_accepts_groups():
    tr, Y = _synthetic(300, 4)
    t1, _ = _synthetic(16, 5)
    t2, _ = _synthetic(40, 6)
    a, b = ve.fit_predict(tr, Y, [t1, t2], n_folds=3, n_jobs=1, groups=np.arange(len(tr)) % 30)
    assert a.shape == (16, 4) and b.shape == (40, 4)
    res = ve.fit_predict(tr, Y, [t1, t2], n_folds=3, n_jobs=1)
    assert len(res) == 2 and len(res.counts) == 2 and res.counts[0].shape == (4,) and res.counts[0].max() <= 16


def test_expected_f1_decision_is_exact_for_certain_labels_and_deterministic():
    rng = np.random.default_rng(0)
    G = (rng.random((16, 6)) < 0.3).astype(int)
    G[0] = 1                                           # every label has a positive
    out, m = ve.expected_f1_decision(G.astype(float), return_counts=True)
    assert np.array_equal(out, G) and np.array_equal(m, G.sum(0))
    P = rng.random((16, 6))
    assert np.array_equal(ve.expected_f1_decision(P, seed=3), ve.expected_f1_decision(P, seed=3))
    big = ve.expected_f1_decision(rng.random((200, 20)))            # large n: draws shrink, still valid
    assert big.shape == (200, 20) and set(np.unique(big)) <= {0, 1}


def test_expected_f1_decision_beats_a_global_cutoff_on_its_own_model():
    rng = np.random.default_rng(1)
    L, n = 12, 16
    prev = rng.uniform(0.03, 0.4, L)
    skill = np.r_[np.full(L // 2, 3.0), np.zeros(L - L // 2)]   # informative labels first, pure-noise labels after
    def draw():
        Y = (rng.random((n, L)) < prev).astype(int)
        z = np.log(prev / (1 - prev)) + skill * (Y - prev) * 2 + rng.normal(0, 0.7, (n, L))
        return Y, 1 / (1 + np.exp(-z))
    dec, cut = [], []
    for _ in range(60):
        Y, P = draw()
        dec.append(ve.official_f1(Y, ve.expected_f1_decision(P)))
        cut.append(ve.official_f1(Y, (P >= 0.3).astype(int)))
    assert np.nanmean(dec) > np.nanmean(cut) + 0.02


def test_column_calibration_recovers_prevalence_and_handles_constant_labels():
    rng = np.random.default_rng(2)
    n = 4000
    Y = np.column_stack([(rng.random(n) < 0.1).astype(int), (rng.random(n) < 0.5).astype(int), np.zeros(n, int)])
    S = np.clip(0.5 + 0.2 * (Y - 0.5) + rng.normal(0, 0.05, Y.shape), 0.01, 0.99)     # uncalibrated scores
    prm = ve.column_calibration(S, Y)
    P = ve.calibrate(S, prm)
    assert prm.shape == (3, 2)
    assert abs(P[:, 0].mean() - 0.1) < 0.02 and abs(P[:, 1].mean() - 0.5) < 0.02
    assert P[:, 2].max() < 0.01                                                        # constant label: prevalence map
    assert np.all(np.diff(P[np.argsort(S[:, 1]), 1]) >= -1e-12)                         # monotone in the score


def test_episode_f1_is_in_unit_interval_and_deterministic():
    rng = np.random.default_rng(3)
    Y = (rng.random((120, 5)) < 0.3).astype(int)
    P = np.clip(0.3 + 0.4 * (Y - 0.3) + rng.normal(0, 0.15, Y.shape), 0.01, 0.99)
    a = ve.episode_f1(Y, P, k=16, n_sub=20)
    assert 0 < a <= 1 and a == ve.episode_f1(Y, P, k=16, n_sub=20)


def test_grouped_oof_never_splits_a_group():
    tr, Y = _synthetic(120, 2)
    X = ve.TextFeatures(min_df_word=1, min_df_char=1).fit_transform(tr)
    from sklearn.model_selection import GroupKFold
    for a, b in GroupKFold(4).split(X, groups=tr["conclusion"].values):
        assert not set(tr["conclusion"].values[a]) & set(tr["conclusion"].values[b])
    assert ve.grouped_oof_scores(X, Y, tr["conclusion"].values, n_folds=4, n_jobs=1).shape == Y.shape
