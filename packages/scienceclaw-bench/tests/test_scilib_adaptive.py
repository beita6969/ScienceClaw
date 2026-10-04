"""scilib.adaptive: item-curve fit, ability posterior, deterministic query selection, prediction, interface text (synthetic data)."""
from __future__ import annotations

import numpy as np
import pytest

import scilib
from scilib import adaptive as ad
from scienceclaw.runtime.integrity import scan_code

Q = 40


def _world(n_students: int, seed: int, answered: float = 0.5):
    """Synthetic 2PL world: returns (full 0/1 matrix, observed -1/0/1 matrix, true a, true b)."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.6, 2.2, Q)
    b = rng.normal(0.0, 1.0, Q)
    theta = rng.normal(size=n_students)
    p = 1 / (1 + np.exp(-a[None, :] * (theta[:, None] - b[None, :])))
    full = (rng.random((n_students, Q)) < p).astype(np.int64)
    seen = rng.random((n_students, Q)) < answered
    return full, np.where(seen, full, -1), a, b


@pytest.fixture(scope="module")
def world():
    full, obs, a, b = _world(700, 0)
    return obs, a, b, ad.fit_item_curves(obs)


def test_fit_recovers_difficulty_and_is_deterministic(world):
    obs, a, b, m = world
    assert m["curves"].shape == (Q, 81) and m["grid"].shape == (81,) and m["prior"].sum() == pytest.approx(1.0)
    assert np.corrcoef(m["difficulty"], b)[0, 1] > 0.95
    assert np.corrcoef(m["discrimination"], a)[0, 1] > 0.7
    assert m["loglik"][-1] > m["loglik"][0]
    # curves are increasing in ability for positive discrimination
    assert np.all(np.diff(m["curves"], axis=1) >= -1e-12)
    m2 = ad.fit_item_curves(obs)
    assert np.array_equal(m["curves"], m2["curves"])
    assert np.array_equal(m["majority"], (((obs == 1).sum(0)) > ((obs == 0).sum(0))).astype(int))


def test_fit_handles_unanswered_questions_and_rejects_bad_values():
    obs = _world(120, 1)[1].copy()
    obs[:, 3] = -1
    m = ad.fit_item_curves(obs, iterations=5)
    assert m["n_answers"][3] == 0 and m["discrimination"][3] == 1.0 and m["difficulty"][3] == 0.0
    assert np.all(np.isfinite(m["curves"]))
    with pytest.raises(ValueError):
        ad.fit_item_curves(np.full((4, 4), 2))
    with pytest.raises(ValueError):
        ad.fit_item_curves(np.zeros(5))


def test_merge_revealed_conventions():
    r1 = np.full((2, 3), -1); r1[0, 0] = 1; r1[1, 2] = 0
    r2 = np.full((2, 3), -1); r2[0, 1] = 0
    out = ad.merge_revealed([r1, r2])
    assert out.tolist() == [[1, 0, -1], [-1, -1, 0]]
    assert ad.merge_revealed(r1).tolist() == r1.tolist()
    assert (ad.merge_revealed(None, (2, 3)) == -1).all() and (ad.merge_revealed([], (2, 3)) == -1).all()
    with pytest.raises(ValueError):
        ad.merge_revealed(np.array([[1, 2, 3, 4]]))          # option ids (revealed_values) are not accepted
    with pytest.raises(ValueError):
        ad.merge_revealed([r1, np.full((3, 3), -1)])


def test_posterior_is_prior_without_reveals_and_moves_with_answers(world):
    obs, a, b, m = world
    rev = np.full((3, Q), -1)
    w = ad.ability_posterior(m, rev)
    assert w.shape == (3, 81) and np.allclose(w.sum(axis=1), 1) and np.allclose(w, m["prior"][None, :])
    hard = np.argsort(m["difficulty"])[-8:]
    easy = np.argsort(m["difficulty"])[:8]
    rev[0, hard] = 1                                          # right on hard questions -> high ability
    rev[1, easy] = 0                                          # wrong on easy questions -> low ability
    w = ad.ability_posterior(m, rev)
    mean = w @ m["grid"]
    assert mean[0] > 0.5 and mean[1] < -0.5 and abs(mean[2]) < 1e-9
    p = ad.predict_proba(m, rev)
    assert p.shape == (3, Q) and p[0].mean() > p[2].mean() > p[1].mean()


@pytest.mark.parametrize("method", ["bald", "batch"])
def test_select_queries_respects_masks_budget_and_is_deterministic(world, method):
    obs, a, b, m = world
    rng = np.random.default_rng(3)
    n = 12
    can = rng.random((n, Q)) < 0.7
    rev = np.full((n, Q), -1)
    for i in range(n):                                        # 3 earlier reveals per student, inside can_query
        for q in np.flatnonzero(can[i])[:3]:
            rev[i, q] = rng.integers(0, 2)
    rev[0] = -1
    s = ad.select_queries(m, can, [rev], k=8, budget=10, method=method)
    assert s.shape == (n, 8) and s.dtype.kind == "i"
    for i in range(n):
        picked = s[i][s[i] >= 0]
        assert len(picked) == len(set(picked.tolist())) == min(8, 10 - (rev[i] >= 0).sum(), can[i].sum() - (rev[i] >= 0).sum())
        assert can[i, picked].all() and (rev[i, picked] < 0).all()
        assert (s[i][len(picked):] == -1).all()               # -1 padding at the end
    assert np.array_equal(s, ad.select_queries(m, can, [rev], k=8, budget=10, method=method))
    # prefix property: the list for a smaller k is the head of the list for a larger k
    s3 = ad.select_queries(m, can, [rev], k=3, budget=10, method=method)
    assert np.array_equal(s3, s[:, :3])
    # a student that has used the whole budget gets nothing
    rev[1, np.flatnonzero(can[1])[:10]] = 1
    assert (ad.select_queries(m, can, [rev], k=5, budget=10, method=method)[1] == -1).all()


def test_select_queries_accepts_empty_and_none_and_validates(world):
    obs, a, b, m = world
    can = np.ones((2, Q), bool)
    s0 = ad.select_queries(m, can, None, 4)
    assert np.array_equal(s0, ad.select_queries(m, can, [], 4)) and np.array_equal(s0, ad.select_queries(m, can, np.full((2, Q), -1), 4))
    assert np.array_equal(s0[0], s0[1])                       # same posterior -> same picks
    with pytest.raises(ValueError):
        ad.select_queries(m, np.ones((2, Q + 1), bool), None, 4)
    with pytest.raises(ValueError):
        ad.select_queries(m, can, None, 4, method="random")
    with pytest.raises(ValueError):
        ad.select_queries(m, can, None, 0)


def test_batch_selection_avoids_duplicate_questions():
    """Two copies of one very informative question: the joint criterion takes one of them and moves on, the marginal one takes both."""
    full, obs, _, _ = _world(500, 2)
    obs = np.concatenate([obs, obs[:, :1], obs[:, :1]], axis=1)          # questions Q, Q+1 duplicate question 0
    m = ad.fit_item_curves(obs, iterations=10)
    can = np.zeros((1, Q + 2), bool)
    can[0, [0, Q, Q + 1, 5, 6, 7]] = True
    bald = ad.select_queries(m, can, None, 3, method="bald")[0].tolist()
    batch = ad.select_queries(m, can, None, 3, method="batch")[0].tolist()
    dup = {0, Q, Q + 1}
    assert len(dup & set(batch)) <= len(dup & set(bald))


def test_adaptive_reveals_beat_the_question_majority(world):
    """Held-out students: 10 selected reveals per student lift accuracy on the other cells above the per-question majority."""
    obs, a, b, m = world
    rng = np.random.default_rng(11)
    theta = rng.normal(size=200)
    p = 1 / (1 + np.exp(-a[None, :] * (theta[:, None] - b[None, :])))
    truth = (rng.random((200, Q)) < p).astype(np.int64)
    can = np.ones((200, Q), bool)
    rev_all = np.full((200, Q), -1)
    sel = ad.select_queries(m, can, None, 10)
    r, c = np.nonzero(sel >= 0)
    rev_all[r, sel[r, c]] = truth[r, sel[r, c]]
    targets = rev_all < 0
    y = ad.predict(m, rev_all, targets)
    assert y.shape == truth.shape and set(np.unique(y)) <= {-1, 0, 1} and (y[~targets] == -1).all()
    acc = (y[targets] == truth[targets]).mean()
    base = (m["majority"][None, :].repeat(200, 0)[targets] == truth[targets]).mean()
    assert acc > base + 0.02


def test_model_may_be_the_training_matrix_and_is_cached(world):
    obs, a, b, m = world
    can = np.ones((3, Q), bool)
    s_dict = ad.select_queries(m, can, None, 5)
    s_raw = ad.select_queries(obs, can, None, 5)
    assert np.array_equal(s_dict, s_raw)
    n_cached = len(ad._CACHE)
    ad.predict(obs, None, can)
    assert len(ad._CACHE) == n_cached
    with pytest.raises(ValueError):
        ad.select_queries({"x": 1}, can, None, 5)


def test_describe_lists_every_public_name_and_conventions():
    text = scilib.describe("adaptive")
    for name in ad.__all__:
        assert name in text
    for fact in ("-1 = NOT revealed", "revealed_values", "that call's cells only", "deterministic"):
        assert fact in text
    low = text.lower()
    for word in ("reference", "baseline", "accept", "best recipe", "should use"):
        assert word not in low


def test_code_node_may_import_scilib():
    assert scan_code("from scilib.adaptive import select_queries\n\ndef run(inputs, config):\n    return {}\n") == []
