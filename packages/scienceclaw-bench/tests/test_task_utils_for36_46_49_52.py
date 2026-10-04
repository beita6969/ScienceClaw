"""Shared helpers of the FoR36 / FoR46 / FoR49 / FoR52 adapters."""
from __future__ import annotations

import numpy as np
import pytest

from scienceclaw.bench.tasks import _adapter_utils_for36_46_49_52 as u


def test_partition_is_disjoint_complete_and_seed_independent():
    ids = [f"x{i}" for i in range(103)]
    p = u.partition_ids(ids, {"a": 0.5, "b": 0.2, "c": 0.3}, "salt")
    assert sorted(sum(p.values(), [])) == sorted(ids)
    assert {k: len(v) for k, v in p.items()} == {"a": 51, "b": 21, "c": 31}    # largest remainder
    assert p == u.partition_ids(list(reversed(ids)), {"a": 0.5, "b": 0.2, "c": 0.3}, "salt")


def test_draw_blocks_disjoint_prefix_stable_and_cycling():
    pool = [f"i{k}" for k in range(40)]
    a = u.draw_blocks({"s": pool}, {"s": 8}, 5, np.random.default_rng(1), "t")
    b = u.draw_blocks({"s": pool}, {"s": 8}, 3, np.random.default_rng(1), "t")
    assert a[:3] == b and len({i for e in a for i in e}) == 40
    with pytest.raises(u.PoolExhausted):
        u.draw_blocks({"s": pool}, {"s": 8}, 6, np.random.default_rng(1), "t")
    c = u.draw_blocks({"s": pool}, {"s": 8}, 12, np.random.default_rng(1), "t", cycle=True)
    assert all(len(set(e)) == 8 for e in c)
    assert len({i for e in c[:5] for i in e}) == 40                  # first cycle covers the pool once
    strat = u.draw_blocks({"p": pool[:10], "n": pool[10:]}, {"p": 2, "n": 6}, 4, np.random.default_rng(0), "t")
    assert all(sum(i in pool[:10] for i in e) == 2 for e in strat)


def test_norm_scores():
    assert u.norm_score(0.8, 0.4, "max") == pytest.approx(2.0)
    assert u.norm_score(0.8, 0.0, "max", floor=0.1) == pytest.approx(8.0)
    assert u.norm_score(0.8, 0.0, "max") == 10.0
    assert u.norm_score(2.0, 1.0, "min") == pytest.approx(0.5)
    assert u.norm_score(None, 1.0, "max") == 0.0
    assert u.norm_score_db(float("inf"), 0.0) == 10.0 and u.norm_score_db(float("nan"), 0.0) == 0.0


def test_label_constraints():
    ok, _ = u.c_str_list(2, "x").check(["a", "b"], None)
    assert ok
    assert not u.c_str_list(2, "x").check(["a", 1], None)[0]
    assert not u.c_str_list(2, "x").check("ab", None)[0]
    assert u.as_str_list(np.array(["a", "b"]), 2)[0] == ["a", "b"]
    c = u.c_allowed_labels(2, [["A", "B"], ["C"]])
    assert c.check(["B", "C"], None)[0] and not c.check(["C", "C"], None)[0]
    g = u.c_allowed_labels(2, ["sat", "unsat"])
    assert g.check(["sat", "unsat"], None)[0] and not g.check(["sat", "maybe"], None)[0]
