"""Shared helpers of the FoR43/45/47/48/50 adapters (pure functions, no data needed)."""
from __future__ import annotations

import numpy as np
import pytest

from scienceclaw.bench.tasks import _adapter_utils_for43_45_47_48_50 as u


def test_partition_groups_is_group_disjoint_deterministic_and_proportional():
    items = {f"i{j}": f"g{j // 3}" for j in range(300)}           # 100 groups of 3
    strata = {k: ("a" if int(k[1:]) < 150 else "b") for k in items}
    p1 = u.partition_groups(items, {"src": 0.5, "val": 0.2, "id": 0.3}, "salt", strata)
    p2 = u.partition_groups(dict(reversed(list(items.items()))), {"src": 0.5, "val": 0.2, "id": 0.3}, "salt", strata)
    assert p1 == p2
    assert sorted(sum(p1.values(), [])) == sorted(items)
    groups = {k: {items[i] for i in v} for k, v in p1.items()}
    assert not groups["src"] & groups["val"] and not groups["src"] & groups["id"] and not groups["val"] & groups["id"]
    assert abs(len(p1["src"]) - 150) <= 6 and abs(len(p1["val"]) - 60) <= 6
    assert u.partition_groups(items, {"src": 0.5, "val": 0.2, "id": 0.3}, "other", strata) != p1


def test_rotating_quotas_and_draw_stratified_prefix_stability():
    assert u.rotating_quotas(16, ["a", "b", "c", "d", "e"], 0) == {"a": 4, "b": 3, "c": 3, "d": 3, "e": 3}
    assert u.rotating_quotas(16, ["a", "b", "c", "d", "e"], 1) == {"a": 3, "b": 4, "c": 3, "d": 3, "e": 3}
    assert sum(u.rotating_quotas(8, ["x", "y", "z"], 5).values()) == 8
    pools = {"a": [f"a{j}" for j in range(40)], "b": [f"b{j}" for j in range(40)]}
    e3 = u.draw_stratified(pools, 3, 10, np.random.default_rng(1), "t")
    e4 = u.draw_stratified(pools, 4, 10, np.random.default_rng(1), "t")
    assert e4[:3] == e3
    flat = [i for e in e4 for i in e]
    assert len(flat) == len(set(flat)) == 40
    assert all(sum(i.startswith("a") for i in e) == 5 for e in e4)
    with pytest.raises(u.PoolExhausted):
        u.draw_stratified(pools, 9, 10, np.random.default_rng(1), "t")


def test_stratified_sample_and_norm_score():
    rng = np.random.default_rng(0)
    s = u.stratified_sample({"a": [f"a{j}" for j in range(100)], "b": ["b0", "b1"]}, 10, rng)
    assert len(s) == 10 and any(x.startswith("b") for x in s) and len(set(s)) == 10
    assert u.norm_score(0.5, 0.25, "max") == pytest.approx(2.0)
    assert u.norm_score(0.02, 0.04, "min") == pytest.approx(2.0)
    assert u.norm_score(0.0, 0.04, "min") == u.NORM_CLIP
    assert u.norm_score(None, 1.0, "max") == 0.0
    assert u.norm_score(100.0, 1.0, "max") == u.NORM_CLIP


def test_as_list_and_list_length_constraint():
    assert u.as_list(np.array(["a", "b"], dtype=object))[0] == ["a", "b"]
    assert u.as_list("abc")[0] is None and u.as_list(None)[0] is None
    c = u.c_list_length(2, "x")
    assert c.check(["a", "b"], None)[0] and not c.check(["a"], None)[0] and not c.check({"a": 1}, None)[0]
    with pytest.raises(ValueError):
        u.check_split("rep")


def test_balanced_sample():
    rng = np.random.default_rng(2)
    pools = {"a": [f"a{j}" for j in range(100)], "b": [f"b{j}" for j in range(3)], "c": [f"c{j}" for j in range(50)]}
    s = u.balanced_sample(pools, 20, rng)
    assert len(s) == 20 and len(set(s)) == 20
    cnt = {k: sum(x.startswith(k) for x in s) for k in pools}
    assert cnt["b"] == 3 and abs(cnt["a"] - cnt["c"]) <= 1
    s2 = u.balanced_sample(pools, 10, np.random.default_rng(3), exclude=set(s))
    assert not set(s2) & set(s)
    assert len(u.balanced_sample({"x": ["x0"]}, 5, rng)) == 1
