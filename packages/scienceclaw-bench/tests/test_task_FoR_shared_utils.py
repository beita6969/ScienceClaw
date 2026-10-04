"""Unit tests of the helpers shared by the FoR34/39/40/44/51 adapters (pure logic, no data needed)."""
from __future__ import annotations

import numpy as np
import pytest

from scienceclaw.bench.registry import get_adapter
from scienceclaw.bench.tasks._adapter_utils_for34_39_40_44_51 import (
    PoolExhausted, allocate_counts, as_float_vector, c_finite, c_range, c_vector, check_split, draw_episodes,
    norm_score, partition_pool,
)


def test_allocate_counts_sums_and_is_stable():
    c = allocate_counts(101, {"a": 0.55, "b": 0.15, "c": 0.30})
    assert sum(c.values()) == 101 and c == allocate_counts(101, {"a": 0.55, "b": 0.15, "c": 0.30})
    assert allocate_counts(0, {"a": 1.0}) == {"a": 0}


def test_partition_pool_disjoint_complete_stratified_and_seed_independent():
    ids = [f"i{j}" for j in range(500)]
    strata = {i: int(i[1:]) % 3 for i in ids}
    p1 = partition_pool(ids, {"x": 0.5, "y": 0.2, "z": 0.3}, "salt", strata)
    p2 = partition_pool(list(reversed(ids)), {"x": 0.5, "y": 0.2, "z": 0.3}, "salt", strata)
    assert p1 == p2                                      # input order does not matter
    allp = sum(p1.values(), [])
    assert sorted(allp) == sorted(ids) and len(set(allp)) == len(ids)
    for s in range(3):                                   # every stratum split in proportion
        n_s = sum(1 for i in p1["x"] if strata[i] == s)
        assert abs(n_s - 0.5 * sum(1 for i in ids if strata[i] == s)) <= 1
    assert partition_pool(ids, {"x": 0.5, "y": 0.5}, "other")["x"] != p1["x"]


def test_draw_episodes_prefix_stable_disjoint_and_exhaustion():
    pools = {1: [f"p{j}" for j in range(10)], 0: [f"n{j}" for j in range(40)]}
    a = draw_episodes(pools, {1: 2, 0: 6}, 4, np.random.default_rng(5))
    b = draw_episodes(pools, {1: 2, 0: 6}, 5, np.random.default_rng(5))
    assert a == b[:4]
    flat = sum(b, [])
    assert len(flat) == len(set(flat)) == 40
    assert all(sum(i.startswith("p") for i in e) == 2 for e in b)
    with pytest.raises(PoolExhausted):
        draw_episodes(pools, {1: 2, 0: 6}, 6, np.random.default_rng(5))


def test_norm_score_directions_and_clip():
    assert norm_score(0.8, 0.4, "max") == pytest.approx(2.0)
    assert norm_score(10.0, 20.0, "min") == pytest.approx(2.0)
    assert norm_score(0.0, 1.0, "min") == 10.0
    assert norm_score(None, 1.0, "max") == 0.0
    assert norm_score(100.0, 1.0, "max") == 10.0


def test_constraint_factories_fire():
    assert c_vector(3, "x").check([1, 2, 3], None)[0]
    assert not c_vector(3, "x").check([1, 2], None)[0]
    assert not c_vector(3, "x").check("abc", None)[0]
    assert not c_finite(2).check([1.0, np.nan], None)[0]
    assert c_range(2, 0, 1, "r", "d").check([0.0, 1.0], None)[0]
    assert not c_range(2, 0, 1, "r", "d").check([0.0, 1.5], None)[0]
    arr, why = as_float_vector([[1.0], [2.0]], 2)
    assert arr is not None and arr.shape == (2,) and why == ""


def test_check_split_rejects_rep_and_unknown():
    with pytest.raises(ValueError):
        check_split("rep")
    with pytest.raises(ValueError):
        check_split("train")


@pytest.mark.parametrize("code", ["FoR34", "FoR39", "FoR40", "FoR44", "FoR51"])
def test_registry_instantiates_adapter_and_available_is_a_reasoned_bool(code):
    a = get_adapter(code)
    ok, reason = a.available()
    assert isinstance(ok, bool) and isinstance(reason, str) and reason
    assert a.discipline == code and a.direction in ("max", "min")
    for attr in ("name", "family", "metric", "task_type"):
        assert isinstance(getattr(a, attr), str) and getattr(a, attr)
