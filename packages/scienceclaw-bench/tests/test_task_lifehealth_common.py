"""Unit tests of the shared Life & health adapter helpers (no dataset access)."""
from __future__ import annotations

import numpy as np
import pytest

from scienceclaw.bench.task import EvalResult
from scienceclaw.bench.tasks._life_health_common import (allocate_groups, beats, compose_episodes, extract_payloads,
                                                         norm_score, stratified_take, unit_constraint)
from scienceclaw.core.trace import NodeRecord, Trace


def test_compose_episodes_disjoint_when_pool_large_enough():
    pool = [f"i{k}" for k in range(64)]
    eps, reused = compose_episodes(pool, 4, 16, ["t", 0])
    assert not reused
    flat = [x for e in eps for x in e]
    assert len(flat) == len(set(flat)) == 64
    assert all(len(e) == 16 and len(set(e)) == 16 for e in eps)


def test_compose_episodes_reuse_and_prefix_stability():
    pool = [f"i{k}" for k in range(20)]
    eps7, reused = compose_episodes(pool, 7, 16, ["t", 1])
    assert reused
    assert all(len(set(e)) == 16 and set(e) <= set(pool) for e in eps7)
    eps3, _ = compose_episodes(pool, 3, 16, ["t", 1])
    assert eps7[:3] == eps3                              # prefix-stable
    assert compose_episodes(pool, 7, 16, ["t", 1])[0] == eps7   # deterministic
    assert compose_episodes(pool, 7, 16, ["t", 2])[0] != eps7   # seed matters


def test_compose_episodes_stratified():
    pos = [f"p{k}" for k in range(8)]
    neg = [f"n{k}" for k in range(56)]
    eps, reused = compose_episodes([], 4, 16, ["s"], strata={"pos": (pos, 2), "neg": (neg, 14)})
    assert not reused
    for e in eps:
        assert sum(x.startswith("p") for x in e) == 2 and len(set(e)) == 16
    with pytest.raises(ValueError):
        compose_episodes([], 1, 16, ["s"], strata={"pos": (pos, 2), "neg": (neg, 13)})


def test_allocate_groups_keeps_groups_whole():
    groups = {f"g{k}": [f"g{k}a", f"g{k}b"] if k % 3 else [f"g{k}a"] for k in range(40)}
    pools = allocate_groups(groups, [("id", 20), ("val", 10)], "src", ["x"])
    assert len(pools["id"]) == 20 and len(pools["val"]) == 10
    owner = {}
    for name, items in pools.items():
        for it in items:
            owner.setdefault(it.rstrip("ab"), set()).add(name)
    assert all(len(v) == 1 for v in owner.values())
    assert sorted(x for v in pools.values() for x in v) == sorted(x for v in groups.values() for x in v)


def test_effective_items():
    from scienceclaw.bench.tasks._life_health_common import effective_items

    assert effective_items("id", 64, 4, 16) == 16
    assert effective_items("ood", 56, 4, 16) == 14          # held-out splits shrink instead of reusing
    assert effective_items("src", 19, 7, 16) == 16          # source episodes may reuse items
    with pytest.raises(ValueError):
        effective_items("val", 1, 2, 16)


def test_stratified_take():
    tp, tn, rp, rn = stratified_take(list("abcd"), list("efghij"), 2, 3, ["k"])
    assert len(tp) == 2 and len(tn) == 3 and set(tp) | set(rp) == set("abcd") and not set(tn) & set(rn)


def test_norm_score_and_beats():
    assert norm_score(0.8, 0.4, "max") == pytest.approx(2.0)
    assert norm_score(2.0, 4.0, "min") == pytest.approx(2.0)
    assert norm_score(100.0, 1.0, "max") == 10.0
    assert norm_score(None, 1.0, "max") == 0.0
    assert norm_score(0.1, -0.2, "max") == pytest.approx(1.3)     # non-positive reference -> shifted
    assert beats(0.5, 0.4, "max", 0.05) and not beats(0.44, 0.4, "max", 0.05)
    assert beats(1.0, 2.0, "min", 0.5) and not beats(1.8, 2.0, "min", 0.5)


def test_make_result_acceptance_floor():
    from scienceclaw.bench.tasks._life_health_common import make_result
    # negative reference: doing nothing (0.0) beats it without a floor, but not with floor=0
    plain = make_result(0.0, -0.13, "max", 0.05, {}, {"pooled_payload": {}}, True)
    floored = make_result(0.0, -0.13, "max", 0.05, {}, {"pooled_payload": {}}, True, floor=0.0)
    assert plain.accepted and not floored.accepted
    assert floored.details["reference"] == pytest.approx(-0.13)          # the true reference is still reported
    assert floored.details["acceptance_floor"] == 0.0
    assert make_result(0.06, -0.13, "max", 0.05, {}, {}, True, floor=0.0).accepted
    assert not make_result(0.04, -0.13, "max", 0.05, {}, {}, True, floor=0.0).accepted
    # a reference above the floor is unchanged
    assert make_result(0.5, 0.4, "max", 0.05, {}, {}, True, floor=0.0).accepted
    assert not make_result(0.44, 0.4, "max", 0.05, {}, {}, True, floor=0.0).accepted
    # min metrics use min(reference, floor)
    assert make_result(0.4, 1.0, "min", 0.05, {}, {}, True, floor=0.5).accepted
    assert not make_result(0.48, 1.0, "min", 0.05, {}, {}, True, floor=0.5).accepted


def test_extract_payloads_accepts_several_shapes():
    p = {"kind": "k", "v": 1}
    ev = EvalResult(details={"pooled_payload": p})
    got = extract_payloads([p, {"pooled_payload": p}, ev, ev.to_dict(), None, {"pooled_payload": None},
                            {"kind": "other"}], "k")
    assert len(got) == 4


def test_unit_constraint_reads_submit_record():
    c = unit_constraint("1")
    assert c.check(None, None)[0]
    rec = NodeRecord("s", "fp", "ok", kind="submit", outputs_summary={"y": {"unit": "K"}})
    assert not c.check(None, Trace(records={"s": rec}))[0]
    rec.outputs_summary = {"y": {"unit": "1"}}
    assert c.check(None, Trace(records={"s": rec}))[0]
    assert np.isfinite(norm_score(1.0, 1.0, "max"))
