"""Unit tests of scienceclaw.bench.tasks._forecast_common and shared checks for the FoR33/35/37/38/41 adapters.

The ``check_*`` helpers are imported by tests/test_task_for33.py … test_task_for41.py.
"""
from __future__ import annotations

import json
import math
import struct
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from scienceclaw.bench.splits import split_seed
from scienceclaw.bench.task import Episode
from scienceclaw.bench.tasks import _forecast_common as fc

COUNTS = {"src": 7, "val": 2, "id": 4, "ood": 4}
BENCH_SEED = 20260928


# ================================================================================================ shared checks
def build_all(adapter, counts: dict[str, int] = COUNTS, seed: int = BENCH_SEED, ipe: int = 16) -> dict[str, list[Episode]]:
    return {s: adapter.build_episodes(s, n, split_seed(seed, adapter.discipline, s), items_per_episode=ipe)
            for s, n in counts.items()}


def check_structure(adapter, eps: dict[str, list[Episode]], ipe: int = 16) -> None:
    ids = set()
    for split, lst in eps.items():
        assert len(lst) == COUNTS[split]
        for ep in lst:
            assert ep.split == split and ep.discipline == adapter.discipline and ep.family == adapter.family
            assert ep.id not in ids
            ids.add(ep.id)
            assert ep.n_items == ipe == len(ep.lineage["item_ids"]) == len(set(ep.lineage["item_ids"]))
            assert ep.direction == adapter.direction == "min"
            assert ep.metric == adapter.metric
            assert ep.objective and "Deliverable" in ep.objective
            assert ep.lineage["rebuilt_split"] is True and ep.lineage["historical_sample_ids_recovered"] is False
            assert ep.lineage["pool"] == ("ood" if split == "ood" else "iid")
            if split == "ood":
                assert ep.lineage["ood_kind"] in ("cross_dataset", "proxy_within_dataset")
            assert ep._dev_evaluate is None
            assert {t.name for t in ep.tools} >= {"load_eval_inputs", "load_dev", "score_dev"}
            assert ep.budget.max_node_s >= 60 and ep.budget.max_llm_items <= 4 * ep.n_items
            json.dumps(ep.lineage)                     # manifest-serializable
            pv = ep.public_view()
            assert "evaluate" not in json.dumps(pv, default=str)


def check_disjoint(eps: dict[str, list[Episode]]) -> None:
    owner: dict[str, str] = {}
    for split, lst in eps.items():
        seen_in_split: set[str] = set()
        for ep in lst:
            for iid in ep.lineage["item_ids"]:
                assert owner.get(iid, split) == split, f"item {iid} in {owner.get(iid)} and {split}"
                owner[iid] = split
                if split != "src":
                    assert iid not in seen_in_split, f"{iid} repeated inside split {split}"
                seen_in_split.add(iid)


def check_determinism(make_adapter: Callable[[], Any], eps: dict[str, list[Episode]]) -> None:
    fresh = make_adapter()
    again = build_all(fresh)
    for split in eps:
        assert [e.lineage["item_ids"] for e in eps[split]] == [e.lineage["item_ids"] for e in again[split]]
        assert [e.id for e in eps[split]] == [e.id for e in again[split]]
        a = eps[split][0].tool("load_eval_inputs").fn({}, {})
        b = again[split][0].tool("load_eval_inputs").fn({}, {})
        assert fingerprint(a) == fingerprint(b)
    # prefix stability and seed sensitivity
    s = split_seed(BENCH_SEED, fresh.discipline, "src")
    two = fresh.build_episodes("src", 2, s)
    three = fresh.build_episodes("src", 3, s)
    assert [e.lineage["item_ids"] for e in two] == [e.lineage["item_ids"] for e in three[:2]]
    other = fresh.build_episodes("src", 2, s + 1)
    assert [e.lineage["item_ids"] for e in other] != [e.lineage["item_ids"] for e in two]


def fingerprint(obj: Any) -> str:
    import hashlib

    h = hashlib.sha256()

    def walk(o: Any) -> None:
        if isinstance(o, np.ndarray):
            h.update(np.ascontiguousarray(o).tobytes())
            h.update(str(o.shape).encode())
        elif hasattr(o, "to_numpy") and hasattr(o, "columns"):
            walk(o.to_numpy(dtype=float, na_value=np.nan) if o.select_dtypes("number").shape[1] == o.shape[1]
                 else o.astype(str).to_numpy().astype("U"))
        elif isinstance(o, dict):
            for k in sorted(o):
                h.update(str(k).encode())
                walk(o[k])
        elif isinstance(o, (list, tuple)):
            for x in o:
                walk(x)
        else:
            h.update(repr(o).encode())
    walk(obj)
    return h.hexdigest()


def check_tool_schemas(ep: Episode) -> dict[str, dict]:
    """Call every data tool; outputs must match the declared ports (keys, int dims, numeric finiteness mix ok)."""
    outs = {}
    for t in ep.tools:
        if t.inputs:
            continue
        out = t.fn({}, {})
        assert set(out) == set(t.outputs), (t.name, set(out) ^ set(t.outputs))
        for port, schema in t.outputs.items():
            v = out[port]
            if schema.shape is not None and schema.type in ("array", "list", "table"):
                shape = np.shape(v) if schema.type != "list" else (len(v),)
                assert len(shape) >= 1
                for got, want in zip(shape, schema.shape):
                    if isinstance(want, int):
                        assert got == want, (t.name, port, shape, schema.shape)
            if schema.type == "array":
                assert isinstance(v, np.ndarray) and v.dtype.kind == "f", (t.name, port)
        outs[t.name] = out
    return outs


def contains_run(hay: np.ndarray, needle: np.ndarray, axis: int = -1, tol: float = 0.0) -> bool:
    """True if the finite pattern ``needle`` (1-D) appears contiguously along ``axis`` of ``hay``."""
    hay = np.moveaxis(np.asarray(hay, float), axis, -1)
    rows = hay.reshape(-1, hay.shape[-1])
    needle = np.asarray(needle, float)
    m = np.isfinite(needle)
    if m.sum() < 3 or rows.shape[1] < needle.size:
        return False
    win = np.lib.stride_tricks.sliding_window_view(rows, needle.size, axis=1)   # (r, k, H)
    diff = np.abs(win[..., m] - needle[m])
    return bool(np.any(np.all(diff <= tol, axis=-1)))


def check_eval_paths(ep: Episode, ref_y: Any, oracle_y: Any, malformed_ys: list[Any], adapter,
                     violating_ys: list[Any] = ()) -> None:
    """reference -> not accepted, norm 1; oracle -> z = 1; malformed (shape/type/NaN) -> no primary, constraint
    fires; violating (well-formed but outside the hard constraints) -> constraint fires, z = 0."""
    r = ep.evaluate(ref_y, None)
    assert r.completed and r.primary is not None
    assert math.isclose(r.primary, r.details["reference"], rel_tol=1e-9, abs_tol=1e-12)
    assert r.accepted is False and r.z == 0 and all(r.h.values()), r.h_msgs
    assert math.isclose(r.details["norm_score"], 1.0, rel_tol=1e-9)
    assert r.details["pooled_payload"] is not None
    json.dumps(r.to_dict())
    pooled = adapter.pooled_metric([r.details])
    assert math.isclose(pooled, r.primary, rel_tol=1e-9), (pooled, r.primary)
    assert math.isclose(adapter.pooled_metric([r.details["pooled_payload"], None]), r.primary, rel_tol=1e-9)
    o = ep.evaluate(oracle_y, None)
    assert o.accepted is True and o.z == 1 and all(o.h.values()), (o.primary, o.h_msgs)
    assert o.primary < r.primary and o.details["norm_score"] > 1.0
    for bad in malformed_ys:
        b = ep.evaluate(bad, None)
        assert b.z == 0 and not b.accepted and b.primary is None
        assert not all(b.h.values()), f"no constraint fired for {type(bad).__name__}"
        # LEAK-5: a malformed output pools the *reference* payload (failed=True), it never drops out of the pool
        assert b.details["pooled_payload"] is not None and b.details["norm_score"] == 0.0
        assert b.details["pooled_payload"] == b.details["reference_payload"] and b.details["failed"] is True
        assert math.isclose(adapter.pooled_metric([b.details]), b.details["reference"], rel_tol=1e-9)
    for bad in violating_ys:
        b = ep.evaluate(bad, None)
        assert b.z == 0 and not all(b.h.values()), f"no constraint fired: {b.h_msgs}"
    nb = ep.evaluate(None, None)
    assert nb.z == 0 and nb.details["failed"] is True and nb.details["pooled_payload"] == nb.details["reference_payload"]
    assert math.isclose(adapter.pooled_metric([nb.details]), nb.details["reference"], rel_tol=1e-9)


def check_score_dev(ep: Episode, dev_pred: Any, bad_pred: Any) -> None:
    out = ep.tool("score_dev").fn({"pred": dev_pred}, {})
    assert math.isfinite(out["score"]) and "reference_score" in out["report"]
    with pytest.raises(ValueError):
        ep.tool("score_dev").fn({"pred": bad_pred}, {})


# ================================================================================================ unit tests
def test_draw_episodes_prefix_stable_balanced_and_exhaustion() -> None:
    pool = [fc.PoolItem(f"i{g}-{k}", f"g{g}") for g in range(4) for k in range(8)]
    e3, reused = fc.draw_episodes(pool, 3, 8, fc.make_rng("t", 1), balance_groups=True)
    e2, _ = fc.draw_episodes(pool, 2, 8, fc.make_rng("t", 1), balance_groups=True)
    assert not reused and [[i.id for i in e] for e in e2] == [[i.id for i in e] for e in e3[:2]]
    for ep in e3:
        assert sorted(sum(1 for i in ep if i.group == g) for g in ("g0", "g1", "g2", "g3")) == [2, 2, 2, 2]
    ids = [i.id for e in e3 for i in e]
    assert len(ids) == len(set(ids))
    with pytest.raises(fc.PoolExhausted):
        fc.draw_episodes(pool, 5, 8, fc.make_rng("t", 1))
    many, reused = fc.draw_episodes(pool, 6, 8, fc.make_rng("t", 1), allow_reuse=True)
    assert reused and all(len({i.id for i in e}) == 8 for e in many)


def test_draw_episodes_distinct_groups() -> None:
    pool = [fc.PoolItem(f"s{g}-{k}", f"s{g}") for g in range(20) for k in range(5)]
    eps, _ = fc.draw_episodes(pool, 5, 16, fc.make_rng("d", 2), distinct_groups=True)
    for ep in eps:
        assert len({i.group for i in ep}) == 16


def _min_gap(ep) -> float:
    ts = sorted(i.meta[0] for i in ep)
    return min(b - a for a, b in zip(ts, ts[1:]))


def test_draw_episodes_conflict_min_spacing_greedy() -> None:
    """LEAK-4: a conflict rule keeps conflicting items out of one episode, prefix-stable and deterministic."""
    pool = [fc.PoolItem(f"t{h}", "g", (h,)) for h in range(0, 400, 4)]          # 100 items, 4 h apart
    conflict = fc.time_spacing_conflict(lambda it: it.meta[0], 40)
    e3, _ = fc.draw_episodes(pool, 3, 5, fc.make_rng("c", 1), conflict=conflict)
    e2, _ = fc.draw_episodes(pool, 2, 5, fc.make_rng("c", 1), conflict=conflict)
    assert [[i.id for i in e] for e in e2] == [[i.id for i in e] for e in e3[:2]]
    assert all(len(ep) == 5 and _min_gap(ep) >= 40 for ep in e3)
    ids = [i.id for e in e3 for i in e]
    assert len(ids) == len(set(ids))
    assert fc.time_spacing_conflict(lambda it: it.meta[0], 40)(pool[0], pool[9]) and not conflict(pool[0], pool[10])
    # distinct groups + conflict together
    pool2 = [fc.PoolItem(f"s{g}-{k}", f"s{g}", (10 * g + 1000 * k,)) for g in range(8) for k in range(4)]
    eps, _ = fc.draw_episodes(pool2, 2, 4, fc.make_rng("c", 2), distinct_groups=True,
                              conflict=fc.time_spacing_conflict(lambda it: it.meta[0], 25))
    assert all(len({i.group for i in ep}) == 4 and _min_gap(ep) >= 25 for ep in eps)


def test_draw_episodes_conflict_too_tight_raises() -> None:
    pool = [fc.PoolItem(f"t{h}", "g", (h,)) for h in range(6)]
    conflict = fc.time_spacing_conflict(lambda it: it.meta[0], 10)              # every pair clashes
    with pytest.raises(fc.PoolExhausted, match="conflict-free"):
        fc.draw_episodes(pool, 1, 3, fc.make_rng("c", 3), conflict=conflict, what="toy")
    with pytest.raises(fc.PoolExhausted, match="conflict-free"):
        fc.draw_episodes(pool, 1, 3, fc.make_rng("c", 3), conflict=conflict, allow_reuse=True, what="toy")
    assert fc.draw_episodes(pool, 1, 1, fc.make_rng("c", 3), conflict=conflict)[0][0][0].id.startswith("t")


def test_draw_episodes_lanes() -> None:
    """Lane drawing: every episode comes from one lane; capacity counts whole ipe-sized chunks per lane."""
    pool = [fc.PoolItem(f"L{g}-{k}", f"lane{g}", (120 * k + 12 * g,)) for g in range(4) for k in range(16)]
    conflict = fc.time_spacing_conflict(lambda it: it.meta[0], 120)
    eps, reused = fc.draw_episodes(pool, 4, 16, fc.make_rng("l", 1), lanes=True, conflict=conflict)
    assert not reused and all(len({i.group for i in ep}) == 1 and _min_gap(ep) >= 120 for ep in eps)
    assert sorted(ep[0].group for ep in eps) == ["lane0", "lane1", "lane2", "lane3"]
    ids = [i.id for e in eps for i in e]
    assert len(ids) == len(set(ids))
    e2, _ = fc.draw_episodes(pool, 2, 16, fc.make_rng("l", 1), lanes=True, conflict=conflict)
    assert [[i.id for i in e] for e in e2] == [[i.id for i in e] for e in eps[:2]]        # prefix-stable
    half, _ = fc.draw_episodes(pool, 8, 8, fc.make_rng("l", 1), lanes=True, conflict=conflict)
    assert len(half) == 8 and all(len({i.group for i in ep}) == 1 for ep in half)         # two chunks per lane
    with pytest.raises(fc.PoolExhausted, match="capacity 4"):
        fc.draw_episodes(pool, 5, 16, fc.make_rng("l", 1), lanes=True, conflict=conflict, what="toy")
    with pytest.raises(fc.PoolExhausted, match="no lane holds 17"):
        fc.draw_episodes(pool, 1, 17, fc.make_rng("l", 1), lanes=True, conflict=conflict, what="toy")
    many, reused = fc.draw_episodes(pool, 6, 16, fc.make_rng("l", 1), lanes=True, conflict=conflict, allow_reuse=True)
    assert reused and len(many) == 6 and all(len({i.id for i in ep}) == 16 for ep in many)


def test_draw_episodes_lane_that_violates_the_conflict_rule_is_rejected() -> None:
    """The conflict rule is re-checked on the drawn episodes whatever the draw mode."""
    pool = [fc.PoolItem(f"x{k}", "lane", (10 * k,)) for k in range(16)]          # lane spaced only 10 h apart
    with pytest.raises(ValueError, match="conflicting items"):
        fc.draw_episodes(pool, 1, 16, fc.make_rng("l", 2), lanes=True,
                         conflict=fc.time_spacing_conflict(lambda it: it.meta[0], 120), what="toy")


def test_mase_smape_known_values() -> None:
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8], float)            # m = 2 -> scale = mean |x_t - x_{t-2}| = 2
    assert fc.mase(np.array([10.0, 12.0]), np.array([9.0, 14.0]), x, 2) == pytest.approx(1.5 / 2)
    assert fc.smape(np.array([100.0, 0.0]), np.array([50.0, 0.0])) == pytest.approx(200 * (50 / 150) / 2)
    with pytest.raises(ValueError):
        fc.mase(np.ones(2), np.ones(2), np.ones(8), 2)


def test_crps_normal_matches_numerical_integral() -> None:
    from scipy.stats import norm

    mu, sig, y = 1.3, 0.7, 2.1
    xs = np.linspace(mu - 12 * sig, mu + 12 * sig, 400001)
    num = np.trapezoid((norm.cdf(xs, mu, sig) - (xs >= y)) ** 2, xs)
    assert float(fc.crps_normal(np.array(mu), np.array(sig), np.array(y))) == pytest.approx(num, rel=1e-5)
    with pytest.raises(ValueError):
        fc.crps_normal(np.array([0.0]), np.array([0.0]), np.array([0.0]))


def test_wb2_lat_weights() -> None:
    lat = np.linspace(-87.1875, 87.1875, 32)
    w = fc.wb2_lat_weights(lat)
    assert w.mean() == pytest.approx(1.0) and w[0] < w[15] and np.allclose(w, w[::-1])


def test_norm_score_and_acceptance() -> None:
    assert fc.norm_score(2.0, 4.0, "min") == 2.0 and fc.norm_score(0.0, 4.0, "min") == fc.NORM_CLIP
    assert fc.norm_score(None, 4.0, "min") == 0.0 and fc.norm_score(3.0, 1.0, "max") == 3.0
    assert fc.beats_reference(0.9, 1.0, "min", 0.05) and not fc.beats_reference(0.96, 1.0, "min", 0.05)
    assert not fc.beats_reference(float("nan"), 1.0, "min", 0.0)


def test_constraints_messages() -> None:
    cs = [fc.c_shape((2, 3)), fc.c_finite(), fc.c_range(0.0, 10.0, "kWh"), fc.c_declared_unit("kWh")]
    good = np.ones((2, 3))
    assert all(c.check(good, None)[0] for c in cs)
    assert not cs[0].check(np.ones((3, 2)), None)[0]
    assert not cs[1].check(np.array([[1, np.nan, 1], [1, 1, 1]]), None)[0]
    assert not cs[2].check(-good, None)[0]
    assert not cs[0].check({"a": 1}, None)[0] and not cs[0].check("text", None)[0]
    from scienceclaw.core.trace import NodeRecord, Trace

    tr = Trace(records={"s": NodeRecord("s", "fp", "ok", kind="submit", outputs_summary={"y": {"unit": "Wh"}})})
    assert not cs[3].check(good, tr)[0]


def test_lz4_and_blosc_decoder_synthetic() -> None:
    from scienceclaw.bench.tasks.for37_weatherbench import blosc_decompress, lz4_block_decompress

    assert lz4_block_decompress(b"\x50hello", 5) == b"hello"
    assert lz4_block_decompress(b"\x35abc\x03\x00", 12) == b"abcabcabcabc"      # overlapping match
    vals = np.arange(200, dtype="<f4") * 1.5
    planes = vals.view(np.uint8).reshape(200, 4).T                                # byte-shuffled layout
    nbytes = vals.nbytes
    body = b"".join(struct.pack("<i", 200) + planes[k].tobytes() for k in range(4))   # stored splits
    header_len = 16 + 4
    frame = bytes([2, 1, 0x21, 4]) + struct.pack("<III", nbytes, nbytes, header_len + len(body))
    frame += struct.pack("<i", header_len) + body
    out = np.frombuffer(blosc_decompress(frame), dtype="<f4")
    assert np.array_equal(out, vals)
    memcpy = bytes([2, 1, 0x02, 4]) + struct.pack("<III", nbytes, nbytes, 16 + nbytes) + vals.tobytes()
    assert np.array_equal(np.frombuffer(blosc_decompress(memcpy), "<f4"), vals)


def test_unwrap_payloads() -> None:
    assert fc.unwrap_payloads([None, {"pooled_payload": None}, {"pooled_payload": {"a": 1}}, {"b": 2}]) == [{"a": 1}, {"b": 2}]
    with pytest.raises(ValueError):
        fc.unwrap_payloads([[1, 2]])


def test_split_plan_integration() -> None:
    """SplitPlan (lead/other engineer) must accept the five adapters without lineage overlap."""
    from scienceclaw.bench.registry import get_adapter
    from scienceclaw.bench.splits import SplitPlan
    from scienceclaw.config import BenchConfig

    adapters = {}
    for code in ("FoR33", "FoR35", "FoR37", "FoR38", "FoR41"):
        a = get_adapter(code)
        if a.available()[0]:
            adapters[code] = a
    if not adapters:
        pytest.skip("no forecasting dataset available")
    plan = SplitPlan.build(BenchConfig(disciplines=list(adapters)), adapters)
    m = plan.manifest()
    assert not [w for w in m["warnings"] if "reused" in w or "returned" in w]
    for code in adapters:
        assert len(plan.episodes["src"][code]) == 7 and len(plan.episodes["ood"][code]) == 4
