"""FoR37 WeatherBench 2 adapter: zarr decoding, time-disjoint splits, leakage, evaluator (lat-weighted RMSE)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_task_forecast_common import (build_all, check_determinism, check_disjoint, check_eval_paths,  # noqa: E402
                                       check_score_dev, check_structure, check_tool_schemas)

from scienceclaw.bench.tasks._forecast_common import PoolExhausted  # noqa: E402
from scienceclaw.bench.tasks.for37_weatherbench import (CONTEXT_OFFSETS_H, GUARD_SLOTS, LANE_ITEMS, LEAD_H,  # noqa: E402
                                                         MIN_SPACING_H, PATTERN_PERIOD_H, T_ORIGIN, TRAIN_END,
                                                         Adapter, _tidx, read_zarr_array, weighted_rmse)

_ADAPTER = Adapter()
OK, WHY = _ADAPTER.available()
pytestmark = pytest.mark.skipif(not OK, reason=f"FoR37 data unavailable: {WHY}")


@pytest.fixture(scope="module")
def eps():
    return build_all(_ADAPTER)


def _targets(ep) -> np.ndarray:
    d = _ADAPTER.data()
    return np.stack([d.t2m[_tidx(d.items[i] + np.timedelta64(LEAD_H, "h"))] for i in ep.lineage["item_ids"]]).astype(float)


def test_decoded_store_matches_receipt():
    d = _ADAPTER.data()
    assert d.t2m.shape == (4384, 64, 32) and d.t2m.dtype == np.float32
    assert float(d.t2m[0, 0, 0]) == pytest.approx(250.15936279296875)
    assert np.allclose(np.diff(d.lat), 5.625) and d.lon[0] == 0.0 and d.lon[-1] == pytest.approx(354.375)
    root = _ADAPTER.data_root / "for37-weatherbench2" / "era5_2m_temperature_2018-2020_6h_64x32.zarr"
    lat = read_zarr_array(root / "latitude")     # pure-python decoder path on a small array
    assert np.allclose(lat, d.lat)
    from scienceclaw.bench.tasks.for37_weatherbench import blosc_decompress

    for key, sl in (("0.0.0", slice(0, 100)), ("43.0.0", slice(4300, 4384))):   # first and (padded) last chunk
        chunk = np.frombuffer(blosc_decompress((root / "2m_temperature" / key).read_bytes()), "<f4").reshape(100, 64, 32)
        assert np.array_equal(chunk[:sl.stop - sl.start], d.t2m[sl])


def test_structure_disjoint_deterministic(eps):
    check_structure(_ADAPTER, eps)
    check_disjoint(eps)
    check_determinism(Adapter, eps)
    _ADAPTER.verify_disjoint()
    d = _ADAPTER.data()
    for s in ("src", "val", "id"):
        assert all(str(d.items[i])[:4] == "2019" for e in eps[s] for i in e.lineage["item_ids"])
    assert all(str(d.items[i])[:4] == "2020" for e in eps["ood"] for i in e.lineage["item_ids"])


def _hours(ep) -> list[int]:
    d = _ADAPTER.data()
    return sorted(int((d.items[i] - T_ORIGIN).astype(int)) for i in ep.lineage["item_ids"])


def test_min_spacing_inside_every_episode(eps):
    """no two initialisations of one episode are closer than MIN_SPACING_H (another item's context field
    would be a near-future observation of the first item's target)."""
    for split, es in eps.items():
        for ep in es:
            hrs = _hours(ep)
            assert len(hrs) == ep.n_items == 16
            assert min(b - a for a, b in zip(hrs, hrs[1:])) >= MIN_SPACING_H == 120
            assert ep.lineage["min_item_spacing_h"] == min(b - a for a, b in zip(hrs, hrs[1:]))
            assert ep.lineage["min_spacing_h"] == MIN_SPACING_H
            assert len(ep.lineage["groups"]) == 1                     # one lane per episode
    # the constraint holds for other seeds and for smaller episodes as well
    for seed in (1, 20260929):
        for ep in _ADAPTER.build_episodes("src", 8, seed) + _ADAPTER.build_episodes("ood", 16, seed):
            hrs = _hours(ep)
            assert min(b - a for a, b in zip(hrs, hrs[1:])) >= MIN_SPACING_H
    small = _ADAPTER.build_episodes("id", 8, 5, items_per_episode=8)
    assert len(small) == 8 and all(min(b - a for a, b in zip(_hours(e), _hours(e)[1:])) >= MIN_SPACING_H for e in small)
    assert len({i for e in small for i in e.lineage["item_ids"]}) == 64
    with pytest.raises(PoolExhausted):
        _ADAPTER.build_episodes("id", 1, 5, items_per_episode=17)     # no conflict-free lane holds 17 items


def _pool_hours() -> dict[str, list[int]]:
    d = _ADAPTER.data()
    return {k: sorted(int((d.items[i.id] - T_ORIGIN).astype(int)) for i in v) for k, v in _ADAPTER.pools().items()}


def test_range_and_lane_structure_and_capacity():
    pools = _ADAPTER.pools()
    assert {k: len(v) for k, v in pools.items()} == {"val": 64, "id": 64, "src": 128, "ood": 256}
    assert {k: _ADAPTER.capacity(k, 16) for k in pools} == {"val": 4, "id": 4, "src": 8, "ood": 16}
    assert {k: _ADAPTER.capacity(k, 8) for k in pools} == {"val": 8, "id": 8, "src": 16, "ood": 32}
    for pool in pools.values():                                          # lanes: 16 items exactly 120 h apart
        lanes: dict[str, list[int]] = {}
        for it in pool:
            lanes.setdefault(it.group, []).append(it.meta[0])
        assert all(len(v) == LANE_ITEMS == 16 and set(np.diff(sorted(v))) == {MIN_SPACING_H} for v in lanes.values())
    assert [i.id for i in Adapter().pools()["val"]] == [i.id for i in pools["val"]]     # fixed partition


def test_splits_are_time_disjoint_with_a_guard_margin():
    """Whichever pools two neighbouring initialisation times belong to, a change of pool crosses a guard gap, and
    the ranges (runs closer than one 60 h slot pair) are 8 in total: 4 per year, 2019 ones split 1 / 1 / 2."""
    t = _pool_hours()
    assert max(t["val"] + t["id"] + t["src"]) < min(t["ood"])                        # the whole OOD year is later
    allh = sorted((h, k) for k, v in t.items() for h in v)
    n_ranges = 1
    for (h0, k0), (h1, k1) in zip(allh, allh[1:]):
        if h1 - h0 > 48:                                # inside a range consecutive inits are 12 h or 48 h apart
            n_ranges += 1
            assert h1 - h0 >= (GUARD_SLOTS + 1) * PATTERN_PERIOD_H - 12          # >= 288 h between ranges
        else:
            assert k0 == k1                                                      # a range belongs to one pool
    assert n_ranges == 8
    assert sorted(len(v) for k, v in t.items() if k != "ood") == [64, 64, 128]


def test_no_target_time_visible_anywhere():
    """No item's target field time lies in the training period, the dev inputs or any item's context window."""
    d = _ADAPTER.data()
    ctx = {t + np.timedelta64(o, "h") for t in d.items.values() for o in CONTEXT_OFFSETS_H}
    ctx |= {t + np.timedelta64(o, "h") for t in d.dev_t0 for o in CONTEXT_OFFSETS_H}
    tg = {t + np.timedelta64(LEAD_H, "h") for t in d.items.values()}
    assert not tg & ctx and min(tg) > TRAIN_END


def test_no_label_leak_in_tool_outputs(eps):
    for split in ("val", "id", "ood"):
        ep = eps[split][0]
        outs = check_tool_schemas(ep)
        tgt = _targets(ep).reshape(ep.n_items, -1)
        for out in outs.values():
            for v in out.values():
                if isinstance(v, np.ndarray) and v.ndim >= 3:
                    fields = v.reshape(-1, 64 * 32).astype(float)
                    for row in tgt:
                        assert not np.any(np.all(np.isclose(fields, row, atol=1e-6), axis=1)), f"leak in {split}"


def test_evaluator_reference_oracle_malformed(eps):
    for split in ("val", "id", "ood"):
        ep = eps[split][0]
        ctx = ep.tool("load_eval_inputs").fn({}, {})["context"]
        ref = ctx[:, -1]
        n = ep.n_items
        check_eval_paths(ep, ref, _targets(ep), [np.zeros((n, 32, 64)), ref[:, :, :, None], np.full_like(ref, np.nan)],
                         _ADAPTER, violating_ys=[ref - 273.15])


def test_rmse_weighting():
    w = np.ones(32)
    a, b = np.zeros((1, 64, 32)), np.ones((1, 64, 32)) * 2.0
    assert weighted_rmse(a, b, w)[0] == pytest.approx(2.0)


def test_score_dev(eps):
    ep = eps["src"][0]
    dev = ep.tool("load_dev").fn({}, {})
    check_score_dev(ep, dev["context"][:, -1], np.zeros((16, 64)))


def test_objective_documents_the_domain_library(eps):
    ep = eps["src"][0]
    assert "scilib.weather" in ep.objective and "fit_patch_ridge" in ep.objective and "blocked_cv" in ep.objective
    desc = ep.tool("score_dev").description
    assert "load_eval_inputs" in desc and "(context, init_time) -> forecast" in desc     # dev and final share one path
    # the documented layout is what the tools deliver: the library runs on load_train / load_dev output and score_dev accepts it
    import scilib.weather as W

    tr, dev = ep.tool("load_train").fn({}, {}), ep.tool("load_dev").fn({}, {})
    assert np.allclose(W.lat_weights(tr["latitude"]), tr["lat_weights"])
    model = W.fit_patch_ridge(tr["t2m"][:400], tr["time"][:400], radius=1)
    pred = W.predict_patch_ridge(model, dev["context"], dev["init_time"])
    assert pred.shape == (len(dev["init_time"]), 64, 32) and np.isfinite(pred).all()
    out = ep.tool("score_dev").fn({"pred": pred}, {})
    assert np.isfinite(out["score"]) and out["score"] == pytest.approx(
        float(np.mean(out["report"]["per_item"])))


def test_unavailable_reports_reason(tmp_path):
    ok, why = Adapter(data_root=tmp_path).available()
    assert not ok and "missing" in why
