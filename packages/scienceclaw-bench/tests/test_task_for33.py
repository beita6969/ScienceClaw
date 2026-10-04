"""FoR33 BuildingsBench adapter (data team's reconstructed_v2): role-based pools, splits, determinism, leakage,
evaluator (category-median balanced CVRMSE), constraints."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_task_forecast_common import (COUNTS, build_all, check_determinism, check_disjoint, check_eval_paths,  # noqa: E402
                                       check_score_dev, check_tool_schemas, contains_run)

from scienceclaw.bench.tasks.for33_buildingsbench import (MIN_DELAY_H, Adapter, _conflict, average_persistence, balanced_nrmse,  # noqa: E402
                                                           balanced_score, min_delay_observed, nrmse_pct, pack_groups,
                                                           persistence, reserved_units)
from scienceclaw.bench.tasks._forecast_common import PoolExhausted, PoolItem, make_rng  # noqa: E402

_ADAPTER = Adapter()
OK, WHY = _ADAPTER.available()
pytestmark = pytest.mark.skipif(not OK, reason=f"FoR33 data unavailable: {WHY}")

ROLE_BUILDINGS = {"src": 8, "val": 3, "id": 11, "ood": 5}
ROLE_WINDOWS = {"src": 64, "val": 24, "id": 88, "ood": 40}
ITEMS_16 = {"src": 8, "val": 6, "id": 11, "ood": 10}      # M_TARGET windows per building at 16 items per episode


@pytest.fixture(scope="module")
def eps():
    return build_all(_ADAPTER)


def _windows(ep):
    d = _ADAPTER.data()
    return [d.windows[i] for i in ep.lineage["item_ids"]]


def _targets(ep) -> np.ndarray:
    return np.stack([w.target for w in _windows(ep)])


def test_pools_follow_the_role_files():
    ok, why = _ADAPTER.available()
    assert ok and why.startswith("FoR33 reconstructed_v") and "windows src=64" in why and why.endswith("buildings 8/3/11/5")
    pools = _ADAPTER.pools()
    assert {k: len(v) for k, v in pools.items()} == ROLE_WINDOWS
    assert {k: len({it.group for it in v}) for k, v in pools.items()} == ROLE_BUILDINGS
    _ADAPTER.verify_disjoint()                                  # window ids and buildings disjoint across roles
    dl = _ADAPTER.delivery()
    assert dl.version.startswith("reconstructed_v") and dl.base.is_dir()
    d = _ADAPTER.data()
    assert d.history_hours == 3601
    assert set(reserved_units(dl)) == {"MT_070"} and "MT_070" not in d.buildings
    cats = {s: {d.buildings[b].category for b in bs} for s, bs in d.split_buildings.items()}
    assert all(c == {"residential", "commercial"} for c in cats.values())
    assert {d.buildings[b].group for b in d.split_buildings["ood"]} == {"smart", "electricity"}
    assert _ADAPTER.capacity("id") == 8 and _ADAPTER.capacity("val") == 4 and _ADAPTER.capacity("ood") == 4


def test_structure_disjoint_deterministic(eps):
    ids = set()
    for split, lst in eps.items():
        assert len(lst) == COUNTS[split]
        for ep in lst:
            assert ep.split == split and ep.discipline == "FoR33" and ep.family == _ADAPTER.family
            assert ep.id not in ids
            ids.add(ep.id)
            assert ep.n_items == ITEMS_16[split] == len(ep.lineage["item_ids"]) == len(set(ep.lineage["item_ids"]))
            assert ep.direction == "min" and ep.metric == _ADAPTER.metric
            assert "Deliverable" in ep.objective and "balanced NRMSE" in ep.objective
            assert ep.lineage["rebuilt_split"] is True and ep.lineage["historical_sample_ids_recovered"] is False
            assert ep.lineage["pool"] == ("ood" if split == "ood" else "iid")
            assert ep.lineage["items_reused_from_earlier_episodes"] is False
            assert ep._dev_evaluate is None
            assert {t.name for t in ep.tools} == {"load_history", "load_dev", "score_dev", "load_eval_inputs"}
            assert ep.budget.max_node_s >= 60 and ep.budget.max_llm_items <= 4 * ep.n_items
            json.dumps(ep.lineage)
            assert "evaluate" not in json.dumps(ep.public_view(), default=str)
    assert eps["ood"][0].lineage["ood_kind"] == "proxy_within_dataset"
    check_disjoint(eps)
    check_determinism(Adapter, eps)


def test_buildings_disjoint_complete_and_window_disjoint(eps):
    d = _ADAPTER.data()
    role_b = {s: set(bs) for s, bs in d.split_buildings.items()}
    seen_b: dict[str, set[str]] = {}
    for split, lst in eps.items():
        used = set()
        for ep in lst:
            ws = _windows(ep)
            blds = [w.building for w in ws]
            seen_b.setdefault(split, set()).update(blds)
            assert set(blds) == role_b[split]                                   # every building of the role
            counts = np.unique(blds, return_counts=True)[1]
            assert len(set(counts)) == 1                                        # equal windows per building
            for w in ws:
                assert w.id not in used                                         # window-disjoint across episodes
                used.add(w.id)
            for b in set(blds):                                                 # conflict-free within a building
                own = sorted((np.datetime64(w.target_start, "h").astype("int64") for w in ws if w.building == b))
                # LEAK-4: the later context starts >= MIN_DELAY_H after the earlier target ended (168-h context + 24-h target)
                assert all(y - x >= 24 + MIN_DELAY_H + 168 for x, y in zip(own, own[1:])), (ep.id, b, own)
    names = list(seen_b)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not seen_b[a] & seen_b[b], (a, b)
    assert "MT_070" not in set().union(*seen_b.values())


def test_capacity_and_pool_exhaustion():
    a = _ADAPTER
    assert len(a.build_episodes("src", 20, 5)) == 20 and a.build_episodes("src", 9, 5)[8].lineage["items_reused_from_earlier_episodes"]
    for split in ("val", "id", "ood"):
        cap = a.capacity(split)
        assert len(a.build_episodes(split, cap, 3)) == cap
        with pytest.raises(PoolExhausted):
            a.build_episodes(split, cap + 1, 3)
    # a larger episode holds two windows per building of the id role (22 items) and still packs disjointly
    big = a.build_episodes("id", a.capacity("id", 24), 3, items_per_episode=24)
    assert {e.n_items for e in big} == {22}
    assert len({i for e in big for i in e.lineage["item_ids"]}) == 22 * len(big)
    small = a.build_episodes("id", 2, 3, items_per_episode=4)
    assert {e.n_items for e in small} == {11}                                   # at least one window per building


def test_pack_groups_respects_conflicts():
    def item(k, start):
        return PoolItem(f"w{k}", "b", ("residential", start, start + 168))
    items = [item(k, 24 * k) for k in range(8)]                                 # consecutive days: all within 7 days
    assert pack_groups(items, 2, 1, make_rng("t", 1), min_delay_h=0) is None
    days = [item(k, 24 * k) for k in range(16)]                                 # min_delay_h = 0: pairs 8+ days apart
    got = pack_groups(days, 2, 4, make_rng("t", 1), min_delay_h=0)
    assert got is not None and len({it.id for g in got for it in g}) == 8
    assert all(abs(int(g[0].id[1:]) - int(g[1].id[1:])) > 7 for g in got)
    spread = [item(k, 24 * 8 * k) for k in range(8)]                            # 8 days apart: no conflicts at delay 0
    got = pack_groups(spread, 2, 4, make_rng("t", 2), min_delay_h=0)
    assert got is not None and len({it.id for g in got for it in g}) == 8


def test_min_delay_between_a_target_and_a_later_context():
    """LEAK-4: a later window of the same building must start its context >= MIN_DELAY_H after the earlier target."""
    assert MIN_DELAY_H == 144

    def item(k, target_day):
        t = 24 * target_day
        return PoolItem(f"w{k}", "b", ("residential", t - 168, t))
    a = item(0, 100)                                                            # target hours [2400, 2424)
    for gap_days, clash in ((7, True), (8, True), (13, True), (14, False), (15, False), (30, False)):
        b = item(1, 100 + gap_days)
        assert _conflict(a, b) is clash is _conflict(b, a), gap_days            # symmetric; 14 days = 24 + 144 + 168 h
        assert _conflict(a, b, 0) is (gap_days <= 7), gap_days                  # the old rule: only overlap with the context
    assert min_delay_observed([a, item(1, 114), item(2, 130)]) == 144
    assert min_delay_observed([a]) is None
    # eight windows 15 days apart pack into four disjoint pairs at the default delay; windows 2 days apart cannot
    # (pairs need >= 14 days = 7 steps, but the old 8-day rule is met by four disjoint pairs 4 steps apart)
    ok = [item(k, 15 * k) for k in range(8)]
    assert pack_groups(ok, 2, 4, make_rng("t", 3)) is not None
    tight = [item(k, 2 * k) for k in range(8)]
    assert pack_groups(tight, 2, 4, make_rng("t", 3)) is None
    assert pack_groups(tight, 2, 4, make_rng("t", 3), min_delay_h=0) is not None


def test_episode_lineage_records_the_delay(eps):
    for split, lst in eps.items():
        for ep in lst:
            lin = ep.lineage
            assert lin["min_window_delay_h"] == MIN_DELAY_H
            obs = lin["min_window_delay_observed_h"]
            if lin["windows_per_building"] >= 2:
                assert obs is not None and obs >= MIN_DELAY_H, (ep.id, obs)
            else:
                assert obs is None                                              # one window per building: no pair
    assert f"{MIN_DELAY_H} h" in eps["val"][0].lineage["split_rule"]


def test_no_label_leak_and_history_before_contexts(eps):
    d = _ADAPTER.data()
    for split, lst in eps.items():
        for ep in lst[:2]:
            outs = check_tool_schemas(ep)
            tgt = _targets(ep)
            for name, out in outs.items():
                for v in out.values():
                    if isinstance(v, np.ndarray) and v.dtype.kind == "f":
                        for row in tgt:
                            assert not contains_run(v, row), f"target leaked in {split}/{name}"
            ctx = outs["load_eval_inputs"]["context"]
            ws = _windows(ep)
            for i, w in enumerate(ws):
                assert np.array_equal(ctx[i], w.context)
                assert outs["load_eval_inputs"]["target_start"][i] == w.target_start
            hist = outs["load_history"]
            for j, b in enumerate(hist["building_id"]):
                bb = d.buildings[b]
                assert np.array_equal(hist["load"][j], bb.history) and hist["category"][j] == bb.category
                end = np.datetime64(hist["history_start"][j], "h") + hist["load"].shape[1]
                first_ctx = min(np.datetime64(w.context_start, "h") for w in ws if w.building == b)
                assert end <= first_ctx                                         # history precedes every context
            dev = outs["load_dev"]
            assert set(dev["building_id"]) == set(hist["building_id"])
            for i, b in enumerate(dev["building_id"]):
                assert np.datetime64(dev["target_start"][i], "h") < min(
                    np.datetime64(w.context_start, "h") for w in ws if w.building == b)


def test_evaluator_reference_oracle_malformed(eps):
    for split in ("src", "val", "id", "ood"):
        ep = eps[split][0]
        ctx = ep.tool("load_eval_inputs").fn({}, {})["context"]
        n = ep.n_items
        check_eval_paths(ep, persistence(ctx), _targets(ep),
                         [np.zeros((n, 23)), np.full((n, 24), np.nan), "text", {"y": 1}], _ADAPTER,
                         violating_ys=[-np.ones((n, 24)), np.ones((n, 24)) * 5000.0, persistence(ctx) * 1000.0])


def test_reference_never_accepted_and_profile_can_be(eps):
    """previous-day persistence is the reference (never accepted); a 7-day hourly profile is a sensible attempt."""
    wins = 0
    for split in ("val", "id", "ood"):
        for ep in eps[split]:
            ctx = ep.tool("load_eval_inputs").fn({}, {})["context"]
            assert not ep.evaluate(persistence(ctx), None).accepted
            prof = ctx.reshape(len(ctx), 7, 24).mean(axis=1)
            wins += bool(ep.evaluate(prof, None).accepted)
    assert 0 < wins < 10


def test_avg7_is_reported_as_a_metric_and_matches_the_definition(eps):
    ep = eps["id"][0]
    ctx = ep.tool("load_eval_inputs").fn({}, {})["context"]
    n = len(ctx)
    avg = average_persistence(ctx)
    assert avg.shape == (n, 24)
    for h in (0, 5, 23):
        assert avg[0, h] == pytest.approx(np.mean([ctx[0, 24 * d + h] for d in range(7)]))
    r = ep.evaluate(avg, None)
    assert r.metrics["avg7_balanced_nrmse_pct"] == pytest.approx(r.primary)
    assert ep.evaluate(persistence(ctx), None).metrics["avg7_balanced_nrmse_pct"] == pytest.approx(r.primary)


def test_score_dev_and_budget(eps):
    ep = eps["id"][0]
    dev = ep.tool("load_dev").fn({}, {})
    n_dev = len(dev["building_id"])
    assert n_dev == 4 * len(set(dev["building_id"]))                    # four dev windows per building
    check_score_dev(ep, persistence(dev["context"]), np.zeros((3, 24)))
    good = ep.tool("score_dev").fn({"pred": persistence(dev["context"])}, {})
    assert good["score"] == pytest.approx(good["report"]["reference_score"])
    assert ep.budget.max_llm_items == 2 * ep.n_items


def test_metric_is_category_median_of_per_building_cvrmse():
    rng = np.random.default_rng(0)
    y = rng.uniform(1, 3, (6, 24))
    yhat = y * rng.uniform(0.5, 1.5, (6, 24))
    blds = ["a", "a", "b", "b", "c", "c"]
    cats = {"a": "residential", "b": "residential", "c": "commercial"}
    per = {b: 100 * np.sqrt(np.mean((y[i:i + 2] - yhat[i:i + 2]) ** 2)) / np.mean(y[i:i + 2]) for b, i in (("a", 0), ("b", 2), ("c", 4))}
    want = 0.5 * (np.median([per["a"], per["b"]]) + per["c"])
    got, per_got, med = balanced_nrmse(y, yhat, blds, cats)
    assert got == pytest.approx(want) and per_got == pytest.approx(per)
    assert med["residential"] == pytest.approx(np.median([per["a"], per["b"]]))
    assert nrmse_pct(y[:2], yhat[:2]) == pytest.approx(per["a"])          # pooled over the building's hours, not per window
    assert balanced_score({"a": 10.0, "b": 30.0}, {"a": "residential", "b": "residential"})[0] == pytest.approx(20.0)
    with pytest.raises(ValueError):
        nrmse_pct(np.zeros((1, 24)), np.ones((1, 24)))


def test_pooled_metric_matches_balanced_definition(eps):
    d = _ADAPTER.data()
    cats = {b: bb.category for b, bb in d.buildings.items()}
    dets, yt, yp, bl = [], [], [], []
    for ep in eps["id"]:
        ctx = ep.tool("load_eval_inputs").fn({}, {})["context"]
        pred = persistence(ctx) * 0.9
        dets.append(ep.evaluate(pred, None).details)
        yt.append(_targets(ep))
        yp.append(pred)
        bl += [w.building for w in _windows(ep)]
    want, _, _ = balanced_nrmse(np.concatenate(yt), np.concatenate(yp), bl, cats)
    assert _ADAPTER.pooled_metric(dets) == pytest.approx(want, rel=1e-9)


def test_unavailable_reports_reason(tmp_path):
    ok, why = Adapter(data_root=tmp_path).available()
    assert not ok and "missing" in why and why.startswith("FoR33")


def test_tampered_role_file_is_rejected(tmp_path):
    dl = _ADAPTER.delivery()
    root = dl.delivery_root
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / dl.base.parent.name).symlink_to(dl.base.parent, target_is_directory=True)
    for src in [dl.catalog_path, *dl.role_files.values()]:
        dst = tmp_path / src.relative_to(root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
    fake = Adapter(data_root=tmp_path / "datasets")
    ok, why = fake.available()
    assert ok, why                                                      # a faithful copy of the delivery is accepted
    with (tmp_path / dl.role_files["id"].relative_to(root)).open("a") as fh:
        fh.write("\n")
    ok, why = Adapter(data_root=tmp_path / "datasets").available()
    assert not ok and "sha256" in why


def test_objective_documents_the_domain_library(eps):
    ep = eps["val"][0]
    assert "scilib.loadforecast" in ep.objective and "forecast_candidates" in ep.objective and "history_windows" in ep.objective
