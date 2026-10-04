"""FoR41 NEON aquatics adapter: site/time splits, global label masking, evaluator (CRPS of normal forecasts)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_task_forecast_common import (build_all, check_determinism, check_disjoint, check_eval_paths,  # noqa: E402
                                       check_score_dev, check_structure, check_tool_schemas)

from scienceclaw.bench.tasks.for41_neon import (H, KEYS, SITE_TYPE, WINDOWS, Adapter, climatology,  # noqa: E402
                                                crps_sums, score_from_sums)

_ADAPTER = Adapter()
OK, WHY = _ADAPTER.available()
pytestmark = pytest.mark.skipif(not OK, reason=f"FoR41 data unavailable: {WHY}")


@pytest.fixture(scope="module")
def eps():
    return build_all(_ADAPTER)


def _obs(ep) -> np.ndarray:
    d = _ADAPTER.data()
    return np.stack([d.raw[d.items[i][0]][d.items[i][1] + 1:d.items[i][1] + 1 + H, :2] for i in ep.lineage["item_ids"]])


def _clim_pred(ep, inp: dict) -> dict:
    hist = inp["history"]
    out = {k: np.empty((ep.n_items, H)) for k in KEYS}
    for i, ref in enumerate(inp["reference_date"]):
        t0 = np.datetime64(ref, "D")
        hd = t0 - np.arange(hist.shape[1] - 1, -1, -1).astype("timedelta64[D]")
        td = t0 + np.arange(1, H + 1).astype("timedelta64[D]")
        for vi, v in enumerate(("oxygen", "temperature")):
            out[f"{v}_mu"][i], out[f"{v}_sigma"][i] = climatology(hist[i, :, vi], hd, td)
    return out


def test_structure_disjoint_deterministic(eps):
    check_structure(_ADAPTER, eps)
    check_disjoint(eps)
    check_determinism(Adapter, eps)
    _ADAPTER.verify_disjoint()
    d = _ADAPTER.data()
    for split, lst in eps.items():
        lo, hi = (np.datetime64(x, "D") for x in WINDOWS[split])
        for e in lst:
            sites = [d.items[i][0] for i in e.lineage["item_ids"]]
            for i in e.lineage["item_ids"]:
                s, ti = d.items[i]
                assert lo <= d.days[ti] <= hi
                assert (SITE_TYPE[s] == "wadeable stream") == (split != "ood")
            if split == "src":
                assert len(set(sites)) == len(sites)      # distinct sites (large src pool, 24 stream sites)


def test_eval_targets_masked_everywhere(eps):
    """No val/id/ood target observation is visible in any history of any episode (incl. src and dev inputs)."""
    d = _ADAPTER.data()
    for split in ("val", "id", "ood"):
        for iid in d.pools[split]:
            s, ti = d.items[iid]
            assert np.all(np.isnan(d.visible[s][ti + 1:ti + 1 + H]))
    for split, lst in eps.items():
        for ep in lst[:2]:
            outs = check_tool_schemas(ep)
            for tool in ("load_eval_inputs", "load_dev"):
                hist = outs[tool]["history"]
                for j, iid in enumerate(ep.lineage["item_ids"]):
                    s, ti = d.items[iid]
                    t_end = ti if tool == "load_eval_inputs" else ti - H
                    lo = t_end - hist.shape[1] + 1
                    expect = d.visible[s][max(lo, 0):t_end + 1]
                    got = hist[j][hist.shape[1] - expect.shape[0]:]
                    assert np.all(np.isnan(hist[j][:hist.shape[1] - expect.shape[0]]))     # padding before data start
                    # identical to the globally masked series except for extra per-episode masking
                    both = np.isfinite(got)
                    np.testing.assert_array_equal(got[both], expect[both])
                    assert not np.any(np.isfinite(got) & np.isnan(expect))
            # within the episode, no item's target observations appear in any item's history / dev data
            for j, iid in enumerate(ep.lineage["item_ids"]):
                s_j, t_j = d.items[iid]
                for kk, other in enumerate(ep.lineage["item_ids"]):
                    s_k, t_k = d.items[other]
                    if s_k != s_j:
                        continue
                    for tool, t_end in (("load_eval_inputs", t_k), ("load_dev", t_k - H)):
                        hist = outs[tool]["history"][kk]
                        lo = t_end - hist.shape[0] + 1
                        a, b = max(t_j + 1, lo), min(t_j + H, t_end)
                        if a <= b:
                            assert np.all(np.isnan(hist[a - lo:b - lo + 1])), (split, tool)
            # the item's own target days lie after its history end
            ref_dates = outs["load_eval_inputs"]["reference_date"]
            for j, iid in enumerate(ep.lineage["item_ids"]):
                assert str(d.days[d.items[iid][1]]) == ref_dates[j]


def test_evaluator_reference_oracle_malformed(eps):
    for split in ("val", "id", "ood"):
        ep = eps[split][0]
        inp = ep.tool("load_eval_inputs").fn({}, {})
        ref = _clim_pred(ep, inp)
        obs = _obs(ep)
        oracle = {k: v.copy() for k, v in ref.items()}
        for vi, v in enumerate(("oxygen", "temperature")):
            m = np.isfinite(obs[:, :, vi])
            oracle[f"{v}_mu"][m] = obs[:, :, vi][m]
            oracle[f"{v}_sigma"][m] = 0.05
        n = ep.n_items
        bad_sigma = {k: v.copy() for k, v in ref.items()}
        bad_sigma["oxygen_sigma"][0, 0] = -1.0
        kelvin = {k: v.copy() for k, v in ref.items()}
        kelvin["temperature_mu"] = kelvin["temperature_mu"] + 273.15
        check_eval_paths(ep, ref, oracle,
                         [np.zeros((n, H)), {k: ref[k] for k in KEYS[:3]}, {**ref, "oxygen_mu": np.zeros((n, H - 1))},
                          {**ref, "oxygen_mu": np.full((n, H), np.nan)}, bad_sigma],
                         _ADAPTER, violating_ys=[kelvin])


def test_crps_bookkeeping_and_pooling(eps):
    ep = eps["id"][0]
    inp = ep.tool("load_eval_inputs").fn({}, {})
    ref = _clim_pred(ep, inp)
    r = ep.evaluate(ref, None)
    s, per = score_from_sums(crps_sums(ref, _obs(ep)))
    assert r.primary == pytest.approx(s) and r.metrics["crps_oxygen_mgL"] == pytest.approx(per["oxygen"])
    assert r.metrics["crps_temperature_degC"] == pytest.approx(per["temperature"])
    dets = [e.evaluate(_clim_pred(e, e.tool("load_eval_inputs").fn({}, {})), None).details for e in eps["id"]]
    pv = _ADAPTER.pooled_per_variable(dets)
    assert _ADAPTER.pooled_metric(dets) == pytest.approx((pv["oxygen"] + pv["temperature"]) / 2)


def test_score_dev(eps):
    ep = eps["src"][0]
    dev = ep.tool("load_dev").fn({}, {})
    pred = _clim_pred(ep, dev)
    check_score_dev(ep, pred, {k: v[:, :5] for k, v in pred.items()})


def test_objective_documents_the_domain_library(eps):
    obj = eps["src"][0].objective
    assert "scilib.aquatics" in obj and "fit_predict" in obj and "data_report" in obj


def test_domain_library_forecasts_score_on_real_episodes(eps):
    from scilib import aquatics as aq
    ep = eps["src"][0]
    inp = ep.tool("load_eval_inputs").fn({}, {})
    dev = ep.tool("load_dev").fn({}, {})
    dev_out = ep.tool("score_dev").fn({"pred": aq.fit_predict(**dev)}, {})
    assert np.isfinite(dev_out["score"]) and dev_out["score"] < dev_out["report"]["reference_score"]
    res = ep._evaluate(aq.fit_predict(**inp), None)
    assert res.details["reference"] > res.primary > 0


def test_unavailable_reports_reason(tmp_path):
    ok, why = Adapter(data_root=tmp_path).available()
    assert not ok and "missing" in why
