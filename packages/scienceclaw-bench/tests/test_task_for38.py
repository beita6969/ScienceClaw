"""FoR38 World Bank WDI adapter: economy-disjoint splits, no post-origin values visible, evaluator (sMAPE)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_task_forecast_common import (build_all, check_determinism, check_disjoint, check_eval_paths,  # noqa: E402
                                       check_score_dev, check_structure, check_tool_schemas, contains_run)

from scienceclaw.bench.tasks.for38_worldbank import (IID_REGIONS, OOD_REGIONS, ORIGIN, Adapter, _yi,  # noqa: E402
                                                     damped_trend, history_backtest_forecast, naive)
import scienceclaw.bench.tasks.for38_worldbank as for38  # noqa: E402

_ADAPTER = Adapter()
OK, WHY = _ADAPTER.available()
pytestmark = pytest.mark.skipif(not OK, reason=f"FoR38 data unavailable: {WHY}")


@pytest.fixture(scope="module")
def eps():
    return build_all(_ADAPTER)


def _targets(ep) -> np.ndarray:
    d = _ADAPTER.data()
    return np.stack([d.values[d.items[i][1]][d.items[i][0]][_yi(ORIGIN) + 1:_yi(ORIGIN) + 5] for i in ep.lineage["item_ids"]])


def _period_cols(panel) -> list[str]:
    return [c for c in panel.columns if re.fullmatch(r"t(0|[+-]\d+)", c)]


def _strings(obj) -> list[str]:
    """Every string reachable in a tool output / public view (dict keys and values, list items, DataFrame
    columns and object cells)."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for k, v in obj.items() for s in _strings(k) + _strings(v)]
    if isinstance(obj, (list, tuple)):
        return [s for v in obj for s in _strings(v)]
    if isinstance(obj, pd.DataFrame):
        out = [str(c) for c in obj.columns]
        for c in obj.columns:
            if obj[c].dtype == object:
                out += [str(v) for v in obj[c].tolist()]
        return out
    return []


def test_structure_disjoint_deterministic(eps):
    check_structure(_ADAPTER, eps)
    check_disjoint(eps)
    check_determinism(Adapter, eps)
    _ADAPTER.verify_disjoint()
    d = _ADAPTER.data()
    econ = {s: {d.items[i][0] for e in lst for i in e.lineage["item_ids"]} for s, lst in eps.items()}
    for a in econ:
        for b in econ:
            if a < b:
                assert not econ[a] & econ[b], (a, b)
    assert all(d.meta[c]["region"] in OOD_REGIONS for c in econ["ood"])
    assert all(d.meta[c]["region"] in IID_REGIONS for s in ("src", "val", "id") for c in econ[s])


def test_no_post_origin_value_visible(eps):
    d = _ADAPTER.data()
    for split in ("src", "id", "ood"):
        ep = eps[split][0]
        outs = check_tool_schemas(ep)
        panel = outs["load_train"]["panel"]
        pcols = _period_cols(panel)
        assert pcols[-1] == "t0" and len(pcols) == ORIGIN - 1990 + 1
        assert outs["load_eval_inputs"]["history"].shape[1] == ORIGIN - 1990 + 1
        assert outs["load_eval_inputs"]["time_offsets"][-1] == 0
        tgt = _targets(ep)
        num = panel[pcols].to_numpy(float)
        for arr in (num, outs["load_eval_inputs"]["history"], outs["load_eval_inputs"]["covariates"],
                    outs["load_dev"]["history"]):
            for row in tgt:
                assert not contains_run(arr, row)
        # the panel holds only IID-region economies
        assert set(panel["region"]) <= set(IID_REGIONS)
        assert len(d.items) > 300


def test_evaluator_reference_oracle_malformed(eps):
    for split in ("val", "id", "ood"):
        ep = eps[split][0]
        inp = ep.tool("load_eval_inputs").fn({}, {})
        ref = np.stack([naive(h, 4) for h in inp["history"]])
        n = ep.n_items
        rate_kind = _ADAPTER.data().series_ids["SL.UEM.TOTL.ZS"]
        unemp = np.array([i == rate_kind for i in inp["indicator"]])
        bad_unemp = ref.copy()
        bad_unemp[unemp] = 150.0
        viol = [-ref] + ([bad_unemp] if unemp.any() else [])
        check_eval_paths(ep, ref, _targets(ep), [np.zeros((n, 5)), np.full((n, 4), np.nan), "not an array"],
                         _ADAPTER, violating_ys=viol)


def test_per_indicator_metrics_and_pooled(eps):
    ep = eps["id"][0]
    inp = ep.tool("load_eval_inputs").fn({}, {})
    r = ep.evaluate(np.stack([naive(h, 4) for h in inp["history"]]) * 1.01, None)
    assert {"mean_smape_pct", "reference_mean_smape_pct"} <= set(r.metrics)
    assert any(k.startswith("smape_") for k in r.metrics)
    dets = [e.evaluate(np.stack([naive(h, 4) for h in e.tool("load_eval_inputs").fn({}, {})["history"]]), None).details
            for e in eps["id"]]
    want = np.mean([v for dd in dets for v in dd["pooled_payload"]["smape"]])
    assert _ADAPTER.pooled_metric(dets) == pytest.approx(want)


def test_score_dev(eps):
    ep = eps["src"][0]
    dev = ep.tool("load_dev").fn({}, {})
    check_score_dev(ep, np.stack([naive(h, 4) for h in dev["history"]]), np.zeros((16, 3)))


def test_damped_trend_and_pooled_reference_diagnostics_are_history_only():
    """The stronger reference is deterministic and cannot alter the official pooled primary."""
    hist = np.asarray([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    pred = damped_trend(hist, 4)
    assert pred.shape == (4,) and np.all(np.isfinite(pred))
    assert np.all(np.diff(pred) > 0)
    selected = history_backtest_forecast(hist, 4)
    assert selected.shape == (4,) and np.all(np.isfinite(selected))
    adapter = Adapter()
    payloads = [
        {"smape": [10.0, 12.0], "reference_smape": [14.0, 15.0], "strong_smape": [11.0, 13.0]},
        {"smape": [8.0], "reference_smape": [9.0], "strong_smape": [7.0]},
    ]
    d = adapter.pooled_diagnostics(payloads)
    assert d["n_episodes"] == 2 and d["n_items"] == 3
    assert d["pooled_primary"] == pytest.approx(10.0)
    assert d["pooled_reference"] == pytest.approx(38.0 / 3.0)
    assert d["pooled_strong_reference"] == pytest.approx(31.0 / 3.0)
    assert d["strong_reference_gain_pct"] == pytest.approx(7.0 / 3.0)


def test_full_split_reference_reports_complete_pool_without_changing_primary(eps):
    """Full-pool references give a comparable audit while the official episode scorer remains unchanged."""
    data = _ADAPTER.data()
    for split in ("src", "val", "id", "ood"):
        out = _ADAPTER.full_split_reference(split)
        assert out["diagnostic_only"] is True
        assert out["n_items"] == len(data.pools[split])
        assert out["n_items"] > 0
        for key in ("reference_smape", "damped_trend_smape", "history_backtest_smape",
                    "damped_trend_gain_pct", "history_backtest_gain_pct"):
            assert np.isfinite(out[key])
        # The complete-pool audit is independent of 16-item episode construction and the evaluator's primary.
        ep = _ADAPTER.build_episodes(split, 1, 0)[0]
        pred = np.stack([naive(h, 4) for h in ep.tool("load_eval_inputs").fn({}, {})["history"]])
        assert "reference_mean_smape_pct" in ep.evaluate(pred, None).metrics


def test_fixed_macro_route_is_label_free_and_submit_ready(eps):
    """The specialist route consumes only the two public tool payloads and returns finite y."""
    ep = eps["id"][0]
    train = ep.tool("load_train").fn({}, {})
    eval_inputs = ep.tool("load_eval_inputs").fn({}, {})
    out = ep.tool("macro_fixed_predict").fn({**train, **eval_inputs}, {})
    assert out["y"].shape == (ep.n_items, 4)
    assert np.isfinite(out["y"]).all() and (out["y"] >= 0).all()
    assert out["provenance"]["method"] == "scilib.macro.fit_predict"
    assert out["provenance"]["label_source"] == "none"
    assert out["provenance"]["n_backtest"] == 0


def test_fixed_chronos_route_is_label_free_and_submit_ready(eps, monkeypatch):
    """The frozen pretrained route consumes visible inputs and clips to task units."""
    ep = eps["id"][0]
    train = ep.tool("load_train").fn({}, {})
    eval_inputs = ep.tool("load_eval_inputs").fn({}, {})

    def fake_forecast(history, horizon, quantiles, model, context_length):
        assert model == "chronos_2" and context_length == 32 and quantiles == (0.5,)
        return np.repeat(history[:, -1:, None], horizon, axis=1)

    monkeypatch.setattr(for38.tsfm, "forecast", fake_forecast)
    out = ep.tool("chronos_fixed_predict").fn({**train, **eval_inputs}, {})
    assert out["y"].shape == (ep.n_items, 4)
    assert np.isfinite(out["y"]).all() and (out["y"] >= 0).all()
    assert out["provenance"]["method"] == "scilib.tsfm.forecast"
    assert out["provenance"]["label_source"] == "none"


_BANNED_WORDS = ("World Bank", "WDI", "World Development", "GDP", "unemployment", "Unemployment", "inflation",
                 "Inflation", "consumer price", "labour force", "labor force", "per capita", "constant 2015",
                 "NY.GDP", "SL.UEM", "FP.CPI", "naive", "no-change", "random walk", "Success criterion")


def _visible_strings(ep, outs) -> list[str]:
    return _strings(ep.public_view()) + [s for o in outs.values() for s in _strings(o)]


def test_visible_view_is_anonymised(eps):
    """LEAK-6: the policy sees no economy name/code, no indicator identity, no source name and no calendar year."""
    d = _ADAPTER.data()
    names = sorted({m["name"] for m in d.meta.values()})
    codes = sorted(d.meta)
    year_re = re.compile(r"(?<![\d.])(19[89]\d|20[0-3]\d)(?![\d])")
    for split in ("src", "val", "id", "ood"):
        ep = eps[split][0]
        outs = check_tool_schemas(ep)
        strings = _visible_strings(ep, outs)
        blob = "\n".join(strings)
        for w in _BANNED_WORDS:
            assert w not in blob, (split, w)
        for code in codes:
            assert not re.search(rf"\b{re.escape(code)}\b", blob), (split, code)
        no_regions = blob
        for reg in IID_REGIONS + OOD_REGIONS:     # the (coarse, deliberate) region labels may mention countries
            no_regions = no_regions.replace(reg, " ")
        for nm in names:
            assert not re.search(rf"(?<![\w]){re.escape(nm)}(?![\w])", no_regions), (split, nm)
        assert not year_re.search(blob), (split, year_re.search(blob).group(0))
        # the economy ids are the opaque ones, one per item
        ids = outs["load_eval_inputs"]["economy_id"]
        assert len(ids) == ep.n_items and all(re.fullmatch(r"E\d{3}", i) for i in ids)
        assert set(outs["load_eval_inputs"]["indicator"]) <= set(d.series_ids.values())
        assert not {"country_code", "country_name", "years", "target_years"} & set(outs["load_eval_inputs"])
        assert "country_code" not in outs["load_train"]["panel"].columns
        assert outs["load_eval_inputs"]["time_offsets"] == list(range(-(ORIGIN - 1990), 1))
        assert outs["load_eval_inputs"]["target_offsets"] == [1, 2, 3, 4]
        assert outs["load_dev"]["time_offsets"][-1] == 0 and outs["load_dev"]["target_offsets"] == [1, 2, 3, 4]
        # constraint / required-output texts only speak about the opaque kinds
        assert set(d.series_ids[i] for i in ("NY.GDP.PCAP.KD", "SL.UEM.TOTL.ZS")) <= set(
            re.findall(r"\bT\d\b", " ".join(c for c in ep.public_view()["constraints"])))
    # hidden side unchanged: lineage / payloads still carry the real item ids
    ep = eps["id"][0]
    assert all(i.startswith("WDI:") for i in ep.lineage["item_ids"])
    assert ep.lineage["anonymised_view"]["series_kinds"] == d.series_ids


def test_anonymised_ids_consistent_across_tools_and_panel(eps):
    d = _ADAPTER.data()
    for split in ("src", "id"):
        ep = eps[split][0]
        inp = ep.tool("load_eval_inputs").fn({}, {})
        dev = ep.tool("load_dev").fn({}, {})
        train = ep.tool("load_train").fn({}, {})
        assert dev["economy_id"] == inp["economy_id"] and dev["indicator"] == inp["indicator"]
        # the dev history is the prefix of the evaluation history (same periods, shifted labels only)
        dh = dev["history"]
        np.testing.assert_array_equal(dh, inp["history"][:, :dh.shape[1]])
        assert set(train["indicator_kinds"]) == set(d.series_ids.values())
        panel = train["panel"].set_index(["economy_id", "indicator"])
        pcols = _period_cols(train["panel"])
        for j, iid in enumerate(ep.lineage["item_ids"]):
            iso, ind = d.items[iid]
            assert inp["economy_id"][j] == d.econ_ids[iso] and inp["indicator"][j] == d.series_ids[ind]
            row = panel.loc[(d.econ_ids[iso], d.series_ids[ind]), pcols].to_numpy(float)
            np.testing.assert_array_equal(row, inp["history"][j])                  # NaN-aware equality
            for c, code in enumerate(inp["covariate_indicators"]):
                cov_row = panel.loc[(d.econ_ids[iso], code), pcols].to_numpy(float)
                np.testing.assert_array_equal(cov_row, inp["covariates"][j, c])
    # OOD economies are not in the visible panel
    ood = eps["ood"][0]
    ood_ids = set(ood.tool("load_eval_inputs").fn({}, {})["economy_id"])
    assert not ood_ids & set(ood.tool("load_train").fn({}, {})["panel"]["economy_id"])


def test_anonymisation_is_deterministic_injective_and_unordered():
    d = _ADAPTER.data()
    same = Adapter().data()
    assert same.econ_ids == d.econ_ids and same.series_ids == d.series_ids
    assert len(set(d.econ_ids.values())) == len(d.econ_ids) == len(d.meta)
    assert len(set(d.series_ids.values())) == 4 and set(d.series_ids.values()) == {"T1", "T2", "C1", "C2"}
    assert all(d.series_ids[i].startswith("T") for i in ("NY.GDP.PCAP.KD", "SL.UEM.TOTL.ZS"))
    # ids are a hash permutation, not alphabetical: rank correlation with the ISO3 order is ~0
    iso = sorted(d.meta)
    rank = np.array([int(d.econ_ids[c][1:]) for c in iso], float)
    assert abs(np.corrcoef(np.arange(len(iso)), rank)[0, 1]) < 0.3
    other = Adapter(partition_seed=7).data()
    assert other.econ_ids != d.econ_ids


def test_evaluation_unchanged_by_anonymisation(eps):
    """Hidden scoring still works on the real series: reference score equals the naive sMAPE recomputed from the
    raw data, and the pooled payload carries the real item ids."""
    from scienceclaw.bench.tasks._forecast_common import smape

    d = _ADAPTER.data()
    ep = eps["id"][0]
    ref = np.mean([smape(_targets(ep)[j], naive(d.values[d.items[i][1]][d.items[i][0]][:_yi(ORIGIN) + 1], 4))
                   for j, i in enumerate(ep.lineage["item_ids"])])
    r = ep.evaluate(np.stack([naive(h, 4) for h in ep.tool("load_eval_inputs").fn({}, {})["history"]]), None)
    assert r.details["reference"] == pytest.approx(ref)
    assert r.details["pooled_payload"]["ids"] == ep.lineage["item_ids"]


def test_unavailable_reports_reason(tmp_path):
    ok, why = Adapter(data_root=tmp_path).available()
    assert not ok and "missing" in why


def test_objective_documents_the_domain_library(eps):
    for split in ("src", "ood"):
        ep = eps[split][0]
        assert "scilib.macro" in ep.objective and "fit_predict" in ep.objective
