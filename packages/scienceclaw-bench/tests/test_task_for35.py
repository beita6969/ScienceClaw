"""FoR35 Monash tourism adapter: splits, determinism, leakage, evaluator (MASE), OOD auto-detection."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_task_forecast_common import (build_all, check_determinism, check_disjoint, check_eval_paths,  # noqa: E402
                                       check_score_dev, check_structure, check_tool_schemas, contains_run)

from scienceclaw.bench.tasks import _forecast_common as fc  # noqa: E402
from scienceclaw.bench.tasks.for35_tourism import Adapter, read_tsf, snaive  # noqa: E402

_ADAPTER = Adapter()
OK, WHY = _ADAPTER.available()
pytestmark = pytest.mark.skipif(not OK, reason=f"FoR35 data unavailable: {WHY}")


@pytest.fixture(scope="module")
def eps():
    return build_all(_ADAPTER)


def _targets(ep) -> np.ndarray:
    return np.stack([_ADAPTER.data().series[i].target for i in ep.lineage["item_ids"]])


def test_structure_disjoint_deterministic(eps):
    check_structure(_ADAPTER, eps)
    check_disjoint(eps)
    check_determinism(Adapter, eps)
    _ADAPTER.verify_disjoint()
    d = _ADAPTER.data()
    assert len(d.series) == 366 and len(d.iid_ids) == 264
    for e in eps["ood"]:
        assert all(d.series[i].start[:4] != "1980" for i in e.lineage["item_ids"])
        assert e.lineage["ood_kind"] in ("proxy_within_dataset", "cross_dataset")


def test_snaive_reproduces_monash_benchmark():
    """Monash archive: SNaive mean MASE on tourism monthly = 1.631 (Godahewa et al. 2021, Table)."""
    d = _ADAPTER.data()
    vals = [fc.mase(s.target, snaive(s.history, 24, 12), s.history, 12)
            for s in d.series.values() if s.id.startswith("tourism_monthly")]
    assert np.mean(vals) == pytest.approx(1.631, abs=5e-4)


def test_no_label_leak(eps):
    for split, lst in eps.items():
        for ep in lst[:2]:
            outs = check_tool_schemas(ep)
            tgt = _targets(ep)
            for tool, out in outs.items():
                for port, v in out.items():
                    arrays = v if isinstance(v, list) and v and isinstance(v[0], np.ndarray) else [v]
                    for a in arrays:
                        if isinstance(a, np.ndarray) and a.dtype.kind == "f":
                            for row in tgt:
                                assert not contains_run(a, row), f"target leaked via {tool}.{port} ({split})"


def test_evaluator_reference_oracle_malformed(eps):
    for split in ("val", "id", "ood"):
        ep = eps[split][0]
        inp = ep.tool("load_eval_inputs").fn({}, {})
        ref = np.stack([snaive(h, inp["horizon"], inp["period"]) for h in inp["history"]])
        n, H = ref.shape
        check_eval_paths(ep, ref, _targets(ep), [np.zeros((n, H + 1)), np.full((n, H), np.inf), [[1, 2], [3]]],
                         _ADAPTER, violating_ys=[-ref, ref * 100.0])


def test_score_dev(eps):
    ep = eps["src"][0]
    dev = ep.tool("load_dev").fn({}, {})
    check_score_dev(ep, np.stack([snaive(h, 24, 12) for h in dev["history"]]), np.zeros((16, 12)))


def _write_tsf(path: Path, freq: str, horizon: int, series: list[tuple[str, str, np.ndarray]]) -> None:
    lines = ["# synthetic test fixture", "@relation Test", "@attribute series_name string",
             "@attribute start_timestamp date", f"@frequency {freq}", f"@horizon {horizon}", "@missing false",
             "@equallength false", "@data"]
    lines += [f"{n}:{s} 00-00-00:" + ",".join(f"{v:.4f}" for v in vals) for n, s, vals in series]
    path.write_text("\n".join(lines) + "\n", encoding="latin-1")


def test_quarterly_ood_autodetected(tmp_path):
    """With a tourism quarterly file present, OOD switches to it (cross-dataset, horizon 8, period 4)."""
    src = _ADAPTER._monthly_path()
    mdir = tmp_path / "for35-monash-tourism-monthly" / "data"
    mdir.mkdir(parents=True)
    (mdir / src.name).write_bytes(src.read_bytes())
    rng = np.random.default_rng(0)
    qs = [(f"Q{i}", "1990-01-01", 100 + 10 * np.tile([1, 3, 2, 0], 10) + rng.normal(0, 1, 40).cumsum())
          for i in range(40)]
    _write_tsf(tmp_path / "for35-monash-tourism-monthly" / "data" / "tourism_quarterly_dataset.tsf", "quarterly", 8, qs)
    a = Adapter(data_root=tmp_path)
    assert a.available()[0] and a.data().ood_kind == "cross_dataset"
    ep = a.build_episodes("ood", 1, 3)[0]
    assert ep.required_output.shape == (16, 8) and ep.lineage["ood_kind"] == "cross_dataset"
    inp = ep.tool("load_eval_inputs").fn({}, {})
    assert inp["horizon"] == 8 and inp["period"] == 4
    r = ep.evaluate(np.stack([snaive(h, 8, 4) for h in inp["history"]]), None)
    assert r.primary == pytest.approx(r.details["reference"])
    assert len(read_tsf(tmp_path / "for35-monash-tourism-monthly" / "data" / "tourism_quarterly_dataset.tsf", "q")) == 40


def test_unavailable_reports_reason(tmp_path):
    ok, why = Adapter(data_root=tmp_path).available()
    assert not ok and "missing" in why


def test_objective_documents_the_domain_library(eps):
    for split in ("src", "ood"):
        ep = eps[split][0]
        assert "scilib.forecast" in ep.objective and "fit_predict" in ep.objective and "backtest_panel" in ep.objective


def test_tool_text_separates_deliverable_inputs_from_dev_inputs(eps):
    """The dev backtest histories end H values before the eval histories; the tool text says which tool feeds the deliverable
    and that load_train holds the values dropped from the dev histories (factual wording only)."""
    for split in ("src", "ood"):
        ep = eps[split][0]
        assert [t.name for t in ep.tools][0] == "load_eval_inputs"
        assert "deliverable y is a forecast of these histories" in ep.tool("load_eval_inputs").description
        assert "not the inputs of the deliverable" in ep.tool("load_dev").description
        assert "appear in load_train" in ep.tool("load_dev").description
        assert "load_dev histories" in ep.tool("score_dev").description
        dev, ev = ep.tool("load_dev").fn({}, {}), ep.tool("load_eval_inputs").fn({}, {})
        H = ev["horizon"]
        for d, e in zip(dev["history"], ev["history"]):
            assert d.size == e.size - H and np.array_equal(d, e[:-H])
