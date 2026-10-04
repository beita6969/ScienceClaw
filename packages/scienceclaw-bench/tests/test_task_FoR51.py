"""FoR51 matbench_phonons adapter: splits, leave-element-out OOD, leakage, evaluator, constraints (real data)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from scienceclaw.bench.splits import split_seed
from scienceclaw.bench.tasks.for51_matbench import FEATURE_NAMES, OOD_ELEMENTS, Adapter, _group_period, parse_structure

SPLITS = {"src": 7, "val": 2, "id": 4, "ood": 4}


@pytest.fixture(scope="module")
def adapter():
    a = Adapter()
    ok, why = a.available()
    if not ok:
        pytest.skip(why)
    return a


@pytest.fixture(scope="module")
def plan(adapter):
    return {s: adapter.build_episodes(s, n, split_seed(20260928, "FoR51", s)) for s, n in SPLITS.items()}


def _tools(ep):
    return {t.name: t for t in ep.tools}


def test_group_period():
    assert _group_period(1) == (1, 1) and _group_period(2) == (18, 1) and _group_period(8) == (16, 2)
    assert _group_period(26) == (8, 4) and _group_period(57) == (3, 6) and _group_period(72) == (4, 6)
    assert _group_period(83) == (15, 6) and _group_period(55) == (1, 6)


def test_parse_structure_rejects_disorder():
    d = {"lattice": {"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
         "sites": [{"species": [{"element": "Na", "occu": 0.5}, {"element": "K", "occu": 0.5}], "abc": [0, 0, 0]}]}
    with pytest.raises(ValueError):
        parse_structure(d)


def test_pools_splits_disjoint_and_ood_is_leave_element_out(adapter, plan):
    data, parts = adapter._data.get(), adapter._parts.get()
    names = ["train", "dev", "src", "val", "id", "ood"]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not set(parts[a]) & set(parts[b]), (a, b)
    assert sum(len(parts[n]) for n in names) == len(data.ids) == 1265
    has = lambda i: any(s in OOD_ELEMENTS for s in data.structures[i]["species"])  # noqa: E731
    assert all(has(i) for i in parts["ood"]) and not any(has(i) for n in names[:-1] for i in parts[n])
    owner = {}
    for s, eps in plan.items():
        assert len(eps) == SPLITS[s]
        for ep in eps:
            assert ep.n_items == 16 and len(set(ep.lineage["item_ids"])) == 16
            for i in ep.lineage["item_ids"]:
                assert owner.setdefault(i, s) == s
                assert i in parts["ood" if s == "ood" else s]
            assert (ep.lineage["ood_kind"] == "proxy_within_dataset") == (s == "ood")


def test_deterministic(plan):
    again = Adapter().build_episodes("id", 4, split_seed(20260928, "FoR51", "id"))
    assert [e.lineage["item_ids"] for e in again] == [e.lineage["item_ids"] for e in plan["id"]]


def test_tools_visible_only(adapter, plan):
    data, parts = adapter._data.get(), adapter._parts.get()
    ep = plan["ood"][1]
    t = _tools(ep)
    ev = t["load_eval_inputs"].fn({}, {})
    assert set(ev) == {"structures"} and len(ev["structures"]) == 16
    for st in ev["structures"]:
        assert set(st) == {"formula", "lattice", "species", "frac_coords"}
    blob = json.dumps(ev)
    targets = [data.targets[i] for i in ep.lineage["item_ids"]]
    assert not any(repr(v) in blob for v in targets)                      # no hidden target values leak
    tr = t["load_train"].fn({}, {})
    assert len(tr["structures"]) == len(parts["train"]) == tr["targets"].shape[0]
    assert set(t["load_dev_inputs"].fn({}, {})) == {"dev_structures"}
    ev["structures"][0]["species"][0] = "Xx"                              # tools hand out copies
    assert t["load_eval_inputs"].fn({}, {})["structures"][0]["species"][0] != "Xx"
    X = t["featurize_structures"].fn({"structures": tr["structures"][:5]}, {})["X"]
    assert X.shape == (5, len(FEATURE_NAMES)) and np.isfinite(X).all()
    assert "fit_log_extra_trees" in t and "fit_hist_gradient_boosting" in t
    assert "fit_sevennet_mlip" in t
    n_dev = len(parts["dev"])
    out = t["score_dev"].fn({"dev_pred": np.full(n_dev, 500.0)}, {})
    assert out["dev_mae"] > 0 and out["dev_reference_mae"] > 0


def test_objective_wires_sevennet_prediction_to_submit(adapter, plan):
    objective = plan["id"][0].objective
    assert "`pred` output port" in objective and "submitted directly as y" in objective


def test_sevennet_tool_uses_frozen_features_and_visible_targets_only(adapter, plan, monkeypatch):
    ep = plan["id"][0]
    t = _tools(ep)["fit_sevennet_mlip"]
    tr = _tools(ep)["load_train"].fn({}, {})
    ev = _tools(ep)["load_eval_inputs"].fn({}, {})
    calls = []

    def fake_phonon_features(structures, **kwargs):
        calls.append((len(structures), kwargs))
        return {"X": np.ones((len(structures), 2), dtype=float), "names": ["f0", "f1"]}

    def fake_fit_predict(Xtr, ytr, Xev, return_info=True):
        assert Xtr.shape[0] == len(tr["structures"])
        assert Xev.shape[0] == len(ev["structures"])
        assert ytr.shape == tr["targets"].shape
        assert return_info is True
        return {"predictions": np.full(len(ev["structures"]), 300.0)}

    import scilib.matphonon_mlip as mlip
    import scilib.matphonon_regression as reg
    monkeypatch.setattr(mlip, "phonon_features", fake_phonon_features)
    monkeypatch.setattr(reg, "fit_predict", fake_fit_predict)
    out = t.fn({"train_structures": tr["structures"], "train_targets": tr["targets"],
                "eval_structures": ev["structures"]}, {})
    assert out["pred"].shape == (len(ev["structures"]),)
    assert out["provenance"]["model"] == "sevennet"
    assert out["provenance"]["pretrained"] is True
    assert calls == [(len(tr["structures"]), {"mesh": 8, "min_len": 7.0, "disp": 0.01, "model": "sevennet"}),
                     (len(ev["structures"]), {"mesh": 8, "min_len": 7.0, "disp": 0.01, "model": "sevennet"})]


def test_mf0_tool_selects_explicit_modal(adapter, plan, monkeypatch):
    ep = plan["id"][0]
    t = _tools(ep)["fit_sevennet_mf0_mlip"]
    tr = _tools(ep)["load_train"].fn({}, {})
    ev = _tools(ep)["load_eval_inputs"].fn({}, {})
    calls = []

    def fake_phonon_features(structures, **kwargs):
        calls.append(kwargs)
        return {"X": np.ones((len(structures), 2), dtype=float), "names": ["f0", "f1"]}

    def fake_fit_predict(Xtr, ytr, Xev, return_info=True):
        return {"predictions": np.full(len(Xev), 300.0)}

    import scilib.matphonon_mlip as mlip
    import scilib.matphonon_regression as reg
    monkeypatch.setattr(mlip, "phonon_features", fake_phonon_features)
    monkeypatch.setattr(reg, "fit_predict", fake_fit_predict)
    out = t.fn({"train_structures": tr["structures"], "train_targets": tr["targets"],
                "eval_structures": ev["structures"]}, {"modal": "r2scan"})
    assert out["provenance"]["model"] == "sevennet_mf0_r2scan"
    assert calls == [{"mesh": 8, "min_len": 7.0, "disp": 0.01, "model": "sevennet_mf0_r2scan"},
                     {"mesh": 8, "min_len": 7.0, "disp": 0.01, "model": "sevennet_mf0_r2scan"}]


def test_evaluator_reference_oracle_pooled(adapter, plan):
    pays = []
    for ep in plan["val"] + plan["ood"]:
        pp = ep.evaluate(np.full(16, 500.0), None).details["pooled_payload"]
        r = ep.evaluate(np.asarray(pp["y_ref"]), None)
        assert r.primary == pytest.approx(r.details["reference"]) and not r.accepted and r.z == 0
        o = ep.evaluate(np.asarray(pp["y_true"]), None)
        assert o.primary == pytest.approx(0.0) and o.accepted and o.hard_ok() and o.z == 1
        assert o.details["norm_score"] == 10.0
        pays.append(o.details["pooled_payload"])
    assert adapter.pooled_metric(pays) == pytest.approx(0.0)
    bad = plan["val"][0].evaluate(np.ones(5), None)
    assert bad.primary is None
    assert adapter.pooled_metric([bad.details["pooled_payload"]]) == pytest.approx(bad.details["reference"])


def test_constraints(plan):
    ep = plan["src"][0]
    thz = ep.evaluate(np.full(16, 15.0) / 33.356, None)                   # values in THz instead of cm^-1
    assert thz.h["plausible_frequency_cm-1"] is False and thz.z == 0
    assert ep.evaluate(np.full(15, 300.0), None).h["output_shape"] is False
    assert ep.evaluate(np.r_[np.nan, np.full(15, 300.0)], None).h["finite"] is False
    assert ep.required_output.unit == "cm^-1"
