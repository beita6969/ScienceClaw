"""FoR34 ogbg-molhiv adapter: splits, determinism, leakage, evaluator, constraints (real OGB data; skipped if absent)."""
from __future__ import annotations

import numpy as np
import pytest

from scienceclaw.bench.splits import split_seed
from scienceclaw.bench.tasks.for34_molhiv import Adapter, _idx

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
    return {s: adapter.build_episodes(s, n, split_seed(20260928, "FoR34", s)) for s, n in SPLITS.items()}


def _tools(ep):
    return {t.name: t for t in ep.tools}


def test_splits_disjoint_and_stratified(adapter, plan):
    owner = {}
    data = adapter._data.get()
    for s, eps in plan.items():
        assert len(eps) == SPLITS[s]
        for ep in eps:
            ids = ep.lineage["item_ids"]
            assert len(ids) == 16 == ep.n_items and len(set(ids)) == 16
            y = data.labels[[_idx(i) for i in ids]]
            assert y.sum() == 4 and (1 - y).sum() == 12          # both classes present
            for i in ids:
                assert owner.setdefault(i, s) == s, f"{i} in {owner[i]} and {s}"
    valid, test = set(data.split["valid"].tolist()), set(data.split["test"].tolist())
    for s, eps in plan.items():
        for ep in eps:
            src = test if s == "ood" else valid
            assert all(_idx(i) in src for i in ep.lineage["item_ids"])
            assert (ep.lineage["ood_kind"] == "proxy_within_dataset") == (s == "ood")


def test_deterministic_and_prefix_stable(adapter, plan):
    again = Adapter().build_episodes("src", 3, split_seed(20260928, "FoR34", "src"))
    assert [e.lineage["item_ids"] for e in again] == [e.lineage["item_ids"] for e in plan["src"][:3]]
    assert [e.id for e in again] == [e.id for e in plan["src"][:3]]
    other = adapter.build_episodes("src", 1, 999)
    assert other[0].lineage["item_ids"] != plan["src"][0].lineage["item_ids"]


def test_tools_expose_no_eval_labels_and_train_is_disjoint(adapter, plan):
    data = adapter._data.get()
    train_part = set(data.split["train"].tolist())
    ep = plan["id"][0]
    t = _tools(ep)
    ev = t["load_eval_inputs"].fn({}, {})
    assert set(ev) == {"smiles"} and all(isinstance(s, str) for s in ev["smiles"]) and len(ev["smiles"]) == 16
    assert ev["smiles"] == [str(data.smiles[_idx(i)]) for i in ep.lineage["item_ids"]]
    tr = t["load_train"].fn({}, {})
    dev = t["load_dev_inputs"].fn({}, {})
    assert set(dev) == {"dev_smiles"}
    assert len(tr["smiles"]) == adapter.n_train and len(dev["dev_smiles"]) == adapter.n_dev
    tr_idx, dev_idx = adapter._train_sample(data, ep.id)
    assert set(tr_idx.tolist()) <= train_part and set(dev_idx.tolist()) <= train_part
    assert not set(tr_idx.tolist()) & set(dev_idx.tolist())
    assert not {_idx(i) for i in ep.lineage["item_ids"]} & train_part
    assert 0.0 < tr["labels"].mean() < 0.1                        # natural class rate of the training partition
    for spec in ep.tools:                                          # declared schemas match returned ports
        if not spec.inputs:
            assert set(spec.fn({}, {})) == set(spec.outputs)


def test_featurizer_and_score_dev(plan):
    t = _tools(plan["src"][0])
    f = t["featurize_molecules"].fn({"smiles": ["CCO", "c1ccccc1", "not-a-smiles"]}, {"kind": "morgan", "n_bits": 64})
    assert f["X"].shape == (3, 64) and f["valid"].tolist() == [True, True, False]
    d = t["featurize_molecules"].fn({"smiles": ["CCO"]}, {"kind": "descriptors"})
    assert d["X"].shape == (1, len(d["feature_names"]))
    with pytest.raises(ValueError):
        t["featurize_molecules"].fn({"smiles": ["CCO"]}, {"kind": "bogus"})
    n_dev = len(t["load_dev_inputs"].fn({}, {})["dev_smiles"])
    out = t["score_dev"].fn({"dev_scores": np.zeros(n_dev)}, {})
    assert out["dev_roc_auc"] == pytest.approx(0.5) and 0.0 <= out["dev_reference_roc_auc"] <= 1.0
    with pytest.raises(ValueError):
        t["score_dev"].fn({"dev_scores": np.zeros(3)}, {})


def test_evaluator_reference_oracle_and_pooled(adapter, plan):
    payloads = []
    for ep in plan["val"] + plan["ood"]:
        ref = ep.evaluate(None, None)                                        # no output -> all constraints fail
        assert ref.completed is False and not any(ref.h.values())
        probe = ep.evaluate(np.full(16, 0.5), None)
        pp = probe.details["pooled_payload"]
        ref_scores = np.asarray(pp["y_ref"])
        r = ep.evaluate(ref_scores, None)
        assert r.primary == pytest.approx(r.details["reference"]) and r.accepted is False and r.z == 0
        assert r.details["norm_score"] == pytest.approx(1.0)
        oracle = ep.evaluate(np.asarray(pp["y_true"], dtype=float), None)
        assert oracle.primary == 1.0 and oracle.hard_ok()
        assert oracle.accepted == (1.0 >= r.details["reference"] + adapter.margin)
        payloads.append(oracle.details["pooled_payload"])
    assert adapter.pooled_metric(payloads) == pytest.approx(1.0)
    bad = plan["val"][0].evaluate([0.5] * 3, None)
    assert bad.primary is None and bad.details["pooled_payload"]["y_score"] is None
    assert 0.0 <= adapter.pooled_metric([bad.details["pooled_payload"]]) <= 1.0


@pytest.mark.parametrize("y,failing", [([0.5] * 15, "output_shape"), ([np.nan] + [0.5] * 15, "finite"),
                                       ([1.5] + [0.5] * 15, "probability_range"), ("text", "output_shape")])
def test_constraints_fire_on_malformed_output(plan, y, failing):
    r = plan["src"][0].evaluate(y, None)
    assert r.h[failing] is False and r.z == 0 and r.hard_ok() is False
    if failing != "probability_range":          # a range violation is still scorable, but never a success
        assert r.primary is None


def test_objective_documents_the_domain_library(plan):
    ep = plan["val"][0]
    assert "scilib.molecules" in ep.objective and "fit_predict" in ep.objective and "grouped_cv_auc" in ep.objective
    assert "scaffold" in ep.objective and "featurize" in ep.objective


def test_about_half_of_the_dev_molecules_share_a_scaffold_with_the_training_molecules(plan):
    from scilib.molecules import scaffold_groups
    ep = plan["val"][0]
    t = _tools(ep)
    tr, dv, ev = t["load_train"].fn({}, {})["smiles"], t["load_dev_inputs"].fn({}, {})["dev_smiles"], \
        t["load_eval_inputs"].fn({}, {})["smiles"]
    g = scaffold_groups(list(tr) + list(dv) + list(ev))
    n_tr, n_dv = len(tr), len(dv)
    gt, gd, ge = set(g[:n_tr].tolist()), g[n_tr:n_tr + n_dv], g[n_tr + n_dv:]
    shared = float(np.mean([x in gt for x in gd]))
    assert 0.35 < shared < 0.65                                      # dev shares scaffolds with train (statement in the objective)
    assert not any(x in gt for x in ge.tolist())                    # evaluation scaffolds never occur in the training molecules
