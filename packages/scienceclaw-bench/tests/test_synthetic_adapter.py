"""Synthetic adapter: episodes, tools, hidden evaluation, dev evaluation, pooled metric."""
from __future__ import annotations

import numpy as np
import pytest

from scienceclaw.bench.task import passes
from synthetic_adapter import ACCEPT_RATIO, IID_RANGE, OOD_RANGE, SyntheticAdapter, synthetic_truth


def _episode(split="id", k=0, seed=11, ipe=16, **kw):
    return SyntheticAdapter(**kw).build_episodes(split, k + 1, seed, items_per_episode=ipe)[k]


def _data(ep):
    tr = ep.tool("load_train").fn({}, {})
    ev = ep.tool("load_eval_inputs").fn({}, {})
    return tr["X_train"], tr["y_train"], ev["X_eval"]


def test_protocol_fields_and_schema():
    ad = SyntheticAdapter()
    assert (ad.discipline, ad.family, ad.metric, ad.direction) == ("SYN", "Engineering & computing", "RMSE", "min")
    assert ad.available()[0]
    ep = _episode(ipe=16)
    assert ep.n_items == 16 + ad.n_dev
    assert ep.required_output.type == "array" and ep.required_output.shape == ("n_items",)
    assert ep.required_output.dtype == "float"
    assert {t.name for t in ep.tools} == {"load_train", "load_eval_inputs"}
    assert {c.name for c in ep.constraints} == {"length", "finite"}
    Xt, yt, Xe = _data(ep)
    assert Xt.shape == (ad.n_train, 2) and yt.shape == (ad.n_train,) and Xe.shape == (ep.n_items, 2)
    assert len(ep.lineage["item_ids"]) == 16 and len(set(ep.lineage["item_ids"])) == 16
    with pytest.raises(ValueError):
        ad.build_episodes("rep", 1, 0)


def test_perfect_predictor_is_accepted():
    ep = _episode()
    _, _, Xe = _data(ep)
    ev = ep.evaluate(synthetic_truth(Xe), None)
    assert ev.completed and ev.hard_ok() and ev.accepted and ev.z == 1
    assert ev.direction == "min"
    assert ev.primary < 0.25                      # ~ noise sd (0.1)
    assert ev.primary <= ACCEPT_RATIO * ev.details["reference"]
    assert ev.details["norm_score"] == pytest.approx(min(ev.details["reference"] / ev.primary, 10.0))  # clipped
    pay = ev.details["pooled_payload"]
    assert len(pay["y_true"]) == len(pay["y_pred"]) == 16
    ev.reproducible = True
    assert passes(ev)


def test_trivial_predictor_equals_reference_and_is_rejected():
    ep = _episode()
    _, yt, Xe = _data(ep)
    ev = ep.evaluate(np.full(len(Xe), yt.mean()), None)
    assert ev.completed and ev.hard_ok()
    assert ev.primary == pytest.approx(ev.details["reference"])
    assert ev.details["norm_score"] == pytest.approx(1.0)
    assert not ev.accepted and ev.z == 0


def test_hard_constraints():
    ep = _episode()
    _, _, Xe = _data(ep)
    short = ep.evaluate(np.zeros(3), None)
    assert short.h["length"] is False and short.z == 0 and short.primary is None
    assert short.details["norm_score"] == 0.0 and short.details["pooled_payload"]["y_pred"] is None
    y = synthetic_truth(Xe)
    y[2] = np.nan
    nan = ep.evaluate(y, None)
    assert nan.h["finite"] is False and nan.h["length"] is True and nan.z == 0
    none = ep.evaluate(None, None)
    assert none.completed is False and none.z == 0


def test_dev_evaluator_uses_reserved_training_rows_only():
    ep = _episode()
    Xt, yt, Xe = _data(ep)
    good = ep.dev_evaluate(synthetic_truth(Xe))
    bad = ep.dev_evaluate(np.full(len(Xe), yt.mean()))
    assert good["dev_rmse"] < 0.3 < bad["dev_rmse"]
    assert bad["dev_rmse"] == pytest.approx(bad["dev_reference_rmse"])
    assert "error" in ep.dev_evaluate(np.zeros(2))
    # dev rows are not part of the visible training data
    assert not any((Xt == row).all(axis=1).any() for row in Xe)


def test_deterministic_and_ood_shift():
    a, b = _episode(seed=5), _episode(seed=5)
    assert a.id == b.id and a.lineage == b.lineage
    np.testing.assert_array_equal(_data(a)[2], _data(b)[2])
    assert _episode(seed=6).id != a.id
    ood = _episode("ood", seed=5)
    Xt, _, Xe = _data(ood)
    (lo1, hi1), (lo2, hi2) = OOD_RANGE
    assert Xe[:, 0].min() >= lo1 and Xe[:, 0].max() <= hi1 and Xe[:, 1].min() >= lo2
    (ilo, ihi), _ = IID_RANGE
    assert _data(a)[2][:, 0].max() <= ihi and ood.lineage["pool"] == "ood" and a.lineage["pool"] == "iid"


def test_pooled_metric_is_pooled_rmse_with_reference_fallback():
    ad = SyntheticAdapter()
    p1 = {"y_true": [0.0, 0.0], "y_pred": [1.0, 1.0], "y_ref": [0.0, 0.0]}
    p2 = {"y_true": [0.0, 0.0], "y_pred": None, "y_ref": [3.0, 3.0]}
    assert ad.pooled_metric([p1]) == pytest.approx(1.0)
    assert ad.pooled_metric([p1, p2]) == pytest.approx(np.sqrt((1 + 1 + 9 + 9) / 4))
    assert ad.pooled_metric([]) is None
