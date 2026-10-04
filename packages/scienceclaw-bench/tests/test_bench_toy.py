"""TOY adapter: episodes, tools, hidden evaluation, dev evaluation, pooled metric, scripted policy."""
from __future__ import annotations

import json

import numpy as np
import pytest

from scienceclaw.bench.task import passes
from scienceclaw.bench.tasks.toy import (ACCEPT_RATIO, IID_RANGE, OOD_RANGE, ToyAdapter, make_toy_plan,
                                         scripted_toy_responder, toy_truth)


def _episode(split="id", k=0, seed=11, ipe=16, **kw):
    return ToyAdapter(**kw).build_episodes(split, k + 1, seed, items_per_episode=ipe)[k]


def _data(ep):
    tr = ep.tool("load_train").fn({}, {})
    ev = ep.tool("load_eval_inputs").fn({}, {})
    return tr["X_train"], tr["y_train"], ev["X_eval"]


def test_protocol_fields_and_schema():
    ad = ToyAdapter()
    assert (ad.discipline, ad.family, ad.metric, ad.direction) == ("TOY", "Engineering & computing", "RMSE", "min")
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
    ev = ep.evaluate(toy_truth(Xe), None)
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
    y = toy_truth(Xe)
    y[2] = np.nan
    nan = ep.evaluate(y, None)
    assert nan.h["finite"] is False and nan.h["length"] is True and nan.z == 0
    none = ep.evaluate(None, None)
    assert none.completed is False and none.z == 0


def test_dev_evaluator_uses_reserved_training_rows_only():
    ep = _episode()
    Xt, yt, Xe = _data(ep)
    good = ep.dev_evaluate(toy_truth(Xe))
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
    ad = ToyAdapter()
    p1 = {"y_true": [0.0, 0.0], "y_pred": [1.0, 1.0], "y_ref": [0.0, 0.0]}
    p2 = {"y_true": [0.0, 0.0], "y_pred": None, "y_ref": [3.0, 3.0]}
    assert ad.pooled_metric([p1]) == pytest.approx(1.0)
    assert ad.pooled_metric([p1, p2]) == pytest.approx(np.sqrt((1 + 1 + 9 + 9) / 4))
    assert ad.pooled_metric([]) is None


def test_make_toy_plan():
    plan = make_toy_plan(rounds=3, n_val=1, n_id=2, n_ood=2, items_per_episode=4)
    assert plan.order == ["TOY"]
    assert [r for r, _ in plan.source_stream()] == [1, 2, 3]
    assert len(plan.episodes["id"]["TOY"]) == 2


def _canvas_msg(text):
    return [{"role": "system", "content": "tools ..."}, {"role": "user", "content": text}]


def test_scripted_responder_walks_the_script():
    respond = scripted_toy_responder()
    a0 = json.loads(respond("policy", _canvas_msg("## Current canvas\n(empty canvas)")))
    assert a0["action"] == {"type": "add_node", "node": {"id": "tr", "kind": "tool", "ref": "load_train"}}
    canvas = ("## Current canvas\n[tr] kind=tool ref=load_train\n[ev] kind=tool ref=load_eval_inputs\n"
              "[fit] kind=code\n    code:\n      import numpy\n[out] kind=submit\nedges:\n"
              "  tr.X_train -> fit.X_train\n  tr.y_train -> fit.y_train\n  ev.X_eval -> fit.X_eval\n"
              "  fit.y_pred -> out.y\n")
    a = json.loads(respond("policy", _canvas_msg(canvas + "\n### skill:fit_models — Fit\n")))
    # repair = replace the model node: first a new code node with the least-squares model ...
    assert a["action"]["type"] == "add_node" and a["action"]["node"]["id"] == "fit2"
    assert "np.linalg.solve" in a["action"]["node"]["code"]
    assert a["uses"] == ["skill:fit_models"]
    # ... then its input wiring, removal of the old node, and the submit wiring (control edits)
    with_fit2 = canvas.replace("[out] kind=submit", "[fit2] kind=code\n    code:\n      np.linalg.solve\n[out] kind=submit")
    a = json.loads(respond("policy", _canvas_msg(with_fit2)))
    assert a["action"] == {"type": "add_edge", "edge": {"src": "tr", "src_port": "X_train", "dst": "fit2",
                                                         "dst_port": "X_train"}}
    wired = with_fit2 + "  tr.X_train -> fit2.X_train\n  tr.y_train -> fit2.y_train\n  ev.X_eval -> fit2.X_eval\n"
    assert json.loads(respond("policy", _canvas_msg(wired)))["action"] == {"type": "remove_node", "id": "fit"}
    no_fit = (wired.replace("[fit] kind=code\n    code:\n      import numpy\n", "")
              .replace("  tr.X_train -> fit.X_train\n  tr.y_train -> fit.y_train\n  ev.X_eval -> fit.X_eval\n"
                       "  fit.y_pred -> out.y\n", ""))
    a = json.loads(respond("policy", _canvas_msg(no_fit)))
    assert a["action"]["type"] == "add_edge" and a["action"]["edge"]["src"] == "fit2"
    done = json.loads(respond("policy", _canvas_msg(no_fit + "  fit2.y_pred -> out.y\n")))
    assert done["action"] == {"type": "finish"}
    assert json.loads(respond("patch", []))["title"]


def test_scripted_responder_repair_rules():
    submitted = ("## Feedback of the last action\nDev score on visible data: {\"dev_rmse\": 0.9, \"n_dev\": 8}\n"
                 "## Current canvas\n[tr] kind=tool\n[ev] kind=tool\n[fit] kind=code\n[out] kind=submit\nedges:\n"
                 "  tr.X_train -> fit.X_train\n  tr.y_train -> fit.y_train\n  ev.X_eval -> fit.X_eval\n"
                 "  fit.y_pred -> out.y\n")

    def act(rule, text):
        return json.loads(scripted_toy_responder(rule)("policy", _canvas_msg(text)))["action"]["type"]

    assert act("always", submitted) == "add_node"
    assert act("library_only", submitted) == "finish"
    assert act("library_only", submitted + "### skill:s1 — t\n") == "add_node"
    assert act("library_or_hash", submitted + "### op:o1 — t\n") == "add_node"
    # without a library the decision is a deterministic function of the visible dev score
    outcomes = {act("library_or_hash", submitted.replace("0.9", f"0.{i}")) for i in range(1, 10)}
    assert outcomes == {"add_node", "finish"}
    with pytest.raises(ValueError):
        scripted_toy_responder("sometimes")
