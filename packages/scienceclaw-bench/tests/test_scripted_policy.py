"""Scripted policy: the deterministic fake policy used to drive the pipeline tests."""
from __future__ import annotations

import json

import pytest

from scripted_policy import scripted_responder


def _canvas_msg(text):
    return [{"role": "system", "content": "tools ..."}, {"role": "user", "content": text}]


def test_scripted_responder_walks_the_script():
    respond = scripted_responder()
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
        return json.loads(scripted_responder(rule)("policy", _canvas_msg(text)))["action"]["type"]

    assert act("always", submitted) == "add_node"
    assert act("library_only", submitted) == "finish"
    assert act("library_only", submitted + "### skill:s1 — t\n") == "add_node"
    assert act("library_or_hash", submitted + "### op:o1 — t\n") == "add_node"
    # without a library the decision is a deterministic function of the visible dev score
    outcomes = {act("library_or_hash", submitted.replace("0.9", f"0.{i}")) for i in range(1, 10)}
    assert outcomes == {"add_node", "finish"}
    with pytest.raises(ValueError):
        scripted_responder("sometimes")
