"""modify_node patch.code_edit (review F4) and 'operator nodes take no config' (review RT-7 / F7)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_actions_apply import CODE, act, add, base_graph, make_episode, make_program  # noqa: E402

from scienceclaw.core.actions import (  # noqa: E402
    EXEC_PATCH_KEYS, PATCH_KEYS, apply_action, apply_code_edit, is_control_edit, is_exec_edit, parse_action,
    touched_nodes,
)
from scienceclaw.core.graph import WorkflowGraph  # noqa: E402


@pytest.fixture()
def env():
    return make_episode(), make_program()


def test_code_edit_replaces_one_exact_occurrence(env) -> None:
    ep, prog = env
    g = base_graph(ep, prog)
    edit = {"find": "mean(axis=1)", "replace": "sum(axis=1)"}
    g2, err = apply_action(g, act("modify_node", ["skill:z"], id="c", patch={"code_edit": edit}), ep, prog, step=3)
    assert err is None and g2.nodes["c"].code == CODE.replace("mean(axis=1)", "sum(axis=1)")
    assert g.nodes["c"].code == CODE                                         # input graph untouched
    assert g2.nodes["c"].origin["modified_steps"] == [3] and "skill:z" in g2.nodes["c"].origin["uses"]
    # combined with a config change in one patch
    g3, err = apply_action(g2, act("modify_node", id="c", patch={"code_edit": {"find": "sum", "replace": "np.sum"},
                                                                  "config": {"k": 1}}), ep, prog)
    assert err is None and "np.sum(axis=1)" in g3.nodes["c"].code and g3.nodes["c"].config == {"k": 1}
    g4, err = apply_action(g, act("modify_node", id="c", patch={"code_edit": {
        "find": "'m': inputs['x'].mean(axis=1)", "replace": "'m': inputs['x'][:, 0]"}}), ep, prog)
    assert err is None and "inputs['x'][:, 0]" in g4.nodes["c"].code


def test_code_edit_errors_leave_the_graph_unchanged(env) -> None:
    ep, prog = env
    g = base_graph(ep, prog)

    def edit(**e):
        return act("modify_node", id="c", patch={"code_edit": e})

    g1, err = apply_action(g, edit(find="not in the code", replace="x"), ep, prog)
    assert g1 is g and "does not occur" in err
    two = g.copy()
    two.nodes["c"].code = "def run(inputs, config):\n    a = inputs['x']\n    b = inputs['x']\n    return {'m': a + b}\n"
    g1, err = apply_action(two, edit(find="inputs['x']", replace="inputs['y']"), ep, prog)
    assert g1 is two and "occurs 2 times" in err
    g1, err = apply_action(g, edit(find="def run(inputs, config):", replace="def go(inputs, config):"), ep, prog)
    assert g1 is g and "leaves the code invalid" in err and "top-level function run" in err
    g1, err = apply_action(g, edit(find="mean(axis=1)", replace="mean(axis=1"), ep, prog)
    assert g1 is g and "leaves the code invalid" in err and "does not parse" in err
    for bad, msg in (({"find": "", "replace": "x"}, "non-empty string"), ({"find": "a"}, "got keys ['find']"),
                     ({"find": "a", "replace": 3}, "must be a string"), ({"find": "a", "replace": "b", "x": 1}, "got keys")):
        _, err = apply_action(g, act("modify_node", id="c", patch={"code_edit": bad}), ep, prog)
        assert err is not None and msg in err, (bad, err)
    _, err = apply_action(g, act("modify_node", id="c", patch={"code_edit": "mean->sum"}), ep, prog)
    assert "must be an object" in err
    _, err = apply_action(g, act("modify_node", id="c", patch={"code": CODE, "code_edit": {"find": "a", "replace": "b"}}),
                          ep, prog)
    assert "cannot be combined" in err
    _, err = apply_action(g, act("modify_node", id="t", patch={"code_edit": {"find": "a", "replace": "b"}}), ep, prog)
    assert "only to code nodes" in err


def test_code_edit_is_an_executable_edit_and_public_helper(env) -> None:
    ep, prog = env
    g = base_graph(ep, prog)
    a = act("modify_node", id="c", patch={"code_edit": {"find": "mean", "replace": "sum"}})
    assert is_exec_edit(a, g) and not is_control_edit(a, g) and touched_nodes(a) == {"c"}
    assert "code_edit" in PATCH_KEYS and "code_edit" in EXEC_PATCH_KEYS
    assert apply_code_edit(CODE, {"find": "mean", "replace": "sum"}) == (CODE.replace("mean", "sum"), None)
    assert apply_code_edit(CODE, {"find": "zzz", "replace": ""})[0] is None
    assert apply_code_edit(CODE, {"find": "zzz"})[0] is None
    assert a.describe() == "modify_node c patch=['code_edit']"


def test_code_edit_parses_from_policy_text_and_in_a_batch(env) -> None:
    ep, prog = env
    text = json.dumps({"thought": "t", "action": {"type": "modify_node", "id": "c",
                                                   "patch": {"code_edit": {"find": "mean", "replace": "sum"}}}})
    a, err = parse_action(text)
    assert err is None and a.payload["patch"]["code_edit"]["replace"] == "sum"
    a, err = parse_action(json.dumps({"action": {"type": "modify_node", "id": "c", "patch": {"code_edit": {"find": "x"}}}}))
    assert a is None and "modify_node: patch.code_edit" in err
    g = base_graph(ep, prog)
    batch = act("batch", actions=[
        {"type": "modify_node", "id": "c", "patch": {"code_edit": {"find": "mean", "replace": "sum"}}},
        {"type": "modify_node", "id": "c", "patch": {"config": {"k": 2}}}])
    g2, err = apply_action(g, batch, ep, prog)
    assert err is None and "sum(axis=1)" in g2.nodes["c"].code and g2.nodes["c"].config == {"k": 2}
    bad = act("batch", actions=[
        {"type": "modify_node", "id": "c", "patch": {"code_edit": {"find": "mean", "replace": "sum"}}},
        {"type": "modify_node", "id": "c", "patch": {"code_edit": {"find": "mean", "replace": "min"}}}])
    g3, err = apply_action(g, bad, ep, prog)
    assert g3 is g and "batch action #1" in err and "does not occur" in err              # atomic: the first edit is rolled back


def test_operator_nodes_reject_config(env) -> None:
    ep, prog = env
    _, err = add(WorkflowGraph(), ep, prog, {"id": "o", "kind": "operator", "ref": "op:rowmean", "config": {"axis": 0}})
    assert err is not None and "operator nodes take no config" in err and "['axis']" in err
    g, err = add(WorkflowGraph(), ep, prog, {"id": "o", "kind": "operator", "ref": "op:rowmean", "config": {}})
    assert err is None and g.nodes["o"].config == {}
    _, err = apply_action(g, act("modify_node", id="o", patch={"config": {"axis": 0}}), ep, prog)
    assert err is not None and "operator nodes take no config" in err
    g2, err = apply_action(g, act("modify_node", id="o", patch={"config": {"axis": None}}), ep, prog)   # deleting is harmless
    assert err is None and g2.nodes["o"].config == {}
    g3, err = add(g, ep, prog, {"id": "t", "kind": "tool", "ref": "load", "config": {"k": 1}})            # tool nodes do
    assert err is None and g3.nodes["t"].config == {"k": 1}


def test_repair_code_tail_strips_leaked_reply_envelope():
    from scienceclaw.core.actions import repair_code_tail
    good = "def run(inputs, config):\n    return {'w': {1: 2}}"
    assert repair_code_tail(good) == good
    leaked = good + "}}}</invoke></div>"
    assert repair_code_tail(leaked) == good
    assert repair_code_tail(good + '"}\n}') == good
    junk = "def run(inputs, config):\n    return {'w': (1"           # genuinely broken code is left untouched
    assert repair_code_tail(junk) == junk
