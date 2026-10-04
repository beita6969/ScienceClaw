"""apply_action / edit classes: every action type, automatic ports, and every error path."""
from __future__ import annotations

import json

import numpy as np
import pytest

from scienceclaw.bench.task import Budget, EvalResult, Episode, ToolSpec
from scienceclaw.core.actions import (
    Action,
    apply_action,
    apply_action_detailed,
    is_control_edit,
    is_exec_edit,
    touched_nodes,
)
from scienceclaw.core.graph import Edge, Node, WorkflowGraph
from scienceclaw.core.operators import Contract, OperatorSpec
from scienceclaw.core.program import AgentProgram
from scienceclaw.core.schema import PortSchema

CODE = "def run(inputs, config):\n    return {'m': inputs['x'].mean(axis=1)}\n"


def make_episode() -> Episode:
    load = ToolSpec("load", "load visible data", {}, {"data": PortSchema("array", ("n", 3), unit="K")},
                    lambda i, c: {"data": np.ones((4, 3))})
    return Episode(id="ep", discipline="FoR34", family="Physical & Earth", split="src", task_type="regression",
                   objective="predict", required_output=PortSchema("array", ("n",)), tools=[load], constraints=[],
                   budget=Budget(), _evaluate=lambda y, t: EvalResult(primary=1.0))


def make_program() -> AgentProgram:
    body = WorkflowGraph({"c": Node("c", "code", code=CODE, inputs={"x": PortSchema("array")},
                                    outputs={"m": PortSchema("array")})})
    op = OperatorSpec("rowmean", 3, "row mean", "mean over columns", body, {"x": PortSchema("array")},
                      {"m": PortSchema("array")}, {"x": [("c", "x")]}, {"m": ("c", "m")}, Contract())
    return AgentProgram(operators={"rowmean": op})


def act(atype: str, uses: list[str] | None = None, **payload) -> Action:
    return Action(atype, payload, uses or [])


def add(graph, ep, prog, node: dict, step=None, uses=None):
    return apply_action(graph, act("add_node", uses, node=node), ep, prog, step=step)


@pytest.fixture()
def env():
    return make_episode(), make_program()


def base_graph(ep, prog) -> WorkflowGraph:
    g = WorkflowGraph()
    for nd in ({"id": "t", "kind": "tool", "ref": "load"},
               {"id": "c", "kind": "code", "code": CODE, "inputs": {"x": {"type": "array", "shape": ["n", 3], "unit": "K"}},
                "outputs": {"m": {"type": "array", "shape": ["n"]}}},
               {"id": "s", "kind": "submit"}):
        g, err = add(g, ep, prog, nd)
        assert err is None, err
    for e in ({"src": "t", "src_port": "data", "dst": "c", "dst_port": "x"}, {"src": "c", "dst": "s"}):
        g, err = apply_action(g, act("add_edge", edge=e), ep, prog)
        assert err is None, err
    return g


# --------------------------------------------------------------------------- add_node
def test_add_tool_node_ports_from_toolspec(env) -> None:
    ep, prog = env
    g0 = WorkflowGraph()
    g, err = add(g0, ep, prog, {"id": "t", "kind": "tool", "ref": "tool:load", "inputs": {"z": {"type": "text"}}},
                 step=2, uses=["skill:s"])
    assert err is None
    n = g.nodes["t"]
    assert n.ref == "load" and n.inputs == {} and n.outputs == {"data": PortSchema("array", ("n", 3), unit="K")}
    assert n.origin["step"] == 2 and n.origin["uses"] == ["skill:s"] and n.origin["generated"] is False
    assert g0.nodes == {}                                   # input graph untouched
    g2, err = add(g, ep, prog, {"id": "u", "kind": "tool", "ref": "missing_tool"})
    assert g2 is g and "unknown tool 'missing_tool'" in err and "['load']" in err


def test_add_operator_node_ports_from_program(env) -> None:
    ep, prog = env
    g, err = add(WorkflowGraph(), ep, prog, {"id": "o", "kind": "operator", "ref": "op:rowmean@v3"})
    assert err is None
    n = g.nodes["o"]
    assert n.ref == "op:rowmean" and set(n.inputs) == {"x"} and set(n.outputs) == {"m"}
    assert n.origin["op_version"] == "op:rowmean@v3"
    g, err = add(WorkflowGraph(), ep, prog, {"id": "o", "kind": "operator", "ref": "rowmean"})
    assert err is None and g.nodes["o"].ref == "op:rowmean"
    _, err = add(WorkflowGraph(), ep, prog, {"id": "o", "kind": "operator", "ref": "op:nope"})
    assert "unknown operator 'op:nope'" in err and "op:rowmean" in err
    _, err = add(WorkflowGraph(), ep, None, {"id": "o", "kind": "operator", "ref": "op:rowmean"})
    assert "unknown operator" in err


def test_add_submit_node_and_single_submit(env) -> None:
    ep, prog = env
    g, err = add(WorkflowGraph(), ep, prog, {"id": "s", "kind": "submit"})
    assert err is None
    assert g.nodes["s"].inputs == {"y": ep.required_output} and g.nodes["s"].outputs == {}
    g2, err = add(g, ep, prog, {"id": "s2", "kind": "submit"})
    assert g2 is g and "submit node already exists" in err


def test_add_code_node_validation(env) -> None:
    ep, prog = env
    ports = {"inputs": {"x": {"type": "array"}}, "outputs": {"m": {"type": "array"}}}
    g, err = add(WorkflowGraph(), ep, prog, {"id": "c", "kind": "code", "code": CODE, **ports})
    assert err is None and g.nodes["c"].origin["generated"] is True
    _, err = add(WorkflowGraph(), ep, prog, {"id": "c", "kind": "code", "code": "def go(x):\n    return {}", **ports})
    assert "must define a top-level function run" in err
    _, err = add(WorkflowGraph(), ep, prog, {"id": "c", "kind": "code", "code": "def run(inputs):\n    return {}", **ports})
    assert "two positional arguments" in err
    _, err = add(WorkflowGraph(), ep, prog, {"id": "c", "kind": "code", "code": "def run(inputs, config)\n  pass", **ports})
    assert "does not parse: line 1" in err


def test_add_llm_node_defaults_and_config_checks(env) -> None:
    ep, prog = env
    g, err = add(WorkflowGraph(), ep, prog, {"id": "l", "kind": "llm", "prompt": "Label {text}", "config": {"parse": "json"}})
    assert err is None
    assert set(g.nodes["l"].inputs) == {"items"} and set(g.nodes["l"].outputs) == {"outputs"}
    _, err = add(WorkflowGraph(), ep, prog, {"id": "l", "kind": "llm", "prompt": "p", "config": {"parse": "xml"}})
    assert "config.parse must be one of" in err
    _, err = add(WorkflowGraph(), ep, prog, {"id": "l", "kind": "llm", "prompt": "p", "config": {"parse": "choice"}})
    assert "config.choices" in err
    _, err = add(WorkflowGraph(), ep, prog, {"id": "l", "kind": "llm", "prompt": "p", "inputs": {"docs": {"type": "list"}}})
    assert "input port 'items'" in err
    _, err = add(WorkflowGraph(), ep, prog, {"id": "l", "kind": "llm", "prompt": "p", "outputs": {"labels": {"type": "list"}}})
    assert "exactly one output port 'outputs'" in err


def test_add_node_duplicate_and_invalid_id(env) -> None:
    ep, prog = env
    g = base_graph(ep, prog)
    g2, err = add(g, ep, prog, {"id": "t", "kind": "tool", "ref": "load"})
    assert g2 is g and "already exists" in err
    g3, err = add(g, ep, prog, {"id": "a-b", "kind": "tool", "ref": "load"})
    assert g3 is g and "node.id must match" in err


# --------------------------------------------------------------------------- remove / modify
def test_remove_node_removes_incident_edges(env) -> None:
    ep, prog = env
    g = base_graph(ep, prog)
    g2, err = apply_action(g, act("remove_node", id="c"), ep, prog)
    assert err is None
    assert "c" not in g2.nodes and all("c" not in (e.src, e.dst) for e in g2.edges)
    assert "c" in g.nodes and len(g.edges) == 2                      # original untouched
    _, err = apply_action(g, act("remove_node", id="zz"), ep, prog)
    assert "unknown node 'zz'" in err


def test_modify_node_patches(env) -> None:
    ep, prog = env
    g = base_graph(ep, prog)
    g, err = apply_action(g, act("modify_node", id="c", patch={"config": {"a": 1, "b": 2}}), ep, prog, step=1)
    assert err is None and g.nodes["c"].config == {"a": 1, "b": 2}
    g, err = apply_action(g, act("modify_node", ["skill:z"], id="c", patch={"config": {"a": None, "c": [3]}}), ep, prog, step=4)
    assert err is None and g.nodes["c"].config == {"b": 2, "c": [3]}
    assert g.nodes["c"].origin["modified_steps"] == [1, 4] and "skill:z" in g.nodes["c"].origin["uses"]
    new_code = CODE.replace("mean", "sum")
    g, err = apply_action(g, act("modify_node", id="c", patch={"code": new_code}), ep, prog)
    assert err is None and g.nodes["c"].code == new_code
    # wrong kinds
    _, err = apply_action(g, act("modify_node", id="t", patch={"code": CODE}), ep, prog)
    assert "only to code nodes" in err
    _, err = apply_action(g, act("modify_node", id="c", patch={"prompt": "x"}), ep, prog)
    assert "only to llm nodes" in err
    _, err = apply_action(g, act("modify_node", id="t", patch={"outputs": {"q": {"type": "array"}}}), ep, prog)
    assert "fixed by its specification" in err
    _, err = apply_action(g, act("modify_node", id="c", patch={"code": "x = 1"}), ep, prog)
    assert "must define a top-level function run" in err
    _, err = apply_action(g, act("modify_node", id="nope", patch={"config": {}}), ep, prog)
    assert "unknown node 'nope'" in err


def test_modify_ports_that_break_edges_is_rejected_unchanged(env) -> None:
    ep, prog = env
    g = base_graph(ep, prog)
    before = json.dumps(g.to_dict(), sort_keys=True, default=str)
    g2, err, verrs = apply_action_detailed(g, act("modify_node", id="c", patch={"outputs": {"z": {"type": "array"}}}), ep, prog)
    assert g2 is g and err.startswith("graph invalid after edit") and verrs
    assert "no output port 'm'" in err
    assert json.dumps(g.to_dict(), sort_keys=True, default=str) == before


# --------------------------------------------------------------------------- edges
def test_add_edge_errors_and_auto_ports(env) -> None:
    ep, prog = env
    g = WorkflowGraph()
    for nd in ({"id": "t", "kind": "tool", "ref": "load"},
               {"id": "c", "kind": "code", "code": CODE, "inputs": {"x": {"type": "array", "unit": "K"}, "w": {"type": "number"}},
                "outputs": {"m": {"type": "array"}}},
               {"id": "d", "kind": "code", "code": CODE, "inputs": {"x": {"type": "array", "unit": "degC"}},
                "outputs": {"m": {"type": "array"}}},
               {"id": "txt", "kind": "code", "code": CODE, "inputs": {"x": {"type": "text"}}, "outputs": {"m": {"type": "array"}}}):
        g, err = add(g, ep, prog, nd)
        assert err is None, err
    cases = [
        ({"src": "zz", "dst": "c"}, "unknown source node 'zz'"),
        ({"src": "t", "dst": "zz"}, "unknown destination node 'zz'"),
        ({"src": "t", "dst": "t"}, "cannot connect a node to itself"),
        ({"src": "t", "src_port": "nope", "dst": "c", "dst_port": "x"}, "no output port 'nope'"),
        ({"src": "t", "dst": "c"}, "specify the input port of node 'c'"),
        ({"src": "t", "dst": "d"}, "unit mismatch K -> degC"),
        ({"src": "t", "dst": "txt"}, "type mismatch array -> text"),
    ]
    for e, needle in cases:
        g2, err = apply_action(g, act("add_edge", edge=e), ep, prog)
        assert g2 is g and err and needle in err, (e, err)
    g2, err = apply_action(g, act("add_edge", edge={"src": "t", "dst": "c", "dst_port": "x"}), ep, prog)
    assert err is None and g2.edges == [Edge.make("t", "data", "c", "x")]
    _, err = apply_action(g2, act("add_edge", edge={"src": "t", "dst": "c", "dst_port": "x"}), ep, prog)
    assert "already exists" in err
    _, err = apply_action(g2, act("add_edge", edge={"src": "d", "dst": "c", "dst_port": "x"}), ep, prog)
    assert "already wired from t.data" in err
    conv = {"from": "K", "to": "degC", "factor": 1.0, "offset": -273.15}
    g3, err = apply_action(g2, act("add_edge", edge={"src": "t", "dst": "d", "conversion": conv}), ep, prog)
    assert err is None and g3.edges[-1].conversion_dict == conv
    _, err = apply_action(g2, act("add_edge", edge={"src": "t", "dst": "d",
                                                    "conversion": {"from": "g", "to": "kg", "factor": 1e-3}}), ep, prog)
    assert "does not map K -> degC" in err


def test_cycle_rejected(env) -> None:
    ep, prog = env
    g = WorkflowGraph()
    for nid in ("a", "b"):
        g, err = add(g, ep, prog, {"id": nid, "kind": "code", "code": CODE, "inputs": {"x": {"type": "array"}},
                                   "outputs": {"m": {"type": "array"}}})
    g, err = apply_action(g, act("add_edge", edge={"src": "a", "dst": "b"}), ep, prog)
    assert err is None
    g2, err = apply_action(g, act("add_edge", edge={"src": "b", "dst": "a"}), ep, prog)
    assert g2 is g and "cycle" in err


def test_remove_edge(env) -> None:
    ep, prog = env
    g = base_graph(ep, prog)
    g2, err = apply_action(g, act("remove_edge", edge={"src": "c", "dst": "s"}), ep, prog)
    assert err is None and len(g2.edges) == 1
    _, err = apply_action(g2, act("remove_edge", edge={"src": "c", "dst": "s"}), ep, prog)
    assert "no such edge" in err


# --------------------------------------------------------------------------- finish / batch
def test_finish_returns_equal_copy(env) -> None:
    ep, prog = env
    g = base_graph(ep, prog)
    g2, err = apply_action(g, act("finish"), ep, prog)
    assert err is None and g2 is not g and g2.to_dict() == g.to_dict()


def test_batch_is_atomic(env) -> None:
    ep, prog = env
    good = act("batch", actions=[{"type": "add_node", "node": {"id": "t", "kind": "tool", "ref": "load"}},
                                 {"type": "add_node", "node": {"id": "s", "kind": "submit"}}])
    g, err = apply_action(WorkflowGraph(), good, ep, prog)
    assert err is None and set(g.nodes) == {"t", "s"}
    badb = act("batch", actions=[{"type": "add_node", "node": {"id": "t2", "kind": "tool", "ref": "load"}},
                                 {"type": "add_edge", "edge": {"src": "t2", "dst": "nope"}}])
    g2, err = apply_action(g, badb, ep, prog)
    assert g2 is g and err.startswith("batch action #1 (add_edge)") and "t2" not in g.nodes


def test_unknown_type_and_invalid_payload_via_direct_action(env) -> None:
    ep, prog = env
    g = WorkflowGraph()
    _, err = apply_action(g, Action("explode", {}), ep, prog)
    assert "unknown action type" in err
    _, err = apply_action(g, Action("add_node", {"node": {"id": "x", "kind": "code", "code": CODE,
                                                          "outputs": {"m": {"type": "tensor"}}}}), ep, prog)
    assert "unknown port type" in err


# --------------------------------------------------------------------------- edit classes
def test_edit_classes_and_touched_nodes(env) -> None:
    ep, prog = env
    g = base_graph(ep, prog)
    control = [
        act("add_edge", edge={"src": "a", "dst": "b"}), act("remove_edge", edge={"src": "a", "dst": "b"}),
        act("remove_node", id="c"), act("modify_node", id="c", patch={"config": {"k": 1}}), act("finish"),
        act("add_node", node={"id": "t2", "kind": "tool", "ref": "load"}),
        act("add_node", node={"id": "o", "kind": "operator", "ref": "op:rowmean"}),
        act("add_node", node={"id": "s2", "kind": "submit"}),
    ]
    execs = [
        act("add_node", node={"id": "c2", "kind": "code", "code": CODE, "outputs": {"m": {}}}),
        act("add_node", node={"id": "l", "kind": "llm", "prompt": "p"}),
        act("modify_node", id="c", patch={"code": CODE}), act("modify_node", id="l", patch={"prompt": "q"}),
        act("modify_node", id="c", patch={"outputs": {"m": {}}, "config": {"a": 1}}),
        act("modify_node", id="c", patch={"inputs": {"x": {}}}),
    ]
    for a in control:
        assert is_control_edit(a, g) and not is_exec_edit(a, g), a
    for a in execs:
        assert is_exec_edit(a, g) and not is_control_edit(a, g), a
    mixed = act("batch", actions=[{"type": "add_edge", "edge": {"src": "a", "dst": "b"}},
                                  {"type": "add_node", "node": {"id": "c3", "kind": "code", "code": CODE, "outputs": {"m": {}}}}])
    assert is_exec_edit(mixed, g) and not is_control_edit(mixed, g)
    assert touched_nodes(control[0]) == {"a", "b"}
    assert touched_nodes(control[2]) == {"c"}
    assert touched_nodes(execs[0]) == {"c2"}
    assert touched_nodes(control[4]) == set()
    assert touched_nodes(mixed) == {"a", "b", "c3"}
