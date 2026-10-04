"""The optional "wire" field: a node is added / rewired together with its input dependencies atomically."""
from __future__ import annotations

import json

from scienceclaw.core.actions import Action, apply_action, is_control_edit, is_exec_edit, parse_action
from scienceclaw.core.graph import WorkflowGraph

from test_actions_apply import CODE, make_episode, make_program


def _add(g, node, wire=None):
    payload = {"node": node}
    if wire is not None:
        payload["wire"] = wire
    return apply_action(g, Action("add_node", payload, []), make_episode(), make_program())


def _graph_with_tool():
    g, err = _add(WorkflowGraph(), {"id": "t", "kind": "tool", "ref": "load"})
    assert err is None
    return g


def test_add_node_with_wire_creates_edges():
    g = _graph_with_tool()
    node = {"id": "c", "kind": "code", "code": CODE, "inputs": {"x": {"type": "array", "shape": ["n", 3], "unit": "K"}},
            "outputs": {"m": {"type": "array", "shape": ["n"]}}}
    g, err = _add(g, node, {"x": "t.data"})
    assert err is None
    assert [e.render() for e in g.edges] == ["t.data -> c.x"]
    g, err = _add(g, {"id": "s", "kind": "submit"}, {"y": "c.m"})
    assert err is None and len(g.edges) == 2 and g.validate() == []


def test_wire_error_is_atomic():
    g = _graph_with_tool()
    node = {"id": "c", "kind": "code", "code": CODE, "inputs": {"x": {"type": "array"}}, "outputs": {"m": {"type": "array"}}}
    g2, err = _add(g, node, {"x": "nope.data"})
    assert err is not None and "wire['x']" in err
    assert set(g2.nodes) == {"t"} and g2.edges == []          # node not added either


def test_wire_with_unit_conversion():
    g = _graph_with_tool()
    node = {"id": "c", "kind": "code", "code": CODE, "inputs": {"x": {"type": "array", "unit": "degC"}},
            "outputs": {"m": {"type": "array"}}}
    g2, err = _add(g, node, {"x": "t.data"})
    assert err is not None and "conversion" in err
    g3, err = _add(g, node, {"x": {"from": "t.data", "conversion": {"from": "K", "to": "degC", "factor": 1.0, "offset": -273.15}}})
    assert err is None and g3.edges[0].conversion_dict["offset"] == -273.15


def test_modify_node_rewire_replaces_incoming_edge():
    g = _graph_with_tool()
    g, err = _add(g, {"id": "t2", "kind": "tool", "ref": "load"})
    node = {"id": "c", "kind": "code", "code": CODE, "inputs": {"x": {"type": "array"}}, "outputs": {"m": {"type": "array"}}}
    g, err = _add(g, node, {"x": "t.data"})
    assert err is None
    a = Action("modify_node", {"id": "c", "patch": {"wire": {"x": "t2.data"}}}, [])
    g, err = apply_action(g, a, make_episode(), make_program())
    assert err is None and [e.render() for e in g.edges] == ["t2.data -> c.x"]
    assert is_control_edit(a) and not is_exec_edit(a)


def test_parse_action_keeps_wire():
    txt = '{"thought": "x", "action": {"type": "add_node", "node": {"id": "s", "kind": "submit"}, "wire": {"y": "c.m"}}, "uses": []}'
    a, err = parse_action(txt)
    assert err is None and a.payload["wire"] == {"y": {"src": "c", "src_port": "m"}}


def test_parsed_wire_action_applies_end_to_end():
    """parse_action normalizes the wire; applying the parsed Action must accept the normalized form."""
    g = _graph_with_tool()
    txt = json.dumps({"thought": "t", "action": {"type": "add_node", "node": {
        "id": "c", "kind": "code", "code": CODE, "inputs": {"x": {"type": "array", "shape": ["n", 3], "unit": "K"}},
        "outputs": {"m": {"type": "array", "shape": ["n"]}}}, "wire": {"x": "t.data"}}, "uses": []})
    a, err = parse_action(txt)
    assert err is None
    g2, err = apply_action(g, a, make_episode(), make_program())
    assert err is None, err
    assert [e.render() for e in g2.edges] == ["t.data -> c.x"]


def test_modify_node_inputs_patch_drops_edges_of_removed_ports():
    """A port dropped by patch.inputs takes its edge with it (FoR31: the edit used to fail forever on the dangling edge)."""
    g = _graph_with_tool()
    g, err = _add(g, {"id": "t2", "kind": "tool", "ref": "load"})
    node = {"id": "c", "kind": "code", "code": CODE, "inputs": {"x": {"type": "array"}, "z": {"type": "array"}},
            "outputs": {"m": {"type": "array"}}}
    g, err = _add(g, node, {"x": "t.data", "z": "t2.data"})
    assert err is None and len(g.edges) == 2
    a = Action("modify_node", {"id": "c", "patch": {"inputs": {"x": {"type": "array"}}}}, [])
    g, err = apply_action(g, a, make_episode(), make_program())
    assert err is None and [e.render() for e in g.edges] == ["t.data -> c.x"] and g.validate() == []
    b = Action("modify_node", {"id": "c", "patch": {"inputs": {"w": {"type": "array"}}, "wire": {"w": "t2.data"}}}, [])
    g, err = apply_action(g, b, make_episode(), make_program())
    assert err is None and [e.render() for e in g.edges] == ["t2.data -> c.w"]


def test_modify_node_outputs_patch_still_rejects_removing_a_consumed_output():
    g = _graph_with_tool()
    node = {"id": "c", "kind": "code", "code": CODE, "inputs": {"x": {"type": "array"}}, "outputs": {"m": {"type": "array"}, "k": {"type": "array"}}}
    g, err = _add(g, node, {"x": "t.data"})
    g, err = _add(g, {"id": "s", "kind": "submit"}, {"y": "c.m"})
    assert err is None
    a = Action("modify_node", {"id": "c", "patch": {"outputs": {"k": {"type": "array"}}}}, [])
    g2, err = apply_action(g, a, make_episode(), make_program())
    assert g2 is g and "no output port 'm'" in err and "batch" in err
