"""Eq. 10: control edits vs executable components CC_Gamma."""
from __future__ import annotations

from scienceclaw.core.actions import Action
from scienceclaw.core.graph import Edge, Node, WorkflowGraph
from scienceclaw.core.operators import OperatorSpec
from scienceclaw.core.program import AgentProgram
from scienceclaw.core.schema import PortSchema
from scienceclaw.core.trace import Evidence, Trace
from scienceclaw.bench.task import EvalResult
from scienceclaw.evolution import extract_instances, split_edits
from scienceclaw.evolution.attribution import EvolutionInstance
from scienceclaw.evolution.operator_abstraction import operator_candidate_with_log
from scienceclaw.evolution.split import _cycle_members, apply_structural, convex_components, is_convex, split_details

from test_evolution_support import build_solve_result, make_episode

L = PortSchema("list")


def test_canonical_repair_split(tmp_path):
    ep = make_episode()
    [inst] = extract_instances(build_solve_result(ep, AgentProgram(), tmp_path))
    control, comps = split_edits(inst, AgentProgram())
    assert [a.type for a in control] == ["modify_node", "remove_edge", "add_edge", "add_edge"]
    assert comps == [{"prep", "post"}]
    det = split_details(inst, AgentProgram())
    assert det.control_steps == [5, 7, 8, 9]
    assert det.touched_nodes == {"post", "prep"}          # post added, prep re-configured inside delta
    assert det.exec_nodes == {"post", "prep"}


def _code(nid: str, ins: list[str], outs: list[str], origin: dict | None = None, kind: str = "code") -> Node:
    return Node(nid, kind, code="def run(inputs, config):\n    return {}\n" if kind == "code" else None,
                prompt="Summarize {x}" if kind == "llm" else None,
                inputs={p: L for p in ins}, outputs={p: L for p in outs}, origin=origin or {"generated": True})


def _g_plus() -> WorkflowGraph:
    nodes = {
        "load": Node("load", "tool", ref="load_data", outputs={"table": L}),
        "a": _code("a", ["x"], ["y"]),
        "b": _code("b", ["y"], ["z"]),
        "c": _code("c", ["x"], ["o"]),
        "d": _code("d", ["items"], ["outputs"], kind="llm"),
        "reg": _code("reg", ["x"], ["o"], origin={"generated": True, "from_operator": "op:known@v2"}),
        "opn": Node("opn", "operator", ref="op:known", inputs={"i": L, "j": L}, outputs={"r": L}),
        "sub": Node("sub", "submit", inputs={"y": L}),
    }
    edges = [Edge.make("load", "table", "a", "x"), Edge.make("a", "y", "b", "y"), Edge.make("b", "z", "sub", "y"),
             Edge.make("load", "table", "c", "x"), Edge.make("c", "o", "opn", "i"),
             Edge.make("load", "table", "reg", "x"), Edge.make("reg", "o", "opn", "j"),
             Edge.make("opn", "r", "d", "items")]
    return WorkflowGraph(nodes, edges)


def _known_program() -> AgentProgram:
    body = WorkflowGraph({"k": _code("k", ["x"], ["o"])})
    op = OperatorSpec("known", 2, "known", "", body, {"x": L}, {"o": L}, {"x": [("k", "x")]}, {"o": ("k", "o")})
    return AgentProgram(operators={"known": op})


def _inst(delta: list[Action], g_plus: WorkflowGraph, e_minus_graph: WorkflowGraph | None = None) -> EvolutionInstance:
    e_plus = Evidence(9, g_plus.to_dict(), None, Trace(), EvalResult(), True)
    e_minus = Evidence(4, e_minus_graph.to_dict(), None, Trace(), EvalResult(), False) if e_minus_graph else None
    return EvolutionInstance("e", 4 if e_minus else None, 9, e_minus, e_plus, delta,
                             list(range(len(delta))), set())


def test_components_exclude_registered_operator_nodes():
    control, comps = split_edits(_inst([], _g_plus()), _known_program())
    assert control == []
    # a-b connected; c and d not connected to each other inside U (opn is outside U); reg is registered
    assert comps == [{"a", "b"}, {"c"}, {"d"}]


def test_registered_node_repaired_in_delta_is_included():
    fix = Action("modify_node", {"id": "reg", "patch": {"code": "def run(inputs, config):\n    return {'o': 1}\n"}})
    control, comps = split_edits(_inst([fix], _g_plus(), _g_plus()), _known_program())
    assert control == []
    assert comps == [{"reg"}]      # the repaired registered node is a candidate; unchanged a/b/c/d are not (F7)


def test_unregistered_operator_origin_counts_as_generated():
    # the operator that produced "reg" is not in the program any more -> reg is an unregistered exec node
    control, comps = split_edits(_inst([], _g_plus()), AgentProgram())
    assert {"reg"} in comps


def test_batch_actions_are_expanded_and_finish_is_not_control():
    batch = Action("batch", {"actions": [
        {"type": "add_node", "node": {"id": "load", "kind": "tool", "ref": "load_data"}},
        {"type": "add_node", "node": {"id": "a", "kind": "code", "code": "def run(i, c):\n    return {}\n",
                                      "outputs": {"y": {"type": "list"}}}},
        {"type": "add_edge", "edge": {"src": "load", "src_port": "table", "dst": "a", "dst_port": "x"}},
    ]})
    g = _g_plus()
    control, comps = split_edits(_inst([batch, Action("finish", {})], g), _known_program())
    assert [a.type for a in control] == ["add_node", "add_edge"]
    assert {"a", "b"} in comps


def test_apply_structural_tracks_graph_before():
    g = WorkflowGraph()
    g = apply_structural(g, Action("add_node", {"node": {"id": "a", "kind": "code", "code": "x"}}))
    g = apply_structural(g, Action("add_node", {"node": {"id": "b", "kind": "tool", "ref": "t"}}))
    g = apply_structural(g, Action("add_edge", {"edge": {"src": "b", "src_port": "o", "dst": "a", "dst_port": "i"}}))
    g = apply_structural(g, Action("modify_node", {"id": "a", "patch": {"config": {"k": 1}}}))
    assert set(g.nodes) == {"a", "b"} and len(g.edges) == 1 and g.nodes["a"].config == {"k": 1}
    g = apply_structural(g, Action("remove_edge", {"edge": {"src": "b", "dst": "a"}}))   # ports omitted
    assert g.edges == []
    g = apply_structural(g, Action("remove_node", {"id": "b"}))
    assert set(g.nodes) == {"a"}
    # malformed payloads are logged and skipped, never raised
    g2 = apply_structural(g, Action("add_node", {"node": {"id": "z", "kind": "bogus"}}))
    assert set(g2.nodes) == {"a"}


# ---------------------------------------------------------------------------------------------- F7: Pi_exec follows delta
def test_unchanged_unrelated_exec_nodes_are_not_operator_candidates():
    """delta only re-configures c: the unrelated loader/formatter-like blocks (a-b, d) must not become operators."""
    fix = Action("modify_node", {"id": "c", "patch": {"config": {"k": 1}}})
    det = split_details(_inst([fix], _g_plus(), _g_plus()), _known_program())
    assert det.components == [{"c"}]
    assert det.exec_nodes == {"c"}
    assert {"a", "b", "d"} <= det.excluded


def test_weakly_connected_exec_neighbours_of_a_touched_node_travel_with_it():
    fix = Action("modify_node", {"id": "a", "patch": {"config": {"k": 1}}})
    det = split_details(_inst([fix], _g_plus(), _g_plus()), _known_program())
    assert det.components == [{"a", "b"}] and det.excluded >= {"c", "d"}


def test_no_e_minus_keeps_every_generated_exec_node():
    det = split_details(_inst([], _g_plus()), _known_program())
    assert det.components == [{"a", "b"}, {"c"}, {"d"}] and det.excluded == {"reg"}


# ---------------------------------------------------------------------------------------------- F2: convex CC_Gamma
def _nonconvex_g() -> WorkflowGraph:
    """A(code) -> T(tool featurize) -> B(code), A.labels -> B.labels, B -> submit."""
    nodes = {
        "A": _code("A", [], ["smiles", "labels"]),
        "T": Node("T", "tool", ref="featurize_molecules", inputs={"smiles": L}, outputs={"X": L}),
        "B": _code("B", ["X", "labels"], ["y"]),
        "sub": Node("sub", "submit", inputs={"y": L}),
    }
    edges = [Edge.make("A", "smiles", "T", "smiles"), Edge.make("T", "X", "B", "X"),
             Edge.make("A", "labels", "B", "labels"), Edge.make("B", "y", "sub", "y")]
    return WorkflowGraph(nodes, edges)


def test_code_tool_code_component_is_split_into_convex_parts():
    g = _nonconvex_g()
    assert g.weak_components({"A", "B"}) == [{"A", "B"}]      # the plain weak component is not convex
    assert not is_convex(g, {"A", "B"})
    det = split_details(_inst([], g), AgentProgram())
    assert det.components == [{"A"}, {"B"}]
    assert all(is_convex(g, c) for c in det.components)
    assert det.convex_splits and det.convex_splits[0]["reentrant_via"] == ["T"]
    assert "convex_splits" in det.summary()


def test_convex_component_is_left_alone():
    g = _g_plus()
    comps, cuts = convex_components(g, {"a", "b"})
    assert comps == [{"a", "b"}] and cuts == []


def test_contracted_components_must_stay_acyclic():
    """Two individually convex components that cycle through outside tools are refined further."""
    nodes = {n: _code(n, [], ["o"]) for n in ("a1", "a2", "m", "b1", "b2", "n")}
    nodes["x"] = Node("x", "tool", ref="tx", outputs={"o": L})
    nodes["y"] = Node("y", "tool", ref="ty", outputs={"o": L})
    edges = [Edge.make("a1", "o", "m", "i"), Edge.make("a2", "o", "m", "j"),
             Edge.make("b1", "o", "n", "i"), Edge.make("b2", "o", "n", "j"),
             Edge.make("a1", "o", "x", "i"), Edge.make("x", "o", "b1", "i"),
             Edge.make("b2", "o", "y", "i"), Edge.make("y", "o", "a2", "i")]
    g = WorkflowGraph(nodes, edges)
    U = {"a1", "a2", "m", "b1", "b2", "n"}
    plain = g.weak_components(U)
    assert len(plain) == 2 and all(is_convex(g, c) for c in plain) and _cycle_members(g, plain) == [0, 1]
    comps, cuts = convex_components(g, U)
    assert set().union(*comps) == U and sum(len(c) for c in comps) == len(U)
    assert _cycle_members(g, comps) == [] and all(is_convex(g, c) for c in comps)
    assert len(comps) >= 3 and any("cycle between operators" in c["why"] for c in cuts)


def test_nonconvex_operator_candidate_is_dropped_and_logged():
    g = _nonconvex_g()
    inst = _inst([], g)
    op, logd = operator_candidate_with_log(inst, {"A", "B"}, AgentProgram(), None, make_episode())
    assert op is None and "not convex" in logd["reason"] and logd["formed"] is False
