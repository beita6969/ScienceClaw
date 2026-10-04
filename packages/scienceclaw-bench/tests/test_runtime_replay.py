"""runtime.replay: reset replay (Eq. 8) and isolated operator execution (boundary replay, Eq. 12)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scienceclaw.bench.task import Budget, ConstraintSpec, EvalResult, Episode, ToolSpec
from scienceclaw.core.graph import Edge, Node, WorkflowGraph
from scienceclaw.core.operators import Contract, OperatorSpec
from scienceclaw.core.program import AgentProgram
from scienceclaw.core.schema import PortSchema
from scienceclaw.runtime.executor import Executor
from scienceclaw.runtime.replay import fresh_dir, replay, run_operator_isolated
from scienceclaw.runtime.values import load_value, outputs_match


def make_episode() -> Episode:
    def load(inputs: dict, config: dict) -> dict:
        rng = np.random.default_rng(3)
        return {"X": rng.normal(size=(30, 4)), "y": rng.normal(size=30)}

    return Episode(id="ep-r", discipline="FoR49", family="Physical & Earth", split="src", task_type="regression",
                   objective="fit", required_output=PortSchema("array", ("n",)),
                   tools=[ToolSpec("load", "visible data", {}, {"X": PortSchema("array", ("n", 4)),
                                                              "y": PortSchema("array", ("n",))}, load)],
                   constraints=[ConstraintSpec("finite", "finite", lambda y, t: (bool(np.isfinite(y).all()), ""))],
                   budget=Budget(max_node_s=30), tolerance={"rtol": 1e-9, "atol": 1e-12},
                   _evaluate=lambda y, t: EvalResult(primary=1.0, accepted=True))


STANDARDIZE = ("import numpy as np\n"
               "def run(inputs, config):\n"
               "    X = inputs['X']\n"
               "    return {'Z': (X - X.mean(0)) / X.std(0)}\n")
FIT = ("import numpy as np\n"
       "def run(inputs, config):\n"
       "    Z, y = inputs['Z'], inputs['y']\n"
       "    noise = np.random.normal(scale=1e-3, size=Z.shape[1])   # seeded by the worker\n"
       "    w = np.linalg.lstsq(Z, y, rcond=None)[0] + noise\n"
       "    return {'pred': Z @ w}\n")
LLM_MAP = "Scale {item}"


def build_graph(ep: Episode) -> WorkflowGraph:
    spec = ep.tool("load")
    nodes = [
        Node("t", "tool", ref="load", inputs={}, outputs=dict(spec.outputs)),
        Node("std", "code", code=STANDARDIZE, config={"seed": 1}, inputs={"X": PortSchema("array")},
             outputs={"Z": PortSchema("array")}),
        Node("fit", "code", code=FIT, config={"seed": 5}, inputs={"Z": PortSchema("array"), "y": PortSchema("array")},
             outputs={"pred": PortSchema("array")}),
        Node("s", "submit", inputs={"y": ep.required_output}),
    ]
    edges = [Edge.make("t", "X", "std", "X"), Edge.make("std", "Z", "fit", "Z"), Edge.make("t", "y", "fit", "y"),
             Edge.make("fit", "pred", "s", "y")]
    g = WorkflowGraph({n.id: n for n in nodes}, edges)
    assert g.validate() == []
    return g


def test_replay_equals_execute_for_deterministic_graph(tmp_path) -> None:
    ep = make_episode()
    g = build_graph(ep)
    ex = Executor(ep, AgentProgram(), None, tmp_path / "session")
    # multi-turn style execution: first a partial graph, then the full one reusing the checkpoint
    partial = WorkflowGraph({k: g.nodes[k] for k in ("t", "std")}, [g.edges[0]])
    cp, _, _ = ex.execute(partial, ex.new_checkpoint())
    _, records_exec, y_exec = ex.execute(g, cp)
    assert records_exec["std"].cached and not records_exec["fit"].cached
    y_rep, trace = replay(g, ep, AgentProgram(), None, tmp_path / "replay")
    assert outputs_match(y_exec, y_rep, ep.tolerance)
    assert trace.order == g.topo_order() and list(trace.records) == trace.order
    assert all(r.status == "ok" and not r.cached for r in trace.records.values())
    assert trace.wall_s > 0 and Path(trace.run_dir) == tmp_path / "replay"
    # the trace points into the fresh replay directory, never into the session checkpoint
    for r in trace.records.values():
        for p in list(r.output_refs.values()) + list(r.input_refs.values()):
            assert Path(p).is_relative_to(tmp_path / "replay")
    # input_refs hold exactly the values fed to each node
    np.testing.assert_allclose(load_value(trace.records["fit"].input_refs["Z"]),
                               load_value(trace.records["std"].output_refs["Z"]))
    ev = ep.evaluate(y_rep, trace)
    assert ev.completed and ev.h == {"finite": True}


def test_replay_uses_a_fresh_directory_each_time(tmp_path) -> None:
    ep = make_episode()
    g = build_graph(ep)
    y1, t1 = replay(g, ep, AgentProgram(), None, tmp_path / "rep")
    y2, t2 = replay(g, ep, AgentProgram(), None, tmp_path / "rep")
    assert t1.run_dir != t2.run_dir and Path(t2.run_dir).name == "rep-1"
    assert outputs_match(y1, y2, ep.tolerance)
    assert fresh_dir(tmp_path / "unused") == tmp_path / "unused"


def test_replay_aggregates_llm_usage(tmp_path) -> None:
    ep = make_episode()

    class Stub:
        def chat_many(self, role, batch, **kw):
            return [SimpleNamespace(text="2", error=None, usage={"prompt_tokens": 5, "completion_tokens": 1, "calls": 1})
                    for _ in batch]

    items = Node("i", "code", code="def run(inputs, config):\n    return {'items': ['a', 'b']}\n",
                 outputs={"items": PortSchema("list")})
    llm_node = Node("l", "llm", prompt=LLM_MAP, config={"parse": "number"}, inputs={"items": PortSchema("list")},
                    outputs={"outputs": PortSchema("list")})
    g = WorkflowGraph({"i": items, "l": llm_node}, [Edge.make("i", "items", "l", "items")])
    y, trace = replay(g, ep, AgentProgram(), Stub(), tmp_path / "r")
    assert y is None
    assert trace.llm_usage == {"prompt_tokens": 10, "completion_tokens": 2, "calls": 2}
    assert load_value(trace.records["l"].output_refs["outputs"]) == [2.0, 2.0]


def test_boundary_replay_of_a_component_reproduces_recorded_outputs(tmp_path) -> None:
    """The evolution use case: component U = {std, fit} of G+ becomes an Operator; restoring the
    recorded boundary inputs from tau+ and executing it in isolation reproduces tau+ at the boundary."""
    ep = make_episode()
    g = build_graph(ep)
    y, trace = replay(g, ep, AgentProgram(), None, tmp_path / "plus")
    U = {"std", "fit"}
    incoming, outgoing = g.boundary(U)
    body = g.subgraph(U)
    inputs_schema = {f"{e.dst}_{e.dst_port}": g.nodes[e.dst].inputs[e.dst_port] for e in incoming}
    input_map = {f"{e.dst}_{e.dst_port}": [(e.dst, e.dst_port)] for e in incoming}
    output_map = {f"{e.src}_{e.src_port}": (e.src, e.src_port) for e in outgoing}
    outputs_schema = {k: g.nodes[s].outputs[p] for k, (s, p) in output_map.items()}
    op = OperatorSpec("fit_std", 1, "standardize+fit", "candidate", body, inputs_schema, outputs_schema,
                      input_map, output_map, Contract(post=[{"port": "fit_pred", "check": "finite"}]))
    boundary_inputs = {k: load_value(trace.records[t[0][0]].input_refs[t[0][1]]) for k, t in input_map.items()}
    outs, rec = run_operator_isolated(op, boundary_inputs, AgentProgram(), None, tmp_path / "breplay", episode=ep)
    assert rec.status == "ok", rec.error
    assert rec.kind == "operator" and rec.node_id == "op:fit_std" and rec.contract_violations == []
    for k, (s, p) in output_map.items():
        assert outputs_match(outs[k], load_value(trace.records[s].output_refs[p]), ep.tolerance)
    assert set(rec.input_refs) == set(input_map) and all(Path(v).exists() for v in rec.output_refs.values())
    # repeated boundary replay (stochastic replication with fixed seeds) is identical
    outs2, _ = run_operator_isolated(op, boundary_inputs, AgentProgram(), None, tmp_path / "breplay", episode=ep)
    assert outputs_match(outs, outs2, ep.tolerance)


def test_run_operator_isolated_failure_and_missing_episode(tmp_path) -> None:
    body = WorkflowGraph({"t": Node("t", "tool", ref="load", outputs={"X": PortSchema("array")})})
    op = OperatorSpec("needs_tool", 1, "n", "d", body, {}, {"X": PortSchema("array")}, {}, {"X": ("t", "X")}, Contract())
    outs, rec = run_operator_isolated(op, {}, AgentProgram(), None, tmp_path / "a")
    assert outs == {} and rec.status == "error" and "unknown tool 'load'" in rec.error
    outs, rec = run_operator_isolated(op, {}, AgentProgram(), None, tmp_path / "b", episode=make_episode())
    assert rec.status == "ok" and outs["X"].shape == (30, 4)
