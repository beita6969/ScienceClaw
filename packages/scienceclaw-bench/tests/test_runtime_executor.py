"""runtime.executor: Eq. 7 execution semantics, checkpoint reuse, node kinds, visible feedback."""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scienceclaw.bench.task import Budget, ConstraintSpec, EvalResult, Episode, ToolSpec
from scienceclaw.core.actions import Action
from scienceclaw.core.graph import Edge, Node, WorkflowGraph
from scienceclaw.core.operators import Contract, OperatorSpec
from scienceclaw.core.program import AgentProgram
from scienceclaw.core.schema import PortSchema
from scienceclaw.runtime.executor import Checkpoint, Executor, parse_llm_text, render_template
from scienceclaw.runtime.values import load_value

HIDDEN_SCORE = 0.987654321


# ----------------------------------------------------------------------------- fixtures
class Counter:
    def __init__(self) -> None:
        self.n = 0


def make_episode(tool_calls: Counter, eval_calls: Counter, max_node_s: float = 30.0, max_llm_items: int = 256,
                 required=PortSchema("array", ("n",))) -> Episode:
    def load(inputs: dict, config: dict) -> dict:
        tool_calls.n += 1
        rng = np.random.default_rng(config.get("seed", 0))
        return {"data": rng.uniform(250.0, 300.0, size=(12, 3))}

    def slow(inputs: dict, config: dict) -> dict:
        time.sleep(3.0)
        return {"v": 1.0}

    def evaluator(y, trace) -> EvalResult:
        eval_calls.n += 1
        return EvalResult(metrics={"hidden_auc": HIDDEN_SCORE}, primary=HIDDEN_SCORE, accepted=True)

    tools = [ToolSpec("load", "visible temperatures", {}, {"data": PortSchema("array", ("n", 3), unit="K")}, load),
             ToolSpec("slow", "slow tool", {}, {"v": PortSchema("number")}, slow)]
    constraints = [
        ConstraintSpec("finite_output", "y must be finite", lambda y, t: (bool(np.isfinite(np.asarray(y, float)).all()), "all finite")),
        ConstraintSpec("hidden_check", "secret", lambda y, t: (False, "HIDDEN_MSG"), visible=False),
    ]
    return Episode(id="ep-x", discipline="FoR37", family="Physical & Earth", split="src", task_type="regression",
                   objective="compute row means", required_output=required, tools=tools, constraints=constraints,
                   budget=Budget(max_node_s=max_node_s, max_llm_items=max_llm_items),
                   _evaluate=evaluator, _dev_evaluate=lambda y: {"dev_mae": 0.25})


def tool(nid: str, ep: Episode, name: str = "load", config=None) -> Node:
    spec = ep.tool(name)
    return Node(nid, "tool", ref=name, config=dict(config or {}), inputs=dict(spec.inputs), outputs=dict(spec.outputs))


def code(nid: str, src: str, ins: dict, outs: dict, config=None) -> Node:
    return Node(nid, "code", code=src, config=dict(config or {}),
                inputs={k: PortSchema.from_dict(v) for k, v in ins.items()},
                outputs={k: PortSchema.from_dict(v) for k, v in outs.items()})


def submit(ep: Episode, nid: str = "s") -> Node:
    return Node(nid, "submit", inputs={"y": ep.required_output})


def graph(nodes: list[Node], edges: list[tuple]) -> WorkflowGraph:
    g = WorkflowGraph({n.id: n for n in nodes}, [Edge.make(*e) for e in edges])
    assert g.validate() == [], g.validate()
    return g


ROWMEAN = "def run(inputs, config):\n    return {'m': inputs['x'].mean(axis=1) * config.get('scale', 1.0)}\n"
ARR = {"type": "array"}


class StubLLM:
    """Minimal chat_many stub (no network): answers with fn(user_text)."""

    def __init__(self, fn, fail: bool = False) -> None:
        self.fn, self.fail, self.calls = fn, fail, []

    def chat_many(self, role, batch, **kw):
        self.calls.append({"role": role, "batch": batch, "kw": kw})
        if self.fail:
            raise RuntimeError("gateway down")
        return [SimpleNamespace(text=self.fn(m[-1]["content"]), error=None,
                                usage={"prompt_tokens": 10, "completion_tokens": 2, "calls": 1}) for m in batch]


@pytest.fixture()
def env(tmp_path):
    tc, evc = Counter(), Counter()
    ep = make_episode(tc, evc)
    return SimpleNamespace(ep=ep, tc=tc, evc=evc, prog=AgentProgram(), run_dir=tmp_path / "run")


# ----------------------------------------------------------------------------- basic
def test_pipeline_via_apply_and_visible_feedback(env) -> None:
    ex = Executor(env.ep, env.prog, None, env.run_dir)
    g, cp = WorkflowGraph(), ex.new_checkpoint()
    actions = [
        Action("add_node", {"node": {"id": "t", "kind": "tool", "ref": "load"}}),
        Action("add_node", {"node": {"id": "c", "kind": "code", "code": ROWMEAN, "inputs": {"x": {"type": "array", "unit": "K"}},
                                     "outputs": {"m": {"type": "array", "shape": ["n"]}}}}, ["skill:rowmean"]),
        Action("add_edge", {"edge": {"src": "t", "dst": "c", "dst_port": "x"}}),
        Action("add_node", {"node": {"id": "s", "kind": "submit"}}),
        Action("add_edge", {"edge": {"src": "c", "dst": "s"}}),
    ]
    for k, a in enumerate(actions):
        g, cp, fb, y = ex.apply(g, cp, a, k)
        assert fb.action_ok, fb.action_error
    data = load_value(fb.records["t"].output_refs["data"])
    np.testing.assert_allclose(y, data.mean(axis=1))
    assert [r.status for r in fb.records.values()] == ["ok", "ok", "ok"]
    assert fb.records["t"].cached and fb.records["c"].cached and not fb.records["s"].cached
    assert fb.node_runs == 1 and fb.submit_ready
    assert g.nodes["c"].origin["step"] == 1 and g.nodes["c"].origin["uses"] == ["skill:rowmean"]
    # input_refs point at the pickled values actually fed in
    assert fb.records["c"].input_refs["x"] == fb.records["t"].output_refs["data"]
    np.testing.assert_allclose(load_value(fb.records["s"].input_refs["y"]), y)
    assert fb.visible_constraints == {"finite_output": [True, "all finite"]}
    assert fb.dev == {"dev_mae": 0.25}
    text = fb.render()
    assert "[c] code ok" in text and "finite_output: PASS" in text and "dev_mae" in text
    assert env.tc.n == 1


def test_feedback_never_contains_hidden_information(env) -> None:
    ex = Executor(env.ep, env.prog, None, env.run_dir)
    g = graph([tool("t", env.ep), code("c", ROWMEAN, {"x": ARR}, {"m": ARR}), submit(env.ep)],
              [("t", "data", "c", "x"), ("c", "m", "s", "y")])
    _, cp, fb, y = ex.apply(g, ex.new_checkpoint(), Action("finish", {}), 0)
    assert y is not None and fb.submit_ready
    blob = fb.render(max_chars=100_000) + json.dumps(fb.to_dict(), default=str)
    for secret in (str(HIDDEN_SCORE), "0.98765", "hidden_auc", "HIDDEN_MSG", "hidden_check"):
        assert secret not in blob
    assert env.evc.n == 0                      # the hidden evaluator is never called by the runtime


# ----------------------------------------------------------------------------- conversion
def test_unit_conversion_edge_is_applied(env) -> None:
    ex = Executor(env.ep, env.prog, None, env.run_dir)
    ident = "def run(inputs, config):\n    return {'m': inputs['x'][:, 0]}\n"
    conv = {"from": "K", "to": "degC", "factor": 1.0, "offset": -273.15}
    g = graph([tool("t", env.ep), code("c", ident, {"x": {"type": "array", "unit": "degC"}}, {"m": ARR}), submit(env.ep)],
              [("t", "data", "c", "x", conv), ("c", "m", "s", "y")])
    _, records, y = ex.execute(g, ex.new_checkpoint())
    data = load_value(records["t"].output_refs["data"])
    np.testing.assert_allclose(y, data[:, 0] - 273.15)
    ref = records["c"].input_refs["x"]
    assert ref != records["t"].output_refs["data"] and Path(ref).parent.name == "inputs"
    np.testing.assert_allclose(load_value(ref), data - 273.15)


# ----------------------------------------------------------------------------- checkpoint reuse
def test_fingerprint_reuse_reruns_only_modified_node_and_descendants(env) -> None:
    ex = Executor(env.ep, env.prog, None, env.run_dir)
    plus = "def run(inputs, config):\n    return {'m': inputs['x'] + config.get('k', 0)}\n"
    colsum = "def run(inputs, config):\n    return {'m': inputs['x'].sum(axis=1)}\n"
    g = graph([tool("t", env.ep), code("a", ROWMEAN, {"x": ARR}, {"m": ARR}), code("b", plus, {"x": ARR}, {"m": ARR}),
               code("c", colsum, {"x": ARR}, {"m": ARR}), submit(env.ep)],
              [("t", "data", "a", "x"), ("a", "m", "b", "x"), ("t", "data", "c", "x"), ("b", "m", "s", "y")])
    cp, records, y0 = ex.execute(g, ex.new_checkpoint())
    assert all(not r.cached and r.status == "ok" for r in records.values())
    # modify b's config -> only b and its descendant s re-run
    g2, cp2, fb, y1 = ex.apply(g, cp, Action("modify_node", {"id": "b", "patch": {"config": {"k": 10}}}), 1)
    rerun = {n for n, r in fb.records.items() if not r.cached}
    assert rerun == {"b", "s"}
    np.testing.assert_allclose(y1, y0 + 10)
    # modify a's code -> a, b, s re-run; t and c stay cached
    g3, cp3, fb, y2 = ex.apply(g2, cp2, Action("modify_node", {"id": "a", "patch": {"code": ROWMEAN.replace("mean", "max")}}), 2)
    assert {n for n, r in fb.records.items() if not r.cached} == {"a", "b", "s"}
    assert env.tc.n == 1                                     # the tool ran exactly once
    # reverting to the first graph is fully cached
    _, records4, y4 = ex.execute(g, cp3)
    assert all(r.cached for r in records4.values())
    np.testing.assert_allclose(y4, y0)
    assert set(cp3.records) >= {r.fingerprint for r in records.values()}


def test_checkpoint_from_other_operator_library_is_not_reused(env) -> None:
    ex = Executor(env.ep, env.prog, None, env.run_dir)
    g = graph([tool("t", env.ep), code("c", ROWMEAN, {"x": ARR}, {"m": ARR}), submit(env.ep)],
              [("t", "data", "c", "x"), ("c", "m", "s", "y")])
    cp, _, _ = ex.execute(g, ex.new_checkpoint())
    other = AgentProgram(operators={"k": make_scale_op("k")})
    ex2 = Executor(env.ep, other, None, env.run_dir)
    _, records, _ = ex2.execute(g, cp)
    assert not any(r.cached for r in records.values())
    same = Executor(env.ep, AgentProgram(), None, env.run_dir)
    _, records, _ = same.execute(g, cp)
    assert all(r.cached for r in records.values())


# ----------------------------------------------------------------------------- pending / skipped
def test_pending_and_skipped_semantics(env) -> None:
    ex = Executor(env.ep, env.prog, None, env.run_dir)
    boom = "def run(inputs, config):\n    raise ValueError('bad input ' + str(inputs['x'].shape))\n"
    g = graph([tool("t", env.ep), code("a", boom, {"x": ARR}, {"m": ARR}), code("b", ROWMEAN, {"x": ARR}, {"m": ARR}),
               submit(env.ep), code("p", ROWMEAN, {"x": ARR}, {"m": ARR}), code("q", ROWMEAN, {"x": ARR}, {"m": ARR})],
              [("t", "data", "a", "x"), ("a", "m", "b", "x"), ("b", "m", "s", "y"), ("p", "m", "q", "x")])
    _, records, y = ex.execute(g, ex.new_checkpoint())
    st = {n: r.status for n, r in records.items()}
    assert st == {"t": "ok", "a": "error", "b": "skipped", "s": "skipped", "p": "pending", "q": "pending"}
    assert y is None
    assert "ValueError: bad input (12, 3)" in records["a"].error and 'File "node_code.py"' in records["a"].error
    assert "['a']" in records["b"].error
    assert "not connected" in records["p"].error and "waiting on pending node(s) ['p']" in records["q"].error
    # an input listed in config.optional_inputs may stay unwired: the node runs without it
    g.nodes["p"].config["optional_inputs"] = ["x"]
    g.nodes["p"].code = ("import numpy as np\ndef run(inputs, config):\n"
                         "    return {'m': np.full((2, 3), 1.0 if 'x' in inputs else 2.0)}\n")
    _, records, _ = ex.execute(g, ex.new_checkpoint())
    assert records["p"].status == "ok" and records["q"].status == "ok"
    np.testing.assert_allclose(load_value(records["q"].output_refs["m"]), [2.0, 2.0])


def test_submit_without_input_is_pending_and_errors_are_cached(env) -> None:
    ex = Executor(env.ep, env.prog, None, env.run_dir)
    g = graph([tool("t", env.ep), submit(env.ep)], [])
    cp, records, y = ex.execute(g, ex.new_checkpoint())
    assert records["s"].status == "pending" and y is None
    boom = "def run(inputs, config):\n    raise RuntimeError('deterministic failure')\n"
    g = graph([tool("t", env.ep), code("a", boom, {"x": ARR}, {"m": ARR})], [("t", "data", "a", "x")])
    cp, records, _ = ex.execute(g, cp)
    assert records["a"].status == "error" and not records["a"].cached
    _, records, _ = ex.execute(g, cp)
    assert records["a"].status == "error" and records["a"].cached      # unchanged failing node is not re-run


def test_submit_schema_mismatch_is_reported(env) -> None:
    ex = Executor(env.ep, env.prog, None, env.run_dir)
    ident = "def run(inputs, config):\n    return {'m': inputs['x']}\n"
    g = graph([tool("t", env.ep), code("c", ident, {"x": ARR}, {"m": ARR}), submit(env.ep)],
              [("t", "data", "c", "x"), ("c", "m", "s", "y")])
    _, records, y = ex.execute(g, ex.new_checkpoint())
    assert y.shape == (12, 3)
    assert any("shape [12, 3] does not match required ['n']" in v for v in records["s"].contract_violations)


# ----------------------------------------------------------------------------- integrity + timeouts
def test_integrity_violation_blocks_execution(env) -> None:
    ex = Executor(env.ep, env.prog, None, env.run_dir)
    leaky = "import socket\ndef run(inputs, config):\n    return {'m': open('/Users/admin/Datasets/y.csv').read()}\n"
    g = graph([tool("t", env.ep), code("c", leaky, {"x": ARR}, {"m": ARR}), submit(env.ep)],
              [("t", "data", "c", "x"), ("c", "m", "s", "y")])
    _, _, fb, y = ex.apply(g, ex.new_checkpoint(), Action("finish", {}), 0)
    assert y is None
    assert fb.records["c"].status == "error" and "integrity scan rejected" in fb.records["c"].error
    assert fb.records["s"].status == "skipped"
    assert any(v.startswith("c: line 1") and "socket" in v for v in fb.integrity_violations)
    assert not (env.run_dir / "work").exists()               # no subprocess was started
    assert "Integrity violations" in fb.render()


def test_tool_and_code_timeouts(tmp_path) -> None:
    tc, evc = Counter(), Counter()
    ep = make_episode(tc, evc, max_node_s=0.5)
    ex = Executor(ep, AgentProgram(), None, tmp_path)
    g = graph([tool("w", ep, "slow")], [])
    t0 = time.monotonic()
    _, records, _ = ex.execute(g, ex.new_checkpoint())
    assert time.monotonic() - t0 < 2.5
    assert records["w"].status == "error" and "timeout" in records["w"].error
    sleepy = "import time\ndef run(inputs, config):\n    time.sleep(30)\n    return {'m': 1}\n"
    g = graph([code("z", sleepy, {}, {"m": ARR}, config={"timeout_s": 100})], [])   # capped by budget.max_node_s
    _, records, _ = ex.execute(g, ex.new_checkpoint())
    assert records["z"].status == "error" and "timeout" in records["z"].error


def test_independent_nodes_run_concurrently(env) -> None:
    ex = Executor(env.ep, env.prog, None, env.run_dir, max_parallel_subprocs=4)
    stamp = "import time\ndef run(inputs, config):\n    t0 = time.time()\n    time.sleep(1.0)\n    return {'m': [t0, time.time()]}\n"
    g = graph([code("a", stamp, {}, {"m": {"type": "list"}}), code("b", stamp, {}, {"m": {"type": "list"}})], [])
    _, records, _ = ex.execute(g, ex.new_checkpoint())
    (a0, a1), (b0, b1) = (load_value(records[n].output_refs["m"]) for n in ("a", "b"))
    assert max(a0, b0) < min(a1, b1)                        # the two runs overlapped in time


# ----------------------------------------------------------------------------- rejected actions
def test_rejected_action_keeps_graph_and_does_not_rerun(env) -> None:
    ex = Executor(env.ep, env.prog, None, env.run_dir)
    g = graph([tool("t", env.ep), code("c", ROWMEAN, {"x": ARR}, {"m": ARR}), submit(env.ep)],
              [("t", "data", "c", "x"), ("c", "m", "s", "y")])
    g1, cp1, fb1, y1 = ex.apply(g, ex.new_checkpoint(), Action("finish", {}), 0)
    runs_before = ex.usage()["node_runs"]
    g2, cp2, fb2, y2 = ex.apply(g1, cp1, Action("add_edge", {"edge": {"src": "t", "dst": "nope"}}), 1)
    assert not fb2.action_ok and "unknown destination node 'nope'" in fb2.action_error
    assert g2 is g1 and cp2 is cp1 and ex.usage()["node_runs"] == runs_before and fb2.node_runs == 0
    np.testing.assert_allclose(y2, y1)
    assert all(r.cached for r in fb2.records.values())
    assert "REJECTED" in fb2.render()
    # an invalid graph handed to execute directly is reported, not run
    bad = WorkflowGraph({"s": submit(env.ep)}, [Edge.make("x", "o", "s", "y")])
    _, records, y = ex.execute(bad, cp2)
    assert y is None and records["s"].status == "pending" and "graph is invalid" in records["s"].error


# ----------------------------------------------------------------------------- llm nodes
def test_llm_node_maps_prompt_over_items(env) -> None:
    items_code = ("def run(inputs, config):\n"
                  "    return {'items': [{'name': f's{i}', 'text': f'sample {i}'} for i in range(3)]}\n")
    llm = StubLLM(lambda text: "not sure" if "sample 1" in text else "Answer: 3")
    ex = Executor(env.ep, env.prog, llm, env.run_dir)
    g = WorkflowGraph()
    g.nodes["mk"] = code("mk", items_code, {}, {"items": {"type": "list"}})
    g.nodes["l"] = Node("l", "llm", prompt="Rate {text} ({name}) in {unit}; raw={item}", config={"parse": "number", "max_tokens": 50},
                        inputs={"items": PortSchema("list"), "unit": PortSchema("any")}, outputs={"outputs": PortSchema("list")})
    g.nodes["u"] = code("u", "def run(inputs, config):\n    return {'unit': 'kelvin'}\n", {}, {"unit": {"type": "text"}})
    g.edges = [Edge.make("mk", "items", "l", "items"), Edge.make("u", "unit", "l", "unit")]
    assert g.validate() == []
    cp, records, _ = ex.execute(g, ex.new_checkpoint())
    rec = records["l"]
    assert rec.status == "ok", rec.error
    assert load_value(rec.output_refs["outputs"]) == [3.0, None, 3.0]
    assert rec.outputs_summary["outputs"]["parse_failures"] == 1 and rec.outputs_summary["outputs"]["n_items"] == 3
    assert rec.llm_usage == {"prompt_tokens": 30, "completion_tokens": 6, "calls": 3}
    call = llm.calls[0]
    assert call["role"] == "executor" and call["kw"]["tag"] == "ep-x" and call["kw"]["max_tokens"] == 50
    user = call["batch"][0][1]["content"]
    assert user.startswith("Rate sample 0 (s0) in kelvin; raw={") and '"name": "s0"' in user
    assert "single number" in call["batch"][0][0]["content"]
    # cached on the next execution: no new LLM calls
    ex.execute(g, cp)
    assert len(llm.calls) == 1


def test_llm_node_budget_and_transient_failure(tmp_path) -> None:
    tc, evc = Counter(), Counter()
    ep = make_episode(tc, evc, max_llm_items=2)
    mk = code("mk", "def run(inputs, config):\n    return {'items': ['a', 'b', 'c']}\n", {}, {"items": {"type": "list"}})
    ln = Node("l", "llm", prompt="Echo {item}", config={}, inputs={"items": PortSchema("list")},
              outputs={"outputs": PortSchema("list")})
    g = graph([mk, ln], [("mk", "items", "l", "items")])
    llm = StubLLM(lambda t: t)
    ex = Executor(ep, AgentProgram(), llm, tmp_path / "a")
    _, records, _ = ex.execute(g, ex.new_checkpoint())
    assert records["l"].status == "error" and "at most 2 items" in records["l"].error and not llm.calls
    failing = StubLLM(lambda t: t, fail=True)
    ep2 = make_episode(tc, evc)
    ex2 = Executor(ep2, AgentProgram(), failing, tmp_path / "b")
    cp, records, _ = ex2.execute(g, ex2.new_checkpoint())
    assert records["l"].status == "error" and "gateway down" in records["l"].error
    ex2.execute(g, cp)
    assert len(failing.calls) == 2                          # transient failures are not cached
    _, records, _ = Executor(ep2, AgentProgram(), None, tmp_path / "c").execute(g, None)
    assert "no LLM client" in records["l"].error


def test_template_and_parsing_helpers() -> None:
    assert render_template("Classify: {text}", {"text": "abc"}) == "Classify: abc"
    assert render_template("x={item}", "plain") == "x=plain"
    assert render_template('Return {"label": ...} for {text}', {"text": "t"}) == 'Return {"label": ...} for t'
    assert render_template("{v:.2f} {missing}", {"v": 1.2345}) == "1.23 {missing}"
    assert parse_llm_text("```json\n{\"a\": 1}\n```", "json") == {"a": 1}
    assert parse_llm_text("[1, 2]", "json") == [1, 2]
    assert parse_llm_text("the value is -3.5e-2 units", "number") == pytest.approx(-0.035)
    assert parse_llm_text("no digits", "number") is None
    assert parse_llm_text("Positive.", "choice", ["positive", "negative"]) == "positive"
    assert parse_llm_text("I'd say negative overall", "choice", ["positive", "negative"]) == "negative"
    assert parse_llm_text("positive or negative", "choice", ["positive", "negative"]) is None
    assert parse_llm_text("   ", "text") is None


# ----------------------------------------------------------------------------- operators
def make_scale_op(oid: str = "scale2", body_code: str | None = None, contract: Contract | None = None) -> OperatorSpec:
    src = body_code or "def run(inputs, config):\n    return {'z': inputs['v'] * 2.0}\n"
    body = WorkflowGraph({"k": Node("k", "code", code=src, inputs={"v": PortSchema("array")}, outputs={"z": PortSchema("array")})})
    return OperatorSpec(oid, 1, "scale by two", "multiplies the input by two", body,
                        {"x": PortSchema("array")}, {"y": PortSchema("array")}, {"x": [("k", "v")]}, {"y": ("k", "z")},
                        contract or Contract(pre=[{"port": "x", "check": "finite"}],
                                             post=[{"port": "y", "check": "range", "value": [0, 1]}]))


def test_operator_node_expands_body_and_reports_contract_violations(env) -> None:
    prog = AgentProgram(operators={"scale2": make_scale_op()})
    ex = Executor(env.ep, prog, None, env.run_dir)
    frac = "def run(inputs, config):\n    import numpy as np\n    x = np.linspace(0.1, 0.9, 5)\n    x[0] = config.get('first', 0.1)\n    return {'m': x}\n"
    g = WorkflowGraph()
    g.nodes["f"] = code("f", frac, {}, {"m": ARR})
    g.nodes["o"] = Node("o", "operator", ref="op:scale2", inputs=dict(prog.operators["scale2"].inputs),
                        outputs=dict(prog.operators["scale2"].outputs))
    g.nodes["s"] = submit(env.ep)
    g.edges = [Edge.make("f", "m", "o", "x"), Edge.make("o", "y", "s", "y")]
    cp, records, y = ex.execute(g, ex.new_checkpoint())
    rec = records["o"]
    assert rec.status == "ok" and rec.kind == "operator"
    np.testing.assert_allclose(y, np.linspace(0.1, 0.9, 5) * 2)
    assert any(v.startswith("post: y: max 1.8") for v in rec.contract_violations), rec.contract_violations
    assert set(rec.input_refs) == {"x"} and set(rec.output_refs) == {"y"}
    # a non-finite input violates the precondition; execution still proceeds (diagnostic, not crash)
    g.nodes["f"].config["first"] = float("nan")
    _, records, _ = ex.execute(g, cp)
    assert any(v == "pre: x: non-finite values" for v in records["o"].contract_violations)
    # feedback shows the contract diagnostics
    _, _, fb, _ = ex.apply(g, cp, Action("finish", {}), 3)
    assert "contract/schema: pre: x: non-finite values" in fb.render()


def test_operator_internal_failure_and_nesting(env) -> None:
    broken = make_scale_op("broken", body_code="def run(inputs, config):\n    raise KeyError('v2')\n", contract=Contract())
    inner = make_scale_op("inner", contract=Contract())
    outer_body = WorkflowGraph({"o": Node("o", "operator", ref="op:inner", inputs={"x": PortSchema("array")},
                                          outputs={"y": PortSchema("array")})})
    outer = OperatorSpec("outer", 1, "outer", "wraps inner", outer_body, {"a": PortSchema("array")}, {"b": PortSchema("array")},
                         {"a": [("o", "x")]}, {"b": ("o", "y")}, Contract())
    prog = AgentProgram(operators={"broken": broken, "inner": inner, "outer": outer})
    ex = Executor(env.ep, prog, None, env.run_dir)
    src = "import numpy as np\ndef run(inputs, config):\n    return {'m': np.array([1.0, 2.0])}\n"
    g = WorkflowGraph()
    g.nodes["f"] = code("f", src, {}, {"m": ARR})
    g.nodes["bad"] = Node("bad", "operator", ref="op:broken", inputs={"x": PortSchema("array")}, outputs={"y": PortSchema("array")})
    g.nodes["nest"] = Node("nest", "operator", ref="op:outer", inputs={"a": PortSchema("array")}, outputs={"b": PortSchema("array")})
    g.edges = [Edge.make("f", "m", "bad", "x"), Edge.make("f", "m", "nest", "a")]
    _, records, _ = ex.execute(g, ex.new_checkpoint())
    assert records["bad"].status == "error"
    assert "operator op:broken failed inside its body" in records["bad"].error and "KeyError" in records["bad"].error
    assert records["nest"].status == "ok"
    np.testing.assert_allclose(load_value(records["nest"].output_refs["b"]), [2.0, 4.0])
    # a self-recursive operator is stopped by the depth limit
    rec_body = WorkflowGraph({"o": Node("o", "operator", ref="op:loop", inputs={"x": PortSchema("array")},
                                        outputs={"y": PortSchema("array")})})
    loop = OperatorSpec("loop", 1, "loop", "recursive", rec_body, {"x": PortSchema("array")}, {"y": PortSchema("array")},
                        {"x": [("o", "x")]}, {"y": ("o", "y")}, Contract())
    ex2 = Executor(env.ep, AgentProgram(operators={"loop": loop}), None, env.run_dir / "r")
    g2 = WorkflowGraph({"f": g.nodes["f"], "l": Node("l", "operator", ref="op:loop", inputs={"x": PortSchema("array")},
                                                     outputs={"y": PortSchema("array")})}, [Edge.make("f", "m", "l", "x")])
    _, records, _ = ex2.execute(g2, None)
    assert records["l"].status == "error" and "nesting deeper" in records["l"].error


def test_checkpoint_serialization_roundtrip(env) -> None:
    ex = Executor(env.ep, env.prog, None, env.run_dir)
    g = graph([tool("t", env.ep), code("c", ROWMEAN, {"x": ARR}, {"m": ARR})], [("t", "data", "c", "x")])
    cp, _, _ = ex.execute(g, ex.new_checkpoint())
    cp2 = Checkpoint.from_dict(json.loads(json.dumps(cp.to_dict())))
    _, records, _ = ex.execute(g, cp2)
    assert all(r.cached for r in records.values())


def test_llm_node_with_project_fake_llm(env) -> None:
    """Keyword compatibility with the project's LLM client API (llm.fake.FakeLLM mirrors LLMClient)."""
    fake_mod = pytest.importorskip("scienceclaw.llm.fake")
    client_mod = pytest.importorskip("scienceclaw.llm.client")

    def respond(role, messages):
        text = messages[-1]["content"]
        if "bad" in text:
            return client_mod.LLMError("HTTP 500 upstream", status=500)
        return '```json\n{"label": "%s"}\n```' % text.split()[-1]

    llm = fake_mod.FakeLLM(respond)
    mk = code("mk", "def run(inputs, config):\n    return {'items': ['a x', 'b bad', 'c y']}\n", {}, {"items": {"type": "list"}})
    ln = Node("l", "llm", prompt="Label {item}", config={"parse": "json", "max_tokens": 64, "seed": 3},
              inputs={"items": PortSchema("list")}, outputs={"outputs": PortSchema("list")})
    g = graph([mk, ln], [("mk", "items", "l", "items")])
    ex = Executor(env.ep, env.prog, llm, env.run_dir)
    cp, records, _ = ex.execute(g, ex.new_checkpoint())
    rec = records["l"]
    assert rec.status == "ok", rec.error
    assert load_value(rec.output_refs["outputs"]) == [{"label": "x"}, None, {"label": "y"}]
    assert rec.outputs_summary["outputs"]["llm_errors"] == 1
    call = llm.calls[0]
    assert call["role"] == "executor" and call["json_mode"] is True and call["max_tokens"] == 64
    assert call["tag"] == "ep-x" and call["cache_salt"] == "seed=3"
    assert rec.llm_usage["calls"] == 2
    ex.execute(g, cp)                                   # a partially failed llm node is re-run (not cached)
    assert len(llm.calls) == 6


def test_relative_run_dir_is_resolved_for_sandbox_workers(env, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    ex = Executor(env.ep, env.prog, None, Path("rel_run"))
    assert ex.run_dir.is_absolute()
    g, cp = WorkflowGraph(), ex.new_checkpoint()
    actions = [
        Action("add_node", {"node": {"id": "t", "kind": "tool", "ref": "load"}}),
        Action("add_node", {"node": {"id": "c", "kind": "code", "code": ROWMEAN, "inputs": {"x": {"type": "array", "unit": "K"}},
                                     "outputs": {"m": {"type": "array", "shape": ["n"]}}}}),
        Action("add_edge", {"edge": {"src": "t", "dst": "c", "dst_port": "x"}}),
    ]
    for k, a in enumerate(actions):
        g, cp, fb, _ = ex.apply(g, cp, a, k)
        assert fb.action_ok, fb.action_error
    assert fb.records["c"].status == "ok", fb.records["c"].error
