"""Offline fakes for the agent tests: scripted LLM, toy episode, stub parser / executor / replay.

The stubs implement only what the solver needs so that the agent tests do not depend on the runtime
modules (written in parallel). ``tests/test_agent_integration.py`` exercises the real modules.
"""
from __future__ import annotations

import copy
import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from scienceclaw.bench.task import Budget, ConstraintSpec, EvalResult, Episode, ToolSpec
from scienceclaw.core.graph import Edge, Node, WorkflowGraph
from scienceclaw.core.schema import PortSchema
from scienceclaw.core.trace import NodeRecord, Trace

# Values that exist only inside the hidden evaluator; they must never reach the policy.
SECRET_METRIC = "secret_metric_zq"
SECRET_SCORE = 0.9876543
SECRET_LABELS = [1.25, 2.5, 3.75, 5.0]
ACCEPTANCE_TEXT = "ACCEPT_RULE_HIDDEN: primary beats the reference"

GOOD_CODE = "def run(inputs, config):\n    return {'y': [v * 1.25 for v in inputs['x']]}\n"
WRONG_CODE = "def run(inputs, config):\n    return {'y': [v * 0.5 for v in inputs['x']]}\n"
BUGGY_CODE = "def run(inputs, config):\n    raise ValueError('boom in code node')\n"


# ------------------------------------------------------------------------------------------ LLM
@dataclass
class FakeResponse:
    text: str
    usage: dict
    cached: bool = False
    latency_s: float = 0.0
    model: str = "scripted"
    finish_reason: str | None = "stop"


class ScriptedLLM:
    """``chat`` returns scripted replies; every call's messages are recorded (thread-safe).

    ``script`` is a list of replies (consumed in order; dict replies are JSON-encoded) or a callable
    ``(role, messages) -> str | dict``.
    """

    def __init__(self, script: list | Callable[[str, list[dict]], Any], tokens_per_call: int = 100,
                 cached: bool = False) -> None:
        self._script = list(script) if not callable(script) else script
        self.calls: list[dict] = []
        self.tokens_per_call = tokens_per_call
        self.cached = cached
        self._lock = threading.Lock()

    def chat(self, role: str, messages: list[dict], *, json_mode: bool | None = None, max_tokens: int | None = None,
             temperature: float | None = None, cache_salt: str = "", tag: str = "") -> FakeResponse:
        with self._lock:
            self.calls.append({"role": role, "messages": copy.deepcopy(messages), "json_mode": json_mode, "tag": tag})
            if callable(self._script):
                item = self._script(role, messages)
            else:
                if not self._script:
                    item = {"thought": "done", "action": {"type": "finish"}, "uses": []}
                else:
                    item = self._script.pop(0)
        text = item if isinstance(item, str) else json.dumps(item)
        n = self.tokens_per_call
        if self.cached:
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "cached_calls": 1,
                     "cached_prompt_tokens": n, "cached_completion_tokens": n // 2}
        else:
            usage = {"prompt_tokens": n, "completion_tokens": n // 2, "calls": 1}
        return FakeResponse(text, usage, cached=self.cached)

    def all_text(self) -> str:
        return "\n".join(m.get("content", "") for c in self.calls for m in c["messages"])


def act(action: dict, uses: list[str] | None = None, thought: str = "t") -> dict:
    return {"thought": thought, "action": action, "uses": list(uses or [])}


def add_node(**node: Any) -> dict:
    return act({"type": "add_node", "node": node})


def add_edge(src: str, sp: str, dst: str, dp: str) -> dict:
    return act({"type": "add_edge", "edge": {"src": src, "src_port": sp, "dst": dst, "dst_port": dp}})


def modify_code(nid: str, code: str, uses: list[str] | None = None) -> dict:
    return act({"type": "modify_node", "id": nid, "patch": {"code": code}}, uses)


FINISH = act({"type": "finish"})


def build_script(first_code: str = BUGGY_CODE, fix: bool = True) -> list[dict]:
    """tool -> code (first_code) -> submit, then (optionally) fix the code, then finish."""
    s = [
        add_node(id="load", kind="tool", ref="load_x"),
        add_node(id="f", kind="code", code=first_code, inputs={"x": {"type": "list"}}, outputs={"y": {"type": "list"}}),
        add_edge("load", "x", "f", "x"),
        add_node(id="sub", kind="submit"),
        add_edge("f", "y", "sub", "y"),
    ]
    if fix:
        s.append(modify_code("f", GOOD_CODE))
    s.append(FINISH)
    return s


# -------------------------------------------------------------------------------------- episode
def make_episode(eid: str = "ep1", objective: str = "Predict the scaled quantity y for each visible x value.",
                 max_steps: int = 12, max_policy_tokens: int = 200_000) -> Episode:
    xs = [1.0, 2.0, 3.0, 4.0]

    def load_x(inputs: dict, config: dict) -> dict:
        return {"x": list(xs)}

    def evaluate(y: Any, trace: Any) -> EvalResult:
        arr = np.asarray(y, dtype=float)
        truth = np.asarray(SECRET_LABELS)
        err = float(np.mean(np.abs(arr - truth))) if arr.shape == truth.shape else float("inf")
        ok = err < 1e-6
        return EvalResult(metrics={SECRET_METRIC: SECRET_SCORE if ok else 0.1234567}, primary=SECRET_SCORE if ok else 0.1234567,
                          accepted=ok, details={"reference": 0.5, "norm_score": 1.0,
                                                "pooled_payload": {"y_true": list(SECRET_LABELS), "y_pred": arr.tolist()},
                                                "labels": list(SECRET_LABELS)})

    def dev_eval(y: Any) -> dict:
        return {"score": float(len(y)), "direction": "max"}

    tool = ToolSpec(name="load_x", description="returns the visible x values", inputs={},
                    outputs={"x": PortSchema(type="list")}, fn=load_x)
    cons = [
        ConstraintSpec("finite", "all values finite", lambda y, tr: (bool(np.isfinite(np.asarray(y, dtype=float)).all()), "")),
        ConstraintSpec("hidden_rule_q", "INVISIBLE_CONSTRAINT_TEXT", lambda y, tr: (True, ""), visible=False),
    ]
    return Episode(id=eid, discipline="FoR49", family="Physical & Earth", split="src", task_type="regression",
                   objective=objective, required_output=PortSchema(type="list", description="one value per x"),
                   tools=[tool], constraints=cons, budget=Budget(max_steps=max_steps, max_policy_tokens=max_policy_tokens),
                   acceptance=ACCEPTANCE_TEXT, tags=["toy", "scaling"], metric=SECRET_METRIC,
                   _evaluate=evaluate, _dev_evaluate=dev_eval)


# ------------------------------------------------------------------------------ stub action model
@dataclass
class StubAction:
    type: str
    payload: dict = field(default_factory=dict)
    uses: list[str] = field(default_factory=list)
    thought: str = ""
    raw: str = ""

    def to_dict(self) -> dict:
        return {"type": self.type, "payload": copy.deepcopy(self.payload), "uses": list(self.uses),
                "thought": self.thought, "raw": self.raw}


def stub_parse(text: str) -> tuple[StubAction | None, str | None]:
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError) as ex:
        return None, f"invalid JSON ({ex.msg if hasattr(ex, 'msg') else ex})"
    if not isinstance(obj, dict) or not isinstance(obj.get("action"), dict) or "type" not in obj["action"]:
        return None, "missing action.type"
    a = obj["action"]
    return StubAction(a["type"], {k: v for k, v in a.items() if k != "type"}, list(obj.get("uses") or []),
                      str(obj.get("thought", "")), text), None


# ------------------------------------------------------------------------------ stub executor
@dataclass
class StubFeedback:
    step: int
    action_ok: bool
    action_error: str | None
    validation_errors: list[str]
    records: dict
    submit_ready: bool
    y_summary: dict | None
    visible_constraints: dict
    dev: dict | None
    integrity_violations: list = field(default_factory=list)
    llm_usage: dict = field(default_factory=dict)
    wall_s: float = 0.0

    def render(self, max_chars: int = 6000) -> str:
        lines = [f"action_ok={self.action_ok}" + (f" error={self.action_error}" if self.action_error else "")]
        for nid in sorted(self.records):
            r = self.records[nid]
            lines.append(f"{nid}: {r.status}" + (f" {r.error}" if r.error else ""))
        if self.submit_ready:
            lines.append(f"y: {self.y_summary}")
        lines.append(f"visible constraints: {self.visible_constraints}")
        if self.dev is not None:
            lines.append(f"dev: {self.dev}")
        return "\n".join(lines)[:max_chars]


def _apply(graph: WorkflowGraph, typ: str, p: dict, episode: Episode, program: Any) -> tuple[WorkflowGraph, str | None]:
    g = graph.copy()
    if typ == "batch":
        for sub in p.get("actions", []):
            g, err = _apply(g, sub["type"], {k: v for k, v in sub.items() if k != "type"}, episode, program)
            if err:
                return graph, err
        return g, None
    if typ == "add_node":
        nd = dict(p["node"])
        if nd["id"] in g.nodes:
            return graph, "duplicate id"
        if nd["kind"] == "tool":
            spec = episode.tool(nd["ref"])
            if spec is None:
                return graph, "unknown tool"
            g.nodes[nd["id"]] = Node(nd["id"], "tool", ref=spec.name, inputs=dict(spec.inputs), outputs=dict(spec.outputs))
        elif nd["kind"] == "submit":
            g.nodes[nd["id"]] = Node(nd["id"], "submit", inputs={"y": episode.required_output})
        elif nd["kind"] == "operator":
            oid = str(nd["ref"]).removeprefix("op:")
            op = program.operators.get(oid)
            if op is None:
                return graph, "unknown operator"
            g.nodes[nd["id"]] = Node(nd["id"], "operator", ref=f"op:{oid}", inputs=dict(op.inputs), outputs=dict(op.outputs))
        else:
            g.nodes[nd["id"]] = Node.from_dict(nd)
    elif typ == "modify_node":
        n = g.nodes.get(p["id"])
        if n is None:
            return graph, "unknown node"
        for k, v in p["patch"].items():
            setattr(n, k, v if k not in ("inputs", "outputs") else {a: PortSchema.from_dict(b) for a, b in v.items()})
    elif typ == "remove_node":
        g.nodes.pop(p["id"], None)
        g.edges = [e for e in g.edges if p["id"] not in (e.src, e.dst)]
    elif typ == "add_edge":
        e = p["edge"]
        g.edges.append(Edge.make(e["src"], e["src_port"], e["dst"], e["dst_port"], e.get("conversion")))
    elif typ == "remove_edge":
        e = p["edge"]
        g.edges = [x for x in g.edges if (x.src, x.src_port, x.dst, x.dst_port) != (e["src"], e["src_port"], e["dst"], e["dst_port"])]
    else:
        return graph, f"unknown action {typ}"
    errs = g.validate()
    if errs:
        return graph, "; ".join(errs)
    return g, None


def execute_graph(graph: WorkflowGraph, episode: Episode) -> tuple[dict[str, NodeRecord], Any]:
    values: dict[tuple[str, str], Any] = {}
    records: dict[str, NodeRecord] = {}
    y = None
    for nid in graph.topo_order():
        n = graph.nodes[nid]
        fp = graph.fingerprint(nid)
        ins: dict[str, Any] = {}
        missing = False
        for e in graph.in_edges(nid):
            if (e.src, e.src_port) not in values:
                missing = True
                break
            ins[e.dst_port] = values[(e.src, e.src_port)]
        if missing or graph.missing_inputs(nid):
            records[nid] = NodeRecord(nid, fp, "pending", kind=n.kind)
            continue
        try:
            if n.kind == "tool":
                out = episode.tool(n.ref).fn(ins, n.config)
            elif n.kind == "code":
                ns: dict[str, Any] = {}
                exec(n.code, ns)  # noqa: S102 - test stub only
                out = ns["run"](ins, n.config)
            elif n.kind == "submit":
                out = {}
                y = ins.get("y")
            else:
                raise RuntimeError(f"stub executor cannot run {n.kind} nodes")
            for port, v in out.items():
                values[(nid, port)] = v
            records[nid] = NodeRecord(nid, fp, "ok", kind=n.kind)
        except Exception as ex:
            records[nid] = NodeRecord(nid, fp, "error", error=f"{type(ex).__name__}: {ex}", kind=n.kind)
    return records, y


class StubExecutor:
    instances: list["StubExecutor"] = []

    def __init__(self, episode: Episode, program: Any, llm: Any, run_dir: str) -> None:
        self.episode, self.program, self.run_dir = episode, program, run_dir
        self.applied: list[Any] = []
        StubExecutor.instances.append(self)

    def new_checkpoint(self) -> dict:
        return {}

    def apply(self, graph: WorkflowGraph, checkpoint: Any, action: Any, step: int):
        self.applied.append(action)
        g, err = _apply(graph, action.type, action.payload, self.episode, self.program)
        records, y = execute_graph(g, self.episode)
        vis, msgs = self.episode.check_constraints(y, None, visible_only=True) if y is not None else ({}, {})
        fb = StubFeedback(step, err is None, err, [], records, y is not None,
                          {"len": len(y)} if y is not None else None,
                          {k: [v, msgs.get(k, "")] for k, v in vis.items()},
                          self.episode.dev_evaluate(y))
        return g, checkpoint, fb, y


def stub_replay(graph: WorkflowGraph, episode: Episode, program: Any, llm: Any, run_dir: str) -> tuple[Any, Trace]:
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    records, y = execute_graph(graph, episode)
    return y, Trace(records=records, order=graph.topo_order(), run_dir=str(run_dir))


def stub_match(a: Any, b: Any, tol: dict) -> bool:
    try:
        return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                                rtol=tol.get("rtol", 1e-6), atol=tol.get("atol", 1e-8)))
    except (TypeError, ValueError):
        return a == b


def make_solver(cfg: Any, llm: Any, evo_cfg: Any = None, **kw: Any):
    from scienceclaw.agent.solver import Solver

    kw.setdefault("executor_factory", StubExecutor)
    kw.setdefault("replay_fn", stub_replay)
    kw.setdefault("outputs_match_fn", stub_match)
    kw.setdefault("parser", stub_parse)
    return Solver(cfg, llm, evo_cfg, **kw)


STEP_RE = re.compile(r"## Step (\d+)")


def step_of(messages: list[dict]) -> int:
    """0-based step index from the latest step message (retries reuse the step)."""
    for m in reversed(messages):
        if m["role"] == "user":
            mm = STEP_RE.search(m["content"])
            if mm:
                return int(mm.group(1)) - 1
    return -1


def test_support_stub_executor_runs_a_graph() -> None:
    """Sanity check of the stub itself."""
    ep = make_episode()
    g = WorkflowGraph()
    for a in build_script(GOOD_CODE, fix=False)[:-1]:
        g, err = _apply(g, a["action"]["type"], {k: v for k, v in a["action"].items() if k != "type"}, ep, None)
        assert err is None
    _, y = execute_graph(g, ep)
    assert y == SECRET_LABELS
