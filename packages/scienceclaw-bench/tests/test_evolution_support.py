"""Shared builders and fakes for the evolution tests (this module defines no tests).

* a toy episode (tool ``load_data`` -> code nodes -> submit) with a hidden evaluator,
* an in-process graph executor that records a trace with pickled port values (like a reset replay),
* a fail -> pass source trajectory built with the real ``core.actions`` (Eq. 9 test material),
* a stub solver / plan / retriever for the validation gate and the evolver loop,
* a fake ``run_operator_isolated`` used when the runtime implementation is not importable.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import threading
import types
from pathlib import Path
from typing import Any

from scienceclaw.agent.solver import SolveResult, StepRecord
from scienceclaw.bench.task import ConstraintSpec, EvalResult, Episode, ToolSpec, passes
from scienceclaw.core.actions import Action, apply_action
from scienceclaw.core.graph import WorkflowGraph
from scienceclaw.core.program import AgentProgram
from scienceclaw.core.schema import PortSchema, apply_conversion
from scienceclaw.core.trace import Evidence, NodeRecord, Trace
from scienceclaw.llm.fake import FakeLLM
from scienceclaw.runtime.values import save_value

DATA = [1.0, 2.0, 3.0, 4.0]
TARGET = [3.0, 5.0, 7.0, 9.0]
HIDDEN_PRIMARY_FAIL = 0.8765
HIDDEN_MSG = "SECRET-HIDDEN-MSG"

PREP_CODE = ("def run(inputs, config):\n"
             "    s = float(config.get('scale', 1.0))\n"
             "    return {'y': [v * s for v in inputs['x']]}\n")
POST_CODE = ("def run(inputs, config):\n"
             "    return {'z': [v + 1.0 for v in inputs['y']]}\n")


# ------------------------------------------------------------------------------------------ episode
def make_episode(eid: str = "ep1", discipline: str = "FoR99", task_type: str = "toy_regression",
                 split: str = "src") -> Episode:
    tool = ToolSpec("load_data", "Load the visible input vector.", inputs={},
                    outputs={"table": PortSchema("list")}, fn=lambda i, c: {"table": list(DATA)})
    constraints = [
        ConstraintSpec("finite_output", "all outputs finite",
                       lambda y, tr: (all(math.isfinite(float(v)) for v in y), "finite check")),
        ConstraintSpec("secret_check", "hidden check", lambda y, tr: (len(y) == 4, HIDDEN_MSG), visible=False),
        ConstraintSpec("output_schema", "y is a list of numbers", lambda y, tr: (isinstance(y, list), "schema")),
    ]

    def evaluate(y: Any, trace: Any) -> EvalResult:
        ok = isinstance(y, list) and len(y) == 4 and all(abs(a - b) < 1e-9 for a, b in zip(y, TARGET))
        return EvalResult(metrics={"score": 1.0 if ok else 0.0}, primary=0.1234 if ok else HIDDEN_PRIMARY_FAIL,
                          accepted=ok, details={"norm_score": 1.0 if ok else 0.2, "reference": 0.5})

    return Episode(id=eid, discipline=discipline, family="Physical & Earth", split=split, task_type=task_type,
                   objective="Transform the loaded vector into the required output vector.",
                   required_output=PortSchema("list"), tools=[tool], constraints=constraints, tags=["toy"],
                   lineage={"item_ids": [f"{eid}-item-0001"]}, _evaluate=evaluate,
                   _dev_evaluate=lambda y: {"dev_score": 0.5})


# ---------------------------------------------------------------------------------- execution
def _exec_code(code: str, ins: dict, config: dict) -> dict:
    ns: dict[str, Any] = {}
    exec(compile(code, "<node>", "exec"), ns)  # noqa: S102 - test fixture code only
    return ns["run"](ins, dict(config))


def execute_graph(graph: WorkflowGraph, episode: Episode, run_dir: str | Path) -> tuple[Any, Trace]:
    """In-process 'reset replay': runs tool/code/submit nodes in topo order, pickling every port value."""
    rd = Path(run_dir)
    rd.mkdir(parents=True, exist_ok=True)
    vals: dict[tuple[str, str], Any] = {}
    trace = Trace(run_dir=str(rd))
    fps = graph.fingerprints()
    y = None
    for nid in graph.topo_order():
        node = graph.nodes[nid]
        ins: dict[str, Any] = {}
        missing = False
        for e in graph.in_edges(nid):
            if (e.src, e.src_port) not in vals:
                missing = True
                continue
            ins[e.dst_port] = apply_conversion(vals[(e.src, e.src_port)], e.conversion_dict)
        rec = NodeRecord(node_id=nid, fingerprint=fps[nid], status="ok", kind=node.kind)
        for p, v in ins.items():
            path = rd / f"{nid}.in.{p}.pkl"
            save_value(v, path)
            rec.input_refs[p] = str(path)
        if missing:
            rec.status = "pending"
            trace.records[nid] = rec
            trace.order.append(nid)
            continue
        outs: dict[str, Any] = {}
        try:
            if node.kind == "tool":
                outs = episode.tool(node.ref).fn(ins, node.config)
            elif node.kind == "code":
                outs = _exec_code(node.code or "", ins, node.config)
            elif node.kind == "submit":
                y = ins.get("y")
            elif node.kind == "operator":
                outs = {}        # a dangling operator node: recorded ok, contributes nothing to y
            else:
                raise NotImplementedError(node.kind)
        except Exception as ex:  # recorded like the runtime does
            rec.status, rec.error = "error", f"{type(ex).__name__}: {ex}"
            outs = {}
        for p, v in (outs or {}).items():
            vals[(nid, p)] = v
            path = rd / f"{nid}.out.{p}.pkl"
            save_value(v, path)
            rec.output_refs[p] = str(path)
        trace.records[nid] = rec
        trace.order.append(nid)
    return y, trace


def make_evidence(graph: WorkflowGraph, episode: Episode, step: int, run_dir: str | Path) -> Evidence:
    y, trace = execute_graph(graph, episode, Path(run_dir) / f"replay_k{step:03d}")
    ev = episode.evaluate(y, trace)
    ev.reproducible = True if y is not None else None
    ev.z = int(ev.completed and ev.hard_ok() and ev.accepted and ev.within_budget)
    return Evidence(step=step, graph_dict=graph.to_dict(), y=y, trace=trace, eval=ev, passed=passes(ev),
                    graph_fp=graph.graph_fingerprint())


# ------------------------------------------------------------------------------- trajectories
def repair_steps(uses_at_repair: list[str] | None = None, rejected_at: int | None = None,
                 parse_error_at: int | None = None, op_refs: list[str] | None = None) -> list[dict]:
    """Step specs of the canonical fail -> pass trajectory.

    Steps 0-4 build load -> prep -> submit (replay at step 4: FAIL, y = DATA). Steps 5-9 repair it: a config
    edit (scale = 2), a new code node ``post`` (+1) and three routing edits (replay at step 9: PASS).
    ``op_refs`` ("op:<id>") add one dangling operator node each just before the passing replay (so they are
    part of e_src's graph and ran ok in its trace; the replay step moves to 9 + len(op_refs)).
    """
    uses = list(uses_at_repair or [])
    specs = [
        {"action": {"type": "add_node", "node": {"id": "load", "kind": "tool", "ref": "load_data"}}},
        {"action": {"type": "add_node", "node": {"id": "prep", "kind": "code", "code": PREP_CODE,
                                                 "inputs": {"x": {"type": "list"}}, "outputs": {"y": {"type": "list"}}}}},
        {"action": {"type": "add_edge", "edge": {"src": "load", "src_port": "table", "dst": "prep", "dst_port": "x"}}},
        {"action": {"type": "add_node", "node": {"id": "sub", "kind": "submit"}}},
        {"action": {"type": "add_edge", "edge": {"src": "prep", "src_port": "y", "dst": "sub", "dst_port": "y"}},
         "replay": True},
        {"action": {"type": "modify_node", "id": "prep", "patch": {"config": {"scale": 2.0}}}, "uses": uses},
        {"action": {"type": "add_node", "node": {"id": "post", "kind": "code", "code": POST_CODE,
                                                 "inputs": {"y": {"type": "list"}}, "outputs": {"z": {"type": "list"}}}},
         "uses": uses},
        {"action": {"type": "remove_edge", "edge": {"src": "prep", "src_port": "y", "dst": "sub", "dst_port": "y"}}},
        {"action": {"type": "add_edge", "edge": {"src": "prep", "src_port": "y", "dst": "post", "dst_port": "y"}}},
        {"action": {"type": "add_edge", "edge": {"src": "post", "src_port": "z", "dst": "sub", "dst_port": "y"}},
         "replay": True},
        {"action": {"type": "finish"}},
    ]
    for i, ref in enumerate(op_refs or []):
        specs.insert(len(specs) - 2 + i, {"action": {"type": "add_node",
                                                    "node": {"id": f"useop{i}", "kind": "operator", "ref": ref}}})
    if rejected_at is not None:
        specs.insert(rejected_at, {"action": {"type": "add_edge", "edge": {"src": "nope", "src_port": "a",
                                                                          "dst": "sub", "dst_port": "y"}},
                                   "uses": uses})
    if parse_error_at is not None:
        specs.insert(parse_error_at, {"action": None, "parse_error": "no JSON object found"})
    return specs


def build_solve_result(episode: Episode, program: AgentProgram, run_dir: str | Path, specs: list[dict] | None = None,
                       *, mode: str = "source", extra_uses: list[str] | None = None) -> SolveResult:
    """Apply the step specs with the real ``core.actions.apply_action`` and replay where requested."""
    specs = repair_steps() if specs is None else specs
    rd = Path(run_dir)
    g = WorkflowGraph()
    steps: list[StepRecord] = []
    actions: list[Action] = []
    action_steps: list[int] = []
    evidence: list[Evidence] = []
    nu: list[str] = []
    for k, spec in enumerate(specs):
        uses = list(spec.get("uses", []))
        for u in uses:
            if u not in nu:
                nu.append(u)
        if spec.get("action") is None:
            steps.append(StepRecord(k, None, spec.get("parse_error", "parse error"), {"action_ok": False}, [], {},
                                    g.graph_fingerprint(), None, 0.0))
            continue
        a = Action.from_dict({"action": spec["action"], "uses": uses})
        actions.append(a)
        action_steps.append(k)
        if a.type == "finish":
            steps.append(StepRecord(k, a.to_dict(), None, {"action_ok": True, "finish": True}, uses, {},
                                    g.graph_fingerprint(), None, 0.0))
            break
        g2, err = apply_action(g, a, episode, program, step=k)
        fb: dict[str, Any] = {"action_ok": err is None, "action_error": err, "dev": {"dev_score": 0.1 * k}}
        ev_idx = None
        if err is None:
            g = g2
            if spec.get("replay"):
                evidence.append(make_evidence(g, episode, k, rd))
                ev_idx = len(evidence) - 1
        steps.append(StepRecord(k, a.to_dict(), None, fb, uses, {}, g.graph_fingerprint(), ev_idx, 0.0))
    passing = [e for e in evidence if e.passed]
    final = passing[0] if passing else (evidence[-1] if evidence else None)
    ev_final = final.eval if final is not None else episode.evaluate(None, None)
    uses_all = set(nu) | set(extra_uses or [])
    return SolveResult(episode_id=episode.id, mode=mode, program_version=program.version,
                       final_graph=WorkflowGraph.from_dict(final.graph_dict) if final else g,
                       y=final.y if final else None, trace=final.trace if final else None, eval=ev_final,
                       evidence=evidence, steps=steps, actions=actions, retrieved={}, uses=uses_all,
                       usage={"total_tokens": 100, "wall_s": 0.5, "llm_calls": 3}, run_dir=str(rd),
                       action_steps=action_steps)


# ------------------------------------------------------------------------------------ patch LLM
SKILL_REPLY = {"skills": [{"revises": None, "title": "Rescale then offset vector outputs",
                           "applicability": "Vector-to-vector transformation tasks with a scale parameter.",
                           "procedure": ["Set the scale in the node config before adding new code.",
                                         "Route the scaled output through an offset node before submit."],
                           "pitfalls": ["Submitting the unscaled vector fails the acceptance rule."],
                           "tags": ["vector", "rescale"]}]}
OP_REPLY = {"name": "scale_then_offset", "description": "Scales a numeric list and adds a constant offset.",
            "tags": ["vector"]}


def patch_responder(skill_reply: dict | None = None, op_reply: dict | str | None = None):
    """FakeLLM responder answering the Skill-patch and Operator-doc prompts by their system text."""
    def respond(role: str, messages: list[dict]) -> str:
        system = messages[0]["content"] if messages else ""
        if "reusable Skills" in system:
            reply = skill_reply if skill_reply is not None else SKILL_REPLY
            return reply if isinstance(reply, str) else json.dumps(reply)
        if "documentation for a reusable, typed workflow operator" in system:
            reply = op_reply if op_reply is not None else OP_REPLY
            return reply if isinstance(reply, str) else json.dumps(reply)
        return "{}"
    return respond


def make_llm(**kw: Any) -> FakeLLM:
    return FakeLLM(patch_responder(**kw))


# ------------------------------------------------------------------------ isolated operator runs
ISOLATED_CALLS: list[dict] = []


def fake_run_operator_isolated(op: Any, inputs: dict, program: Any, llm: Any, run_dir: str,
                               episode: Any = None) -> tuple[dict, NodeRecord]:
    """In-process isolated execution of an operator body (code nodes only)."""
    ISOLATED_CALLS.append({"op": op.id, "run_dir": run_dir, "inputs": sorted(inputs)})
    body = op.body
    feed: dict[tuple[str, str], Any] = {}
    for name, targets in op.input_map.items():
        for nid, port in targets:
            feed[(nid, port)] = inputs[name]
    vals: dict[tuple[str, str], Any] = {}
    for nid in body.topo_order():
        node = body.nodes[nid]
        ins = {port: v for (n, port), v in feed.items() if n == nid}
        for e in body.in_edges(nid):
            ins[e.dst_port] = apply_conversion(vals[(e.src, e.src_port)], e.conversion_dict)
        for p, v in _exec_code(node.code or "", ins, node.config).items():
            vals[(nid, p)] = v
    outs = {name: vals[(s, p)] for name, (s, p) in op.output_map.items() if (s, p) in vals}
    return outs, NodeRecord(node_id=op.id, fingerprint="", status="ok", kind="operator")


def install_fake_isolated(monkeypatch: Any) -> None:
    """Route ``runtime.replay.run_operator_isolated`` to the in-process fake for this test."""
    mod = types.ModuleType("scienceclaw.runtime.replay")
    mod.run_operator_isolated = fake_run_operator_isolated  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scienceclaw.runtime.replay", mod)


def real_isolated_available() -> bool:
    try:
        from scienceclaw.runtime.replay import run_operator_isolated  # noqa: F401
        return True
    except ImportError:
        try:
            from scienceclaw.runtime.executor import run_operator_isolated  # noqa: F401,F811
            return True
        except ImportError:
            return False


# --------------------------------------------------------------------- stub solver / retriever
class SimulatedCrash(BaseException):
    """Raised by the stub solver to simulate a process crash (not caught by the evolver)."""


class StubRetriever:
    """slice_hash(ep) depends only on the components tagged with the episode's discipline."""

    def __init__(self, program: AgentProgram) -> None:
        self.program = program

    def slice_hash(self, episode: Any, k_skills: int, k_ops: int) -> str:
        d = episode.discipline
        ids = sorted(s.version_id for s in self.program.skills.values() if d in s.tags)
        ids += sorted(o.version_id for o in self.program.operators.values() if d in o.tags)
        return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def val_result(episode: Episode, program: AgentProgram, run_dir: str, *, z: int, norm: float,
               h: dict[str, bool] | None = None, completed: bool = True, code_nodes: dict[str, str] | None = None,
               tokens: float = 10.0) -> SolveResult:
    ev = EvalResult(primary=0.5, h=dict(h or {"finite_output": True, "output_schema": True}), accepted=bool(z),
                    z=z, completed=completed, reproducible=True, details={"norm_score": norm})
    g = WorkflowGraph()
    for nid, code in (code_nodes or {}).items():
        from scienceclaw.core.graph import Node

        g.nodes[nid] = Node(nid, "code", code=code, outputs={"o": PortSchema()})
    return SolveResult(episode_id=episode.id, mode="val", program_version=program.version, final_graph=g,
                       y=[1.0] if completed else None, trace=None, eval=ev, evidence=[], steps=[], actions=[],
                       retrieved={}, uses=set(), usage={"total_tokens": tokens, "wall_s": 1.0}, run_dir=run_dir)


class StubSolver:
    """Source mode: the canonical fail -> pass trajectory (claiming use of every program component unless
    ``omit_use`` says otherwise). Val mode: z = 1 iff the program has a Skill tagged with the discipline."""

    def __init__(self, evo_cfg: Any = None, *, omit_use: set[str] | None = None,
                 crash_once: set[tuple[str, str]] | None = None, fail_source_replay: bool = False) -> None:
        self.evo_cfg = evo_cfg
        self.omit_use = set(omit_use or ())
        self.crash_once = set(crash_once or ())
        self.fail_source_replay = fail_source_replay
        self.calls: list[tuple[str, str, str]] = []
        self._lock = threading.Lock()

    def solve(self, episode: Episode, program: AgentProgram, mode: str, run_dir: str | Path) -> SolveResult:
        with self._lock:
            self.calls.append((mode, episode.id, program.version))
            key = (mode, episode.id)
            if key in self.crash_once:
                self.crash_once.discard(key)
                raise SimulatedCrash(f"simulated crash in {key}")
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        if mode == "val":
            ok = any(episode.discipline in s.tags for s in program.skills.values())
            return val_result(episode, program, str(run_dir), z=int(ok), norm=1.0 if ok else 0.2)
        refs = [s.ref for s in program.skills.values()] + [o.ref for o in program.operators.values()]
        refs = [r for r in refs if not any(r.startswith(p) for p in self.omit_use)]
        op_refs = [f"op:{o.id}" for o in program.operators.values()
                   if not any(o.ref.startswith(p) for p in self.omit_use)]
        is_replay = bool(program.skills or program.operators) and program.version != "A0"
        if is_replay and self.fail_source_replay:
            specs = repair_steps()[:5]   # never repaired: only the failing evidence
            return build_solve_result(episode, program, run_dir, specs, extra_uses=refs)
        return build_solve_result(episode, program, run_dir, repair_steps(uses_at_repair=refs, op_refs=op_refs))


class StubPlan:
    def __init__(self, stream: list[tuple[int, Episode]], val: dict[str, list[Episode]]) -> None:
        self._stream = list(stream)
        self.episodes = {"val": val}

    def source_stream(self) -> list[tuple[int, Episode]]:
        return list(self._stream)
