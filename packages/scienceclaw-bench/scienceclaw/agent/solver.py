"""Solver: Z_t = (G*_t, y_t, τ_t) = Solve_Θ0(D_t | A_r)   (paper Eq. 1, realized through Eq. 6–8).

Per episode the solver

1. retrieves Skills and Operators of the program (``core.retrieval.Retriever``) and builds the system prompt
   (``agent.prompts``; interface description only, DESIGN decision 9);
2. runs the multi-turn loop k = 0 .. K-1: the policy proposes one atomic canvas edit a_{t,k} with its
   attribution ν_{t,k} (Eq. 6), the executor applies and executes it (Eq. 7), and the visible feedback
   f_{t,k} is shown in the next turn;
3. in ``mode="source"`` regenerates evidence e_{t,k} by *reset replay* (Eq. 8) whenever a new complete
   workflow is submitted (see "Replay trigger" below), evaluates it with the hidden evaluator and records
   ``Pass`` (``bench.task.passes``). The policy is never told anything from the hidden evaluation;
4. selects the final solution G*_t (DESIGN decision 7): the replay-verified passing graph with the best
   *visible* dev score (ties: latest step); otherwise the latest graph that produced an output (replayed
   now if it was not yet); otherwise failure (y = None, z = 0). In modes ``"val"``/``"eval"`` only this
   final graph is replayed and hidden-evaluated.

Replay trigger (source mode). A reset replay runs whenever the canvas holds a *submitted* workflow — a
valid graph whose submit node has its input ``y`` wired — whose submit fingerprint (submit node plus all
its ancestors) differs from the last replayed one. This includes submitted workflows whose execution
failed, so that crash -> repair sequences yield a replay-verified failure e⁻ before the success e⁺
(paper Eq. 9: "the latest replay-verified failure"). ``Solver(..., replay_failed_submits=False)`` restricts
replays to graphs that produced an output (the literal wording of DESIGN decision 6).

Pass / z. After the hidden evaluation the solver sets ``EvalResult.reproducible`` (clean replay output ==
executor output within ``episode.tolerance``) and ``within_budget`` (policy tokens and agent wall time at
the step that produced the graph), and recomputes ``z = completed ∧ all(h) ∧ accepted ∧ within_budget ∧
reproducible≠False`` so that z and Pass agree (DESIGN decision 4). The final output y_t is the output of
the clean replay.

Budgets. Steps: ``cfg.max_steps`` (fallback ``episode.budget.max_steps``) (``single_turn``: 1). Policy tokens are
counted *logically* (spent + cached) so a cached rerun stops at the same step. Wall time counts the agent's
own time (policy calls + executor), not the evaluation replays.

Thread safety: ``Solver.solve`` keeps all per-episode state in a private run object; many episodes can be
solved concurrently with separate run directories (the LLM client is shared and thread-safe).
"""
from __future__ import annotations

import copy
import dataclasses
import functools
import inspect
import json
import logging
import math
import pickle
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ..bench.task import EvalResult, passes
from ..core.graph import Edge, Node, WorkflowGraph
from ..core.retrieval import Retriever
from ..core.trace import Evidence, Trace
from . import prompts
from .policy import Policy, add_usage, empty_usage, logical_tokens

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import EvolutionConfig, SolverConfig
    from ..core.program import AgentProgram

__all__ = ["MODES", "StepRecord", "SolveResult", "Solver", "build_fixed_workflow", "normalize_uses",
           "check_orchestration", "scrub_volatile", "sanitize_eval", "FIXED_CODE_NODE", "FIXED_SUBMIT_NODE"]

log = logging.getLogger(__name__)

MODES = ("source", "val", "eval")
FIXED_CODE_NODE = "code"
FIXED_SUBMIT_NODE = "submit"
_FIXED_STUB = (
    "def run(inputs: dict, config: dict) -> dict:\n"
    "    raise NotImplementedError(\"node 'code' has no implementation yet\")\n"
)
_VERSION_SUFFIX = re.compile(r"@v\d+$")
# Volatile, run-specific fragments of rendered feedback (see scrub_volatile).
_WALL_RE = re.compile(r" \((?:\d+(?:\.\d+)?|\.\d+)s(, cached)?\)")
# "<run>/work/<node>-<uid>/": the 8-hex uid ends at any non-name character or at the end of the text; a uid cut short
# by a truncation upstream (1-7 hex digits at the very end of the text) is removed as well.
_WORK_UID_RE = re.compile(r"(work/[A-Za-z0-9_.-]+?)-(?:[0-9a-f]{8}(?![0-9A-Za-z_-])|[0-9a-f]{1,7}\Z)")
_LABEL_KEYS = re.compile(r"(^|_)(y_true|true|truth|labels?|gold|targets?|answers?|hidden|ground)($|_)", re.IGNORECASE)


# ============================================================================================ records
@dataclass
class StepRecord:
    """One interaction step k (DESIGN 8.4; the fields after ``wall_s`` are additive)."""
    step: int
    action: dict | None
    parse_error: str | None
    feedback: dict                      # visible feedback f_{t,k} (+ "text": what the policy was shown)
    uses: list[str]                     # ν_{t,k}, restricted to ids of the program
    policy_usage: dict
    graph_fp: str                       # graph fingerprint after the step
    evidence_idx: int | None            # index into SolveResult.evidence of the replay triggered here
    wall_s: float
    raw: str = ""                       # raw policy text of the last attempt
    thought: str = ""
    dropped_uses: list[str] = field(default_factory=list)
    submit_fp: str | None = None        # fingerprint of the submit node (+ ancestors) if submitted
    y_available: bool = False           # executor produced a submit output at this step

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class SolveResult:
    """Z_t = (G*_t, y_t, τ_t) plus the full step history and the evidence stream (DESIGN 8.4)."""
    episode_id: str
    mode: str
    program_version: str
    final_graph: WorkflowGraph
    y: Any
    trace: Trace | None
    eval: EvalResult
    evidence: list[Evidence]
    steps: list[StepRecord]
    actions: list[Any]                  # parsed actions (core.actions.Action), in step order
    retrieved: dict
    uses: set[str]
    usage: dict
    run_dir: str
    # ---- additive fields
    action_steps: list[int] = field(default_factory=list)   # step index of each entry of ``actions``
    uses_versions: set[str] = field(default_factory=set)    # ω of the used components under the solved program
    final_step: int | None = None       # step that produced G*
    final_source: str = "none"          # "pass" | "last_output" | "none"
    passed: bool = False                # Pass of the final solution
    stop_reason: str = ""
    notes: list[str] = field(default_factory=list)
    infra_error: str | None = None      # gateway / LLM outage that cut the solve short (NOT a task failure)

    @property
    def z(self) -> int:
        return int(getattr(self.eval, "z", 0) or 0)

    def action_at(self, step: int) -> Any | None:
        for s, a in zip(self.action_steps, self.actions):
            if s == step:
                return a
        return None

    def summary(self) -> dict:
        ev = self.eval
        return {
            "episode_id": self.episode_id, "mode": self.mode, "program_version": self.program_version,
            "z": self.z, "passed": self.passed, "primary": getattr(ev, "primary", None),
            "completed": getattr(ev, "completed", None), "reproducible": getattr(ev, "reproducible", None),
            "within_budget": getattr(ev, "within_budget", None), "hard_ok": ev.hard_ok() if ev else None,
            "final_step": self.final_step, "final_source": self.final_source, "stop_reason": self.stop_reason,
            "n_steps": len(self.steps), "n_evidence": len(self.evidence),
            "n_pass_evidence": sum(1 for e in self.evidence if e.passed),
            "retrieved": self.retrieved, "uses": sorted(self.uses), "uses_versions": sorted(self.uses_versions),
            "notes": list(self.notes), "run_dir": self.run_dir, "infra_error": self.infra_error,
        }

    def save(self, path: str | Path) -> None:
        """Write the run receipts (no hidden labels): trajectory.jsonl, final_graph.json, eval.json,
        evidence.jsonl, usage.json, result.json and y.pkl."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        _write_text(p / "trajectory.jsonl", "".join(_dumps(s.to_dict()) + "\n" for s in self.steps))
        _write_text(p / "final_graph.json", json.dumps(self.final_graph.to_dict(), indent=1, default=_json_default))
        _write_text(p / "eval.json", json.dumps(sanitize_eval(self.eval), indent=1, default=_json_default))
        ev_lines = []
        for i, e in enumerate(self.evidence):
            d = dict(e.summary())
            d.update({"idx": i, "reproducible": getattr(e.eval, "reproducible", None),
                      "within_budget": getattr(e.eval, "within_budget", None),
                      "accepted": getattr(e.eval, "accepted", None),
                      "completed": getattr(e.eval, "completed", None),
                      "details": _sanitize(dict(getattr(e.eval, "details", {}) or {}), drop_pooled=True),
                      "graph": e.graph_dict})
            ev_lines.append(_dumps(d) + "\n")
        _write_text(p / "evidence.jsonl", "".join(ev_lines))
        _write_text(p / "usage.json", json.dumps(self.usage, indent=1, default=_json_default))
        _write_text(p / "result.json", json.dumps(self.summary(), indent=1, default=_json_default))
        _save_value(self.y, p / "y.pkl")


# ============================================================================================ helpers
def _json_default(o: Any) -> Any:
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=str)
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    if isinstance(o, Path):
        return str(o)
    tolist = getattr(o, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception as ex:  # fall through to str() but keep a trace of why
            log.debug("tolist() failed for %s: %s", type(o).__name__, ex)
    item = getattr(o, "item", None)
    if callable(item):
        try:
            return item()
        except Exception as ex:
            log.debug("item() failed for %s: %s", type(o).__name__, ex)
    return str(o)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, default=_json_default, ensure_ascii=False)


def _write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _save_value(value: Any, path: Path) -> None:
    try:
        from ..runtime.values import save_value  # lazy: runtime is owned by another module
    except ImportError:
        save_value = None
    try:
        if save_value is not None:
            save_value(value, path)
        else:
            with open(path, "wb") as fh:
                pickle.dump(value, fh)
    except Exception as ex:  # an unpicklable y must not lose the other receipts
        log.warning("could not save y to %s: %s: %s", path, type(ex).__name__, ex)
        _write_text(path.with_suffix(".error.txt"), f"{type(ex).__name__}: {ex}\n")


def _sanitize(obj: Any, drop_pooled: bool = True) -> Any:
    """Recursively drop keys that may carry hidden labels (y_true, labels, targets, gold, answers, ...)."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            ks = str(k)
            if drop_pooled and ks == "pooled_payload":
                continue
            if _LABEL_KEYS.search(ks):
                continue
            out[k] = _sanitize(v, drop_pooled)
        return out
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v, drop_pooled) for v in obj]
    return obj


def sanitize_eval(ev: Any) -> dict:
    """EvalResult as a JSON-safe dict without label-carrying detail keys (``pooled_payload`` is dropped)."""
    d = ev.to_dict() if hasattr(ev, "to_dict") else dict(getattr(ev, "__dict__", {}))
    d["details"] = _sanitize(dict(d.get("details") or {}), drop_pooled=True)
    return d


def _fb_dict(fb: Any) -> dict:
    if fb is None:
        return {}
    if isinstance(fb, dict):
        return dict(fb)
    if hasattr(fb, "to_dict"):
        return dict(fb.to_dict())
    if dataclasses.is_dataclass(fb):
        return dataclasses.asdict(fb)
    return dict(getattr(fb, "__dict__", {}))


def scrub_volatile(text: str, run_dir: str | Path | None = None) -> str:
    """Remove run-specific, non-deterministic fragments from text shown to the policy.

    Per-node wall times ("(0.09s, cached)" -> " (cached)"), absolute run-directory paths (-> "<run>") and
    random work-dir suffixes ("work/f-1a2b3c4d" -> "work/f") would otherwise make identical contexts differ
    between runs, defeating the LLM response cache that reproducible re-solves and lazy re-validation rely
    on (DESIGN decision 8). They carry no task information.
    """
    if run_dir is not None:
        for root in {str(Path(run_dir)), str(Path(run_dir).resolve())}:
            if root and root != ".":
                text = text.replace(root, "<run>")
    text = _WORK_UID_RE.sub(r"\1", text)
    return _WALL_RE.sub(lambda m: " (cached)" if m.group(1) else "", text)


def _visible_feedback(fb: Any, show_dev: bool) -> Any:
    """``fb`` as the policy may see it: without the dev score unless ``show_dev`` (every rendering of it)."""
    if show_dev:
        return fb
    if isinstance(fb, dict):
        return {k: v for k, v in fb.items() if k != "dev"}
    try:
        return dataclasses.replace(fb, dev=None) if dataclasses.is_dataclass(fb) else fb
    except TypeError as ex:
        log.debug("cannot hide dev score on %s: %s", type(fb).__name__, ex)
        return fb


def _render_feedback(fb: Any, show_dev: bool, scrub: Callable[[str], str] | None = None) -> str:
    """Feedback text of the last action. ``scrub`` runs before any clipping / truncation inside the rendering."""
    fb = _visible_feedback(fb, show_dev)
    render = getattr(fb, "render", None)
    if callable(render):
        try:
            takes_scrub = "scrub" in inspect.signature(render).parameters
        except (TypeError, ValueError):
            takes_scrub = False
        text = str(render(scrub=scrub)) if takes_scrub and scrub is not None else str(render())
        return scrub(text) if scrub is not None else text
    return prompts.summarize_feedback(fb, max_chars=6000, scrub=scrub)


def _sum_llm_usage(acc: dict, u: dict | None) -> None:
    for k, v in (u or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            acc[k] = acc.get(k, 0) + v


# Messages of runtime.executor for an LLM outage inside an llm node: "LLM call failed: ..." (whole batch raised) and
# "all N LLM calls failed: ..." (every item of the batch failed). Partial failures are not outages.
_LLM_OUTAGE_RE = re.compile(r"\b(all \d+ LLM calls failed|LLM call failed)\b")


def llm_outage_error(records: dict | None) -> str | None:
    """First executor node error that is an LLM/gateway outage (transient, not a property of the workflow)."""
    for nid, r in (records or {}).items():
        st = getattr(r, "status", None) if not isinstance(r, dict) else r.get("status")
        err = getattr(r, "error", None) if not isinstance(r, dict) else r.get("error")
        if st == "error" and err and _LLM_OUTAGE_RE.search(str(err)):
            return f"node {nid}: {str(err)[:300]}"
    return None


def _count_node_runs(records: dict | None) -> int:
    n = 0
    for r in (records or {}).values():
        st = getattr(r, "status", None) if not isinstance(r, dict) else r.get("status")
        cached = getattr(r, "cached", False) if not isinstance(r, dict) else r.get("cached", False)
        if st in ("ok", "error") and not cached:
            n += 1
    return n


def _dev_score(dev: Any, default_direction: str) -> float | None:
    """Signed (higher = better) visible dev score from a dev dict, or None."""
    if not isinstance(dev, dict):
        return None
    for key in ("score", "dev_score", "primary", "value", "metric"):
        v = dev.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
            direction = str(dev.get("direction", default_direction) or default_direction)
            return float(v) if direction != "min" else -float(v)
    return None


def normalize_uses(uses: Any, program: "AgentProgram") -> tuple[list[str], list[str]]:
    """ν restricted to the program: returns (kept refs "skill:<id>"/"op:<id>", dropped raw entries).

    Accepts "skill:<id>", "op:<id>", "operator:<id>", optional "@vN" suffixes and bare ids that match exactly
    one skill or operator id.
    """
    kept: list[str] = []
    dropped: list[str] = []
    for u in list(uses or []):
        s = _VERSION_SUFFIX.sub("", str(u).strip())
        ref: str | None = None
        if s.startswith("skill:"):
            ref = f"skill:{s[6:]}" if s[6:] in program.skills else None
        elif s.startswith("op:") or s.startswith("operator:"):
            oid = s.split(":", 1)[1]
            ref = f"op:{oid}" if oid in program.operators else None
        elif s in program.skills and s not in program.operators:
            ref = f"skill:{s}"
        elif s in program.operators and s not in program.skills:
            ref = f"op:{s}"
        if ref is None:
            dropped.append(str(u))
        elif ref not in kept:
            kept.append(ref)
    return kept, dropped


def _strip_prefix(ref: Any, prefixes: tuple[str, ...]) -> Any:
    if not isinstance(ref, str):
        return ref
    for p in prefixes:
        if ref.startswith(p):
            ref = ref[len(p):]
            break
    return _VERSION_SUFFIX.sub("", ref)


def _atomic_items(action: Any) -> list[tuple[str, dict]]:
    """(type, payload) of an action, expanding a batch one level (dict or Action sub-items)."""
    typ = getattr(action, "type", None)
    payload = getattr(action, "payload", None) or {}
    if typ != "batch":
        return [(str(typ), payload)]
    out: list[tuple[str, dict]] = []
    for sub in payload.get("actions", None) or []:
        if isinstance(sub, dict):
            inner = sub.get("action") if isinstance(sub.get("action"), dict) else sub
            sp = inner.get("payload") if isinstance(inner.get("payload"), dict) else inner
            out.append((str(inner.get("type")), sp))
        else:
            out.append((str(getattr(sub, "type", None)), getattr(sub, "payload", None) or {}))
    return out


def check_orchestration(action: Any, graph: WorkflowGraph, orchestration: str,
                        fixed_code_node: str | None = None) -> str | None:
    """Return a rejection reason if ``action`` is not allowed under the orchestration, else None."""
    typ = getattr(action, "type", None)
    if typ == "batch" and orchestration != "single_turn":
        return "a batch action is accepted only in single-turn orchestration; send one atomic action per reply"
    if orchestration == "single_operator":
        n_code = sum(1 for n in graph.nodes.values() if n.kind == "code")
        for t, p in _atomic_items(action):
            if t != "add_node":
                continue
            kind = (p.get("node") or {}).get("kind")
            if kind in ("operator", "llm"):
                return f"single-operator orchestration: {kind} nodes cannot be added (allowed: tool nodes, one code node, submit)"
            if kind == "code":
                if n_code >= 1:
                    return "single-operator orchestration: the workflow already has a code node (at most one)"
                n_code += 1
    elif orchestration == "fixed_workflow":
        code_id = fixed_code_node or FIXED_CODE_NODE
        if typ == "finish":
            return None
        payload = getattr(action, "payload", None) or {}
        patch = payload.get("patch") or {}
        if typ != "modify_node" or payload.get("id") != code_id:
            return f"fixed-workflow orchestration: only modify_node on node {code_id!r} and finish are accepted"
        if not isinstance(patch, dict) or not patch or not set(patch) <= {"code", "code_edit", "config"}:
            return (f"fixed-workflow orchestration: the patch may change only 'code', 'code_edit' and/or 'config' "
                    f"of node {code_id!r}")
    return None


def build_fixed_workflow(episode: Any) -> tuple[WorkflowGraph, str]:
    """Pre-built canvas for ``fixed_workflow``: every input-free tool -> one code node -> submit.

    Tools that require inputs cannot be wired by a fixed template and are left out (they stay unused); the
    prompt lists exactly the wired tools (``prompts.fixed_workflow_tools``).
    Code-node input ports are named after the tool output ports ("<tool>__<port>" on collisions); its
    single output "y" has the required output schema.
    """
    origin = {"step": -1, "uses": [], "generated": False, "from_operator": None, "template": "fixed_workflow"}
    nodes: dict[str, Node] = {}
    ports: list[tuple[str, str, str, Any]] = []
    for t in prompts.fixed_workflow_tools(episode):
        nid = "tool_" + re.sub(r"[^A-Za-z0-9_]+", "_", t.name)
        nodes[nid] = Node(id=nid, kind="tool", ref=t.name, inputs={}, outputs=dict(t.outputs), origin=dict(origin))
        ports.extend((nid, t.name, p, sch) for p, sch in t.outputs.items())
    names = Counter(p for _, _, p, _ in ports)
    code_inputs = {}
    edges: list[Edge] = []
    for nid, tname, p, sch in ports:
        port = p if names[p] == 1 else f"{re.sub(r'[^A-Za-z0-9_]+', '_', tname)}__{p}"
        code_inputs[port] = sch
        edges.append(Edge.make(nid, p, FIXED_CODE_NODE, port))
    nodes[FIXED_CODE_NODE] = Node(id=FIXED_CODE_NODE, kind="code", code=_FIXED_STUB, config={}, inputs=code_inputs,
                                  outputs={"y": episode.required_output}, origin=dict(origin, generated=True))
    nodes[FIXED_SUBMIT_NODE] = Node(id=FIXED_SUBMIT_NODE, kind="submit", inputs={"y": episode.required_output},
                                    origin=dict(origin))
    edges.append(Edge.make(FIXED_CODE_NODE, "y", FIXED_SUBMIT_NODE, "y"))
    return WorkflowGraph(nodes, edges), FIXED_CODE_NODE


def _submitted_fp(graph: WorkflowGraph) -> str | None:
    """Fingerprint of the submit node (+ ancestors) if the graph is a valid, submitted workflow.

    Submitted = the graph validates, the submit input ``y`` is wired and no node feeding the deliverable
    has an unwired (non-optional) input port.
    """
    sid = graph.submit_node()
    if sid is None or not any(e.dst_port == "y" for e in graph.in_edges(sid)):
        return None
    if graph.validate():
        return None
    if any(graph.missing_inputs(n) for n in graph.ancestors([sid]) | {sid}):
        return None  # still under construction: some input port feeding the deliverable is unwired
    return graph.fingerprint(sid)


# ============================================================================================ solver
class Solver:
    """Solve_Θ0(D_t | A_r) — see the module docstring.

    Args:
        cfg: :class:`~scienceclaw.config.SolverConfig`.
        llm: shared ``LLMClient`` / ``FakeLLM`` (policy and executor roles).
        evo_cfg: :class:`~scienceclaw.config.EvolutionConfig` (``pass_requires_acceptance``); default: acceptance required.
        executor_factory, replay_fn, outputs_match_fn, parser: dependency injection for tests; default to
            ``runtime.executor.Executor``, ``runtime.replay.replay``, ``runtime.values.outputs_match`` and
            ``core.actions.parse_action`` (imported lazily).
        replay_failed_submits: also replay submitted workflows whose execution produced no output (see
            "Replay trigger" in the module docstring).
        max_parallel_subprocs: passed to the executor.
    """

    def __init__(self, cfg: "SolverConfig", llm: Any, evo_cfg: "EvolutionConfig | None" = None, *,
                 executor_factory: Callable[..., Any] | None = None, replay_fn: Callable[..., Any] | None = None,
                 outputs_match_fn: Callable[[Any, Any, dict], bool] | None = None,
                 parser: Callable[[str], Any] | None = None, replay_failed_submits: bool = True,
                 max_parallel_subprocs: int = 4) -> None:
        if cfg.orchestration not in prompts.ORCHESTRATIONS:
            raise ValueError(f"unknown orchestration {cfg.orchestration!r}; expected one of {prompts.ORCHESTRATIONS}")
        self.cfg = cfg
        self.llm = llm
        self.evo_cfg = evo_cfg
        self.policy = Policy(llm, cfg, parser)
        self._executor_factory = executor_factory
        self._replay_fn = replay_fn
        self._outputs_match_fn = outputs_match_fn
        self.replay_failed_submits = bool(replay_failed_submits)
        self.max_parallel_subprocs = int(max_parallel_subprocs)

    # -------------------------------------------------------------- dependencies (lazy imports)
    def make_executor(self, episode: Any, program: Any, run_dir: Path) -> Any:
        if self._executor_factory is not None:
            return self._executor_factory(episode, program, self.llm, str(run_dir))
        from ..runtime.executor import Executor

        return Executor(episode, program, self.llm, str(run_dir), max_parallel_subprocs=self.max_parallel_subprocs)

    def replay(self, graph: WorkflowGraph, episode: Any, program: Any, run_dir: Path) -> tuple[Any, Trace]:
        fn = self._replay_fn
        if fn is None:
            from ..runtime.replay import replay as fn
        return fn(graph, episode, program, self.llm, str(run_dir))

    def outputs_match(self, a: Any, b: Any, tol: dict) -> bool:
        fn = self._outputs_match_fn
        if fn is None:
            from ..runtime.values import outputs_match as fn
        return bool(fn(a, b, tol))

    @property
    def require_acceptance(self) -> bool:
        return bool(getattr(self.evo_cfg, "pass_requires_acceptance", True)) if self.evo_cfg is not None else True

    # ------------------------------------------------------------------------------ public API
    def solve(self, episode: Any, program: "AgentProgram", mode: str, run_dir: str | Path) -> SolveResult:
        """Solve one episode from a reset state and write receipts under ``run_dir``."""
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        return _SolveRun(self, episode, program, mode, Path(run_dir)).run()


class _SolveRun:
    """All mutable state of one solve (never shared between threads)."""

    def __init__(self, solver: Solver, episode: Any, program: "AgentProgram", mode: str, run_dir: Path) -> None:
        self.s = solver
        self.cfg = solver.cfg
        self.ep = episode
        self.program = program
        self.mode = mode
        self.run_dir = run_dir
        self.orch = self.cfg.orchestration
        # The experiment protocol (SolverConfig.max_steps) sets the step budget; adapters' Budget.max_steps
        # is only used when the solver config gives none.
        steps = int(self.cfg.max_steps) if int(self.cfg.max_steps or 0) > 0 else int(episode.budget.max_steps)
        self.max_steps = 1 if self.orch == "single_turn" else max(0, steps)
        # accounting
        self.policy_usage = empty_usage()
        self.last_reply_tokens = 0
        self.exec_usage: dict = {}
        self.policy_wall = 0.0
        self.agent_wall = 0.0
        self.replay_wall = 0.0
        self.node_runs = 0
        self.replays = 0
        self.parse_failures = 0
        self.rejected = 0
        self.budget_at_step: dict[int, tuple[int, float]] = {}
        # trajectory
        self.steps: list[StepRecord] = []
        self.actions: list[Any] = []
        self.action_steps: list[int] = []
        self.history: list[dict] = []
        self.evidence: list[Evidence] = []
        self.evidence_submit_fp: list[str | None] = []
        self.evidence_y_exec: list[Any] = []
        self.nu: list[str] = []
        self.infra_last: str | None = None     # LLM outage seen in the most recent applied step
        self.notes: list[str] = []
        self.last_replayed_fp: str | None = None
        self.last_output: tuple[int, WorkflowGraph, Any, str] | None = None   # (step, graph, y_exec, submit_fp)
        self.dev_by_step: dict[int, float | None] = {}

    # ------------------------------------------------------------------------------ main loop
    def run(self) -> SolveResult:
        t_start = time.monotonic()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        ep, cfg = self.ep, self.cfg

        retriever = Retriever(self.program)
        skills = retriever.skills(ep, cfg.retrieve_skills_k)
        ops = retriever.operators(ep, cfg.retrieve_ops_k)
        retrieved = {
            "skills": [s.ref for s in skills], "operators": [o.ref for o in ops],
            "skill_versions": [s.version_id for s in skills], "operator_versions": [o.version_id for o in ops],
            "slice_hash": retriever.slice_hash(ep, cfg.retrieve_skills_k, cfg.retrieve_ops_k),
        }

        fixed_code: str | None = None
        graph = WorkflowGraph()
        if self.orch == "fixed_workflow":
            graph, fixed_code = build_fixed_workflow(ep)
            errs = graph.validate()
            if errs:
                self.notes.append(f"fixed workflow template has validation errors: {errs}")
                log.warning("episode %s: fixed workflow template invalid: %s", ep.id, errs)
        system = prompts.build_system_prompt(ep, skills, ops, self.orch, max_steps=self.max_steps,
                                             fixed_code_node=fixed_code, show_dev_score=bool(cfg.show_dev_score))
        scrub = functools.partial(scrub_volatile, run_dir=self.run_dir)
        _write_text(self.run_dir / "system_prompt.txt", system)

        exec_dir = self.run_dir / "exec"
        exec_dir.mkdir(parents=True, exist_ok=True)
        executor = self.s.make_executor(ep, self.program, exec_dir)
        checkpoint = self._new_checkpoint(executor, exec_dir)

        feedback_text: str | None = None
        stop_reason = "step_budget"
        io_log = open(self.run_dir / "policy_io.jsonl", "w")
        traj_log = open(self.run_dir / "trajectory.jsonl", "w")
        try:
            for k in range(self.max_steps):
                if self.agent_wall >= float(ep.budget.max_wall_s):
                    stop_reason = "wall_budget"
                    break
                if logical_tokens(self.policy_usage) >= int(ep.budget.max_policy_tokens):
                    stop_reason = "token_budget"
                    break
                state = {"max_steps": self.max_steps, "steps_left": self.max_steps - k,
                         "policy_tokens_left": max(0, int(ep.budget.max_policy_tokens) - logical_tokens(self.policy_usage)) // 1000 * 1000}
                if self.last_reply_tokens:
                    state["last_reply_tokens"] = int(round(self.last_reply_tokens, -3))
                msg = prompts.build_step_message(
                    k, graph, feedback_text, self.history, state,
                    window=cfg.history_window,
                    scrub=scrub)
                t0 = time.monotonic()
                try:
                    action, perr, raw, pusage = self.s.policy.propose(system, [{"role": "user", "content": msg}],
                                                                      tag=f"policy:{self.mode}")
                except Exception as ex:  # transport failure after the client's retries: stop, keep the record
                    dt = time.monotonic() - t0
                    self.policy_wall += dt
                    self.agent_wall += dt
                    err = f"policy call failed: {type(ex).__name__}: {ex}"
                    log.error("episode %s step %d: %s", ep.id, k, err)
                    self.notes.append(err)
                    self._record(traj_log, StepRecord(k, None, err, {"action_ok": False, "action_error": err,
                                                                   "policy_error": True}, [], empty_usage(),
                                                      graph.graph_fingerprint(), None, dt))
                    stop_reason = "policy_error"
                    break
                dt_pol = time.monotonic() - t0
                self.policy_wall += dt_pol
                self.agent_wall += dt_pol
                self.policy_usage = add_usage(self.policy_usage, pusage)
                self.last_reply_tokens = logical_tokens(pusage)
                io_log.write(_dumps({"step": k, "message": msg, "raw": raw, "parse_error": perr}) + "\n")
                io_log.flush()

                if action is None:
                    self.parse_failures += 1
                    text = f"Your reply could not be parsed as an action: {perr}"
                    fb = {"action_ok": False, "action_error": f"parse error: {perr}", "parse_error": True, "text": text}
                    self._mark_budget(k)
                    self._record(traj_log, StepRecord(k, None, perr, fb, [], pusage, graph.graph_fingerprint(),
                                                      None, dt_pol, raw=raw))
                    self.history.append({"step": k, "action": "(unparsable reply)", "result": f"parse error: {perr}"})
                    feedback_text = text
                    continue

                kept, dropped = normalize_uses(getattr(action, "uses", None), self.program)
                action = self._normalized_action(action, kept)
                note = prompts.uses_note(dropped)
                action_dict = action.to_dict() if hasattr(action, "to_dict") else {
                    "type": action.type, "payload": action.payload, "uses": kept}
                action_dict.pop("raw", None)   # the raw text is kept once, in StepRecord.raw
                thought = str(getattr(action, "thought", "") or "")
                self.actions.append(action)
                self.action_steps.append(k)

                if action.type == "finish":
                    self._mark_budget(k)
                    fb = {"action_ok": True, "action_error": None, "finish": True, "text": "finish"}
                    self._record(traj_log, StepRecord(k, action_dict, None, fb, kept, pusage, graph.graph_fingerprint(),
                                                      None, dt_pol, raw=raw, thought=thought, dropped_uses=dropped))
                    stop_reason = "finish"
                    break

                reason = check_orchestration(action, graph, self.orch, fixed_code)
                if reason is not None:
                    self.rejected += 1
                    self._mark_budget(k)
                    text = f"Action rejected: {reason}" + (f"\n{note}" if note else "")
                    fb = {"action_ok": False, "action_error": reason, "rejected_by": "orchestration", "text": text}
                    self._record(traj_log, StepRecord(k, action_dict, None, fb, kept, pusage, graph.graph_fingerprint(),
                                                      None, dt_pol, raw=raw, thought=thought, dropped_uses=dropped))
                    self.history.append({"step": k, "action": prompts.summarize_action(action_dict),
                                         "thought": thought, "result": f"rejected: {reason}"})
                    feedback_text = text
                    if self.orch == "single_turn":
                        stop_reason = "single_turn"
                        break
                    continue

                t1 = time.monotonic()
                try:
                    graph2, ckpt2, fbo, y = executor.apply(graph, checkpoint, action, k)
                except Exception as ex:  # executor bug or unexpected payload: graph unchanged, policy informed
                    dt_ex = time.monotonic() - t1
                    self.agent_wall += dt_ex
                    err = scrub_volatile(f"executor internal error: {type(ex).__name__}: {ex}", self.run_dir)
                    log.exception("episode %s step %d: %s", ep.id, k, err)
                    self.notes.append(f"step {k}: {err}")
                    self._mark_budget(k)
                    text = f"Action not applied: {err}" + (f"\n{note}" if note else "")
                    fb = {"action_ok": False, "action_error": err, "internal_error": True, "text": text}
                    self._record(traj_log, StepRecord(k, action_dict, None, fb, kept, pusage, graph.graph_fingerprint(),
                                                      None, dt_pol + dt_ex, raw=raw, thought=thought,
                                                      dropped_uses=dropped))
                    self.history.append({"step": k, "action": prompts.summarize_action(action_dict),
                                         "thought": thought, "result": err})
                    feedback_text = text
                    if self.orch == "single_turn":
                        stop_reason = "single_turn"
                        break
                    continue
                dt_ex = time.monotonic() - t1
                self.agent_wall += dt_ex
                graph, checkpoint = graph2, ckpt2
                fb = _fb_dict(fbo)
                if fb.get("action_ok", True):   # nu only counts uses of actions that were actually applied
                    for u in kept:
                        if u not in self.nu:
                            self.nu.append(u)
                self.infra_last = llm_outage_error(getattr(fbo, "records", None) or fb.get("records"))
                _sum_llm_usage(self.exec_usage, getattr(fbo, "llm_usage", None))
                self.node_runs += _count_node_runs(getattr(fbo, "records", None))
                self._mark_budget(k)
                self.dev_by_step[k] = _dev_score(getattr(fbo, "dev", None), getattr(ep, "direction", "max"))

                text = _render_feedback(fbo, bool(cfg.show_dev_score), scrub)
                text += f"\n{note}" if note else ""
                if not cfg.show_dev_score:
                    fb.pop("dev", None)
                fb["text"] = text

                sub_fp = _submitted_fp(graph)
                if y is not None and sub_fp is not None:
                    self.last_output = (k, graph.copy(), y, sub_fp)
                ev_idx: int | None = None
                want_replay = (self.mode == "source" and cfg.replay_on_new_submit and sub_fp is not None
                               and sub_fp != self.last_replayed_fp
                               and (y is not None or self.s.replay_failed_submits))
                if want_replay:
                    ev_idx = self._add_evidence(graph, y, k, sub_fp)
                self._record(traj_log, StepRecord(k, action_dict, None, fb, kept, pusage, graph.graph_fingerprint(),
                                                  ev_idx, dt_pol + dt_ex, raw=raw, thought=thought,
                                                  dropped_uses=dropped, submit_fp=sub_fp, y_available=y is not None))
                self.history.append({"step": k, "action": prompts.summarize_action(action_dict), "thought": thought,
                                     "result": prompts.summarize_feedback(
                                         _visible_feedback(fbo, bool(cfg.show_dev_score)), scrub=scrub)})
                feedback_text = text

                if self.orch == "single_turn":
                    stop_reason = "single_turn"
                    break
                if (self.mode == "source" and cfg.stop_on_first_pass and ev_idx is not None
                        and self.evidence[ev_idx].passed):
                    stop_reason = "first_pass"
                    break
        finally:
            io_log.close()
            traj_log.close()

        result = self._finalize(graph, retrieved, stop_reason, t_start)
        result.save(self.run_dir)
        return result

    # ------------------------------------------------------------------------------ pieces
    @staticmethod
    def _new_checkpoint(executor: Any, exec_dir: Path) -> Any:
        make = getattr(executor, "new_checkpoint", None)
        if callable(make):
            return make()
        from ..runtime.executor import Checkpoint

        values = exec_dir / "values"
        values.mkdir(parents=True, exist_ok=True)
        return Checkpoint(records={}, values_dir=str(values))

    @staticmethod
    def _normalized_action(action: Any, kept: list[str]) -> Any:
        """Copy of the action whose ``uses`` is ν restricted to the program (node origins record it)."""
        a = copy.copy(action)
        try:
            a.uses = list(kept)
        except (AttributeError, dataclasses.FrozenInstanceError) as ex:
            log.debug("action object is immutable (%s); keeping its original uses", ex)
            return action
        return a

    def _mark_budget(self, k: int) -> None:
        self.budget_at_step[k] = (logical_tokens(self.policy_usage), self.agent_wall)

    def _record(self, fh: Any, rec: StepRecord) -> None:
        self.steps.append(rec)
        fh.write(_dumps(rec.to_dict()) + "\n")
        fh.flush()

    def _within_budget(self, k: int) -> bool:
        tokens, wall = self.budget_at_step.get(k, (logical_tokens(self.policy_usage), self.agent_wall))
        b = self.ep.budget
        return tokens <= int(b.max_policy_tokens) and wall <= float(b.max_wall_s) and k < self.max_steps

    def _replay_eval(self, graph: WorkflowGraph, y_exec: Any, k: int) -> Evidence:
        """Eq. 8: reset replay of G_{t,k} + hidden evaluation -> e_{t,k} (never shown to the policy)."""
        ep = self.ep
        base = self.run_dir / "replay" / f"k{k:03d}"
        rdir, i = base, 1
        while rdir.exists():
            rdir = base.with_name(f"{base.name}_{i}")
            i += 1
        rdir.mkdir(parents=True)
        t0 = time.monotonic()
        replay_error: str | None = None
        try:
            y_rep, trace = self.s.replay(graph, ep, self.program, rdir)
        except Exception as ex:
            replay_error = f"{type(ex).__name__}: {ex}"
            log.exception("episode %s step %d: replay failed", ep.id, k)
            y_rep, trace = None, Trace(run_dir=str(rdir))
        self.replay_wall += time.monotonic() - t0
        self.replays += 1
        self.node_runs += _count_node_runs(getattr(trace, "records", None))
        _sum_llm_usage(self.exec_usage, getattr(trace, "llm_usage", None))

        if y_rep is not None and hasattr(ep, "run_probes"):
            def _probe_runner(derived: Any, name: str) -> tuple[Any, Any]:
                # sibling of the replay dir: the FoR39 ledger walks <run_dir>/exec and <run_dir>/replay/kNNN
                t1 = time.monotonic()
                try:
                    y_p, tr_p = self.s.replay(graph, derived, self.program, rdir.with_name(f"{rdir.name}_probe_{name}"))
                finally:
                    self.replay_wall += time.monotonic() - t1
                _sum_llm_usage(self.exec_usage, getattr(tr_p, "llm_usage", None))
                return y_p, tr_p
            try:
                ep.run_probes(y_rep, trace, _probe_runner)
            except Exception as ex:
                log.warning("episode %s step %d: probes raised %s: %s", ep.id, k, type(ex).__name__, ex)

        if y_exec is None:
            reproducible: bool | None = None if y_rep is None else False
        elif y_rep is None:
            reproducible = False
        else:
            try:
                reproducible = self.s.outputs_match(y_exec, y_rep, dict(ep.tolerance or {}))
            except Exception as ex:
                log.warning("episode %s step %d: outputs_match raised %s: %s", ep.id, k, type(ex).__name__, ex)
                reproducible = False
        try:
            ev = ep.evaluate(y_rep, trace)
        except Exception as ex:  # a deliverable the evaluator cannot score is a failed evaluation
            ev = EvalResult(direction=getattr(ep, "direction", "max"), completed=False,
                            details={"eval_error": f"{type(ex).__name__}: {ex}"})
        if replay_error:
            ev.details = dict(ev.details or {}, replay_error=replay_error)
        ev.reproducible = reproducible
        ev.within_budget = bool(ev.within_budget and self._within_budget(k))
        ev.z = int(bool(ev.completed and ev.hard_ok() and ev.accepted and ev.within_budget
                        and ev.reproducible is not False))
        ev.cost = dict(ev.cost or {}, replay_wall_s=float(getattr(trace, "wall_s", 0.0) or 0.0),
                       policy_tokens=float(self.budget_at_step.get(k, (logical_tokens(self.policy_usage), 0.0))[0]),
                       agent_wall_s=float(self.budget_at_step.get(k, (0, self.agent_wall))[1]))
        passed = passes(ev, self.s.require_acceptance)
        return Evidence(step=k, graph_dict=graph.to_dict(), y=y_rep, trace=trace, eval=ev, passed=passed,
                        graph_fp=graph.graph_fingerprint())

    def _add_evidence(self, graph: WorkflowGraph, y_exec: Any, k: int, sub_fp: str) -> int:
        ev = self._replay_eval(graph, y_exec, k)
        self.evidence.append(ev)
        self.evidence_submit_fp.append(sub_fp)
        self.evidence_y_exec.append(y_exec)
        self.last_replayed_fp = sub_fp
        return len(self.evidence) - 1

    def _finalize(self, graph: WorkflowGraph, retrieved: dict, stop_reason: str, t_start: float) -> SolveResult:
        ep = self.ep
        final_graph, y, trace, ev_final = graph, None, None, None
        final_step: int | None = None
        source = "none"
        passed = False
        passing = [(i, e) for i, e in enumerate(self.evidence) if e.passed]
        if passing:
            def key(item: tuple[int, Evidence]) -> tuple[float, int, int]:
                i, e = item
                d = self.dev_by_step.get(e.step)
                return (d if d is not None else -math.inf, e.step, i)

            _, best = max(passing, key=key)
            final_graph = WorkflowGraph.from_dict(best.graph_dict)
            y, trace, ev_final, final_step, source, passed = best.y, best.trace, best.eval, best.step, "pass", True
        elif self.last_output is not None:
            k, g, y_exec, sfp = self.last_output
            idx = next((i for i in range(len(self.evidence) - 1, -1, -1)
                        if self.evidence_submit_fp[i] == sfp and self.evidence_y_exec[i] is not None), None)
            if idx is None:
                idx = self._add_evidence(g, y_exec, k, sfp)
            e = self.evidence[idx]
            final_graph, y, trace, ev_final, final_step, source, passed = g, e.y, e.trace, e.eval, k, "last_output", e.passed
        if ev_final is None:
            ev_final = ep.evaluate(None, None)
            ev_final.within_budget = True
            ev_final.z = 0

        uses = set(self.nu)
        for n in final_graph.nodes.values():
            if n.kind == "operator" and n.ref:
                oid = _strip_prefix(n.ref, ("op:", "operator:"))
                if oid in self.program.operators:
                    uses.add(f"op:{oid}")
        uses_versions = set()
        for u in uses:
            kind, _, cid = u.partition(":")
            comp = self.program.skills.get(cid) if kind == "skill" else self.program.operators.get(cid)
            if comp is not None:
                uses_versions.add(comp.version_id)

        infra_error: str | None = None
        if stop_reason == "policy_error":
            infra_error = next((n for n in reversed(self.notes) if n.startswith("policy call failed")),
                               "policy call failed")
        elif trace is not None and llm_outage_error(getattr(trace, "records", None)):
            infra_error = llm_outage_error(trace.records)
        elif source == "none" and self.infra_last:
            infra_error = self.infra_last     # no output at all and the last executed graph died on an LLM outage
        if infra_error:
            self.notes.append(f"infrastructure error (not a task failure): {infra_error}")

        pu = self.policy_usage
        xu = self.exec_usage
        pol_p, pol_c = int(pu.get("prompt_tokens", 0)), int(pu.get("completion_tokens", 0))
        ex_p, ex_c = int(xu.get("prompt_tokens", 0)), int(xu.get("completion_tokens", 0))
        usage = {
            "policy_prompt_tokens": pol_p, "policy_completion_tokens": pol_c,
            "policy_reasoning_tokens": int(pu.get("reasoning_tokens", 0)),
            "executor_prompt_tokens": ex_p, "executor_completion_tokens": ex_c,
            "total_tokens": pol_p + pol_c + ex_p + ex_c,
            "llm_calls": int(pu.get("calls", 0)) + int(xu.get("calls", 0)),
            "wall_s": time.monotonic() - t_start, "policy_wall_s": self.policy_wall,
            "node_runs": self.node_runs, "replays": self.replays,
            # additive detail (cached work is recorded, not counted as spent)
            "policy_cached_prompt_tokens": int(pu.get("cached_prompt_tokens", 0)),
            "policy_cached_completion_tokens": int(pu.get("cached_completion_tokens", 0)),
            "policy_calls": int(pu.get("calls", 0)), "policy_cached_calls": int(pu.get("cached_calls", 0)),
            "executor_cached_prompt_tokens": int(xu.get("cached_prompt_tokens", 0)),
            "executor_cached_completion_tokens": int(xu.get("cached_completion_tokens", 0)),
            "executor_calls": int(xu.get("calls", 0)), "executor_cached_calls": int(xu.get("cached_calls", 0)),
            "cached_llm_calls": int(pu.get("cached_calls", 0)) + int(xu.get("cached_calls", 0)),
            "policy_logical_tokens": logical_tokens(pu), "executor_logical_tokens": logical_tokens(xu),
            "logical_tokens": logical_tokens(pu) + logical_tokens(xu), "agent_wall_s": self.agent_wall,
            "replay_wall_s": self.replay_wall, "steps": len(self.steps), "parse_failures": self.parse_failures,
            "rejected_actions": self.rejected,
        }
        return SolveResult(
            episode_id=str(ep.id), mode=self.mode, program_version=str(self.program.version),
            final_graph=final_graph, y=y, trace=trace, eval=ev_final, evidence=list(self.evidence),
            steps=list(self.steps), actions=list(self.actions), retrieved=retrieved, uses=uses, usage=usage,
            run_dir=str(self.run_dir), action_steps=list(self.action_steps), uses_versions=uses_versions,
            final_step=final_step, final_source=source, passed=passed, stop_reason=stop_reason,
            notes=list(self.notes), infra_error=infra_error,
        )
