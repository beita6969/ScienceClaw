"""A canvas orchestration session: the multi-turn loop of the paper (Eq. 5-8) driven from outside.

In a batch solver the engine owns the loop: it asks a policy for an atomic canvas edit, executes it, and shows the feedback. Inside a
gateway the policy is the agent that is already talking to the user, so the loop is inverted: the agent *acts* on a session
(one atomic edit per call) and gets the visible feedback back. Everything below that boundary is the engine unchanged:

* the typed workflow graph and its atomic edits (``core.actions``), executed incrementally against a checkpoint so that only
  the affected descendants rerun (``runtime.executor``, Eq. 7);
* reset replay of the submitted workflow in a fresh directory, with the reproducibility check and the hard-constraint
  verdict (``runtime.replay``, Eq. 8);
* the bookkeeping of the solver (steps, uses nu, evidence stream, budgets), reused so that a finished session yields a
  :class:`~scienceclaw.agent.solver.SolveResult` the evolution engine can learn from (Eq. 9-13).
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from scienceclaw.agent import prompts
from scienceclaw.agent.solver import (SolveResult, Solver, StepRecord, _SolveRun, _fb_dict, _render_feedback, _submitted_fp,
                                      normalize_uses, scrub_volatile)
from scienceclaw.config import SolverConfig
from scienceclaw.core.actions import parse_action
from scienceclaw.core.graph import WorkflowGraph
from scienceclaw.core.program import AgentProgram
from scienceclaw.core.retrieval import Retriever
from scienceclaw.llm.client import LLMError, empty_usage

log = logging.getLogger(__name__)


class NoModel:
    """Stand-in chat model for sessions without a configured language model: ``llm`` nodes then fail with an explanation."""

    cfg = None

    def role_config(self, role: str) -> Any:
        raise LLMError("no language model is configured for llm nodes: set SCIENCECLAW_MODEL and SCIENCECLAW_API_BASE_URL/KEY")

    def chat(self, role: str, messages: list[dict], **kw: Any) -> Any:
        self.role_config(role)

    def usage(self) -> dict:
        return {}

    def close(self) -> None:
        return None


def _jsonable(value: Any, limit: int = 20) -> Any:
    """A bounded, JSON-friendly preview of a deliverable."""
    try:
        import numpy as np
        import pandas as pd
        if isinstance(value, np.ndarray):
            return {"type": "array", "shape": list(value.shape), "dtype": str(value.dtype),
                    "head": value.reshape(-1)[:limit].tolist()}
        if isinstance(value, pd.DataFrame):
            return {"type": "table", "shape": list(value.shape), "columns": [str(c) for c in value.columns],
                    "head": json.loads(value.head(limit).to_json(orient="records"))}
        if isinstance(value, pd.Series):
            return {"type": "series", "length": len(value), "head": value.head(limit).tolist()}
    except Exception:                                   # a preview must never break a session
        pass
    if isinstance(value, (list, tuple)):
        return {"type": "list", "length": len(value), "head": [_jsonable(v, 5) for v in value[:limit]]}
    if isinstance(value, dict):
        return {"type": "dict", "keys": [str(k) for k in list(value)[:limit]]}
    return value if isinstance(value, (int, float, str, bool)) or value is None else repr(value)[:300]


class CanvasSession:
    """One task being solved on a typed workflow canvas, one atomic edit per :meth:`act`."""

    def __init__(self, episode: Any, program: AgentProgram, run_dir: str | Path, *, llm: Any = None,
                 cfg: SolverConfig | None = None, session_id: str | None = None, reveal_verdict: bool = True,
                 kind: str = "live", spec: dict | None = None) -> None:
        self.id = session_id or uuid.uuid4().hex[:12]
        self.episode = episode
        self.program = program
        self.kind = kind
        self.spec = spec
        self.reveal_verdict = reveal_verdict
        self.llm = llm if llm is not None else NoModel()
        steps = int(episode.budget.max_steps)
        self.cfg = cfg or SolverConfig(max_steps=steps)
        self.solver = Solver(self.cfg, self.llm)
        self.run_dir = Path(run_dir)
        self.run = _SolveRun(self.solver, episode, program, "source", self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        retriever = Retriever(program)
        self.skills = retriever.skills(episode, self.cfg.retrieve_skills_k)
        self.operators = retriever.operators(episode, self.cfg.retrieve_ops_k)
        self.retrieved = {"skills": [s.ref for s in self.skills], "operators": [o.ref for o in self.operators],
                          "skill_versions": [s.version_id for s in self.skills],
                          "operator_versions": [o.version_id for o in self.operators],
                          "slice_hash": retriever.slice_hash(episode, self.cfg.retrieve_skills_k, self.cfg.retrieve_ops_k)}
        self.graph = WorkflowGraph()
        exec_dir = self.run_dir / "exec"
        exec_dir.mkdir(parents=True, exist_ok=True)
        self.executor = self.solver.make_executor(episode, program, exec_dir)
        self.checkpoint = self.run._new_checkpoint(self.executor, exec_dir)
        self._scrub = lambda t: scrub_volatile(t, self.run_dir)
        self._traj = open(self.run_dir / "trajectory.jsonl", "w")
        self._t0 = time.monotonic()
        self.k = 0
        self.closed = False
        self.result: SolveResult | None = None
        self._last_y: Any = None
        self._last_fp: str | None = None

    # ------------------------------------------------------------------------------------------- context
    def context(self) -> str:
        """The protocol, the task, its tools and the retrieved Skills/Operators: everything the acting agent needs."""
        return prompts.build_system_prompt(self.episode, self.skills, self.operators, max_steps=self.run.max_steps,
                                           show_dev_score=bool(self.cfg.show_dev_score))

    def render(self) -> str:
        return self.graph.render_compact()

    @property
    def steps_left(self) -> int:
        return max(0, self.run.max_steps - self.k)

    # ---------------------------------------------------------------------------------------------- act
    def act(self, action: str | dict) -> dict[str, Any]:
        """Apply one atomic edit (the JSON of the Actions section of :meth:`context`) and execute what it affects."""
        if self.closed:
            return {"ok": False, "text": "The session is finished.", "steps_left": 0}
        r, k = self.run, self.k
        if k >= r.max_steps:
            return {"ok": False, "text": "The step budget is exhausted: call finish.", "steps_left": 0, "budget_exhausted": True}
        raw = action if isinstance(action, str) else json.dumps(action)
        t_start = time.monotonic()
        parsed, perr = parse_action(raw)
        if parsed is None:
            r.parse_failures += 1
            text = f"Your reply could not be parsed as an action: {perr}"
            fb = {"action_ok": False, "action_error": f"parse error: {perr}", "parse_error": True, "text": text}
            r._mark_budget(k)
            r._record(self._traj, StepRecord(k, None, perr, fb, [], empty_usage(), self.graph.graph_fingerprint(), None,
                                             time.monotonic() - t_start, raw=raw))
            r.history.append({"step": k, "action": "(unparsable reply)", "result": f"parse error: {perr}"})
            self.k += 1
            return self._reply(False, text, k)

        kept, dropped = normalize_uses(getattr(parsed, "uses", None), self.program)
        parsed = r._normalized_action(parsed, kept)
        note = prompts.uses_note(dropped)
        adict = parsed.to_dict() if hasattr(parsed, "to_dict") else {"type": parsed.type, "payload": parsed.payload, "uses": kept}
        adict.pop("raw", None)
        thought = str(getattr(parsed, "thought", "") or "")
        r.actions.append(parsed)
        r.action_steps.append(k)

        if parsed.type == "finish":
            r._mark_budget(k)
            r._record(self._traj, StepRecord(k, adict, None, {"action_ok": True, "action_error": None, "finish": True, "text": "finish"},
                                             kept, empty_usage(), self.graph.graph_fingerprint(), None, 0.0, raw=raw,
                                             thought=thought, dropped_uses=dropped))
            self.k += 1
            return self.finish()

        t1 = time.monotonic()
        try:
            graph2, ckpt2, fbo, y = self.executor.apply(self.graph, self.checkpoint, parsed, k)
        except Exception as ex:                          # an executor defect must not kill the session
            r.agent_wall += time.monotonic() - t1
            err = scrub_volatile(f"executor internal error: {type(ex).__name__}: {ex}", self.run_dir)
            log.exception("session %s step %d: %s", self.id, k, err)
            r.notes.append(f"step {k}: {err}")
            r._mark_budget(k)
            text = f"Action not applied: {err}" + (f"\n{note}" if note else "")
            r._record(self._traj, StepRecord(k, adict, None, {"action_ok": False, "action_error": err, "internal_error": True, "text": text},
                                             kept, empty_usage(), self.graph.graph_fingerprint(), None, time.monotonic() - t1,
                                             raw=raw, thought=thought, dropped_uses=dropped))
            r.history.append({"step": k, "action": prompts.summarize_action(adict), "thought": thought, "result": err})
            self.k += 1
            return self._reply(False, text, k)

        dt = time.monotonic() - t1
        r.agent_wall += dt
        self.graph, self.checkpoint = graph2, ckpt2
        fb = _fb_dict(fbo)
        if fb.get("action_ok", True):
            for u in kept:
                if u not in r.nu:
                    r.nu.append(u)
        from scienceclaw.agent.solver import _count_node_runs, _dev_score, _sum_llm_usage, _visible_feedback, llm_outage_error
        r.infra_last = llm_outage_error(getattr(fbo, "records", None) or fb.get("records"))
        _sum_llm_usage(r.exec_usage, getattr(fbo, "llm_usage", None))
        r.node_runs += _count_node_runs(getattr(fbo, "records", None))
        r._mark_budget(k)
        r.dev_by_step[k] = _dev_score(getattr(fbo, "dev", None), getattr(self.episode, "direction", "max"))

        text = _render_feedback(fbo, bool(self.cfg.show_dev_score), self._scrub) + (f"\n{note}" if note else "")
        if not self.cfg.show_dev_score:
            fb.pop("dev", None)
        fb["text"] = text
        sub_fp = _submitted_fp(self.graph)
        if y is not None and sub_fp is not None:
            r.last_output = (k, self.graph.copy(), y, sub_fp)
            self._last_y, self._last_fp = y, sub_fp
        r._record(self._traj, StepRecord(k, adict, None, fb, kept, empty_usage(), self.graph.graph_fingerprint(), None,
                                         dt, raw=raw, thought=thought, dropped_uses=dropped, submit_fp=sub_fp,
                                         y_available=y is not None))
        r.history.append({"step": k, "action": prompts.summarize_action(adict), "thought": thought,
                          "result": prompts.summarize_feedback(_visible_feedback(fbo, bool(self.cfg.show_dev_score)), scrub=self._scrub)})
        self.k += 1
        self._save_state()
        out = self._reply(bool(fb.get("action_ok", True)), text, k)
        out["submit_ready"] = bool(getattr(fbo, "submit_ready", False))
        return out

    def _reply(self, ok: bool, text: str, step: int) -> dict[str, Any]:
        return {"ok": ok, "step": step, "text": text, "steps_left": self.steps_left,
                "graph_fingerprint": self.graph.graph_fingerprint()}

    # ------------------------------------------------------------------------------------------ replay
    def replay(self) -> dict[str, Any]:
        """Reset replay of the submitted workflow (Eq. 8): regenerates the output from a clean state and verifies it."""
        sub_fp = _submitted_fp(self.graph)
        if sub_fp is None:
            return {"ok": False, "text": "Nothing to replay: wire the result into the submit node's input y first."}
        if self._last_fp != sub_fp or self._last_y is None:
            self._last_y = None                         # the executor has not produced an output for this graph
        idx = self.run._add_evidence(self.graph, self._last_y, max(self.k - 1, 0), sub_fp)
        ev = self.run.evidence[idx]
        e = ev.eval
        h, msgs = self._visible_constraints(e)
        out: dict[str, Any] = {"ok": True, "replayed": True, "evidence_index": idx, "reproducible": e.reproducible,
                               "hard_constraints": h, "constraint_messages": msgs}
        if self.reveal_verdict:
            out.update(verdict="PASS" if ev.passed else "FAIL", passed=bool(ev.passed), z=int(e.z), completed=bool(e.completed))
            out["text"] = (f"Replay from a reset state: {'PASS' if ev.passed else 'FAIL'}; reproducible={e.reproducible}; "
                           f"constraints={h}")
        else:
            out["text"] = f"Replay recorded from a reset state; reproducible={e.reproducible}. The evaluator verdict is not shown."
        return out

    def _visible_constraints(self, e: Any) -> tuple[dict[str, bool], dict[str, str]]:
        """Hard-constraint results as the acting agent may see them: all of them once the verdict is revealed, otherwise
        only the constraints the task declares visible (evaluator-only checks stay sealed)."""
        if self.reveal_verdict:
            return dict(e.h), dict(e.h_msgs)
        shown = {c.name for c in self.episode.constraints if c.visible}
        return ({k: v for k, v in e.h.items() if k in shown}, {k: v for k, v in e.h_msgs.items() if k in shown})

    # ----------------------------------------------------------------------------------------- finish
    def finish(self) -> dict[str, Any]:
        """Close the session: select the final solution G*, replay it if needed and write the receipts."""
        if self.closed:
            return self._summary()
        self._traj.close()
        self.result = self.run._finalize(self.graph, self.retrieved, "finish", self._t0)
        self.result.save(self.run_dir)
        self.closed = True
        self._save_state()
        return self._summary()

    def _summary(self) -> dict[str, Any]:
        res = self.result
        out: dict[str, Any] = {"ok": True, "done": True, "session_id": self.id, "steps": len(self.run.steps),
                               "uses": sorted(res.uses) if res else [], "run_dir": str(self.run_dir),
                               "deliverable": _jsonable(res.y) if res else None,
                               "reproducible": getattr(res.eval, "reproducible", None) if res else None,
                               "hard_constraints": self._visible_constraints(res.eval)[0] if res else {}}
        if self.reveal_verdict and res:
            out.update(verdict="PASS" if res.passed else "FAIL", passed=bool(res.passed), z=res.z)
        out["text"] = ("Session finished. " + (f"Verdict: {out['verdict']}. " if "verdict" in out else "")
                       + f"Receipts: {self.run_dir}")
        return out

    # ----------------------------------------------------------------------------------------- state
    def _save_state(self) -> None:
        try:
            (self.run_dir / "graph.json").write_text(json.dumps(self.graph.to_dict(), indent=1, default=str))
            (self.run_dir / "session.json").write_text(json.dumps({
                "id": self.id, "kind": self.kind, "program_version": self.program.version, "step": self.k,
                "closed": self.closed, "episode": getattr(self.episode, "id", ""), "spec": self.spec,
                "retrieved": self.retrieved}, indent=1, default=str))
        except OSError as ex:
            log.warning("session %s: cannot persist state: %s", self.id, ex)

    def status(self) -> dict[str, Any]:
        return {"session_id": self.id, "kind": self.kind, "step": self.k, "steps_left": self.steps_left, "closed": self.closed,
                "nodes": len(self.graph.nodes), "episode": getattr(self.episode, "id", ""), "program": self.program.version}
