"""Line-delimited JSON-RPC service: the single entry point of the gateway plugin into the ScienceClaw engine.

``python -m scienceclaw.rpc`` reads one JSON request per line on stdin and writes one JSON response per line on stdout::

    {"id": 7, "method": "canvas.act", "params": {"session_id": "ab12", "action": {...}}}
    {"id": 7, "ok": true,  "result": {...}}
    {"id": 7, "ok": false, "error": {"type": "ValueError", "message": "..."}}

The process is long-lived so that canvas sessions keep their graphs, checkpoints and evidence between calls. Methods:

* ``canvas.open|act|render|replay|finish|status|list``: typed workflow orchestration of live tasks;
* ``tools.search|show|status`` and ``weights.status|plan``: the scientific tool library and its pretrained weights;
* ``program.summary|skills|operators|show|history|rollback``: the versioned agent program (Skills and Operators);
* ``evolve.val_add|val_list|val_remove|propose|gate|run|status|candidates|candidate``: program self-evolution from finished
  live sessions (``propose``, ``gate`` and ``run`` are background jobs polled with ``evolve.status``).

Environment: ``SCIENCECLAW_HOME`` (state, default ``~/.scienceclaw``), ``SCIENCECLAW_INPUT_ROOTS`` (directories live tasks may
read, ``os.pathsep``-separated, default: the working directory), ``SCIENCECLAW_CONFIG`` (optional run config with the ``llm``
section), ``SCIENCECLAW_DATA_ROOT`` (protected from code nodes) and ``SCIENCECLAW_MODELS``.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("scienceclaw.rpc")
_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{4,40}$")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    return value


class Service:
    """All engine state of one worker process."""

    def __init__(self) -> None:
        self.home = Path(os.environ.get("SCIENCECLAW_HOME") or Path.home() / ".scienceclaw").expanduser()
        self.sessions: dict[str, Any] = {}
        self._store = None
        self._llm = None
        self._cfg = None
        self._evolution = None
        roots = os.environ.get("SCIENCECLAW_INPUT_ROOTS")
        self.input_roots = [Path(p).expanduser() for p in roots.split(os.pathsep) if p] if roots else [Path.cwd()]
        from scienceclaw.canvas.live import check_roots, deny_paths
        deny_paths([self.home, Path.home() / ".config" / "scienceclaw", Path.home() / ".ssh"])
        self.root_problems = check_roots(self.input_roots)
        self.methods: dict[str, Callable[[dict], Any]] = {
            "ping": self.ping, "setup.status": self.setup_status, "setup.start": self.setup_start,
            "canvas.open": self.canvas_open, "canvas.act": self.canvas_act, "canvas.render": self.canvas_render,
            "canvas.replay": self.canvas_replay, "canvas.finish": self.canvas_finish, "canvas.status": self.canvas_status,
            "canvas.list": self.canvas_list,
            "tools.search": self.tools_search, "tools.show": self.tools_show, "tools.status": self.tools_status,
            "weights.status": self.weights_status, "weights.plan": self.weights_plan,
            "program.summary": self.program_summary, "program.skills": self.program_skills,
            "program.operators": self.program_operators, "program.show": self.program_show,
            "program.history": self.program_history, "program.rollback": self.program_rollback,
            "evolve.val_add": self.evolve_val_add, "evolve.val_list": self.evolve_val_list,
            "evolve.val_remove": self.evolve_val_remove, "evolve.propose": self.evolve_propose,
            "evolve.gate": self.evolve_gate, "evolve.run": self.evolve_run, "evolve.status": self.evolve_status,
            "evolve.candidates": self.evolve_candidates, "evolve.candidate": self.evolve_candidate,
        }

    # ------------------------------------------------------------------------------------------ shared
    @property
    def store(self):
        if self._store is None:
            from scienceclaw.program import ProgramStore
            self._store = ProgramStore(self.home / "program")
        return self._store

    def program(self):
        return self.store.open()

    def run_config(self):
        if self._cfg is None:
            from scienceclaw.config import LLMConfig, RunConfig, load_config
            cfg_path = os.environ.get("SCIENCECLAW_CONFIG")
            self._cfg = load_config(cfg_path) if cfg_path else RunConfig(llm=LLMConfig(cache_path=str(self.home / "llm_cache.sqlite")))
        return self._cfg

    def llm(self):
        if self._llm is None:
            from scienceclaw.llm import build_chat_model
            self._llm = build_chat_model(self.run_config().llm)
        return self._llm

    @property
    def evolution(self):
        if self._evolution is None:
            from scienceclaw.evolution import LiveEvolution
            self._evolution = LiveEvolution(self.store, self.home, self.llm(), self.run_config(), self.input_roots,
                                            auto_promote=os.environ.get("SCIENCECLAW_AUTO_PROMOTE") == "1")
        return self._evolution

    def _session(self, params: dict):
        sid = str(params.get("session_id", ""))
        if sid not in self.sessions:
            raise KeyError(f"unknown session {sid!r}; open one with canvas.open")
        return self.sessions[sid]

    @staticmethod
    def _need(params: dict, *names: str) -> None:
        missing = [n for n in names if params.get(n) in (None, "")]
        if missing:
            raise ValueError(f"missing parameter(s): {', '.join(missing)}")

    def ping(self, p: dict) -> dict:
        return {"pong": True, "pid": os.getpid(), "sessions": len(self.sessions), "home": str(self.home)}

    # ------------------------------------------------------------------------------------------ canvas
    def setup_status(self, p: dict) -> dict:
        from scienceclaw import bootstrap
        st = bootstrap.status()
        rec = bootstrap.read_state() or {}
        return {**st, "download_gb": rec.get("download_gb"), "results": [{"id": r["id"], "status": r["status"]} for r in rec.get("results", [])],
                "model_root": rec.get("model_root")}

    def setup_start(self, p: dict) -> dict:
        from scienceclaw import bootstrap
        return bootstrap.start_background(p.get("profile"))

    def canvas_open(self, p: dict) -> dict:
        from scienceclaw import bootstrap
        from scienceclaw.canvas import CanvasSession, build_live_episode
        bootstrap.require()
        if self.root_problems:
            raise ValueError("unsafe input roots: " + "; ".join(self.root_problems) + " (set inputRoots to a dedicated workspace)")
        task = p.get("task")
        if not isinstance(task, dict):
            raise ValueError("canvas.open needs a 'task' object (objective, inputs, required_output, constraints)")
        self._evict_closed()
        sid = uuid.uuid4().hex[:12]
        run_dir = self.home / "sessions" / sid
        program = self.program()
        ep = build_live_episode(task, task_id=f"live-{sid}", input_roots=self.input_roots)
        session = CanvasSession(ep, program, run_dir, llm=self.llm(), session_id=sid, kind="live", spec=task)
        self.sessions[sid] = session
        return {"session_id": sid, "kind": session.kind, "program": program.version, "steps": session.run.max_steps,
                "retrieved": session.retrieved, "context": session.context()}

    def _evict_closed(self, keep: int = 64) -> None:
        """Forget the oldest finished sessions once more than ``keep`` are held (their receipts stay on disk)."""
        closed = [sid for sid, s in self.sessions.items() if s.closed]
        for sid in closed[: max(0, len(self.sessions) - keep)]:
            del self.sessions[sid]

    def canvas_act(self, p: dict) -> dict:
        self._need(p, "session_id", "action")
        return self._session(p).act(p["action"])

    def canvas_render(self, p: dict) -> dict:
        s = self._session(p)
        return {"graph": s.render(), "status": s.status()}

    def canvas_replay(self, p: dict) -> dict:
        s = self._session(p)
        if s.closed:
            raise ValueError("the session is finished; open a new one to replay again")
        return s.replay()

    def canvas_finish(self, p: dict) -> dict:
        return self._session(p).finish()

    def canvas_status(self, p: dict) -> dict:
        return self._session(p).status()

    def canvas_list(self, p: dict) -> dict:
        return {"sessions": [s.status() for s in self.sessions.values()]}

    # ------------------------------------------------------------------------------------------ tools
    def tools_search(self, p: dict) -> dict:
        from scienceclaw import tools
        self._need(p, "query")
        hits = tools.search(str(p["query"]), int(p.get("k", 8)), kind=p.get("kind"),
                            available_only=bool(p.get("available_only", False)))
        return {"tools": [dict(e.to_dict(), card=e.card()) for e in hits]}

    def tools_show(self, p: dict) -> dict:
        from scienceclaw import tools
        self._need(p, "target")
        target = str(p["target"])
        try:
            e = tools.get(target)
            return {"tool": e.to_dict(), "card": e.card(full=True), "probe": tools.probe(e.module)}
        except KeyError:
            return {"module": target, "doc": tools.module_doc(target), "probe": tools.probe(target),
                    "functions": [e.name for e in tools.catalog() if e.module == target]}

    def tools_status(self, p: dict) -> dict:
        from scienceclaw import tools
        return {"modules": tools.status_table(p.get("modules") or None)}

    def weights_status(self, p: dict) -> dict:
        from scienceclaw.tools import weights as W
        return {"model_root": str(W.model_root()), "assets": W.status_all()}

    def weights_plan(self, p: dict) -> dict:
        from scienceclaw.tools import weights as W
        return {"commands": W.plan(p.get("ids") or None, p.get("root"))}

    # ----------------------------------------------------------------------------------------- program
    def program_summary(self, p: dict) -> dict:
        prog = self.program()
        return {**prog.summary(), "head": self.store.head(), "versions": self.store.versions()}

    def program_skills(self, p: dict) -> dict:
        return {"skills": [{"id": s.id, "version": s.version, "title": s.title, "tags": s.tags, "source": s.provenance.get("source")}
                           for s in self.program().skills.values()]}

    def program_operators(self, p: dict) -> dict:
        return {"operators": [{"id": o.id, "version": o.version, "signature": o.signature(), "tags": o.tags,
                               "source": o.provenance.get("source")} for o in self.program().operators.values()]}

    def program_show(self, p: dict) -> dict:
        self._need(p, "ref")
        prog, ref = self.program(), str(p["ref"])
        kind, _, cid = ref.partition(":")
        comp = prog.skills.get(cid) if kind == "skill" else prog.operators.get(cid) if kind in ("op", "operator") else None
        if comp is None:
            raise KeyError(f"unknown component {ref!r}; use skill:<id> or op:<id>")
        return {"ref": ref, "version": comp.version_id, "text": comp.render()}

    def program_history(self, p: dict) -> dict:
        return {"head": self.store.head(), "versions": self.store.versions(), "receipts": self.store.history()}

    def program_rollback(self, p: dict) -> dict:
        self._need(p, "version")
        return {"head": self.store.rollback(str(p["version"]))}

    # ------------------------------------------------------------------------------------------ evolve
    def _finished_live_session(self, p: dict):
        s = self._session(p)
        if s.kind != "live" or not s.closed:
            raise ValueError("this needs a finished live session (canvas.finish)")
        return s

    def evolve_val_add(self, p: dict) -> dict:
        from scienceclaw import bootstrap
        bootstrap.require()
        if p.get("session_id"):
            s = self._finished_live_session(p)
            if not (s.result is not None and s.result.passed):
                raise ValueError("only a task whose session passed verification can serve as a validation task")
            spec = s.spec
        else:
            spec = p.get("task")
            if not isinstance(spec, dict):
                raise ValueError("evolve.val_add needs 'task' (a task declaration) or 'session_id' (a finished, passed session)")
        return self.evolution.val_add(spec, p.get("id"))

    def evolve_val_list(self, p: dict) -> dict:
        return {"tasks": [{"id": r["id"], "objective": r["spec"].get("objective", ""), "added": r["added"]}
                          for r in self.evolution.val_list()]}

    def evolve_val_remove(self, p: dict) -> dict:
        self._need(p, "id")
        return self.evolution.val_remove(str(p["id"]))

    def evolve_propose(self, p: dict) -> dict:
        from scienceclaw import bootstrap
        bootstrap.require()
        self._need(p, "session_id")
        s, evo = self._finished_live_session(p), self.evolution
        return evo.start_job("propose", lambda: evo.propose(s, p.get("variant")))

    def evolve_gate(self, p: dict) -> dict:
        self._need(p, "candidate_id")
        evo, cid = self.evolution, str(p["candidate_id"])
        evo.candidate(cid)
        return evo.start_job("gate", lambda: evo.gate(cid))

    def evolve_run(self, p: dict) -> dict:
        self._need(p, "session_id")
        s, evo = self._finished_live_session(p), self.evolution
        return evo.start_job("run", lambda: evo.run(s, p.get("variant")))

    def evolve_status(self, p: dict) -> dict:
        return self.evolution.job(p.get("job_id"), float(p.get("wait_s") or 0))

    def evolve_candidates(self, p: dict) -> dict:
        return {"candidates": self.evolution.candidates(), "head": self.program().version}

    def evolve_candidate(self, p: dict) -> dict:
        self._need(p, "candidate_id")
        rec = self.evolution.candidate(str(p["candidate_id"]))
        return {k: v for k, v in rec.items() if k != "source"} | {"source_session": rec["source"]["session_id"]}

    # ---------------------------------------------------------------------------------------- dispatch
    def handle(self, line: str) -> dict:
        rid: Any = None
        try:
            req = json.loads(line)
            if not isinstance(req, dict):
                raise ValueError("a request must be a JSON object")
            rid = req.get("id")
            method = str(req.get("method", ""))
            fn = self.methods.get(method)
            if fn is None:
                raise ValueError(f"unknown method {method!r}; known: {sorted(self.methods)}")
            params = req.get("params") or {}
            if not isinstance(params, dict):
                raise ValueError("params must be a JSON object")
            return {"id": rid, "ok": True, "result": _json_safe(fn(params))}
        except Exception as ex:                          # every failure becomes a structured answer
            if not isinstance(ex, (ValueError, KeyError, RuntimeError)) or os.environ.get("SCIENCECLAW_RPC_DEBUG"):
                log.error("request failed:\n%s", traceback.format_exc())
            msg = ex.args[0] if isinstance(ex, KeyError) and ex.args else str(ex)
            return {"id": rid, "ok": False, "error": {"type": type(ex).__name__, "message": str(msg)}}


def serve(stdin=None, stdout=None) -> int:
    """Answer requests until stdin closes. Library output on stdout is redirected to stderr so the protocol stays clean."""
    stdin, out = stdin or sys.stdin, stdout or sys.stdout
    sys.stdout = sys.stderr
    svc = Service()
    for line in stdin:
        if not line.strip():
            continue
        resp = svc.handle(line)
        out.write(json.dumps(resp, ensure_ascii=False, default=str, allow_nan=False) + "\n")
        out.flush()
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(name)s %(levelname)s %(message)s")
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
