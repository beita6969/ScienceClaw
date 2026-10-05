"""Program self-evolution driven by live canvas sessions (paper Section 5.2, Eq. 2-3 and 9-13).

The batch :class:`~scienceclaw.evolution.evolver.Evolver` learns from a pre-defined source stream. In a gateway the source
stream is the user's own work: each finished, replay-verified canvas session is a source episode e_src, and the validation
set D_val is a set of tasks the user registered as representative. The same machinery applies unchanged:

``finished session -> evolution instance (e-, e+, delta) -> bundle B = (dS, dO) -> Apply(A_r; B) -> R_src -> D_val gate -> promote``

* **propose** builds the linked Skill/Operator bundle from the session (attribution, edit split, Patch_Theta0 skill candidates,
  boundary-replayed operator candidates) and stores it as a pending candidate. Nothing about the active program changes.
* **gate** re-solves the source task with the candidate (R_src = Pass and Use), then solves D_val with the incumbent and the
  candidate under the frozen language model, and admits the candidate only if H_val holds (absolute: every hard constraint on
  every validation task), the cost stays within the budget B and Q_val = MacroSR strictly improves. A candidate is never
  admitted without enough validation tasks, the task it was learned from can never validate it, and a candidate derived from an
  older program is refused when a component it changes has moved on.
* **promote** makes an admitted candidate the head of the :class:`ProgramStore` with a receipt. It is the user's decision
  (``scienceclaw live promote``); only with ``auto_promote`` does the gate promote by itself.

Both stages call the language model, so they run as background jobs while the engine keeps serving canvas sessions.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from scienceclaw.canvas.live import SpecError, build_live_episode, resolve_input
from scienceclaw.config import RunConfig
from scienceclaw.core.program import AgentProgram, Bundle
from scienceclaw.evolution.attribution import extract_instances
from scienceclaw.evolution.bundle import build_bundle, bundle_summary
from scienceclaw.evolution.validation import (ValidationGate, ValReport, infra_error_of, source_pass, source_replay_check,
                                              use_check)
from scienceclaw.evolution.variants import bundles_for_variant, check_variant
from scienceclaw.llm.client import LLMError
from scienceclaw.program.store import ProgramStore, StaleHead

log = logging.getLogger(__name__)

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,39}$")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str, ensure_ascii=False))
    tmp.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def task_key(spec: dict, input_roots: list[Path]) -> str:
    """Identity of a live task: its objective and the resolved input files (the key that keeps D_src and D_val disjoint)."""
    inputs = sorted([str(item.get("name", "")), str(resolve_input(str(item.get("path", "")), input_roots))]
                    for item in spec.get("inputs") or [])
    payload = {"objective": " ".join(str(spec.get("objective", "")).split()), "inputs": inputs}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# Eq. 2-3 as written: every hard constraint holds on every validation task, and MacroSR must strictly improve.
LIVE_GATE = {"hval_mode": "absolute", "qval": "macrosr"}


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def task_digest(spec: dict, input_roots: list[Path]) -> str:
    """Content identity of a live task: the whole declaration and the bytes of every input and held-out file it reads."""
    files = {}
    for item in spec.get("inputs") or []:
        files[f"input:{item.get('name')}"] = _file_digest(resolve_input(str(item.get("path", "")), input_roots))
    for c in spec.get("constraints") or []:
        if c.get("check") == "metric":
            files[f"holdout:{c.get('name') or 'metric'}"] = _file_digest(resolve_input(str(c.get("target", "")), input_roots))
    payload = {"spec": spec, "files": files}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _brief(rep: ValReport) -> dict[str, Any]:
    return {"program": rep.program_version, "macro_sr": rep.macro_sr, "norm_score": rep.norm_score, "h_val": rep.h_val,
            "logical_tokens": rep.cost.get("logical_tokens"), "wall_s": rep.wall_s, "resolved": rep.n_resolved,
            "reused": rep.n_reused}


class LiveEvolution:
    """Candidate generation, gating and promotion for one program store."""

    def __init__(self, store: ProgramStore, home: Path, llm: Any, cfg: RunConfig, input_roots: list[Path], *,
                 auto_promote: bool = False, min_val_tasks: int = 2) -> None:
        self.auto_promote = auto_promote
        self.min_val_tasks = max(1, int(min_val_tasks))
        self.store = store
        self.root = Path(home) / "evolution"
        self.llm = llm
        self.cfg = cfg
        self.input_roots = input_roots
        self.jobs: dict[str, dict[str, Any]] = {}
        self._done: dict[str, threading.Event] = {}
        self._jobs_lock = threading.Lock()

    # ------------------------------------------------------------------------------------ validation tasks
    def _val_dir(self) -> Path:
        return self.root / "val"

    def val_add(self, spec: dict, vid: str | None = None) -> dict[str, Any]:
        """Register a task as a member of D_val. Its spec must build (inputs readable, constraints valid)."""
        key = task_key(spec, self.input_roots)
        vid = vid or f"v{key[:8]}"
        if not _ID.match(vid):
            raise SpecError(f"validation id {vid!r} must be letters, digits, '_', '.' or '-'")
        for row in self.val_list():
            if row["key"] == key:
                raise SpecError(f"this task is already registered as {row['id']!r}")
            if row["id"] == vid:
                raise SpecError(f"validation id {vid!r} is taken")
        digest = task_digest(spec, self.input_roots)
        build_live_episode(spec, task_id=f"val-{vid}-{digest[:10]}", input_roots=self.input_roots)
        row = {"id": vid, "key": key, "digest": digest, "added": time.strftime("%Y-%m-%dT%H:%M:%S"), "spec": spec}
        _write_json(self._val_dir() / f"{vid}.json", row)
        return {k: v for k, v in row.items() if k != "spec"} | {"objective": spec.get("objective", "")}

    def val_list(self) -> list[dict[str, Any]]:
        d = self._val_dir()
        return [_read_json(p) for p in sorted(d.glob("*.json"))] if d.is_dir() else []

    def val_remove(self, vid: str) -> dict[str, Any]:
        path = self._val_dir() / f"{vid}.json"
        if not _ID.match(vid) or not path.is_file():
            raise KeyError(f"no validation task {vid!r}")
        path.unlink()
        return {"removed": vid, "remaining": len(self.val_list())}

    def _val_episodes(self) -> tuple[list[Any], dict[str, str]]:
        """The episodes of D_val and, per task that cannot be used as registered, why (the gate refuses to run then)."""
        episodes, errors = [], {}
        for row in self.val_list():
            try:
                digest = task_digest(row["spec"], self.input_roots)
                if digest != row.get("digest"):
                    raise SpecError("its inputs or held-out data changed since it was registered; remove it and register it again")
                episodes.append(build_live_episode(row["spec"], task_id=f"val-{row['id']}-{digest[:10]}",
                                                   input_roots=self.input_roots))
            except (SpecError, OSError) as ex:
                errors[row["id"]] = str(ex)
        return episodes, errors

    # --------------------------------------------------------------------------------------- candidates
    def _cdir(self, cid: str) -> Path:
        return self.root / "candidates" / cid

    def _new_id(self) -> str:
        d = self.root / "candidates"
        n = len(list(d.iterdir())) if d.is_dir() else 0
        while (d / f"c{n + 1:04d}").exists():
            n += 1
        cid = f"c{n + 1:04d}"
        (d / cid).mkdir(parents=True)
        return cid

    def candidates(self) -> list[dict[str, Any]]:
        d = self.root / "candidates"
        out = []
        for p in sorted(d.glob("c*/record.json")) if d.is_dir() else []:
            r = _read_json(p)
            out.append({"id": r["id"], "status": r["status"], "variant": r.get("variant"), "part": r.get("part"),
                        "bundle": r.get("bundle"), "created": r.get("created"),
                        "decision": (r.get("decision") or {}).get("reason")})
        return out

    def candidate(self, cid: str) -> dict[str, Any]:
        path = self._cdir(cid) / "record.json"
        if not re.fullmatch(r"c\d{4,}", cid) or not path.is_file():
            raise KeyError(f"no candidate {cid!r}; known: {[c['id'] for c in self.candidates()]}")
        return _read_json(path)

    def _save(self, rec: dict[str, Any]) -> None:
        _write_json(self._cdir(rec["id"]) / "record.json", rec)

    def propose(self, session: Any, variant: str | None = None) -> dict[str, Any]:
        """Build candidate bundles from a finished live session (Eq. 9-12) and store them as pending."""
        if getattr(session, "kind", "") != "live":
            raise ValueError("only finished live sessions can be evolved here")
        if not session.closed or session.result is None:
            raise ValueError("finish the session first: evolution learns from its replay-verified result")
        evo = self.cfg.evolution
        variant = check_variant(variant or evo.variant)
        if variant == "frozen":
            raise ValueError("variant 'frozen' learns nothing")
        res = session.result
        req = bool(getattr(evo, "pass_requires_acceptance", True))
        if not source_pass(res, req):
            return {"status": "no_instance", "candidates": [],
                    "reason": "the session has no replay-verified success to learn from: replay a workflow that satisfies "
                              "every constraint, then finish"}
        instances = extract_instances(res, int(evo.instances_per_episode))
        key = task_key(session.spec, self.input_roots)
        made: list[dict[str, Any]] = []
        notes: list[str] = []
        for j, inst in enumerate(instances):
            scratch = self.root / "scratch" / f"{session.id}_{j}"
            bundle, blog = build_bundle(inst, session.program, self.llm, session.episode, scratch, variant,
                                        repeats=int(evo.breplay_repeats))
            parts = bundles_for_variant(bundle, variant)
            if not parts:
                notes.append(f"instance {j}: empty bundle (no abstractable repair)")
            for part in parts:
                cid = self._new_id()
                _write_json(self._cdir(cid) / "bundle.json", part.to_dict())
                _write_json(self._cdir(cid) / "build_log.json", blog)
                rec = {"id": cid, "status": "pending", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "variant": variant,
                       "part": part.meta.get("part", "bundle"), "bundle": bundle_summary(part), "instance": inst.summary(),
                       "source": {"session_id": session.id, "task_key": key, "spec": session.spec,
                                  "program_version": session.program.version, "run_dir": str(session.run_dir)},
                       "decision": None}
                self._save(rec)
                made.append({"id": cid, "bundle": rec["bundle"], "part": rec["part"]})
        status = "proposed" if made else "empty"
        return {"status": status, "candidates": made, "notes": notes,
                "reason": None if made else "the repair did not yield a reusable Skill or Operator"}

    # --------------------------------------------------------------------------------------------- gate
    def _next_version(self) -> str:
        nums = [int(m.group(1)) for v in self.store.versions() if (m := re.fullmatch(r"A(\d+)", v))]
        return f"A{max(nums, default=0) + 1}"

    def _config_digest(self) -> str:
        """What a stored report depends on besides the program and the tasks: the solver settings and the model behind Theta_0."""
        roles = {}
        for role in ("policy", "executor", "patch"):
            try:
                roles[role] = self.llm.role_config(role).model
            except Exception:                           # an unconfigured role has no model to pin
                roles[role] = ""
        llm = self.cfg.llm
        ident = {"solver": dataclasses.asdict(self.cfg.solver), "roles": roles, "backend": llm.backend,
                 "command": llm.command or os.environ.get("SCIENCECLAW_LLM_COMMAND", ""),
                 "pass_requires_acceptance": bool(self.cfg.evolution.pass_requires_acceptance)}
        return hashlib.sha256(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def _report_path(self, program: AgentProgram) -> Path:
        return self.root / "reports" / f"{program.fingerprint()}-{self._config_digest()}.json"

    def _cached_report(self, program: AgentProgram) -> ValReport | None:
        p = self._report_path(program)
        return ValReport.from_dict(_read_json(p)) if p.is_file() else None

    def _save_report(self, program: AgentProgram, rep: ValReport) -> None:
        _write_json(self._report_path(program), rep.to_dict())

    def _conflicts(self, bundle: Bundle, source: AgentProgram, head: AgentProgram) -> list[str]:
        """Components the bundle changes whose version moved since the session's program (the bundle is not based on head)."""
        out = []
        for kind, comps, src, cur in (("skill", bundle.skills, source.skills, head.skills),
                                      ("op", bundle.operators, source.operators, head.operators)):
            for c in comps:
                if c.id in cur and (c.id not in src or src[c.id].version != cur[c.id].version):
                    out.append(f"{kind}:{c.id}")
        return out

    def gate(self, cid: str) -> dict[str, Any]:
        """Validate a pending candidate on top of the active program; promote it iff the gate admits it and promotion is automatic.

        Without ``auto_promote`` an admitted candidate becomes ``ready`` and waits for :meth:`promote` (the user, outside the
        agent's tools): the agent that proposes candidates and registers validation tasks never promotes by itself.
        """
        rec = self.candidate(cid)
        if rec["status"] != "pending":
            raise ValueError(f"candidate {cid} is {rec['status']}; only pending candidates can be gated")
        spec = rec["source"]["spec"]
        key = rec["source"]["task_key"]
        evo_cfg = self.cfg.evolution
        t0 = time.monotonic()

        def settle(status: str, reason: str, **extra: Any) -> dict[str, Any]:
            rec["status"] = status
            rec["decision"] = {"reason": reason, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                               "wall_s": round(time.monotonic() - t0, 2), **extra}
            self._save(rec)
            return {"candidate": cid, "status": status, "reason": reason, **extra}

        val_eps, val_errors = self._val_episodes()
        if val_errors:
            return settle("pending", "blocked: validation tasks cannot be used as registered, and a smaller D_val would weaken "
                                     "the gate: " + "; ".join(f"{k}: {v}" for k, v in val_errors.items()), val_errors=val_errors)
        need = max(self.min_val_tasks, int(evo_cfg.min_improved_episodes))
        if len(val_eps) < need:
            return settle("pending", f"blocked: {len(val_eps)} validation task(s) registered, at least {need} are required; "
                                     "register representative tasks (not the source task) and gate again")
        if any(row["key"] == key for row in self.val_list()):
            return settle("pending", "blocked: the source task is itself a validation task, but D_src and D_val must be "
                                     "disjoint; remove it from the validation tasks and gate again")

        head = self.store.open()
        source_prog = self.store.load(rec["source"]["program_version"])
        if source_prog is None:
            return settle("pending", f"blocked: program {rec['source']['program_version']} the candidate was derived from "
                                     "is no longer in the store")
        bundle = Bundle.from_dict(_read_json(self._cdir(cid) / "bundle.json"))
        stale = self._conflicts(bundle, source_prog, head)
        if stale:
            return settle("pending", f"blocked: {stale} changed in program {head.version} since the candidate was derived "
                                     f"from {source_prog.version}; learn it again from a new session", stale=stale)
        cand, omega = head.apply(bundle, new_version=self._next_version())
        source_ep = build_live_episode(spec, task_id=f"src-{key[:8]}", input_roots=self.input_roots)

        from scienceclaw.agent.solver import Solver
        solver = Solver(self.cfg.solver, self.llm, evo_cfg)
        eff = dataclasses.replace(evo_cfg, **LIVE_GATE)
        work = self._cdir(cid) / "gate"
        try:
            ok_src, sres = source_replay_check(cand, omega, source_ep, solver, work / "source_replay",
                                               retries=int(evo_cfg.infra_retries), backoff_s=float(evo_cfg.infra_backoff_s))
            used, missing = use_check(omega, sres)
            r_src = {"R_src": bool(ok_src), "Pass": bool(source_pass(sres, bool(evo_cfg.pass_requires_acceptance))),
                     "Use": used, "use_missing": missing}
            if infra_error_of(sres):
                return settle("pending", f"blocked: the model was unavailable during source replay ({infra_error_of(sres)})",
                              **r_src)
            if not ok_src:
                return settle("rejected", "R_src failed: the candidate program does not re-solve the source task or does "
                                          "not use the new components", **r_src)

            gate = ValidationGate(eff, self.cfg.solver, solver, val_eps, work / "val", max_workers=2)
            inc = gate.evaluate(head, reuse_from=self._cached_report(head))
            self._save_report(head, inc)
            errored = sorted(e for e, v in inc.per_episode.items() if v.get("error"))
            if errored:
                return settle("pending", f"blocked: the incumbent could not be evaluated on {errored}", **r_src)
            crep = gate.evaluate(cand, reuse_from=inc)
            self._save_report(cand, crep)
            admitted, reasons = gate.admit(crep, inc)
        except LLMError as ex:
            return settle("pending", f"blocked: language model unavailable ({ex})")

        gate_info = {**r_src, "H_val": crep.h_val, "admitted": admitted, "validation_tasks": [e.id for e in val_eps],
                     "gate": LIVE_GATE, "min_improved_episodes": eff.min_improved_episodes, "reasons": reasons,
                     "incumbent": _brief(inc), "candidate_report": _brief(crep)}
        if not admitted:
            return settle("rejected", "the validation gate did not admit the candidate "
                                      f"(feasible={reasons.get('feasible')}, improved={reasons.get('improved')})", **gate_info)
        record = {"version": cand.version, "omega": omega, "base": head.version, "fingerprint": cand.fingerprint(), **gate_info}
        if not self.auto_promote:
            return settle("ready", f"admitted over {head.version}; promote it with `scienceclaw live promote {cid}`", **record)
        return self._promote(rec, cand, record, settle)

    def _promote(self, rec: dict[str, Any], cand: AgentProgram, record: dict[str, Any], settle: Callable[..., dict]) -> dict[str, Any]:
        try:
            self.store.commit(cand, {"event": "promote", "candidate": rec["id"], "parent": record["base"], "omega": record["omega"],
                                     "source": {"session_id": rec["source"]["session_id"], "task_key": rec["source"]["task_key"]},
                                     "gate": record}, expected_parent=record["base"])
        except StaleHead as ex:
            return settle("pending", f"blocked: {ex}; gate again", **record)
        return settle("promoted", f"admitted: program {record['base']} -> {cand.version}", **record)

    def promote(self, cid: str) -> dict[str, Any]:
        """Make an admitted (``ready``) candidate the active program. This is the user's decision, not the agent's."""
        rec = self.candidate(cid)
        if rec["status"] != "ready":
            raise ValueError(f"candidate {cid} is {rec['status']}; only candidates admitted by the gate ('ready') can be promoted")
        record = {k: v for k, v in (rec["decision"] or {}).items() if k not in ("reason", "time", "wall_s")}
        head = self.store.open()
        if head.version != record["base"]:
            raise StaleHead(f"the active program is {head.version}, the candidate was validated on {record['base']}; gate it again")
        cand, omega = head.apply(Bundle.from_dict(_read_json(self._cdir(cid) / "bundle.json")), new_version=record["version"])
        if cand.fingerprint() != record["fingerprint"]:
            raise ValueError("the program that would result is not the one the gate validated")
        t0 = time.monotonic()

        def settle(status: str, reason: str, **extra: Any) -> dict[str, Any]:
            rec["status"] = status
            rec["decision"] = {"reason": reason, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                               "wall_s": round(time.monotonic() - t0, 2), **extra}
            self._save(rec)
            return {"candidate": cid, "status": status, "reason": reason, **extra}

        return self._promote(rec, cand, record, settle)

    # ------------------------------------------------------------------------------------------- jobs
    def start_job(self, kind: str, fn: Callable[[], Any]) -> dict[str, Any]:
        """Run ``fn`` in the background (one evolution job at a time)."""
        with self._jobs_lock:
            running = [j for j in self.jobs.values() if j["state"] == "running"]
            if running:
                raise RuntimeError(f"evolution job {running[0]['id']} ({running[0]['kind']}) is still running; "
                                   "check it with evolve status")
            jid = uuid.uuid4().hex[:10]
            job = {"id": jid, "kind": kind, "state": "running", "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "result": None, "error": None}
            self.jobs[jid] = job
            self._done[jid] = threading.Event()
        threading.Thread(target=self._run_job, args=(job, fn), name=f"evolve-{jid}", daemon=True).start()
        return dict(job)

    def _run_job(self, job: dict[str, Any], fn: Callable[[], Any]) -> None:
        try:
            job["result"] = fn()
            job["state"] = "done"
        except Exception as ex:                          # reported through status, never lost
            log.exception("evolution job %s failed", job["id"])
            job["error"] = f"{type(ex).__name__}: {ex}"[:800]
            job["state"] = "error"
        finally:
            job["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._done[job["id"]].set()

    def job(self, jid: str | None = None, wait_s: float = 0.0) -> dict[str, Any]:
        """State of a job (the latest one if ``jid`` is omitted), optionally waiting up to ``wait_s`` seconds for it."""
        with self._jobs_lock:
            if jid is None:
                if not self.jobs:
                    raise KeyError("no evolution job has been started")
                jid = list(self.jobs)[-1]
            if jid not in self.jobs:
                raise KeyError(f"unknown job {jid!r}; known: {list(self.jobs)}")
        if wait_s > 0:
            self._done[jid].wait(min(float(wait_s), 300.0))
        return dict(self.jobs[jid])

    def run(self, session: Any, variant: str | None = None) -> dict[str, Any]:
        """Propose from a session and gate every candidate in turn."""
        proposal = self.propose(session, variant)
        decisions = [self.gate(c["id"]) for c in proposal["candidates"]]
        return {"proposal": proposal, "decisions": decisions}
