"""Bundle construction B_i = (dS_i, dO_i) from one evolution instance (paper Section 5.2).

``build_bundle`` orchestrates Eq. 10 (split), Eq. 11 (Skill patch), Eq. 12 (Operator abstraction) and
boundary replay. Only boundary-replay-verified operators enter the bundle. The bundle couples the strategy change with its executable capability: every Skill of the bundle
gets a "Linked operators" section naming the operators of the same bundle, and every operator records the
Skills it is linked with in its provenance; both are applied and gated atomically.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ..core.operators import OperatorSpec
from ..core.program import AgentProgram, Bundle
from ..core.skills import Skill
from .attribution import EvolutionInstance
from .operator_abstraction import boundary_replay, operator_candidate_with_log
from .skill_patch import LINK_HEADER, skill_candidates_with_log
from .split import split_details

__all__ = ["Bundle", "build_bundle", "link_skills_to_operators", "bundle_summary", "LINK_HEADER"]

log = logging.getLogger(__name__)

_LINK_LINE = re.compile(r"^- (op:[^\s]+)")


def _created(round_idx: int | None, episode: Any) -> str:
    ep = getattr(episode, "id", "")
    return f"r{round_idx}:e{ep}" if round_idx is not None else f"e{ep}"


def link_skills_to_operators(skills: list[Skill], operators: list[OperatorSpec]) -> None:
    """Couple Skills and Operators of one bundle (in place).

    Each Skill body gets (or has merged) a "Linked operators:" section listing ``op:<id> - name: signature``
    of the bundle's operators; each operator records the linked Skill refs in ``provenance["linked_skills"]``.
    """
    if not skills or not operators:
        return
    new_lines = {o.ref: f"- {o.ref} - {o.name}: {o.signature()}" for o in operators}
    for s in skills:
        head, sep, tail = s.body.partition(LINK_HEADER)
        existing: dict[str, str] = {}
        rest: list[str] = []
        if sep:
            for ln in tail.splitlines():
                m = _LINK_LINE.match(ln.strip())
                if m:
                    existing[m.group(1)] = ln.strip()
                elif ln.strip():
                    rest.append(ln)
        merged = {**existing, **new_lines}
        body = head.rstrip()
        if rest:
            body += "\n" + "\n".join(rest)
        s.body = body + "\n" + LINK_HEADER + "\n" + "\n".join(merged[k] for k in sorted(merged))
    for o in operators:
        o.provenance = {**o.provenance, "linked_skills": sorted(s.ref for s in skills)}


def bundle_summary(bundle: Bundle) -> dict:
    return {"skills": [{"id": s.id, "title": s.title, "tags": s.tags} for s in bundle.skills],
            "operators": [{"id": o.id, "name": o.name, "signature": o.signature(),
                           "n_nodes": len(o.body.nodes), "n_pre": len(o.contract.pre), "n_post": len(o.contract.post)}
                          for o in bundle.operators],
            "meta": bundle.meta}


def build_bundle(inst: EvolutionInstance, program: AgentProgram, llm: Any, episode: Any, run_dir: str | Path,
                 *, repeats: int = 1, round_idx: int | None = None) -> tuple[Bundle, dict]:
    """Build the linked bundle of one evolution instance; returns (bundle, JSON-safe log)."""
    created = _created(round_idx, episode)
    meta = {"source_episode": inst.episode_id, "k_minus": inst.k_minus, "k_plus": inst.k_plus,
            "delta_steps": list(inst.delta_steps), "linked": True,
            "round": round_idx, "program_version": program.version}
    logd: dict[str, Any] = {"instance": inst.summary(), "split": None, "skills": None,
                            "operators": [], "llm_usage": []}
    split = split_details(inst, program)
    logd["split"] = split.summary()

    operators: list[OperatorSpec] = []
    for U in split.components:
        op, oplog = operator_candidate_with_log(inst, U, program, llm, episode)
        if oplog.get("llm"):
            logd["llm_usage"].append(oplog["llm"].get("usage", {}))
        entry: dict[str, Any] = {"candidate": oplog}
        if op is not None:
            ok, det = boundary_replay(op, inst, U, program, llm, episode, run_dir, repeats=repeats)
            entry["breplay"] = det
            entry["admitted_to_bundle"] = ok
            if ok:
                operators.append(op)
        else:
            entry["admitted_to_bundle"] = False
        logd["operators"].append(entry)

    skills, sklog = skill_candidates_with_log(inst, split.control_edits, program, llm, episode, created=created,
                                              control_steps=split.control_steps)
    logd["skills"] = sklog
    if sklog.get("llm"):
        logd["llm_usage"].append(sklog["llm"].get("usage", {}))

    for o in operators:
        o.provenance = {**o.provenance, "created": created}
    link_skills_to_operators(skills, operators)
    meta["skill_ids"] = [s.id for s in skills]
    meta["operator_ids"] = [o.id for o in operators]
    return Bundle(skills, operators, meta), logd
