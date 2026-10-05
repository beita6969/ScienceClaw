"""Bundle construction B_i = (dS_i, dO_i) from one evolution instance (paper Section 5.2).

``build_bundle`` orchestrates Eq. 10 (split), Eq. 11 (Skill patch), Eq. 12 (Operator abstraction) and
boundary replay. Only boundary-replay-verified operators enter the bundle. In the linked (``full``)
variant the bundle couples the strategy change with its executable capability: every Skill of the bundle
gets a "Linked operators" section naming the operators of the same bundle, and every operator records the
Skills it is linked with in its provenance; both are applied and gated atomically.

Variant handling (see ``variants.py``): ``skill_only`` / ``operator_only`` build only one side,
``unlinked`` builds both without linking (``variants.bundles_for_variant`` splits them for independent
gating), ``workflow_only`` stores the whole G+ as one "workflow exemplar" Skill, ``frozen`` builds nothing.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from ..core.graph import WorkflowGraph
from ..core.operators import OperatorSpec
from ..core.program import AgentProgram, Bundle
from ..core.skills import Skill
from .attribution import EvolutionInstance
from .operator_abstraction import boundary_replay, operator_candidate_with_log
from .skill_patch import LINK_HEADER, short_hash, skill_candidates_with_log, slugify
from .split import split_details
from .variants import check_variant, wants_components

__all__ = ["Bundle", "build_bundle", "workflow_exemplar_skill", "link_skills_to_operators", "bundle_summary",
           "LINK_HEADER", "MAX_EXEMPLAR_CODE_CHARS"]

log = logging.getLogger(__name__)

MAX_EXEMPLAR_CODE_CHARS = 4000
_LINK_LINE = re.compile(r"^- (op:[^\s]+)")


def _created(round_idx: int | None, episode: Any) -> str:
    ep = getattr(episode, "id", "")
    return f"r{round_idx}:e{ep}" if round_idx is not None else f"e{ep}"


def workflow_exemplar_skill(inst: EvolutionInstance, episode: Any, created: str = "") -> Skill:
    """``workflow_only`` variant: the whole replay-verified G+ (compact JSON) as one retrievable Skill."""
    g = WorkflowGraph.from_dict(inst.e_plus.graph_dict or {})
    gd = g.to_dict(include_origin=False)
    for nd in gd["nodes"]:
        for key in ("code", "prompt"):
            if isinstance(nd.get(key), str) and len(nd[key]) > MAX_EXEMPLAR_CODE_CHARS:
                nd[key] = nd[key][:MAX_EXEMPLAR_CODE_CHARS] + "\n# ... (truncated)"
    compact = json.dumps(gd, separators=(",", ":"), sort_keys=True, default=str)
    disc = str(getattr(episode, "discipline", "") or "")
    ttype = str(getattr(episode, "task_type", "") or "")
    req = getattr(episode, "required_output", None)
    req_txt = req.render() if hasattr(req, "render") else str(req)
    title = f"Workflow exemplar: {ttype or 'task'} ({disc})".strip()
    body = "\n".join([
        f"Applicability: {ttype or 'tasks'} in {disc or 'this discipline'} whose deliverable is {req_txt}.",
        "Workflow graph (replay-verified passing workflow, JSON: nodes with kind/ref/code/prompt/config/ports, "
        "edges src.src_port -> dst.dst_port):",
        "```json", compact, "```",
    ])
    tags = [t for t in dict.fromkeys([disc, ttype, "workflow_exemplar", *(getattr(episode, "tags", None) or [])[:5]])
            if t]
    sid = f"workflow-{slugify(ttype or disc or 'task', 30)}-{short_hash(compact, 8)}"
    prov = {"source": "workflow_only", "episode": inst.episode_id, "k_plus": inst.k_plus, "k_minus": inst.k_minus,
            "graph_fp": getattr(inst.e_plus, "graph_fp", ""), "program_version": inst.program_version}
    return Skill(id=sid, version=1, title=title, body=body, tags=tags, provenance=prov, created=created)


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
                 variant: str, *, repeats: int = 1, round_idx: int | None = None) -> tuple[Bundle, dict]:
    """Build the bundle of one evolution instance for ``variant``; returns (bundle, JSON-safe log)."""
    check_variant(variant)
    created = _created(round_idx, episode)
    meta = {"source_episode": inst.episode_id, "k_minus": inst.k_minus, "k_plus": inst.k_plus,
            "delta_steps": list(inst.delta_steps), "variant": variant, "linked": variant == "full",
            "round": round_idx, "program_version": program.version}
    logd: dict[str, Any] = {"variant": variant, "instance": inst.summary(), "split": None, "skills": None,
                            "operators": [], "llm_usage": []}
    if variant == "frozen":
        return Bundle([], [], meta), logd
    if variant == "workflow_only":
        sk = workflow_exemplar_skill(inst, episode, created)
        logd["skills"] = {"formed": True, "workflow_exemplar": sk.id}
        return Bundle([sk], [], meta), logd

    want_skills, want_ops = wants_components(variant)
    split = split_details(inst, program)
    logd["split"] = split.summary()

    operators: list[OperatorSpec] = []
    if want_ops:
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

    skills: list[Skill] = []
    if want_skills:
        skills, sklog = skill_candidates_with_log(inst, split.control_edits, program, llm, episode, created=created,
                                                  control_steps=split.control_steps)
        logd["skills"] = sklog
        if sklog.get("llm"):
            logd["llm_usage"].append(sklog["llm"].get("usage", {}))

    for o in operators:
        o.provenance = {**o.provenance, "created": created}
    if variant == "full":
        link_skills_to_operators(skills, operators)
    meta["skill_ids"] = [s.id for s in skills]
    meta["operator_ids"] = [o.id for o in operators]
    return Bundle(skills, operators, meta), logd
