"""Skill candidates dS_i = Patch_Theta0(S_r[nu^S_i]; Pi_ctrl(delta_i), e-_i, e+_i)  (paper Eq. 11).

One call of the fixed foundation model (role ``patch``) receives

* the attributed existing Skills S_r[nu^S] (Skill ids in the nu of the steps of delta),
* the control edits of the reproduced repair (rendered one per line),
* a summary of the replay-verified failure e- (node errors, violated *visible* hard constraints, visible
  dev score) and of the replay-verified success e+,
* the public view of the episode (never hidden labels; the hidden evaluation is summarized as pass/fail),

and writes a SHORT reusable Skill with sections Applicability / Procedure / Pitfalls. With attributed
Skills the output revises them (same id; ``AgentProgram.apply`` bumps the version), otherwise exactly one
new Skill is created (id = slug(title) + short hash). No e- or no control edit -> no Skill candidate.

This module also hosts small helpers shared by the other evolution modules (slug, hashes, tolerant JSON
parsing of patch-model outputs, compact renderings of evidence).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

from ..core.graph import WorkflowGraph
from ..core.program import AgentProgram
from ..core.skills import Skill
from .attribution import EvolutionInstance

__all__ = ["make_skill_candidates", "skill_candidates_with_log", "build_patch_messages", "slugify",
           "short_hash", "parse_json_object", "call_patch_model", "render_action", "evidence_summary",
           "assemble_skill_body", "SKILL_SECTIONS", "MAX_SKILL_BODY_CHARS", "LINK_HEADER"]

log = logging.getLogger(__name__)

SKILL_SECTIONS = ("Applicability", "Procedure", "Pitfalls")
MAX_SKILL_BODY_CHARS = 2400
MAX_TITLE_CHARS = 120
MAX_TAGS = 8
MAX_TOTAL_TAGS = 16
_GRAPH_CHARS_PLUS = 6000
_GRAPH_CHARS_MINUS = 3500


# --------------------------------------------------------------------------------------------- helpers
def slugify(text: str, max_len: int = 40) -> str:
    """Lower-case ``[a-z0-9-]`` slug (``"x"`` if nothing is left)."""
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    s = s[:max_len].rstrip("-")
    return s or "x"


def short_hash(obj: Any, n: int = 8) -> str:
    blob = obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:n]


def _local_extract_json(text: str) -> dict | None:
    """Tolerant JSON object extraction (fallback when ``llm.client.extract_json`` is unavailable)."""
    if not text:
        return None
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    candidates = [fence.group(1)] if fence else []
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        candidates.append(t[i:j + 1])
    candidates.append(t)
    for c in candidates:
        for variant in (c, re.sub(r",\s*([}\]])", r"\1", c)):
            try:
                obj = json.loads(variant)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(obj, dict):
                return obj
    return None


def parse_json_object(text: str) -> dict | None:
    """Parse the JSON object of a patch-model response (``llm.client.extract_json`` when available)."""
    try:
        from ..llm.client import extract_json
    except ImportError:
        return _local_extract_json(text)
    obj = extract_json(text)
    return obj if isinstance(obj, dict) else _local_extract_json(text)


def call_patch_model(llm: Any, messages: list[dict], *, tag: str, max_tokens: int | None = None,
                     cache_salt: str = "") -> tuple[dict | None, dict]:
    """One ``llm.chat(role="patch")`` call returning (parsed JSON object or None, call log).

    Failures (transport errors, unparseable output) are logged and reported in the call log; they never
    propagate, because evolution must continue with the next instance.
    """
    info: dict[str, Any] = {"tag": tag, "ok": False, "usage": {}, "error": None, "cached": None}
    try:
        kwargs: dict[str, Any] = {"json_mode": True, "tag": tag}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if cache_salt:
            kwargs["cache_salt"] = cache_salt
        resp = llm.chat("patch", messages, **kwargs)
    except Exception as ex:  # transport/model failure: recorded, the candidate is simply not formed
        info["error"] = f"{type(ex).__name__}: {ex}"[:500]
        log.warning("patch model call %s failed: %s", tag, info["error"])
        return None, info
    info["usage"] = dict(getattr(resp, "usage", {}) or {})
    info["cached"] = getattr(resp, "cached", None)
    text = getattr(resp, "text", "") or ""
    obj = parse_json_object(text)
    if obj is None:
        info["error"] = "unparseable patch-model output"
        info["raw_head"] = text[:300]
        log.warning("patch model call %s returned unparseable output: %r", tag, text[:200])
        return None, info
    info["ok"] = True
    return obj, info


def _trunc(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n] + f"\n... ({len(text) - n} more chars truncated)"


def render_action(action: Any, step: int | None = None) -> str:
    """One-line (plus short code excerpt) rendering of an atomic edit for the patch prompt."""
    t = getattr(action, "type", "?")
    p = getattr(action, "payload", None) or {}
    head = f"[step {step}] " if step is not None and step >= 0 else ""
    try:
        if t == "add_node":
            nd = p.get("node", p)
            s = f"add_node {nd.get('id')} kind={nd.get('kind')}"
            if nd.get("ref"):
                s += f" ref={nd.get('ref')}"
            if nd.get("config"):
                s += f" config={json.dumps(nd.get('config'), default=str)[:300]}"
            if nd.get("code"):
                s += "\n      code: " + _trunc(str(nd["code"]), 400).replace("\n", "\n      ")
            if nd.get("prompt"):
                s += f"\n      prompt: {str(nd['prompt'])[:300]}"
        elif t == "remove_node":
            s = f"remove_node {p.get('id')}"
        elif t == "modify_node":
            patch = p.get("patch", {}) or {}
            s = f"modify_node {p.get('id')} fields={sorted(patch)} patch={json.dumps(patch, default=str)[:500]}"
        elif t in ("add_edge", "remove_edge"):
            ed = p.get("edge", p)
            s = f"{t} {ed.get('src')}.{ed.get('src_port')} -> {ed.get('dst')}.{ed.get('dst_port')}"
            if ed.get("conversion"):
                s += f" conversion={json.dumps(ed.get('conversion'), default=str)}"
        else:
            s = f"{t} {json.dumps(p, default=str)[:400]}"
    except AttributeError:
        s = f"{t} {str(p)[:400]}"
    uses = list(getattr(action, "uses", []) or [])
    return head + s + (f"   (uses: {', '.join(uses)})" if uses else "")


def _visible_constraint_names(episode: Any) -> set[str]:
    return {c.name for c in (getattr(episode, "constraints", None) or []) if getattr(c, "visible", True)}


def _record_view(rec: Any) -> dict:
    if isinstance(rec, dict):
        return {"status": rec.get("status"), "error": (rec.get("error") or None)}
    return {"status": getattr(rec, "status", None), "error": getattr(rec, "error", None)}


def evidence_summary(ev: Any, episode: Any, feedback: dict | None = None, *, graph_chars: int = 4000) -> dict:
    """Policy-safe summary of an evidence e: graph, node errors, visible constraint outcomes, dev score.

    Hidden information is reduced to a pass/fail bit ("hidden_evaluation") and the number of violated
    hidden hard constraints; hidden scores, labels and hidden constraint messages are never included.
    """
    visible = _visible_constraint_names(episode)
    evr = getattr(ev, "eval", None)
    h = dict(getattr(evr, "h", {}) or {})
    msgs = dict(getattr(evr, "h_msgs", {}) or {})
    trace = getattr(ev, "trace", None)
    errors = {}
    if trace is not None and hasattr(trace, "errors"):
        errors = {k: _trunc(str(v), 400) for k, v in trace.errors().items()}
    try:
        graph_text = WorkflowGraph.from_dict(getattr(ev, "graph_dict", {}) or {}).render_compact(max_code_chars=900)
    except (ValueError, KeyError, TypeError) as ex:
        graph_text = f"(graph could not be rendered: {type(ex).__name__}: {ex})"
    out: dict[str, Any] = {
        "step": getattr(ev, "step", None),
        "graph": _trunc(graph_text, graph_chars),
        "node_errors": errors,
        "completed": bool(getattr(evr, "completed", False)),
        "visible_constraints": {k: {"ok": bool(v), "msg": _trunc(str(msgs.get(k, "")), 300)}
                                for k, v in h.items() if k in visible},
        "hidden_constraints_violated": sum(1 for k, v in h.items() if k not in visible and not v),
        "reproducible": getattr(evr, "reproducible", None),
        "within_budget": getattr(evr, "within_budget", None),
        "hidden_evaluation": "PASS" if getattr(ev, "passed", False) else "FAIL",
    }
    fb = feedback or {}
    if fb.get("dev") is not None:
        out["visible_dev_score"] = fb.get("dev")
    if fb.get("integrity_violations"):
        out["integrity_violations"] = list(fb.get("integrity_violations"))[:10]
    if fb.get("action_error"):
        out["action_error"] = _trunc(str(fb.get("action_error")), 300)
    return out


_ENUM_PREFIX = re.compile(r"^[0-9]+[.)]\s*")


def assemble_skill_body(item: dict) -> str:
    """Body with sections Applicability / Procedure / Pitfalls from a structured patch-model item."""
    def _lines(v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [ln.strip(" -*\t") for ln in v.strip().splitlines() if ln.strip()]
        return [str(x).strip() for x in v if str(x).strip()]

    structured = any(k in item for k in ("applicability", "procedure", "pitfalls"))
    if not structured and isinstance(item.get("body"), str):
        return item["body"].strip()
    app = " ".join(_lines(item.get("applicability"))) or "(not stated)"
    proc = _lines(item.get("procedure"))
    pit = _lines(item.get("pitfalls"))
    parts = [f"Applicability: {app}", "Procedure:"]
    parts += [f"{i}. {_ENUM_PREFIX.sub('', s)}" for i, s in enumerate(proc, 1)] or ["1. (not stated)"]
    parts.append("Pitfalls:")
    parts += [f"- {s}" for s in pit] or ["- (none recorded)"]
    return "\n".join(parts)


_ABS_PATH = re.compile(r"(?<![\w.])/(?:[\w.-]+/){2,}[\w.-]*")


def _scrub(text: str, tokens: list[str]) -> tuple[str, list[str]]:
    """Remove instance-specific tokens (episode id, lineage item ids, absolute paths)."""
    found: list[str] = []
    for tok in sorted({t for t in tokens if t and len(t) >= 4}, key=len, reverse=True):
        if tok in text:
            found.append(tok)
            text = text.replace(tok, "<instance>")
    for m in set(_ABS_PATH.findall(text)):
        found.append(m)
        text = text.replace(m, "<path>")
    return text, found


def _instance_tokens(episode: Any) -> list[str]:
    toks = [str(getattr(episode, "id", ""))]
    lineage = getattr(episode, "lineage", None) or {}
    for key in ("item_ids", "items", "ids"):
        vals = lineage.get(key)
        if isinstance(vals, (list, tuple)):
            toks += [str(v) for v in vals[:500] if isinstance(v, (str, int))]
    return toks


# ------------------------------------------------------------------------------------------ prompts
_SYSTEM = """You maintain a library of reusable Skills for a scientific workflow agent that builds typed workflow graphs (tool, operator, code, llm and submit nodes connected by typed ports) through atomic edits.
A Skill is a SHORT, general instruction about decomposition, workflow construction or recovery that the agent reads before working on a new task.
You receive one reproduced repair: a replay-verified FAILING workflow, the control edits (topology, routing and configuration edits) that turned it into a replay-verified PASSING workflow, and the existing Skills the agent was following during the repair (if any).
Write the Skill so that the agent would apply this repair directly on future tasks of the same kind.
Rules:
- Generalize beyond this instance: no instance-specific numbers, node ids, file names, item ids, data values or answers. Refer to nodes by their role.
- Keep it short: at most about 200 words per Skill.
- Structure: "applicability" (when the Skill applies), "procedure" (ordered steps), "pitfalls" (failure modes seen in the failing workflow and how to avoid them).
- If existing Skills are given, revise them: keep their id in "revises", keep what is still correct and change only what the repair shows is missing or wrong. Otherwise create exactly one new Skill with "revises": null.
Output ONLY one JSON object:
{"skills": [{"revises": "<existing skill id or null>", "title": "<short title>", "applicability": "<text>", "procedure": ["<step>", ...], "pitfalls": ["<pitfall>", ...], "tags": ["<tag>", ...]}]}"""


def build_patch_messages(inst: EvolutionInstance, control_edits: list[Any], attributed: list[Skill],
                         episode: Any, control_steps: list[int] | None = None) -> list[dict]:
    """Messages of the Patch_Theta0 call (Eq. 11)."""
    view = dict(episode.public_view()) if hasattr(episode, "public_view") else {}
    view.pop("id", None)
    if control_steps and len(control_steps) == len(control_edits):
        steps: list[int | None] = list(control_steps)
    else:   # recover the step of each control edit from delta (same Action objects)
        by_obj = {id(a): k for a, k in zip(inst.delta, inst.delta_steps)}
        steps = [by_obj.get(id(a)) for a in control_edits]
    edits = "\n".join(f"{i}. {render_action(a, s)}" for i, (s, a) in enumerate(zip(steps, control_edits), 1))
    fail = evidence_summary(inst.e_minus, episode, inst.feedback_minus, graph_chars=_GRAPH_CHARS_MINUS) \
        if inst.e_minus is not None else None
    succ = evidence_summary(inst.e_plus, episode, inst.feedback_plus, graph_chars=_GRAPH_CHARS_PLUS)
    parts = [
        "## Task (public description)",
        json.dumps(view, indent=1, default=str)[:4000],
        "## Existing Skills the agent was following during the repair",
        "\n\n".join(f"id: {s.id}\n{s.render()}" for s in attributed) if attributed
        else "(none: create one new Skill)",
        "## Replay-verified FAILING workflow (before the repair)",
        json.dumps({k: v for k, v in (fail or {}).items() if k != "graph"}, indent=1, default=str),
        (fail or {}).get("graph", "(none)"),
        "## Control edits of the repair (in order)",
        edits or "(none)",
    ]
    if inst.rejected:
        rej = "\n".join(f"- [step {r.get('step')}] {json.dumps(r.get('action'), default=str)[:300]} -> rejected: "
                        f"{_trunc(str(r.get('error') or ''), 300)}" for r in inst.rejected[:8])
        parts += ["## Edits rejected by validation during the repair", rej]
    parts += [
        "## Replay-verified PASSING workflow (after the repair)",
        json.dumps({k: v for k, v in succ.items() if k != "graph"}, indent=1, default=str),
        succ["graph"],
        "Write the Skill JSON now.",
    ]
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


# ------------------------------------------------------------------------------------------- main
LINK_HEADER = "Linked operators:"


def _carry_links(old_body: str, new_body: str) -> str:
    """Keep the "Linked operators" section of a revised Skill if the patch model dropped it."""
    if LINK_HEADER not in old_body or LINK_HEADER in new_body:
        return new_body
    section = old_body[old_body.index(LINK_HEADER):].strip()
    return new_body.rstrip() + "\n" + section


def _attributed_skills(inst: EvolutionInstance, program: AgentProgram) -> list[Skill]:
    ids: list[str] = []
    for u in sorted(inst.uses_in_delta):
        if not str(u).startswith("skill:"):
            continue
        sid = str(u)[len("skill:"):].split("@v", 1)[0]
        if sid in program.skills and sid not in ids:
            ids.append(sid)
    return [program.skills[s] for s in ids]


def skill_candidates_with_log(inst: EvolutionInstance, control_edits: list[Any], program: AgentProgram,
                              llm: Any, episode: Any, *, created: str = "",
                              control_steps: list[int] | None = None) -> tuple[list[Skill], dict]:
    """``make_skill_candidates`` plus a JSON-safe log of what happened (for candidates.jsonl)."""
    logd: dict[str, Any] = {"formed": False, "reason": None, "attributed": [], "llm": None, "leaks": []}
    if inst.e_minus is None:
        logd["reason"] = "no replay-verified failure e- before e+ (Eq. 9: no Skill candidate)"
        return [], logd
    if not control_edits:
        logd["reason"] = "no control edits in delta"
        return [], logd
    attributed = _attributed_skills(inst, program)
    logd["attributed"] = [s.version_id for s in attributed]
    messages = build_patch_messages(inst, control_edits, attributed, episode, control_steps)
    obj, call = call_patch_model(llm, messages, tag="evo.skill_patch")
    logd["llm"] = call
    if obj is None:
        logd["reason"] = f"patch model failed: {call.get('error')}"
        return [], logd
    items = obj.get("skills")
    if items is None and ("title" in obj or "body" in obj):
        items = [obj]
    if not isinstance(items, list):
        logd["reason"] = "patch output has no 'skills' list"
        return [], logd
    items = [it for it in items if isinstance(it, dict)]
    tokens = _instance_tokens(episode)
    by_id = {s.id: s for s in attributed}
    unassigned = [s.id for s in attributed]
    base_tags = [t for t in (getattr(episode, "discipline", ""), getattr(episode, "task_type", "")) if t]
    prov_base = {"source": "Patch_Theta0", "episode": inst.episode_id, "k_minus": inst.k_minus,
                 "k_plus": inst.k_plus, "delta_steps": list(inst.delta_steps),
                 "program_version": program.version, "n_control_edits": len(control_edits)}
    out: list[Skill] = []
    for it in items:
        title = " ".join(str(it.get("title") or "").split())[:MAX_TITLE_CHARS]
        body = assemble_skill_body(it)
        if not title or not body.strip():
            logd.setdefault("dropped", []).append("item without title/body")
            continue
        target: str | None = None
        if attributed:
            if not unassigned:
                break
            target = str(it.get("revises") or "").removeprefix("skill:").split("@v", 1)[0]
            if target not in unassigned:
                target = unassigned[0]
            unassigned.remove(target)
            body = _carry_links(by_id[target].body, body)
        title, leaks_t = _scrub(title, tokens)
        body, leaks_b = _scrub(body, tokens)
        logd["leaks"] += leaks_t + leaks_b
        missing = [sec for sec in SKILL_SECTIONS if sec.lower() not in body.lower()]
        if missing:
            logd.setdefault("notes", []).append(f"skill body lacks sections {missing}")
        body = _trunc(body, MAX_SKILL_BODY_CHARS)
        raw_tags = it.get("tags") if isinstance(it.get("tags"), list) else []
        tags: list[str] = []
        for t in [*(str(x).strip()[:40] for x in raw_tags if str(x).strip())][:MAX_TAGS] + base_tags:
            if t not in tags:
                tags.append(t)
        if target is not None:
            old = by_id[target]
            prov = {**prov_base, "revised_from": old.version_id}
            # a revision keeps the applicability it already had (tags are a union, old tags first)
            merged = list(dict.fromkeys([*old.tags, *tags]))[:MAX_TOTAL_TAGS]
            out.append(Skill(id=old.id, version=old.version, title=title, body=body, tags=merged,
                             provenance=prov, created=created or old.created))
        else:
            sid = f"{slugify(title)}-{short_hash(title + chr(10) + body, 6)}"
            out.append(Skill(id=sid, version=1, title=title, body=body, tags=tags, provenance=prov_base,
                             created=created))
            break  # without attributed skills exactly one new Skill is created
    logd["formed"] = bool(out)
    logd["skills"] = [{"id": s.id, "title": s.title, "revision": s.id in by_id} for s in out]
    if not out:
        logd["reason"] = "patch output contained no usable skill"
    return out, logd


def make_skill_candidates(inst: EvolutionInstance, control_edits: list[Any], program: AgentProgram, llm: Any,
                          episode: Any) -> list[Skill]:
    """Eq. 11: Skill candidates (new Skill, or revisions of the attributed Skills); [] if none is formed."""
    skills, _ = skill_candidates_with_log(inst, control_edits, program, llm, episode)
    return skills
