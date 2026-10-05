"""Action model a_{t,k} for multi-turn atomic canvas edits (paper Eq. 6-7).

The policy emits exactly one JSON object per turn::

    {"thought": "<=3 sentences",
     "action": {"type": "add_node", "node": {...}}
             | {"type": "remove_node", "id": "n3"}
             | {"type": "modify_node", "id": "n3", "patch": {"code"|"code_edit"|"prompt"|"config"|"inputs"|"outputs"|"wire": ...}}
             | {"type": "add_edge", "edge": {"src", "src_port", "dst", "dst_port", "conversion"?}}
             | {"type": "remove_edge", "edge": {...}}
             | {"type": "finish"}
             | {"type": "batch", "actions": [<action>, ...]},      # single_turn orchestration only
     "uses": ["skill:<id>", "op:<id>"]}                             # nu_{t,k}: contributing Skills/Operators

``parse_action`` turns policy text into an :class:`Action` (or a precise error string);
``apply_action`` applies one atomic edit to a *copy* of the workflow graph, filling ports
automatically for tool / operator / submit nodes, and re-validates the whole graph (Eq. 5).
On any error the input graph is returned unchanged together with the error.

Edit classes (DESIGN.md section 5, paper Eq. 10):
  * control edits Pi_ctrl: add/remove edge, remove_node, modify_node touching only ``config``,
    add_node of kind tool/operator/submit, finish;
  * executable edits Pi_exec: add_node of kind code/llm, modify_node touching code/prompt/ports.

Patch semantics of ``modify_node``: ``code`` / ``prompt`` replace the payload; ``code_edit``
``{"find": s, "replace": r}`` replaces the one occurrence of ``s`` in the node's current code by ``r``
(an exact substring that must occur exactly once; not combinable with ``code``); ``config`` is a
shallow merge where a JSON ``null`` value deletes the key; ``inputs`` / ``outputs`` replace the
declared port dictionary of a code/llm node (ports of tool/operator/submit nodes are fixed by
their specification); ``wire`` replaces the incoming edges of the listed input ports. Tool and
submit nodes take configuration; operator nodes take none (an operator is applied exactly as its
specification says, DESIGN.md section 3.2).
"""
from __future__ import annotations

import ast
import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .graph import NODE_KINDS, SUBMIT_PORT, Edge, Node, WorkflowGraph
from .schema import PortSchema

ACTION_TYPES = ("add_node", "remove_node", "modify_node", "add_edge", "remove_edge", "finish", "batch")
NODE_ID_RE = re.compile(r"[A-Za-z0-9_]{1,40}")
PORT_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
PATCH_KEYS = ("code", "code_edit", "prompt", "config", "inputs", "outputs", "wire")
NODE_FIELDS = ("id", "kind", "ref", "code", "prompt", "config", "inputs", "outputs")
LLM_PARSE_MODES = ("text", "json", "number", "choice")
LLM_ITEMS_PORT = "items"
LLM_OUTPUT_PORT = "outputs"
USES_PREFIXES = ("skill:", "op:")
CONTROL_NODE_KINDS = ("tool", "operator", "submit")
EXEC_NODE_KINDS = ("code", "llm")
EXEC_PATCH_KEYS = ("code", "code_edit", "prompt", "inputs", "outputs")

_DECODER = json.JSONDecoder(strict=False)   # strict=False: tolerate raw newlines inside code strings
_FENCE_RE = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\n?(.*?)```", re.S)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_OP_VERSION_RE = re.compile(r"@v\d+$")


# --------------------------------------------------------------------------------------- Action
@dataclass
class Action:
    """One atomic edit a_{t,k} with its attribution nu_{t,k} (``uses``)."""
    type: str
    payload: dict = field(default_factory=dict)
    uses: list[str] = field(default_factory=list)
    thought: str = ""
    raw: str = ""

    def to_dict(self) -> dict:
        return {"type": self.type, "payload": copy.deepcopy(self.payload), "uses": list(self.uses),
                "thought": self.thought, "raw": self.raw}

    @classmethod
    def from_dict(cls, d: dict) -> "Action":
        """Accepts ``to_dict`` output, a policy container ``{"action": {...}, "uses": ...}`` or a bare
        policy action ``{"type": ..., <payload fields>}``."""
        if "payload" in d and "type" in d:
            return cls(str(d["type"]), dict(d.get("payload") or {}), list(d.get("uses") or []),
                       str(d.get("thought") or ""), str(d.get("raw") or ""))
        container = d if isinstance(d.get("action"), dict) else {"action": d}
        act = container["action"]
        payload = {k: v for k, v in act.items() if k != "type"}
        return cls(str(act.get("type", "")), payload, _clean_uses(container.get("uses")),
                   str(container.get("thought") or ""), str(d.get("raw") or ""))

    def action_json(self) -> dict:
        """The action in the policy's JSON format (``{"type": ..., **payload}``)."""
        return {"type": self.type, **copy.deepcopy(self.payload)}

    def describe(self) -> str:
        """One-line human-readable description used in histories and feedback."""
        p = self.payload
        if self.type == "add_node":
            n = p.get("node", {})
            ref = f" ref={n.get('ref')}" if n.get("ref") else ""
            return f"add_node {n.get('id')} (kind={n.get('kind')}{ref})"
        if self.type in ("remove_node",):
            return f"remove_node {p.get('id')}"
        if self.type == "modify_node":
            return f"modify_node {p.get('id')} patch={sorted((p.get('patch') or {}).keys())}"
        if self.type in ("add_edge", "remove_edge"):
            e = p.get("edge", {})
            conv = " [conversion]" if e.get("conversion") else ""
            src = e.get("src", "?") + (f".{e['src_port']}" if e.get("src_port") else "")
            dst = e.get("dst", "?") + (f".{e['dst_port']}" if e.get("dst_port") else "")
            return f"{self.type} {src} -> {dst}{conv}"
        if self.type == "batch":
            return f"batch of {len(p.get('actions', []))} actions"
        return self.type


def _clean_uses(uses: Any) -> list[str]:
    """Keep attribution ids of program components (``skill:`` / ``op:``), in order, without duplicates."""
    if uses is None:
        return []
    if isinstance(uses, str):
        uses = [uses]
    out: list[str] = []
    if not isinstance(uses, (list, tuple)):
        return out
    for u in uses:
        if isinstance(u, str) and u.strip().startswith(USES_PREFIXES) and u.strip() not in out:
            out.append(u.strip())
    return out


# ----------------------------------------------------------------------------------- JSON parse
def _loads_dict(s: str) -> dict | None:
    s = s.strip()
    if not s:
        return None
    for cand in (s, _TRAILING_COMMA_RE.sub(r"\1", s)):
        try:
            obj = _DECODER.decode(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def extract_json_object(text: str) -> dict | None:
    """Tolerant extraction of one JSON object from model text.

    Tries, in order: the whole text, fenced code blocks, the span between the first ``{`` and the
    last ``}``, and finally every embedded object (``raw_decode`` from each ``{``), preferring objects
    that look like an action container. Raw newlines inside strings and trailing commas are tolerated.
    """
    if not isinstance(text, str):
        return None
    for cand in [text] + [m.group(1) for m in _FENCE_RE.finditer(text)]:
        obj = _loads_dict(cand)
        if obj is not None:
            return obj
    i, j = text.find("{"), text.rfind("}")
    if 0 <= i < j:
        obj = _loads_dict(text[i:j + 1])
        if obj is not None:
            return obj
    first: dict | None = None
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = _DECODER.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            if "action" in obj or "type" in obj:
                return obj
            first = first if first is not None else obj
    return first


# ------------------------------------------------------------------------------ validation
def _port_dict_error(ports: Any, where: str) -> str | None:
    if not isinstance(ports, dict):
        return f"{where} must be an object mapping port names to schemas, got {type(ports).__name__}"
    for name, sch in ports.items():
        if not isinstance(name, str) or not PORT_NAME_RE.fullmatch(name):
            return f"{where}: invalid port name {name!r} (use [A-Za-z_][A-Za-z0-9_]*, at most 64 chars)"
        if not isinstance(sch, (dict, str, PortSchema)):
            return f"{where}[{name!r}] must be a schema object like {{\"type\": \"array\", \"shape\": [\"n\"], \"unit\": \"K\"}}"
        try:
            PortSchema.from_dict(sch)
        except (ValueError, TypeError) as ex:
            return f"{where}[{name!r}]: {ex}"
    return None


def _conversion_error(conv: Any) -> str | None:
    if conv is None:
        return None
    if not isinstance(conv, dict):
        return "edge.conversion must be an object {\"from\": u1, \"to\": u2, \"factor\": a, \"offset\": b}"
    missing = [k for k in ("from", "to", "factor") if k not in conv]
    if missing:
        return f"edge.conversion is missing {missing} (needs from, to, factor and optional offset)"
    for k in ("factor", "offset"):
        if k in conv and (isinstance(conv[k], bool) or not isinstance(conv[k], (int, float))):
            return f"edge.conversion.{k} must be a number, got {conv[k]!r}"
    return None


def _normalize_wire(wire: Any, where: str) -> tuple[dict | None, str | None]:
    """Validate a wire map {input_port: "src_node.src_port" | {"from": "src_node.src_port", "conversion": {...}}}.

    Wiring a node's inputs together with adding/modifying it is one atomic edit of the node and its
    dependencies (paper Sec. 5.1: a_{t,k} adds, removes or modifies a node, dependency or configuration).
    """
    if wire is None:
        return {}, None
    if not isinstance(wire, dict):
        return None, f"{where}: \"wire\" must be an object {{input_port: \"src_node.src_port\"}}"
    out: dict[str, dict] = {}
    for port, spec in wire.items():
        conv = None
        if isinstance(spec, dict):
            conv = spec.get("conversion")
            if "src" in spec and "from" not in spec:      # already-normalized form (idempotent)
                spec = f"{spec['src']}.{spec.get('src_port') or ''}"
            else:
                spec = spec.get("from")
        if not isinstance(spec, str) or "." not in spec:
            return None, f"{where}: wire[{port!r}] must be \"src_node.src_port\" (got {spec!r})"
        src, sport = spec.split(".", 1)
        if conv is not None:
            err = _conversion_error(conv)
            if err:
                return None, f"{where}: wire[{port!r}]: {err}"
        out[str(port)] = {"src": src, "src_port": sport, **({"conversion": conv} if conv else {})}
    return out, None


def _code_edit_error(edit: Any) -> str | None:
    """Shape check of ``patch.code_edit`` = {"find": <exact substring>, "replace": <text>}."""
    hint = "patch.code_edit must be an object {\"find\": <exact text in the current code>, \"replace\": <new text>}"
    if not isinstance(edit, dict):
        return hint
    if set(edit) != {"find", "replace"}:
        return f"{hint}; got keys {sorted(map(str, edit))}"
    if not isinstance(edit["find"], str) or not edit["find"]:
        return "patch.code_edit.find must be a non-empty string"
    if not isinstance(edit["replace"], str):
        return "patch.code_edit.replace must be a string (empty deletes the found text)"
    return None


def apply_code_edit(code: str, edit: dict) -> tuple[str | None, str | None]:
    """Apply ``{"find", "replace"}`` to ``code``: returns ``(new_code, None)`` or ``(None, error)``.

    ``find`` is an exact substring (whitespace and indentation included) that must occur exactly once, so an
    edit cannot silently change the wrong place. The result must still be valid code defining ``run``.
    """
    err = _code_edit_error(edit)
    if err:
        return None, err
    find, repl = edit["find"], edit["replace"]
    n = code.count(find)
    if n == 0:
        return None, ("code_edit.find does not occur in the node's current code (exact match, including "
                      "whitespace and indentation); send the full \"code\" instead if unsure")
    if n > 1:
        return None, (f"code_edit.find occurs {n} times in the node's current code; extend it with neighbouring "
                      "text so that it matches exactly once")
    new = repair_code_tail(code.replace(find, repl, 1))
    err = _check_code(new)
    if err:
        return None, f"code_edit leaves the code invalid: {err}"
    return new, None


def _normalize_payload(atype: str, act: dict) -> tuple[dict | None, str | None]:
    """Validate the payload fields of one action object (policy format) and normalize it."""
    payload = {k: v for k, v in act.items() if k != "type"}
    if atype == "add_node":
        node = payload.get("node")
        if node is None and "id" in payload and "kind" in payload:
            node = {k: v for k, v in payload.items() if k in NODE_FIELDS}
        if not isinstance(node, dict):
            return None, "add_node requires \"node\": {\"id\", \"kind\", ...}"
        node = {k: v for k, v in node.items() if k in NODE_FIELDS}
        nid, kind = node.get("id"), node.get("kind")
        if not isinstance(nid, str) or not NODE_ID_RE.fullmatch(nid):
            return None, f"add_node: node.id must match [A-Za-z0-9_]{{1,40}}, got {nid!r}"
        if kind not in NODE_KINDS:
            return None, f"add_node: node.kind must be one of {list(NODE_KINDS)}, got {kind!r}"
        if kind in ("tool", "operator") and (not isinstance(node.get("ref"), str) or not node["ref"].strip()):
            what = "tool name" if kind == "tool" else "operator id (\"op:<id>\")"
            return None, f"add_node: a {kind} node requires \"ref\" = the {what}"
        if kind == "code":
            if not isinstance(node.get("code"), str) or not node["code"].strip():
                return None, "add_node: a code node requires \"code\" defining run(inputs, config) -> dict"
            if not isinstance(node.get("outputs"), dict) or not node["outputs"]:
                return None, "add_node: a code node requires non-empty \"outputs\" (port name -> schema)"
        if kind == "llm" and (not isinstance(node.get("prompt"), str) or not node["prompt"].strip()):
            return None, "add_node: an llm node requires a non-empty \"prompt\" template"
        if node.get("config") is None:
            node["config"] = {}
        if not isinstance(node["config"], dict):
            return None, "add_node: node.config must be an object"
        if kind in EXEC_NODE_KINDS:
            for key in ("inputs", "outputs"):
                if key in node and node[key] is not None:
                    err = _port_dict_error(node[key], f"add_node: node.{key}")
                    if err:
                        return None, err
        else:   # ports of tool / operator / submit nodes come from their specification
            node.pop("inputs", None)
            node.pop("outputs", None)
        wire, err = _normalize_wire(payload.get("wire", (payload.get("node") or {}).get("wire")), "add_node")
        if err:
            return None, err
        return ({"node": node, "wire": wire} if wire else {"node": node}), None
    if atype == "remove_node":
        nid = payload.get("id", payload.get("node_id"))
        if not isinstance(nid, str) or not nid:
            return None, "remove_node requires \"id\": <node id>"
        return {"id": nid}, None
    if atype == "modify_node":
        nid, patch = payload.get("id", payload.get("node_id")), payload.get("patch")
        if not isinstance(nid, str) or not nid:
            return None, "modify_node requires \"id\": <node id>"
        if not isinstance(patch, dict) or not patch:
            return None, f"modify_node requires a non-empty \"patch\" object with keys from {list(PATCH_KEYS)}"
        bad = sorted(set(patch) - set(PATCH_KEYS))
        if bad:
            return None, f"modify_node: unsupported patch key(s) {bad}; allowed: {list(PATCH_KEYS)}"
        for key in ("code", "prompt"):
            if key in patch and (not isinstance(patch[key], str) or not patch[key].strip()):
                return None, f"modify_node: patch.{key} must be a non-empty string"
        if "code_edit" in patch:
            err = _code_edit_error(patch["code_edit"])
            if err:
                return None, f"modify_node: {err}"
            if "code" in patch:
                return None, "modify_node: patch.code and patch.code_edit cannot be combined; send the full code or the edit"
        if "config" in patch and not isinstance(patch["config"], dict):
            return None, "modify_node: patch.config must be an object (null values delete keys)"
        for key in ("inputs", "outputs"):
            if key in patch:
                err = _port_dict_error(patch[key], f"modify_node: patch.{key}")
                if err:
                    return None, err
        if "wire" in patch:
            wire, err = _normalize_wire(patch["wire"], "modify_node: patch")
            if err:
                return None, err
            patch = {**patch, "wire": wire}
        return {"id": nid, "patch": patch}, None
    if atype in ("add_edge", "remove_edge"):
        edge = payload.get("edge")
        if edge is None and "src" in payload and "dst" in payload:
            edge = {k: payload[k] for k in ("src", "src_port", "dst", "dst_port", "conversion") if k in payload}
        if not isinstance(edge, dict):
            return None, f"{atype} requires \"edge\": {{\"src\", \"src_port\", \"dst\", \"dst_port\"}}"
        for k in ("src", "dst"):
            if not isinstance(edge.get(k), str) or not edge[k]:
                return None, f"{atype}: edge.{k} must be a node id"
        for k in ("src_port", "dst_port"):
            if k in edge and edge[k] is not None and not isinstance(edge[k], str):
                return None, f"{atype}: edge.{k} must be a port name"
        edge = {k: v for k, v in edge.items() if k in ("src", "src_port", "dst", "dst_port", "conversion") and v is not None}
        if atype == "add_edge":
            err = _conversion_error(edge.get("conversion"))
            if err:
                return None, f"add_edge: {err}"
        return {"edge": edge}, None
    if atype == "finish":
        return {}, None
    if atype == "batch":
        subs = payload.get("actions")
        if not isinstance(subs, list) or not subs:
            return None, "batch requires a non-empty \"actions\" list"
        norm: list[dict] = []
        for i, sub in enumerate(subs):
            if isinstance(sub, dict) and isinstance(sub.get("action"), dict):
                sub = sub["action"]
            if not isinstance(sub, dict):
                return None, f"batch action #{i}: must be an object"
            st = sub.get("type")
            if st not in ACTION_TYPES or st == "batch":
                return None, f"batch action #{i}: type must be one of {[t for t in ACTION_TYPES if t != 'batch']}, got {st!r}"
            p, err = _normalize_payload(st, sub)
            if err:
                return None, f"batch action #{i} ({st}): {err}"
            norm.append({"type": st, **p})
        return {"actions": norm}, None
    return None, f"unknown action type {atype!r}; expected one of {list(ACTION_TYPES)}"


def parse_action(text: str, *, allow_batch: bool = True) -> tuple[Action | None, str | None]:
    """Parse policy output into an :class:`Action`.

    Returns ``(action, None)`` on success, or ``(None, error)`` with a precise, policy-facing message.
    Code fences, leading prose, raw newlines inside strings and trailing commas are tolerated.
    """
    if not isinstance(text, str) or not text.strip():
        return None, "empty response: reply with one JSON object {\"thought\", \"action\", \"uses\"}"
    obj = extract_json_object(text)
    if obj is None:
        return None, "no JSON object found: reply with exactly one JSON object {\"thought\", \"action\": {...}, \"uses\": [...]}"
    if "action" in obj:
        act = obj["action"]
        if not isinstance(act, dict):
            return None, f"\"action\" must be an object with a \"type\" field, got {type(act).__name__}"
    elif "type" in obj:
        act = obj
    else:
        return None, f"JSON object has no \"action\" field (keys: {sorted(obj)[:10]})"
    atype = act.get("type")
    if atype not in ACTION_TYPES:
        return None, f"action.type must be one of {list(ACTION_TYPES)}, got {atype!r}"
    if atype == "batch" and not allow_batch:
        return None, "action type \"batch\" is not allowed here: emit exactly one atomic edit per turn"
    payload, err = _normalize_payload(atype, act)
    if err:
        return None, err
    # thought / uses live on the container (for a bare action object the container is the action itself)
    return Action(atype, payload or {}, _clean_uses(obj.get("uses")), str(obj.get("thought") or "")[:2000], text), None


# ----------------------------------------------------------------------------------- apply
_ENVELOPE_TAIL_RE = re.compile(r"(?:\s*</[A-Za-z_][\w:.-]*>)+\s*$")


def repair_code_tail(code: str) -> str:
    """Strip reply-envelope debris that leaked into the end of a ``code`` string (``}}}}</invoke></div>``).

    Only applied to code that does not parse; the repaired text is used only if it parses (and only closing
    brackets/braces and closing tags are removed from the very end, so valid code is never altered).
    """
    if not isinstance(code, str):
        return code
    try:
        ast.parse(code)
        return code
    except SyntaxError:
        pass
    cand = _ENVELOPE_TAIL_RE.sub("", code).rstrip()
    for _ in range(8):
        try:
            ast.parse(cand)
            return cand
        except SyntaxError:
            if not cand or cand[-1] not in "}])\"'":
                break
            cand = cand[:-1].rstrip()
    return code


def _check_code(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError as ex:
        return f"code does not parse: line {ex.lineno}: {ex.msg}"
    for st in tree.body:
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)) and st.name == "run":
            if isinstance(st, ast.AsyncFunctionDef):
                return "run must be a regular (non-async) function run(inputs, config) -> dict"
            a = st.args
            n_pos = len(a.posonlyargs) + len(a.args)
            if n_pos >= 2 or a.vararg is not None:
                return None
            return "run must accept two positional arguments: run(inputs, config)"
    return "code must define a top-level function run(inputs, config) returning a dict of output ports"


def _schemas(ports: dict | None) -> dict[str, PortSchema]:
    return {k: PortSchema.from_dict(v) for k, v in (ports or {}).items()}


def _llm_node_error(node: Node) -> str | None:
    parse = node.config.get("parse", "text")
    if parse not in LLM_PARSE_MODES:
        return f"llm node config.parse must be one of {list(LLM_PARSE_MODES)}, got {parse!r}"
    if parse == "choice":
        ch = node.config.get("choices")
        if not isinstance(ch, list) or not ch or not all(isinstance(c, str) and c for c in ch):
            return "llm node with parse=\"choice\" needs config.choices = a non-empty list of strings"
    if LLM_ITEMS_PORT not in node.inputs:
        return f"llm node must declare an input port {LLM_ITEMS_PORT!r} (the list mapped over)"
    if set(node.outputs) != {LLM_OUTPUT_PORT}:
        return f"llm node has exactly one output port {LLM_OUTPUT_PORT!r} (list of parsed answers)"
    mt = node.config.get("max_tokens")
    if mt is not None and (isinstance(mt, bool) or not isinstance(mt, int) or mt <= 0):
        return "llm node config.max_tokens must be a positive integer"
    return None


def _operator_id(ref: str) -> str:
    ref = ref.strip()
    if ref.startswith("op:"):
        ref = ref[3:]
    return _OP_VERSION_RE.sub("", ref)


def _add_node(g: WorkflowGraph, nd: dict, action: Action, episode: Any, program: Any, step: int | None) -> str | None:
    nid, kind = nd["id"], nd["kind"]
    if nid in g.nodes:
        return f"node id {nid!r} already exists; choose a new id or use modify_node"
    cfg = copy.deepcopy(nd.get("config") or {})
    origin: dict[str, Any] = {"step": step, "uses": list(action.uses), "generated": kind in EXEC_NODE_KINDS,
                              "from_operator": None}
    if kind == "tool":
        name = nd["ref"].strip()
        name = name[5:] if name.startswith("tool:") else name
        spec = episode.tool(name) if episode is not None else None
        if spec is None:
            avail = [t.name for t in getattr(episode, "tools", [])] if episode is not None else []
            return f"unknown tool {name!r}; available tools: {avail}"
        node = Node(nid, "tool", ref=name, config=cfg, inputs=dict(spec.inputs), outputs=dict(spec.outputs), origin=origin)
    elif kind == "operator":
        oid = _operator_id(nd["ref"])
        ops = getattr(program, "operators", {}) if program is not None else {}
        op = ops.get(oid)
        if op is None:
            return f"unknown operator {nd['ref']!r}; available operators: {sorted('op:' + k for k in ops)[:40]}"
        if cfg:
            return (f"operator nodes take no config (op:{oid} is applied exactly as specified); got "
                    f"{sorted(cfg)}; add a code node or a tool node for configurable steps")
        origin["op_version"] = op.version_id
        node = Node(nid, "operator", ref=f"op:{oid}", config=cfg, inputs=dict(op.inputs), outputs=dict(op.outputs),
                    origin=origin)
    elif kind == "submit":
        existing = g.submit_node()
        if existing is not None:
            return f"a submit node already exists ({existing!r}); at most one submit node is allowed"
        req = episode.required_output if episode is not None else PortSchema()
        node = Node(nid, "submit", config=cfg, inputs={SUBMIT_PORT: req}, outputs={}, origin=origin)
    elif kind == "code":
        nd["code"] = repair_code_tail(nd["code"])
        err = _check_code(nd["code"])
        if err:
            return err
        node = Node(nid, "code", code=nd["code"], config=cfg, inputs=_schemas(nd.get("inputs")),
                    outputs=_schemas(nd.get("outputs")), origin=origin)
    else:  # llm
        ins = _schemas(nd.get("inputs")) if nd.get("inputs") else {LLM_ITEMS_PORT: PortSchema(type="list")}
        outs = _schemas(nd.get("outputs")) if nd.get("outputs") else {LLM_OUTPUT_PORT: PortSchema(type="list")}
        node = Node(nid, "llm", prompt=nd["prompt"], config=cfg, inputs=ins, outputs=outs, origin=origin)
        err = _llm_node_error(node)
        if err:
            return err
    g.nodes[nid] = node
    return None


def _modify_node(g: WorkflowGraph, nid: str, patch: dict, action: Action, step: int | None) -> str | None:
    node = g.nodes.get(nid)
    if node is None:
        return f"unknown node {nid!r}; existing nodes: {sorted(g.nodes)}"
    if "code" in patch:
        if node.kind != "code":
            return f"patch.code applies only to code nodes; {nid!r} is a {node.kind} node"
        patch["code"] = repair_code_tail(patch["code"])
        err = _check_code(patch["code"])
        if err:
            return err
        node.code = patch["code"]
    if "code_edit" in patch:
        if node.kind != "code":
            return f"patch.code_edit applies only to code nodes; {nid!r} is a {node.kind} node"
        new, err = apply_code_edit(node.code, patch["code_edit"])
        if err:
            return err
        node.code = new
    if "prompt" in patch:
        if node.kind != "llm":
            return f"patch.prompt applies only to llm nodes; {nid!r} is a {node.kind} node"
        node.prompt = patch["prompt"]
    for key in ("inputs", "outputs"):
        if key in patch:
            if node.kind not in EXEC_NODE_KINDS:
                return f"ports of {node.kind} node {nid!r} are fixed by its specification; only config can be patched"
            setattr(node, key, _schemas(patch[key]))
            # A node that stops declaring an input port stops depending on it: its own edge into that port goes
            # away (otherwise the edit fails validation on the dangling edge for ever). Removing an *output* port
            # that other nodes consume stays an error: those consumers must be rewired explicitly.
            if key == "inputs":
                g.edges = [e for e in g.edges if not (e.dst == nid and e.dst_port not in node.inputs)]
    if "config" in patch:
        merged = dict(node.config)
        for k, v in patch["config"].items():
            if v is None:
                merged.pop(k, None)
            else:
                merged[k] = copy.deepcopy(v)
        if node.kind == "operator" and any(v is not None for v in patch["config"].values()):
            return (f"operator nodes take no config ({node.ref} is applied exactly as specified); got "
                    f"{sorted(k for k, v in patch['config'].items() if v is not None)}")
        node.config = merged
    if node.kind == "code" and not node.outputs:
        return "a code node must declare at least one output port"
    if node.kind == "llm":
        err = _llm_node_error(node)
        if err:
            return err
    origin = dict(node.origin)
    origin.setdefault("modified_steps", [])
    origin["modified_steps"] = list(origin["modified_steps"]) + ([step] if step is not None else [])
    origin["uses"] = list(dict.fromkeys(list(origin.get("uses", [])) + list(action.uses)))
    node.origin = origin
    return None


def _resolve_port(ports: dict, given: str | None, node_id: str, side: str) -> tuple[str | None, str | None]:
    if given:
        if given not in ports:
            return None, f"node {node_id!r} has no {side} port {given!r} (has {list(ports)})"
        return given, None
    if len(ports) == 1:
        return next(iter(ports)), None
    return None, f"specify the {side} port of node {node_id!r}: one of {list(ports)}"


def _add_edge(g: WorkflowGraph, e: dict) -> str | None:
    src, dst = e["src"], e["dst"]
    for nid, role in ((src, "source"), (dst, "destination")):
        if nid not in g.nodes:
            return f"unknown {role} node {nid!r}; existing nodes: {sorted(g.nodes)}"
    if src == dst:
        return "an edge cannot connect a node to itself"
    sp, err = _resolve_port(g.nodes[src].outputs, e.get("src_port"), src, "output")
    if err:
        return err
    dp, err = _resolve_port(g.nodes[dst].inputs, e.get("dst_port"), dst, "input")
    if err:
        return err
    for old in g.in_edges(dst):
        if old.dst_port == dp:
            if old.src == src and old.src_port == sp:
                return f"edge {old.render()} already exists"
            return f"input port {dst}.{dp} is already wired from {old.src}.{old.src_port}; remove that edge first"
    g.edges.append(Edge.make(src, sp, dst, dp, e.get("conversion")))
    return None


def _remove_edge(g: WorkflowGraph, e: dict) -> str | None:
    matches = [x for x in g.edges if x.src == e["src"] and x.dst == e["dst"]
               and (not e.get("src_port") or x.src_port == e["src_port"])
               and (not e.get("dst_port") or x.dst_port == e["dst_port"])]
    if not matches:
        between = [x.render() for x in g.edges if x.src == e["src"] and x.dst == e["dst"]]
        return f"no such edge {e['src']}.{e.get('src_port', '?')} -> {e['dst']}.{e.get('dst_port', '?')}; edges between them: {between}"
    if len(matches) > 1:
        return f"ambiguous edge; specify ports. candidates: {[x.render() for x in matches]}"
    g.edges.remove(matches[0])
    return None


def _apply_wire(g: WorkflowGraph, nid: str, wire: dict, replace: bool) -> str | None:
    for port, spec in wire.items():
        if replace:
            g.edges = [e for e in g.edges if not (e.dst == nid and e.dst_port == port)]
        err = _add_edge(g, {"src": spec["src"], "src_port": spec["src_port"], "dst": nid, "dst_port": port,
                            **({"conversion": spec["conversion"]} if spec.get("conversion") else {})})
        if err:
            return f"wire[{port!r}]: {err}"
    return None


def _apply_inplace(g: WorkflowGraph, atype: str, payload: dict, action: Action, episode: Any, program: Any,
                   step: int | None) -> str | None:
    if atype == "add_node":
        err = _add_node(g, payload["node"], action, episode, program, step)
        if err:
            return err
        return _apply_wire(g, payload["node"]["id"], payload.get("wire") or {}, replace=False)
    if atype == "remove_node":
        nid = payload["id"]
        if nid not in g.nodes:
            return f"unknown node {nid!r}; existing nodes: {sorted(g.nodes)}"
        del g.nodes[nid]
        g.edges = [e for e in g.edges if e.src != nid and e.dst != nid]
        return None
    if atype == "modify_node":
        patch = dict(payload["patch"])
        wire = patch.pop("wire", None) or {}
        if patch:
            err = _modify_node(g, payload["id"], patch, action, step)
            if err:
                return err
        elif payload["id"] not in g.nodes:
            return f"unknown node {payload['id']!r}; existing nodes: {sorted(g.nodes)}"
        return _apply_wire(g, payload["id"], wire, replace=True)
    if atype == "add_edge":
        return _add_edge(g, payload["edge"])
    if atype == "remove_edge":
        return _remove_edge(g, payload["edge"])
    if atype == "finish":
        return None
    if atype == "batch":
        for i, sub in enumerate(payload["actions"]):
            st = sub["type"]
            sp = {k: v for k, v in sub.items() if k != "type"}
            err = _apply_inplace(g, st, sp, action, episode, program, step)
            if err:
                return f"batch action #{i} ({st}): {err}"
        return None
    return f"unknown action type {atype!r}"


def apply_action_detailed(graph: WorkflowGraph, action: Action, episode: Any, program: Any, *,
                          step: int | None = None) -> tuple[WorkflowGraph, str | None, list[str]]:
    """Like :func:`apply_action` but also returns the list of graph-validation errors (if any)."""
    if action.type not in ACTION_TYPES:
        return graph, f"unknown action type {action.type!r}; expected one of {list(ACTION_TYPES)}", []
    payload, err = _normalize_payload(action.type, {"type": action.type, **(action.payload or {})})
    if err:
        return graph, err, []
    g = graph.copy()
    try:
        err = _apply_inplace(g, action.type, payload or {}, action, episode, program, step)
    except (ValueError, TypeError) as ex:   # e.g. an invalid PortSchema value that slipped through
        err = f"invalid {action.type}: {type(ex).__name__}: {ex}"
    if err:
        return graph, err, []
    verrs = g.validate()
    if verrs:
        return graph, ("graph invalid after edit (edit not applied): " + "; ".join(verrs)
                       + " -- fix the offending edges in the same edit (a batch of remove_edge / add_edge / modify_node)"), verrs
    return g, None, []


def apply_action(graph: WorkflowGraph, action: Action, episode: Any, program: Any, *,
                 step: int | None = None) -> tuple[WorkflowGraph, str | None]:
    """Apply one atomic edit to a copy of ``graph``.

    Returns ``(new_graph, None)`` on success or ``(graph, error)`` with the *unchanged* input graph.
    ``step`` (optional) is recorded into ``node.origin`` of added / modified nodes.
    """
    g, err, _ = apply_action_detailed(graph, action, episode, program, step=step)
    return g, err


# -------------------------------------------------------------------------- edit classes
def _sub_actions(action: Action) -> list[Action]:
    subs = action.payload.get("actions") or []
    out = []
    for s in subs:
        if isinstance(s, dict):
            out.append(Action(str(s.get("type", "")), {k: v for k, v in s.items() if k != "type"}, list(action.uses)))
    return out


def is_control_edit(action: Action, graph_before: WorkflowGraph | None = None) -> bool:
    """Pi_ctrl membership: topology / routing / configuration edits that preserve executable payloads."""
    t = action.type
    if t == "batch":
        subs = _sub_actions(action)
        return bool(subs) and all(is_control_edit(s, graph_before) for s in subs)
    if t in ("add_edge", "remove_edge", "remove_node", "finish"):
        return True
    if t == "add_node":
        return (action.payload.get("node") or {}).get("kind") in CONTROL_NODE_KINDS
    if t == "modify_node":
        keys = set((action.payload.get("patch") or {}).keys())
        return bool(keys) and keys <= {"config", "wire"}
    return False


def is_exec_edit(action: Action, graph_before: WorkflowGraph | None = None) -> bool:
    """Pi_exec membership: edits that generate or repair executable payloads (code / llm prompt / ports)."""
    t = action.type
    if t == "batch":
        return any(is_exec_edit(s, graph_before) for s in _sub_actions(action))
    if t == "add_node":
        return (action.payload.get("node") or {}).get("kind") in EXEC_NODE_KINDS
    if t == "modify_node":
        return bool(set((action.payload.get("patch") or {}).keys()) & set(EXEC_PATCH_KEYS))
    return False


def touched_nodes(action: Action) -> set[str]:
    """Node ids an action adds, removes, modifies or (for edges) connects."""
    t, p = action.type, action.payload
    if t == "add_node":
        nid = (p.get("node") or {}).get("id")
        return {nid} if isinstance(nid, str) else set()
    if t in ("remove_node", "modify_node"):
        return {p["id"]} if isinstance(p.get("id"), str) else set()
    if t in ("add_edge", "remove_edge"):
        e = p.get("edge") or {}
        return {x for x in (e.get("src"), e.get("dst")) if isinstance(x, str)}
    if t == "batch":
        out: set[str] = set()
        for s in _sub_actions(action):
            out |= touched_nodes(s)
        return out
    return set()
