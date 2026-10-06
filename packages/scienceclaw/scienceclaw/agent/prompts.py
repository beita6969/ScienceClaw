"""Policy prompts (paper Eq. 6 context): interface description only — no how-to strategies.

DESIGN decision 9: the system prompt contains only
  (A) what the Workflow Canvas, each node kind, each action and each tool does,
  (B) the deliverable format (required output schema of y, the submit node) and the acceptance rule,
  (C) interface rules (port-schema syntax, Python types of port values, edge compatibility / unit conversion,
      action JSON, how the deliverable is selected, "uses").
Everything procedural ("how to solve a task") must come from retrieved Skills and Operators, which are
rendered verbatim under "Skills from your library" / "Operators from your library".

:data:`STRATEGY_PATTERNS` / :func:`find_strategy_phrases` scan a prompt for imperative strategy phrasing;
the unit tests assert that the A_0 system prompt (no Skills) contains none of them.

Determinism: prompts depend only on the episode's public view, the retrieved components, the canvas, the set of
Python packages installed in the interpreter that runs code nodes, and execution feedback. No wall-clock
readings are rendered and free text is scrubbed of run-specific fragments *before* it is shortened, so
identical contexts give identical prompts (the LLM response cache relies on this).
"""
from __future__ import annotations

import functools
import importlib.util
import json
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..runtime.executor import format_summary

__all__ = [
    "STRATEGY_PATTERNS",
    "PARSE_RETRY_TEMPLATE",
    "find_strategy_phrases",
    "available_packages",
    "build_system_prompt",
    "build_step_message",
    "summarize_action",
    "summarize_feedback",
    "uses_note",
]

# Imperative strategy / advice phrasing that must never appear in the built-in prompt text.
STRATEGY_PATTERNS: tuple[str, ...] = (
    r"\bstart (by|with)\b", r"\bbegin (by|with)\b", r"\bfirst(ly)?\b", r"\bearly\b",
    r"\bcheck for\b", r"\bnans?\b", r"\bmissing values\b", r"\bcross[- ]?validat", r"\bhold[- ]?out\b",
    r"\bshould\b", r"\bought to\b", r"\brecommend", r"\bbest practice", r"\bmake sure\b", r"\bensure\b",
    r"\btry\b", r"\bconsider\b", r"\btips?\b", r"\bhints?\b", r"\bstrateg", r"\bgood idea\b", r"\bbaseline\b",
    r"\bsanity\b", r"\binspect", r"\bexplor", r"\bload(ing)? the data\b", r"\bstep[- ]by[- ]step\b",
    r"\biterat", r"\bimprov", r"\bfallback\b", r"\brobust", r"\bavoid\b", r"\bprefer", r"\bforget\b",
    r"\bremember to\b", r"\bdebug", r"\bhyper-?parameter", r"\btun(e|ing)\b", r"\bnormali[sz]",
    r"\bstandardi[sz]", r"\bfeature engineering\b", r"\bincremental", r"\bcareful", r"\bimportant\b",
    r"\bplan (your|the|ahead)\b", r"\bhelpful\b", r"\buseful\b", r"\bsimple (model|approach|method)\b",
    r"\bbefore (writing|adding|building|submitting)\b", r"\bgood (approach|way|practice)\b",
    r"\bit is (wise|advisable)\b", r"\bkeep (it|things) simple\b",
)
_STRATEGY_RE = [re.compile(p, re.IGNORECASE) for p in STRATEGY_PATTERNS]

PARSE_RETRY_TEMPLATE = (
    "Your previous reply could not be parsed as an action: {error}\n"
    "Reply with exactly one JSON object in the action format "
    '({{"thought": ..., "action": {{"type": ...}}, "uses": [...]}}) and nothing else.'
)


def find_strategy_phrases(text: str) -> list[str]:
    """Return the matched strategy phrases in ``text`` (empty list if none)."""
    hits: list[str] = []
    for rx in _STRATEGY_RE:
        for m in rx.finditer(text or ""):
            hits.append(m.group(0))
    return hits


# ---------------------------------------------------------------------------- execution environment (F10)
# (import name, name shown in the prompt): the packages a code node may import when they are installed. Code
# nodes run in worker processes of the interpreter that runs this program (``sys.executable``), so the same
# lookup answers "what can a code node import".
_PACKAGE_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"), ("sklearn", "scikit-learn"),
    ("statsmodels", "statsmodels"), ("networkx", "networkx"), ("sympy", "sympy"), ("joblib", "joblib"),
    ("torch", "torch"), ("lightgbm", "lightgbm"), ("xgboost", "xgboost"),
)


@functools.lru_cache(maxsize=1)
def available_packages() -> tuple[str, ...]:
    """Display names of the candidate packages that are importable here (looked up, not imported)."""
    found: list[str] = []
    for module, shown in _PACKAGE_CANDIDATES:
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            spec = None
        if spec is not None:
            found.append(shown)
    return tuple(found)


# ----------------------------------------------------------------------------------------- system prompt
_CANVAS = """\
# Workflow Canvas
You are the policy of a scientific workflow agent. You build the solution to one task as a workflow on the Workflow Canvas by emitting edit actions as JSON.
The Workflow Canvas is a typed directed acyclic graph (DAG). Each node has named, typed input ports and output ports; an edge connects one output port of a node to one input port of another node. After every edit the canvas is validated and executed: nodes whose specification or upstream values changed are (re)executed in topological order, results of unchanged nodes are reused from the checkpoint, and nodes with unwired or unavailable inputs stay "pending". Node ids are unique strings chosen by you (letters, digits and "_", at most 40 characters); port names match [A-Za-z_][A-Za-z0-9_]*. Data of the task is available only through tool nodes; values move between nodes only along edges."""

# Node-kind descriptions: (description, node JSON). ``@...@`` tokens are filled in by _node_kinds().
_KINDS: dict[str, tuple[str, str]] = {
    "tool": (
        '- tool: a task tool provided by the environment (listed under "Tools"). Input and output ports are filled in from the tool signature; "config" holds the tool options given in its signature.',
        'Node: {"id": "<id>", "kind": "tool", "ref": "<tool name>", "config": {...}}.'),
    "operator": (
        '- operator: a learned reusable subgraph from your library, shown as op:<id> under "Operators from your library". Operator nodes take no config. Ports are filled in from the operator signature; the subgraph is expanded and executed inline, and violations of its pre/postconditions are reported in the feedback.',
        'Node: {"id": "<id>", "kind": "operator", "ref": "op:<id>"}.'),
    "code": (
        '- code: Python source that defines a top-level function `def run(inputs: dict, config: dict) -> dict`. `inputs` maps each declared input port name to its value and `config` is the node config; the returned dict has exactly the declared output port names as keys. It runs in an isolated process with a limit of @MAX_NODE_S@ s; @PACKAGES@ The code may use only its inputs and config: no files outside its working directory, no network access, no subprocesses, no environment variables.',
        'Node: {"id": "<id>", "kind": "code", "code": "<python source>", "config": {...}, "inputs": {"<port>": <PortSchema>, ...}, "outputs": {"<port>": <PortSchema>, ...}}.'),
    "llm": (
        '- llm: a prompt template that the executor language model applies to every element of inputs["items"] (a list of dicts or strings, at most @MAX_LLM_ITEMS@ per run); {field} placeholders are filled from the fields of a dict item, {item} with the whole item, and {<port>} with the value of another input port of the node. config: "parse" ("text" | "json" | "number" | "choice"), "choices" (list, for parse = "choice"), "max_tokens". The single output port "outputs" is the list of parsed results, one per item.',
        'Node: {"id": "<id>", "kind": "llm", "prompt": "<template>", "config": {"parse": "text"}, "inputs": {"items": <PortSchema>}, "outputs": {"outputs": <PortSchema>}}; "inputs" and "outputs" may be omitted (default: list schemas).'),
    "submit": (
        '- submit: the single terminal node. Its one input port "y" carries the required output schema (see "Deliverable") and is filled in automatically; the value arriving at "y" is the deliverable of the task.',
        'Node: {"id": "<id>", "kind": "submit"}.'),
}


def _node_kinds(max_node_s: float, max_llm_items: int) -> str:
    """The "Node kinds" section."""
    pk = list(available_packages())
    packages = ("the Python standard library is available" if not pk else
                "the Python packages " + ", ".join(pk) + " and the standard library are available") + "."
    lines = ["# Node kinds"]
    for desc, node_json in _KINDS.values():
        line = desc + " " + node_json
        lines.append(line.replace("@MAX_NODE_S@", f"{float(max_node_s):g}").replace("@MAX_LLM_ITEMS@", str(int(max_llm_items)))
                     .replace("@PACKAGES@", packages))
    return "\n".join(lines)


_PORT_SCHEMA = """\
# Port schemas
PortSchema JSON: {"type": T, "shape": [...], "unit": "<unit>", "dtype": "<dtype>"}; only "type" is required.
- type T is one of: any, number, text, array, table, list, dict, series.
- shape is a list of integers or symbolic names such as "n"; a symbol binds to one size across an edge; null or absent = unspecified.
- unit is a unit string ("1" = dimensionless); null or absent = not applicable.
- dtype is an optional element type such as "float", "int", "str", "prob".
Python values: a port of type number carries an int or float (a 0-d numpy array is accepted); text a str; array a numpy.ndarray (a list, a tuple or a pandas.Series is accepted); table a pandas.DataFrame; list a list (a tuple is accepted); dict a dict; series a pandas.Series (a numpy.ndarray or a list is accepted); any carries any picklable object. The value at the submit node's input "y" is checked against the deliverable schema: type, shape (a symbol binds to one size) and, for dtype "prob", the range [0, 1]."""

_EDGES = """\
# Edges
An edge {"src": "<node id>", "src_port": "<output port>", "dst": "<node id>", "dst_port": "<input port>"} is accepted only if the two port schemas are compatible:
- the types are equal, or one of them is "any" (array <-> series and array <-> list are also compatible);
- the shapes unify;
- the units are equal or at least one is null. If both units are given and differ, the edge carries an explicit conversion "conversion": {"from": "<source unit>", "to": "<destination unit>", "factor": a, "offset": b}; the value is converted as x_dst = a * x_src + b and the conversion is recorded in the value's provenance.
Each input port accepts at most one incoming edge."""

# How the deliverable is selected (matches Solver._finalize): the latest step with a computed, validly wired y;
# a run that verifies workflows by hidden replay prefers the verified workflow with the best visible dev score.
_FINISH = ('"finish" ends the episode, and the episode also ends when the budget is exhausted. The deliverable is '
           'selected when the episode ends and is not always the current canvas: it is the workflow of the latest step '
           'at which the value at the submit node\'s input "y" was computed by a valid, fully wired workflow (an edit '
           'that leaves "y" uncomputed does not replace it); if some earlier workflow passed the run\'s hidden '
           'verification, the passing workflow with the best development score on visible data (ties: the later '
           'step) is selected instead. An episode without any such step counts as failed.')

_USES = ('"uses": list the {ids} ids whose guidance or code you are using in this action ([] if none). Only ids '
         'listed in this prompt are valid; other ids are ignored.')

_CODE_EDIT_RULE = ('"code_edit" is {"find": "<text>", "replace": "<text>"} and replaces the one exact occurrence of '
                   '"find" in the current code of a code node (it is rejected when "find" occurs zero or several times '
                   'or when the resulting code is not valid Python source with a top-level function run, and it cannot '
                   'be combined with "code")')
_CONFIG_RULE = '"config" is merged into the current config (a null value deletes that key)'


_PATCH_RULES = (f'"code" and "prompt" replace the current value; {_CODE_EDIT_RULE}; {_CONFIG_RULE}; "inputs" / '
                '"outputs" replace the declared ports of a code or llm node (edges into an input port that is '
                'no longer declared are dropped; ports of tool, operator and submit nodes are fixed by their specification).')
_PATCH_KEYS = '"code" | "code_edit" | "prompt" | "config" | "inputs" | "outputs" | "wire"'

_ACTIONS_CANVAS = """\
# Actions
Each reply is exactly one JSON object and nothing else:
{"thought": "<at most 3 sentences>", "action": <ACTION>, "uses": [@USES_IDS@]}
<ACTION> is one of:
  {"type": "add_node", "node": {<node fields as above>}, "wire": {"<input port>": "<src node>.<src port>", ...}}
  {"type": "remove_node", "id": "<node id>"}   (edges attached to the node are removed with it)
  {"type": "modify_node", "id": "<node id>", "patch": {@PATCH_KEYS@: <new value>, ...}}
  {"type": "add_edge", "edge": {"src": "<node id>", "src_port": "<port>", "dst": "<node id>", "dst_port": "<port>", "conversion": {...}}}   ("conversion" only when units differ)
  {"type": "remove_edge", "edge": {"src": "<node id>", "src_port": "<port>", "dst": "<node id>", "dst_port": "<port>"}}
  {"type": "finish"}
In modify_node, @PATCH_RULES@ "wire" (optional) connects input ports of the added / modified node to output ports of existing nodes in the same action; each entry is equivalent to one add_edge into that node (a value may also be {"from": "<src node>.<src port>", "conversion": {...}}); in modify_node, a wired port replaces the port's current incoming edge. In add_edge, "src_port" / "dst_port" may be omitted when that node has exactly one output / input port. An action that fails validation is not applied and the canvas stays unchanged. @FINISH@
@USES@"""

def _feedback_section(show_dev_score: bool) -> str:
    items = ["whether the action was applied (or the reason it was rejected)", "graph validation errors",
             "the status of every node (ok, error, pending, skipped) with error messages and the tail of its stdout",
             "summaries of node outputs (type, shape, dtype, unit, finite fraction, min/max/mean, column or key "
             "names, head)"]
    items.append("operator contract violations")
    items.append("the results of the visible constraint checks on the submit input")
    if show_dev_score:
        items.append("a development score on visible data when the task provides one")
    items[-1] += (". Output summaries are truncated (about 160 characters of a text value, a handful of elements of a "
                  "list or array); print() from a code node returns more (the last 800 characters of stdout are returned "
                  "for the node that just ran). A node whose code and inputs are unchanged is not run again; its earlier "
                  "result is reused and marked 'cached'. The step line also shows how many policy tokens are left in the budget")
    history = ("Earlier actions are listed in compact form with their results: the summary of y, "
               + ("the development score, " if show_dev_score else "")
               + "the failed constraints and the nodes that ran again or failed.")
    return "# Feedback\nAfter each action you receive: " + ", ".join(items[:-1]) + ", and " + items[-1] + ". " + history


_ACCEPTANCE = """\
# Acceptance
The deliverable counts as solved when all of the following hold:
- every constraint holds, including any that are not listed under "Constraints" (for example a quality bar that is checked
  on held-out data the workflow never loads; its value is not shown to you);
- the final workflow, replayed from its stored specification, reproduces the deliverable within the budget."""

_ORCH = """\
# Orchestration: canvas
One action per reply. Each action is applied and the canvas is executed before your next reply."""


def _json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)


def _schema_line(schema: Any) -> str:
    render = schema.render() if hasattr(schema, "render") else str(schema)
    d = schema.to_dict() if hasattr(schema, "to_dict") else {}
    d = {k: v for k, v in d.items() if k not in ("provenance", "description")}
    desc = getattr(schema, "description", "") or ""
    line = f"{render}  JSON: {_json(d)}"
    if desc:
        line += f"\n  description: {desc}"
    return line


def _actions_section() -> str:
    return (_ACTIONS_CANVAS.replace("@PATCH_KEYS@", _PATCH_KEYS).replace("@PATCH_RULES@", _PATCH_RULES)
            .replace("@FINISH@", _FINISH).replace("@USES@", _USES.format(ids="skill:<id> and op:<id>"))
            .replace("@USES_IDS@", '"skill:<id>", "op:<id>", ...'))


def build_system_prompt(episode: Any, skills: Sequence[Any], operators: Sequence[Any], *,
                        max_steps: int | None = None, show_dev_score: bool = True) -> str:
    """System prompt for the policy π_Θ0 on one episode.

    Args:
        episode: a :class:`scienceclaw.task.Episode`; only its public view is rendered.
        skills: retrieved Skills (rendered in full with ``Skill.render()``).
        operators: retrieved Operators (``OperatorSpec.render()``: signature, description, contract).
        max_steps: effective step budget (defaults to ``episode.budget.max_steps``).
        show_dev_score: whether the feedback carries the development score (the solver's ``show_dev_score``).
    """
    budget = episode.budget
    steps = int(max_steps if max_steps is not None else budget.max_steps)
    parts: list[str] = [
        _CANVAS,
        _node_kinds(float(budget.max_node_s), int(budget.max_llm_items)),
        _PORT_SCHEMA + "\n\n" + _EDGES,
        _actions_section(),
        _feedback_section(bool(show_dev_score)),
        _ORCH,
    ]

    # ------------------------------------------------------------------ task (public view only)
    task = ["# Task"]
    task.append(f"discipline: {episode.discipline}")
    task.append(f"task type: {episode.task_type}")
    if getattr(episode, "tags", None):
        task.append("tags: " + ", ".join(str(t) for t in episode.tags))
    task.append("objective:\n" + str(episode.objective).strip())
    parts.append("\n".join(task))

    parts.append("# Deliverable\nThe submit node's input y must have the schema:\n  y: " + _schema_line(episode.required_output))

    cons = [c for c in (episode.constraints or []) if getattr(c, "visible", True)]
    if cons:
        parts.append("# Constraints (checked on the deliverable)\n" + "\n".join(f"- {c.name}: {c.description}" for c in cons))

    parts.append(_ACCEPTANCE)

    tools = episode.tools or []
    parts.append("# Tools\n" + ("\n".join(f"- {t.signature()}" for t in tools) if tools else "(none)"))

    limits = ["# Budget",
              f"- replies (steps): at most {steps}",
              f"- policy tokens: at most {int(budget.max_policy_tokens)}",
              f"- wall-clock time: at most {float(budget.max_wall_s):g} s",
              f"- time per node run: at most {float(budget.max_node_s):g} s",
              f"- llm node items per run: at most {int(budget.max_llm_items)}"]
    parts.append("\n".join(limits))

    parts.append("# Skills from your library\n" + ("\n\n".join(s.render() for s in skills) if skills else "(none)"))
    parts.append("# Operators from your library\n" + ("\n\n".join(o.render() for o in operators) if operators else "(none)"))
    return "\n\n".join(parts).strip() + "\n"


# ---------------------------------------------------------------------------------------- step message
def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def summarize_action(action: Any) -> str:
    """One-line description of an action (Action object or dict)."""
    if action is None:
        return "(no parsable action)"
    if not isinstance(action, Mapping) and hasattr(action, "to_dict"):
        action = action.to_dict()
    a = dict(action)
    if "action" in a and isinstance(a["action"], Mapping) and "type" not in a:
        a = dict(a["action"])
    typ = a.get("type", "?")
    payload = a.get("payload") if isinstance(a.get("payload"), Mapping) else a
    if typ == "add_node":
        n = payload.get("node") or {}
        extra = f" ref={n.get('ref')}" if n.get("ref") else ""
        return f"add_node {n.get('id', '?')} kind={n.get('kind', '?')}{extra}"
    if typ == "remove_node":
        return f"remove_node {payload.get('id', '?')}"
    if typ == "modify_node":
        keys = ",".join(sorted((payload.get("patch") or {}).keys()))
        return f"modify_node {payload.get('id', '?')} [{keys}]"
    if typ in ("add_edge", "remove_edge"):
        e = payload.get("edge") or {}
        return f"{typ} {e.get('src', '?')}.{e.get('src_port', '?')} -> {e.get('dst', '?')}.{e.get('dst_port', '?')}"
    return str(typ)


_MAX_NODES_IN_SUMMARY = 8


def summarize_feedback(feedback: Any, max_chars: int = 400, scrub: Callable[[str], str] | None = None) -> str:
    """Compact result line of one step from visible feedback (runtime Feedback, dict, or plain text).

    The line carries, in this order: the outcome, the first validation error, the summary of the submitted ``y``,
    the development score (when the feedback has one), failed visible constraints, the first integrity violation
    and last the nodes that ran again or failed (nodes served from the checkpoint are only counted). ``scrub``
    (e.g. ``solver.scrub_volatile``) is applied to every free-text fragment BEFORE it is shortened, so a cut can
    never split a run-specific path; without it the text is used as given.
    """
    S: Callable[[str], str] = scrub if scrub is not None else (lambda t: t)

    def clip(s: str) -> str:
        return s if len(s) <= max_chars else s[: max_chars - 3] + "..."

    if feedback is None:
        return ""
    if isinstance(feedback, str):
        return clip(S(feedback))
    if not _get(feedback, "action_ok", True):
        return clip(S(f"rejected: {_get(feedback, 'action_error') or 'unknown error'}"))
    bits: list[str] = ["applied"]
    verrs = _get(feedback, "validation_errors") or []
    if verrs:
        bits.append(f"{len(verrs)} validation error(s): {S(str(verrs[0]))}")
    records = _get(feedback, "records") or {}
    if _get(feedback, "submit_ready"):
        ys = _get(feedback, "y_summary")
        bits.append("y: " + format_summary({k: v for k, v in ys.items() if k != "head"} if isinstance(ys, Mapping) else ys))
    elif records:
        bits.append("no y")
    dev = _get(feedback, "dev")
    if dev is not None:
        bits.append("dev: " + S(json.dumps(dev, sort_keys=True, default=str, ensure_ascii=False))[:120])
    vc = _get(feedback, "visible_constraints") or {}
    failed = sorted(k for k, v in vc.items() if isinstance(v, (list, tuple)) and v and not v[0])
    if failed:
        bits.append("constraints failed: " + ", ".join(failed))
    integ = _get(feedback, "integrity_violations") or []
    if integ:
        bits.append(f"integrity violations: {S(str(integ[0]))}")
    if records:
        changed: list[str] = []
        n_cached = 0
        for nid in sorted(records):
            r = records[nid]
            st = _get(r, "status", "?")
            if st == "ok" and _get(r, "cached", False):
                n_cached += 1
                continue
            err = _get(r, "error")
            item = f"{nid}={st}"
            if st == "error" and err:
                lines = S(str(err)).rstrip().splitlines()
                if lines:
                    item += f" ({lines[-1][:120]})"
            changed.append(item)
        shown = changed[:_MAX_NODES_IN_SUMMARY]
        if len(changed) > len(shown):
            shown.append(f"+{len(changed) - len(shown)} more")
        if n_cached:
            shown.append(f"+{n_cached} cached ok" if shown else f"{n_cached} cached ok")
        bits.append("nodes: " + ", ".join(shown))
    return clip(S("; ".join(bits)))


def _render_budget(budget_state: Mapping[str, Any] | None, step: int) -> str:
    b = dict(budget_state or {})
    max_steps = b.pop("max_steps", None)
    left = b.pop("steps_left", None)
    b.pop("step", None)
    head = f"Step {step + 1}" + (f" of {max_steps}" if max_steps is not None else "")
    if left is not None:
        head += f" ({left} left including this one)"
    extra = ", ".join(f"{k}: {b[k]}" for k in sorted(b))
    return head + (f"; {extra}" if extra else "")


def build_step_message(step: int, graph: Any, feedback_text: str | None, history: Sequence[Mapping[str, Any]],
                       budget_state: Mapping[str, Any] | None, *, window: int = 6,
                       scrub: Callable[[str], str] | None = None) -> str:
    """User message for step k: budget, compact history, last feedback, current canvas.

    Args:
        step: 0-based step index k.
        graph: current :class:`~scienceclaw.core.graph.WorkflowGraph` G_{t,k}.
        feedback_text: rendered feedback of the previous step (None at k = 0).
        history: earlier steps, oldest first; dicts with keys "step", "action" (one-line summary),
            "result" (one-line result) and optionally "thought". The last ``window`` entries are shown
            with thought and a longer result; older ones as one-liners.
        budget_state: e.g. {"max_steps": 12, "steps_left": 9}; rendered verbatim (no wall-clock values).
        scrub: removes run-specific fragments from history text; applied to every history field BEFORE it is
            shortened (DESIGN decision 8), so identical contexts give identical messages.
    """
    S: Callable[[str], str] = scrub if scrub is not None else (lambda t: t)
    out: list[str] = [f"## {_render_budget(budget_state, step)}"]
    hist = list(history or [])
    if hist:
        lines = ["## Earlier actions"]
        cut = max(0, len(hist) - max(0, int(window)))
        for i, h in enumerate(hist):
            k = h.get("step", i)
            act = S(str(h.get("action", "")))
            res = S(str(h.get("result", "") or ""))
            if i < cut:
                res = res if len(res) <= 160 else res[:157] + "..."
                lines.append(f"- step {int(k) + 1}: {act} -> {res}")
            else:
                res = res if len(res) <= 600 else res[:597] + "..."
                thought = S(str(h.get("thought", "") or "")).strip()
                lines.append(f"- step {int(k) + 1}: {act}" + (f"\n    thought: {thought[:300]}" if thought else "")
                             + f"\n    result: {res}")
        out.append("\n".join(lines))
    if feedback_text:
        out.append("## Feedback of the last action\n" + feedback_text.strip())
    render = graph.render_compact() if hasattr(graph, "render_compact") else str(graph)
    out.append("## Current canvas\n" + render)
    out.append("Reply with exactly one JSON object in the action format.")
    return "\n\n".join(out) + "\n"


def uses_note(dropped: Iterable[str]) -> str:
    """Interface note appended to the feedback when unknown ids were given in "uses"."""
    d = sorted(set(dropped))
    return f"Note: these \"uses\" ids are not in your library and were ignored: {', '.join(d)}" if d else ""
