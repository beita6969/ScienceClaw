"""Deterministic scripted policy for driving the evolution pipeline without a language model.

``scripted_responder`` is a ``(role, messages) -> str`` callable for ``scienceclaw.llm.fake.FakeLLM``; it plays the
policy and patch roles against the synthetic regression task in ``synthetic_adapter``. ``operator_instantiating``
adapts it so that a retrieved fit-and-predict Operator is placed as a node.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable

_TRIVIAL_CODE = """import numpy as np

def run(inputs, config):
    y = np.asarray(inputs["y_train"], dtype=float)
    X = np.asarray(inputs["X_eval"], dtype=float)
    return {"y_pred": np.full(X.shape[0], float(y.mean()))}
"""

_GOOD_CODE = """import numpy as np

def _features(X):
    x1, x2 = X[:, 0], X[:, 1]
    return np.column_stack([np.ones(len(X)), x1, x1 ** 2, x1 ** 3, x2, x2 ** 2, x1 * x2])

def run(inputs, config):
    Xt = np.asarray(inputs["X_train"], dtype=float)
    yt = np.asarray(inputs["y_train"], dtype=float)
    Xe = np.asarray(inputs["X_eval"], dtype=float)
    A = _features(Xt)
    w = np.linalg.solve(A.T @ A + 1e-6 * np.eye(A.shape[1]), A.T @ yt)
    return {"y_pred": _features(Xe) @ w}
"""

_REF_RE = re.compile(r"\b(?:skill|op):[A-Za-z0-9_\-\.]*[A-Za-z0-9_\-]")
_NODE_RE = re.compile(r"\[([A-Za-z0-9_\-]+)\] kind=")
_EDGE_RE = re.compile(r"([A-Za-z0-9_\-]+)\.([A-Za-z0-9_]+) -> ([A-Za-z0-9_\-]+)\.([A-Za-z0-9_]+)")


def _msg_text(m: dict) -> str:
    c = m.get("content", "")
    if isinstance(c, list):
        return "\n".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in c)
    return str(c)


def _node(nid: str, kind: str, **kw: Any) -> dict:
    return {"id": nid, "kind": kind, **kw}


_DEV_RE = re.compile(r'"dev_rmse":\s*([-+0-9.eE]+)')
REPAIR_RULES = ("always", "library_only", "library_or_hash")
_FIT_INPUTS = {"X_train": {"type": "array", "shape": ["n", 2]}, "y_train": {"type": "array", "shape": ["n"]},
               "X_eval": {"type": "array", "shape": ["m", 2]}}
_FIT_OUTPUTS = {"y_pred": {"type": "array", "shape": ["m"], "dtype": "float"}}
_FIT_WIRING = (("tr", "X_train", "X_train"), ("tr", "y_train", "y_train"), ("ev", "X_eval", "X_eval"))


def _edge(src: str, src_port: str, dst: str, dst_port: str) -> dict:
    return {"type": "add_edge", "edge": {"src": src, "src_port": src_port, "dst": dst, "dst_port": dst_port}}


def scripted_responder(repair: str = "library_or_hash") -> Callable[[str, list[dict]], str]:
    """Return a deterministic ``(role, messages) -> str`` responder for ``scienceclaw.llm.fake.FakeLLM``.

    policy role: reads the current canvas from the most recent message that contains one (the text after
    ``## Current canvas``: node headers ``[id] kind=...`` and ``edges:`` lines, as printed by
    ``WorkflowGraph.render_compact``) and emits the next missing canvas edit of a fixed script: load tools ->
    a *trivial* mean predictor ``fit`` -> submit (a replay-verified failure: it does not beat the reference) ->
    [repair by replacing the model node: add a polynomial least-squares code node ``fit2``, wire its inputs,
    remove ``fit``, wire ``fit2`` to the submit node (a replay-verified success whose window holds both
    control edits and an executable edit)] -> finish. The repair needs 14 canvas edits, so the episode budget
    must allow >= 14 steps (the pipeline tests use 16). It cites every
    ``skill:``/``op:`` reference found in the prompt in ``uses`` (so the source-replay Use check can observe
    evolved components). Whether the repair happens is set by ``repair``:

    * ``"always"``: always repair (every episode is solved, independent of the program);
    * ``"library_only"``: repair iff the prompt cites a library component (Skill/Operator);
    * ``"library_or_hash"`` (default): repair iff a library component is cited, or — emulating an LLM that
      only sometimes finds the fix on its own — iff a deterministic hash of the visible dev score of the
      trivial predictor is even (~50% of episodes). This makes evolution observable: A_0 solves only some
      episodes, a promoted program with a retrieved component solves all of them.

    patch role (Skill/Operator documentation): a generic, instance-independent JSON document.
    executor role: ``"0"``.
    """
    if repair not in REPAIR_RULES:
        raise ValueError(f"repair must be one of {REPAIR_RULES}")

    def respond(role: str, messages: list[dict]) -> str:
        if role == "patch":
            return json.dumps({
                "title": "Fit a flexible model on the labelled data before submitting",
                "body": "When a baseline prediction does not beat the reference, replace it with a model that "
                        "can represent non-linear structure in the features, check the visible dev score, and "
                        "keep the output aligned with the evaluation rows.",
                "tags": ["regression", "tabular"],
                "name": "fit_and_predict", "description": "Fits a regression model and predicts the eval rows.",
                "skills": [{"title": "Fit a flexible model on the labelled data before submitting",
                            "body": "Replace a baseline that does not beat the reference with a model that can "
                                    "represent non-linear feature effects; verify with the dev score.",
                            "tags": ["regression", "tabular"]}],
            })
        if role != "policy":
            return "0"
        texts = [_msg_text(m) for m in messages]
        latest = next((t for t in reversed(texts) if "(empty canvas)" in t or " kind=" in t), "")
        canvas = latest[latest.rfind("## Current canvas"):] if "## Current canvas" in latest else latest
        nodes = set(_NODE_RE.findall(canvas))
        edges = {(a, b, c, d) for a, b, c, d in _EDGE_RE.findall(canvas)}
        refs = sorted({m.group(0) for t in texts for m in _REF_RE.finditer(t)})
        repaired = "np.linalg.solve" in canvas
        dev = _DEV_RE.findall(latest)
        if repair == "always" or refs:
            do_repair = True
        elif repair == "library_only" or not dev:
            do_repair = False
        else:
            do_repair = int(hashlib.sha256(dev[-1].encode()).hexdigest(), 16) % 2 == 0
        repaired = repaired or "fit2" in nodes
        build: list[tuple[bool, dict]] = [
            ("tr" in nodes, {"type": "add_node", "node": _node("tr", "tool", ref="load_train")}),
            ("ev" in nodes, {"type": "add_node", "node": _node("ev", "tool", ref="load_eval_inputs")}),
            ("fit" in nodes, {"type": "add_node", "node": _node(
                "fit", "code", code=_TRIVIAL_CODE, inputs=_FIT_INPUTS, outputs=_FIT_OUTPUTS)}),
            *[((src, sp, "fit", dp) in edges, _edge(src, sp, "fit", dp)) for src, sp, dp in _FIT_WIRING],
            ("out" in nodes, {"type": "add_node", "node": _node("out", "submit")}),
            (("fit", "y_pred", "out", "y") in edges, _edge("fit", "y_pred", "out", "y")),
        ]
        # repair = replace the model node: a new code node (executable edit) plus re-wiring and removal of the
        # old node (control edits). The intermediate canvases never feed the submit node, so the next replay
        # happens only when fit2 is wired to it -> delta(e-, e+) holds both control and executable edits.
        fix: list[tuple[bool, dict]] = [
            ("fit2" in nodes, {"type": "add_node", "node": _node(
                "fit2", "code", code=_GOOD_CODE, inputs=_FIT_INPUTS, outputs=_FIT_OUTPUTS)}),
            *[((src, sp, "fit2", dp) in edges, _edge(src, sp, "fit2", dp)) for src, sp, dp in _FIT_WIRING],
            ("fit" not in nodes, {"type": "remove_node", "id": "fit"}),
            (("fit2", "y_pred", "out", "y") in edges, _edge("fit2", "y_pred", "out", "y")),
        ]
        if repaired:
            script = fix
        else:
            script = build + ([] if not do_repair else fix)
        action = next((a for done, a in script if not done), {"type": "finish"})
        return json.dumps({"thought": "Next canvas edit of the scripted policy.", "action": action,
                           "uses": refs})

    return respond


_OP_CARD_RE = re.compile(r"\bop:([A-Za-z0-9_\-\.]*[A-Za-z0-9_\-])\(([^\n]*?)\) -> \(([^\n]*?)\)")
_PORT_NAME_RE = re.compile(r"(?:^|, )([A-Za-z_][A-Za-z0-9_]*): ")
_MODEL_NODE = "fit2"


def _port_by_suffix(ports: list[str], name: str) -> str | None:
    return next((p for p in ports if p == name or p.endswith("__" + name)), None)


def operator_instantiating(base: Any) -> Any:
    """Wrap ``scripted_responder`` so that a *retrieved* fit-and-predict Operator is instantiated.

    The scripted policy only cites the ``skill:``/``op:`` references it sees in ``uses``; it never places an
    operator node. Since Use(omega) is judged on the passing evidence (an Operator counts as used only if an
    operator node with that ref is in the passing graph and ran ok), a cited-but-unused Operator makes every
    pipeline candidate fail R_src. When the prompt carries an Operator card whose ports match the model node
    (``*X_train``, ``*y_train``, ``*X_eval`` -> ``*y_pred``), the model node of the scripted repair
    (``fit2``) becomes an ``operator`` node with that ref and the wiring actions are re-targeted to the
    Operator's port names; the canvas shown to the scripted policy is de-prefixed so its script is unchanged.
    """
    def _text(m: dict) -> str:
        c = m.get("content", "")
        if isinstance(c, list):
            return "\n".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in c)
        return str(c)

    def respond(role: str, messages: list[dict]) -> str:
        if role != "policy":
            return base(role, messages)
        system = _text(messages[0]) if messages and messages[0].get("role") == "system" else ""
        card = _OP_CARD_RE.search(system)
        if card is None:
            return base(role, messages)
        oid = card.group(1)
        ins = _PORT_NAME_RE.findall(card.group(2))
        outs = _PORT_NAME_RE.findall(card.group(3))
        pmap_in = {n: _port_by_suffix(ins, n) for n in ("X_train", "y_train", "X_eval")}
        pmap_out = {"y_pred": _port_by_suffix(outs, "y_pred")}
        if not all(pmap_in.values()) or not all(pmap_out.values()):
            return base(role, messages)
        strip = re.compile(r"\b" + _MODEL_NODE + r"\.(?:" + "|".join(
            re.escape(v) for v in [*pmap_in.values(), *pmap_out.values()]) + r")\b")
        seen = [dict(m, content=strip.sub(lambda mo: _MODEL_NODE + "." + mo.group(0).split(".", 1)[1].split("__")[-1], _text(m)))
                if m.get("role") != "system" else m for m in messages]
        reply = json.loads(base(role, seen))
        act = reply.get("action") or {}
        if act.get("type") == "add_node" and (act.get("node") or {}).get("id") == _MODEL_NODE:
            act["node"] = {"id": _MODEL_NODE, "kind": "operator", "ref": f"op:{oid}"}
        elif act.get("type") == "add_edge":
            e = act["edge"]
            if e.get("dst") == _MODEL_NODE and pmap_in.get(e.get("dst_port")):
                e["dst_port"] = pmap_in[e["dst_port"]]
            if e.get("src") == _MODEL_NODE and pmap_out.get(e.get("src_port")):
                e["src_port"] = pmap_out[e["src_port"]]
        return json.dumps(reply)

    return respond
