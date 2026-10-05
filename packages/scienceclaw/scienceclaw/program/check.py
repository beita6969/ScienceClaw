"""Run a typed library operator through the real canvas on concrete input values.

``check_operator`` builds a small task whose tools hand the given Python values to the operator's input ports, wires
``feeds -> operator -> submit`` on a :class:`~scienceclaw.canvas.CanvasSession` and replays it from a reset state. It therefore
exercises exactly what a workflow does with the operator: port type/shape/unit compatibility on the edges, the pre/post
conditions of the contract, the sandboxed execution and the reproducibility check.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from scienceclaw.task import Budget, Episode, EvalResult, ToolSpec
from scienceclaw.canvas.session import CanvasSession
from scienceclaw.core.program import AgentProgram
from scienceclaw.core.schema import PortSchema


def _episode(op_id: str, inputs: dict[str, Any], schemas: dict[str, PortSchema], output: str) -> Episode:
    tools = []
    for name, value in inputs.items():
        def feed(_inputs: dict, _config: dict, _v: Any = value) -> dict:
            return {"data": _v}
        tools.append(ToolSpec(name=f"feed_{name}", description=f"Supplies the value of input {name}.", inputs={},
                              outputs={"data": schemas[name]}, fn=feed))

    def evaluate(y: Any, trace: Any) -> EvalResult:
        return EvalResult(primary=1.0, h={}, accepted=True, completed=True, details={"norm_score": 1.0})

    return Episode(id=f"check-{op_id}", discipline="CHECK", family="Check", split="live", task_type="operator_check",
                   objective=f"Apply the operator {op_id} to the supplied inputs.", required_output=PortSchema(type="any"),
                   tools=tools, constraints=[], budget=Budget(max_steps=len(inputs) * 2 + 6), lineage={"source": "check"},
                   acceptance="The operator runs, its contract holds and the run reproduces.", tags=[], metric="ok", direction="max",
                   _evaluate=evaluate, _dev_evaluate=None)


def check_operator(op_id: str, inputs: dict[str, Any], *, output: str | None = None, program: AgentProgram | None = None,
                   work_dir: str | Path | None = None) -> dict[str, Any]:
    """Execute operator ``op_id`` on ``inputs`` (port name -> value) and report what happened.

    Returns ``{"ok", "feedback", "reproducible", "output" (the value of ``output`` or the only output port), "steps"}``.
    """
    from scienceclaw.program.seed import seed_program

    program = program or seed_program()
    op = program.operators.get(op_id)
    if op is None:
        raise KeyError(f"unknown operator {op_id!r}; have {sorted(program.operators)}")
    missing = sorted(set(op.inputs) - set(inputs))
    if missing:
        raise ValueError(f"operator {op_id} needs inputs {sorted(op.inputs)}; missing {missing}")
    out_port = output or (next(iter(op.outputs)) if len(op.outputs) == 1 else None)
    if out_port is None or out_port not in op.outputs:
        raise ValueError(f"name the output port to deliver, one of {sorted(op.outputs)}")

    ep = _episode(op_id, {k: inputs[k] for k in op.inputs}, op.inputs, out_port)
    tmp = None
    if work_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="sc-opcheck-")
        work_dir = tmp.name
    try:
        session = CanvasSession(ep, program, Path(work_dir) / "run", session_id=f"chk{abs(hash(op_id)) % 10**8}", kind="check")
        steps: list[dict] = []

        def act(action: dict) -> dict:
            r = session.act({"thought": "check", "action": action, "uses": []})
            steps.append({"ok": r["ok"], "text": r["text"].splitlines()[0] if r["text"] else ""})
            return r

        act({"type": "add_node", "node": {"id": "op", "kind": "operator", "ref": f"op:{op_id}"}})
        for name in op.inputs:
            act({"type": "add_node", "node": {"id": f"in_{name}", "kind": "tool", "ref": f"feed_{name}"}})
            act({"type": "add_edge", "edge": {"src": f"in_{name}", "src_port": "data", "dst": "op", "dst_port": name}})
        act({"type": "add_node", "node": {"id": "out", "kind": "submit"}})
        last = act({"type": "add_edge", "edge": {"src": "op", "src_port": out_port, "dst": "out", "dst_port": "y"}})
        replay = session.replay()
        ok = bool(last.get("submit_ready")) and replay.get("reproducible") is not False and all(s["ok"] for s in steps)
        return {"ok": ok, "feedback": last["text"], "reproducible": replay.get("reproducible"), "output": session._last_y,
                "steps": steps, "replay": replay.get("text")}
    finally:
        if tmp is not None:
            tmp.cleanup()
