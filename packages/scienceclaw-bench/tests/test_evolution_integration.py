"""End-to-end evolution with the real Solver, executor, replay, retriever and a scripted FakeLLM.

The scripted policy (role "policy") behaves like a weak agent without library support:

* on source episodes it builds load -> prep -> submit (replay-verified FAILURE), then repairs it with a config
  edit, a new code node and three routing edits (replay-verified SUCCESS);
* on validation episodes it stops after the failing workflow;
* whenever an Operator is retrieved into its prompt it wires that Operator between the loader and submit and
  cites the retrieved Skill/Operator ids in "uses".

So A_0 has MacroSR 0 on D_val; the linked bundle from the source repair passes boundary replay, source-task
replay (Pass and Use) and strictly improves Q_val, and is accepted into A_1.
"""
from __future__ import annotations

import json
import re

import pytest

pytest.importorskip("scienceclaw.runtime.executor")
pytest.importorskip("scienceclaw.runtime.replay")

from scienceclaw.config import RunConfig  # noqa: E402
from scienceclaw.core.program import AgentProgram  # noqa: E402
from scienceclaw.evolution import Evolver  # noqa: E402
from scienceclaw.llm.fake import FakeLLM  # noqa: E402

from test_evolution_support import StubPlan, make_episode, patch_responder, repair_steps  # noqa: E402

STEP_RE = re.compile(r"## Step (\d+)")
OP_RE = re.compile(r"\bop:([a-z0-9][a-z0-9-]*)")
SKILL_RE = re.compile(r"\bskill:([a-z0-9][a-z0-9-]*)")


def _step(messages: list[dict]) -> int:
    for m in reversed(messages):
        if m["role"] == "user":
            mm = STEP_RE.search(m["content"])
            if mm:
                return int(mm.group(1)) - 1
    return -1


def _operator_script(op_id: str, uses: list[str]) -> list[dict]:
    a = [{"type": "add_node", "node": {"id": "load", "kind": "tool", "ref": "load_data"}},
         {"type": "add_node", "node": {"id": "o", "kind": "operator", "ref": f"op:{op_id}"}},
         {"type": "add_edge", "edge": {"src": "load", "src_port": "table", "dst": "o", "dst_port": "prep__x"}},
         {"type": "add_node", "node": {"id": "sub", "kind": "submit"}},
         {"type": "add_edge", "edge": {"src": "o", "src_port": "post__z", "dst": "sub", "dst_port": "y"}},
         {"type": "finish"}]
    return [{"thought": "t", "action": x, "uses": uses} for x in a]


def _policy(role: str, messages: list[dict]) -> str:
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    k = _step(messages)
    ops = sorted(set(OP_RE.findall(system)))
    if ops:
        uses = [f"skill:{s}" for s in sorted(set(SKILL_RE.findall(system)))] + [f"op:{ops[0]}"]
        script = _operator_script(ops[0], uses)
    else:
        specs = repair_steps()
        if "(validation)" in system:
            specs = specs[:5]                      # a weak agent stops at the failing workflow
        script = [{"thought": "t", "action": s["action"], "uses": []} for s in specs]
    return json.dumps(script[k] if 0 <= k < len(script) else {"thought": "t", "action": {"type": "finish"}, "uses": []})


def _responder():
    patch = patch_responder()

    def respond(role: str, messages: list[dict]) -> str:
        return _policy(role, messages) if role == "policy" else patch(role, messages)
    return respond


def _episode(eid: str, kind: str, split: str):
    ep = make_episode(eid, "FoR99", split=split)
    ep.objective = f"Transform the loaded vector into the required output vector ({kind})."
    return ep


def test_real_stack_accepts_linked_bundle(tmp_path):
    cfg = RunConfig()
    cfg.solver.max_steps = 12
    llm = FakeLLM(_responder())
    plan = StubPlan([(1, _episode("src1", "source", "src"))],
                    {"FoR99": [_episode("val1", "validation", "val"), _episode("val2", "validation", "val")]})
    snaps = Evolver(cfg, llm, plan, tmp_path / "run", val_workers=2).run(AgentProgram())
    rd = tmp_path / "run"
    rows = [json.loads(x) for x in (rd / "candidates.jsonl").read_text().splitlines()]
    assert len(rows) == 1, rows
    c = rows[0]
    assert [b["ok"] for b in c["breplay"]] == [True], c["breplay"]
    assert c["R_src"] and c["Pass"] and c["Use"], c
    assert c["qval"]["inc"]["macro_sr"] == 0.0 and c["qval"]["cand"]["macro_sr"] == 1.0
    assert c["accepted"], c["reasons"]
    assert len(snaps) == 2 and len(snaps[1].skills) == 1 and len(snaps[1].operators) == 1
    op = next(iter(snaps[1].operators.values()))
    skill = next(iter(snaps[1].skills.values()))
    assert f"- {op.ref} -" in skill.body                           # linked bundle
    # the real solver also replays after the config edit (the submit fingerprint changed): e- is that
    # latest replay-verified failure, so delta = steps 6..9 (the shortest reproduced repair)
    stream = [json.loads(x) for x in (rd / "stream.jsonl").read_text().splitlines()]
    assert stream[0]["solve"]["evidence"] == [{"step": 4, "passed": False}, {"step": 5, "passed": False},
                                              {"step": 9, "passed": True}]
    assert op.provenance["episode"] == "src1" and op.provenance["k_minus"] == 5 and op.provenance["k_plus"] == 9
    assert stream[0]["instances"][0]["summary"]["delta_steps"] == [6, 7, 8, 9]
    # the policy never saw hidden information; the patch model saw it only as pass/fail
    policy_text = "\n".join(m["content"] for call in llm.calls if call["role"] == "policy" for m in call["messages"])
    assert "SECRET-HIDDEN-MSG" not in policy_text and "0.8765" not in policy_text
