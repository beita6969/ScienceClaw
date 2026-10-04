"""Solver + real runtime: code_edit under fixed_workflow (F4), scrub-before-truncate of run paths (F9), the dev score
staying out of the policy context when the config hides it. Skipped while a dependency module is missing."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

pytest.importorskip("scienceclaw.core.actions")
pytest.importorskip("scienceclaw.runtime.executor")
pytest.importorskip("scienceclaw.runtime.replay")
fake = pytest.importorskip("scienceclaw.llm.fake")

from test_agent_support import GOOD_CODE, WRONG_CODE, act, make_episode, step_of  # noqa: E402

from scienceclaw.agent.solver import FIXED_CODE_NODE, Solver, check_orchestration  # noqa: E402
from scienceclaw.config import EvolutionConfig, SolverConfig  # noqa: E402
from scienceclaw.core.actions import parse_action  # noqa: E402
from scienceclaw.core.graph import WorkflowGraph  # noqa: E402
from scienceclaw.core.program import AgentProgram  # noqa: E402

FINISH = {"thought": "done", "action": {"type": "finish"}, "uses": []}


def _responder(script: list[dict]):
    def respond(role: str, messages: list[dict]) -> str:
        assert role == "policy"
        k = step_of(messages)
        return json.dumps(script[k] if 0 <= k < len(script) else FINISH)
    return respond


def _patch(**patch) -> dict:
    return act({"type": "modify_node", "id": FIXED_CODE_NODE, "patch": patch})


def test_fixed_workflow_accepts_code_edit_and_rejects_other_keys(tmp_path: Path) -> None:
    script = [
        _patch(code=WRONG_CODE),
        _patch(code_edit={"find": "not in the code", "replace": "x"}),        # zero occurrences: rejected
        _patch(inputs={"x": {"type": "list"}}),                                # not a fixed-workflow key: rejected
        _patch(code_edit={"find": "0.5", "replace": "1.25"}),                  # applied: the code becomes GOOD_CODE
        FINISH,
    ]
    llm = fake.FakeLLM(_responder(script))
    res = Solver(SolverConfig(orchestration="fixed_workflow"), llm, EvolutionConfig()).solve(
        make_episode(), AgentProgram(), "source", tmp_path / "run")
    fbs = [s.feedback for s in res.steps]
    assert fbs[0]["action_ok"] is True
    assert fbs[1]["action_ok"] is False and "find" in fbs[1]["action_error"]
    assert fbs[2]["action_ok"] is False and "'code', 'code_edit' and/or 'config'" in fbs[2]["action_error"]
    assert fbs[3]["action_ok"] is True
    assert res.final_graph.nodes[FIXED_CODE_NODE].code == GOOD_CODE
    assert res.passed and res.y == [1.25, 2.5, 3.75, 5.0]
    system = llm.calls[0]["messages"][0]["content"]
    assert '"code" | "code_edit" | "config"' in system


@pytest.mark.parametrize("patch,ok", [({"code_edit": {"find": "a", "replace": "b"}}, True), ({"code": "x"}, True),
                                      ({"config": {"k": 1}}, True),
                                      ({"code": "x", "code_edit": {"find": "a", "replace": "b"}}, False),
                                      ({"prompt": "p"}, False), ({"wire": {}}, False), ({}, False)])
def test_check_orchestration_fixed_workflow_patch_keys(patch: dict, ok: bool) -> None:
    action, err = parse_action(json.dumps({"action": {"type": "modify_node", "id": "code", "patch": patch}}))
    if action is None:                    # the parser already rejects "code" with "code_edit" and empty patches
        assert not ok and err
        return
    reason = check_orchestration(action, WorkflowGraph(), "fixed_workflow", "code")
    assert (reason is None) == ok, reason


def _cwd_crash_script() -> list[dict]:
    crash = "import os\n\ndef run(inputs, config):\n    raise RuntimeError('cwd=' + os.getcwd())\n"
    return [_patch(code=crash), _patch(code=GOOD_CODE), FINISH]


def test_run_paths_and_uids_never_reach_the_policy_and_contexts_are_identical(tmp_path: Path) -> None:
    contexts = []
    for name in ("a", "a-much-longer-run-directory-name"):
        llm = fake.FakeLLM(_responder(_cwd_crash_script()))
        run_dir = tmp_path / name
        res = Solver(SolverConfig(orchestration="fixed_workflow"), llm, EvolutionConfig()).solve(
            make_episode(), AgentProgram(), "eval", run_dir)
        text = "\n".join(m["content"] for c in llm.calls for m in c["messages"])
        assert "RuntimeError: cwd=" in text                                      # the crash itself is visible
        assert str(run_dir) not in text and str(run_dir.resolve()) not in text
        assert not re.search(r"work/[A-Za-z0-9_]+-[0-9a-f]{8}", text)
        assert res.steps[0].feedback["text"].count("<run>") >= 1
        contexts.append([c["messages"] for c in llm.calls])
    assert contexts[0] == contexts[1]


def test_dev_score_is_absent_from_every_rendering_when_hidden(tmp_path: Path) -> None:
    script = [_patch(code=GOOD_CODE), _patch(code_edit={"find": "1.25", "replace": "1.25"}), FINISH]
    seen = {}
    for show in (True, False):
        llm = fake.FakeLLM(_responder(script))
        Solver(SolverConfig(orchestration="fixed_workflow", show_dev_score=show), llm, EvolutionConfig()).solve(
            make_episode(), AgentProgram(), "source", tmp_path / f"dev{int(show)}")
        seen[show] = "\n".join(m["content"] for c in llm.calls for m in c["messages"])
    assert "Dev score on visible data" in seen[True] and "dev: {" in seen[True]
    assert "Dev score on visible data" not in seen[False] and "dev: {" not in seen[False]
    assert "development score on visible data when the task provides one" in seen[True]
    assert "development score on visible data when the task provides one" not in seen[False]
