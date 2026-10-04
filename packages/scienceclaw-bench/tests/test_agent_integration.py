"""Solver with the real modules (core.actions, runtime executor/replay/values, llm.fake.FakeLLM).

Skipped automatically while a dependency module is missing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

pytest.importorskip("scienceclaw.core.actions")
pytest.importorskip("scienceclaw.runtime.executor")
pytest.importorskip("scienceclaw.runtime.replay")
pytest.importorskip("scienceclaw.runtime.values")
fake = pytest.importorskip("scienceclaw.llm.fake")

from test_agent_support import (  # noqa: E402
    ACCEPTANCE_TEXT, BUGGY_CODE, SECRET_METRIC, SECRET_SCORE, build_script, make_episode, step_of,
)

from scienceclaw.agent.solver import Solver  # noqa: E402
from scienceclaw.config import EvolutionConfig, SolverConfig  # noqa: E402
from scienceclaw.core.program import AgentProgram  # noqa: E402


def _responder(script: list[dict]):
    def respond(role: str, messages: list[dict]) -> str:
        assert role == "policy"
        k = step_of(messages)
        return json.dumps(script[k] if 0 <= k < len(script) else {"action": {"type": "finish"}, "uses": []})
    return respond


def _texts(llm) -> str:
    return "\n".join(m["content"] for c in llm.calls for m in c["messages"])


def test_real_runtime_failure_then_pass(tmp_path: Path) -> None:
    llm = fake.FakeLLM(_responder(build_script(BUGGY_CODE, fix=True)))
    solver = Solver(SolverConfig(), llm, EvolutionConfig())
    res = solver.solve(make_episode(), AgentProgram(), "source", tmp_path / "run")
    assert [e.passed for e in res.evidence] == [False, True], [e.summary() for e in res.evidence]
    assert res.passed and res.z == 1 and res.final_source == "pass"
    assert res.y == [1.25, 2.5, 3.75, 5.0]
    assert res.evidence[1].eval.reproducible is True
    assert "boom in code node" in res.steps[3].feedback["text"]      # the policy saw the crash (visible)
    text = _texts(llm)
    for hidden in (SECRET_METRIC, str(SECRET_SCORE), ACCEPTANCE_TEXT, "INVISIBLE_CONSTRAINT_TEXT"):
        assert hidden not in text
    # policy contexts contain no volatile run-specific content (wall times, absolute run paths)
    assert str(tmp_path) not in text and str(tmp_path.resolve()) not in text
    assert res.usage["node_runs"] > 0 and res.usage["replays"] == 2


def test_real_runtime_contexts_identical_across_run_dirs(tmp_path: Path) -> None:
    """Identical episodes/programs give byte-identical policy contexts (LLM cache + lazy re-validation)."""
    contexts = []
    for name in ("a", "bb"):
        llm = fake.FakeLLM(_responder(build_script(BUGGY_CODE, fix=True)))
        Solver(SolverConfig(), llm).solve(make_episode(), AgentProgram(), "eval", tmp_path / name)
        contexts.append([c["messages"] for c in llm.calls])
    assert contexts[0] == contexts[1]
