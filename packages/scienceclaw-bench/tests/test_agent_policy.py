"""Policy tests: one proposal per step, single re-ask on parse errors with summed usage."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_agent_support import FINISH, ScriptedLLM, add_node, stub_parse  # noqa: E402

from scienceclaw.agent.policy import Policy, logical_tokens, response_usage  # noqa: E402
from scienceclaw.config import SolverConfig  # noqa: E402


def test_propose_parses_first_reply() -> None:
    llm = ScriptedLLM([add_node(id="a", kind="tool", ref="load_x")])
    action, err, raw, usage = Policy(llm, SolverConfig(), stub_parse).propose("SYS", [{"role": "user", "content": "U"}])
    assert err is None and action.type == "add_node" and '"add_node"' in raw
    assert usage["calls"] == 1 and usage["attempts"] == 1 and usage["prompt_tokens"] == 100
    msgs = llm.calls[0]["messages"]
    assert msgs == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "U"}]
    assert llm.calls[0]["role"] == "policy"


def test_reask_once_with_parse_error_and_summed_usage() -> None:
    llm = ScriptedLLM(["not json at all", FINISH])
    action, err, raw, usage = Policy(llm, SolverConfig(), stub_parse).propose("SYS", [{"role": "user", "content": "U"}])
    assert err is None and action.type == "finish"
    assert usage["calls"] == 2 and usage["attempts"] == 2 and usage["prompt_tokens"] == 200
    retry = llm.calls[1]["messages"]
    assert retry[2] == {"role": "assistant", "content": "not json at all"}
    assert retry[3]["role"] == "user" and "could not be parsed" in retry[3]["content"]
    assert "invalid JSON" in retry[3]["content"]


def test_two_parse_errors_return_none() -> None:
    llm = ScriptedLLM(["bad", ""])
    action, err, raw, usage = Policy(llm, SolverConfig(), stub_parse).propose("SYS", [])
    assert action is None and err and raw == "" and usage["attempts"] == 2
    assert llm.calls[1]["messages"][-2]["content"] == "bad"


def test_parser_exception_is_a_parse_error() -> None:
    def crash(text):
        raise RuntimeError("parser bug")

    action, err, _, _ = Policy(ScriptedLLM(["x", "y"]), SolverConfig(), crash).propose("S", [])
    assert action is None and "parser bug" in err


def test_cached_usage_accounting() -> None:
    llm = ScriptedLLM([FINISH], tokens_per_call=1000, cached=True)
    _, _, _, usage = Policy(llm, SolverConfig(), stub_parse).propose("S", [])
    assert usage["prompt_tokens"] == 0 and usage["cached_prompt_tokens"] == 1000 and usage["cached_calls"] == 1
    assert logical_tokens(usage) == 1500

    class R:  # a client that flags cached=True but reports tokens under the plain keys
        text, cached, latency_s = "x", True, 0.0
        usage = {"prompt_tokens": 10, "completion_tokens": 5}

    u = response_usage(R())
    assert u["prompt_tokens"] == 0 and u["cached_prompt_tokens"] == 10 and u["cached_calls"] == 1
    assert logical_tokens(u) == 15


def test_default_parser_batch_only_in_single_turn() -> None:
    pytest.importorskip("scienceclaw.core.actions")
    batch = {"thought": "t", "action": {"type": "batch", "actions": [{"type": "finish"}]}, "uses": []}
    llm = ScriptedLLM([batch, batch])
    action, err, _, usage = Policy(llm, SolverConfig(orchestration="canvas")).propose("S", [])
    assert action is None and "batch" in err and usage["attempts"] == 2
    llm = ScriptedLLM([batch])
    action, err, _, _ = Policy(llm, SolverConfig(orchestration="single_turn")).propose("S", [])
    assert err is None and action.type == "batch"
