"""Tests for scienceclaw.llm.FakeLLM."""
from __future__ import annotations

import threading

import pytest

from scienceclaw.config import LLMConfig
from scienceclaw.llm import USAGE_KEYS, FakeLLM, FakeLLMExhausted, LLMError, LLMResponse, empty_usage

M = [{"role": "user", "content": "x" * 40}]


def test_list_responder_consumed_in_order_and_recorded() -> None:
    llm = FakeLLM(["first", "second"])
    assert llm.remaining == 2
    r1 = llm.chat("policy", M, json_mode=True, tag="step0")
    r2 = llm.chat("patch", M, max_tokens=5, temperature=0.5, cache_salt="s")
    assert (r1.text, r2.text) == ("first", "second")
    assert r1.finish_reason == "stop" and not r1.cached and r1.role == "policy" and r1.tag == "step0"
    assert llm.remaining == 0
    assert [c["role"] for c in llm.calls] == ["policy", "patch"]
    assert llm.calls[0]["json_mode"] is True and llm.calls[0]["tag"] == "step0" and llm.calls[0]["text"] == "first"
    assert llm.calls[1]["max_tokens"] == 5 and llm.calls[1]["temperature"] == 0.5
    assert llm.calls[1]["cache_salt"] == "s" and llm.calls[1]["index"] == 1
    with pytest.raises(FakeLLMExhausted):
        llm.chat("policy", M)


def test_callable_responder_and_usage() -> None:
    seen = []

    def responder(role, messages):
        seen.append((role, messages[-1]["content"]))
        return "y" * 21

    llm = FakeLLM(responder)
    assert llm.remaining is None
    r = llm.chat("executor", M, tag="node")
    assert seen == [("executor", "x" * 40)]
    assert set(r.usage) == set(USAGE_KEYS)
    assert r.usage["prompt_tokens"] == 10 and r.usage["completion_tokens"] == 5 and r.usage["calls"] == 1
    u = llm.usage()
    assert u["total"]["completion_tokens"] == 5 and u["by_role"]["executor"]["calls"] == 1
    assert u["by_tag"]["node"]["prompt_tokens"] == 10
    llm.reset_usage()
    assert llm.usage()["total"]["calls"] == 0


def test_response_objects_and_exceptions_as_items() -> None:
    llm = FakeLLM([LLMResponse(text="", usage=empty_usage(), cached=False, latency_s=0, model="m",
                               finish_reason="length"),
                   LLMError("gateway down", status=503, retryable=True)])
    r = llm.chat("policy", M)
    assert r.text == "" and r.finish_reason == "length"
    with pytest.raises(LLMError, match="gateway down"):
        llm.chat("policy", M)
    assert llm.usage()["total"]["errors"] == 1 and "gateway down" in llm.calls[1]["error"]


def test_chat_many_sequential_order_and_errors() -> None:
    llm = FakeLLM(["a", LLMError("bad", status=400), "c"])
    out = llm.chat_many("executor", [M, M, M], return_exceptions=True, tag="b")
    assert [r.text for r in out] == ["a", "", "c"]
    assert out[1].finish_reason == "error" and "bad" in out[1].error and out[1].tag == "b"
    llm2 = FakeLLM(["a", LLMError("bad", status=400), "c"])
    with pytest.raises(LLMError):
        llm2.chat_many("executor", [M, M, M])
    assert llm.chat_many("executor", []) == []


def test_chat_many_concurrent_preserves_order() -> None:
    barrier = threading.Barrier(4, timeout=5)

    def responder(role, messages):
        barrier.wait()          # proves 4 calls are in flight at once
        return messages[0]["content"].upper()

    llm = FakeLLM(responder, concurrent=True, cfg=LLMConfig(concurrency=4))
    batch = [[{"role": "user", "content": f"q{i}"}] for i in range(8)]
    assert [r.text for r in llm.chat_many("policy", batch)] == [f"Q{i}" for i in range(8)]
    assert llm.usage()["total"]["calls"] == 8 and len(llm.calls) == 8


def test_validation() -> None:
    with pytest.raises(TypeError):
        FakeLLM("just one string")
    with pytest.raises(TypeError):
        FakeLLM(42)
    llm = FakeLLM(lambda role, messages: 123)
    with pytest.raises(TypeError):
        llm.chat("policy", M)
    with pytest.raises(ValueError):
        FakeLLM(["x"]).chat("judge", M)
    assert isinstance(FakeLLM(["x"]).cfg, LLMConfig)
