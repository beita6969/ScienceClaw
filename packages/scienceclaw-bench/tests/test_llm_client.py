"""Tests for scienceclaw.llm.client.LLMClient against a local fake HTTP gateway (offline)."""
from __future__ import annotations

import json
import logging
import pickle
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import pytest

from scienceclaw.config import LLMConfig, ModelRole
from scienceclaw.llm import (
    USAGE_KEYS,
    LLMClient,
    LLMError,
    build_request_body,
    cache_key,
    is_reasoning_model,
    logical_usage,
)
from scienceclaw.llm.client import DEFAULT_USER_AGENT

SECRET = "sk-test-SECRET-0123456789abcdef"
MSGS = [{"role": "system", "content": "You answer tersely."}, {"role": "user", "content": "Reply with OK"}]


# ----------------------------------------------------------------- fake gateway

def ok(content: str | None = "OK", finish: str = "stop", usage: tuple[int, int, int] = (10, 5, 0)) -> dict:
    p, c, r = usage
    return {"status": 200, "json": {
        "id": "cmpl-1", "object": "chat.completion", "model": "served-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c,
                  "completion_tokens_details": {"reasoning_tokens": r}}}}


def err(status: int, body: str = '{"error": "nope"}', headers: dict | None = None) -> dict:
    return {"status": status, "text": body, "headers": headers or {}}


class GatewayState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.script: list[dict] = []
        self.responder: Callable[[dict], dict] | None = None
        self.requests: list[dict] = []
        self.inflight = 0
        self.max_inflight = 0
        self.base_url = ""

    def next_action(self, body: dict) -> dict:
        if self.responder is not None:
            return self.responder(body)
        with self.lock:
            if not self.script:
                return err(599, "script exhausted")
            return self.script.pop(0)


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - http.server API
        state: GatewayState = self.server.state  # type: ignore[attr-defined]
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with state.lock:
            state.requests.append({"path": self.path, "headers": dict(self.headers.items()), "body": body})
            state.inflight += 1
            state.max_inflight = max(state.max_inflight, state.inflight)
        try:
            action = state.next_action(body)
            if action.get("delay"):
                time.sleep(action["delay"])
            if action.get("drop"):
                self.close_connection = True
                return                       # close the socket without any response
            payload = json.dumps(action["json"]) if "json" in action else action.get("text", "")
            data = payload.encode()
            self.send_response(action["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for k, v in action.get("headers", {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)
        finally:
            with state.lock:
                state.inflight -= 1

    def log_message(self, *args: Any) -> None:
        return


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        return  # broken pipes after client-side timeouts are expected in tests


@pytest.fixture()
def gateway():
    srv = _Server(("127.0.0.1", 0), _Handler)
    srv.state = GatewayState()  # type: ignore[attr-defined]
    srv.state.base_url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    t.start()
    yield srv.state
    srv.shutdown()
    srv.server_close()


@pytest.fixture()
def creds(tmp_path: Path, gateway: GatewayState) -> Path:
    p = tmp_path / "client.json"
    p.write_text(json.dumps({"base_url": gateway.base_url, "api_key": SECRET}))
    return p


def make_client(tmp_path: Path, creds: Path, *, sleeps: list | None = None, cache: bool = True,
                **cfg_kw: Any) -> LLMClient:
    cfg = LLMConfig(credentials_file=str(creds), use_cache=cache, max_retries=cfg_kw.pop("max_retries", 4),
                    timeout_s=cfg_kw.pop("timeout_s", 5.0), **cfg_kw)
    cfg.policy = ModelRole(model="lab-gpt-5.4-mini", max_tokens=100, temperature=0.0, reasoning_effort="low",
                           json_mode=False)
    cfg.executor = ModelRole(model="lab-gpt-4.1-mini", max_tokens=50, temperature=0.2, reasoning_effort=None)
    rec = sleeps if sleeps is not None else []
    return LLMClient(cfg, cache_dir=tmp_path / "cache" if cache else None, sleep=rec.append,
                     backoff_base_s=0.01, backoff_max_s=1.0, seed=0)


# ------------------------------------------------------------ request building

@pytest.mark.parametrize("name,expected", [
    ("lab-gpt-5.4-mini", True), ("gpt-5", True), ("o3-mini", True), ("lab-o4-mini", True), ("openai/o3", True),
    ("o1", True), ("lab-gpt-4.1-mini", False), ("gpt-4o-mini", False), ("llama-3.1-70b", False),
    ("qwen2.5-72b", False),
])
def test_is_reasoning_model(name: str, expected: bool) -> None:
    assert is_reasoning_model(name) is expected


def test_body_reasoning_model() -> None:
    role = ModelRole(model="lab-gpt-5.4-mini", max_tokens=6000, temperature=0.0, reasoning_effort="low",
                     json_mode=True)
    body = build_request_body(role, MSGS)
    assert body == {"model": "lab-gpt-5.4-mini", "messages": MSGS, "max_completion_tokens": 6000,
                    "reasoning": {"effort": "low"}, "temperature": 0.0, "response_format": {"type": "json_object"}}
    assert "max_tokens" not in body


def test_body_reasoning_model_omits_none_temperature_and_effort() -> None:
    role = ModelRole(model="o3-mini", max_tokens=300, temperature=None, reasoning_effort=None)
    body = build_request_body(role, MSGS)
    assert body["max_completion_tokens"] == 300
    assert "temperature" not in body and "reasoning" not in body and "response_format" not in body


def test_body_non_reasoning_model() -> None:
    role = ModelRole(model="lab-gpt-4.1-mini", max_tokens=2000, temperature=0.3, reasoning_effort="low")
    body = build_request_body(role, MSGS)
    assert body["max_tokens"] == 2000 and body["temperature"] == 0.3
    assert "max_completion_tokens" not in body and "reasoning" not in body


def test_body_call_overrides() -> None:
    role = ModelRole(model="lab-gpt-4.1-mini", max_tokens=2000, temperature=0.3, json_mode=True)
    body = build_request_body(role, MSGS, json_mode=False, max_tokens=17, temperature=0.9)
    assert body["max_tokens"] == 17 and body["temperature"] == 0.9 and "response_format" not in body
    body = build_request_body(ModelRole(model="lab-gpt-4.1-mini"), MSGS, json_mode=True)
    assert body["response_format"] == {"type": "json_object"}
    with pytest.raises(ValueError):
        build_request_body(role, MSGS, max_tokens=0)


def test_body_min_completion_limit() -> None:
    # the gateway rejects limits < 16 with a 400 upstream_error, so they are raised to 16
    assert build_request_body(ModelRole(model="lab-gpt-4.1-mini"), MSGS, max_tokens=5)["max_tokens"] == 16
    assert build_request_body(ModelRole(model="lab-gpt-5.4-mini", max_tokens=3), MSGS)["max_completion_tokens"] == 16


def test_cache_key_is_canonical_and_sensitive() -> None:
    role = ModelRole(model="lab-gpt-4.1-mini", max_tokens=50, temperature=0.0)
    b = build_request_body(role, MSGS)
    k = cache_key(b)
    assert k == cache_key(dict(reversed(list(b.items()))))
    assert len(k) == 64
    assert cache_key(b, "salt") != k
    assert cache_key({**b, "model": "other"}) != k
    assert cache_key({**b, "max_tokens": 51}) != k
    assert cache_key({**b, "messages": MSGS[:1]}) != k


# ----------------------------------------------------------------- transport

def test_request_headers_path_and_body(tmp_path, gateway, creds) -> None:
    gateway.script = [ok("OK")]
    client = make_client(tmp_path, creds)
    r = client.chat("policy", MSGS, tag="t")
    assert r.text == "OK" and r.finish_reason == "stop" and not r.cached and r.model == "lab-gpt-5.4-mini"
    req = gateway.requests[0]
    assert req["path"] == "/v1/chat/completions"
    assert req["headers"]["Authorization"] == f"Bearer {SECRET}"
    assert req["headers"]["User-Agent"] == DEFAULT_USER_AGENT
    assert req["headers"]["Content-Type"] == "application/json"
    assert req["body"]["max_completion_tokens"] == 100 and req["body"]["reasoning"] == {"effort": "low"}
    assert set(r.usage) == set(USAGE_KEYS)
    assert r.usage["prompt_tokens"] == 10 and r.usage["completion_tokens"] == 5 and r.usage["calls"] == 1


def test_retry_429_then_200(tmp_path, gateway, creds) -> None:
    gateway.script = [err(429, '{"error":"rate limited"}'), ok("fine")]
    sleeps: list[float] = []
    client = make_client(tmp_path, creds, sleeps=sleeps)
    r = client.chat("executor", MSGS)
    assert r.text == "fine"
    assert len(gateway.requests) == 2 and len(sleeps) == 1
    assert 0.005 <= sleeps[0] <= 0.01
    tot = client.usage()["total"]
    assert tot["retries"] == 1 and tot["calls"] == 1 and tot["errors"] == 0


def test_retry_after_header_is_honoured(tmp_path, gateway, creds) -> None:
    gateway.script = [err(503, "busy", headers={"Retry-After": "0.7"}), ok()]
    sleeps: list[float] = []
    make_client(tmp_path, creds, sleeps=sleeps).chat("executor", MSGS)
    assert sleeps == [pytest.approx(0.7)]


def test_retries_5xx_reset_timeout_and_malformed(tmp_path, gateway, creds) -> None:
    gateway.script = [err(500), {"drop": True}, {"delay": 0.6, **ok("late")},
                      {"status": 200, "json": {"error": {"message": "upstream"}}}, err(408), ok("done")]
    sleeps: list[float] = []
    client = make_client(tmp_path, creds, sleeps=sleeps, max_retries=6, timeout_s=0.25)
    r = client.chat("executor", MSGS)
    assert r.text == "done"
    assert len(gateway.requests) == 6 and len(sleeps) == 5
    assert client.usage()["total"]["retries"] == 5


def test_403_is_not_retried_and_key_not_leaked(tmp_path, gateway, creds, caplog) -> None:
    echo = json.dumps({"error": "forbidden", "auth": f"Bearer {SECRET}", "raw": SECRET, "pad": "x" * 2000})
    gateway.script = [err(403, echo), ok()]
    sleeps: list[float] = []
    client = make_client(tmp_path, creds, sleeps=sleeps)
    with caplog.at_level(logging.DEBUG), pytest.raises(LLMError) as ei:
        client.chat("policy", MSGS)
    e = ei.value
    assert e.status == 403 and not e.retryable and e.role == "policy" and e.model == "lab-gpt-5.4-mini"
    assert len(gateway.requests) == 1 and sleeps == []
    text = str(e) + repr(e.args) + e.body + caplog.text
    assert SECRET not in text and "[REDACTED]" in e.body
    assert len(e.body) < 600
    assert client.usage()["total"]["errors"] == 1
    assert SECRET not in repr(client)


@pytest.mark.parametrize("status", [400, 401, 404])
def test_other_4xx_not_retried(tmp_path, gateway, creds, status) -> None:
    gateway.script = [err(status), ok()]
    with pytest.raises(LLMError) as ei:
        make_client(tmp_path, creds).chat("executor", MSGS)
    assert ei.value.status == status and not ei.value.retryable
    assert len(gateway.requests) == 1


def test_gives_up_after_max_retries(tmp_path, gateway, creds, caplog) -> None:
    gateway.script = [err(503)] * 3 + [ok()]
    sleeps: list[float] = []
    with caplog.at_level(logging.WARNING), pytest.raises(LLMError) as ei:
        make_client(tmp_path, creds, sleeps=sleeps, max_retries=2).chat("executor", MSGS)
    assert ei.value.retryable and ei.value.status == 503 and "gave up after 3 attempts" in str(ei.value)
    assert len(gateway.requests) == 3
    assert 0.005 <= sleeps[0] <= 0.01 and 0.01 <= sleeps[1] <= 0.02     # exponential with jitter
    assert SECRET not in caplog.text
    assert ei.value.attempts == 3


def test_failed_call_accounts_retries(tmp_path, gateway, creds) -> None:
    gateway.script = [err(503)] * 3
    client = make_client(tmp_path, creds, max_retries=2)
    with pytest.raises(LLMError):
        client.chat("executor", MSGS)
    tot = client.usage()["total"]
    assert tot["errors"] == 1 and tot["retries"] == 2 and tot["calls"] == 0


UPSTREAM = json.dumps({"error": {"message": "Upstream request failed", "type": "compat_error",
                                 "param": "", "code": "upstream_error"}})


def test_gateway_upstream_error_400_gets_limited_retries(tmp_path, gateway, creds) -> None:
    gateway.script = [err(400, UPSTREAM), ok("recovered")]
    assert make_client(tmp_path, creds, cache=False, max_retries=6).chat("executor", MSGS).text == "recovered"
    gateway.requests.clear()
    gateway.script = [err(400, UPSTREAM)] * 5 + [ok("too late")]
    with pytest.raises(LLMError) as ei:
        make_client(tmp_path, creds, cache=False, max_retries=6).chat("executor", MSGS)
    assert ei.value.status == 400 and ei.value.attempts == 3 and len(gateway.requests) == 3
    assert "gave up after 3 attempts" in str(ei.value)


def test_upstream_error_code_never_retried_on_auth_errors(tmp_path, gateway, creds) -> None:
    gateway.script = [err(403, UPSTREAM), ok()]
    with pytest.raises(LLMError) as ei:
        make_client(tmp_path, creds).chat("executor", MSGS)
    assert not ei.value.retryable and len(gateway.requests) == 1


def test_retry_logs_do_not_contain_key(tmp_path, gateway, creds, caplog) -> None:
    gateway.script = [err(502, f"bad gateway Bearer {SECRET}"), ok()]
    with caplog.at_level(logging.DEBUG):
        make_client(tmp_path, creds).chat("executor", MSGS)
    assert "attempt 1/5" in caplog.text and SECRET not in caplog.text


# --------------------------------------------------------------- length retry

def test_empty_length_is_retried_once_with_doubled_limit(tmp_path, gateway, creds) -> None:
    gateway.script = [ok("", "length", (10, 100, 100)), ok("ANSWER", "stop", (10, 30, 20))]
    client = make_client(tmp_path, creds)
    r = client.chat("policy", MSGS)
    assert r.text == "ANSWER" and r.finish_reason == "stop"
    assert [q["body"]["max_completion_tokens"] for q in gateway.requests] == [100, 200]
    assert r.usage["calls"] == 2 and r.usage["prompt_tokens"] == 20
    assert r.usage["completion_tokens"] == 130 and r.usage["reasoning_tokens"] == 120
    assert client.usage()["total"]["length_retries"] == 1


def test_length_retry_happens_only_once_and_empty_is_not_cached(tmp_path, gateway, creds) -> None:
    gateway.script = [ok("", "length"), ok(None, "length"), ok("x")]
    client = make_client(tmp_path, creds)
    r = client.chat("policy", MSGS)
    assert r.text == "" and r.finish_reason == "length" and len(gateway.requests) == 2
    r2 = client.chat("policy", MSGS)     # not served from cache: a fresh request is made
    assert r2.text == "x" and not r2.cached and len(gateway.requests) == 3


def test_length_retry_non_reasoning_doubles_max_tokens(tmp_path, gateway, creds) -> None:
    gateway.script = [ok("", "length"), ok("y")]
    make_client(tmp_path, creds).chat("executor", MSGS)
    assert [q["body"]["max_tokens"] for q in gateway.requests] == [50, 100]


def test_length_is_inferred_when_gateway_reports_stop(tmp_path, gateway, creds) -> None:
    # the lab gateway says "stop" even when the completion limit was exhausted by reasoning
    gateway.script = [ok("", "stop", (10, 100, 100)), ok("ANSWER", "stop", (10, 40, 30))]
    client = make_client(tmp_path, creds)
    r = client.chat("policy", MSGS)
    assert r.text == "ANSWER" and r.finish_reason == "stop"
    assert [q["body"]["max_completion_tokens"] for q in gateway.requests] == [100, 200]
    tot = client.usage()["total"]
    assert tot["length_inferred"] == 1 and tot["length_retries"] == 1
    # truncated non-empty text is surfaced as "length" but not retried
    gateway.script = [ok("1 2 3 4 5 6 7 8 ", "stop", (10, 50, 0))]
    r2 = client.chat("executor", MSGS)
    assert r2.finish_reason == "length" and r2.text.startswith("1 2 3") and len(gateway.requests) == 3


def test_nonempty_length_is_returned_without_retry(tmp_path, gateway, creds) -> None:
    gateway.script = [ok("partial answ", "length"), ok("never")]
    r = make_client(tmp_path, creds).chat("executor", MSGS)
    assert r.text == "partial answ" and r.finish_reason == "length" and len(gateway.requests) == 1


def test_failed_length_retry_still_accounts_first_request(tmp_path, gateway, creds) -> None:
    gateway.script = [ok("", "length", (10, 100, 100)), err(400)]
    client = make_client(tmp_path, creds)
    with pytest.raises(LLMError):
        client.chat("policy", MSGS)
    tot = client.usage()["total"]
    assert tot["calls"] == 1 and tot["completion_tokens"] == 100 and tot["errors"] == 1


# ---------------------------------------------------------------------- cache

def test_cache_hit_accounting_and_persistence(tmp_path, gateway, creds) -> None:
    gateway.script = [ok("A", usage=(12, 7, 3)), ok("B", usage=(12, 7, 3))]
    client = make_client(tmp_path, creds)
    r1 = client.chat("policy", MSGS, tag="x")
    r2 = client.chat("policy", MSGS, tag="x")
    assert len(gateway.requests) == 1
    assert r2.cached and not r1.cached and r2.text == "A" and r2.finish_reason == "stop"
    assert r2.usage["calls"] == 0 and r2.usage["prompt_tokens"] == 0 and r2.usage["completion_tokens"] == 0
    assert r2.usage["cached_calls"] == 1 and r2.usage["cached_prompt_tokens"] == 12
    assert r2.usage["cached_completion_tokens"] == 7 and r2.usage["cached_reasoning_tokens"] == 3
    assert logical_usage(r2.usage) == logical_usage(r1.usage)
    tot = client.usage()["total"]
    assert tot["calls"] == 1 and tot["cached_calls"] == 1
    assert tot["prompt_tokens"] == 12 and tot["cached_prompt_tokens"] == 12
    # a different salt is a different key
    r3 = client.chat("policy", MSGS, cache_salt="sample-2")
    assert not r3.cached and r3.text == "B" and len(gateway.requests) == 2
    client.close()
    # persistence across client instances; the key never reaches the cache file
    client2 = make_client(tmp_path, creds)
    assert client2.chat("policy", MSGS).cached
    assert len(gateway.requests) == 2
    for f in (tmp_path / "cache").iterdir():
        assert SECRET.encode() not in f.read_bytes()


def test_cache_disabled_and_cfg_cache_path(tmp_path, gateway, creds) -> None:
    gateway.script = [ok("a"), ok("b"), ok("c")]
    client = make_client(tmp_path, creds, cache=False)
    assert client.cache_path is None
    assert [client.chat("executor", MSGS).text for _ in range(2)] == ["a", "b"]
    cfg = LLMConfig(credentials_file=str(creds), cache_path=str(tmp_path / "sub" / "c.sqlite"))
    c2 = LLMClient(cfg, sleep=lambda s: None)
    assert c2.cache_path == tmp_path / "sub" / "c.sqlite" and c2.cache_path.exists()


# ------------------------------------------------------------ concurrency etc.

def test_chat_many_preserves_order_and_bounds_concurrency(tmp_path, gateway, creds) -> None:
    n = 9

    def responder(body: dict) -> dict:
        i = int(body["messages"][-1]["content"].split()[-1])
        return {"delay": 0.02 * (n - i), **ok(f"echo {i}")}

    gateway.responder = responder
    client = make_client(tmp_path, creds, concurrency=3)
    batch = [[{"role": "user", "content": f"item {i}"}] for i in range(n)]
    out = client.chat_many("executor", batch, tag="many")
    assert [r.text for r in out] == [f"echo {i}" for i in range(n)]
    assert 2 <= gateway.max_inflight <= 3
    u = client.usage()
    assert u["total"]["calls"] == n and u["by_tag"]["many"]["calls"] == n
    assert client.chat_many("executor", []) == []


def test_chat_many_errors(tmp_path, gateway, creds) -> None:
    def responder(body: dict) -> dict:
        return err(400, "bad item") if body["messages"][-1]["content"] == "bad" else ok("good")

    gateway.responder = responder
    client = make_client(tmp_path, creds, cache=False, concurrency=4)
    batch = [[{"role": "user", "content": c}] for c in ("a", "bad", "b")]
    out = client.chat_many("executor", batch, return_exceptions=True)
    assert [r.text for r in out] == ["good", "", "good"]
    assert out[1].finish_reason == "error" and "400" in out[1].error and out[0].error is None
    with pytest.raises(LLMError):
        client.chat_many("executor", batch)
    with pytest.raises(LLMError):                       # sequential path (max_workers=1)
        client.chat_many("executor", batch, max_workers=1)


def test_usage_by_role_tag_model_is_threadsafe(tmp_path, gateway, creds) -> None:
    gateway.responder = lambda body: ok("r", usage=(3, 2, 1))
    client = make_client(tmp_path, creds, cache=False, concurrency=8)
    threads = [threading.Thread(target=lambda i=i: client.chat("policy" if i % 2 else "executor", MSGS,
                                                               tag=f"t{i % 3}")) for i in range(24)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    u = client.usage()
    assert u["total"]["calls"] == 24 and u["total"]["prompt_tokens"] == 72
    assert u["by_role"]["policy"]["calls"] == 12 and u["by_role"]["executor"]["calls"] == 12
    assert sum(v["calls"] for v in u["by_tag"].values()) == 24
    assert u["by_model"]["lab-gpt-5.4-mini"]["reasoning_tokens"] == 12
    client.reset_usage()
    assert client.usage()["total"]["calls"] == 0


def test_credentials_errors_and_base_url_override(tmp_path, gateway, creds) -> None:
    gateway.script = [ok("via override")]
    missing = LLMConfig(credentials_file=str(tmp_path / "nope.json"), use_cache=False)
    with pytest.raises(LLMError) as ei:
        LLMClient(missing, sleep=lambda s: None).chat("executor", MSGS)
    assert not ei.value.retryable and "not found" in str(ei.value)
    bogus = tmp_path / "bogus.json"
    bogus.write_text(json.dumps({"base_url": "http://127.0.0.1:9/v1", "api_key": SECRET}))
    cfg = LLMConfig(credentials_file=str(bogus), base_url=gateway.base_url + "/", use_cache=False)
    assert LLMClient(cfg, sleep=lambda s: None).chat("executor", MSGS).text == "via override"
    nokey = tmp_path / "nokey.json"
    nokey.write_text(json.dumps({"base_url": gateway.base_url}))
    with pytest.raises(LLMError, match="no api_key"):
        LLMClient(LLMConfig(credentials_file=str(nokey), use_cache=False)).chat("executor", MSGS)
    assert len(gateway.requests) == 1


def test_input_validation_and_pickling(tmp_path, creds) -> None:
    client = make_client(tmp_path, creds)
    with pytest.raises(ValueError):
        client.chat("judge", MSGS)
    with pytest.raises(ValueError):
        client.chat("policy", [])
    with pytest.raises(ValueError):
        client.chat("policy", [{"content": "no role"}])
    with pytest.raises(TypeError):
        pickle.dumps(client)
    with pytest.raises(ValueError):
        LLMClient(LLMConfig(use_cache=False), user_agent="")


# ------------------------------------------------------- self-hosted endpoints

def test_endpoints_replace_gateway_and_never_send_key(tmp_path, gateway, creds) -> None:
    gateway.script = [ok("local")]
    cfg = LLMConfig(credentials_file=str(tmp_path / "does-not-exist.json"), endpoints=[gateway.base_url + "/"],
                    use_cache=False)
    cfg.executor = ModelRole(model="qwen3.8-27b", max_tokens=50, temperature=0.0, reasoning_effort=None,
                             extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    r = LLMClient(cfg, sleep=lambda s: None).chat("executor", MSGS)
    assert r.text == "local"
    req = gateway.requests[0]
    assert req["path"] == "/v1/chat/completions" and SECRET not in json.dumps(req["headers"])
    assert req["body"]["chat_template_kwargs"] == {"enable_thinking": False} and req["body"]["max_tokens"] == 50


def test_endpoints_spread_and_fail_over(tmp_path, gateway, creds) -> None:
    gateway.responder = lambda body: ok("up")
    dead = "http://127.0.0.1:9/v1"
    cfg = LLMConfig(credentials_file=str(tmp_path / "x.json"), endpoints=[dead, gateway.base_url], use_cache=False,
                    max_retries=3, concurrency=4)
    cfg.executor = ModelRole(model="qwen3.8-27b", max_tokens=50, reasoning_effort=None)
    client = LLMClient(cfg, sleep=lambda s: None, backoff_base_s=0.0, backoff_max_s=0.0)
    out = client.chat_many("executor", [MSGS] * 6)
    assert [r.text for r in out] == ["up"] * 6 and len(gateway.requests) == 6


def test_extra_body_in_cache_key_and_body() -> None:
    role = ModelRole(model="qwen3.8-27b", max_tokens=64, reasoning_effort=None,
                     extra_body={"chat_template_kwargs": {"enable_thinking": True}})
    b1 = build_request_body(role, MSGS)
    role.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    b2 = build_request_body(role, MSGS)
    assert b1["chat_template_kwargs"] != b2["chat_template_kwargs"] and cache_key(b1) != cache_key(b2)
    assert role.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}      # not mutated by the body builder
