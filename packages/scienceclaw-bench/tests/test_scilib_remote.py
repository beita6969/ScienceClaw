"""File-spool bridge that runs pretrained scilib tools on a GPU host: request keys, array coding, the client wait loop, and the
udparse_pretrained routing (a thread plays the broker; no network, torch or weights)."""
from __future__ import annotations

import gzip
import io
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "remote"))

from scilib import _remote
from scilib import udparse_pretrained as upp
import worker


def _serve(spool: Path, handler, stop: threading.Event, seen: list):
    while not stop.is_set():
        for req in spool.glob("req-*.json.gz"):
            key = req.name[4:-8]
            payload = json.loads(gzip.decompress(req.read_bytes()))
            seen.append(payload)
            try:
                out = {"ok": True, "result": _remote.encode(handler(payload["fn"], _remote.decode(payload["args"])))}
            except Exception as e:  # noqa: BLE001
                out = {"ok": False, "error": str(e)}
            _remote._write_atomic(spool / f"resp-{key}.json.gz", gzip.compress(json.dumps(out).encode()))
            req.unlink()
        time.sleep(0.02)


@pytest.fixture
def broker(tmp_path, monkeypatch):
    spool = tmp_path / "spool"
    spool.mkdir()
    monkeypatch.setenv(_remote.SPOOL_ENV, str(spool))
    monkeypatch.delenv("SCIENCECLAW_NO_PRETRAINED", raising=False)
    monkeypatch.delenv("SCIENCECLAW_NO_SCILIB", raising=False)
    stop, seen = threading.Event(), []
    box = {"handler": lambda fn, args: None}
    t = threading.Thread(target=_serve, args=(spool, lambda fn, a: box["handler"](fn, a), stop, seen), daemon=True)
    t.start()
    yield box, seen, spool
    stop.set()
    t.join(2)


def test_array_and_scalar_coding_round_trip():
    obj = {"a": np.arange(6, dtype=np.float32).reshape(2, 3), "b": [np.int64(3), np.float32(0.5), (1, 2)], "c": None}
    back = _remote.decode(json.loads(json.dumps(_remote.encode(obj))))
    assert np.array_equal(back["a"], obj["a"]) and back["a"].dtype == np.float32
    assert back["b"] == [3, 0.5, [1, 2]] and back["c"] is None


def test_request_key_is_stable_and_argument_sensitive():
    k1, _ = _remote.request_key("m", "f", {"x": [1, 2], "y": 3})
    k2, _ = _remote.request_key("m", "f", {"y": 3, "x": [1, 2]})
    k3, _ = _remote.request_key("m", "f", {"x": [1, 2], "y": 4})
    assert k1 == k2 != k3


def test_disabled_without_spool_or_when_switched_off(tmp_path, monkeypatch):
    monkeypatch.delenv(_remote.SPOOL_ENV, raising=False)
    assert not _remote.enabled()
    monkeypatch.setenv(_remote.SPOOL_ENV, str(tmp_path))
    assert _remote.enabled()
    monkeypatch.setenv("SCIENCECLAW_NO_PRETRAINED", "1")
    assert not _remote.enabled()


def test_call_returns_result_and_stored_answer_is_reused(broker):
    box, seen, spool = broker
    box["handler"] = lambda fn, a: {"n": int(np.sum(a["x"]))}
    x = np.arange(4)
    assert _remote.call("m", "f", {"x": x}, timeout=5) == {"n": 6}
    assert _remote.call("m", "f", {"x": x}, timeout=5) == {"n": 6}
    assert len(seen) == 1


def test_call_raises_remote_errors_and_times_out(broker, tmp_path):
    box, _, spool = broker

    def boom(fn, a):
        raise ValueError("bad input")
    box["handler"] = boom
    with pytest.raises(RuntimeError, match="bad input"):
        _remote.call("m", "f", {"x": 1}, timeout=5)


def test_call_times_out_when_nobody_answers(tmp_path, monkeypatch):
    monkeypatch.setenv(_remote.SPOOL_ENV, str(tmp_path))
    monkeypatch.delenv("SCIENCECLAW_NO_PRETRAINED", raising=False)
    monkeypatch.delenv("SCIENCECLAW_NO_SCILIB", raising=False)
    with pytest.raises(RuntimeError, match="no answer"):
        _remote.call("m", "f", {"x": 1}, timeout=0.05, poll=0.01)


def test_transient_failures_are_not_kept(broker):
    box, seen, spool = broker
    key, _ = _remote.request_key("m", "f", {"x": 1})
    _remote._write_atomic(spool / f"resp-{key}.json.gz", gzip.compress(json.dumps({"ok": False, "transient": True, "error": "ssh down"}).encode()))
    with pytest.raises(RuntimeError, match="ssh down"):
        _remote.call("m", "f", {"x": 1}, timeout=5)
    assert not (spool / f"resp-{key}.json.gz").exists()


def test_udparse_routes_to_the_gpu_host_when_it_cannot_run_locally(broker, monkeypatch):
    box, seen, _ = broker
    monkeypatch.setattr(upp, "_local_ok", lambda: False)
    assert upp.available()

    def handler(fn, a):
        if fn == "parse_gold_tokens":
            return [{"head": [0] * len(f), "deprel": ["root"] * len(f), "upos": ["X"] * len(f)} for f in a["sentences"]]
        return "/remote/model.pt"
    box["handler"] = handler
    out = upp.parse_gold_tokens([["a", "b"], [], {"words": [{"form": "c"}]}])
    assert [len(o["head"]) for o in out] == [2, 0, 1]
    assert seen[0]["fn"] == "parse_gold_tokens"
    sents = [{"words": [{"form": "x", "head": 0, "deprel": "root", "misc": "secret"}]}]
    before = len(seen)
    with pytest.raises(RuntimeError, match="finetune.*disabled.*frozen-inference"):
        upp.finetune(sents, steps=3)
    # A disabled finetune must not enqueue a remote request (and therefore
    # cannot leak labelled fields or start a worker-side trainer).
    assert len(seen) == before


def test_udparse_unavailable_without_local_stack_or_spool(monkeypatch):
    monkeypatch.delenv(_remote.SPOOL_ENV, raising=False)
    monkeypatch.setattr(upp, "_local_ok", lambda: False)
    assert not upp.available()
    with pytest.raises(RuntimeError, match="not available"):
        upp.parse_gold_tokens([["a"]])


def test_remote_worker_rejects_finetune_before_import_or_output(tmp_path, monkeypatch):
    """The GPU worker must fail closed even if a caller bypasses the client wrapper."""
    assert ("udparse_pretrained", "finetune") not in worker.ALLOWED

    payload = {"module": "udparse_pretrained", "fn": "finetune", "args": {"train_sentences": []}}
    encoded = gzip.compress(json.dumps(payload).encode())

    class _NativeStream:
        def __init__(self, raw=b""):
            self.buffer = io.BytesIO(raw)

    stdin, stdout = _NativeStream(encoded), _NativeStream()
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(worker, "OUT_ROOT", tmp_path / "remote_models")
    monkeypatch.setattr(worker.importlib, "import_module", lambda *_: (_ for _ in ()).throw(AssertionError("imported")))

    worker.main()
    reply = json.loads(gzip.decompress(stdout.buffer.getvalue()))
    assert reply["ok"] is False and "not served remotely" in reply["error"]
    assert not (tmp_path / "remote_models").exists()
