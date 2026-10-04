"""Run a ``scilib`` function on a GPU host through a file spool (nothing here opens a connection).

When ``$SCIENCECLAW_REMOTE_SPOOL`` names an existing directory, a pretrained-model module that cannot run in this
interpreter writes its request there as ``req-<key>.json.gz`` and waits for ``resp-<key>.json.gz``; a broker process that
runs outside the sandbox (``scripts/remote/broker.py``) executes the request on the GPU host and writes the response.
``<key>`` is the hash of module, function and arguments, so an identical call returns the stored answer.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np

from ._pretrained import switched_off

SPOOL_ENV = "SCIENCECLAW_REMOTE_SPOOL"


def spool_dir() -> Path | None:
    p = os.environ.get(SPOOL_ENV)
    if switched_off() or not p or not Path(p).is_dir():
        return None
    return Path(p)


def enabled() -> bool:
    return spool_dir() is not None


def encode(obj):
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__nd__": base64.b64encode(buf.getvalue()).decode()}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode(v) for v in obj]
    return obj


def decode(obj):
    if isinstance(obj, dict):
        if set(obj) == {"__nd__"}:
            return np.load(io.BytesIO(base64.b64decode(obj["__nd__"])), allow_pickle=False)
        return {k: decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode(v) for v in obj]
    return obj


def request_key(module: str, fn: str, args: dict) -> tuple[str, bytes]:
    raw = json.dumps({"module": module, "fn": fn, "args": encode(args)}, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:24], raw


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def call(module: str, fn: str, args: dict, timeout: float = 780.0, poll: float = 0.5):
    """Execute ``scilib.<module>.<fn>(**args)`` on the GPU host and return its (decoded) result."""
    d = spool_dir()
    if d is None:
        raise RuntimeError("remote execution is not configured")
    key, raw = request_key(module, fn, args)
    req, resp = d / f"req-{key}.json.gz", d / f"resp-{key}.json.gz"
    if not resp.exists() and not req.exists():
        _write_atomic(req, gzip.compress(raw, 6))
    deadline = time.time() + timeout
    while not resp.exists():
        if time.time() > deadline:
            raise RuntimeError(f"remote {module}.{fn}: no answer from the GPU host within {timeout:.0f} s")
        time.sleep(poll)
    data = json.loads(gzip.decompress(resp.read_bytes()))
    if not data.get("ok"):
        if data.get("transient"):
            resp.unlink(missing_ok=True)
        raise RuntimeError(f"remote {module}.{fn} failed: {data.get('error', 'unknown error')}")
    return decode(data["result"])
