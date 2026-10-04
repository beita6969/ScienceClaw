"""Subprocess entry point for ``code`` nodes: ``python -m scienceclaw.runtime.node_worker <spec.json>``.

The spec (written by :mod:`scienceclaw.runtime.sandbox`) holds::

    {"node_id": str, "code_path": str, "inputs_path": str, "outputs_path": str, "meta_path": str,
     "config": dict, "declared_outputs": [str, ...]}

The worker loads the node source, unpickles the input values, seeds ``random`` / ``numpy`` from
``config.get("seed", 0)``, executes ``run(inputs, config)`` with stdout/stderr captured, checks that
every declared output port is present, pickles the declared outputs and writes ``meta.json``::

    {"status": "ok"|"error", "error": str|None (traceback tail <= 4000 chars),
     "stdout_tail": str (<= 4000 chars), "wall_s": float, "extra_outputs": [...]}

This module deliberately imports only the standard library (numpy only if installed, for seeding).
It never imports the LLM client or any other ScienceClaw module, it blocks imports of the
``scienceclaw`` package from node code (adapters / evaluators must stay out of reach) and it refuses
outbound network connections / DNS lookups (defense in depth behind the static integrity scan).
"""
from __future__ import annotations

import importlib.abc
import io
import json
import os
import pickle
import random
import re
import sys
import time
import traceback

TAIL_CHARS = 4000
CODE_FILENAME = "node_code.py"
PICKLE_PROTOCOL = 5
BLOCKED_IMPORT_ROOTS = ("scienceclaw",)


class _TailBuffer(io.TextIOBase):
    """A text sink that keeps only the last ``limit`` characters (bounded memory for chatty code)."""

    def __init__(self, limit: int = 4 * TAIL_CHARS) -> None:
        super().__init__()
        self._limit = limit
        self._parts: list[str] = []
        self._size = 0

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        if not isinstance(s, str):
            s = str(s)
        self._parts.append(s)
        self._size += len(s)
        if self._size > 2 * self._limit:
            joined = "".join(self._parts)[-self._limit:]
            self._parts, self._size = [joined], len(joined)
        return len(s)

    def flush(self) -> None:
        return None

    def getvalue(self) -> str:
        return "".join(self._parts)


class _ImportBlocker(importlib.abc.MetaPathFinder):
    """Refuses imports of the ScienceClaw package (not already loaded) from inside node code."""

    def find_spec(self, fullname, path=None, target=None):  # noqa: ANN001 - importlib protocol
        if fullname.split(".")[0] in BLOCKED_IMPORT_ROOTS and fullname not in sys.modules:
            raise ImportError(f"code nodes may not import {fullname!r}")
        return None


def tail(s: str, n: int = TAIL_CHARS) -> str:
    """Last ``n`` characters of ``s`` (with a leading marker when truncated)."""
    if len(s) <= n:
        return s
    return "...[truncated]..." + s[-(n - 17):]


_SITE_RE = re.compile(r"[^\s\"']*/(?:site|dist)-packages/")


def format_node_exception(exc: BaseException) -> str:
    """Traceback restricted to frames from the node source onwards (worker frames are dropped)."""
    frames = traceback.extract_tb(exc.__traceback__)
    idx = next((i for i, f in enumerate(frames) if os.path.basename(f.filename) == CODE_FILENAME), None)
    if idx is None:   # raised by the worker's own checks (or a syntax error): the message is enough
        return "".join(traceback.format_exception_only(type(exc), exc))
    frames = frames[idx:]
    # Keep every frame of the node's own source (the line the author must fix) plus the last few library frames;
    # the deep library middle only pushes the node frame out of the clipped tail the policy gets to see.
    keep = {i for i, f in enumerate(frames) if os.path.basename(f.filename) == CODE_FILENAME} | set(range(max(0, len(frames) - 3), len(frames)))
    lines = ["Traceback (most recent call last):\n"]
    prev = -1
    for i in sorted(keep):
        if i - prev > 1:
            lines.append(f"  ... [{i - prev - 1} library frame(s) omitted]\n")
        lines += traceback.format_list([frames[i]])
        prev = i
    lines += traceback.format_exception_only(type(exc), exc)
    return _SITE_RE.sub("<site>/", "".join(lines))


def _seed_all(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:  # numpy is optional for the worker itself
        return
    np.random.seed(seed % (2 ** 32))


_LOOPBACK = ("127.0.0.1", "localhost", "::1", "")


def _disable_network() -> None:
    """Refuse outbound INET/INET6 connections and DNS lookups from node code (loopback / AF_UNIX allowed)."""
    import socket

    def _guard(orig):  # noqa: ANN001, ANN202 - wraps socket.socket.connect / connect_ex
        def guarded(self, address, *args, **kwargs):  # noqa: ANN001 - socket API
            if self.family in (socket.AF_INET, socket.AF_INET6):
                host = address[0] if isinstance(address, tuple) and address else address
                if host not in _LOOPBACK:
                    raise PermissionError("network access is disabled inside code nodes")
            return orig(self, address, *args, **kwargs)
        return guarded

    socket.socket.connect = _guard(socket.socket.connect)
    socket.socket.connect_ex = _guard(socket.socket.connect_ex)
    orig_gai = socket.getaddrinfo

    def getaddrinfo(host, *args, **kwargs):  # noqa: ANN001 - socket API
        if host is not None and host not in _LOOPBACK:
            raise PermissionError("network access (DNS) is disabled inside code nodes")
        return orig_gai(host, *args, **kwargs)
    socket.getaddrinfo = getaddrinfo


def _write_json(path: str, obj: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, default=str)
    os.replace(tmp, path)


def run_spec(spec: dict) -> dict:
    """Execute one node spec and return the meta dict (also used in-process by tests)."""
    t0 = time.monotonic()
    meta: dict = {"status": "error", "error": None, "stdout_tail": "", "wall_s": 0.0, "extra_outputs": []}
    buf = _TailBuffer()
    try:
        config = spec.get("config") or {}
        seed = config.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, (int, float, str)):
            raise ValueError(f"config.seed must be an integer, got {seed!r}")
        _seed_all(int(seed))
        with open(spec["inputs_path"], "rb") as f:
            inputs = pickle.load(f)
        with open(spec["code_path"], encoding="utf-8") as f:
            source = f.read()
        _disable_network()
        blocker = _ImportBlocker()
        sys.meta_path.insert(0, blocker)
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = buf
        try:
            namespace: dict = {"__name__": "__node__", "__file__": CODE_FILENAME}
            exec(compile(source, CODE_FILENAME, "exec"), namespace)  # noqa: S102 - the sandboxed node itself
            fn = namespace.get("run")
            if not callable(fn):
                raise NameError("code must define a function run(inputs, config) returning a dict")
            result = fn(inputs, dict(config))
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            if blocker in sys.meta_path:
                sys.meta_path.remove(blocker)
        if not isinstance(result, dict):
            raise TypeError(f"run() must return a dict mapping output port names to values, got {type(result).__name__}")
        declared = list(spec.get("declared_outputs") or [])
        missing = [p for p in declared if p not in result]
        if missing:
            raise ValueError(f"run() returned no value for declared output port(s) {missing}; "
                           f"returned keys: {sorted(map(str, result))[:20]}")
        meta["extra_outputs"] = sorted(str(k) for k in result if k not in declared)[:20]
        outputs = {p: result[p] for p in declared}
        try:
            payload = pickle.dumps(outputs, protocol=PICKLE_PROTOCOL)
        except Exception as ex:  # noqa: BLE001 - reported as a node error with a clear message
            raise TypeError("output values must be picklable (numbers, strings, lists, dicts, numpy / pandas "
                            f"objects); pickling failed: {type(ex).__name__}: {ex}") from None
        tmp = spec["outputs_path"] + ".tmp"
        with open(tmp, "wb") as f:
            f.write(payload)
        os.replace(tmp, spec["outputs_path"])
        meta["status"] = "ok"
    except BaseException as ex:  # noqa: BLE001 - any failure of node code (incl. SystemExit) is reported
        meta["error"] = tail(format_node_exception(ex))
    meta["stdout_tail"] = tail(buf.getvalue())
    meta["wall_s"] = time.monotonic() - t0
    return meta


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m scienceclaw.runtime.node_worker <spec.json>", file=sys.stderr)
        return 2
    with open(argv[1], encoding="utf-8") as f:
        spec = json.load(f)
    meta = run_spec(spec)
    _write_json(spec["meta_path"], meta)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
