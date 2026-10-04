"""Subprocess sandbox for ``code`` nodes.

``run_code_node(node, input_values, run_dir, timeout_s)`` writes the node source, pickled inputs and
a JSON spec into a fresh per-node working directory ``<run_dir>/work/<node>-<uid>/`` and launches
``python -m scienceclaw.runtime.node_worker <spec.json>`` with

* ``cwd`` = the working directory, ``HOME`` and ``TMPDIR`` inside it;
* a minimal allow-listed environment (no API keys / tokens / credentials variables), with
  ``PYTHONHASHSEED=0`` and 2 BLAS/OpenMP threads for reproducible, bounded numerics, HTTP(S)
  proxies pointing at a dead local port and data/model hubs in offline mode;
* a hard timeout: the whole process group is killed when ``timeout_s`` elapses.

A process-wide :class:`threading.BoundedSemaphore` (default 6, env ``SCIENCECLAW_MAX_SUBPROCS``)
limits the number of concurrently running workers across all executors / threads.
"""
from __future__ import annotations

import contextlib
import json
import os
import pickle
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .values import load_value, save_value

DEFAULT_MAX_SUBPROCS = int(os.environ.get("SCIENCECLAW_MAX_SUBPROCS", "6"))
TAIL_CHARS = 4000
WORKER_MODULE = "scienceclaw.runtime.node_worker"
_ENV_PASSTHROUGH = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "SYSTEMROOT",
                    # locally staged pretrained weights (scilib/_pretrained.py), their off switches and GPU visibility
                    "SCIENCECLAW_MODELS", "SCIENCECLAW_NO_PRETRAINED", "SCIENCECLAW_NO_SCILIB", "CUDA_VISIBLE_DEVICES",
                    # spool directory served by the GPU-host broker (scilib/_remote.py)
                    "SCIENCECLAW_REMOTE_SPOOL",
                    # immutable, content-addressed frozen MLIP feature cache
                    "SCIENCECLAW_MLIP_CACHE")
_SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSW|CREDENTIAL|AUTH|COOKIE", re.I)
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]")
_PACKAGE_PARENT = str(Path(__file__).resolve().parents[2])
_DEAD_PROXY = "http://127.0.0.1:9"     # discard port: connections fail immediately

_sem_lock = threading.Lock()
_semaphore = threading.BoundedSemaphore(max(1, DEFAULT_MAX_SUBPROCS))
_max_subprocs = max(1, DEFAULT_MAX_SUBPROCS)


def set_max_concurrent_subprocesses(n: int) -> None:
    """Replace the global concurrency limit (workers already running keep their old slot)."""
    global _semaphore, _max_subprocs
    with _sem_lock:
        _max_subprocs = max(1, int(n))
        _semaphore = threading.BoundedSemaphore(_max_subprocs)


def max_concurrent_subprocesses() -> int:
    return _max_subprocs


def _current_semaphore() -> threading.BoundedSemaphore:
    with _sem_lock:
        return _semaphore


def _tail(s: str, n: int = TAIL_CHARS) -> str:
    return s if len(s) <= n else "...[truncated]..." + s[-(n - 17):]


def _read_tail(path: Path, n: int = TAIL_CHARS) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 4 * n))
        return _tail(f.read().decode("utf-8", errors="replace"), n)


def build_env(work_dir: Path) -> dict[str, str]:
    """Allow-listed environment for a worker: nothing credential-like is inherited."""
    env = {k: os.environ[k] for k in _ENV_PASSTHROUGH if k in os.environ and not _SECRET_NAME.search(k)}
    tmp = work_dir / "tmp"
    env.update({
        "HOME": str(work_dir), "TMPDIR": str(tmp), "TMP": str(tmp), "TEMP": str(tmp),
        "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
        "PYTHONIOENCODING": "utf-8", "PYTHONPATH": _PACKAGE_PARENT, "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2", "VECLIB_MAXIMUM_THREADS": "2",
        # defense in depth against network access: dead proxy for HTTP clients, offline model/data hubs
        "HTTP_PROXY": _DEAD_PROXY, "HTTPS_PROXY": _DEAD_PROXY, "ALL_PROXY": _DEAD_PROXY, "http_proxy": _DEAD_PROXY,
        "https_proxy": _DEAD_PROXY, "all_proxy": _DEAD_PROXY, "NO_PROXY": "", "no_proxy": "",
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    })
    hf = Path(env["SCIENCECLAW_MODELS"]) / "hf" if "SCIENCECLAW_MODELS" in env else None
    if hf is not None and hf.is_dir():
        env["HF_HOME"] = str(hf)
    return env


_protected: list[str] = []


def protect(path: str | Path) -> None:
    """Make a directory (run receipts, evaluator payloads) unreadable to code nodes, except their own work directory."""
    resolved = str(Path(path).expanduser().resolve())
    if resolved not in _protected:
        _protected.append(resolved)


def protected_paths() -> list[str]:
    """Locations code nodes must not read or write: benchmark data, evaluator payloads and runs, engine state, credentials."""
    home = Path.home()
    out = [os.environ.get("SCIENCECLAW_DATA_ROOT"), os.environ.get("SCIENCECLAW_RUN_ROOT"), os.environ.get("SCIENCECLAW_HOME"),
           str(home / ".cache" / "scienceclaw" / "datasets"), str(home / ".config" / "scienceclaw"), str(home / ".scienceclaw"),
           str(home / ".ssh"), str(home / ".aws"), str(home / ".gnupg")]
    return [p for p in out if p] + list(_protected)


_isolation_prefix: list[str] | None = None


def isolation_prefix() -> list[str]:
    """Command prefix that runs a worker in new user, network and pid namespaces (no network at all, no view of other processes).

    ``SCIENCECLAW_SANDBOX_ISOLATION``: ``auto`` (default) uses it when ``unshare`` works here, ``off`` never, ``require`` fails
    when it is unavailable. Namespaces drop supplementary groups, so files readable only through a group are not visible.
    """
    global _isolation_prefix
    mode = os.environ.get("SCIENCECLAW_SANDBOX_ISOLATION", "auto")
    if mode == "off":
        return []
    if _isolation_prefix is None:
        prefix = ["unshare", "--user", "--map-root-user", "--net", "--pid", "--fork", "--mount-proc", "--kill-child"]
        try:
            ok = subprocess.run(prefix + ["true"], capture_output=True, timeout=20).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            ok = False
        _isolation_prefix = prefix if ok else []
    if mode == "require" and not _isolation_prefix:
        raise RuntimeError("SCIENCECLAW_SANDBOX_ISOLATION=require but user/net/pid namespaces are not available here")
    return list(_isolation_prefix)


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        # the group is already gone (or not ours to signal): fall back to the direct child
        with contextlib.suppress(ProcessLookupError):
            proc.kill()


def run_code_node(node: Any, input_values: dict, run_dir: str | Path, timeout_s: float, *,
                  keep_io: bool = False) -> tuple[dict | None, dict]:
    """Run one ``code`` node in a subprocess worker.

    Returns ``(outputs, meta)``: ``outputs`` maps every declared output port to its value (None on
    failure); ``meta`` has ``status`` ("ok" | "error" | "timeout"), ``error``, ``stdout_tail``,
    ``wall_s``, ``returncode``, ``timed_out`` and ``work_dir``. Input/output pickles in the working
    directory are removed afterwards unless ``keep_io`` (the executor keeps its own copies).
    """
    safe = _SAFE_NAME.sub("_", str(node.id))[:40] or "node"
    # absolute: the worker runs with cwd=work, so relative run dirs would break every path in the spec
    work = Path(run_dir).resolve() / "work" / f"{safe}-{uuid.uuid4().hex[:8]}"
    (work / "tmp").mkdir(parents=True, exist_ok=False)
    code_path = work / "node_code.py"
    code_path.write_text(node.code or "", encoding="utf-8")
    in_path, out_path, meta_path = work / "_inputs.pkl", work / "_outputs.pkl", work / "_meta.json"
    save_value(dict(input_values), in_path)
    spec = {"node_id": str(node.id), "code_path": str(code_path), "inputs_path": str(in_path),
            "outputs_path": str(out_path), "meta_path": str(meta_path), "config": dict(node.config or {}),
            "declared_outputs": list(node.outputs), "deny_paths": protected_paths(), "allow_paths": [str(work)]}
    spec_path = work / "_spec.json"
    spec_path.write_text(json.dumps(spec, default=str), encoding="utf-8")
    stdout_path, stderr_path = work / "_stdout.txt", work / "_stderr.txt"

    timed_out = False
    sem = _current_semaphore()
    with sem:
        t0 = time.monotonic()
        with open(stdout_path, "wb") as out_f, open(stderr_path, "wb") as err_f:
            proc = subprocess.Popen(
                [*isolation_prefix(), sys.executable, "-m", WORKER_MODULE, str(spec_path)], cwd=str(work), env=build_env(work),
                stdin=subprocess.DEVNULL, stdout=out_f, stderr=err_f, start_new_session=True)
            try:
                rc = proc.wait(timeout=max(0.1, float(timeout_s)))
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_group(proc)
                rc = proc.wait()
        wall = time.monotonic() - t0

    meta: dict[str, Any] = {"status": "error", "error": None, "stdout_tail": "", "wall_s": wall,
                            "returncode": rc, "timed_out": timed_out, "work_dir": str(work)}
    fd_out = _read_tail(stdout_path)
    fd_err = _read_tail(stderr_path)
    outputs: dict | None = None
    if timed_out:
        meta["status"] = "timeout"
        meta["error"] = f"timeout: node exceeded {float(timeout_s):.1f} s and was killed"
        meta["stdout_tail"] = _tail(fd_out + fd_err)
    elif not meta_path.exists():
        meta["error"] = _tail(f"worker exited with code {rc} without writing results\n{fd_err}")
        meta["stdout_tail"] = _tail(fd_out)
    else:
        wmeta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.update({k: wmeta.get(k) for k in ("status", "error", "extra_outputs")})
        extra = (fd_out + fd_err).strip()   # output written at file-descriptor level (C extensions, warnings)
        meta["stdout_tail"] = _tail((wmeta.get("stdout_tail") or "") + (("\n" + extra) if extra else ""))
        meta["worker_wall_s"] = wmeta.get("wall_s")
        if meta["status"] == "ok":
            try:
                outputs = load_value(out_path)
            except (OSError, EOFError, ValueError, ImportError, AttributeError, pickle.UnpicklingError) as ex:
                meta["status"] = "error"
                meta["error"] = f"could not load node outputs: {type(ex).__name__}: {ex}"
    if not keep_io:
        for p in (in_path, out_path):
            p.unlink(missing_ok=True)   # temporary copies created above; values are stored by the executor
    return outputs, meta
