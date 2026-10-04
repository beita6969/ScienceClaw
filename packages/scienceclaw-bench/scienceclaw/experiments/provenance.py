"""Run provenance receipts and the append-only usage ledger (DESIGN §8.6 "receipts").

Every phase of a run (evolve, each resume segment, evaluate, family transfer, report) writes a *provenance receipt*
so that any number in a report can be traced to the exact code, configuration, data and model settings that produced
it (review findings m3+m4 and M6):

* ``code``     git commit, dirty flag and a sha256 of the working-tree diff (tracked changes + untracked files),
               python / platform and the versions of the packages the pipeline imports;
* ``config``   sha256 of the canonical RunConfig JSON (+ ``bench`` seed / items per episode);
* ``data``     sha256 of the dataset manifests under ``<data_root>/../manifests`` and of every per-dataset
               ``receipt.json``;
* ``llm``      role -> model / decoding settings, gateway host (never credentials), concurrency, cache path and the
               cache-salt policy (per-call salts are set by the callers, e.g. ``seed=<n>`` for executor nodes);
* ``segment``  phase name, start time and process id.

``UsageLedger`` is the append-only ``usage.jsonl``: one record per (process segment, phase) holding start/end
time and the *delta* of ``llm.usage()`` since the segment began, so a resumed run accumulates instead of overwriting
``usage.json``. ``sum_ledger`` adds the records up.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

PACKAGES = ("numpy", "scipy", "pandas", "scikit-learn", "pyyaml", "torch", "xarray", "zarr", "rdkit", "ogb",
            "librosa", "soundfile", "nibabel", "h5py", "pyarrow", "networkx", "statsmodels", "lightgbm", "xgboost")
MANIFEST_FILES = ("dataset_manifest.json", "local-files.sha256.jsonl", "reconstruction-policy.json",
                  "downloaded-scope.json")
_PKG_ROOT = Path(__file__).resolve().parents[2]

_hash_cache: dict[tuple[str, int, int], str] = {}
_hash_lock = threading.Lock()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str | Path) -> str | None:
    """sha256 of a file (cached by path/size/mtime); None if unreadable."""
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return None
    key = (str(p), st.st_size, int(st.st_mtime_ns))
    with _hash_lock:
        hit = _hash_cache.get(key)
    if hit is not None:
        return hit
    h = hashlib.sha256()
    try:
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    with _hash_lock:
        _hash_cache[key] = h.hexdigest()
    return h.hexdigest()


def _git(args: list[str], root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=20)


def code_version(root: str | Path | None = None) -> dict:
    """Code provenance: git commit, dirty flag, sha256 of the diff and package versions (best effort, never raises)."""
    root = Path(root) if root is not None else _PKG_ROOT
    info: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform(),
                            "packages": package_versions()}
    try:
        commit = _git(["rev-parse", "HEAD"], root)
        if commit.returncode != 0:
            info["git_error"] = commit.stderr.strip()[:200]
            return info
        info["git_commit"] = commit.stdout.strip()
        status = _git(["status", "--porcelain"], root).stdout
        info["git_dirty"] = bool(status.strip())
        if info["git_dirty"]:
            diff = _git(["diff", "HEAD", "--binary"], root).stdout
            h = hashlib.sha256(diff.encode("utf-8", "replace"))
            untracked = sorted(line[3:] for line in status.splitlines() if line.startswith("??"))
            for rel in untracked:
                h.update(rel.encode())
                fp = sha256_file(root / rel)
                h.update((fp or "?").encode())
            info["git_diff_sha256"] = h.hexdigest()
            info["git_dirty_files"] = sorted(line[3:] for line in status.splitlines())[:50]
    except (OSError, subprocess.SubprocessError) as ex:
        info["git_error"] = f"{type(ex).__name__}: {ex}"
    return info


def package_versions(names: Iterable[str] = PACKAGES) -> dict[str, str]:
    """Installed versions of the packages the pipeline may import (absent ones are omitted)."""
    from importlib import metadata

    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            continue
        except Exception:  # pragma: no cover - broken dist-info must not break a run
            continue
    return out


def config_hash(cfg: Any) -> str:
    """sha256 of the canonical JSON of a RunConfig (or any dataclass/dict)."""
    d = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    return sha256_bytes(canonical_json(d).encode())


def dataset_hashes(data_root: str | Path | None) -> dict:
    """sha256 of the dataset manifests next to ``data_root`` and of every ``<dataset>/receipt.json``."""
    out: dict[str, Any] = {"data_root": str(data_root) if data_root else None, "manifests": {}, "receipts": {}}
    if not data_root:
        return out
    root = Path(data_root).expanduser()
    mdir = root.parent / "manifests"
    for name in MANIFEST_FILES:
        h = sha256_file(mdir / name)
        if h is not None:
            out["manifests"][name] = h
    try:
        for d in sorted(root.iterdir()):
            h = sha256_file(d / "receipt.json") if d.is_dir() else None
            if h is not None:
                out["receipts"][d.name] = h
    except OSError:
        pass
    return out


def llm_receipt(cfg: Any, llm: Any = None) -> dict:
    """LLM settings that determine outputs: roles, gateway host, concurrency, cache path and salt policy."""
    from urllib.parse import urlparse

    lc = getattr(cfg, "llm", cfg)
    roles = {}
    for r in ("policy", "executor", "patch"):
        m = getattr(lc, r, None)
        if m is not None:
            roles[r] = {"model": m.model, "max_tokens": m.max_tokens, "temperature": m.temperature,
                        "reasoning_effort": m.reasoning_effort, "json_mode": m.json_mode}
    host = urlparse(getattr(lc, "base_url", "") or "").netloc or None
    cache = getattr(llm, "cache_path", None)
    return {"roles": roles, "gateway_host": host, "concurrency": getattr(lc, "concurrency", None),
            "use_cache": getattr(lc, "use_cache", None),
            "cache_path": str(cache) if cache is not None else (getattr(lc, "cache_path", None) or None),
            "cache_salt": "per-call (executor nodes: 'seed=<node seed>'; policy/patch: caller-supplied, default '')",
            "client": type(llm).__name__ if llm is not None else None}


def provenance(cfg: Any, phase: str, *, llm: Any = None, extra: Mapping[str, Any] | None = None) -> dict:
    """The provenance receipt of one phase / process segment (JSON-safe)."""
    bench = getattr(cfg, "bench", None)
    rec: dict[str, Any] = {
        "phase": phase, "at": datetime.now().isoformat(timespec="seconds"), "pid": os.getpid(),
        "code": code_version(), "config_sha256": config_hash(cfg) if cfg is not None else None,
        "data": dataset_hashes(getattr(bench, "data_root", None)),
        "llm": llm_receipt(cfg, llm) if cfg is not None else {},
    }
    if bench is not None:
        rec["bench"] = {"seed": bench.seed, "items_per_episode": bench.items_per_episode,
                        "disciplines": list(bench.disciplines)}
    if extra:
        rec.update(extra)
    return json.loads(canonical_json(rec))


# ------------------------------------------------------------------------------------------------ usage ledger
_NUM = (int, float)


def _flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, bool):
            continue
        if isinstance(v, _NUM):
            out[key] = float(v)
        elif isinstance(v, Mapping):
            out.update(_flatten(v, key + "."))
    return out


def usage_delta(now: Mapping[str, Any], before: Mapping[str, Any]) -> dict:
    """``now - before`` for the nested numeric ``llm.usage()`` snapshots (keys absent before count as 0)."""
    def sub(a: Any, b: Any) -> Any:
        if isinstance(a, Mapping):
            bb = b if isinstance(b, Mapping) else {}
            return {k: sub(v, bb.get(k)) for k, v in a.items()}
        if isinstance(a, bool) or not isinstance(a, _NUM):
            return a
        return a - (b if isinstance(b, _NUM) and not isinstance(b, bool) else 0)

    return sub(dict(now), dict(before))


def add_usage_trees(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, Mapping):
            out[k] = add_usage_trees(out.get(k, {}) if isinstance(out.get(k), Mapping) else {}, v)
        elif isinstance(v, _NUM) and not isinstance(v, bool):
            cur = out.get(k)
            out[k] = (cur if isinstance(cur, _NUM) and not isinstance(cur, bool) else 0) + v
        else:
            out.setdefault(k, v)
    return out


class UsageLedger:
    """Append-only ``usage.jsonl``; one record per (process segment, phase).

    Use as a context manager around a phase::

        with UsageLedger(run_dir, "evolve", llm, prov) as led:
            ...work...
        # -> {"segment": "...", "phase": "evolve", "start": ..., "end": ..., "wall_s": ..., "llm": <delta>, ...}

    A ``status="started"`` marker is appended on entry and the closing record (``done`` / ``failed``) on exit, also
    on exceptions. A process that is killed (OOM, SIGKILL, wall limit) leaves the marker without a closing record:
    its spend is unknowable from the ledger, so :func:`sum_ledger` lists it under ``unclosed`` and the report warns
    that the totals are a lower bound. A resume adds its own segment instead of overwriting anything.
    """

    FILE = "usage.jsonl"

    def __init__(self, run_dir: str | Path, phase: str, llm: Any = None, prov: Mapping[str, Any] | None = None,
                 extra: Mapping[str, Any] | None = None) -> None:
        self.path = Path(run_dir) / self.FILE
        self.phase, self.llm, self.prov, self.extra = phase, llm, prov, dict(extra or {})
        self._before: dict = {}
        self.record: dict | None = None
        self._t0 = 0.0
        self._start = ""
        self._uid = uuid.uuid4().hex[:6]

    def _usage(self) -> dict:
        try:
            return dict(self.llm.usage()) if self.llm is not None and hasattr(self.llm, "usage") else {}
        except Exception:  # pragma: no cover
            return {}

    @property
    def segment(self) -> str:
        return f"{self._start}#{os.getpid()}#{self._uid}"

    def _append(self, rec: dict) -> None:
        line = json.dumps(rec, default=str, sort_keys=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(line + "\n")
            f.flush()

    def __enter__(self) -> "UsageLedger":
        self._before = self._usage()
        self._t0 = time.time()
        self._start = datetime.now().isoformat(timespec="seconds")
        self._append({"segment": self.segment, "phase": self.phase, "status": "started", "start": self._start,
                      "provenance": self.prov, **self.extra})
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.write("failed" if exc_type is not None else "done", error=None if exc is None else
                   f"{exc_type.__name__}: {exc}")

    def write(self, status: str = "done", error: str | None = None) -> dict:
        wall = time.time() - self._t0
        rec = {"segment": self.segment, "phase": self.phase, "status": status, "start": self._start,
               "end": datetime.now().isoformat(timespec="seconds"), "wall_s": wall,
               "llm": usage_delta(self._usage(), self._before), "provenance": self.prov, **self.extra}
        if error:
            rec["error"] = error
        self._append(rec)
        self.record = rec
        return rec


def read_ledger(path: str | Path) -> list[dict]:
    """Read ``usage.jsonl`` (tolerates a torn last line)."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    lines = p.read_text().splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                continue
            raise
    return out


def sum_ledger(records: Iterable[Mapping[str, Any]]) -> dict:
    """Sum a ledger: ``{"llm": <summed usage tree>, "wall_s": ..., "segments": n, "by_phase": {phase: {...}},
    "unclosed": [segment, ...], "failed": [segment, ...]}``.

    ``started`` markers without a closing record (killed processes) are listed under ``unclosed``; their spend is
    not in the sums (lower bound). Closed records with ``status="failed"`` are summed (their spend is real).
    """
    total: dict = {}
    wall = 0.0
    by_phase: dict[str, dict] = {}
    n = 0
    records = list(records)
    closed = {r.get("segment") for r in records if r.get("status") != "started"}
    unclosed = [r.get("segment") for r in records if r.get("status") == "started" and r.get("segment") not in closed]
    failed = [r.get("segment") for r in records if r.get("status") == "failed"]
    for r in records:
        if r.get("status") == "started":
            continue
        n += 1
        llm = r.get("llm") or {}
        total = add_usage_trees(total, llm)
        wall += float(r.get("wall_s") or 0.0)
        ph = by_phase.setdefault(str(r.get("phase")), {"segments": 0, "wall_s": 0.0, "llm": {}})
        ph["segments"] += 1
        ph["wall_s"] += float(r.get("wall_s") or 0.0)
        ph["llm"] = add_usage_trees(ph["llm"], llm)
    return {"llm": total, "wall_s": wall, "segments": n, "by_phase": by_phase, "unclosed": unclosed, "failed": failed}
