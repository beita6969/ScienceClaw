"""Local SWE-bench Verified OOD adapter for FoR46.

The legacy ``for46_code`` HumanEval/MBPP adapter is intentionally untouched.
This adapter consumes a local ``for46-swebench-verified/instances.jsonl``
delivery, exposes issue text plus repository inspection tools, and evaluates
one candidate unified diff per issue in a throw-away checkout. Gold patches,
hidden tests and evaluator commands never leave this module.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...core.schema import PortSchema
from ...core.trace import Trace
from ..task import Budget, ConstraintSpec, Episode, EvalResult, ToolSpec
from ._adapter_utils_for36_46_49_52 import as_str_list, check_split, draw_blocks, episode_id, episode_rng, resolve_data_root

CODE = "FoR46"
FAMILY = "Engineering & computing"
DATASET_DIR = "for46-swebench-verified"
DEFAULT_TIMEOUT_S = 600.0
DEFAULT_TEST_COMMAND = (sys.executable, "-m", "pytest", "-q")
MAX_PATCH_BYTES = 2 << 20
MAX_READ_BYTES = 256 << 10
MAX_TOTAL_READ_BYTES = 1 << 20


@dataclass(frozen=True)
class Instance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    version: str
    repo_path: Path
    test_patch: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    eval_command: tuple[str, ...] | None

    def public(self) -> dict[str, str]:
        # Evaluator-only fields are deliberately absent.
        return {"instance_id": self.instance_id, "repo": self.repo, "base_commit": self.base_commit,
                "problem_statement": self.problem_statement, "version": self.version}


@dataclass(frozen=True)
class _Data:
    instances: dict[str, Instance]
    ids: tuple[str, ...]


def _tests(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            value = value.splitlines()
    if not isinstance(value, (list, tuple)):
        raise ValueError("test list must be a list or JSON list")
    return tuple(str(x).strip() for x in value if str(x).strip())


def _command(value: Any) -> tuple[str, ...] | None:
    if value in (None, ""):
        return None
    out = tuple(shlex.split(value)) if isinstance(value, str) else tuple(str(x) for x in value)
    return out or None


def _relative(path: str) -> str:
    p = str(path).replace("\\", "/")
    if p in ("", "/dev/null", "dev/null"):
        return ""
    if p.startswith(("/", "~")):
        raise ValueError("absolute path")
    if p.startswith(("a/", "b/")):
        p = p[2:]
    parts = [x for x in p.split("/") if x not in ("", ".")]
    if not parts or any(x == ".." for x in parts):
        raise ValueError("path traversal")
    return "/".join(parts)


def _test_path(path: str) -> bool:
    parts = path.lower().split("/")
    name = parts[-1]
    return "tests" in parts or name.startswith("test_") or name.endswith("_test.py")


def patch_paths(patch: str, *, allow_tests: bool = False) -> tuple[bool, str, list[str]]:
    """Static patch checks before git sees candidate text."""
    if not isinstance(patch, str) or not patch.strip():
        return False, "empty patch", []
    if len(patch.encode("utf-8", "replace")) > MAX_PATCH_BYTES:
        return False, "patch is too large", []
    if "\x00" in patch or "GIT binary patch" in patch or "Binary files " in patch:
        return False, "binary patch", []
    paths: list[str] = []
    try:
        for line in patch.splitlines():
            if line.startswith("diff --git "):
                f = line.split()
                if len(f) < 4:
                    return False, "malformed diff header", []
                paths.extend(x for x in (_relative(f[2]), _relative(f[3])) if x)
            elif line.startswith(("--- ", "+++ ")):
                x = line[4:].split("\t", 1)[0].strip()
                if x != "/dev/null":
                    paths.append(_relative(x))
    except ValueError as ex:
        return False, str(ex), []
    if not paths:
        return False, "no changed paths", []
    unique = sorted(set(paths))
    if not allow_tests and any(_test_path(x) for x in unique):
        return False, "candidate patch modifies tests", unique
    return True, "ok", unique


def _load(root: Path) -> _Data:
    d = root / DATASET_DIR
    source = next((d / x for x in ("instances.jsonl", "instances.json") if (d / x).exists()), None)
    if source is None:
        raise FileNotFoundError(f"missing {d}/instances.jsonl")
    if source.suffix == ".json":
        raw = json.loads(source.read_text(encoding="utf-8"))
    else:
        raw = [json.loads(x) for x in source.read_text(encoding="utf-8").splitlines() if x.strip()]
    if isinstance(raw, dict):
        raw = raw.get("instances", raw.get("data", []))
    if not isinstance(raw, list) or not raw:
        raise ValueError("SWE-bench delivery is empty")
    root_real = d.resolve()
    out: dict[str, Instance] = {}
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError("instance row must be an object")
        iid, repo, base, statement = (str(row.get(k, "")).strip() for k in
                                      ("instance_id", "repo", "base_commit", "problem_statement"))
        if not iid or not repo or not base or not statement:
            raise ValueError("instance requires instance_id/repo/base_commit/problem_statement")
        if iid in out:
            raise ValueError(f"duplicate instance_id: {iid}")
        explicit = row.get("repo_path", row.get("snapshot_dir", row.get("repo_dir")))
        candidates = [d / str(explicit)] if explicit else [d / "repos" / repo, d / "snapshots" / iid]
        repo_path = next((p.resolve() for p in candidates if p.exists()), candidates[0].resolve())
        if not repo_path.is_relative_to(root_real):
            raise ValueError(f"repository path escapes dataset root: {iid}")
        out[iid] = Instance(iid, repo, base, statement, str(row.get("version", "")), repo_path,
                            str(row.get("test_patch", "") or ""),
                            _tests(row.get("FAIL_TO_PASS", row.get("fail_to_pass"))),
                            _tests(row.get("PASS_TO_PASS", row.get("pass_to_pass"))),
                            _command(row.get("eval_command", row.get("test_command"))))
    return _Data(out, tuple(sorted(out)))


def _env() -> dict[str, str]:
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/tmp", "TMPDIR": "/tmp",
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "LANG": "C.UTF-8",
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0", "NO_PROXY": "*",
            "http_proxy": "http://127.0.0.1:9", "https_proxy": "http://127.0.0.1:9",
            "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9"}


def _git(args: list[str], cwd: Path, *, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=str(cwd), env=_env(), input=input_text, text=True,
                          capture_output=True, timeout=180, check=False)


def _checkout(src: Path, commit: str, dst: Path) -> tuple[bool, str]:
    if not src.is_dir():
        return False, "repository snapshot is missing"
    try:
        clone = subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(src), str(dst)], env=_env(),
                               text=True, capture_output=True, timeout=180, check=False)
        if clone.returncode:
            return False, "repository snapshot is not a git checkout"
        r = _git(["checkout", "--quiet", "--detach", commit], dst)
        return (True, "ok") if r.returncode == 0 else (False, "base commit is unavailable")
    except (OSError, subprocess.SubprocessError) as ex:
        return False, f"checkout failed: {type(ex).__name__}"


def _apply(repo: Path, patch: str, *, allow_tests: bool) -> tuple[bool, str]:
    ok, why, _ = patch_paths(patch, allow_tests=allow_tests)
    if not ok:
        return False, why
    check = _git(["apply", "--check", "--whitespace=nowarn", "-"], repo, input_text=patch)
    if check.returncode:
        return False, "patch does not apply cleanly"
    run = _git(["apply", "--whitespace=nowarn", "-"], repo, input_text=patch)
    return (True, "ok") if run.returncode == 0 else (False, "patch application failed")


def _cmd(template: tuple[str, ...] | None, tests: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    expanded = False
    for token in list(template or DEFAULT_TEST_COMMAND):
        if token == "{tests}":
            out.extend(tests); expanded = True
        elif token in ("{test}", "{test_id}"):
            out.append(tests[0] if tests else ""); expanded = True
        else:
            out.append(token.replace("{instance_id}", ""))
    if tests and not expanded:
        out.extend(tests)
    return [x for x in out if x]


def _run(repo: Path, inst: Instance, tests: tuple[str, ...], timeout: float) -> bool:
    try:
        r = subprocess.run(_cmd(inst.eval_command, tests), cwd=str(repo), env=_env(), text=True,
                           capture_output=True, timeout=max(1.0, min(float(timeout), 3600.0)), check=False)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _one(inst: Instance, patch: str, timeout: float) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="scienceclaw-for46-swe-") as td:
        ok, why = _checkout(inst.repo_path, inst.base_commit, Path(td) / "repo")
        if not ok:
            return {"passed": False, "error": why}
        repo = Path(td) / "repo"
        ok, why = _apply(repo, patch, allow_tests=False)
        if not ok:
            return {"passed": False, "error": why}
        if inst.test_patch:
            ok, _ = _apply(repo, inst.test_patch, allow_tests=True)
            if not ok:
                return {"passed": False, "error": "evaluator test patch failed"}
        f2p, p2p = _run(repo, inst, inst.fail_to_pass, timeout), _run(repo, inst, inst.pass_to_pass, timeout)
        return {"passed": bool(f2p and p2p), "error": None if f2p and p2p else "test_failed"}


def _parallel(fn, args: list[Any]) -> list[Any]:
    if len(args) < 2:
        return [fn(x) for x in args]
    with ThreadPoolExecutor(max_workers=min(8, len(args))) as pool:
        return list(pool.map(fn, args))


class Adapter:
    """FoR46 SWE-bench Verified OOD adapter; legacy MBPP remains ``for46_code.Adapter``."""
    discipline = CODE
    name = "SWE-bench Verified (OOD)"
    family = FAMILY
    metric = "execution patch pass@1"
    direction = "max"
    task_type = "repository_patch"

    def __init__(self, data_root: str | os.PathLike | None = None, hidden_timeout_s: float = DEFAULT_TIMEOUT_S,
                 accept_margin: float = 0.0, budget: Budget | None = None) -> None:
        self.root = resolve_data_root(data_root)
        self.hidden_timeout_s, self.accept_margin, self.budget = float(hidden_timeout_s), float(accept_margin), budget
        self._data: _Data | None = None

    def _get(self) -> _Data:
        if self._data is None:
            self._data = _load(self.root)
        return self._data

    def available(self) -> tuple[bool, str]:
        try:
            d = self._get()
            missing = [i for i in d.ids if not d.instances[i].repo_path.is_dir()]
            return (False, f"missing repository snapshots: {missing[:3]}") if missing else (True, f"SWE-bench local instances: {len(d.ids)}")
        except (OSError, ValueError, json.JSONDecodeError) as ex:
            return False, f"FoR46 SWE-bench unreadable: {type(ex).__name__}: {ex}"

    def max_disjoint_episodes(self, split: str, items_per_episode: int = 16) -> int:
        check_split(split)
        if split != "ood":
            return 0
        ipe = int(items_per_episode)
        if ipe < 1:
            raise ValueError("items_per_episode must be >= 1")
        return len(self._get().ids) // ipe

    def build_episodes(self, split: str, n: int, seed: int, items_per_episode: int = 16) -> list[Episode]:
        check_split(split)
        if split != "ood":
            return []
        ipe, n = int(items_per_episode), int(n)
        if ipe < 1 or n < 0:
            raise ValueError("items_per_episode must be >= 1 and n >= 0")
        d = self._get()
        blocks = draw_blocks({"all": list(d.ids)}, {"all": ipe}, n, episode_rng(CODE, "swebench", seed), f"{CODE}/swebench/ood")
        return [self._episode(d, seed, k, ids) for k, ids in enumerate(blocks)]

    def pooled_metric(self, payloads: list[dict]) -> float | None:
        vals: list[int] = []
        for p in payloads:
            ids, passed = p.get("item_ids") or [], p.get("passed") or []
            if len(ids) != len(passed):
                raise ValueError("pooled payload item_ids/passed mismatch")
            vals.extend(int(bool(x)) for x in passed)
        return float(np.mean(vals)) if vals else None

    def _episode(self, d: _Data, seed: int, k: int, ids: list[str]) -> Episode:
        inst = [d.instances[x] for x in ids]; n = len(inst)
        def load(_i: dict, _c: dict) -> dict: return {"instances": [x.public() for x in inst]}
        def list_files(i: dict, _c: dict) -> dict:
            wanted = i.get("instance_ids", ids); out: dict[str, list[str]] = {}
            for iid in wanted:
                if iid not in d.instances: raise ValueError("unknown instance_id")
                root = d.instances[iid].repo_path
                out[iid] = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and ".git" not in p.parts and not p.is_symlink())[:5000]
            return {"files": out}
        def read_files(i: dict, _c: dict) -> dict:
            iid, paths = str(i.get("instance_id", "")), i.get("paths")
            if iid not in d.instances or not isinstance(paths, (list, tuple)): raise ValueError("instance_id/paths required")
            root = d.instances[iid].repo_path.resolve(); out: dict[str, str] = []; total = 0; result: dict[str, str] = {}
            for raw in paths:
                rel = _relative(str(raw)); p = (root / rel).resolve()
                if not p.is_relative_to(root) or not p.is_file() or p.is_symlink(): raise ValueError("file unavailable")
                size = p.stat().st_size
                if size > MAX_READ_BYTES or total + size > MAX_TOTAL_READ_BYTES: raise ValueError("read limit exceeded")
                result[rel] = p.read_text(encoding="utf-8", errors="replace"); total += size
            return {"files": result}
        def c_patch(y: Any, _t: Trace | None) -> tuple[bool, str]:
            v, why = as_str_list(y, n)
            if v is None: return False, why
            bad = [f"item {j}: {patch_paths(p)[1]}" for j, p in enumerate(v) if not patch_paths(p)[0]]
            return (not bad, "all patches are safe unified diffs" if not bad else "; ".join(bad[:3]))
        def evaluate(y: Any, _t: Trace | None) -> EvalResult:
            v, why = as_str_list(y, n); payload = {"item_ids": list(ids), "passed": [0] * n}
            if v is None: return EvalResult(metrics={"pass@1": 0.0, "n_passed": 0.0, "n_items": float(n)}, primary=0.0, accepted=False, details={"pooled_payload": payload, "invalid": why})
            result = _parallel(lambda x: _one(x[0], x[1], self.hidden_timeout_s), list(zip(inst, v)))
            passed = [bool(x["passed"]) for x in result]; score = float(np.mean(passed)) if passed else 0.0; payload["passed"] = [int(x) for x in passed]
            return EvalResult(metrics={"pass@1": score, "n_passed": float(sum(passed)), "n_items": float(n)}, primary=score, accepted=bool(score > self.accept_margin), details={"pooled_payload": payload, "per_item_error_kind": [None if x["passed"] else x.get("error", "failed") for x in result]})
        tools = [
            ToolSpec("load_eval_inputs", f"Returns {n} issue statements and repository revisions; hidden gold patches/tests are withheld.", {}, {"instances": PortSchema("list", (n,), dtype="dict")}, load),
            ToolSpec("list_repo_files", "Lists files in selected local repository snapshots.", {"instance_ids": PortSchema("list", dtype="str")}, {"files": PortSchema("dict", dtype="dict")}, list_files),
            ToolSpec("read_repo_files", "Reads repository-relative UTF-8 source files with a size limit.", {"instance_id": PortSchema("text"), "paths": PortSchema("list", dtype="str")}, {"files": PortSchema("dict", dtype="str")}, read_files),
        ]
        objective = f"Repair {n} SWE-bench repository issues. Inspect each repository with the tools and return one unified diff per issue. The evaluator applies it to the base commit, then applies hidden tests and runs FAIL_TO_PASS/PASS_TO_PASS. Do not modify tests or use network access."
        return Episode(id=episode_id(CODE, "swebench-ood", seed, k), discipline=CODE, family=FAMILY, split="ood", task_type=self.task_type, objective=objective, required_output=PortSchema("list", (n,), dtype="str"), tools=tools, constraints=[ConstraintSpec("patch_format", "every output is a safe unified diff and does not modify tests", c_patch)], budget=self.budget or Budget(max_steps=16, max_policy_tokens=300_000, max_wall_s=7200.0, max_node_s=900.0, max_llm_items=4*n), lineage={"dataset": "SWE-bench Verified", "pool": "ood", "ood_kind": "cross_dataset", "item_ids": list(ids), "seed": seed, "index": k, "source": "https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified"}, acceptance=f"pass@1 >= {self.accept_margin:g}", tolerance={"rtol": 0.0, "atol": 0.0}, tags=["code", "repository_patch", "swebench", "unit_tests", "llm"], metric=self.metric, direction=self.direction, n_items=n, _evaluate=evaluate)
