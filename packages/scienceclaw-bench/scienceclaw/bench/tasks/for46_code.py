"""FoR46 Information and computing sciences — HumanEval -> MBPP program synthesis, metric execution pass@1 (max).

Data (data-team receipts under ``<DATA_ROOT>/for46-humaneval-mbpp``): OpenAI HumanEval (``HumanEval.jsonl.gz``,
164 problems, openai/human-eval commit 6d43fb9, MIT) and Google MBPP sanitized (``sanitized-mbpp.json``, 427
problems, google-research commit f46ca83, CC-BY-4.0), plus the data team's normative role assignment
``reconstructed_v2/sampling-manifest.json`` (seed 20260928, ``partitions[role].selected_ids``; the per-role
``*.jsonl`` files are not read at run time, the problems come from the two raw dataset files and the roles only
choose ids). ``reconstructed_v1`` is the fallback when v2 is absent, and a sha256 ranking of the native ids the
last resort (both documented in ``docs/tasks/FoR46.md``).

* **Item** = one Python programming problem with hidden unit tests. HumanEval: prompt = signature + docstring,
  hidden test = the official ``check(candidate)`` function. MBPP (sanitized): prompt = natural-language task,
  hidden tests = all ``test_list`` asserts (with ``test_imports``).
* **Pools (v2 roles).** IID = HumanEval, OOD = MBPP (a genuinely different dataset, ``ood_kind = "cross_dataset"``):
  ``src`` = ``source_train`` (36 HumanEval), ``val`` = ``validation`` (64), ``id`` = ``heldout_id`` (64),
  ``ood`` = ``heldout_ood`` (64 MBPP sanitized, official test range 11-510) followed, if more than 4 OOD episodes
  of 16 are requested, by the ``heldout_extra`` reserve (64 further official-test MBPP problems). The manifest's
  ``replication`` role is D_rep = the already seen source prefix (NOT new items); it is not a pool here (the
  framework's ``rep`` episodes are frozen copies of ``src`` episodes). Maximum item-disjoint episodes of 16:
  val 4, id 4, ood 4 (+4 from the reserve = 8), src 2 per cycle. Episodes are drawn with
  :func:`draw_blocks` from a seeded permutation, so they are deterministic and prefix-stable; the OOD reserve is a
  second layer with its own seeded stream, so adding OOD episodes never changes the first ones. The 36 source
  problems cannot supply 7 disjoint episodes of 16 (2 per cycle), so ``src`` recycles: a fresh seeded permutation
  starts after every 2 episodes (items recur across *source* episodes only, never inside one episode and never
  in val/id/ood; ``lineage.src_cycle`` tells which pass an episode belongs to).
* **Visible data (D_E).** ``load_eval_inputs`` returns the problems (prompt, entry point, visible tests).
  Visible tests: HumanEval = the input/output examples written in the docstring (doctest ``>>>`` examples and
  ``f(x) ➞ y`` / ``==`` / ``=>`` / ``returns`` lines whose arguments and result are Python literals); MBPP = the
  first ``test_list`` assert (the EvalPlus MBPP+ prompt convention). ``run_visible_tests`` runs candidate code
  against the visible tests only, in a subprocess with a timeout. There is no labelled training set: the visible
  dev signal is the visible-test pass rate (``Episode._dev_evaluate``), which uses no hidden test. Canonical
  solutions and hidden tests never leave the adapter (``Problem.public`` is the only thing exposed).
* **Metric (D_V).** pass@1 with one sample per problem, as in ``human_eval.evaluation`` (Chen et al. 2021):
  a problem is solved iff its program runs all hidden tests without exception within the time limit. HumanEval
  program = ``prompt + y[i] + test + "check(entry_point)"`` (the official concatenation; a complete function
  definition in ``y[i]`` overrides the docstring-only stub); MBPP program = ``y[i] + test_imports + asserts``.
  The data team's official scorer (``estimate_pass_at_k(n=1, k=1)`` over native assertions, 3 s native timeout)
  gives the same per-problem verdicts; here the timeout is 6 s per statement (see the sandbox paragraph).
* **Sandbox.** Hidden tests run in a fresh ``python -I`` subprocess per problem (own session, minimal environment,
  dead proxy variables, RLIMIT_CPU / RLIMIT_FSIZE, SIGALRM per statement, wall-clock kill of the process group,
  destructive os/shutil/subprocess functions disabled as in the official ``reliability_guard``, socket connect /
  bind / DNS functions disabled in-process). On macOS the subprocess additionally runs under ``sandbox-exec`` with
  ``(deny network*)`` and file writes limited to its two private temp directories, when a start-up probe shows the
  OS boundary is enforceable (nested/unsupported hosts fall back silently; ``SCIENCECLAW_FOR46_SEATBELT=0``
  disables it). Best-effort isolation of LLM-written code, not a hard security boundary.
* **Reference baseline** (deterministic, visible data only): the *visible-example memorizer* — a function that
  returns the documented output for argument tuples that occur in the visible examples and ``None`` otherwise.
  **Acceptance:** ``pass@1 >= reference + 0.5``. ``norm_score = pass@1 / max(reference, 0.1)`` clipped to
  [0, 10] (the floor keeps the ratio informative when the memorizer solves nothing: then norm = 10 * pass@1).
* **Hard constraints:** ``output_format`` (list of n strings) and ``implemented`` (every assembled program is
  valid Python whose last top-level definition of the entry point has a body beyond a docstring).
"""
from __future__ import annotations

import ast
import doctest
import gzip
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import scilib
from ...core.schema import PortSchema
from ...core.trace import Trace
from ..task import Budget, ConstraintSpec, EvalResult, Episode, ToolSpec
from ._adapter_utils_for36_46_49_52 import (
    PARTITION_SEED, Lazy, PoolExhausted, as_str_list, atomic_write_json, c_str_list, cache_dir, check_split,
    draw_blocks, episode_id, episode_rng, hash_rank, norm_score, read_json, receipt_summary, resolve_data_root,
)

CODE = "FoR46"
FAMILY = "Engineering & computing"
DATASET_DIR = "for46-humaneval-mbpp"
ACCEPT_MARGIN = 0.5
NORM_FLOOR = 0.1
HIDDEN_TIMEOUT_S = 6.0
VISIBLE_TIMEOUT_S = 5.0
MAX_WORKERS = max(1, min(8, (os.cpu_count() or 4) - 2))
_CACHE_VERSION = "v1"        # keys: problem ids; contents depend only on the raw datasets, not on the role files
_ERR_CHARS = 300
ROLES_V2 = "reconstructed_v2"
ROLES_V1 = "reconstructed_v1"
ROLES_HASH = "hash_rank"
MBPP_TEST_RANGE = (11, 510)  # MBPP official test partition (native ids), the source of every v2 OOD problem


# ================================================================================================ problems
@dataclass
class Problem:
    item_id: str                    # "HumanEval/12" or "MBPP/590"
    dataset: str                    # "HumanEval" | "MBPP"
    prompt: str                     # HumanEval: code stub; MBPP: natural-language task
    entry_point: str
    visible_tests: list[str]        # assert statements shown to the agent
    visible_setup: str              # code executed before the visible tests (MBPP test_imports)
    memo_pairs: list[tuple[str, str, str]] = field(default_factory=list)  # (repr(args), repr(expected), test)
    hidden_test: str = field(default="", repr=False)                  # never exposed
    canonical: str = field(default="", repr=False)                    # reference solution (never exposed)

    def public(self) -> dict:
        return {"prompt": self.prompt, "entry_point": self.entry_point,
                "visible_tests": "\n".join(self.visible_tests),
                "kind": "python_function_stub" if self.dataset == "HumanEval" else "task_description"}


def _parse(src: str, mode: str = "exec") -> ast.AST:
    """``ast.parse`` without SyntaxWarnings (dataset code contains e.g. non-raw regex strings)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(src, mode=mode)


def _balanced_call(line: str, start: int) -> int | None:
    """Index just past the ')' closing the call whose '(' is at ``start`` (strings respected), else None."""
    depth = 0
    quote: str | None = None
    i = start
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def _literal(src: str) -> tuple[bool, Any]:
    try:
        return True, ast.literal_eval(src)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return False, None


def _call_args(call_src: str, entry: str) -> str | None:
    """repr of the literal positional-argument tuple of ``entry(...)``; None if not a pure-literal call."""
    try:
        node = _parse(call_src.strip(), mode="eval").body
    except SyntaxError:
        return None
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == entry
            and not node.keywords):
        return None
    vals = []
    for a in node.args:
        ok, v = _literal(ast.unparse(a))
        if not ok:
            return None
        vals.append(v)
    return repr(tuple(vals))


_SEP = re.compile(r"^\s*(?:#\s*)?(?:➞|==>|=>|->|==|=|should\s+return|returns|return)\s*(.+)$")


def _parse_expected(rest: str) -> tuple[bool, Any, str]:
    rest = rest.split("#", 1)[0].strip() if not rest.lstrip().startswith(("'", '"')) else rest.strip()
    rest = rest.rstrip().rstrip(".").strip()
    for cand in (rest, {"true": "True", "false": "False", "none": "None"}.get(rest.lower(), rest)):
        ok, v = _literal(cand)
        if ok:
            return True, v, repr(v)
    return False, None, ""


def humaneval_visible(prompt: str, entry: str) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Visible tests (assert statements) and memorizer pairs from the docstring of the entry point."""
    try:
        tree = _parse(prompt)
    except SyntaxError:
        return [], []
    doc = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == entry:
            doc = ast.get_docstring(node, clean=True)
    if not doc:
        return [], []
    tests: list[str] = []
    pairs: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    def add(call_src: str, exp_repr: str) -> None:
        t = f"assert {call_src} == {exp_repr}"
        if t in seen:
            return
        try:
            _parse(t)
        except SyntaxError:
            return
        seen.add(t)
        tests.append(t)
        args = _call_args(call_src, entry)
        if args is not None:
            pairs.append((args, exp_repr, t))

    try:
        examples = doctest.DocTestParser().get_examples(doc)
    except ValueError:
        examples = []
    for ex in examples:
        src = ex.source.strip()
        try:
            _parse(src, mode="eval")
        except SyntaxError:
            continue
        if f"{entry}(" not in src:
            continue
        ok, _v, rep = _parse_expected(ex.want.strip())
        if ok and ex.want.strip():
            add(src, rep)
    if not examples:
        for raw in doc.splitlines():
            line = raw.strip()
            k = line.find(f"{entry}(")
            if k < 0 or line.startswith(("def ", ">>>")):
                continue
            if k > 0 and (line[k - 1].isalnum() or line[k - 1] == "_"):
                continue
            end = _balanced_call(line, k + len(entry))
            if end is None:
                continue
            call_src = line[k:end]
            m = _SEP.match(line[end:])
            if not m:
                continue
            ok, _v, rep = _parse_expected(m.group(1))
            if not ok:
                continue
            if _call_args(call_src, entry) is None:      # arguments must be python literals
                continue
            add(call_src, rep)
    return tests, pairs


def _mbpp_entry(code: str, first_test: str) -> str:
    try:
        defs = [n.name for n in _parse(code).body if isinstance(n, ast.FunctionDef)]
    except SyntaxError:
        defs = []
    for name in defs:
        if re.search(rf"\b{re.escape(name)}\s*\(", first_test):
            return name
    m = re.findall(r"\b([A-Za-z_]\w*)\s*\(", first_test)
    builtin = {"set", "sorted", "list", "tuple", "str", "abs", "round", "math", "isclose", "len", "int", "float"}
    for name in m:
        if name not in builtin:
            return name
    return defs[-1] if defs else "solution"


def _mbpp_pairs(first_test: str, entry: str) -> list[tuple[str, str, str]]:
    try:
        node = _parse(first_test).body[0]
    except (SyntaxError, IndexError):
        return []
    if not (isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.Eq)):
        return []
    args = _call_args(ast.unparse(node.test.left), entry)
    ok, v = _literal(ast.unparse(node.test.comparators[0]))
    return [(args, repr(v), first_test)] if args is not None and ok else []


# ================================================================================================ sandbox
_RUNNER = r"""
import builtins, json, os, resource, signal, sys
def _main():
    job = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    out = open(job.pop("result_path"), "w", encoding="utf-8")
    nonce = job.pop("nonce")
    t = float(job["timeout"])
    try:
        cpu = int(job["cpu_limit"])
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
    except (ValueError, OSError):
        pass

    class _Timeout(BaseException):
        pass

    def _alarm(signum, frame):
        raise _Timeout("time limit exceeded")

    signal.signal(signal.SIGALRM, _alarm)
    import shutil as _sh, subprocess as _sp, socket as _so

    def _no_net(*a, **k):
        raise OSError(1, "network access is disabled in the sandbox")

    for name in ("connect", "connect_ex", "bind", "listen", "sendto", "sendmsg"):
        setattr(_so.socket, name, _no_net)
    for name in ("create_connection", "create_server", "getaddrinfo", "gethostbyname", "gethostbyname_ex",
                 "gethostbyaddr", "getnameinfo"):
        setattr(_so, name, _no_net)
    for name in ("kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir", "setuid", "fork", "forkpty",
                 "killpg", "rename", "renames", "truncate", "replace", "unlink", "fchmod", "fchown", "chmod", "chown",
                 "chroot", "lchown", "getcwd", "chdir"):
        if hasattr(os, name):
            setattr(os, name, None)
    _sh.rmtree = None
    _sh.move = None
    _sh.chown = None
    _sp.Popen = None
    builtins.exit = None
    builtins.quit = None
    sys._getframe = None
    for mod in ("ipdb", "joblib", "resource", "psutil", "tkinter", "inspect", "gc"):
        sys.modules[mod] = None
    program, tests, stop = job["program"], job["tests"], bool(job.get("stop_on_fail"))
    del job

    def _err(ex):
        return (type(ex).__name__ + ": " + str(ex))[:%(errc)d]

    res = {"compiled": False, "error": None, "tests": [], "passed": False}
    ns = {"__name__": "__candidate__", "__builtins__": builtins}
    try:
        signal.setitimer(signal.ITIMER_REAL, t)
        exec(compile(program, "<candidate>", "exec"), ns)
        signal.setitimer(signal.ITIMER_REAL, 0)
        res["compiled"] = True
    except BaseException as ex:
        signal.setitimer(signal.ITIMER_REAL, 0)
        res["error"] = _err(ex)
    if res["compiled"]:
        ok_all = True
        for test in tests:
            try:
                signal.setitimer(signal.ITIMER_REAL, t)
                exec(compile(test, "<test>", "exec"), ns)
                signal.setitimer(signal.ITIMER_REAL, 0)
                res["tests"].append([True, ""])
            except BaseException as ex:
                signal.setitimer(signal.ITIMER_REAL, 0)
                ok_all = False
                res["tests"].append([False, _err(ex)])
                if stop:
                    break
        res["passed"] = ok_all and len(res["tests"]) == len(tests)
    res["nonce"] = nonce
    out.write(json.dumps(res))
    out.close()
    os._exit(0)
_main()
""" % {"errc": _ERR_CHARS}

_DEAD_PROXY = "http://127.0.0.1:9"


def _child_env(work: str) -> dict[str, str]:
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": work, "TMPDIR": work, "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "http_proxy": _DEAD_PROXY, "https_proxy": _DEAD_PROXY,
            "HTTP_PROXY": _DEAD_PROXY, "HTTPS_PROXY": _DEAD_PROXY, "no_proxy": "", "LANG": "C.UTF-8"}


_SANDBOX_EXEC = "/usr/bin/sandbox-exec"
_SEATBELT_ENV = "SCIENCECLAW_FOR46_SEATBELT"        # "0" disables the macOS OS-level boundary, default "auto"
_seatbelt_lock = threading.Lock()
_seatbelt_state: bool | None = None


def _sb_quote(path: str) -> str:
    return path.replace("\\", "\\\\").replace('"', '\\"')


def _seatbelt_profile(*writable: str) -> str:
    """macOS Seatbelt profile: everything allowed except network and file writes outside ``writable``."""
    allow = " ".join(f'(subpath "{_sb_quote(os.path.realpath(w))}")' for w in writable)
    return ("(version 1)(allow default)(deny network*)(deny file-write*)"
            f'(allow file-write* {allow} (literal "/dev/null") (literal "/dev/dtracehelper"))')


def _seatbelt_usable() -> bool:
    """True iff ``sandbox-exec`` exists and demonstrably blocks a loopback connect (probed once per process)."""
    global _seatbelt_state
    with _seatbelt_lock:
        if _seatbelt_state is None:
            ok = False
            if sys.platform == "darwin" and os.environ.get(_SEATBELT_ENV, "auto").strip().lower() not in (
                    "0", "off", "false", "no") and os.path.exists(_SANDBOX_EXEC):
                probe = ("import socket\ntry:\n    socket.socket().connect(('127.0.0.1', 9))\n"
                         "except PermissionError:\n    print('blocked')\nexcept OSError:\n    print('open')\n")
                try:
                    r = subprocess.run([_SANDBOX_EXEC, "-p", "(version 1)(allow default)(deny network*)",
                                        sys.executable, "-I", "-c", probe], capture_output=True, timeout=20,
                                       env={"PATH": "/usr/bin:/bin"}, stdin=subprocess.DEVNULL)
                    ok = r.stdout.decode("utf-8", "replace").strip() == "blocked"
                except (OSError, subprocess.SubprocessError):
                    ok = False
            _seatbelt_state = ok
        return _seatbelt_state


def run_program(program: str, tests: list[str], timeout_s: float, stop_on_fail: bool = False) -> dict:
    """Execute ``program`` then each test statement in a fresh python subprocess (one namespace).

    Returns ``{"compiled", "error", "tests": [[ok, err], ...], "passed", "timeout", "sandbox"}``. ``passed`` =
    program ran and every test ran without exception. Each statement has its own ``timeout_s`` (SIGALRM) and the
    process a CPU limit of ``timeout_s * (len(tests) + 1) + 2`` s; it is killed after
    ``timeout_s * (len(tests) + 1) + 5`` s of wall time. The job (program, tests) is sent on stdin, never written
    into the candidate's working directory; the verdict is written to a file in a separate private directory and
    must echo a per-run random nonce. ``sandbox`` is ``"seatbelt"`` when the macOS ``sandbox-exec`` boundary (no
    network, writes only to the two private directories) was applied, else ``"none"`` (in-process guards only).
    A job whose ``sandbox-exec`` wrapper itself failed to start is re-run once without it, never scored as failed.
    """
    if _seatbelt_usable():
        r = _run_program(program, tests, timeout_s, stop_on_fail, True)
        if not r.pop("_wrapper_failed", False):
            return r
    r = _run_program(program, tests, timeout_s, stop_on_fail, False)
    r.pop("_wrapper_failed", None)
    return r


def _run_program(program: str, tests: list[str], timeout_s: float, stop_on_fail: bool, seatbelt: bool) -> dict:
    work = tempfile.mkdtemp(prefix="sc46_")
    private = tempfile.mkdtemp(prefix="sc46r_")
    sandbox = "seatbelt" if seatbelt else "none"
    try:
        res = Path(private) / f"{secrets.token_hex(8)}.json"
        nonce = secrets.token_hex(16)
        n_stmt = len(tests) + 1
        payload = json.dumps({"program": program, "tests": tests, "timeout": float(timeout_s),
                              "stop_on_fail": bool(stop_on_fail), "result_path": str(res), "nonce": nonce,
                              "cpu_limit": int(float(timeout_s) * n_stmt) + 2}).encode("utf-8")
        wall = float(timeout_s) * n_stmt + 5.0
        cmd = [sys.executable, "-I", "-c", _RUNNER]
        if seatbelt:
            cmd = [_SANDBOX_EXEC, "-p", _seatbelt_profile(work, private)] + cmd
        try:
            proc = subprocess.Popen(cmd, cwd=work, env=_child_env(work),
                                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                    start_new_session=True)
        except OSError as ex:
            return {"compiled": False, "error": f"could not start interpreter: {ex}", "tests": [], "passed": False,
                    "timeout": False, "sandbox": sandbox, "_wrapper_failed": seatbelt}
        try:
            _out, err = proc.communicate(payload, timeout=wall)
            timed_out = False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            _out, err = proc.communicate()
            timed_out = True
        out: dict = {}
        if res.exists():
            try:
                out = json.loads(res.read_text(encoding="utf-8") or "{}")
            except json.JSONDecodeError:
                out = {}
        if out and out.get("nonce") != nonce:
            out = {}
        if not out:
            wrapper_failed = bool(seatbelt and not timed_out and (err or b"").lstrip().startswith(b"sandbox-exec:"))
            tail = (err or b"").decode("utf-8", "replace")[-_ERR_CHARS:]
            why = "time limit exceeded (process killed)" if timed_out else (
                f"process ended without a result (exit {proc.returncode}; e.g. CPU limit, os._exit or crash)"
                + (f": {tail}" if tail else ""))
            return {"compiled": False, "error": why, "tests": [], "passed": False, "timeout": timed_out,
                    "sandbox": sandbox, "_wrapper_failed": wrapper_failed}
        out.pop("nonce", None)
        out["timeout"] = timed_out
        out["sandbox"] = sandbox
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(private, ignore_errors=True)


def _parallel(fn, args: list, workers: int = MAX_WORKERS) -> list:
    if len(args) <= 1:
        return [fn(a) for a in args]
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(args)))) as ex:
        return list(ex.map(fn, args))


# ================================================================================================ programs
def assemble(p: Problem, code: str) -> str:
    """The program executed for problem ``p`` given candidate text ``code`` (see module docstring)."""
    if p.dataset == "HumanEval":
        return p.prompt + ("" if p.prompt.endswith("\n") else "\n") + code
    return code


def implemented(p: Problem, code: str) -> tuple[bool, str]:
    """Assembled program parses and its last top-level def of the entry point has a non-docstring statement."""
    try:
        tree = _parse(assemble(p, code))
    except SyntaxError as ex:
        return False, f"syntax error: {ex.msg} (line {ex.lineno})"
    defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == p.entry_point]
    if not defs:
        return False, f"no top-level function named {p.entry_point!r}"
    body = defs[-1].body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    if not body:
        return False, f"function {p.entry_point!r} has no body beyond its docstring"
    return True, ""


def memorizer_code(p: Problem) -> str:
    """Reference baseline source: look up the visible examples' outputs by argument tuple, else return None."""
    table = "{" + ", ".join(f"{a!r}: {e}" for a, e, _t in p.memo_pairs) + "}"
    return (f"def {p.entry_point}(*args, **kwargs):\n"
            f"    _table = {table}\n"
            f"    return _table.get(repr(tuple(args)))\n")


def hidden_tests(p: Problem) -> tuple[str, list[str]]:
    """(extra program suffix, test statements) of the hidden evaluation."""
    if p.dataset == "HumanEval":
        return p.hidden_test, [f"check({p.entry_point})"]
    lines = p.hidden_test.split("\n\x00\n")
    setup, asserts = lines[0], [t for t in lines[1:] if t.strip()]
    return setup, asserts


def run_hidden(p: Problem, code: str, timeout_s: float = HIDDEN_TIMEOUT_S) -> dict:
    suffix, tests = hidden_tests(p)
    program = assemble(p, code) + "\n\n" + suffix + "\n"
    r = run_program(program, tests, timeout_s, stop_on_fail=True)
    return {"passed": bool(r.get("passed")), "error": (r.get("error") or next(
        (e for ok, e in r.get("tests", []) if not ok), None))}


def run_visible(p: Problem, code: str, timeout_s: float = VISIBLE_TIMEOUT_S) -> dict:
    program = assemble(p, code) + ("\n\n" + p.visible_setup + "\n" if p.visible_setup else "\n")
    if not p.visible_tests:
        r = run_program(program, [], timeout_s)
        return {"passed": bool(r.get("compiled")), "n_tests": 0, "n_passed": 0,
                "error": r.get("error") or "no visible tests for this problem (program executed only)"}
    r = run_program(program, list(p.visible_tests), timeout_s)
    oks = [bool(ok) for ok, _ in r.get("tests", [])]
    first_err = r.get("error") or next((e for ok, e in r.get("tests", []) if not ok), None)
    failing = [p.visible_tests[i] for i, ok in enumerate(oks) if not ok][:3]
    fail_idx = [i for i in range(len(p.visible_tests)) if i >= len(oks) or not oks[i]]
    return {"passed": bool(r.get("passed")), "n_tests": len(p.visible_tests), "n_passed": int(sum(oks)),
            "error": first_err, "failing_tests": failing, "failing_idx": fail_idx}


# ================================================================================================ data
def _filter_visible(problems: dict[str, Problem]) -> None:
    """Drop visible tests that the dataset's own reference solution fails (docstring errata / parser artefacts).

    Deterministic; the kept test indices per problem are cached in ``cache/tasks/FoR46``. Only public examples
    are removed — hidden tests are untouched and never shown.
    """
    f = cache_dir(CODE) / f"visible_filter_{_CACHE_VERSION}.json"
    try:
        kept: dict[str, list[int]] = read_json(f) if f.exists() else {}
    except (OSError, json.JSONDecodeError):
        kept = {}
    todo = [p for k, p in problems.items() if k not in kept and p.visible_tests]
    if todo:
        res = _parallel(lambda p: run_visible(p, p.canonical), todo)
        for p, r in zip(todo, res):
            fails = set(r.get("failing_idx", []))
            kept[p.item_id] = [i for i in range(len(p.visible_tests)) if i not in fails]
        atomic_write_json(f, kept)
    for k, p in problems.items():
        if k in kept and p.visible_tests:
            idx = [i for i in kept[k] if 0 <= i < len(p.visible_tests)]
            p.visible_tests = [p.visible_tests[i] for i in idx]
            kept_tests = set(p.visible_tests)
            p.memo_pairs = [pr for pr in p.memo_pairs if pr[2] in kept_tests]


@dataclass
class _Data:
    problems: dict[str, Problem]
    pools: dict[str, list[str]]        # src / val / id / ood (ordered) ; ood_extra (OOD reserve, second layer)
    roles: str                         # ROLES_V2 | ROLES_V1 | ROLES_HASH : where the pools came from
    receipt: dict

    @property
    def manifest_used(self) -> bool:
        return self.roles != ROLES_HASH


def _load(root: Path, roles: str | None = None) -> _Data:
    d = root / DATASET_DIR
    problems: dict[str, Problem] = {}
    with gzip.open(d / "HumanEval.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            tests, pairs = humaneval_visible(r["prompt"], r["entry_point"])
            problems[r["task_id"]] = Problem(item_id=r["task_id"], dataset="HumanEval", prompt=r["prompt"],
                                             entry_point=r["entry_point"], visible_tests=tests, visible_setup="",
                                             memo_pairs=pairs, hidden_test=r["test"],
                                             canonical=r["canonical_solution"])
    san = read_json(d / "sanitized-mbpp.json")
    for r in san:
        tid = f"MBPP/{int(r['task_id'])}"
        first = r["test_list"][0]
        entry = _mbpp_entry(r["code"], first)
        setup = "\n".join(r.get("test_imports") or [])
        hidden = setup + "\n\x00\n" + "\n\x00\n".join(r["test_list"])
        problems[tid] = Problem(item_id=tid, dataset="MBPP", prompt=r["prompt"].strip(), entry_point=entry,
                                visible_tests=[first], visible_setup=setup, memo_pairs=_mbpp_pairs(first, entry),
                                hidden_test=hidden, canonical=r["code"])
    _filter_visible(problems)
    pools, label, info = _select_roles(d, problems, roles)
    receipt = receipt_summary(d / "receipt.json")
    receipt.update(info)
    return _Data(problems=problems, pools=pools, roles=label, receipt=receipt)


# ---------------------------------------------------------------------------------------------- roles
def _manifest_ids(parts: dict, role: str, mbpp: bool = False) -> list[str]:
    p = parts[role]
    ids = [f"MBPP/{int(x)}" if mbpp else str(x) for x in p["selected_ids"]]
    if "count" in p and int(p["count"]) != len(ids):
        raise ValueError(f"FoR46 manifest role {role!r}: count {p['count']} != {len(ids)} selected_ids")
    return ids


def _mbpp_native(i: str) -> int:
    return int(i.split("/", 1)[1])


def _select_roles(d: Path, problems: dict[str, Problem], want: str | None) -> tuple[dict[str, list[str]], str, dict]:
    """Pools ``src / val / id / ood / ood_extra`` (lists of item ids), their origin label and receipt fields.

    ``want`` = ``None`` (auto: v2 manifest, else v1 manifest, else hash ranking) | ``"v2"`` | ``"v1"`` | ``"hash"``.
    A manifest that exists but is malformed raises ``ValueError`` (reported by ``available()``) instead of silently
    switching to another role assignment.
    """
    if want not in (None, "v2", "v1", "hash"):
        raise ValueError(f"FoR46 roles must be None, 'v2', 'v1' or 'hash', got {want!r}")
    he_ids = [k for k in problems if k.startswith("HumanEval/")]
    mbpp_ids = [k for k in problems if k.startswith("MBPP/")]
    info: dict[str, Any] = {}
    label: str | None = None
    pools: dict[str, list[str]] = {}
    for version, tag in ((ROLES_V2, "v2"), (ROLES_V1, "v1")):
        man_path = d / version / "sampling-manifest.json"
        if want not in (None, tag) or not man_path.exists():
            continue
        try:
            parts = read_json(man_path)["partitions"]
            if version == ROLES_V2:
                pools = {"src": _manifest_ids(parts, "source_train"), "val": _manifest_ids(parts, "validation"),
                         "id": _manifest_ids(parts, "heldout_id"), "ood": _manifest_ids(parts, "heldout_ood", True),
                         "ood_extra": _manifest_ids(parts, "heldout_extra", True)}
            else:
                pools = {"src": _manifest_ids(parts, "source_train"), "val": _manifest_ids(parts, "validation"),
                         "id": _manifest_ids(parts, "heldout_id"),
                         "ood": _manifest_ids(parts, "cross_dataset_mbpp", True), "ood_extra": []}
        except (KeyError, TypeError, ValueError, AttributeError) as ex:
            raise ValueError(f"FoR46 {version}/sampling-manifest.json is malformed: {type(ex).__name__}: {ex}") from ex
        label = version
        info = {"roles_version": version, "roles_manifest_sha256": hashlib.sha256(man_path.read_bytes()).hexdigest()}
        try:
            info["roles_status"] = str(read_json(d / version / "receipt.json").get("status", ""))
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
        break
    if label is None:
        if want in ("v1", "v2"):
            raise ValueError(f"FoR46 roles {want!r} requested but reconstructed_{want}/sampling-manifest.json is missing")
        label = ROLES_HASH                       # fallback: the v2 sizes from a seeded sha256 ranking (documented)
        ranked = hash_rank(he_ids, f"{CODE}|{PARTITION_SEED}|humaneval")
        test_range = [i for i in mbpp_ids if MBPP_TEST_RANGE[0] <= _mbpp_native(i) <= MBPP_TEST_RANGE[1]]
        mranked = hash_rank(test_range, f"{CODE}|{PARTITION_SEED}|mbpp")
        pools = {"id": ranked[:64], "val": ranked[64:128], "src": ranked[128:], "ood": mranked[:64],
                 "ood_extra": mranked[64:128]}
        info = {"roles_version": ROLES_HASH}
    if label == ROLES_V1:                        # v1 has no reserve: further sanitized-MBPP problems by sha256 rank
        taken = set(pools["ood"])
        pools["ood_extra"] = [i for i in hash_rank(mbpp_ids, f"{CODE}|{PARTITION_SEED}|mbpp-extra")
                              if i not in taken]
    _check_pools(pools, problems, label)
    return pools, label, info


def _check_pools(pools: dict[str, list[str]], problems: dict[str, Problem], label: str) -> None:
    for k in ("src", "val", "id", "ood"):
        if not pools[k]:
            raise ValueError(f"FoR46 role pool {k!r} is empty ({label})")
    allv = [i for k in ("src", "val", "id", "ood", "ood_extra") for i in pools[k]]
    missing = [i for i in allv if i not in problems]
    if missing:
        raise ValueError(f"FoR46 sampling manifest lists unknown problems: {missing[:5]}")
    if len(allv) != len(set(allv)):
        raise ValueError("FoR46 pools overlap")
    for k, prefix in (("src", "HumanEval/"), ("val", "HumanEval/"), ("id", "HumanEval/"), ("ood", "MBPP/"),
                      ("ood_extra", "MBPP/")):
        bad = [i for i in pools[k] if not i.startswith(prefix)]
        if bad:
            raise ValueError(f"FoR46 role pool {k!r} must contain {prefix}* problems only, got {bad[:3]}")
    if label == ROLES_V2:
        out = [i for k in ("ood", "ood_extra") for i in pools[k]
               if not MBPP_TEST_RANGE[0] <= _mbpp_native(i) <= MBPP_TEST_RANGE[1]]
        if out:
            raise ValueError(f"FoR46 v2 OOD problems must be in the MBPP official test range "
                             f"{MBPP_TEST_RANGE[0]}-{MBPP_TEST_RANGE[1]}, got {out[:3]}")


# ================================================================================================ adapter
class CodeAdapter:
    """HumanEval (IID) -> MBPP (OOD) execution pass@1 adapter (``bench.task.TaskAdapter`` protocol)."""

    discipline = CODE
    name = "HumanEval -> MBPP"
    family = FAMILY
    metric = "execution pass@1"
    direction = "max"
    task_type = "program_synthesis"

    def __init__(self, data_root: str | os.PathLike | None = None, accept_margin: float = ACCEPT_MARGIN,
                 hidden_timeout_s: float = HIDDEN_TIMEOUT_S, budget: Budget | None = None,
                 roles: str | None = None) -> None:
        self.root = resolve_data_root(data_root)
        self.accept_margin = float(accept_margin)
        self.hidden_timeout_s = float(hidden_timeout_s)
        self.budget = budget
        self.roles = roles          # None = auto (reconstructed_v2 -> reconstructed_v1 -> hash ranking) | "v2" | "v1" | "hash"
        self._data = Lazy(lambda: _load(self.root, roles))
        self._ref_lock = threading.Lock()
        self._ref_cache: dict[str, bool] | None = None

    # ------------------------------------------------------------------------------------------ protocol
    def available(self) -> tuple[bool, str]:
        d = self.root / DATASET_DIR
        for f in ("HumanEval.jsonl.gz", "sanitized-mbpp.json"):
            if not (d / f).exists():
                return False, f"missing {d / f}"
        try:
            data = self._data.get()
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as ex:
            return False, f"FoR46 data unreadable: {type(ex).__name__}: {ex}"
        sizes = "/".join(str(len(data.pools[k])) for k in ("src", "val", "id", "ood"))
        origin = (f"data-team {data.roles} roles" if data.roles != ROLES_HASH else "fallback sha256-rank roles")
        return True, (f"HumanEval {sum(p.dataset == 'HumanEval' for p in data.problems.values())} + MBPP sanitized "
                      f"{sum(p.dataset == 'MBPP' for p in data.problems.values())} problems ({origin}; "
                      f"src/val/id/ood pools {sizes}, OOD reserve {len(data.pools['ood_extra'])})")

    def _layers(self, data: _Data, split: str) -> list[list[str]]:
        """Item pools of a split, in the order episodes consume them (OOD: role pool, then the reserve)."""
        layers = [list(data.pools[split])]
        if split == "ood" and data.pools["ood_extra"]:
            layers.append(list(data.pools["ood_extra"]))
        return layers

    def max_disjoint_episodes(self, split: str, items_per_episode: int = 16) -> int:
        """Most mutually item-disjoint episodes of ``items_per_episode`` items ``build_episodes`` can return.

        For ``src`` this is the number per recycling cycle: more episodes are still built, by recycling the source
        items with a fresh seeded permutation after every ``max_disjoint_episodes`` episodes.
        """
        check_split(split)
        return sum(len(p) // int(items_per_episode) for p in self._layers(self._data.get(), split))

    def build_episodes(self, split: str, n: int, seed: int, items_per_episode: int = 16) -> list[Episode]:
        """``n`` episodes of ``items_per_episode`` problems; deterministic in ``(split, seed)`` and prefix-stable.

        val / id / ood episodes are item-disjoint (``PoolExhausted`` beyond :meth:`max_disjoint_episodes`); ood
        first uses ``heldout_ood`` and continues with the ``heldout_extra`` reserve (own seeded stream, so the
        first episodes never change). src recycles its 36 items (see the module docstring).
        """
        check_split(split)
        ipe = int(items_per_episode)
        if ipe < 1:
            raise ValueError("items_per_episode must be >= 1")
        n = int(n)
        data = self._data.get()
        what = f"{CODE}/{split}"
        layers = self._layers(data, split)
        blocks: list[list[str]] = []
        if len(layers) == 1:
            blocks = draw_blocks({"all": layers[0]}, {"all": ipe}, n, episode_rng(CODE, split, seed), what,
                                 cycle=(split == "src"))
        else:
            for li, pool in enumerate(layers):
                take = min(len(pool) // ipe, n - len(blocks))
                if take > 0:
                    rng = episode_rng(CODE, split, seed, *(("reserve",) if li else ()))
                    blocks += draw_blocks({"all": pool}, {"all": ipe}, take, rng, what)
            if len(blocks) < n:
                raise PoolExhausted(f"{what}: requested {n} episodes of {ipe} items but the role pool and its reserve "
                                    f"support only {len(blocks)} (sizes {[len(p) for p in layers]})")
        per_cycle = max(1, len(layers[0]) // ipe)
        return [self._episode(data, split, int(seed), k, ids, cycle=(k // per_cycle if split == "src" else None))
                for k, ids in enumerate(blocks)]

    def pooled_metric(self, per_episode: list[dict]) -> float | None:
        """pass@1 over all items of the episodes (invalid outputs count as failures)."""
        passed: list[int] = []
        for p in per_episode:
            if not p:
                continue
            v = p.get("passed")
            ids = p.get("item_ids") or []
            if v is None:
                v = [0] * len(ids)
            if ids and len(v) != len(ids):
                raise ValueError("FoR46 pooled payload: 'passed' and 'item_ids' differ in length")
            passed.extend(int(bool(x)) for x in v)
        return float(np.mean(passed)) if passed else None

    # ------------------------------------------------------------------------------------------ reference
    def _ref_file(self) -> Path:
        return cache_dir(CODE) / f"memorizer_pass_{_CACHE_VERSION}.json"

    def reference_passes(self, problems: list[Problem]) -> list[bool]:
        """Hidden-test outcome of the memorizer baseline per problem (computed once, cached on disk)."""
        with self._ref_lock:
            if self._ref_cache is None:
                f = self._ref_file()
                try:
                    self._ref_cache = {str(k): bool(v) for k, v in read_json(f).items()} if f.exists() else {}
                except (OSError, json.JSONDecodeError):
                    self._ref_cache = {}
            todo = [p for p in problems if p.item_id not in self._ref_cache]
        if todo:
            res = _parallel(lambda p: run_hidden(p, memorizer_code(p), self.hidden_timeout_s)["passed"], todo)
            with self._ref_lock:
                assert self._ref_cache is not None
                for p, r in zip(todo, res):
                    self._ref_cache[p.item_id] = bool(r)
                atomic_write_json(self._ref_file(), self._ref_cache)
        with self._ref_lock:
            assert self._ref_cache is not None
            return [self._ref_cache[p.item_id] for p in problems]

    # ------------------------------------------------------------------------------------------ episodes
    def _episode(self, data: _Data, split: str, seed: int, k: int, ids: list[str],
                 cycle: int | None = None) -> Episode:
        probs = [data.problems[i] for i in ids]
        n = len(probs)
        pool = "ood" if split == "ood" else "iid"
        public = [p.public() for p in probs]
        timeout_h = self.hidden_timeout_s

        def load_eval_inputs(inputs: dict, config: dict) -> dict:
            return {"problems": [dict(x) for x in public], "entry_points": [p.entry_point for p in probs]}

        def run_visible_tests(inputs: dict, config: dict) -> dict:
            codes, why = as_str_list(inputs.get("codes"), n)
            if codes is None:
                raise ValueError(f"run_visible_tests: codes must be a list of {n} strings ({why})")
            t = float(min(max(float(config.get("test_timeout_s", VISIBLE_TIMEOUT_S)), 0.5), 30.0))
            res = _parallel(lambda a: run_visible(a[0], a[1], t), list(zip(probs, codes)))
            rate = float(np.mean([r["passed"] for r in res])) if res else 0.0
            return {"results": res, "passed": [bool(r["passed"]) for r in res], "visible_pass_rate": rate}

        tools = [
            ToolSpec("load_eval_inputs",
                     f"The {n} programming problems to solve, in output order. Each problem dict has 'prompt' "
                     "(HumanEval-style python stub with signature and docstring, or a task description), "
                     "'entry_point' (name of the function the tests call), 'visible_tests' (newline-separated "
                     "python assert statements that are public examples), 'kind'.",
                     {}, {"problems": PortSchema("list", (n,), dtype="dict", description="problem dicts (str fields)"),
                          "entry_points": PortSchema("list", (n,), dtype="str", description="function names")},
                     load_eval_inputs),
            ToolSpec("run_visible_tests",
                     "Runs candidate programs against the PUBLIC visible tests only (one fresh python subprocess "
                     "per problem, assembled exactly like the hidden evaluation). Returns per-problem results "
                     "{passed, n_tests, n_passed, error, failing_tests}, a pass list and the visible pass rate.",
                     {"codes": PortSchema("list", (n,), dtype="str", description="one candidate source per problem")},
                     {"results": PortSchema("list", (n,), dtype="dict"),
                      "passed": PortSchema("list", (n,), dtype="bool"),
                      "visible_pass_rate": PortSchema("number", unit="1", description="fraction of problems passing "
                                                                                       "all visible tests")},
                     run_visible_tests, config_doc="test_timeout_s: time limit per test statement in seconds (0.5-30, default 5)"),
        ]

        def c_impl(y: Any, trace: Trace | None) -> tuple[bool, str]:
            v, why = as_str_list(y, n)
            if v is None:
                return False, why
            bad = []
            for i, (p, c) in enumerate(zip(probs, v)):
                ok, msg = implemented(p, c)
                if not ok:
                    bad.append(f"item {i}: {msg}")
            return (not bad), ("all items implement their entry point" if not bad
                               else f"{len(bad)} items invalid; " + "; ".join(bad[:3]))

        constraints = [
            c_str_list(n, "python source per problem"),
            ConstraintSpec("implemented", "for every problem the assembled program is valid Python and defines the "
                                          "entry-point function with a body beyond its docstring", c_impl),
        ]

        def evaluate(yv: Any, trace: Trace | None) -> EvalResult:
            ref_pass = self.reference_passes(probs)
            ref = float(np.mean(ref_pass))
            payload: dict[str, Any] = {"item_ids": list(ids), "passed": None, "ref_passed": [int(x) for x in ref_pass]}
            v, why = as_str_list(yv, n)
            if v is None:
                payload["passed"] = [0] * n
                return EvalResult(metrics={"pass@1": 0.0, "reference_pass@1": ref}, primary=0.0, direction="max",
                                  accepted=False, details={"reference": ref, "norm_score": 0.0,
                                                           "pooled_payload": payload, "invalid": why})
            res = _parallel(lambda a: run_hidden(a[0], a[1], timeout_h), list(zip(probs, v)))
            passed = [bool(r["passed"]) for r in res]
            score = float(np.mean(passed))
            payload["passed"] = [int(x) for x in passed]
            return EvalResult(
                metrics={"pass@1": score, "reference_pass@1": ref, "n_passed": float(sum(passed)), "n_items": float(n)},
                primary=score, direction="max", accepted=bool(score >= ref + self.accept_margin - 1e-12),
                details={"reference": ref, "norm_score": norm_score(score, ref, "max", floor=NORM_FLOOR),
                         "pooled_payload": payload,
                         "per_item_error_kind": [None if r["passed"] else (r["error"] or "failed").split(":", 1)[0]
                                                 for r in res]})

        def dev_evaluate(yv: Any) -> dict:
            v, why = as_str_list(yv, n)
            if v is None:
                return {"error": why}
            res = _parallel(lambda a: run_visible(a[0], a[1]), list(zip(probs, v)))
            with_tests = [r for r in res if r["n_tests"] > 0]
            return {"visible_pass_rate": float(np.mean([r["passed"] for r in res])),
                    "visible_pass_rate_items_with_tests": (float(np.mean([r["passed"] for r in with_tests]))
                                                           if with_tests else None),
                    "n_items_with_visible_tests": len(with_tests), "n_items": n,
                    "note": "fraction of problems whose program passes all PUBLIC visible tests (hidden tests are "
                            "never run here)"}

        kinds = sorted({p.dataset for p in probs})
        objective = (
            f"Write Python 3 code for {n} programming problems. Each problem (tool load_eval_inputs) gives a prompt, "
            "the name of the function to implement (entry_point) and public example tests (visible_tests). "
            "Every problem also has hidden unit tests that call the entry-point function; a problem counts as "
            "solved iff all its hidden tests run without exception within a time limit of "
            f"{timeout_h:g} s per test statement. Metric: execution pass@1 = fraction of solved problems (one "
            "program per problem). Deliverable y: a list of exactly "
            f"{n} Python source strings, y[i] for problem i in the given order. For a python-stub prompt the "
            "program executed is the prompt text followed by y[i] (y[i] may be the function body continuing the "
            "stub, or a complete definition of the entry-point function with any imports/helpers it needs); for a "
            "task-description prompt the program is y[i] alone and must define the entry-point function with the "
            "signature used by the example test. Programs run without network access; only the Python standard "
            "library is guaranteed to be importable.\n"
            + scilib.describe("codegen")
        )
        return Episode(
            id=episode_id(CODE, split, seed, k), discipline=CODE, family=FAMILY, split=split,
            task_type=self.task_type, objective=objective,
            required_output=PortSchema("list", (n,), dtype="str", description="python source per problem"),
            tools=tools, constraints=constraints,
            budget=self.budget or Budget(max_steps=12, max_policy_tokens=200_000, max_wall_s=1800.0,
                                         max_node_s=180.0, max_llm_items=4 * n),
            lineage={"dataset": "HumanEval" if pool == "iid" else "MBPP (sanitized)", "datasets": kinds,
                     "pool": pool, "ood_kind": "cross_dataset" if pool == "ood" else None,
                     "item_ids": list(ids), "seed": seed, "index": k,
                     "partition": (f"data-team {data.roles} sampling-manifest roles" if data.manifest_used
                                   else f"sha256 rank, partition seed {PARTITION_SEED}"),
                     "roles": data.roles,
                     "src_items_recycled": split == "src",
                     "src_cycle": cycle,
                     "ood_layer": (None if split != "ood" else ("role" if ids[0] in set(data.pools["ood"])
                                                                else "reserve")),
                     "sources": {"HumanEval": "https://github.com/openai/human-eval",
                                 "MBPP": "https://github.com/google-research/google-research/tree/master/mbpp"},
                     "receipt": data.receipt, "n_visible_tests": [len(p.visible_tests) for p in probs]},
            acceptance=f"pass@1 >= reference pass@1 (visible-example memorizer) + {self.accept_margin:g}",
            tolerance={"rtol": 0.0, "atol": 0.0},
            tags=["code", "program_synthesis", "python", "unit_tests", "llm"],
            metric=self.metric, direction=self.direction, n_items=n,
            _evaluate=evaluate, _dev_evaluate=dev_evaluate,
        )


Adapter = CodeAdapter
