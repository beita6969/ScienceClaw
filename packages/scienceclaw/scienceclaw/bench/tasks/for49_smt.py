"""FoR49 Mathematical sciences — SMT-COMP 2025 QF_NonLinearIntArith single query, metric oracle-agreement accuracy.

Data: the complete SMT-LIB 2025 non-incremental ``QF_NIA`` / ``QF_NIRA`` archives (Zenodo DOI
10.5281/zenodo.15493090, release 2025.05.22, CC-BY-4.0; the library from which SMT-COMP 2025 draws its
QF_NonLinearIntArith single-query division) as extracted by the data team under ``<DATA_ROOT>/
for49-smtcomp2025-nonlinearintarith/source_library`` with the data team's candidate frame
``reconstructed_v1/candidate-frame.jsonl`` (25,455 files; *eligible* = one ``set-logic`` matching the directory, one
``check-sat``, definite ``:status``, byte duplicates removed).

* **Item** = one eligible benchmark of at most ``MAX_BYTES`` (1 MB; 19,930 files) whose official status (the
  SMT-LIB ``(set-info :status ...)`` value that SMT-COMP scores against) is ``sat`` or ``unsat``. The agent sees a
  *sanitized* query: comments and every ``set-info`` (status, source, license, category) and output command removed.
* **Difficulty screen** (SMT-COMP removes trivially easy benchmarks): every eligible item is run once through z3
  5.1.0 with a 1-second CPU-time limit (RLIMIT_CPU, load independent) plus ``rlimit=16e6``; *hard* = not decided,
  *easy* = decided in agreement with the official status, *conflict* = decided against it (excluded). The result
  is frozen in ``for49_screen_v1.json.gz`` next to this module (``python -m scienceclaw.bench.tasks.for49_smt
  screen`` then ``... freeze``).
* **Lineage unit.** Items are grouped by generator lineage (family + program/problem stem, e.g.
  ``20170427-VeryMax/CInteger/Stroeder_15__GCD2.c``); per (group, status, tier) one representative is fixed by a
  seed-independent hash, so near-duplicate variants of one program never fill an episode.
* **Pools.** IID = families ``20170427-VeryMax``, ``20220315-MathProblems``, ``calypto``; their groups are cut once
  (``partition_seed``) into visible ``train`` / ``dev`` and evaluation ``src`` / ``val`` / ``id`` (25/5/35/10/25 %).
  OOD = all other QF_NIA/QF_NIRA families (``AProVE``, ``LassoRanker``, ``leipzig``, ``20230328-sqrtmodinv-hoenicke``,
  ``mcm``, ``UltimateAutomizer``, ...): a generator/family shift inside the same library
  (``lineage["ood_kind"] = "proxy_within_dataset"``; no second SMT benchmark set exists).
* **Episodes.** 16 items = 12 hard (6 sat + 6 unsat) + 4 easy; the total number of ``sat`` items is
  ``6 + (h mod 5)`` (h = hash of split seed and episode index). Strata are drawn block-wise from seeded permutations
  (prefix-stable, item-disjoint).
* **Visible data (D_E).** ``load_train``: 48 labelled sanitized queries (36 hard balanced + 12 easy, from the train
  groups); ``load_dev_inputs`` + ``score_dev``: 16 dev queries (same composition as an episode) and their accuracy;
  ``load_eval_inputs``: the 16 evaluation queries; ``z3_check``: the Z3 binary shipped with the ``z3-solver`` wheel,
  one subprocess per query with a per-query timeout (``-T``), memory cap, optional deterministic ``rlimit`` and
  z3 parameters, in parallel; results are memoized per (query, configuration) inside a process, so replays are
  reproducible.
* **Metric (D_V).** Oracle-agreement accuracy = fraction of items whose predicted status equals the official
  status (``unknown`` never agrees). Wrong definite answers are reported as ``n_wrong_definite``.
* **Reference baseline:** the majority status of the episode's ``load_train`` labels (``sat``). **Acceptance:**
  ``accuracy >= reference + 0.2``.
* **Hard constraints:** 16 labels, each one of ``sat`` / ``unsat`` / ``unknown``.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import scilib
from ...core.schema import PortSchema
from ...core.trace import Trace
from ..task import Budget, EvalResult, Episode, ToolSpec
from ._adapter_utils_for36_46_49_52 import (
    PARTITION_SEED, Lazy, PoolExhausted, as_str_list, atomic_write_json, c_allowed_labels, c_str_list, cache_dir,
    check_split, episode_id, episode_rng, hash_rank, norm_score, partition_ids, read_json, receipt_summary,
    resolve_data_root, sha_hex, stable_int,
)

CODE = "FoR49"
FAMILY = "Engineering & computing"
DATASET_DIR = "for49-smtcomp2025-nonlinearintarith"
LABELS = ("sat", "unsat", "unknown")
IID_FAMILIES = ("20170427-VeryMax", "20220315-MathProblems", "calypto")
IID_FRACTIONS = {"train": 0.25, "dev": 0.05, "src": 0.35, "val": 0.10, "id": 0.25}
MAX_BYTES = 1_000_000
HARD_FRACTION = 0.75           # share of screen-hard items per episode (balanced sat/unsat inside the hard stratum)
N_TRAIN = 48
N_DEV = 16
ACCEPT_MARGIN = 0.2
Z3_DEFAULT_TIMEOUT_S = 10.0
Z3_MAX_TIMEOUT_S = 120.0
Z3_DEFAULT_MEMORY_MB = 2048
_CACHE_VERSION = "v1"
_PARAM_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.]*=[A-Za-z0-9_.\-]+$")
_DROP_COMMANDS = {"set-info", "get-info", "get-model", "get-value", "get-assignment", "get-proof", "get-unsat-core",
                  "get-unsat-assumptions", "get-assertions", "get-option", "echo"}


# ================================================================================================ SMT-LIB text
def strip_comments(text: str) -> str:
    """Remove ``;`` comments (outside string literals and ``|quoted symbols|``)."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == ";":
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if c == '"':
            j = i + 1
            while j < n:
                if text[j] == '"':
                    if j + 1 < n and text[j + 1] == '"':
                        j += 2
                        continue
                    break
                j += 1
            out.append(text[i:j + 1])
            i = j + 1
            continue
        if c == "|":
            j = text.find("|", i + 1)
            j = n - 1 if j < 0 else j
            out.append(text[i:j + 1])
            i = j + 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def status_baseline_diagnostics(y_true: list[str], y_pred: list[str], majority: list[str],
                                pure_z3: list[str] | None, tiers: list[str]) -> dict[str, Any]:
    """Details-only split of solver and majority shortcuts from submitted predictions."""
    n = len(y_true)
    if not (len(y_pred) == len(majority) == len(tiers) == n):
        raise ValueError("baseline diagnostic vectors must have equal length")

    def acc(pred: list[str] | None, sel: list[int] | None = None) -> float | None:
        if pred is None:
            return None
        ii = list(range(n)) if sel is None else sel
        return float(np.mean([pred[i] == y_true[i] for i in ii])) if ii else None

    out: dict[str, Any] = {
        "diagnostic_only": True,
        "tool_shortcut_not_capability_evidence": True,
        "n_items": n,
        "majority_guess_accuracy": acc(majority),
        "submitted_accuracy": acc(y_pred),
        "pure_z3_accuracy": acc(pure_z3),
        "pure_z3_decided": int(sum(x in ("sat", "unsat") for x in pure_z3)) if pure_z3 is not None else None,
        "pure_z3_unknown": int(sum(x == "unknown" for x in pure_z3)) if pure_z3 is not None else None,
    }
    by_tier: dict[str, dict[str, Any]] = {}
    for tier in sorted(set(tiers)):
        sel = [i for i, t in enumerate(tiers) if t == tier]
        by_tier[tier] = {
            "n": len(sel),
            "submitted_accuracy": acc(y_pred, sel),
            "majority_guess_accuracy": acc(majority, sel),
            "pure_z3_accuracy": acc(pure_z3, sel),
            "pure_z3_decided": (int(sum(pure_z3[i] in ("sat", "unsat") for i in sel))
                                if pure_z3 is not None else None),
        }
    out["by_tier"] = by_tier
    return out


def split_commands(text: str) -> list[str]:
    """Top-level s-expressions of a comment-free SMT-LIB script (strings / quoted symbols respected)."""
    cmds: list[str] = []
    depth, start, i, n = 0, -1, 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            j = i + 1
            while j < n:
                if text[j] == '"':
                    if j + 1 < n and text[j + 1] == '"':
                        j += 2
                        continue
                    break
                j += 1
            i = j + 1
            continue
        if c == "|":
            j = text.find("|", i + 1)
            i = (n if j < 0 else j + 1)
            continue
        if c == "(":
            if depth == 0:
                start = i
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0 and start >= 0:
                cmds.append(text[start:i + 1])
                start = -1
            if depth < 0:
                raise ValueError("unbalanced parentheses in SMT-LIB script")
        i += 1
    if depth != 0:
        raise ValueError("unterminated s-expression in SMT-LIB script")
    return cmds


def _head(cmd: str) -> str:
    m = re.match(r"\(\s*([^\s()]+)", cmd)
    return m.group(1) if m else ""


def sanitize_query(text: str) -> str:
    """Agent-visible query: comments, ``set-info`` (incl. ``:status``) and output commands removed."""
    cmds = split_commands(strip_comments(text))
    kept = [" ".join(c.split()) if len(c) < 400 else c.strip() for c in cmds if _head(c) not in _DROP_COMMANDS]
    return "\n".join(kept) + "\n"


def lineage_group(instance_id: str) -> str:
    """Generator lineage of a benchmark path (family + program/problem stem)."""
    parts = instance_id.split("/")
    fam = parts[2] if len(parts) > 2 else ""
    rest = "/".join(parts[3:])
    rest = re.sub(r"__p\d+.*$", "", rest)                 # VeryMax: program + property point
    rest = re.sub(r"\.smt2$", "", rest)
    rest = re.sub(r"_Iteration\d+.*$", "", rest)          # Ultimate/Lasso ranker iterations
    rest = re.sub(r"(?:[._-]\d+)+$", "", rest)            # numbered variants
    if fam == "20220315-MathProblems":
        rest = rest.split("_", 1)[0]                     # one group per puzzle template (STC, MC, ...)
    return f"{fam}/{rest}"


# ================================================================================================ z3
def find_z3() -> str | None:
    cands = [os.environ.get("SCIENCECLAW_Z3", ""), str(Path(sys.executable).parent / "z3")]
    try:
        import z3 as _z3  # noqa: F401 - only to locate the wheel
        cands.append(str(Path(_z3.__file__).parent / "bin" / "z3"))
    except ImportError:
        pass
    w = shutil.which("z3")
    if w:
        cands.append(w)
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


class Z3Runner:
    """Runs the z3 binary on SMT-LIB text (subprocess per query, parallel, memoized per configuration)."""

    def __init__(self, binary: str, memo_size: int = 20_000) -> None:
        self.binary = binary
        self._lock = threading.Lock()
        self._memo: OrderedDict[str, dict] = OrderedDict()
        self._memo_size = memo_size
        self.version = self._version()

    def _version(self) -> str:
        try:
            out = subprocess.run([self.binary, "--version"], capture_output=True, text=True, timeout=20)
            return out.stdout.strip() or out.stderr.strip()
        except (OSError, subprocess.SubprocessError) as ex:
            return f"unknown ({type(ex).__name__})"

    def check(self, query: str, timeout_s: float, memory_mb: int, rlimit: int | None, params: tuple[str, ...]) -> dict:
        key = sha_hex(self.version, timeout_s, memory_mb, rlimit, params, hashlib.sha256(query.encode()).hexdigest())
        with self._lock:
            if key in self._memo:
                self._memo.move_to_end(key)
                return dict(self._memo[key], memoized=True)
        res = self._run(query, timeout_s, memory_mb, rlimit, params)
        with self._lock:
            self._memo[key] = res
            while len(self._memo) > self._memo_size:
                self._memo.popitem(last=False)
        return dict(res, memoized=False)

    def _run(self, query: str, timeout_s: float, memory_mb: int, rlimit: int | None, params: tuple[str, ...],
             cpu_limit_s: int | None = None) -> dict:
        """One z3 process; ``cpu_limit_s`` adds a load-independent CPU-time limit (RLIMIT_CPU, used by the screen)."""
        work = tempfile.mkdtemp(prefix="sc49_")
        cmd = [self.binary, "-smt2", "-in", f"-T:{max(1, int(round(timeout_s)))}", f"-memory:{int(memory_mb)}",
               "smt.random_seed=0", "sat.random_seed=0"]
        preexec = None
        if cpu_limit_s is not None:
            def preexec() -> None:  # runs in the child before exec
                import resource
                resource.setrlimit(resource.RLIMIT_CPU, (int(cpu_limit_s), int(cpu_limit_s) + 1))
        if rlimit is not None:
            cmd.append(f"rlimit={int(rlimit)}")
        cmd.extend(params)
        t0 = time.monotonic()
        try:
            proc = subprocess.Popen(cmd, cwd=work, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                                                                 "HOME": work, "TMPDIR": work},
                                    start_new_session=True, preexec_fn=preexec)
            try:
                out, err = proc.communicate(query.encode(), timeout=timeout_s + 10.0)
                killed = False
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                out, err = proc.communicate()
                killed = True
        except OSError as ex:
            return {"status": "unknown", "reason": f"error: cannot run z3: {ex}", "time_s": 0.0}
        finally:
            shutil.rmtree(work, ignore_errors=True)
        dt = time.monotonic() - t0
        text = out.decode("utf-8", "replace")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        answer = next((ln for ln in lines if ln in ("sat", "unsat", "unknown", "timeout")), None)
        errors = [ln for ln in lines if ln.startswith("(error")]
        if killed:
            return {"status": "unknown", "reason": "timeout (killed)", "time_s": dt}
        if cpu_limit_s is not None and answer is None and proc.returncode in (-signal.SIGXCPU, -signal.SIGKILL):
            return {"status": "unknown", "reason": "cpu limit", "time_s": dt}
        if errors:       # z3 skips erroneous commands and may still answer: the answer is not trustworthy
            return {"status": "unknown", "reason": "error: " + errors[0][:200], "time_s": dt}
        if answer in ("sat", "unsat"):
            return {"status": answer, "reason": "", "time_s": dt}
        if answer == "timeout" or (answer is None and dt >= timeout_s * 0.95):
            return {"status": "unknown", "reason": "timeout", "time_s": dt}
        err_text = err.decode("utf-8", "replace")
        # z3's command-line front end reports a rejected parameter as "ERROR: unknown parameter 'x' at module 'm'" on
        # stderr followed by the list of legal parameters (which itself contains the word "memory"): report the
        # first line as the reason instead of misclassifying it as a memory-limit stop.
        cli_err = next((ln.strip() for ln in err_text.splitlines() if ln.startswith("ERROR")), None)
        if cli_err:
            return {"status": "unknown", "reason": "error: " + cli_err[len("ERROR"):].lstrip(": ")[:200], "time_s": dt}
        if "out of memory" in (text + err_text).lower():
            return {"status": "unknown", "reason": "memout", "time_s": dt}
        return {"status": "unknown", "reason": "unknown" if answer == "unknown" else
                f"no answer (exit {proc.returncode})", "time_s": dt}


# ================================================================================================ data
@dataclass
class _Unit:
    unit_id: str          # "<group>#<status>#<tier>"
    item_id: str          # SMT-LIB path
    group: str
    family: str
    status: str
    tier: str             # "hard" (z3 undecided in the screen) | "easy"
    path: Path


@dataclass
class _SmtData:
    units: dict[str, _Unit]
    pools: dict[str, dict[tuple[str, str], list[str]]]   # role -> (tier, status) -> unit ids (rank order)
    receipt: dict
    n_eligible: int
    n_conflict: int
    screen_meta: dict


FROZEN_SCREEN = Path(__file__).with_name("for49_screen_v1.json.gz")


def _eligible_rows(root: Path) -> list[dict]:
    frame = root / DATASET_DIR / "reconstructed_v1" / "candidate-frame.jsonl"
    out = []
    with open(frame, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("eligible") and r.get("source_status") in ("sat", "unsat") and \
                        int(r.get("bytes", 1 << 30)) <= MAX_BYTES:
                    out.append(r)
    return out


def _ids_digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def load_screen(rows: list[dict]) -> tuple[dict[str, str], dict]:
    """item id -> "hard" | "easy" | "conflict" from the frozen screen manifest (preferred) or the cache screen."""
    ids = [r["instance_id"] for r in rows]
    status = {r["instance_id"]: r["source_status"] for r in rows}
    if FROZEN_SCREEN.exists():
        with gzip.open(FROZEN_SCREEN, "rt", encoding="utf-8") as f:
            fr = json.load(f)
        if fr.get("eligible_digest") != _ids_digest(ids):
            raise ValueError("frozen FoR49 screen does not match the current eligible set (candidate frame / "
                             "MAX_BYTES changed); re-run the screen and re-freeze")
        tiers = {i: "easy" for i in ids}
        tiers.update({i: "hard" for i in fr["hard"]})
        tiers.update({i: "conflict" for i in fr["conflict"]})
        return tiers, dict(fr["meta"], source="frozen manifest " + FROZEN_SCREEN.name)
    path = screen_path()
    if not path.exists():
        raise FileNotFoundError(f"difficulty screen missing ({FROZEN_SCREEN.name} / {path}); run "
                                "`python -m scienceclaw.bench.tasks.for49_smt screen` (~1 CPU-hour)")
    st = read_json(path)
    items = st.get("items", {})
    missing = [i for i in ids if i not in items]
    if missing:
        raise ValueError(f"difficulty screen incomplete: {len(missing)} of {len(ids)} eligible items unscreened")
    tiers = {}
    for i in ids:
        z = items[i]["z3"]
        tiers[i] = "hard" if z not in ("sat", "unsat") else ("easy" if z == status[i] else "conflict")
    return tiers, dict(st.get("meta", {}), source=str(path))


def freeze_screen(data_root: str | os.PathLike | None = None) -> Path:
    """Write the compact frozen screen manifest (hard / conflict ids) next to this module."""
    rows = _eligible_rows(resolve_data_root(data_root))
    st = read_json(screen_path())
    items = st["items"]
    ids = [r["instance_id"] for r in rows]
    status = {r["instance_id"]: r["source_status"] for r in rows}
    hard = sorted(i for i in ids if items[i]["z3"] not in ("sat", "unsat"))
    conflict = sorted(i for i in ids if items[i]["z3"] in ("sat", "unsat") and items[i]["z3"] != status[i])
    out = {"meta": st.get("meta", {}), "eligible_digest": _ids_digest(ids), "n_eligible": len(ids),
           "hard": hard, "conflict": conflict}
    with gzip.open(FROZEN_SCREEN, "wt", encoding="utf-8") as f:
        json.dump(out, f, sort_keys=True)
    return FROZEN_SCREEN


def _load(root: Path) -> _SmtData:
    d = root / DATASET_DIR
    rows = _eligible_rows(root)
    tiers, meta = load_screen(rows)
    by_key: dict[tuple[str, str, str], list[dict]] = {}
    n_conflict = 0
    for r in rows:
        t = tiers[r["instance_id"]]
        if t == "conflict":
            n_conflict += 1
            continue
        by_key.setdefault((lineage_group(r["instance_id"]), r["source_status"], t), []).append(r)
    units: dict[str, _Unit] = {}
    for (g, st, t), rs in by_key.items():
        rep = hash_rank([r["instance_id"] for r in rs], f"{CODE}|{PARTITION_SEED}|rep|{g}|{st}|{t}")[0]
        r = next(x for x in rs if x["instance_id"] == rep)
        uid = f"{g}#{st}#{t}"
        units[uid] = _Unit(unit_id=uid, item_id=rep, group=g, family=r["family_group"], status=st, tier=t,
                           path=d / r["path"])
    groups_iid = sorted({u.group for u in units.values() if u.family in IID_FAMILIES})
    part = partition_ids(groups_iid, IID_FRACTIONS, f"{CODE}|{PARTITION_SEED}|iid-groups")
    role_of = {g: role for role, gs in part.items() for g in gs}
    strata = [(t, st) for t in ("hard", "easy") for st in ("sat", "unsat")]
    pools: dict[str, dict[tuple[str, str], list[str]]] = {k: {s_: [] for s_ in strata}
                                                          for k in (*IID_FRACTIONS, "ood")}
    for uid in hash_rank(list(units), f"{CODE}|{PARTITION_SEED}|units"):
        u = units[uid]
        role = role_of.get(u.group, "ood") if u.family in IID_FAMILIES else "ood"
        pools[role][(u.tier, u.status)].append(uid)
    return _SmtData(units=units, pools=pools, receipt=receipt_summary(d / "receipt.json"), n_eligible=len(rows),
                    n_conflict=n_conflict, screen_meta=meta)


class _QueryCache:
    """Sanitized query text per SMT-LIB path (memory + ``cache/tasks/FoR49/queries_v1/<sha>.smt2``)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mem: dict[str, str] = {}
        self._dir = cache_dir(CODE) / f"queries_{_CACHE_VERSION}"
        self._dir.mkdir(parents=True, exist_ok=True)

    def get(self, u: _Unit) -> str:
        with self._lock:
            if u.item_id in self._mem:
                return self._mem[u.item_id]
        f = self._dir / (hashlib.sha256(u.item_id.encode()).hexdigest()[:32] + ".smt2")
        if f.exists():
            q = f.read_text(encoding="utf-8")
        else:
            q = sanitize_query(u.path.read_text(encoding="utf-8", errors="replace"))
            if ":status" in q:
                raise ValueError(f"sanitizer left a status annotation in {u.item_id}")
            tmp = f.with_name(f.name + f".tmp{os.getpid()}-{threading.get_ident()}")
            tmp.write_text(q, encoding="utf-8")
            tmp.replace(f)
        with self._lock:
            self._mem[u.item_id] = q
        return q


def _draw_visible(pool: dict[tuple[str, str], list[str]], quota: dict[tuple[str, str], int],
                  rng: np.random.Generator) -> list[str]:
    """Visible sample: ``quota`` units per (tier, status) stratum (a short stratum is topped up from the other
    status of the same tier), shuffled."""
    out: list[str] = []
    short: dict[str, int] = {}
    for s_, q in quota.items():
        cand = pool.get(s_, [])
        take = [cand[j] for j in rng.permutation(len(cand))[:q]]
        out.extend(take)
        if len(take) < q:
            short[s_[0]] = short.get(s_[0], 0) + q - len(take)
    for tier, missing in short.items():
        rest = [u for s_, v in sorted(pool.items()) if s_[0] == tier for u in v if u not in set(out)]
        out.extend(rest[j] for j in rng.permutation(len(rest))[:missing])
    return [out[j] for j in rng.permutation(len(out))]


# ================================================================================================ adapter
class SmtAdapter:
    """SMT-LIB QF_NIA satisfiability-status prediction adapter (``bench.task.TaskAdapter`` protocol)."""

    discipline = CODE
    name = "SMT-COMP 2025 QF_NonLinearIntArith (SMT-LIB 2025 QF_NIA/QF_NIRA)"
    family = FAMILY
    metric = "oracle-agreement accuracy"
    direction = "max"
    task_type = "decision_procedure"

    def __init__(self, data_root: str | os.PathLike | None = None, accept_margin: float = ACCEPT_MARGIN,
                 n_train: int = N_TRAIN, n_dev: int = N_DEV, budget: Budget | None = None) -> None:
        self.root = resolve_data_root(data_root)
        self.accept_margin = float(accept_margin)
        self.n_train = int(n_train)
        self.n_dev = int(n_dev)
        self.budget = budget
        self._data = Lazy(lambda: _load(self.root))
        self._queries = Lazy(_QueryCache)
        self._z3 = Lazy(self._make_z3)

    def _make_z3(self) -> Z3Runner:
        b = find_z3()
        if b is None:
            raise FileNotFoundError("z3 binary not found (install z3-solver or set SCIENCECLAW_Z3)")
        return Z3Runner(b)

    # ------------------------------------------------------------------------------------------ protocol
    def available(self) -> tuple[bool, str]:
        d = self.root / DATASET_DIR
        if not (d / "reconstructed_v1" / "candidate-frame.jsonl").exists():
            return False, f"missing candidate frame {d / 'reconstructed_v1' / 'candidate-frame.jsonl'}"
        if not (d / "source_library").is_dir():
            return False, f"missing extracted SMT-LIB files {d / 'source_library'}"
        if find_z3() is None:
            return False, "z3 binary not found (pip install z3-solver or set SCIENCECLAW_Z3)"
        try:
            data = self._data.get()
        except FileNotFoundError as ex:
            return False, str(ex)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as ex:
            return False, f"FoR49 data unreadable: {type(ex).__name__}: {ex}"
        missing = [u.path for u in list(data.units.values())[:50] if not u.path.exists()]
        if missing:
            return False, f"SMT-LIB files missing, e.g. {missing[0]}"
        sizes = {k: {f"{t}/{st}": len(v) for (t, st), v in p.items()} for k, p in data.pools.items()}
        return True, (f"{len(data.units)} lineage units from {data.n_eligible} eligible benchmarks "
                      f"({data.n_conflict} solver/label conflicts excluded); pools {sizes}")

    def build_episodes(self, split: str, n: int, seed: int, items_per_episode: int = 16) -> list[Episode]:
        check_split(split)
        if items_per_episode < 4:
            raise ValueError("items_per_episode must be >= 4")
        data = self._data.get()
        rng = episode_rng(CODE, split, seed)
        pool = data.pools[split]
        perms = {s_: [pool[s_][j] for j in rng.permutation(len(pool[s_]))] for s_ in sorted(pool)}
        used = {s_: 0 for s_ in perms}
        eps: list[Episode] = []
        for k in range(int(n)):
            quota = self.quotas(split, seed, k, items_per_episode)
            ids: list[str] = []
            for s_, q in quota.items():
                if used[s_] + q > len(perms[s_]):
                    raise PoolExhausted(f"{CODE}/{split}: stratum {s_} has {len(perms[s_])} units; episode {k} needs "
                                        f"{used[s_] + q} (quotas per episode {quota})")
                ids.extend(perms[s_][used[s_]:used[s_] + q])
                used[s_] += q
            ids = [ids[j] for j in rng.permutation(len(ids))]
            eps.append(self._episode(data, split, int(seed), k, ids))
        return eps

    def quotas(self, split: str, seed: int, k: int, items: int, n_sat: int | None = None) -> dict[tuple[str, str], int]:
        """Stratum sizes: ``HARD_FRACTION`` hard items (balanced sat/unsat), the rest easy; #sat in [6/16, 10/16]."""
        hard = int(round(items * HARD_FRACTION))
        hs = hard // 2
        easy = items - hard
        n_sat = self._n_sat(split, seed, k, items) if n_sat is None else int(n_sat)
        es = int(min(max(n_sat - hs, 0), easy))
        return {("hard", "sat"): hs, ("hard", "unsat"): hard - hs, ("easy", "sat"): es, ("easy", "unsat"): easy - es}

    @staticmethod
    def _n_sat(split: str, seed: int, k: int, items: int) -> int:
        lo = int(round(items * 6 / 16))
        span = max(1, int(round(items * 10 / 16)) - lo + 1)
        return int(min(items - 1, max(1, lo + stable_int(CODE, "nsat", split, int(seed), k) % span)))

    def pooled_metric(self, per_episode: list[dict]) -> float | None:
        """Oracle-agreement accuracy over all items (an invalid output scores as the reference predictions)."""
        yt: list[str] = []
        yp: list[str] = []
        for p in per_episode:
            if not p:
                continue
            t = list(p.get("y_true") or [])
            pred = p.get("y_pred") if p.get("y_pred") is not None else p.get("y_ref")
            if pred is None or len(pred) != len(t):
                raise ValueError("FoR49 pooled payload needs y_true and y_pred (or y_ref) of equal length")
            yt.extend(t)
            yp.extend(pred)
        if not yt:
            return None
        return float(np.mean([a == b for a, b in zip(yt, yp)]))

    def pooled_diagnostics(self, per_episode: list[dict]) -> dict[str, Any]:
        """Aggregate the solver/majority shortcut audit without changing the official metric.

        ``evaluate`` records a fixed-time pure-Z3 result and the screen tier in the trusted payload.  This helper
        keeps those values separate from the submitted accuracy so a pooled score cannot be mistaken for agent
        reasoning evidence.  Older payloads that lack the optional solver audit still produce the ordinary pooled
        and majority values; ``pure_z3_*`` remains ``None`` in that case.
        """
        y_true: list[str] = []
        y_pred: list[str] = []
        y_ref: list[str] = []
        pure_z3: list[str] = []
        tiers: list[str] = []
        all_have_z3 = True
        all_have_tiers = True
        episode_scores: list[float] = []
        episode_refs: list[float] = []
        for p in per_episode:
            if not p:
                continue
            t = list(p.get("y_true") or [])
            pred = p.get("y_pred") if p.get("y_pred") is not None else p.get("y_ref")
            ref = list(p.get("y_ref") or [])
            if pred is None or len(pred) != len(t) or len(ref) != len(t):
                raise ValueError("FoR49 diagnostics need y_true, y_ref and y_pred (or y_ref) of equal length")
            y_true.extend(t)
            y_pred.extend(list(pred))
            y_ref.extend(ref)
            episode_scores.append(float(np.mean([a == b for a, b in zip(t, pred)])))
            episode_refs.append(float(np.mean([a == b for a, b in zip(t, ref)])))
            z = p.get("pure_z3")
            if z is None or len(z) != len(t):
                all_have_z3 = False
            else:
                pure_z3.extend(list(z))
            tr = p.get("tiers")
            if tr is None or len(tr) != len(t):
                all_have_tiers = False
            else:
                tiers.extend(list(tr))
        if not y_true:
            return {"diagnostic_only": True, "n_episodes": 0, "n_items": 0,
                    "pooled_accuracy": None, "pooled_reference_accuracy": None,
                    "mean_episode_accuracy": None, "mean_episode_reference_accuracy": None,
                    "majority_guess_accuracy": None, "pure_z3_accuracy": None,
                    "pure_z3_decided": None, "pure_z3_unknown": None, "by_tier": {}}

        def acc(values: list[str]) -> float:
            return float(np.mean([a == b for a, b in zip(y_true, values)]))

        out: dict[str, Any] = {
            "diagnostic_only": True,
            "tool_shortcut_not_capability_evidence": True,
            "n_episodes": len(episode_scores),
            "n_items": len(y_true),
            "pooled_accuracy": acc(y_pred),
            "pooled_reference_accuracy": acc(y_ref),
            "majority_guess_accuracy": acc(y_ref),
            "mean_episode_accuracy": float(np.mean(episode_scores)),
            "mean_episode_reference_accuracy": float(np.mean(episode_refs)),
            "slice_to_pool_delta": float(np.mean(episode_scores) - acc(y_pred)),
            "pure_z3_accuracy": (acc(pure_z3) if all_have_z3 and len(pure_z3) == len(y_true) else None),
            "pure_z3_decided": (int(sum(v in ("sat", "unsat") for v in pure_z3))
                                if all_have_z3 and len(pure_z3) == len(y_true) else None),
            "pure_z3_unknown": (int(sum(v == "unknown" for v in pure_z3))
                                if all_have_z3 and len(pure_z3) == len(y_true) else None),
        }
        if all_have_tiers and len(tiers) == len(y_true):
            by_tier: dict[str, dict[str, Any]] = {}
            for tier in sorted(set(tiers)):
                sel = [i for i, value in enumerate(tiers) if value == tier]
                by_tier[tier] = {
                    "n": len(sel),
                    "submitted_accuracy": float(np.mean([y_pred[i] == y_true[i] for i in sel])),
                    "majority_guess_accuracy": float(np.mean([y_ref[i] == y_true[i] for i in sel])),
                    "pure_z3_accuracy": (float(np.mean([pure_z3[i] == y_true[i] for i in sel]))
                                         if all_have_z3 and len(pure_z3) == len(y_true) else None),
                    "pure_z3_decided": (int(sum(pure_z3[i] in ("sat", "unsat") for i in sel))
                                        if all_have_z3 and len(pure_z3) == len(y_true) else None),
                }
            out["by_tier"] = by_tier
        else:
            out["by_tier"] = {}
        return out

    def full_split_reference(self, split: str) -> dict[str, Any]:
        """Summarize the majority reference over every representative in one split.

        Episode references are computed from a sampled visible training pool, so their accuracy is not directly
        comparable across runs with different episode seeds.  This trusted-side helper uses every deterministic
        lineage representative in the requested evaluation pool and reports the same majority-status baseline plus
        its hard/easy composition.  It never enters an episode, tool output, primary score, or acceptance decision;
        labels are read here only for an audit aggregate.  The frozen screen is deliberately used only for tier
        accounting: running Z3 over the whole split is an expensive optional experiment and is already audited on
        each formal episode by ``pooled_diagnostics``.
        """
        check_split(split)
        data = self._data.get()
        pool = data.pools[split]
        units = [data.units[uid] for key in sorted(pool) for uid in pool[key]]
        if not units:
            return {
                "split": split,
                "n_items": 0,
                "status_counts": {"sat": 0, "unsat": 0},
                "reference_status": None,
                "reference_accuracy": None,
                "majority_guess_accuracy": None,
                "tier_counts": {},
                "tier_reference_accuracy": {},
                "pure_z3_known": 0,
                "pure_z3_unknown": 0,
                "pure_z3_decided_lower_bound": 0,
                "pure_z3_decided_upper_bound": 0,
                "pure_z3_accuracy_lower_bound": None,
                "pure_z3_accuracy_upper_bound": None,
                "diagnostic_only": True,
                "tool_shortcut_not_capability_evidence": True,
            }

        truth = [u.status for u in units]
        counts = {label: int(truth.count(label)) for label in ("sat", "unsat")}
        # Deterministic tie-break matches the per-episode majority reference in _episode.
        majority = "sat" if counts["sat"] >= counts["unsat"] else "unsat"
        overall = float(np.mean([label == majority for label in truth]))
        tier_counts: dict[str, int] = {}
        tier_acc: dict[str, float] = {}
        for tier in sorted({u.tier for u in units}):
            labels = [u.status for u in units if u.tier == tier]
            tier_counts[tier] = len(labels)
            tier_acc[tier] = float(np.mean([label == majority for label in labels])) if labels else None
        # The frozen screen proves that every ``easy`` representative was decided by the screen Z3 run and agreed
        # with the published status.  It says nothing about whether the harder representatives become decidable at
        # the longer formal timeout, so expose conservative bounds instead of silently running a full solver sweep.
        n_known = int(tier_counts.get("easy", 0))
        n_unknown = int(tier_counts.get("hard", 0))
        n_total = len(units)
        return {
            "split": split,
            "n_items": len(units),
            "status_counts": counts,
            "reference_status": majority,
            "reference_accuracy": overall,
            "majority_guess_accuracy": overall,
            "tier_counts": tier_counts,
            "tier_reference_accuracy": tier_acc,
            "pure_z3_known": n_known,
            "pure_z3_unknown": n_unknown,
            "pure_z3_decided_lower_bound": n_known,
            "pure_z3_decided_upper_bound": n_total,
            "pure_z3_accuracy_lower_bound": float(n_known / n_total),
            "pure_z3_accuracy_upper_bound": 1.0,
            "pure_z3_bound_definition": "frozen 1s screen: easy=decided/agrees, hard=unknown; no full-split solver run",
            "representative_scope": "one hash-ranked lineage representative per (lineage,status,tier)",
            "diagnostic_only": True,
            "tool_shortcut_not_capability_evidence": True,
        }

    # ------------------------------------------------------------------------------------------ z3 tool
    def z3_statuses(self, queries: list[str], config: dict) -> dict:
        timeout = float(min(max(float(config.get("query_timeout_s", Z3_DEFAULT_TIMEOUT_S)), 1.0), Z3_MAX_TIMEOUT_S))
        mem = int(min(max(int(config.get("memory_mb", Z3_DEFAULT_MEMORY_MB)), 256), 4096))
        workers = int(min(max(int(config.get("workers", 4)), 1), 8))
        rl = config.get("rlimit")
        rlimit = None if rl in (None, "", 0) else int(rl)
        params = tuple(str(p) for p in (config.get("params") or ()))
        bad = [p for p in params if not _PARAM_RE.match(p)]
        if bad:
            raise ValueError(f"z3_check: invalid params {bad}; expected 'name=value' tokens like 'smt.arith.solver=6'")
        z3 = self._z3.get()
        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(queries)))) as ex:
            res = list(ex.map(lambda q: z3.check(q, timeout, mem, rlimit, params), queries))
        return {"status": [r["status"] for r in res], "time_s": [float(r["time_s"]) for r in res],
                "reason": [r["reason"] for r in res], "z3_version": z3.version}

    # ------------------------------------------------------------------------------------------ episodes
    def _episode(self, data: _SmtData, split: str, seed: int, k: int, unit_ids: list[str]) -> Episode:
        qc = self._queries.get()
        units = [data.units[u] for u in unit_ids]
        n = len(units)
        pool = "ood" if split == "ood" else "iid"
        erng = episode_rng(CODE, split, seed, k, "visible")
        tr_quota = self.quotas("train", seed, k, self.n_train, n_sat=int(round(self.n_train * 0.625)))
        train_ids = _draw_visible(data.pools["train"], tr_quota, erng)
        dev_ids = _draw_visible(data.pools["dev"], self.quotas("dev", seed, k, self.n_dev), erng)
        train_units = [data.units[u] for u in train_ids]
        dev_units = [data.units[u] for u in dev_ids]
        y_true = [u.status for u in units]
        y_train = [u.status for u in train_units]
        y_dev = [u.status for u in dev_units]
        counts = {s: y_train.count(s) for s in ("sat", "unsat")}
        majority = "sat" if counts["sat"] >= counts["unsat"] else "unsat"
        ref_pred = [majority] * n
        ref_acc = float(np.mean([a == b for a, b in zip(y_true, ref_pred)]))
        dev_ref_acc = float(np.mean([a == majority for a in y_dev]))
        nd = len(dev_units)
        ntr = len(train_units)

        def load_train(inputs: dict, config: dict) -> dict:
            return {"queries": [qc.get(u) for u in train_units], "status": list(y_train)}

        def load_dev_inputs(inputs: dict, config: dict) -> dict:
            return {"dev_queries": [qc.get(u) for u in dev_units]}

        def load_eval_inputs(inputs: dict, config: dict) -> dict:
            return {"queries": [qc.get(u) for u in units]}

        def score_dev(inputs: dict, config: dict) -> dict:
            v, why = as_str_list(inputs.get("dev_pred"), nd)
            if v is None:
                raise ValueError(f"score_dev: dev_pred must be a list of {nd} labels ({why})")
            bad = [x for x in v if x not in LABELS]
            if bad:
                raise ValueError(f"score_dev: labels must be in {LABELS}, got e.g. {bad[0]!r}")
            acc = float(np.mean([a == b for a, b in zip(v, y_dev)]))
            wrong = int(sum(p in ("sat", "unsat") and p != t for p, t in zip(v, y_dev)))
            return {"dev_accuracy": acc, "dev_reference_accuracy": dev_ref_acc, "dev_n_wrong_definite": wrong,
                    "n_dev": nd}

        def z3_check(inputs: dict, config: dict) -> dict:
            qs, why = as_str_list(inputs.get("queries"), None)
            if qs is None:
                raise ValueError(f"z3_check: queries must be a list of SMT-LIB strings ({why})")
            return self.z3_statuses(qs, config)

        q_list = lambda m, d: PortSchema("list", (m,), dtype="str", description=d)  # noqa: E731
        tools = [
            ToolSpec("load_train", f"{ntr} labelled training queries (SMT-LIB 2.6 text, metadata removed) and their "
                                   "official statuses ('sat'/'unsat').", {},
                     {"queries": q_list(ntr, "SMT-LIB scripts"), "status": q_list(ntr, "official status")}, load_train),
            ToolSpec("load_dev_inputs", f"{nd} further training-pool queries whose statuses are withheld (score them "
                                        "with score_dev).", {}, {"dev_queries": q_list(nd, "SMT-LIB scripts")},
                     load_dev_inputs),
            ToolSpec("score_dev", "Accuracy of dev_pred (one label per dev query, same order) against the official "
                                  "statuses of the dev queries, plus the reference accuracy.",
                     {"dev_pred": q_list(nd, "'sat'|'unsat'|'unknown'")},
                     {"dev_accuracy": PortSchema("number", unit="1"),
                      "dev_reference_accuracy": PortSchema("number", unit="1"),
                      "dev_n_wrong_definite": PortSchema("number", dtype="int"), "n_dev": PortSchema("number", dtype="int")},
                     score_dev),
            ToolSpec("load_eval_inputs", f"The {n} evaluation queries (SMT-LIB 2.6 text, metadata removed), in output "
                                         "order.", {}, {"queries": q_list(n, "SMT-LIB scripts")}, load_eval_inputs),
            ToolSpec("z3_check", "Runs the Z3 SMT solver (one subprocess per query, in parallel) and returns its "
                                 "answer per query: status 'sat'/'unsat'/'unknown' (unknown = timeout, resource "
                                 "limit, memory limit or error; see reason, e.g. 'timeout', 'memout', 'error: unknown parameter ...') "
                                 "and wall time in seconds.",
                     {"queries": PortSchema("list", ("m",), dtype="str", description="SMT-LIB scripts")},
                     {"status": PortSchema("list", ("m",), dtype="str"), "time_s": PortSchema("list", ("m",), dtype="float", unit="s"),
                      "reason": PortSchema("list", ("m",), dtype="str"), "z3_version": PortSchema("text")},
                     z3_check, config_doc=f"query_timeout_s: z3 time limit per query (1-{Z3_MAX_TIMEOUT_S:g}, default {Z3_DEFAULT_TIMEOUT_S:g}); "
                                          "memory_mb (256-4096, default 2048); workers (1-8, default 4); rlimit "
                                          "(optional deterministic resource limit); params (list of z3 'name=value' "
                                          "parameters)"),
        ]
        constraints = [
            c_str_list(n, "status label per query"),
            c_allowed_labels(n, list(LABELS), description="every y[i] is 'sat', 'unsat' or 'unknown'"),
        ]

        def evaluate(yv: Any, trace: Trace | None) -> EvalResult:
            payload: dict[str, Any] = {"item_ids": [u.item_id for u in units], "y_true": list(y_true),
                                       "y_pred": None, "y_ref": list(ref_pred),
                                       "tiers": [u.tier for u in units]}
            v, why = as_str_list(yv, n)
            if v is not None and any(x not in LABELS for x in v):
                why = "labels outside {sat, unsat, unknown}"
                v = None
            if v is None:
                return EvalResult(metrics={"reference_accuracy": ref_acc}, primary=None, direction="max",
                                  accepted=False, details={"reference": ref_acc, "norm_score": 0.0,
                                                           "pooled_payload": payload, "invalid": why})
            acc = float(np.mean([a == b for a, b in zip(v, y_true)]))
            wrong = int(sum(p in ("sat", "unsat") and p != t for p, t in zip(v, y_true)))
            payload["y_pred"] = list(v)
            # Details-only baselines; failures never alter primary/acceptance.
            pure_z3: list[str] | None = None
            z3_error: str | None = None
            try:
                q_eval = [qc.get(u) for u in units]
                pure_z3 = list(self.z3_statuses(q_eval, {
                    "query_timeout_s": Z3_DEFAULT_TIMEOUT_S, "memory_mb": Z3_DEFAULT_MEMORY_MB,
                    "workers": 4, "params": (),
                })["status"])
            except Exception as ex:  # pragma: no cover - optional z3 runtime/data
                z3_error = f"{type(ex).__name__}: {ex}"
            diag = status_baseline_diagnostics(list(y_true), list(v), list(ref_pred), pure_z3,
                                               [u.tier for u in units])
            if z3_error is not None:
                diag["pure_z3_error"] = z3_error
            if pure_z3 is not None:
                payload["pure_z3"] = list(pure_z3)
            return EvalResult(
                metrics={"accuracy": acc, "reference_accuracy": ref_acc, "n_wrong_definite": float(wrong),
                         "n_unknown": float(sum(x == "unknown" for x in v))},
                primary=acc, direction="max", accepted=bool(acc >= ref_acc + self.accept_margin - 1e-12),
                details={"reference": ref_acc, "norm_score": norm_score(acc, ref_acc, "max"),
                         "diagnostics": diag, "pooled_payload": payload})

        objective = (
            f"Decide the satisfiability status of {n} SMT-LIB 2.6 queries in quantifier-free non-linear integer "
            "arithmetic (logic QF_NIA; rarely QF_NIRA). Each query declares integer (or real) constants and asserts "
            "polynomial constraints, then issues one (check-sat). Deliverable y: a list of exactly "
            f"{n} strings, y[i] in {{'sat', 'unsat', 'unknown'}} for query i of load_eval_inputs, in the same order. "
            "Metric: oracle-agreement accuracy = fraction of queries whose label equals the official SMT-LIB "
            "status (every query has a definite official status, so 'unknown' never agrees). Tools provide labelled "
            "training queries, a dev slice scored by score_dev, the evaluation queries and the Z3 solver with a "
            "per-query timeout. For hard QF_NIA cases, prefer query_timeout_s=120 (the documented maximum) when "
            "the episode budget allows it. For the solver route, call z3_check (do not reimplement it through scilib.logic "
            "inside a code node), wire z3_check.status directly to submit.y and finish after that tool result; do "
            "not overwrite solver outputs with a later majority or code-node approximation. If a second "
            "z3_check call is needed for a timeout retry, wire that latest status output directly to submit.y "
            "and finish; never leave a retry solver node disconnected from the submitted y value.\n"
            + scilib.describe("logic")
        )
        tot_timeout = Z3_MAX_TIMEOUT_S * (n + nd) / 4 + 60
        return Episode(
            id=episode_id(CODE, split, seed, k), discipline=CODE, family=FAMILY, split=split,
            task_type=self.task_type, objective=objective,
            required_output=PortSchema("list", (n,), dtype="str", description="'sat' | 'unsat' | 'unknown' per query"),
            tools=tools, constraints=constraints,
            budget=self.budget or Budget(max_steps=12, max_policy_tokens=200_000, max_wall_s=2400.0,
                                         max_node_s=float(min(900.0, tot_timeout)), max_llm_items=2 * n),
            lineage={"dataset": "SMT-LIB 2025 non-incremental QF_NIA/QF_NIRA (SMT-COMP 2025 QF_NonLinearIntArith)",
                     "version": "2025.05.22 (Zenodo 10.5281/zenodo.15493090)", "pool": pool,
                     "ood_kind": "proxy_within_dataset" if pool == "ood" else None,
                     "ood_shift": "benchmark-family/generator shift (IID families "
                                  f"{list(IID_FAMILIES)} vs all other families)" if pool == "ood" else None,
                     "item_ids": [u.item_id for u in units], "lineage_units": list(unit_ids),
                     "families": sorted({u.family for u in units}), "n_sat": y_true.count("sat"),
                     "tiers": [u.tier for u in units], "screen": {k: v for k, v in data.screen_meta.items()
                                                                   if k in ("z3_version", "cpu_limit_s", "rlimit", "source")},
                     "train_item_ids": [u.item_id for u in train_units], "dev_item_ids": [u.item_id for u in dev_units],
                     "seed": seed, "index": k, "partition_seed": PARTITION_SEED, "max_bytes": MAX_BYTES,
                     "label_source": "SMT-LIB (set-info :status) as published (SMT-COMP scoring label)",
                     "receipt": data.receipt},
            acceptance=f"accuracy >= reference accuracy (majority training status) + {self.accept_margin:g}",
            tolerance={"rtol": 0.0, "atol": 0.0},
            tags=["smt", "logic", "nonlinear_integer_arithmetic", "decision_procedure", "solver"],
            metric=self.metric, direction=self.direction, n_items=n,
            _evaluate=evaluate, _dev_evaluate=None,
        )


Adapter = SmtAdapter


# ================================================================================================ difficulty screen
SCREEN_CPU_S = 1                    # CPU-time limit (RLIMIT_CPU) of the easy-benchmark screen (load-independent)
SCREEN_RLIMIT = 16_000_000          # plus z3's deterministic resource limit (~1 s of z3 5.1 work on an Apple-M core)
SCREEN_GUARD_S = 120.0              # wall-clock guard only (machine load must not decide the tier)


def screen_path() -> Path:
    return cache_dir(CODE) / f"screen_{_CACHE_VERSION}.json"


def run_screen(data_root: str | os.PathLike | None = None, workers: int = 8, save_every: int = 250,
               log: Any = None) -> dict:
    """SMT-COMP-style easy-benchmark screen: z3 (1 s CPU limit + ``rlimit``) once on every eligible item.

    Results (``item_id -> {"z3": status, "reason": ...}`` + meta) are written incrementally to
    ``cache/tasks/FoR49/screen_v1.json`` and the run resumes from an existing file. An item is *hard* when z3
    does not decide it within the resource limit; *conflict* when z3 decides it against the official status.
    """
    root = resolve_data_root(data_root)
    rows = _eligible_rows(root)
    z3 = Z3Runner(find_z3() or "z3")
    path = screen_path()
    state = read_json(path) if path.exists() else {}
    meta = {"z3_version": z3.version, "cpu_limit_s": SCREEN_CPU_S, "rlimit": SCREEN_RLIMIT, "guard_s": SCREEN_GUARD_S,
            "memory_mb": Z3_DEFAULT_MEMORY_MB,
            "max_bytes": MAX_BYTES}
    if state.get("meta", {}).get("z3_version") not in (None, z3.version):
        raise RuntimeError(f"screen file was made with {state['meta']['z3_version']}, current z3 is {z3.version}")
    res: dict[str, dict] = state.get("items", {})
    order = {i: k for k, i in enumerate(hash_rank([r["instance_id"] for r in rows], f"{CODE}|{PARTITION_SEED}|screen"))}
    todo = sorted((r for r in rows if r["instance_id"] not in res), key=lambda r: order[r["instance_id"]])
    d = root / DATASET_DIR
    lock = threading.Lock()
    done = [0]

    def one(r: dict) -> None:
        q = sanitize_query((d / r["path"]).read_text(encoding="utf-8", errors="replace"))
        out = z3._run(q, SCREEN_GUARD_S, Z3_DEFAULT_MEMORY_MB, SCREEN_RLIMIT, (), cpu_limit_s=SCREEN_CPU_S)
        with lock:
            res[r["instance_id"]] = {"z3": out["status"], "reason": out["reason"][:80],
                                     "time_s": round(float(out["time_s"]), 3)}
            done[0] += 1
            if done[0] % save_every == 0:
                atomic_write_json(path, {"meta": meta, "items": res})
                if log:
                    log(f"screened {len(res)}/{len(rows)}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, todo))
    atomic_write_json(path, {"meta": {**meta, "complete": True}, "items": res})
    return {"meta": meta, "n": len(res)}


if __name__ == "__main__":      # python -m scienceclaw.bench.tasks.for49_smt screen [workers]
    if len(sys.argv) >= 2 and sys.argv[1] == "screen":
        print(run_screen(workers=int(sys.argv[2]) if len(sys.argv) > 2 else 8, log=lambda s: print(s, flush=True)))
    elif len(sys.argv) >= 2 and sys.argv[1] == "freeze":
        print(freeze_screen())
