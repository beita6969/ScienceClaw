"""Batched Z3 driver for SMT-LIB satisfiability queries (QF_NIA and related logics), run inside the code node.

check(query, timeout_s=10, rlimit=None, memory_mb=None, params=None) -> dict
    One query, one Z3 Python-API solver in its own context. Returns {"status": 'sat'|'unsat'|'unknown',
    "reason": Z3's reason for 'unknown' (e.g. 'timeout', 'max. resource limit exceeded', 'out of memory',
    'error: ...' for a script or parameter Z3 rejects; '' when decided), "seconds": wall time}.
    The solver is built from the script's (set-logic ...) (Z3's default solver when absent). ``params``: dict of Z3
    solver parameters ({'smt.arith.solver': 6}); a name Z3 does not know gives status 'unknown', reason 'error: ...'.
    ``timeout_s`` is wall-clock time; ``rlimit`` is Z3's resource counter (deterministic; the wall time per unit
    varies with the query). Whichever limit is reached first ends the call.
solve_all(queries, threads=2, timeout_s=10, rlimit=None, memory_mb=2048, params=None) -> list[dict]
    ``check`` for every query, ``threads`` queries at a time (Z3 releases the GIL; one context per query).
majority_status(train_status) -> 'sat' | 'unsat'
    The more frequent of 'sat' / 'unsat' in a list of statuses (ties: 'sat').
decide(queries, train_status=None, fallback=None, budget_s=240, threads=2, first_rlimit=16_000_000, first_timeout_s=8,
       factor=4, max_rounds=4, memory_mb=2048) -> dict
    Escalating schedule over the whole batch. Round r checks every still-undecided query with
    rlimit = first_rlimit * factor**r and a wall limit of min(first_timeout_s * factor**r, the remaining share of
    ``budget_s``): the remaining budget is divided over the queries still waiting in the round (``threads`` run at a
    time, cheapest-so-far first), so one query cannot use the budget of the others. Stops when every query is
    decided, after ``max_rounds`` rounds or when ``budget_s`` (wall-clock seconds for the whole call) is used up; the
    call returns within about ``budget_s`` + 5 s (a Z3 call still running then is abandoned and its query stays
    undecided). The node's own time limit must exceed ``budget_s``.
    A returned 'sat' / 'unsat' is Z3's own answer for that query (no bounded or heuristic approximation); the
    queries Z3 did not decide get the ``fallback`` label: 'sat' | 'unsat' | 'unknown', or 'majority' = the more
    frequent status in ``train_status`` (default: 'majority' when ``train_status`` is given, else 'unknown').
    Returns a dict:
      labels    list of str, one per query: Z3's status if decided, else the fallback label
      status    list of str, Z3's status ('unknown' if undecided)
      stage     list of int, the round (0-based) that decided the query, -1 if undecided
      seconds   list of float, wall seconds spent on the query over all rounds
      reason    list of str, Z3's reason from the last attempt ('' if decided)
      n_decided, fallback_label, wall_s

Scripts are parsed with z3.parse_smt2_string, so ``(check-sat)`` and ``(set-info ...)`` commands are ignored and only the
asserted formulas are solved. The scheduling is deterministic given the rlimit schedule; the wall limits make the number
of decided queries depend on the CPU time the process actually gets. ``memory_mb`` is Z3's process-wide memory cap
(a query that exceeds it ends as 'unknown', reason 'out of memory'). Example:
    r = decide(queries, train_status=train_status); y = r["labels"]
"""
from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

__all__ = ["check", "solve_all", "majority_status", "decide"]

_LABELS = ("sat", "unsat", "unknown")
_GRACE_S = 2.0
_LOGIC_RE = re.compile(r"\(\s*set-logic\s+([A-Za-z_]+)\s*\)")
_CORES_RE = re.compile(r"\(\s*set-option\s+:produce-unsat-(?:cores|assumptions)\s+(?:true|false)\s*\)")


def _z3():
    import z3
    return z3


def _set_memory(memory_mb: int | None) -> None:
    if memory_mb:
        _z3().set_param("memory_max_size", int(memory_mb))


def check(query: str, timeout_s: float = 10.0, rlimit: int | None = None, memory_mb: int | None = None,
          params: dict | None = None) -> dict:
    """Status of one SMT-LIB script (see the module docstring)."""
    z3 = _z3()
    t0 = time.perf_counter()
    status, reason = "unknown", ""
    try:
        _set_memory(memory_mb)
        ctx = z3.Context()
        # with unsat-core tracking on, the parser turns a named assertion into (=> name F), which a plain solver may leave unsatisfied
        fs = z3.parse_smt2_string(_CORES_RE.sub("", str(query)), ctx=ctx)
        m = _LOGIC_RE.search(str(query))
        try:
            s = z3.SolverFor(m.group(1), ctx=ctx) if m else z3.Solver(ctx=ctx)
        except z3.Z3Exception:
            s = z3.Solver(ctx=ctx)
        if timeout_s:
            s.set("timeout", max(1, int(float(timeout_s) * 1000)))
        if rlimit:
            s.set("rlimit", int(rlimit))
        for k, v in (params or {}).items():
            s.set(str(k), v)
        s.add(*fs)
        dog = None
        if timeout_s:  # backstop: interrupt the context if Z3's own timeout does not stop the search
            dog = threading.Timer(float(timeout_s) + _GRACE_S, ctx.interrupt)
            dog.daemon = True
            dog.start()
        try:
            r = s.check()
        finally:
            if dog is not None:
                dog.cancel()
        if r == z3.sat:
            status = "sat"
        elif r == z3.unsat:
            status = "unsat"
        else:
            reason = str(s.reason_unknown()) or "unknown"
    except Exception as e:  # parse error, rejected parameter, z3 internal error
        reason = f"error: {str(e).strip().splitlines()[0][:200] if str(e).strip() else type(e).__name__}"
    return {"status": status, "reason": reason, "seconds": time.perf_counter() - t0}


def solve_all(queries, threads: int = 2, timeout_s: float = 10.0, rlimit: int | None = None,
              memory_mb: int | None = 2048, params: dict | None = None) -> list[dict]:
    """``check`` for every query, ``threads`` at a time; result i belongs to queries[i]."""
    qs = list(queries)
    if not qs:
        return []
    _set_memory(memory_mb)
    with ThreadPoolExecutor(max_workers=max(1, min(int(threads), len(qs)))) as ex:
        return list(ex.map(lambda q: check(q, timeout_s, rlimit, None, params), qs))


def majority_status(train_status) -> str:
    st = [str(s) for s in train_status]
    return "sat" if st.count("sat") >= st.count("unsat") else "unsat"


def decide(queries, train_status=None, fallback: str | None = None, budget_s: float = 240.0, threads: int = 2,
           first_rlimit: int = 16_000_000, first_timeout_s: float = 8.0, factor: float = 4.0, max_rounds: int = 4,
           memory_mb: int | None = 2048) -> dict:
    """Escalating batch schedule with Z3 (see the module docstring)."""
    qs = [str(q) for q in queries]
    n = len(qs)
    if fallback is None:
        fallback = "majority" if train_status is not None and len(train_status) else "unknown"
    if fallback == "majority":
        if train_status is None or not len(train_status):
            raise ValueError("decide: fallback='majority' needs train_status")
        fallback = majority_status(train_status)
    if fallback not in _LABELS:
        raise ValueError(f"decide: fallback must be one of {_LABELS} or 'majority', got {fallback!r}")
    _set_memory(memory_mb)
    status = ["unknown"] * n
    stage = [-1] * n
    seconds = [0.0] * n
    reason = ["not attempted"] * n
    t_start = time.perf_counter()
    deadline = t_start + float(budget_s)
    threads = max(1, int(threads))

    closed = threading.Event()
    lock = threading.Lock()
    for rnd in range(int(max_rounds)):
        pending = [i for i in range(n) if status[i] == "unknown"]
        if not pending or deadline - time.perf_counter() < 0.5:
            break
        pending.sort(key=lambda i: (seconds[i], i))
        rlimit = int(first_rlimit * factor ** rnd)
        cap = float(first_timeout_s) * factor ** rnd
        queue = list(pending)

        def worker(rnd=rnd, rlimit=rlimit, cap=cap, queue=queue) -> None:
            while not closed.is_set():
                with lock:
                    if not queue:
                        return
                    i = queue.pop(0)
                    left = deadline - time.perf_counter()
                    # remaining budget divided over the queries still waiting (``threads`` run at a time), never
                    # more than what is left
                    share = min(left, left * threads / (len(queue) + 1))
                if left < 0.5:
                    continue
                res = check(qs[i], min(cap, max(0.5, share)), rlimit)
                with lock:
                    if closed.is_set():
                        return
                    seconds[i] += res["seconds"]
                    reason[i] = res["reason"]
                    if res["status"] != "unknown":
                        status[i] = res["status"]
                        stage[i] = rnd

        workers = [threading.Thread(target=worker, daemon=True) for _ in range(min(threads, len(pending)))]
        for w in workers:
            w.start()
        for w in workers:  # hard stop: a Z3 call that ignores its limits is abandoned (its query stays undecided)
            w.join(max(0.0, deadline + 2 * _GRACE_S - time.perf_counter()))
        if any(w.is_alive() for w in workers):
            break
    with lock:
        closed.set()
        status, stage, seconds, reason = list(status), list(stage), list(seconds), list(reason)

    labels = [s if s != "unknown" else fallback for s in status]
    return {"labels": labels, "status": status, "stage": stage, "seconds": seconds, "reason": reason,
            "n_decided": sum(s != "unknown" for s in status), "fallback_label": fallback,
            "wall_s": time.perf_counter() - t_start}
