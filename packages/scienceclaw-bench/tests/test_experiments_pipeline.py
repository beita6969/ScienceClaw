"""Experiment pipeline integrity: program de-duplication, infrastructure retries, usage ledger and resume
accumulation, provenance receipts, logical vs billed tokens, coverage-aware reporting.

Uses a local stub Evolver / Solver, the synthetic adapter and no LLM.
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scienceclaw.bench.metrics import (coverage_gaps, expected_coverage, performance_index)
from synthetic_adapter import SyntheticAdapter, synthetic_truth
from scienceclaw.config import BenchConfig, RunConfig
from scienceclaw.core.program import AgentProgram, Bundle
from scienceclaw.core.skills import Skill
from scienceclaw.experiments import provenance as P
from scienceclaw.experiments.evaluate import (InfraError, evaluate_snapshots, read_results, run_jobs, _solve_job)
from scienceclaw.experiments.report import (build_report, expected_by_discipline, logical_tokens_of, row_costs)
from scienceclaw.experiments.run_stream import run_stream


# ------------------------------------------------------------------------------------------------- stubs
class CountingEvolver:
    """Adds one Skill per round and 'spends' tokens on the injected llm (so the ledger has something to add up)."""

    def __init__(self, cfg, llm, plan, run_dir):
        self.cfg, self.llm, self.plan, self.run_dir = cfg, llm, plan, Path(run_dir)

    def run(self, program0):
        snaps, prog = [program0], program0
        for r, ep in self.plan.source_stream():
            sk = Skill(id=f"s{r}", version=1, title=f"skill {r}", body="b", created=f"r{r}:e{ep.id}")
            prog, _ = prog.apply(Bundle(skills=[sk]), new_version=f"A{r}")
            snaps.append(prog)
        if hasattr(self.llm, "spend"):
            self.llm.spend(100)
        time.sleep(0.01)
        return snaps


class MeterLLM:
    """Minimal llm with a cumulative ``usage()`` like LLMClient."""

    def __init__(self, cache_path="cache.sqlite"):
        self.n = 0
        self.cache_path = cache_path

    def spend(self, k):
        self.n += k

    def usage(self):
        return {"total": {"calls": self.n // 10, "prompt_tokens": self.n, "completion_tokens": self.n // 2},
                "by_role": {"policy": {"prompt_tokens": self.n}}}


class Solver:
    """Stub solver: true function iff the program has a Skill; usage carries policy/executor token fields."""

    calls: list = []
    lock = threading.Lock()

    def solve(self, episode, program, mode, run_dir):
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        with Solver.lock:
            Solver.calls.append((program.fingerprint(), episode.split, episode.id))
        X = episode.tool("load_eval_inputs").fn({}, {})["X_eval"]
        yt = episode.tool("load_train").fn({}, {})["y_train"]
        y = synthetic_truth(X) if program.skills else np.full(len(X), yt.mean())
        ev = episode.evaluate(y, None)
        ev.reproducible = True
        usage = {"total_tokens": 10, "policy_logical_tokens": 100, "executor_prompt_tokens": 7,
                 "executor_cached_prompt_tokens": 3, "llm_calls": 1, "wall_s": 0.01}
        return SimpleNamespace(eval=ev, usage=usage, program_version=program.version, stop_reason="finish",
                               save=lambda path: (Path(path) / "eval.json").write_text(json.dumps({"z": ev.z})))


def _programs():
    base = AgentProgram()
    skilled, _ = base.apply(Bundle(skills=[Skill(id="s1", version=1, title="t", body="b")]), new_version="A1")
    return base, skilled


def _episodes(n=2, split="id"):
    return SyntheticAdapter().build_episodes(split, n, 3, 6)


def _jobs(snaps, eps, split="id"):
    return [{"snapshot": s, "split": split, "episode": e} for s in snaps for e in eps]


def _cfg(tmp_path, rounds=2):
    cfg = RunConfig(name="t", runs_root=str(tmp_path / "runs"))
    cfg.bench = BenchConfig(disciplines=["SYN"], items_per_episode=6, rounds=rounds, n_val=1, n_id=2, n_ood=1, seed=3)
    return cfg


@pytest.fixture()
def counting_evolver(monkeypatch):
    mod = types.ModuleType("scienceclaw.evolution.evolver")
    mod.Evolver = CountingEvolver
    monkeypatch.setitem(sys.modules, "scienceclaw.evolution.evolver", mod)


@pytest.fixture()
def run_dir(tmp_path, counting_evolver):
    return run_stream(_cfg(tmp_path), adapters={"SYN": SyntheticAdapter()}, llm=MeterLLM())


# ---------------------------------------------------------------------------------- program de-duplication
def test_run_jobs_aliases_identical_programs(tmp_path):
    base, _ = _programs()
    twin = copy.deepcopy(base)
    twin.version = "A9"
    assert twin.fingerprint() == base.fingerprint()
    eps = _episodes(2)
    Solver.calls = []
    out = run_jobs(_jobs(["A_0", "A_9"], eps), {"A_0": base, "A_9": twin}, Solver, tmp_path / "eval", workers=1)
    assert out["done"] == 2 and out["aliased"] == 2 and len(Solver.calls) == 2
    rows = {(r["snapshot"], r["episode"]): r for r in read_results(tmp_path / "eval" / "results.jsonl")}
    assert len(rows) == 4
    for e in eps:
        root, al = rows[("A_0", e.id)], rows[("A_9", e.id)]
        assert "alias_of" not in root and al["alias_of"] == "A_0"
        assert al["usage"] == {} and al["wall_s"] == 0.0 and al["alias_usage"] == root["usage"]
        assert al["z"] == root["z"] and al["pooled_payload"] == root["pooled_payload"]
        assert al["program_version"] == "A9" and al["program_fingerprint"] == root["program_fingerprint"]


def test_run_jobs_concurrent_duplicates_are_solved_once(tmp_path):
    base, _ = _programs()
    progs = {f"A_{i}": copy.deepcopy(base) for i in range(4)}
    eps = _episodes(2)
    gate = threading.Event()
    inflight, peak = [0], [0]
    lock = threading.Lock()

    class Slow(Solver):
        def solve(self, episode, program, mode, run_dir):
            with lock:
                inflight[0] += 1
                peak[0] = max(peak[0], inflight[0])
            gate.wait(0.2)
            try:
                return super().solve(episode, program, mode, run_dir)
            finally:
                with lock:
                    inflight[0] -= 1

    Solver.calls = []
    out = run_jobs(_jobs(list(progs), eps), progs, Slow, tmp_path / "eval", workers=8)
    assert len(Solver.calls) == len(eps) and out["done"] == len(eps) and out["aliased"] == 3 * len(eps)
    assert sorted(c[2] for c in Solver.calls) == sorted(e.id for e in eps)      # one solve per episode, not per snapshot


def test_run_jobs_owner_failure_defers_aliases_and_resume_retries(tmp_path):
    base, _ = _programs()
    progs = {"A_0": base, "A_1": copy.deepcopy(base)}
    eps = _episodes(2)

    class Boom(Solver):
        def solve(self, *a, **k):
            raise RuntimeError("transient")

    ed = tmp_path / "eval"
    out = run_jobs(_jobs(list(progs), eps), progs, Boom, ed, workers=1)
    assert out["failed"] == 2 and out["deferred"] == 2 and out["done"] == 0 and out["aliased"] == 0
    assert read_results(ed / "results.jsonl") == []
    Solver.calls = []
    out = run_jobs(_jobs(list(progs), eps), progs, Solver, ed, workers=1)
    assert out["done"] == 2 and out["aliased"] == 2 and out["failed"] == 0 and len(Solver.calls) == 2


def test_run_jobs_reuses_root_rows_across_calls_and_dedup_off(tmp_path):
    base, _ = _programs()
    twin = copy.deepcopy(base)
    eps = _episodes(2)
    ed = tmp_path / "eval"
    Solver.calls = []
    run_jobs(_jobs(["A_0"], eps), {"A_0": base}, Solver, ed, workers=1)
    assert len(Solver.calls) == 2
    out = run_jobs(_jobs(["A_0", "A_5"], eps), {"A_0": base, "A_5": twin}, Solver, ed, workers=1)
    assert out["skipped"] == 2 and out["aliased"] == 2 and out["done"] == 0 and len(Solver.calls) == 2
    ed2 = tmp_path / "eval2"
    Solver.calls = []
    out = run_jobs(_jobs(["A_0", "A_5"], eps), {"A_0": base, "A_5": twin}, Solver, ed2, workers=1, dedup=False)
    assert out["done"] == 4 and out["aliased"] == 0 and len(Solver.calls) == 4


def test_run_jobs_different_programs_are_not_aliased(tmp_path):
    base, skilled = _programs()
    Solver.calls = []
    out = run_jobs(_jobs(["A_0", "A_1"], _episodes(2)), {"A_0": base, "A_1": skilled}, Solver, tmp_path / "eval",
                   workers=1)
    assert out["done"] == 4 and out["aliased"] == 0 and len(Solver.calls) == 4


# ---------------------------------------------------------------------------------- infrastructure stop reasons
def test_policy_error_is_an_error_not_a_result(tmp_path):
    base, _ = _programs()
    ep = _episodes(1)[0]

    class Broken(Solver):
        def solve(self, episode, program, mode, run_dir):
            res = super().solve(episode, program, mode, run_dir)
            res.stop_reason = "policy_error"
            return res

    ed = tmp_path / "eval"
    with pytest.raises(InfraError) as ei:
        _solve_job({"snapshot": "A_0", "split": "id", "episode": ep}, base, Broken, ed)
    assert ei.value.stop_reason == "policy_error" and ei.value.usage["total_tokens"] == 10
    out = run_jobs(_jobs(["A_0"], [ep]), {"A_0": base}, Broken, ed, workers=1)
    assert out["failed"] == 1 and out["done"] == 0
    assert read_results(ed / "results.jsonl") == []
    err = [json.loads(x) for x in (ed / "errors.jsonl").read_text().splitlines()][-1]
    assert err["infra"] is True and err["stop_reason"] == "policy_error" and err["usage"]["total_tokens"] == 10
    assert err["receipt_dir"] and (ed / err["receipt_dir"]).is_dir()       # the failed attempt stays inspectable
    out = run_jobs(_jobs(["A_0"], [ep]), {"A_0": base}, Solver, ed, workers=1)      # a resume retries it
    assert out["done"] == 1 and len(read_results(ed / "results.jsonl")) == 1


def test_finished_failures_are_results_not_errors(tmp_path):
    """Only infrastructure stop reasons are retried: an agent that ran out of budget really failed the episode."""
    base, _ = _programs()
    ep = _episodes(1)[0]

    class OutOfSteps(Solver):
        def solve(self, episode, program, mode, run_dir):
            res = super().solve(episode, program, mode, run_dir)
            res.stop_reason = "step_budget"
            return res

    out = run_jobs(_jobs(["A_0"], [ep]), {"A_0": base}, OutOfSteps, tmp_path / "eval", workers=1)
    assert out["done"] == 1 and out["failed"] == 0
    assert read_results(tmp_path / "eval" / "results.jsonl")[0]["stop_reason"] == "step_budget"


# ---------------------------------------------------------------------------------- uniform failure payload
def test_solve_without_payload_falls_back_to_the_failure_payload(tmp_path):
    base, _ = _programs()
    ep = _episodes(1)[0]

    class NoPayload(Solver):
        def solve(self, episode, program, mode, run_dir):
            res = super().solve(episode, program, mode, run_dir)
            res.eval = SimpleNamespace(z=0, primary=None, details={}, accepted=False, completed=False, h={})
            return res

    ed = tmp_path / "eval"
    row = _solve_job({"snapshot": "A_0", "split": "id", "episode": ep}, base, NoPayload, ed)
    assert row["payload_source"] == "failure_fallback" and row["failed"] is True and row["z"] == 0
    payload = json.loads((ed / row["pooled_payload"]).read_text())
    assert payload["y_pred"] is None and len(payload["y_true"]) == len(payload["y_ref"])   # SYN failure payload
    ok = _solve_job({"snapshot": "A_1", "split": "id", "episode": ep}, base, Solver, ed)
    assert ok["payload_source"] == "solve" and ok["failed"] is False


def test_episode_evaluate_returns_a_payload_for_failures():
    ad = SyntheticAdapter()
    ep = ad.build_episodes("id", 1, 3, 6)[0]
    for bad in (None, "garbage", [1.0, 2.0], [float("nan")] * 6, {"y": 1}):
        res = ep.evaluate(bad, None)
        assert res.details["pooled_payload"] is not None and res.details["failed"] is True, bad
        assert res.z == 0 and res.details["norm_score"] == 0.0
    pooled = ad.pooled_metric([ep.evaluate(None, None).details["pooled_payload"]])
    assert pooled == pytest.approx(ep.evaluate(None, None).details["reference"])      # reference level, not better


# ---------------------------------------------------------------------------------- coverage-aware PI
def test_expected_coverage_counts_distinct_episodes_and_missing_ones(tmp_path):
    rs = [{"episode": "e1", "pooled_payload": {"a": 1}}, {"episode": "e1", "pooled_payload": {"a": 1}},
          {"episode": "e2", "pooled_payload": None}]
    cov = expected_coverage({"D": rs}, {"D": 3, "E": 2}, tmp_path)
    assert cov == {"D": (1, 3), "E": (0, 2)}
    assert coverage_gaps(cov) == {"D": (1, 3), "E": (0, 2)}
    assert coverage_gaps({"D": (3, 3)}) == {}


def test_performance_index_flags_incomplete_coverage():
    scores = {"ref": {"a": 2.0, "b": 4.0}, "m": {"a": 1.0, "b": 2.0}}
    dirs = {"a": "min", "b": "min"}
    full = {"ref": {"a": (4, 4), "b": (4, 4)}, "m": {"a": (4, 4), "b": (4, 4)}}
    pi = performance_index(scores, "ref", dirs, coverage=full)
    assert pi["ref"] == pytest.approx(100.0) and pi["m"] == pytest.approx(200.0) and not pi.incomplete
    part = {"ref": {"a": (4, 4), "b": (4, 4)}, "m": {"a": (4, 4), "b": (3, 4)}}
    pi = performance_index(scores, "ref", dirs, coverage=part)
    assert np.isnan(pi["m"]) and pi.incomplete == {"m": {"b": (3, 4)}} and pi["ref"] == pytest.approx(100.0)
    assert pi.to_dict()["incomplete"] == {"m": {"b": [3, 4]}}
    ref_part = {"ref": {"a": (4, 4), "b": (2, 4)}, "m": {"a": (4, 4), "b": (4, 4)}}
    pi = performance_index(scores, "ref", dirs, coverage=ref_part)
    assert "b" in pi.excluded and "coverage" in pi.excluded["b"] and pi["m"] == pytest.approx(200.0)
    assert performance_index(scores, "ref", dirs)["m"] == pytest.approx(200.0)         # coverage is optional


def test_expected_by_discipline_from_manifest(run_dir):
    man = json.loads((run_dir / "splits.json").read_text())
    assert expected_by_discipline(man, "id") == {"SYN": 2} and expected_by_discipline(man, "ood") == {"SYN": 1}
    assert expected_by_discipline(man, "rep", "A_0") == {}
    assert expected_by_discipline(man, "rep", "A_2") == {"SYN": 2}
    assert expected_by_discipline({}, "id") is None


# ---------------------------------------------------------------------------------- logical vs billed tokens
def test_logical_tokens_and_row_costs():
    u = {"total_tokens": 10, "policy_logical_tokens": 100, "executor_prompt_tokens": 7,
         "executor_completion_tokens": 1, "executor_cached_prompt_tokens": 3, "executor_cached_completion_tokens": 2}
    assert logical_tokens_of(u) == 113
    old = {"policy_prompt_tokens": 40, "policy_completion_tokens": 10, "policy_cached_prompt_tokens": 5,
           "executor_prompt_tokens": 1}
    assert logical_tokens_of(old) == 56 and logical_tokens_of({}) == 0
    rows = [{"usage": u}, {"usage": {}, "alias_of": "A_0", "alias_usage": u}, {"usage": u}]
    spent, logical, n_alias = row_costs(rows)
    assert spent["total_tokens"] == 20 and logical == 3 * 113 and n_alias == 1


def test_report_separates_logical_billed_and_alias_rows(run_dir):
    """A_3 is a copy of A_2: its rows are aliases (nothing billed), yet its logical cost equals A_2's."""
    shutil.copytree(run_dir / "programs" / "A_2", run_dir / "programs" / "A_3")
    Solver.calls = []
    ads = {"SYN": SyntheticAdapter()}
    evaluate_snapshots(run_dir, "all", adapters=ads, solver_factory=Solver, workers=2, include_rep=False)
    rows = read_results(run_dir / "eval" / "results.jsonl")
    a3 = [r for r in rows if r["snapshot"] == "A_3"]
    assert a3 and all(r["alias_of"] in ("A_2",) for r in a3)
    assert len(Solver.calls) == 3 * 3                                # A_0, A_1, A_2 x (2 id + 1 ood); A_3 not solved
    rep = json.loads((build_report(run_dir, adapters=ads, n_boot=20).parent / "report.json").read_text())
    s = {(r["snapshot"], r["split"]): r for r in rep["summary"]}
    assert s[("A_3", "id")]["n_alias"] == 2 and s[("A_3", "id")]["billed_tokens"] == 0
    assert s[("A_3", "id")]["logical_tokens"] == s[("A_2", "id")]["logical_tokens"] == 2 * 110
    assert s[("A_2", "id")]["billed_tokens"] == 20 and s[("A_2", "id")]["total_tokens"] == 20
    assert s[("A_3", "id")]["macro_sr"] == s[("A_2", "id")]["macro_sr"]
    assert any("alias" in w for w in rep["warnings"])
    assert "logical_tokens" in (run_dir / "report" / "costs.csv").read_text().splitlines()[0]


def test_report_marks_incomplete_snapshot_pi_as_nan(run_dir):
    ads = {"SYN": SyntheticAdapter()}
    evaluate_snapshots(run_dir, "all", adapters=ads, solver_factory=Solver, workers=1, include_rep=False)
    path = run_dir / "eval" / "results.jsonl"
    lines = path.read_text().splitlines()
    drop = next(i for i, ln in enumerate(lines) if json.loads(ln)["snapshot"] == "A_2"
                and json.loads(ln)["split"] == "id")
    path.write_text("\n".join(ln for i, ln in enumerate(lines) if i != drop) + "\n")       # A_2 lost one id episode
    rep = json.loads((build_report(run_dir, adapters=ads, n_boot=20).parent / "report.json").read_text())
    s = {(r["snapshot"], r["split"]): r for r in rep["summary"]}
    assert s[("A_2", "id")]["complete"] is False and s[("A_2", "id")]["incomplete_disciplines"] == ["SYN"]
    assert s[("A_2", "id")]["n_episodes"] == 1 and s[("A_2", "id")]["n_expected"] == 2
    assert s[("A_2", "id")]["pi"] != s[("A_2", "id")]["pi"]                                # NaN
    assert s[("A_0", "id")]["complete"] is True and s[("A_0", "id")]["pi"] == pytest.approx(100.0)
    assert s[("A_2", "ood")]["complete"] is True and s[("A_2", "ood")]["pi"] > 150.0
    assert any("coverage incomplete" in w and "id/A_2" in w for w in rep["warnings"])


# ---------------------------------------------------------------------------------- usage ledger / resume
def test_usage_ledger_records_segments_and_detects_killed_ones(tmp_path):
    llm = MeterLLM()
    with P.UsageLedger(tmp_path, "evolve", llm, {"phase": "evolve"}, extra={"resume": False}) as led:
        llm.spend(100)
    assert led.record["status"] == "done" and led.record["llm"]["total"]["prompt_tokens"] == 100
    with pytest.raises(ValueError):
        with P.UsageLedger(tmp_path, "evolve", llm, {}) as _:
            llm.spend(50)
            raise ValueError("x")
    recs = P.read_ledger(tmp_path / P.UsageLedger.FILE)
    assert [r["status"] for r in recs] == ["started", "done", "started", "failed"]
    # a killed process leaves only the marker
    killed = P.UsageLedger(tmp_path, "evaluate", llm, {})
    killed.__enter__()
    tot = P.sum_ledger(P.read_ledger(tmp_path / P.UsageLedger.FILE))
    assert tot["segments"] == 2 and tot["llm"]["total"]["prompt_tokens"] == 150
    assert tot["unclosed"] == [killed.segment] and len(tot["failed"]) == 1
    assert tot["by_phase"]["evolve"]["segments"] == 2 and tot["wall_s"] > 0


def test_usage_delta_and_add_trees():
    a = {"total": {"prompt_tokens": 10, "calls": 3}, "by_role": {"p": {"x": 4}}, "name": "s"}
    b = {"total": {"prompt_tokens": 4}}
    d = P.usage_delta(a, b)
    assert d["total"] == {"prompt_tokens": 6, "calls": 3} and d["by_role"] == {"p": {"x": 4}}
    s = P.add_usage_trees(d, d)
    assert s["total"]["prompt_tokens"] == 12 and s["by_role"]["p"]["x"] == 8


def test_resume_accumulates_usage_instead_of_overwriting(tmp_path, counting_evolver):
    llm = MeterLLM()
    rd = run_stream(_cfg(tmp_path), adapters={"SYN": SyntheticAdapter()}, llm=llm)
    first = json.loads((rd / "run.json").read_text())
    llm2 = MeterLLM()                      # a new process: its client counters start at zero
    run_stream(None, adapters={"SYN": SyntheticAdapter()}, llm=llm2, resume_dir=rd)
    recs = [r for r in P.read_ledger(rd / "usage.jsonl") if r["status"] != "started"]
    assert len(recs) == 2 and all(r["phase"] == "evolve" for r in recs)
    usage = json.loads((rd / "usage.json").read_text())
    assert usage["llm"]["total"]["prompt_tokens"] == 200 and usage["segments"] == 2 and usage["unclosed"] == []
    run = json.loads((rd / "run.json").read_text())
    assert run["wall_s"] == pytest.approx(sum(r["wall_s"] for r in recs)) and run["wall_s"] > first["wall_s"]
    assert run["segments"] == 2 and len(run["resumes"]) == 1
    assert run["provenance"]["phase"] == "evolve" and run["resumes"][0]["provenance"]["resume"] is True
    assert run["provenance"]["split_sha256"] == run["split_sha256"]


def test_resume_imports_a_legacy_usage_json(tmp_path, counting_evolver):
    llm = MeterLLM()
    rd = run_stream(_cfg(tmp_path), adapters={"SYN": SyntheticAdapter()}, llm=llm)
    (rd / "usage.jsonl").unlink()                                         # a run from before the ledger existed
    (rd / "usage.json").write_text(json.dumps({"llm": {"total": {"prompt_tokens": 1000}}, "wall_s": 5.0}))
    run_stream(None, adapters={"SYN": SyntheticAdapter()}, llm=MeterLLM(), resume_dir=rd)
    usage = json.loads((rd / "usage.json").read_text())
    assert usage["llm"]["total"]["prompt_tokens"] == 1100 and usage["segments"] == 2
    recs = P.read_ledger(rd / "usage.jsonl")
    assert recs[0]["status"] == "legacy" and "pre-ledger" in recs[0]["note"]
    assert json.loads((rd / "run.json").read_text())["wall_s"] >= 5.0


# ---------------------------------------------------------------------------------- provenance receipts
def test_provenance_receipt_fields(tmp_path):
    cfg = _cfg(tmp_path)
    prov = P.provenance(cfg, "evaluate", llm=MeterLLM("/x/cache.sqlite"), extra={"eval_workers": 4})
    assert prov["phase"] == "evaluate" and prov["eval_workers"] == 4
    code = prov["code"]
    assert code["python"] and "packages" in code and "numpy" in code["packages"]
    assert ("git_commit" in code and isinstance(code["git_dirty"], bool)) or "git_error" in code
    assert prov["config_sha256"] == P.config_hash(cfg) and len(prov["config_sha256"]) == 64
    assert prov["llm"]["cache_path"] == "/x/cache.sqlite" and prov["llm"]["concurrency"] == cfg.llm.concurrency
    assert "policy" in prov["llm"]["roles"] and prov["llm"]["cache_salt"]
    assert prov["bench"]["seed"] == 3 and "data" in prov
    assert "api_key" not in json.dumps(prov).lower()
    json.dumps(prov)
    cfg2 = _cfg(tmp_path)
    cfg2.bench.seed = 4
    assert P.config_hash(cfg2) != P.config_hash(cfg)


def test_dirty_tree_is_recorded_with_a_diff_hash(tmp_path):
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.email=a@b.c", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "x"],
                   cwd=tmp_path, check=True)
    clean = P.code_version(tmp_path)
    assert clean["git_dirty"] is False and len(clean["git_commit"]) == 40 and "git_diff_sha256" not in clean
    (tmp_path / "new.py").write_text("x = 1\n")
    d1 = P.code_version(tmp_path)
    assert d1["git_dirty"] is True and d1["git_commit"] == clean["git_commit"] and len(d1["git_diff_sha256"]) == 64
    (tmp_path / "new.py").write_text("x = 2\n")
    assert P.code_version(tmp_path)["git_diff_sha256"] != d1["git_diff_sha256"]


def test_dataset_hashes_cover_manifests_and_receipts(tmp_path):
    root = tmp_path / "data"
    (root / "ds1").mkdir(parents=True)
    (root / "ds1" / "receipt.json").write_text("{}")
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "dataset_manifest.json").write_text("[]")
    h = P.dataset_hashes(root)
    assert set(h["receipts"]) == {"ds1"} and set(h["manifests"]) == {"dataset_manifest.json"}
    assert P.dataset_hashes(None)["manifests"] == {}


def test_evaluate_writes_provenance_to_eval_log_and_ledger(run_dir):
    ads = {"SYN": SyntheticAdapter()}
    evaluate_snapshots(run_dir, ["A_0", "A_1"], adapters=ads, solver_factory=Solver, workers=2, include_rep=False,
                       llm=MeterLLM())
    log = [json.loads(x) for x in (run_dir / "eval" / "eval_log.jsonl").read_text().splitlines()]
    assert len(log) == 1
    prov = log[0]["provenance"]
    assert prov["phase"] == "evaluate" and prov["eval_workers"] == 2 and prov["llm_concurrency"] == 16
    assert set(prov["programs"]) == {"A_0", "A_1"} and prov["splits_json_sha256"]
    assert prov["code"]["python"] and prov["config_sha256"] and prov["warnings"] == []
    assert log[0]["counts"]["done"] == 6 and log[0]["segment"]
    led = [r for r in P.read_ledger(run_dir / "usage.jsonl") if r["phase"] == "evaluate"]
    assert [r["status"] for r in led] == ["started", "done"] and led[-1]["kind"] == "snapshots"


def test_more_workers_than_llm_slots_warns(run_dir):
    from scienceclaw.config import load_config

    conc = load_config(run_dir / "config.yaml").llm.concurrency
    with pytest.warns(RuntimeWarning, match="llm.concurrency"):
        evaluate_snapshots(run_dir, ["A_0"], adapters={"SYN": SyntheticAdapter()}, solver_factory=Solver, workers=conc + 1,
                           include_rep=False)
    log = [json.loads(x) for x in (run_dir / "eval" / "eval_log.jsonl").read_text().splitlines()]
    assert log[-1]["provenance"]["eval_workers"] == conc + 1 and log[-1]["provenance"]["warnings"]
    rep = json.loads((build_report(run_dir, adapters={"SYN": SyntheticAdapter()}, n_boot=10).parent / "report.json")
                     .read_text())
    assert any("llm.concurrency" in w for w in rep["warnings"])


def test_report_contains_ledger_and_provenance(run_dir):
    ads = {"SYN": SyntheticAdapter()}
    evaluate_snapshots(run_dir, "all", adapters=ads, solver_factory=Solver, workers=1, include_rep=False)
    md = build_report(run_dir, adapters=ads, n_boot=10)
    rep = json.loads((md.parent / "report.json").read_text())
    phases = {r["phase"] for r in rep["usage_ledger"]["by_phase"]}
    assert {"evolve", "evaluate"} <= phases
    labels = [r["segment"] for r in rep["provenance"]]
    assert "evolve" in labels and any(lb == "snapshots" for lb in labels)
    assert rep["report_provenance"]["phase"] == "report" and rep["report_provenance"]["code"]["python"]
    text = md.read_text()
    assert "Provenance" in text and "billed" in text.lower() and "logical" in text.lower()
