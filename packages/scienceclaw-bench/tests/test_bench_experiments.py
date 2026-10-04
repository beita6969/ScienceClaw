"""Experiment drivers (run_stream / evaluate / report / cli) with local stub Evolver and Solver.

The real Evolver/Solver are other modules; these tests inject stubs so they run offline and independently.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scienceclaw.bench.splits import SplitPlan
from synthetic_adapter import SyntheticAdapter, synthetic_truth
from scienceclaw.cli import build_parser, parse_overrides
from scienceclaw.config import BenchConfig, RunConfig, load_config
from scienceclaw.core.program import AgentProgram, Bundle
from scienceclaw.core.skills import Skill
from scienceclaw.experiments.evaluate import (evaluate_family_transfer, evaluate_snapshots, family_programs,
                                              read_results)
from scienceclaw.experiments.report import build_report, compare_runs, promotion_rows
from scienceclaw.experiments.run_stream import run_stream, snapshot_dirs


# ------------------------------------------------------------------------------------------------- stubs
class StubEvolver:
    """Adds one Skill per round (provenance = that round's SYN source episode) and logs candidates."""

    def __init__(self, cfg, llm, plan, run_dir):
        self.cfg, self.llm, self.plan, self.run_dir = cfg, llm, plan, Path(run_dir)

    def run(self, program0):
        snaps = [program0]
        prog = program0
        with (self.run_dir / "candidates.jsonl").open("w") as f:
            for r, ep in self.plan.source_stream():
                sk = Skill(id=f"s{r}", version=1, title=f"skill {r}", body="b", created=f"r{r}:e{ep.id}")
                prog, omega = prog.apply(Bundle(skills=[sk]), new_version=f"A{r}")
                f.write(json.dumps({"round": r, "episode": ep.id, "accepted": True}) + "\n")
                f.write(json.dumps({"round": r, "episode": ep.id, "accepted": False}) + "\n")
                snaps.append(prog)
        return snaps


class StubSolver:
    """Predicts with the true function iff the program has a Skill, else the training mean."""

    calls: list = []

    def solve(self, episode, program, mode, run_dir):
        assert mode == "eval"
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        StubSolver.calls.append((program.version, episode.split, episode.id, str(run_dir)))
        X = episode.tool("load_eval_inputs").fn({}, {})["X_eval"]
        yt = episode.tool("load_train").fn({}, {})["y_train"]
        y = synthetic_truth(X) if program.skills else np.full(len(X), yt.mean())
        ev = episode.evaluate(y, None)
        ev.reproducible = True

        def save(path):
            (Path(path) / "eval.json").write_text(json.dumps({"z": ev.z}))

        return SimpleNamespace(eval=ev, usage={"total_tokens": 10, "llm_calls": 1, "wall_s": 0.01},
                               program_version=program.version, save=save)


@pytest.fixture()
def stub_evolver(monkeypatch):
    mod = types.ModuleType("scienceclaw.evolution.evolver")
    mod.Evolver = StubEvolver
    monkeypatch.setitem(sys.modules, "scienceclaw.evolution.evolver", mod)
    return mod


def _cfg(tmp_path, rounds=2):
    cfg = RunConfig(name="t", runs_root=str(tmp_path / "runs"))
    cfg.bench = BenchConfig(disciplines=["SYN"], items_per_episode=6, rounds=rounds, n_val=1, n_id=2, n_ood=1, seed=3)
    return cfg


@pytest.fixture()
def run_dir(tmp_path, stub_evolver):
    return run_stream(_cfg(tmp_path), adapters={"SYN": SyntheticAdapter()}, llm=SimpleNamespace(usage=lambda: {"total": {}}))


# ------------------------------------------------------------------------------------------------- tests
def test_run_stream_writes_receipts(run_dir):
    assert run_dir.name.startswith("t-")
    cfg = load_config(run_dir / "config.yaml")
    assert cfg.bench.disciplines == ["SYN"] and cfg.bench.rounds == 2
    m = SplitPlan.load_manifest(run_dir / "splits.json")
    assert m["disciplines"] == ["SYN"] and len(m["splits"]["src"]["SYN"]) == 2
    assert sorted(snapshot_dirs(run_dir)) == [0, 1, 2]
    assert AgentProgram.load(snapshot_dirs(run_dir)[2]).version == "A2"
    run = json.loads((run_dir / "run.json").read_text())
    assert run["status"] == "done" and run["split_sha256"] == m["sha256"] and "A_2" in run["snapshots"]
    assert "llm" in json.loads((run_dir / "usage.json").read_text())


def test_run_stream_marks_failure(tmp_path, monkeypatch):
    class Boom(StubEvolver):
        def run(self, program0):
            raise RuntimeError("boom")

    mod = types.ModuleType("scienceclaw.evolution.evolver")
    mod.Evolver = Boom
    monkeypatch.setitem(sys.modules, "scienceclaw.evolution.evolver", mod)
    with pytest.raises(RuntimeError):
        run_stream(_cfg(tmp_path), adapters={"SYN": SyntheticAdapter()}, llm=SimpleNamespace())
    rd = next((tmp_path / "runs").iterdir())
    assert json.loads((rd / "run.json").read_text())["status"] == "failed"


def test_evaluate_snapshots_rep_and_resume(run_dir):
    StubSolver.calls = []
    ads = {"SYN": SyntheticAdapter()}
    path = evaluate_snapshots(run_dir, "all", adapters=ads, solver_factory=StubSolver, workers=2)
    rows = read_results(path)
    # 3 snapshots x (2 id + 1 ood) + rep: A_1 -> 1, A_2 -> 2
    assert len(rows) == 3 * 3 + 1 + 2
    keys = {(r["snapshot"], r["split"]) for r in rows}
    assert ("A_0", "rep") not in keys and ("A_2", "rep") in keys
    rep = [r for r in rows if r["split"] == "rep"]
    assert sorted((r["snapshot"], r["round"]) for r in rep) == [("A_1", 1), ("A_2", 1), ("A_2", 2)]
    a0 = [r for r in rows if r["snapshot"] == "A_0" and r["split"] == "id"]
    a2 = [r for r in rows if r["snapshot"] == "A_2" and r["split"] == "id"]
    assert all(r["z"] == 0 for r in a0) and all(r["z"] == 1 for r in a2)
    r0 = rows[0]
    for k in ("z", "primary", "norm_score", "pooled_payload", "usage", "direction", "discipline", "family"):
        assert k in r0
    assert (run_dir / "eval" / r0["pooled_payload"]).exists()
    assert (run_dir / "eval" / r0["receipt_dir"] / "eval.json").exists()
    # fresh directory per (snapshot, split, episode)
    dirs = [c[3] for c in StubSolver.calls]
    assert len(set(dirs)) == len(dirs)
    # resumable: nothing is re-solved
    n_calls = len(StubSolver.calls)
    evaluate_snapshots(run_dir, "all", adapters=ads, solver_factory=StubSolver, workers=2)
    assert len(StubSolver.calls) == n_calls and len(read_results(path)) == len(rows)


def test_evaluate_errors_are_retried(run_dir):
    class Flaky(StubSolver):
        def solve(self, episode, program, mode, run_dir):
            if episode.split == "ood":
                raise RuntimeError("transient")
            return super().solve(episode, program, mode, run_dir)

    ads = {"SYN": SyntheticAdapter()}
    path = evaluate_snapshots(run_dir, ["A_0"], splits=("id", "ood"), adapters=ads, solver_factory=Flaky,
                              include_rep=False, workers=1)
    assert {r["split"] for r in read_results(path)} == {"id"}
    errs = (run_dir / "eval" / "errors.jsonl").read_text().splitlines()
    assert len(errs) == 1 and "transient" in errs[0]
    evaluate_snapshots(run_dir, ["A_0"], splits=("id", "ood"), adapters=ads, solver_factory=StubSolver,
                       include_rep=False, workers=1)
    assert {r["split"] for r in read_results(path)} == {"id", "ood"}
    with pytest.raises(FileNotFoundError):
        evaluate_snapshots(run_dir, ["A_9"], adapters=ads, solver_factory=StubSolver)


def test_manifest_mismatch_is_detected(run_dir):
    class OtherSynthetic(SyntheticAdapter):
        def build_episodes(self, split, n, seed, items_per_episode=16):
            return super().build_episodes(split, n, seed + 1, items_per_episode)

    with pytest.raises(ValueError, match="manifest"):
        evaluate_snapshots(run_dir, "all", adapters={"SYN": OtherSynthetic()}, solver_factory=StubSolver)


def test_report_tables(run_dir):
    ads = {"SYN": SyntheticAdapter()}
    evaluate_snapshots(run_dir, "all", adapters=ads, solver_factory=StubSolver, workers=1)
    evaluate_family_transfer(run_dir, split="ood", adapters=ads, solver_factory=StubSolver, workers=1)
    md = build_report(run_dir, adapters=ads, n_boot=100)
    rep = json.loads((md.parent / "report.json").read_text())
    summ = {(r["snapshot"], r["split"]): r for r in rep["summary"]}
    assert summ[("A_0", "id")]["macro_sr"] == 0.0 and summ[("A_2", "id")]["macro_sr"] == 1.0
    assert summ[("A_0", "ood")]["pi"] == pytest.approx(100.0)
    assert summ[("A_2", "ood")]["pi"] > 150.0
    assert summ[("A_0", "id")]["n_expected"] == 2 and summ[("A_2", "rep")]["n_expected"] == 2
    assert rep["pi_reference"] == "A_0" and rep["normalization"] == "ratio"
    pd_rows = [r for r in rep["per_discipline"] if r["snapshot"] == "A_2" and r["split"] == "id"]
    assert pd_rows[0]["score_kind"] == "pooled" and pd_rows[0]["pooled_score"] < 0.3
    fam = [r for r in rep["per_family"] if r["group_kind"] == "sphere"]
    assert {r["group"] for r in fam} == {"natural"}
    prom = {r["round"]: r for r in rep["promotions"]}
    assert prom[1]["candidates"] == 2 and prom[1]["promoted"] == 1 and prom["total"]["rate"] == 0.5
    ret = rep["rep_retention"]["z"]
    assert ret["matrix"] == [[1.0, None], [1.0, 1.0]] or ret["matrix"][1] == [1.0, 1.0]
    assert ret["BWT"] == pytest.approx(0.0)
    ft = rep["family_transfer"]
    assert ft["families"] == ["Engineering & computing"] and ft["matrix"][0][0] > 0
    assert ft["FWT_allcell"] == pytest.approx(ft["matrix"][0][0])
    for f in ("summary.csv", "per_discipline.csv", "per_family.csv", "costs.csv", "promotions.csv",
              "rep_matrix_z.csv", "family_transfer.csv"):
        assert (md.parent / f).exists(), f
    text = md.read_text()
    assert "computed from this run's receipts" in text and "PI reference: **A_0**" in text
    # method_final reference: the last snapshot gets 100
    md2 = build_report(run_dir, adapters=ads, pi_reference="method_final", n_boot=50)
    rep2 = json.loads((md2.parent / "report.json").read_text())
    assert {(r["snapshot"], r["split"]): r for r in rep2["summary"]}[("A_2", "ood")]["pi"] == pytest.approx(100.0)
    # compare runs (same run twice -> identical ranks)
    cmp = compare_runs({"a": run_dir, "b": run_dir}, split="id", adapters=ads)
    assert cmp["avg_ranks"] == {"a": 1.5, "b": 1.5} and cmp["sign_tests"]["b"]["ties"] == 1


def test_family_programs_attribution():
    base = AgentProgram()
    final, _ = base.apply(Bundle(skills=[Skill("a", 1, "t", "b", created="r1:eE1"),
                                         Skill("b", 1, "t", "b", provenance={"episode": "E2"}),
                                         Skill("c", 1, "t", "b")]))
    progs, info = family_programs(base, final, {"E1": "F1", "E2": "F2"}, ["F1", "F2"])
    assert set(progs["F1"].skills) == {"a"} and set(progs["F2"].skills) == {"b"}
    assert info["unattributed"] == ["skill:c"]


def test_promotion_rows_formats():
    rows = promotion_rows([{"round": 1, "decision": "accept"}, {"round": 1, "status": "rejected"},
                           {"round": 2, "promoted": True}])
    assert rows[0] == {"round": 1, "candidates": 2, "promoted": 1, "rate": 0.5}
    assert rows[-1]["promoted"] == 2


def test_cli_parsing(tmp_path):
    assert parse_overrides(["bench.rounds=3", "llm.policy.model=m", "solver.show_dev_score=false"]) == {
        "bench.rounds": 3, "llm.policy.model": "m", "solver.show_dev_score": False}
    a = build_parser().parse_args(["evaluate", "--run", "x", "--snapshots", "A_0,A_2", "--no-rep"])
    assert a.snapshots == "A_0,A_2" and a.no_rep and a.splits == "id,ood"


def test_cli_list_tasks(capsys):
    from scienceclaw.cli import main

    assert main(["list-tasks"]) == 0
    out = capsys.readouterr().out
    assert "FoR30" in out and "FoR52" in out and "registry disciplines available" in out


def test_configs_load():
    root = Path(__file__).resolve().parents[1] / "configs"
    d = load_config(root / "default.yaml")
    assert d.to_dict() == RunConfig(name="default").to_dict()
    dev = load_config(root / "dev_api.yaml")
    assert (len(dev.bench.disciplines), dev.bench.rounds, dev.bench.n_val, dev.bench.n_id, dev.bench.n_ood) == (3, 2, 1, 1, 1)
    assert dev.solver.max_steps == 8 and dev.llm.policy.model == "lab-gpt-5.4-mini" and dev.llm.policy.json_mode
    full = load_config(root / "full_api.yaml")
    assert (full.bench.disciplines, full.bench.rounds, full.bench.n_val, full.bench.n_id, full.bench.n_ood) == ([], 7, 2, 4, 4)
