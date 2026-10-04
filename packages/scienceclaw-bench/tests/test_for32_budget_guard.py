"""Execution guard for a future fair FoR32 tool-ON rerun.

The profile is deliberately separate from the normal Leonardo profile.  These
checks protect the operational budget only; they do not alter FoR32 scoring,
the atlas reference, acceptance margin, or visible/withheld data boundaries.
"""
from pathlib import Path
import os
import subprocess

from scienceclaw.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_f32_budgeted_profile_is_narrow_and_bounded():
    cfg = load_config(ROOT / "configs" / "toolon_f32_budgeted.yaml")
    assert cfg.bench.disciplines == ["FoR32"]
    assert cfg.solver.max_steps == 12
    assert cfg.bench.items_per_episode == 16
    assert cfg.bench.n_id == 4 and cfg.bench.n_ood == 4
    assert cfg.llm.endpoints == ["http://127.0.0.1:21000/v1", "http://127.0.0.1:21001/v1"]


def test_f32_budgeted_profile_does_not_change_acceptance_configuration():
    cfg = load_config(ROOT / "configs" / "toolon_f32_budgeted.yaml")
    # The acceptance/reference are task-owned; this profile only sets execution.
    assert cfg.evolution.variant == "full"
    assert cfg.evolution.qval == "macrosr_then_score"
    assert cfg.evolution.qval_eps == 0.02


def test_run_batch_fails_closed_for_unreadable_profile(tmp_path):
    script = ROOT / "scripts" / "leonardo" / "run_batch_leo.sh"
    env = os.environ.copy()
    env["SCIENCECLAW_CONFIG_PATH"] = str(tmp_path / "does-not-exist.yaml")
    proc = subprocess.run(["bash", str(script), "H-F32", "FoR32", "id", "1", "1"],
                          env=env, capture_output=True, text=True)
    assert proc.returncode == 2
    assert "config file is not readable" in proc.stderr
