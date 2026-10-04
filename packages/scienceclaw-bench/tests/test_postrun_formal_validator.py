from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "formal_postrun_validator",
    ROOT / "scripts" / "leonardo" / "migration" / "validate_formal_results.py",
)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def _manifest(tmp_path: Path, *, id_episodes: int = 1, ood_episodes: int = 0) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "batches": [
                    {
                        "tag": "SOTAX",
                        "discipline": "FoR30",
                        "mode": "formal_tool_on",
                        "required_tool_refs": ["predict_pretrained"],
                        "splits": {
                            "id": {"episodes": id_episodes},
                            "ood": {"episodes": ood_episodes, "status": "unavailable"}
                            if ood_episodes == 0
                            else {"episodes": ood_episodes},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _episode(
    root: Path,
    split: str,
    name: str,
    *,
    result: dict | None = None,
    ref_status: str = "ok",
    connect_tool: bool = True,
) -> Path:
    episode = root / f"toolon_SOTAX_{split}" / name
    episode.mkdir(parents=True)
    receipt = {
        "z": 1,
        "completed": True,
        "reproducible": True,
        "within_budget": True,
        "hard_ok": True,
    }
    if result:
        receipt.update(result)
    (episode / "result.json").write_text(json.dumps(receipt), encoding="utf-8")
    tool_output = str(episode / "values" / "pre.pkl")
    code_output = str(episode / "values" / "cpu.pkl")
    trajectory = {
        "action": {
            "payload": {
                "node": {"id": "pre", "kind": "tool", "ref": "predict_pretrained"}
            }
        },
        "feedback": {
            "records": {
                "pre": {
                    "status": ref_status,
                    "kind": "tool",
                    "output_refs": {"predictions": tool_output},
                },
                "code": {
                    "status": "ok",
                    "kind": "code",
                    "input_refs": {"x": tool_output if connect_tool else str(episode / "values" / "raw.pkl")},
                    "output_refs": {"y": code_output},
                },
                "submit": {
                    "status": "ok",
                    "kind": "submit",
                    "input_refs": {"y": code_output},
                    "output_refs": {"y": str(episode / "values" / "submitted.pkl")},
                },
            }
        },
    }
    (episode / "trajectory.jsonl").write_text(json.dumps(trajectory) + "\n", encoding="utf-8")
    return episode


def test_postrun_accepts_complete_episode_and_unavailable_split(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    _episode(tmp_path, "id", "FoR30-id-e00")
    assert validator.validate_results(tmp_path, "SOTAX", manifest) == []


def test_postrun_requires_hard_receipt_flags_and_successful_tool_record(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    episode = _episode(
        tmp_path,
        "id",
        "FoR30-id-e00",
        result={"z": 0, "within_budget": False},
        ref_status="error",
    )
    errors = validator.validate_results(tmp_path, "SOTAX", manifest)
    assert any("z must be integer 1" in error for error in errors)
    assert any("within_budget must be true" in error for error in errors)
    assert any("successful feedback.records" in error for error in errors)
    assert episode.exists()


def test_postrun_requires_tool_output_to_reach_submit(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    _episode(tmp_path, "id", "FoR30-id-e00", connect_tool=False)
    errors = validator.validate_results(tmp_path, "SOTAX", manifest)
    assert any("not connected to final submit.y" in error for error in errors)


def test_postrun_split_filter_allows_concurrent_sibling_batch(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, ood_episodes=1)
    _episode(tmp_path, "id", "FoR30-id-e00")
    assert validator.validate_results(tmp_path, "SOTAX", manifest, split_filter="id") == []
    errors = validator.validate_results(tmp_path, "SOTAX", manifest)
    assert any("SOTAX/ood: manifest requires 1 episode(s), found 0" in error for error in errors)


def test_postrun_fails_on_manifest_episode_count_and_missing_receipts(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, id_episodes=3)
    _episode(tmp_path, "id", "FoR30-id-e00")
    incomplete = tmp_path / "toolon_SOTAX_id" / "FoR30-id-e01"
    incomplete.mkdir()
    errors = validator.validate_results(tmp_path, "SOTAX", manifest)
    assert any("requires 3 episode(s), found 2" in error for error in errors)
    assert any("missing result.json" in error for error in errors)
    assert any("missing trajectory.jsonl" in error for error in errors)


def test_cli_returns_nonzero_when_formal_evidence_fails(tmp_path: Path, monkeypatch, capsys) -> None:
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(
        validator.sys,
        "argv",
        ["validate_formal_results.py", "SOTAX", str(tmp_path), "--manifest", str(manifest)],
    )
    assert validator.main() == 2
    assert "ERROR" in capsys.readouterr().err


def test_manifest_lookup_honors_remote_checkout_environment(monkeypatch) -> None:
    monkeypatch.setenv("SCIENCECLAW_REPO", str(ROOT))
    assert validator._manifest_for(None, None) == ROOT / "configs" / "formal_toolon_manifest.json"
