from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "formal_manifest_gate", ROOT / "scripts" / "leonardo" / "migration" / "check_formal_manifest.py"
)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)

MANIFEST = ROOT / "configs" / "formal_toolon_manifest.json"


def test_formal_request_matches_manifest_and_registered_tool():
    assert gate.validate_request(ROOT, MANIFEST, "FoR30", "SOTA30", "id,ood", 4) == []
    assert gate.validate_request(ROOT, MANIFEST, "FoR36", "SOTA36", "id", 4) == []
    assert gate.validate_request(ROOT, MANIFEST, "FoR50", "SOTA50", "id,ood", 4) == []
    assert gate.validate_request(ROOT, MANIFEST, "FoR52", "SOTA52", "id,ood", 4) == []
    assert gate.validate_request(ROOT, MANIFEST, "FoR47", "SOTA47B", "id,ood", 4, 4) == []
    assert gate.validate_request(ROOT, MANIFEST, "FoR51", "SOTA51B", "id", 2, 4) == []
    assert gate.validate_request(ROOT, MANIFEST, "FoR50", "SOTA50B", "id,ood", 4, 4) == []
    assert gate.validate_request(ROOT, MANIFEST, "FoR52", "SOTA52B", "id,ood", 4, 4) == []


def test_unavailable_blocked_and_wrong_count_fail_closed():
    unavailable = gate.validate_request(ROOT, MANIFEST, "FoR36", "SOTA36", "ood", 4)
    assert any("unavailable" in error for error in unavailable)
    blocked = gate.validate_request(ROOT, MANIFEST, "FoR32", "SOTA32", "id", 4)
    assert any("No fair external pretrained weight" in error for error in blocked)
    wrong = gate.validate_request(ROOT, MANIFEST, "FoR30", "SOTA30", "id", 3)
    assert any("requires n=4" in error for error in wrong)
    wrong_skip = gate.validate_request(ROOT, MANIFEST, "FoR47", "SOTA47B", "id", 4, 0)
    assert any("requires skip=4" in error for error in wrong_skip)
    exhausted = gate.validate_request(ROOT, MANIFEST, "FoR52", "SOTA52B", "id", 4, 16)
    assert any("beyond declared pool capacity 16" in error for error in exhausted)


def test_debug_tag_does_not_need_formal_manifest(tmp_path):
    missing = tmp_path / "missing.json"
    assert gate.validate_request(ROOT, missing, "FoR30", "H6", "id", 4) == []
    assert gate.validate_request(ROOT, missing, "FoR30", "SOTA30", "id", 4)


def test_sota51_launch_is_wired_to_fail_closed_cache_audit():
    script = (ROOT / "scripts" / "leonardo" / "run_batch_leo.sh").read_text()
    assert "audit_mlip_cache.py" in script
    assert "--model sevennet --require-complete" in script
    assert '"$TAG" == SOTA51' in script
    single = (ROOT / "scripts" / "leonardo" / "run_single_gpu_toolon.sh").read_text()
    assert '"$TAG" == SOTA51*' in single and '"$MANIFEST_CHECK" "$DISC" "$TAG" "$SPLIT" "$N" --skip "$SKIP"' in single


def test_formal_batch_runs_split_scoped_postrun_validator():
    script = (ROOT / "scripts" / "leonardo" / "run_batch_leo.sh").read_text()
    assert "formal post-run evidence failed" in script
    assert '"$VALIDATOR" "$TAG" "$R" --split "$SPLIT"' in script
