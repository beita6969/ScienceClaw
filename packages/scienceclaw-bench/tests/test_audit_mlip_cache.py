from __future__ import annotations

import json
import sys

import numpy as np

from scienceclaw.bench.tasks.for51_matbench import parse_structure
from scripts.f51.audit_mlip_cache import audit_cache, main
from scilib.matphonon_mlip import NAMES, _cache_key


def _structure(x: float) -> dict:
    return parse_structure({"lattice": {"matrix": [[x, 0, 0], [0, x, 0], [0, 0, x]]},
                            "sites": [{"species": [{"element": "Si", "occu": 1}], "abc": [0, 0, 0]}]})


def test_audit_cache_counts_exact_keys_and_rejects_bad_rows(tmp_path):
    structures = {"a": _structure(4.0), "b": _structure(5.0), "c": _structure(6.0)}
    ids = list(structures)
    good = _cache_key(structures["a"], 8, 7.0, 0.01, "sevennet")
    bad = _cache_key(structures["b"], 8, 7.0, 0.01, "sevennet")
    (tmp_path / f"{good}.json").write_text(json.dumps([1.0] * len(NAMES)))
    (tmp_path / f"{bad}.json").write_text(json.dumps([float("nan")] * len(NAMES)))
    out = audit_cache(structures, ids, tmp_path, "sevennet")
    assert out["n_structures"] == 3
    assert out["valid"] == 1 and out["missing"] == 1 and out["malformed"] == 1
    assert out["complete"] is False


def test_audit_cache_accepts_complete_finite_rows(tmp_path):
    structures = {"a": _structure(4.0)}
    key = _cache_key(structures["a"], 8, 7.0, 0.01, "chgnet")
    (tmp_path / f"{key}.json").write_text(json.dumps(np.ones(len(NAMES)).tolist()))
    out = audit_cache(structures, list(structures), tmp_path, "chgnet")
    assert out["complete"] and out["valid"] == 1


def test_cli_fails_closed_when_cache_env_is_unset(monkeypatch):
    monkeypatch.delenv("SCIENCECLAW_MLIP_CACHE", raising=False)
    monkeypatch.setattr(sys, "argv", ["audit_mlip_cache"])
    assert main() == 2
