"""scilib.matphonon_mlip: availability, per-structure cache key, remote routing and the local cache (the potential itself is faked)."""
from __future__ import annotations

import json

import numpy as np
import pytest

import scilib
from scilib import _remote
from scilib import matphonon_mlip as mm
from test_scilib_remote import broker  # noqa: F401  (fixture)

ST = {"lattice": [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]], "species": ["Na", "Cl"], "frac_coords": [[0, 0, 0], [0.5, 0.5, 0.5]]}


def test_unavailable_without_stack_or_spool(monkeypatch):
    monkeypatch.delenv(_remote.SPOOL_ENV, raising=False)
    monkeypatch.setattr(mm, "_local_ok", lambda *a: False)
    assert not mm.available()
    assert scilib.describe_extra("matphonon_mlip") == ""
    with pytest.raises(RuntimeError, match="not available"):
        mm.phonon_features([ST])


def test_describe_extra_shown_only_when_available(monkeypatch):
    monkeypatch.delenv("SCIENCECLAW_NO_SCILIB", raising=False)
    monkeypatch.setattr(mm, "available", lambda: True)
    assert scilib.describe_extra("matphonon_mlip").startswith("\n\nLibrary `scilib.matphonon_mlip`, importable in code nodes:")
    monkeypatch.setenv("SCIENCECLAW_NO_SCILIB", "1")
    assert scilib.describe_extra("matphonon_mlip") == ""


def test_switched_off(monkeypatch):
    monkeypatch.setenv("SCIENCECLAW_NO_PRETRAINED", "1")
    assert not mm.available()


def test_cache_key_depends_on_structure_and_parameters_only():
    a = mm._cache_key(ST, 8, 7.0, 0.01)
    assert a == mm._cache_key(json.loads(json.dumps(ST)), 8, 7.0, 0.01)
    assert a != mm._cache_key(ST, 6, 7.0, 0.01) and a != mm._cache_key(ST, 8, 7.0, 0.02)
    moved = dict(ST, frac_coords=[[0, 0, 0], [0.5, 0.5, 0.4]])
    assert a != mm._cache_key(moved, 8, 7.0, 0.01)
    assert a == mm._cache_key(dict(ST, frac_coords=[[0, 0, 1e-9], [0.5, 0.5, 0.5]]), 8, 7.0, 0.01)


def test_cache_key_depends_on_model_and_chgnet_keys_are_unchanged():
    assert mm._cache_key(ST, 8, 7.0, 0.01) == mm._cache_key(ST, 8, 7.0, 0.01, "chgnet")
    assert mm._cache_key(ST, 8, 7.0, 0.01, "sevennet") != mm._cache_key(ST, 8, 7.0, 0.01, "chgnet")
    assert mm._cache_key(ST, 8, 7.0, 0.01, "sevennet_mf0_pbe") != mm._cache_key(ST, 8, 7.0, 0.01, "sevennet_mf0_r2scan")


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="model must be"):
        mm.phonon_features([ST], model="nope")


def test_bad_structure_is_rejected():
    with pytest.raises(ValueError):
        mm._structure({"lattice": [[1, 0, 0]], "species": ["H"], "frac_coords": [[0, 0, 0]]})


def test_remote_routing(broker, monkeypatch):  # noqa: F811
    box, seen, _ = broker
    monkeypatch.setattr(mm, "_local_ok", lambda *a: False)
    assert mm.available()
    box["handler"] = lambda fn, a: {"X": np.full((len(a["structures"]), len(mm.NAMES)), 7.0), "names": list(mm.NAMES)}
    out = mm.phonon_features([ST, ST], mesh=6)
    assert out["X"].shape == (2, len(mm.NAMES)) and out["names"] == list(mm.NAMES)
    assert seen[0]["module"] == "matphonon_mlip" and seen[0]["fn"] == "phonon_features"
    assert seen[0]["args"]["model"] == "sevennet"
    mm.phonon_features([ST], mesh=6, model="chgnet")
    assert seen[1]["args"]["model"] == "chgnet"
    assert mm.phonon_features([])["X"].shape == (0, len(mm.NAMES))


def test_local_path_caches_per_structure(tmp_path, monkeypatch):
    calls = []

    def fake(st, mesh, min_len, disp, model):
        calls.append(st["species"])
        if st["species"] == ["Bad"]:
            raise RuntimeError("no")
        return [float(len(st["species"]))] * len(mm.NAMES)
    monkeypatch.setattr(mm, "_local_ok", lambda *a: True)
    monkeypatch.setattr(mm, "_one", fake)
    monkeypatch.setenv("SCIENCECLAW_MLIP_CACHE", str(tmp_path))
    bad = dict(ST, species=["Bad"], frac_coords=[[0, 0, 0]])
    X = mm.phonon_features([ST, bad])["X"]
    assert X.shape == (2, len(mm.NAMES)) and np.all(X[0] == 2.0) and np.all(np.isnan(X[1]))
    n = len(calls)
    X2 = mm.phonon_features([ST])["X"]
    assert np.all(X2 == 2.0) and len(calls) == n                              # served from the cache
    assert len(list(tmp_path.glob("*.json"))) == 1                            # failures are not stored


def test_local_path_recomputes_malformed_cached_rows(tmp_path, monkeypatch):
    calls = []

    def fake(st, mesh, min_len, disp, model):
        calls.append(st["species"])
        return [3.0] * len(mm.NAMES)

    monkeypatch.setattr(mm, "_local_ok", lambda *a: True)
    monkeypatch.setattr(mm, "_one", fake)
    monkeypatch.setenv("SCIENCECLAW_MLIP_CACHE", str(tmp_path))
    key = mm._cache_key(ST, 8, 7.0, 0.01, "sevennet")
    # A stale writer may leave valid JSON with the wrong width, or a non-finite row.
    (tmp_path / f"{key}.json").write_text(json.dumps([float("nan")]))
    out = mm.phonon_features([ST], model="sevennet")["X"]
    assert calls == [ST["species"]] and out.shape == (1, len(mm.NAMES))
    assert np.all(out == 3.0)
    assert np.asarray(json.loads((tmp_path / f"{key}.json").read_text())).shape == (len(mm.NAMES),)


def test_complete_frozen_cache_works_without_optional_packages(tmp_path, monkeypatch):
    monkeypatch.setenv("SCIENCECLAW_MLIP_CACHE", str(tmp_path))
    monkeypatch.setattr(mm, "_local_ok", lambda *a: False)
    monkeypatch.delenv(_remote.SPOOL_ENV, raising=False)
    key = mm._cache_key(ST, 8, 7.0, 0.01, "sevennet")
    (tmp_path / f"{key}.json").write_text(json.dumps([1.0] * len(mm.NAMES)))
    out = mm.phonon_features([ST], model="sevennet")
    assert out["X"].shape == (1, len(mm.NAMES))
    assert np.all(out["X"] == 1.0)


def test_cache_row_write_is_atomic_and_leaves_no_temp_files(tmp_path):
    path = tmp_path / "feature.json"
    mm._write_cache_row(path, [1.0] * len(mm.NAMES))
    assert json.loads(path.read_text()) == [1.0] * len(mm.NAMES)
    assert not list(tmp_path.glob("*.tmp"))


def test_niggli_fallback_grid_is_uniform_and_deterministic():
    mesh = 3
    q = np.stack(np.meshgrid(*[np.arange(mesh, dtype=float) / mesh] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    assert q.shape == (mesh ** 3, 3)
    assert np.all(q >= 0.0) and np.all(q < 1.0)
    q2 = np.stack(np.meshgrid(*[np.arange(mesh, dtype=float) / mesh] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    assert np.array_equal(q, q2)
