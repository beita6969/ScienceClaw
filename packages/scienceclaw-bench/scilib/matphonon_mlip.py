"""Harmonic phonon spectra of crystals from a pretrained universal interatomic potential.

Frozen routes include SevenNet-l3i5, SevenNet-MF-0 PBE/r2SCAN, and CHGNet
0.3.0, all used with phonopy.

Structures are dicts as returned by the task's loaders: ``lattice`` (3x3, lattice vectors as rows, Å), ``species`` (element
symbol per site) and ``frac_coords`` (fractional coordinates per site). The structures are used as given (no relaxation).

available() -> bool
    True when the packages of one of the models (and phonopy) can be imported, or when a remote GPU worker is attached (then
    ``phonon_features`` runs on that worker and returns the same values; results are cached per structure and by arguments).
phonon_features(structures, mesh=8, min_len=7.0, disp=0.01, model="sevennet") -> dict
    ``{"X": float array (n, len(names)), "names": [str]}``. ``model`` is ``"sevennet"`` (SevenNet-l3i5) or ``"chgnet"``.
    Per structure: a supercell whose edges are at least ``min_len`` Å
    (at most 220 atoms), the forces of the potential on the supercells with one displacement of ``disp`` Å per
    symmetry-inequivalent atom and direction (phonopy), the force constants, and the frequencies on a ``mesh``^3
    Γ-centred q-grid with the phonon density of states (Gaussian smearing). The columns, all frequencies in cm^-1:
    highest / 99th / 95th / 90th percentile / mean / median / 25th percentile of the mesh frequencies above 1 cm^-1
    (``f_max`` ... ``f_p25``); statistics of the highest branch over the q-grid (``top_mean``, ``top_median``,
    ``top_p10``, ``top_min``); the share of frequencies below -1 cm^-1 and the lowest frequency (``imag_frac``,
    ``f_min``); the frequency of the highest peak of the density of states and of its last peak (height >= 3 % of the
    maximum) after smearing with sigma = 0.1 / 0.25 / 0.5 THz (``dos_top_s*``, ``dos_last_s*``); the highest
    frequency at Γ (``gamma_max``); ``n_sites``. Rows of structures for which the calculation failed are NaN.
    Universal potentials are known to underestimate phonon frequencies systematically (softening of the potential-energy
    surface, Deng et al. 2024), so the columns are inputs for a regression on labelled frequencies rather than the
    frequencies themselves. Cost on one GPU: about 0.25 s per structure with ``"chgnet"``, about 0.9 s per structure with
    ``"sevennet"``; a request with several hundred new structures takes several minutes.

Models. ``"sevennet"``: SevenNet-l3i5 (Park et al., J. Chem. Theory Comput. 2024; equivariant message passing with
l_max = 3 and 5 interaction layers; 5 Å cutoff), trained on the energies, forces and stresses of the Materials Project
trajectories MPtrj. ``"chgnet"``: CHGNet (Deng et al., Nature Machine Intelligence 2023), trained on the energies, forces,
stresses and magnetic moments of MPtrj (MIT licence). Materials Project structures - the source of many crystals in phonon
benchmarks - were part of the training data of both (energies/forces/stresses, not phonon frequencies).
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from . import _remote
from ._pretrained import have_module, switched_off

__all__ = ["available", "phonon_features", "NAMES"]

_THZ = 33.35641
_VERSION = "1"
_SIGMAS = (0.1, 0.25, 0.5)
NAMES = (["f_max", "f_p99", "f_p95", "f_p90", "f_mean", "f_median", "f_p25", "top_mean", "top_median", "top_p10",
          "top_min", "imag_frac", "f_min"]
         + [f"dos_top_s{s:g}" for s in _SIGMAS] + [f"dos_last_s{s:g}" for s in _SIGMAS] + ["gamma_max", "n_sites"])
MODELS = ("sevennet", "sevennet_mf0_pbe", "sevennet_mf0_r2scan", "chgnet")
_MODEL: dict = {}
_NEEDS = {
    "chgnet": ("chgnet", "phonopy", "pymatgen", "torch", "scipy"),
    "sevennet": ("sevenn", "ase", "phonopy", "torch", "scipy"),
    "sevennet_mf0_pbe": ("sevenn", "ase", "phonopy", "torch", "scipy"),
    "sevennet_mf0_r2scan": ("sevenn", "ase", "phonopy", "torch", "scipy"),
}


def _local_ok(model: str = "chgnet") -> bool:
    return not switched_off() and all(have_module(m) for m in _NEEDS[model])


def available() -> bool:
    return any(_local_ok(m) for m in MODELS) or _remote.enabled()


def _structure(st: dict) -> tuple[np.ndarray, list[str], np.ndarray]:
    L = np.asarray(st["lattice"], dtype=float)
    F = np.asarray(st["frac_coords"], dtype=float)
    sp = [str(s) for s in st["species"]]
    if L.shape != (3, 3) or F.shape != (len(sp), 3) or not len(sp):
        raise ValueError("structure needs lattice (3x3), species and frac_coords (n x 3)")
    return L, sp, F


def _cache_key(st: dict, mesh: int, min_len: float, disp: float, model: str = "chgnet") -> str:
    L, sp, F = _structure(st)
    head = [_VERSION] if model == "chgnet" else [_VERSION, model]
    blob = json.dumps(head + [mesh, round(float(min_len), 6), round(float(disp), 6), np.round(L, 6).tolist(), sp,
                              np.round(F, 6).tolist()], separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def _cache_dir() -> Path | None:
    p = os.environ.get("SCIENCECLAW_MLIP_CACHE")
    return Path(p) if p and Path(p).is_dir() else None


def _write_cache_row(path: Path, row: list[float]) -> None:
    """Write one complete feature row without exposing a partial JSON file.

    Prewarming can be interrupted by a wall-time limit or a node failure.  A
    same-directory temporary file followed by ``os.replace`` makes every cache
    key either absent or valid, and also keeps concurrent resumptions from
    observing a half-written row.
    """
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(row, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def _read_cache_row(path: Path) -> np.ndarray | None:
    """Read one immutable feature row, rejecting stale or partial cache files.

    Cache files are content-addressed, but a process can still be interrupted while an
    older writer is replacing a row.  The complete-cache fast path already validates
    rows; the per-structure local path must apply the same check so a partial cache
    cannot be turned into a malformed feature matrix or silently reach the regressor.
    """
    try:
        row = np.asarray(json.loads(path.read_text()), dtype=float)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if row.shape != (len(NAMES),) or not np.isfinite(row).all():
        return None
    return row


def _model(model: str):
    if model not in _MODEL:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if model == "chgnet":
            from chgnet.model import CHGNet
            _MODEL[model] = CHGNet.load(verbose=False).to(dev)
        else:
            from sevenn.calculator import SevenNetCalculator
            if model == "sevennet":
                _MODEL[model] = SevenNetCalculator("7net-l3i5", device=dev)
            else:
                modal = "PBE" if model.endswith("_pbe") else "R2SCAN"
                _MODEL[model] = SevenNetCalculator("7net-mf-0", modal=modal, device=dev)
    return _MODEL[model]


def _forces(model: str, supercells) -> np.ndarray:
    m = _model(model)
    if model == "chgnet":
        from pymatgen.core import Lattice, Structure
        structs = [Structure(Lattice(s.cell), s.symbols, s.scaled_positions) for s in supercells]
        pred = m.predict_structure(structs, task="ef", batch_size=16) if len(structs) > 1 else [m.predict_structure(structs[0], task="ef")]
        return np.array([p["f"] for p in pred])
    from ase import Atoms
    out = []
    for s in supercells:
        at = Atoms(symbols=s.symbols, cell=s.cell, scaled_positions=s.scaled_positions, pbc=True)
        at.calc = m
        out.append(at.get_forces())
    return np.array(out)


def _one(st: dict, mesh: int, min_len: float, disp: float, model: str = "chgnet") -> list[float]:
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from scipy.signal import find_peaks

    L, sp, F = _structure(st)
    ua = PhonopyAtoms(symbols=sp, cell=L, scaled_positions=F)
    sm = np.diag([max(1, int(np.ceil(min_len / l))) for l in np.linalg.norm(L, axis=1)])
    while abs(round(np.linalg.det(sm))) * len(sp) > 220:
        i = int(np.argmax(np.diag(sm)))
        sm[i, i] = max(1, sm[i, i] - 1)
    ph = Phonopy(ua, supercell_matrix=sm, primitive_matrix="auto", log_level=0)
    ph.generate_displacements(distance=disp)
    ph.forces = _forces(model, ph.supercells_with_displacements)
    ph.produce_force_constants()
    fallback_qpoints = False
    try:
        ph.run_mesh([mesh] * 3, with_eigenvectors=False)
        fr = ph.get_mesh_dict()["frequencies"] * _THZ                       # (n_q, 3 n_atoms)
    except Exception as exc:  # noqa: BLE001
        # A small number of otherwise valid cells trigger a spglib Niggli
        # reduction failure in phonopy's BZ relocation. Keep the exact
        # structure/model/mesh and evaluate the uniform reciprocal grid
        # directly; do not perturb the lattice or hide other failures.
        if "Niggli reduction failed" not in str(exc):
            raise
        q = np.stack(np.meshgrid(*[np.arange(mesh, dtype=float) / mesh] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
        ph.run_qpoints(q, with_eigenvectors=False)
        fr = ph.get_qpoints_dict()["frequencies"] * _THZ
        fallback_qpoints = True
    flat = fr.reshape(-1)
    pos = flat[flat > 1.0]
    q = (lambda p: float(np.percentile(pos, p))) if pos.size else (lambda p: 0.0)
    top = np.sort(fr, axis=1)[:, -1]
    out = [float(pos.max()) if pos.size else 0.0, q(99), q(95), q(90), float(pos.mean()) if pos.size else 0.0, q(50), q(25),
           float(top.mean()), float(np.median(top)), float(np.percentile(top, 10)), float(top.min()),
           float((flat < -1.0).mean()), float(flat.min())]
    tops, lasts = [], []
    for s in _SIGMAS:
        if fallback_qpoints:
            # run_total_dos enters the failing BZ relocation path too. Use the
            # same Gaussian smearing convention over the direct grid in THz.
            thz = fr.reshape(-1) / _THZ
            pitch = 0.05
            lo, hi = float(thz.min() - 4 * s), float(thz.max() + 4 * s)
            grid = np.arange(lo, hi + pitch, pitch)
            z = (grid[:, None] - thz[None, :]) / float(s)
            d = np.exp(-0.5 * z * z).sum(axis=1) / (float(s) * np.sqrt(2 * np.pi) * max(len(thz), 1))
            f = grid * _THZ
        else:
            ph.run_total_dos(sigma=s, freq_pitch=0.05)
            dos = ph.get_total_dos_dict()
            f, d = np.asarray(dos["frequency_points"]) * _THZ, np.asarray(dos["total_dos"])
        pk, _ = find_peaks(d, height=d.max() * 0.03)
        tops.append(float(f[pk[np.argmax(d[pk])]]) if len(pk) else np.nan)
        lasts.append(float(f[pk[-1]]) if len(pk) else np.nan)
    ph.run_qpoints([[0, 0, 0]])
    gamma = float(np.max(ph.get_qpoints_dict()["frequencies"]) * _THZ)
    return out + tops + lasts + [gamma, float(len(sp))]


def _local(structures, mesh, min_len, disp, model="chgnet") -> np.ndarray:
    cache = _cache_dir()
    rows = []
    for st in structures:
        key = _cache_key(st, mesh, min_len, disp, model)
        f = cache / f"{key}.json" if cache else None
        if f is not None and f.is_file():
            cached = _read_cache_row(f)
            if cached is not None:
                rows.append(cached)
                continue
        try:
            row = _one(st, mesh, min_len, disp, model)
        except Exception:  # noqa: BLE001
            row = [float("nan")] * len(NAMES)
        row = np.asarray(row, dtype=float)
        if row.shape != (len(NAMES),) or not np.isfinite(row).all():
            row = np.full(len(NAMES), np.nan, dtype=float)
        elif f is not None:
            _write_cache_row(f, row.tolist())
        rows.append(row)
    return np.array(rows, dtype=float).reshape(len(rows), len(NAMES))


def _cache_complete(structures, mesh: int, min_len: float, disp: float, model: str) -> bool:
    """Return whether every requested row is already a valid frozen-cache entry.

    A complete prewarm is usable from the lightweight sc-run environment even
    when the GPU-only SevenNet/phonopy packages are not importable there.
    Reading those immutable rows does not execute inference or fit anything.
    """
    cache = _cache_dir()
    if cache is None:
        return False
    for st in structures:
        path = cache / f"{_cache_key(st, mesh, min_len, disp, model)}.json"
        try:
            row = np.asarray(json.loads(path.read_text()), dtype=float)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False
        if row.shape != (len(NAMES),) or not np.isfinite(row).all():
            return False
    return True


def phonon_features(structures, mesh: int = 8, min_len: float = 7.0, disp: float = 0.01, model: str = "sevennet") -> dict:
    if not isinstance(structures, (list, tuple)):
        raise ValueError("structures must be a list of structure dicts")
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    sts = [{"lattice": np.asarray(s["lattice"], dtype=float).tolist(), "species": [str(x) for x in s["species"]],
            "frac_coords": np.asarray(s["frac_coords"], dtype=float).tolist()} for s in structures]
    if not sts:
        return {"X": np.zeros((0, len(NAMES))), "names": list(NAMES)}
    mesh, min_len, disp = int(mesh), float(min_len), float(disp)
    # A complete frozen cache is a valid local source of features.  This is
    # intentionally checked before remote routing so formal agent episodes do
    # not fail merely because sc-run lacks the optional GPU packages.
    if _local_ok(model) or _cache_complete(sts, mesh, min_len, disp, model):
        X = _local(sts, mesh, min_len, disp, model)
    elif _remote.enabled():
        X = _remote.call("matphonon_mlip", "phonon_features",
                         {"structures": sts, "mesh": mesh, "min_len": min_len, "disp": disp, "model": model})["X"]
    else:
        raise RuntimeError(f"phonon_features: the packages for model={model!r} are not available here")
    return {"X": np.asarray(X, dtype=float), "names": list(NAMES)}
