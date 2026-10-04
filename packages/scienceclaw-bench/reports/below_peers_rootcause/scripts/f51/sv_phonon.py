"""Phonopy + ASE-calculator phonon features (same 21 columns as scilib.matphonon_mlip); usage: sv_phonon.py MODEL OUT.npy [start end]"""
import sys, json, time, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from scipy.signal import find_peaks
from ase import Atoms

THZ = 33.35641
SIG = (0.1, 0.25, 0.5)
model, outp = sys.argv[1], sys.argv[2]
S = json.load(open("/home/bedicloud/sharestore2/zxc/scienceclaw/f51/structs.json"))
lo = int(sys.argv[3]) if len(sys.argv) > 3 else 0
hi = int(sys.argv[4]) if len(sys.argv) > 4 else len(S)

if model.startswith("7net"):
    from sevenn.calculator import SevenNetCalculator
    kw = {"modal": sys.argv[5]} if len(sys.argv) > 5 else {}
    calc = SevenNetCalculator(model, device="cuda", **kw)
elif model == "mace":
    from mace.calculators import mace_mp
    calc = mace_mp(model="medium", device="cuda", default_dtype="float32")
else:
    raise SystemExit("unknown model")


def forces(cells):
    out = []
    for s in cells:
        at = Atoms(symbols=s.symbols, cell=s.cell, scaled_positions=s.scaled_positions, pbc=True)
        at.calc = calc
        out.append(at.get_forces())
    return np.array(out)


def one(st, mesh=8, min_len=7.0, disp=0.01):
    L = np.asarray(st["lattice"], float); sp = st["species"]; F = np.asarray(st["frac_coords"], float)
    ua = PhonopyAtoms(symbols=sp, cell=L, scaled_positions=F)
    sm = np.diag([max(1, int(np.ceil(min_len / l))) for l in np.linalg.norm(L, axis=1)])
    while abs(round(np.linalg.det(sm))) * len(sp) > 220:
        i = int(np.argmax(np.diag(sm))); sm[i, i] = max(1, sm[i, i] - 1)
    ph = Phonopy(ua, supercell_matrix=sm, primitive_matrix="auto", log_level=0)
    ph.generate_displacements(distance=disp)
    ph.forces = forces(ph.supercells_with_displacements)
    ph.produce_force_constants()
    ph.run_mesh([mesh] * 3, with_eigenvectors=False)
    fr = ph.get_mesh_dict()["frequencies"] * THZ
    flat = fr.reshape(-1); pos = flat[flat > 1.0]
    q = (lambda p: float(np.percentile(pos, p))) if pos.size else (lambda p: 0.0)
    top = np.sort(fr, axis=1)[:, -1]
    out = [float(pos.max()) if pos.size else 0.0, q(99), q(95), q(90), float(pos.mean()) if pos.size else 0.0, q(50), q(25),
           float(top.mean()), float(np.median(top)), float(np.percentile(top, 10)), float(top.min()),
           float((flat < -1.0).mean()), float(flat.min())]
    tops, lasts = [], []
    for s in SIG:
        ph.run_total_dos(sigma=s, freq_pitch=0.05)
        dos = ph.get_total_dos_dict()
        f, d = np.asarray(dos["frequency_points"]) * THZ, np.asarray(dos["total_dos"])
        pk, _ = find_peaks(d, height=d.max() * 0.03)
        tops.append(float(f[pk[np.argmax(d[pk])]]) if len(pk) else np.nan)
        lasts.append(float(f[pk[-1]]) if len(pk) else np.nan)
    ph.run_qpoints([[0, 0, 0]])
    gamma = float(np.max(ph.get_qpoints_dict()["frequencies"]) * THZ)
    return out + tops + lasts + [gamma, float(len(sp))]


rows = np.full((len(S), 21), np.nan)
t0 = time.time()
for i in range(lo, hi):
    try:
        rows[i] = one(S[i])
    except Exception as e:  # noqa
        print("fail", i, repr(e)[:100], flush=True)
    if (i - lo) % 50 == 0:
        print(i, f"{time.time() - t0:.0f}s", flush=True)
        np.save(outp, rows)
np.save(outp, rows)
print("done", time.time() - t0, flush=True)
