---
name: pymatgen-materials
description: "Materials science computation with pymatgen. Use when: (1) crystal structure creation and manipulation, (2) phase diagram construction, (3) electronic structure analysis, (4) symmetry and space group operations, (5) VASP input/output parsing. NOT for: molecular chemistry (use rdkit-chemistry), protein structure (use biopython-bio), or general numerical computation (use scipy-analysis)."
metadata: { "openclaw": { "emoji": "💎", "requires": { "bins": ["python3"] }, "install": [{ "id": "uv-pymatgen", "kind": "uv", "package": "pymatgen" }] } }
---

# Pymatgen Materials Science

Materials science analysis using pymatgen for crystal structures, phase diagrams,
electronic structure, and computational materials workflows.

## When to Use

- Crystal structure creation, manipulation, and visualization
- Phase diagram construction and thermodynamic stability analysis
- Electronic structure (band structure, DOS) parsing and plotting
- Symmetry analysis and space group determination
- Reading/writing VASP, CIF, POSCAR, and other structure formats

## When NOT to Use

- Molecular chemistry or drug design (use rdkit-chemistry)
- Protein or biomolecular structure (use biopython-bio)
- General plotting without materials context (use matplotlib-viz)
- Machine learning on materials data (use dedicated ML frameworks)

## Crystal Structure Creation

```python
from pymatgen.core import Structure, Lattice, Molecule

# Create from spacegroup and Wyckoff positions
structure = Structure.from_spacegroup(
    "Fm-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]]
)

# Create from file
structure = Structure.from_file("POSCAR")
structure = Structure.from_file("structure.cif")

# Create manually with lattice parameters
lattice = Lattice.from_parameters(a=3.84, b=3.84, c=3.84, alpha=90, beta=90, gamma=90)
structure = Structure(lattice, ["Cu", "Cu", "Cu", "Cu"],
                      [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])

# Hexagonal lattice
lattice = Lattice.hexagonal(a=2.46, c=6.71)
```

## Structure Properties and Manipulation

```python
# Basic properties
print(structure.lattice)                      # lattice vectors
print(structure.lattice.abc)                  # a, b, c lengths
print(structure.lattice.angles)               # alpha, beta, gamma
print(structure.volume)                       # unit cell volume
print(structure.density)                      # density in g/cm^3
print(structure.composition)                  # chemical formula
print(len(structure))                         # number of sites

# Space group and symmetry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
sga = SpacegroupAnalyzer(structure)
print(sga.get_space_group_symbol())           # e.g., "Fm-3m"
print(sga.get_space_group_number())           # e.g., 225
print(sga.get_point_group_symbol())           # e.g., "m-3m"
conventional = sga.get_conventional_standard_structure()
primitive = sga.get_primitive_standard_structure()

# Supercell and perturbation
supercell = structure.copy()
supercell.make_supercell([2, 2, 2])
structure.perturb(0.01)                       # random perturbation

# Write to file
structure.to(filename="POSCAR")
structure.to(filename="output.cif")
structure.to(fmt="poscar")                    # return as string
```

## Phase Diagrams

```python
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter, PDEntry
from pymatgen.core import Composition

# Build entries from computed energies
entries = [
    PDEntry(Composition("Li"), -1.9),
    PDEntry(Composition("Fe"), -8.3),
    PDEntry(Composition("O"), -4.95),
    PDEntry(Composition("LiFePO4"), -43.2),
    PDEntry(Composition("FePO4"), -38.5),
    PDEntry(Composition("Li3PO4"), -25.6),
]

# Construct phase diagram
pd = PhaseDiagram(entries)

# Stability analysis
for entry in entries:
    ehull = pd.get_e_above_hull(entry)
    print(f"{entry.composition.reduced_formula}: E_above_hull = {ehull:.3f} eV/atom")

# Decomposition products
decomp, ehull = pd.get_decomp_and_e_above_hull(entries[3])
for comp, amount in decomp.items():
    print(f"  {comp.reduced_formula}: {amount:.3f}")

# Plot
plotter = PDPlotter(pd)
plotter.get_plot().savefig("phase_diagram.pdf")
```

## Electronic Structure

```python
from pymatgen.io.vasp import Vasprun, BSVasprun
from pymatgen.electronic_structure.plotter import BSPlotter, DosPlotter

# Parse VASP output
vasprun = Vasprun("vasprun.xml", parse_dos=True, parse_eigen=True)
print(f"Final energy: {vasprun.final_energy} eV")
print(f"Converged: {vasprun.converged}")

# Band structure
bs_vasprun = BSVasprun("vasprun.xml")
bs = bs_vasprun.get_band_structure(line_mode=True)
print(f"Band gap: {bs.get_band_gap()['energy']:.3f} eV")
print(f"Direct: {bs.get_band_gap()['direct']}")

bs_plotter = BSPlotter(bs)
bs_plotter.get_plot().savefig("band_structure.pdf")

# Density of states
dos = vasprun.complete_dos
dos_plotter = DosPlotter()
dos_plotter.add_dos("Total", dos)
dos_plotter.add_dos_dict(dos.get_element_dos())
dos_plotter.get_plot().savefig("dos.pdf")
```

## VASP I/O

```python
from pymatgen.io.vasp import Poscar, Incar, Kpoints, Outcar

# Read VASP files
poscar = Poscar.from_file("POSCAR")
structure = poscar.structure

incar = Incar.from_file("INCAR")
print(incar["ENCUT"])

kpoints = Kpoints.from_file("KPOINTS")

outcar = Outcar("OUTCAR")
print(f"Total magnetization: {outcar.total_mag}")
print(f"Final energy: {outcar.final_energy}")

# Create VASP input set
from pymatgen.io.vasp.sets import MPRelaxSet
relax_set = MPRelaxSet(structure)
relax_set.write_input("vasp_input/")
```

## Best Practices

1. Always check symmetry with `SpacegroupAnalyzer` after structure creation.
2. Use `get_primitive_standard_structure()` to reduce computational cost.
3. Validate structures with `structure.is_valid()` before running calculations.
4. Use Materials Project API (`MPRester`) to fetch known structures and energies.
5. When comparing energies, normalize per atom (`energy / len(structure)`).
6. Save structures in CIF format for archival and POSCAR for VASP input.
7. Use `structure.get_neighbors()` for local environment analysis.
8. Check convergence flags in `Vasprun` before trusting computed properties.
