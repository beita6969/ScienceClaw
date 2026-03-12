---
name: chemistry-tools
description: Computational chemistry tools including molecular structure, chemical reactions, thermodynamics, spectroscopy analysis, and cheminformatics. Use when user works with chemical formulas, molecular structures, reaction balancing, thermodynamic calculations, or chemical databases (PubChem, ChemSpider). Triggers on "chemical structure", "molecular weight", "balance equation", "reaction", "thermodynamics", "spectroscopy", "SMILES", "PubChem", "chemical formula", "stoichiometry".
---

# Chemistry Tools

Computational chemistry and cheminformatics. Venv: `source /Users/zhangmingda/clawd/.venv/bin/activate`

## Molecular Properties

```python
# Using RDKit if available, otherwise manual calculations
from sympy import symbols, Eq, solve

# Molecular weight calculation (manual)
ATOMIC_WEIGHTS = {
    'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.81,
    'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
    'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.086, 'P': 30.974,
    'S': 32.065, 'Cl': 35.453, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078,
    'Fe': 55.845, 'Cu': 63.546, 'Zn': 65.38, 'Br': 79.904, 'Ag': 107.868,
    'I': 126.904, 'Au': 196.967,
}

import re
def molecular_weight(formula):
    """Calculate MW from chemical formula like 'C6H12O6'"""
    elements = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    mw = sum(ATOMIC_WEIGHTS.get(el, 0) * (int(n) if n else 1) for el, n in elements)
    return mw

# Example
print(f"Glucose (C6H12O6): {molecular_weight('C6H12O6'):.3f} g/mol")
```

## Chemical Equation Balancing

```python
from sympy import Matrix, lcm

def balance_equation(reactants_elements, products_elements):
    """
    Balance using linear algebra (null space method).
    Each compound is a dict of {element: count}.
    """
    all_elements = set()
    for compound in reactants_elements + products_elements:
        all_elements.update(compound.keys())
    all_elements = sorted(all_elements)
    
    n_compounds = len(reactants_elements) + len(products_elements)
    matrix = []
    for el in all_elements:
        row = []
        for comp in reactants_elements:
            row.append(comp.get(el, 0))
        for comp in products_elements:
            row.append(-comp.get(el, 0))
        matrix.append(row)
    
    M = Matrix(matrix)
    null = M.nullspace()
    if null:
        coeffs = null[0]
        # Make integer coefficients
        denom = lcm(*[c.q for c in coeffs if hasattr(c, 'q')] or [1])
        coeffs = [int(c * denom) for c in coeffs]
        return coeffs
    return None
```

## Thermodynamics

```python
import numpy as np

# Ideal gas law: PV = nRT
R = 8.314  # J/(mol·K)

def ideal_gas(P=None, V=None, n=None, T=None):
    """Solve for the missing variable. Units: Pa, m³, mol, K"""
    if P is None: return n * R * T / V
    if V is None: return n * R * T / P
    if n is None: return P * V / (R * T)
    if T is None: return P * V / (n * R)

# Gibbs free energy
def gibbs(dH, T, dS):
    """ΔG = ΔH - TΔS (kJ/mol, K, kJ/(mol·K))"""
    return dH - T * dS

# Nernst equation
def nernst(E0, n_electrons, Q, T=298.15):
    """E = E° - (RT/nF)ln(Q)"""
    F = 96485  # C/mol
    return E0 - (R * T / (n_electrons * F)) * np.log(Q)

# Arrhenius equation
def arrhenius(A, Ea, T):
    """k = A * exp(-Ea/RT), Ea in J/mol"""
    return A * np.exp(-Ea / (R * T))
```

## Chemical Databases

### PubChem
```bash
# Search by name
curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/aspirin/JSON" | python3 -m json.tool

# Search by SMILES
curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/CC(=O)OC1=CC=CC=C1C(=O)O/JSON"

# Get properties
curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/caffeine/property/MolecularFormula,MolecularWeight,IUPACName/JSON"
```

### ChEBI (Chemical Entities of Biological Interest)
```bash
curl -s "https://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:15377" # water
```

## Spectroscopy Reference

| Technique | What it measures | Key info |
|-----------|-----------------|----------|
| IR | Bond vibrations | Functional groups (cm⁻¹) |
| NMR (¹H) | H environments | Chemical shift (δ ppm), splitting |
| NMR (¹³C) | C environments | Chemical shift (δ ppm) |
| UV-Vis | Electronic transitions | λmax, absorbance |
| Mass Spec | m/z ratio | Molecular weight, fragmentation |

### Common IR Absorptions
- O-H stretch: 3200-3600 cm⁻¹ (broad)
- N-H stretch: 3300-3500 cm⁻¹
- C-H stretch: 2850-3000 cm⁻¹
- C=O stretch: 1650-1750 cm⁻¹
- C=C stretch: 1600-1680 cm⁻¹
- C-O stretch: 1000-1300 cm⁻¹

## Tips
- Always check units (SI vs CGS vs practical)
- Use IUPAC nomenclature
- For complex reactions, break into elementary steps
- Verify thermodynamic data against NIST WebBook
- For computational chemistry (DFT, MD), recommend specialized software (Gaussian, ORCA, GROMACS)
