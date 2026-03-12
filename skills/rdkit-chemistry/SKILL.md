---
name: rdkit-chemistry
description: "Molecular chemistry operations via RDKit. Use when: user asks about molecular structures, SMILES, chemical properties, or fingerprints. NOT for: reaction databases or wet lab protocols."
metadata: { "openclaw": { "emoji": "\u2697\uFE0F", "requires": { "bins": ["python3"] }, "install": [{ "id": "uv-rdkit", "kind": "uv", "package": "rdkit" }] } }
---

# RDKit Chemistry

Molecular chemistry operations using RDKit.

## When to Use

- Molecular structures, SMILES parsing, or validation
- Molecular properties (MW, logP, TPSA, HBD/HBA)
- Substructure searching or molecular filtering
- Fingerprints (Morgan, MACCS) and similarity calculations
- 2D depiction or molecular image generation

## When NOT to Use

- Reaction databases or retrosynthesis planning
- Wet lab protocols or experimental procedures
- Protein structure analysis (use biopython-bio)
- Quantum chemistry or DFT calculations

## SMILES Parsing and Properties

```python
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

mol = Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O')  # Aspirin
if mol is None:
    print("Invalid SMILES")

canonical = Chem.MolToSmiles(mol)                      # Canonical SMILES
mw = Descriptors.MolWt(mol)                            # Molecular weight
logp = Descriptors.MolLogP(mol)                        # Partition coefficient
tpsa = Descriptors.TPSA(mol)                           # Topological polar surface area
hbd = rdMolDescriptors.CalcNumHBD(mol)                 # H-bond donors
hba = rdMolDescriptors.CalcNumHBA(mol)                 # H-bond acceptors
rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)

# SMARTS substructure match
pattern = Chem.MolFromSmarts('[OH]')
has_oh = mol.HasSubstructMatch(pattern)
```

## Lipinski Rule of Five

```python
def lipinski(smi):
    mol = Chem.MolFromSmiles(smi)
    return {
        'MW <= 500': Descriptors.MolWt(mol) <= 500,
        'LogP <= 5': Descriptors.MolLogP(mol) <= 5,
        'HBD <= 5': rdMolDescriptors.CalcNumHBD(mol) <= 5,
        'HBA <= 10': rdMolDescriptors.CalcNumHBA(mol) <= 10,
    }
```

## Substructure Search

```python
molecules = [Chem.MolFromSmiles(s) for s in ['CCO', 'CC(=O)O', 'c1ccccc1', 'c1ccccc1O']]
pattern = Chem.MolFromSmarts('c1ccccc1')  # Benzene ring
hits = [m for m in molecules if m.HasSubstructMatch(pattern)]
```

## Fingerprints and Similarity

```python
from rdkit.Chem import AllChem, MACCSkeys
from rdkit import DataStructs

mol1 = Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O')
mol2 = Chem.MolFromSmiles('CC(=O)Nc1ccc(O)cc1')

# Morgan (circular) fingerprints — radius 2 ~ ECFP4
fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)

# MACCS keys
mfp1 = MACCSkeys.GenMACCSKeys(mol1)

# Tanimoto / Dice similarity
tanimoto = DataStructs.TanimotoSimilarity(fp1, fp2)
dice = DataStructs.DiceSimilarity(fp1, fp2)
```

## 2D Depiction

```python
from rdkit.Chem import Draw

Draw.MolToFile(mol, 'molecule.png', size=(300, 300))

# Grid of molecules
mols = [Chem.MolFromSmiles(s) for s in ['CCO', 'CC(=O)O', 'c1ccccc1O']]
img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(300, 300))
img.save('grid.png')
```

## Quick One-liner

```bash
python3 -c "
from rdkit import Chem; from rdkit.Chem import Descriptors
mol = Chem.MolFromSmiles('CCO')
print(f'MW: {Descriptors.MolWt(mol):.2f}, LogP: {Descriptors.MolLogP(mol):.2f}')
"
```

## Best Practices

1. Always check `MolFromSmiles()` return for `None` (invalid SMILES).
2. Use canonical SMILES for consistent comparisons.
3. Morgan radius=2 (ECFP4) is standard for similarity screening.
4. Sanitize molecules before property calculations.
5. Use `Chem.AddHs(mol)` before 3D coordinate generation.
