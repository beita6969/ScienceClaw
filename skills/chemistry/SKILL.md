---
name: chemistry
description: # Chemistry & Drug Discovery
---

# Chemistry & Drug Discovery

## Overview
Computational chemistry, cheminformatics, and drug discovery workflows.

## Key Tools
- **RDKit**: Molecular manipulation, fingerprints, descriptors, substructure search
- **PubChem**: Chemical compound database (100M+ compounds)
- **ChEMBL**: Bioactivity database for drug-like molecules
- **Open Babel**: Format conversion, 3D generation
- **AutoDock Vina**: Molecular docking
- **GROMACS/OpenMM**: Molecular dynamics simulations

## Common Workflows

### Virtual Screening
1. Define target (protein structure from PDB)
2. Prepare compound library (from ChEMBL/ZINC/Enamine)
3. Filter by drug-likeness (Lipinski's Rule of Five)
4. Docking (AutoDock Vina, GNINA)
5. Scoring and ranking
6. ADMET prediction (absorption, distribution, metabolism, excretion, toxicity)
7. Hit validation

### QSAR Modeling
1. Curate activity data (ChEMBL IC50/Ki/EC50)
2. Calculate molecular descriptors (RDKit)
3. Feature selection
4. Model training (Random Forest, XGBoost, neural network)
5. Validation (cross-validation, external test set)
6. Applicability domain assessment

### Molecular Property Prediction
- Lipophilicity (LogP)
- Solubility (LogS)
- Permeability (PAMPA, Caco-2)
- Metabolic stability (CYP inhibition)
- hERG toxicity
- BBB penetration

## Databases
| Database | Content | Access |
|----------|---------|--------|
| PubChem | 100M+ compounds | Free API |
| ChEMBL | Bioactivity data | Free API |
| PDB | 200K+ protein structures | Free API |
| ZINC | Purchasable compounds | Free download |
| DrugBank | Drug information | Free academic |
| Materials Project | Inorganic materials | Free API |
