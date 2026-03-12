---
name: drug-discovery
description: Supports drug discovery workflows including target identification, virtual screening, ADMET prediction, lead optimization, pharmacokinetics modeling, and drug repurposing analyses; trigger when users discuss drug targets, compound libraries, medicinal chemistry, or pharmaceutical development.
---

## When to Trigger

Activate this skill when the user mentions:
- Drug target identification, druggability assessment
- Virtual screening, molecular docking, pharmacophore
- ADMET (absorption, distribution, metabolism, excretion, toxicity)
- Lead optimization, SAR (structure-activity relationship)
- Pharmacokinetics (PK), pharmacodynamics (PD), PK/PD modeling
- Drug repurposing, off-label, drug-disease associations
- SMILES, InChI, compound libraries, chemical fingerprints
- IC50, EC50, Ki, dose-response curves

## Step-by-Step Methodology

1. **Target identification and validation** - Identify therapeutic target from literature, GWAS hits, or omics data. Assess druggability using Open Targets, DGIdb, or structural pocket analysis. Confirm target-disease association strength.
2. **Compound sourcing** - Search ChEMBL, PubChem, ZINC, or DrugBank for known active compounds. For novel scaffolds, consider de novo design tools (REINVENT, MolGPT).
3. **Virtual screening** - Structure-based: dock compound library against target (AutoDock Vina, Glide). Ligand-based: use pharmacophore models or molecular fingerprint similarity. Filter by drug-likeness (Lipinski Ro5, Veber rules).
4. **ADMET prediction** - Predict absorption (Caco-2 permeability, logP), distribution (plasma protein binding, Vd), metabolism (CYP inhibition/induction), excretion (clearance), and toxicity (hERG, hepatotoxicity, AMES mutagenicity). Use SwissADME, pkCSM, or ADMETlab.
5. **Lead optimization** - Analyze SAR from dose-response data. Identify key pharmacophoric features. Suggest modifications to improve potency, selectivity, or ADMET profile while maintaining drug-likeness.
6. **PK/PD modeling** - Build compartmental PK models. Estimate key parameters: Cmax, Tmax, AUC, half-life, bioavailability. For PD, model dose-response (Emax model, Hill equation).
7. **Drug repurposing analysis** - Query drug-gene interaction databases. Analyze shared pathways between drug targets and disease mechanisms. Check clinical trial databases for existing evidence.

## Key Databases and Tools

- **ChEMBL** - Bioactivity data for drug-like compounds
- **PubChem** - Chemical structure and bioassay data
- **DrugBank** - Drug and target information
- **Open Targets** - Target-disease associations
- **ZINC** - Purchasable compound library
- **SwissADME / pkCSM** - ADMET prediction tools
- **BindingDB** - Protein-ligand binding data

## Output Format

- Compound results as tables: SMILES, molecular weight, logP, key activity (IC50/EC50), ADMET flags.
- Docking results: binding energy (kcal/mol), key interactions, pose description.
- PK parameters: Cmax, Tmax, AUC, t1/2, clearance, bioavailability with units.
- SAR analysis: matched molecular pair comparisons with activity changes.

## Quality Checklist

- [ ] Target-disease association supported by evidence (genetic, functional)
- [ ] Drug-likeness filters applied (Lipinski, Veber, PAINS)
- [ ] ADMET predictions include confidence levels or applicability domain
- [ ] Docking validated against known co-crystal structures when available
- [ ] IC50/EC50 reported with assay conditions and confidence intervals
- [ ] PK parameters include units and species (human vs. preclinical)
- [ ] Known liabilities (hERG, CYP inhibition, reactive metabolites) flagged
- [ ] Comparison to existing drugs/compounds for the same target included
