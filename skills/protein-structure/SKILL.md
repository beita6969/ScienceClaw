---
name: protein-structure
description: Analyzes protein 3D structures, performs homology modeling, interprets AlphaFold predictions, conducts molecular docking, and evaluates protein-ligand interactions; trigger when users ask about PDB files, folding, binding sites, or structural biology.
---

## When to Trigger

Activate this skill when the user mentions:
- Protein folding, AlphaFold, ESMFold, RoseTTAFold
- PDB files, structural analysis, Ramachandran plots
- Molecular docking, binding affinity, binding pockets
- Homology modeling, threading, ab initio structure prediction
- Protein-protein interactions (PPI), interface analysis
- Structural alignment, RMSD, TM-score
- Cryo-EM, X-ray crystallography data interpretation

## Step-by-Step Methodology

1. **Retrieve or predict structure** - Search PDB for experimental structures (by UniProt ID or gene name). If unavailable, use AlphaFold DB or run ESMFold. Check pLDDT confidence scores for predicted structures.
2. **Quality assessment** - For experimental structures: check resolution, R-free, and completeness. For predictions: evaluate pLDDT per-residue and PAE (predicted aligned error) matrices.
3. **Structural analysis** - Identify secondary structure elements (helices, sheets, loops). Compute solvent-accessible surface area. Map conserved residues and functional domains.
4. **Binding site identification** - Use fpocket, SiteMap, or DoGSiteScorer for pocket detection. Cross-reference with known ligand binding from PDBe or BindingDB.
5. **Molecular docking** - Recommend AutoDock Vina, GNINA, or Glide. Define grid box around binding site. Report binding energy (kcal/mol) and key interactions (H-bonds, hydrophobic, pi-stacking).
6. **Structural comparison** - Align structures using TM-align or FATCAT. Report RMSD and TM-score. Identify conformational changes between states.
7. **Visualization guidance** - Recommend PyMOL, ChimeraX, or Mol* for rendering. Specify coloring schemes (by chain, B-factor, electrostatics, or conservation).

## Key Databases and Tools

- **PDB / PDBe** - Experimental protein structures
- **AlphaFold DB** - AI-predicted structures for UniProt entries
- **UniProt** - Protein sequences, domains, and annotations
- **InterPro / Pfam** - Domain classification
- **BindingDB / ChEMBL** - Binding affinity data
- **RCSB PDB REST API** - Programmatic structure queries
- **PDBe-KB** - Aggregated structural annotations

## Output Format

- Report structures with PDB ID, resolution, method, and chain identifiers.
- Binding energies in kcal/mol with interaction fingerprints.
- Structural alignments with RMSD (in Angstroms) and TM-score.
- Residue numbering must match the canonical UniProt sequence or PDB SEQRES.
- Include PyMOL/ChimeraX commands for reproducing visualizations.

## Quality Checklist

- [ ] Structure source and method (X-ray, cryo-EM, NMR, predicted) clearly stated
- [ ] Resolution or confidence metric (pLDDT) reported
- [ ] Binding site residues listed with chain and numbering
- [ ] Docking results include top poses and interaction details
- [ ] Limitations of predicted structures acknowledged where relevant
- [ ] All PDB IDs verified as valid and current
- [ ] Biological assembly vs. asymmetric unit distinguished
