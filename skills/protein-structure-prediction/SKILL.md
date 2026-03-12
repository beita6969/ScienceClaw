---
name: protein-structure-prediction
description: COPYRIGHT NOTICE
---

<!--
# COPYRIGHT NOTICE
# This file is part of the "Universal Biomedical Skills" project.
# Copyright (c) 2026 MD BABU MIA, PhD <md.babu.mia@mssm.edu>
# All Rights Reserved.
#
# This code is proprietary and confidential.
# Unauthorized copying of this file, via any medium is strictly prohibited.
#
# Provenance: Authenticated by MD BABU MIA

-->

---
name: 'protein-structure-prediction'
description: 'Predicts 3D protein structures from amino acid sequences using ESMFold or AlphaFold3 (mock).'
measurable_outcome: Execute skill workflow successfully with valid output within 15 minutes.
allowed-tools:
  - read_file
  - run_shell_command
---


# Protein Structure Prediction (ESMFold/AF3)

The **Protein Structure Prediction Skill** provides an interface to state-of-the-art folding models. It takes an amino acid sequence and returns a PDB file or structure metrics (pLDDT).

## When to Use This Skill

*   When you have a protein sequence and need its 3D coordinates.
*   To check if a designed sequence folds into a stable structure.
*   To prepare a receptor for docking simulations.

## Core Capabilities

1.  **Folding**: Generates atomic coordinates (PDB format).
2.  **Confidence Scoring**: Returns pLDDT scores per residue.
3.  **Visualization**: (Optional) Generates a static view of the structure.

## Workflow

1.  **Input**: Amino acid sequence (FASTA string).
2.  **Process**: Sends sequence to ESMFold API (or local inference).
3.  **Output**: Saves `.pdb` file and returns confidence metrics.

## Example Usage

**User**: "Fold this sequence: MKTIIALSY..."

**Agent Action**:
```bash
python3 Skills/Drug_Discovery/Protein_Structure/esmfold_client.py \
    --sequence "MKTIIALSYIFCLVFDYDY" \
    --output structure.pdb
```



<!-- AUTHOR_SIGNATURE: 9a7f3c2e-MD-BABU-MIA-2026-MSSM-SECURE -->