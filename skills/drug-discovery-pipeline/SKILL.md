---
name: drug-discovery-pipeline
description: "Orchestrates a full drug discovery workflow from target identification through lead optimization. Use when searching for drug candidates against a biological target, evaluating compound libraries, or optimizing hits for drug-likeness. NOT for pure protein structure analysis or single-compound lookups."
metadata: { "openclaw": { "emoji": "💉" } }
---

# Drug Discovery Pipeline (Meta Skill)

This meta-skill orchestrates a multi-stage drug discovery workflow by combining
target validation, compound searching, property filtering, and lead optimization
into a single coherent pipeline. It coordinates four specialized skills to move
from a biological target to a ranked list of drug candidates.

## Workflow

### Step 1: Target Validation

Query UniProt for the target protein to gather functional annotations, known
domains, post-translational modifications, and disease associations. Assess
druggability by checking for known binding pockets, ligand-binding domains,
and membership in established druggable protein families (kinases, GPCRs,
ion channels, nuclear receptors).

### Step 2: Known Drug and Compound Survey

Query ChEMBL for existing drugs, clinical candidates, and bioactive compounds
reported against the target. Collect activity data (IC50, Ki, EC50) and note
selectivity profiles. Identify chemical series and mechanism of action classes
already explored in the literature.

### Step 3: Lead Expansion via Similarity Search

Use PubChem similarity and substructure searches to find structural analogs
of the most promising hits from Step 2. Expand the candidate pool by exploring
nearby chemical space using Tanimoto similarity with ECFP4 fingerprints.
Retrieve vendor availability and patent status where possible.

### Step 4: Property Filtering and ADMET Prediction

Apply RDKit to compute molecular descriptors and filter candidates through
established drug-likeness rules:
- Lipinski Rule of Five (MW, LogP, HBD, HBA)
- Veber rules (rotatable bonds, TPSA)
- PAINS filter to remove frequent hitters
- ADMET property estimation (solubility, permeability, CYP inhibition flags)

Remove compounds that violate multiple criteria or show structural alerts.

### Step 5: Compound Ranking and Prioritization

Score remaining candidates using a weighted multi-parameter optimization:
- Potency (pIC50 or pKi against target)
- Selectivity (activity ratio vs. off-targets)
- Drug-likeness (QED score)
- Synthetic accessibility (SA score)
- Novelty (Tanimoto distance from known drugs)

Output a ranked table of top candidates with reasoning for each score.

## Integration Points

- **uniprot-protein** -- Target protein annotation, domain architecture, druggability assessment
- **chembl-drug** -- Bioactivity data, existing drugs, SAR context for the target
- **pubchem-compound** -- Similarity searching, analog identification, vendor availability
- **rdkit-chemistry** -- Descriptor calculation, filtering rules, ADMET prediction, scoring

## Output Formats

- **Target summary**: Protein name, function, druggability assessment, known ligands
- **Compound table**: SMILES, name, source, activity, drug-likeness scores
- **Ranked list**: Top 10-20 candidates with composite scores and rationale
- **SAR notes**: Observed structure-activity trends across chemical series

## Best Practices

1. Always validate the target before searching for compounds to avoid wasted effort
2. Set activity thresholds early (e.g., IC50 < 1 uM) to keep the candidate pool manageable
3. Use multiple fingerprint types for similarity search to capture diverse analogs
4. Apply PAINS filters before investing effort in detailed ADMET analysis
5. Document the rationale for each filtering step to maintain reproducibility
6. Consider the therapeutic area when weighting ranking criteria
7. Flag compounds with known IP restrictions or limited synthetic routes
8. Cross-check top candidates against ChEMBL for any reported toxicity signals
9. Present results with confidence levels reflecting data quality and coverage
10. Iterate the pipeline if initial results are sparse by relaxing similarity thresholds
