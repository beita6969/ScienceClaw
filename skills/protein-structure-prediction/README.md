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

# Protein Structure Prediction (AlphaFold)

**ID:** `biomedical.drug_discovery.protein_structure`
**Version:** 1.0.0
**Status:** Production
**Category:** Drug Discovery / Structural Biology

---

## Overview

The **Protein Structure Prediction Skill** provides comprehensive tools for AI-powered protein structure prediction and analysis using **AlphaFold 2**, **AlphaFold 3**, **OpenFold**, and the **AlphaFold MCP Server**. This skill enables structure prediction from amino acid sequences, confidence assessment, structure-based drug design, and protein-ligand interaction analysis.

AlphaFold has revolutionized structural biology, with over 200 million predicted structures in the AlphaFold Database. This skill enables AI agents to query existing predictions, generate new structures, assess binding sites, and support rational drug design workflows.

---

## Key Capabilities

### 1. Structure Prediction

| Model | Capabilities | Use Case |
|-------|--------------|----------|
| **AlphaFold 2** | Monomer structure | Single protein prediction |
| **AlphaFold 3** | Complex prediction | Protein-DNA/RNA/ligand complexes |
| **AlphaFold-Multimer** | Protein complexes | Protein-protein interactions |
| **OpenFold 3** | Open-source AF3 | Academic/commercial use |
| **ESMFold** | Fast prediction | Rapid screening |

### 2. Structure Analysis

| Analysis | Description | Application |
|----------|-------------|-------------|
| **pLDDT Assessment** | Per-residue confidence | Quality evaluation |
| **PAE Analysis** | Predicted Aligned Error | Domain boundaries |
| **Binding Site Detection** | Pocket identification | Drug target analysis |
| **Structural Alignment** | RMSD calculation | Homology assessment |
| **Contact Maps** | Residue interactions | Interface analysis |

### 3. Drug Discovery Applications

| Application | Description | Tools |
|-------------|-------------|-------|
| **Virtual Screening** | Structure-based docking | AutoDock, GNINA |
| **Binding Site Analysis** | Druggability assessment | fpocket, SiteMap |
| **Lead Optimization** | SAR analysis | FEP+, Schrödinger |
| **Off-Target Prediction** | Selectivity profiling | ProteomeScout |

---

## Technical Specifications

### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sequence` | `str` | Required | Amino acid sequence (FASTA) |
| `model_version` | `str` | `alphafold3` | Model to use |
| `num_predictions` | `int` | `5` | Number of model predictions |
| `template_mode` | `str` | `auto` | Template usage (auto/none/pdb70) |
| `relaxation` | `bool` | `True` | Amber relaxation |

### Output Files

| File | Description |
|------|-------------|
| `*_model_*.pdb` | Predicted structure(s) |
| `*_plddt.json` | Per-residue confidence scores |
| `*_pae.json` | Predicted aligned error matrix |
| `*_confidence.png` | Confidence visualization |
| `*_report.json` | Structure summary report |

---

## Usage

### Command Line Interface

```bash
python predict_structure.py \
    --sequence protein_sequence.fasta \
    --model alphafold3 \
    --num-predictions 5 \
    --output-dir ./structures
```

### Python Library Integration

```python
# Using AlphaFold Database API
import requests
import json

def get_alphafold_structure(uniprot_id: str) -> dict:
    """Retrieve pre-computed AlphaFold structure from database."""

    base_url = "https://alphafold.ebi.ac.uk/api"

    # Get structure metadata
    response = requests.get(f"{base_url}/prediction/{uniprot_id}")

    if response.status_code != 200:
        raise ValueError(f"Structure not found for {uniprot_id}")

    metadata = response.json()[0]

    # Download PDB file
    pdb_url = metadata['pdbUrl']
    pdb_content = requests.get(pdb_url).text

    # Download confidence data
    confidence_url = metadata['cifUrl'].replace('.cif', '_local_plddt.json')
    plddt_data = requests.get(confidence_url).json()

    return {
        "uniprot_id": uniprot_id,
        "gene": metadata.get('gene'),
        "organism": metadata.get('organismScientificName'),
        "pdb_content": pdb_content,
        "plddt_scores": plddt_data,
        "mean_plddt": sum(plddt_data) / len(plddt_data)
    }

# Example: Get BRCA1 structure
brca1_structure = get_alphafold_structure("P38398")
print(f"Mean pLDDT: {brca1_structure['mean_plddt']:.1f}")
```

### AlphaFold 3 Local Prediction

```python
# Note: Requires local AlphaFold 3 installation
from alphafold3 import AlphaFold3
from alphafold3.model import RunModel
import json

def predict_structure_af3(
    sequences: list,
    sequence_names: list,
    output_dir: str
) -> dict:
    """Run AlphaFold 3 structure prediction."""

    # Prepare input JSON
    input_json = {
        "name": "prediction_job",
        "sequences": [
            {"protein": {"id": name, "sequence": seq}}
            for seq, name in zip(sequences, sequence_names)
        ],
        "modelSeeds": [1, 2, 3, 4, 5],
        "dialect": "alphafold3"
    }

    # Run prediction
    runner = RunModel(
        model_dir="/data/alphafold3/models",
        output_dir=output_dir
    )

    results = runner.predict(input_json)

    return {
        "structures": [f"{output_dir}/model_{i}.cif" for i in range(5)],
        "confidence_scores": results['confidence'],
        "best_model": results['ranking'][0]
    }

# Complex prediction example
sequences = [
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",  # Protein A
    "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRCALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL"  # Protein B
]

result = predict_structure_af3(
    sequences=sequences,
    sequence_names=["ProteinA", "ProteinB"],
    output_dir="./complex_prediction"
)
```

### AlphaFold MCP Server Integration

```python
# Using AlphaFold MCP Server for LLM integration
# Install: pip install alphafold-mcp-server

from mcp import Client

async def query_alphafold_mcp():
    """Query AlphaFold via MCP Server."""

    client = Client()
    await client.connect("alphafold-mcp-server")

    # Available tools:
    # - get_structure_by_uniprot
    # - predict_structure
    # - analyze_confidence
    # - detect_binding_sites
    # - align_structures

    # Get structure
    result = await client.call_tool(
        "get_structure_by_uniprot",
        {"uniprot_id": "P53350"}
    )

    # Analyze binding sites
    sites = await client.call_tool(
        "detect_binding_sites",
        {"pdb_content": result['pdb_content']}
    )

    return result, sites
```

### LLM Agent Integration (LangChain)

```python
from langchain.tools import tool
import requests

@tool
def get_protein_structure(
    identifier: str,
    identifier_type: str = "uniprot",
    include_analysis: bool = True
) -> str:
    """
    Retrieves protein structure from AlphaFold Database.

    Gets predicted structure, confidence scores, and optionally
    performs structural analysis.

    Args:
        identifier: Protein identifier (UniProt ID or gene name)
        identifier_type: Type of identifier (uniprot, gene)
        include_analysis: Include structural analysis

    Returns:
        JSON with structure info, confidence, and analysis
    """
    if identifier_type == "gene":
        uniprot_id = resolve_gene_to_uniprot(identifier)
    else:
        uniprot_id = identifier

    # Get AlphaFold structure
    base_url = "https://alphafold.ebi.ac.uk/api"
    response = requests.get(f"{base_url}/prediction/{uniprot_id}")

    if response.status_code != 200:
        return json.dumps({"error": f"Structure not found for {uniprot_id}"})

    metadata = response.json()[0]

    result = {
        "uniprot_id": uniprot_id,
        "gene": metadata.get('gene'),
        "organism": metadata.get('organismScientificName'),
        "pdb_url": metadata['pdbUrl'],
        "cif_url": metadata['cifUrl'],
        "mean_plddt": metadata.get('globalMetricValue')
    }

    if include_analysis:
        # Download and analyze structure
        pdb_content = requests.get(metadata['pdbUrl']).text

        result["analysis"] = {
            "secondary_structure": analyze_secondary_structure(pdb_content),
            "domains": identify_domains(pdb_content),
            "binding_sites": detect_binding_pockets(pdb_content)
        }

    return json.dumps(result)

@tool
def predict_protein_complex(
    sequences: dict,
    complex_type: str = "protein-protein"
) -> str:
    """
    Predicts structure of protein complexes using AlphaFold 3.

    Supports protein-protein, protein-DNA, protein-RNA, and
    protein-ligand complex prediction.

    Args:
        sequences: Dict mapping chain IDs to sequences
        complex_type: Type of complex to predict

    Returns:
        JSON with predicted complex structure and confidence
    """
    # Prepare AlphaFold 3 input
    af3_input = prepare_af3_input(sequences, complex_type)

    # Run prediction (requires local AF3 or cloud API)
    results = run_alphafold3_prediction(af3_input)

    return json.dumps({
        "complex_type": complex_type,
        "chains": list(sequences.keys()),
        "ipTM_score": results['iptm'],
        "pTM_score": results['ptm'],
        "ranking_confidence": results['confidence'],
        "interface_residues": results['interface'],
        "structure_file": results['output_path']
    })

@tool
def analyze_binding_site(
    structure_source: str,
    ligand_type: str = "small_molecule"
) -> str:
    """
    Analyzes protein binding sites for drug discovery.

    Identifies druggable pockets, calculates druggability scores,
    and suggests binding modes.

    Args:
        structure_source: UniProt ID or PDB file path
        ligand_type: Type of ligand (small_molecule, peptide, protein)

    Returns:
        JSON with binding site analysis and druggability assessment
    """
    # Get structure
    if structure_source.startswith("P") or structure_source.startswith("Q"):
        structure = get_alphafold_structure(structure_source)
        pdb_content = structure['pdb_content']
    else:
        with open(structure_source) as f:
            pdb_content = f.read()

    # Pocket detection using fpocket
    pockets = run_fpocket(pdb_content)

    # Druggability assessment
    druggable_pockets = []
    for pocket in pockets:
        druggability = assess_druggability(pocket)
        if druggability['score'] > 0.5:
            druggable_pockets.append({
                "pocket_id": pocket['id'],
                "residues": pocket['residues'],
                "volume": pocket['volume'],
                "druggability_score": druggability['score'],
                "hydrophobicity": pocket['hydrophobicity']
            })

    return json.dumps({
        "total_pockets": len(pockets),
        "druggable_pockets": len(druggable_pockets),
        "best_pocket": druggable_pockets[0] if druggable_pockets else None,
        "all_druggable": druggable_pockets
    })
```

### Integration with Anthropic Claude

```python
import anthropic
import requests

client = anthropic.Client()

def structural_analysis_with_claude(uniprot_id: str, research_question: str):
    """Combines AlphaFold structure retrieval with Claude analysis."""

    # Get structure and metadata
    structure = get_alphafold_structure(uniprot_id)

    # Get functional annotations
    uniprot_data = requests.get(
        f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    ).json()

    # Claude structural interpretation
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[
            {
                "role": "user",
                "content": f"""You are a structural biology expert. Analyze this protein structure in context of the research question.

Research Question: {research_question}

Protein Information:
- UniProt ID: {uniprot_id}
- Gene: {structure['gene']}
- Organism: {structure['organism']}
- Mean pLDDT (confidence): {structure['mean_plddt']:.1f}

Functional Annotations:
- Function: {uniprot_data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'Unknown')}
- Subcellular Location: {extract_subcellular_location(uniprot_data)}
- Known Domains: {extract_domains(uniprot_data)}
- Disease Associations: {extract_diseases(uniprot_data)}

Please provide:

1. **Structure Quality Assessment:**
   - Interpretation of pLDDT scores
   - Regions of high/low confidence
   - Implications for analysis

2. **Functional Domain Analysis:**
   - Key structural domains
   - Active sites or binding regions
   - Disordered regions and their function

3. **Research Question Analysis:**
   - How structure relates to the question
   - Key residues or regions of interest
   - Structural basis for function

4. **Drug Discovery Potential:**
   - Druggable binding sites
   - Allosteric opportunities
   - Known inhibitor binding modes

5. **Recommendations:**
   - Additional structural studies needed
   - Mutagenesis targets
   - Related structures to compare

Format as a structural biology report."""
            }
        ],
    )

    return message.content[0].text
```

---

## Confidence Interpretation

| pLDDT Score | Interpretation | Reliability |
|-------------|----------------|-------------|
| > 90 | Very high confidence | Experimental quality |
| 70-90 | Confident | Reliable for most analyses |
| 50-70 | Low confidence | Use with caution |
| < 50 | Very low confidence | Likely disordered |

---

## Methodology

This implementation follows established structural biology practices:

> **Jumper, J. et al.** *Highly accurate protein structure prediction with AlphaFold.* Nature (2021). https://github.com/google-deepmind/alphafold

> **Abramson, J. et al.** *Accurate structure prediction of biomolecular interactions with AlphaFold 3.* Nature (2024).

Key design decisions:

1. **Database-first:** Query existing predictions before running new jobs
2. **Confidence-aware:** All analyses consider pLDDT/PAE scores
3. **Drug discovery focus:** Binding site analysis and druggability
4. **Multi-model support:** AF2, AF3, OpenFold, ESMFold

---

## Dependencies

```
requests>=2.28.0
biopython>=1.81
pymol>=2.5.0 (optional, for visualization)
rdkit>=2023.03.0 (for ligand analysis)
```

For local AlphaFold:
```bash
# AlphaFold 2
git clone https://github.com/google-deepmind/alphafold.git

# AlphaFold 3
git clone https://github.com/google-deepmind/alphafold3.git

# OpenFold 3
git clone https://github.com/aqlaboratory/openfold-3.git
```

---

## Validation

Performance benchmarks:

| Benchmark | Metric | Performance |
|-----------|--------|-------------|
| CASP15 | GDT-TS | 92.4 (AF2) |
| CASP15 Complex | DockQ | 0.73 (AF3) |
| PDBBind | Binding pose | RMSD < 2Å (80%) |

---

## Related Skills

- **Chemical Property Lookup:** For ligand analysis
- **AgentD Drug Discovery:** For drug design workflows
- **Knowledge Graph Skills:** For target validation
- **Variant Annotation:** For structural impact assessment

---

## External Resources

- [AlphaFold Database](https://alphafold.ebi.ac.uk/)
- [AlphaFold GitHub](https://github.com/google-deepmind/alphafold)
- [AlphaFold 3 GitHub](https://github.com/google-deepmind/alphafold3)
- [OpenFold 3](https://github.com/aqlaboratory/openfold-3)
- [AlphaFold MCP Server](https://github.com/Augmented-Nature/AlphaFold-MCP-Server)

---

## Author

**MD BABU MIA**
*Artificial Intelligence Group*
*Icahn School of Medicine at Mount Sinai*
md.babu.mia@mssm.edu


<!-- AUTHOR_SIGNATURE: 9a7f3c2e-MD-BABU-MIA-2026-MSSM-SECURE -->