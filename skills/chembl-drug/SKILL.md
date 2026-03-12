---
name: chembl-drug
description: "Query the ChEMBL REST API for drug-target interactions, bioactivity data, ADMET properties, and approved drug information. Use when the user needs drug mechanism of action, binding affinity data, target information, or pharmacokinetic properties. NOT for basic compound lookup (use pubchem-compound), NOT for gene-disease associations (use open-targets), NOT for protein 3D structures (use pdb-structure)."
metadata: { "openclaw": { "emoji": "\ud83d\udc8a", "requires": { "bins": ["curl"] } } }
---

# ChEMBL Drug & Bioactivity Lookup

Query the ChEMBL REST API to access curated drug-target interaction data, bioactivity measurements, drug mechanisms of action, and ADMET properties from the European Bioinformatics Institute.

## API Base URL

```
https://www.ebi.ac.uk/chembl/api/data
```

All endpoints accept `.json` suffix and return JSON by default. Use `format=json` as a query parameter alternatively.

## API Endpoints

### Molecule Lookup

Retrieve molecule details by ChEMBL ID or search by name:

```bash
# Get molecule by ChEMBL ID
curl -s "https://www.ebi.ac.uk/chembl/api/data/molecule/CHEMBL25.json" | head -80

# Search molecules by name
curl -s "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q=imatinib" | head -80

# Get molecule by canonical SMILES
curl -s "https://www.ebi.ac.uk/chembl/api/data/molecule.json?molecule_structures__canonical_smiles=CC(=O)Oc1ccccc1C(=O)O" | head -60
```

### Target Lookup

Retrieve drug target information:

```bash
# Get target by ChEMBL ID
curl -s "https://www.ebi.ac.uk/chembl/api/data/target/CHEMBL2034.json" | head -60

# Search targets by gene name
curl -s "https://www.ebi.ac.uk/chembl/api/data/target/search.json?q=EGFR" | head -80

# Get target by UniProt accession
curl -s "https://www.ebi.ac.uk/chembl/api/data/target.json?target_components__accession=P00533" | head -60
```

### Bioactivity Data

Retrieve binding affinity, IC50, Ki, and other activity measurements:

```bash
# Get activities for a molecule (with pagination)
curl -s "https://www.ebi.ac.uk/chembl/api/data/activity.json?molecule_chembl_id=CHEMBL25&limit=20" | head -100

# Get activities for a specific target
curl -s "https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id=CHEMBL2034&limit=20" | head -100

# Filter by activity type (IC50, Ki, Kd, EC50)
curl -s "https://www.ebi.ac.uk/chembl/api/data/activity.json?molecule_chembl_id=CHEMBL941&standard_type=IC50&limit=10" | head -80

# Filter by potency threshold (pChEMBL value >= 6, i.e., activity <= 1 uM)
curl -s "https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id=CHEMBL2034&pchembl_value__gte=6&limit=20" | head -80
```

### Drug Mechanisms of Action

```bash
# Get mechanism of action for a drug
curl -s "https://www.ebi.ac.uk/chembl/api/data/mechanism.json?molecule_chembl_id=CHEMBL941" | head -60

# Get all mechanisms for a target
curl -s "https://www.ebi.ac.uk/chembl/api/data/mechanism.json?target_chembl_id=CHEMBL2034" | head -80
```

### Approved Drugs

Filter for approved drugs and clinical candidates:

```bash
# Get approved drugs only (max_phase = 4)
curl -s "https://www.ebi.ac.uk/chembl/api/data/molecule.json?max_phase=4&limit=20" | head -80

# Approved drugs for a specific target
curl -s "https://www.ebi.ac.uk/chembl/api/data/mechanism.json?target_chembl_id=CHEMBL2034" | head -60

# Filter by molecule type (small molecule, antibody, etc.)
curl -s "https://www.ebi.ac.uk/chembl/api/data/molecule.json?max_phase=4&molecule_type=Small%20molecule&limit=20" | head -60
```

### ADMET and Drug Properties

```bash
# Get computed molecular properties (Lipinski, PSA, ALogP are in molecule_properties)
curl -s "https://www.ebi.ac.uk/chembl/api/data/molecule/CHEMBL25.json" | python3 -c "import sys,json; [print(f'{k}: {v}') for k,v in json.load(sys.stdin).get('molecule_properties',{}).items()]"

# Get drug indications
curl -s "https://www.ebi.ac.uk/chembl/api/data/drug_indication.json?molecule_chembl_id=CHEMBL941&limit=10" | head -60
```

## Common Queries

```bash
# Find all drugs targeting a specific protein
curl -s "https://www.ebi.ac.uk/chembl/api/data/mechanism.json?target_chembl_id=CHEMBL1862" | head -80

# Get the ChEMBL ID for a drug by name
curl -s "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q=metformin&limit=5" | head -40

# Get molecule image URL: https://www.ebi.ac.uk/chembl/api/data/image/CHEMBL25.svg
```

## Best Practices

1. Always include `limit` parameter to control result size; default may return thousands of records.
2. Use `offset` with `limit` for pagination through large result sets.
3. Filter bioactivity by `pchembl_value__gte=5` (10 uM) or `pchembl_value__gte=6` (1 uM) for meaningful hits.
4. Use `max_phase` to filter clinical status: 4 = approved, 3 = Phase III, 2 = Phase II, 1 = Phase I.
5. Prefer ChEMBL IDs over name searches for precise lookups; name searches are fuzzy.
6. Parse `molecule_properties` for pre-computed Lipinski descriptors, PSA, and ALogP.
7. Rate limit to 1 request per second to avoid throttling from EBI servers.

## Data Integrity Rule

NEVER fabricate database results from training data. Every protein ID, gene name, compound property, pathway ID, structure detail, and metadata MUST come from an actual API response in this conversation. If the API returns no results, errors, or partial data, report exactly what happened. Do not "fill in" missing data from memory or make up identifiers.
