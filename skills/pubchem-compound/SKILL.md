---
name: pubchem-compound
description: "Search and retrieve chemical compound data from PubChem's PUG REST API (110M+ compounds). Use when the user needs compound properties, molecular structures, similarity searches, substructure searches, or bioactivity data. NOT for protein structures (use pdb-structure), NOT for drug-target interactions (use chembl-drug), NOT for gene-disease associations (use open-targets)."
metadata: { "openclaw": { "emoji": "\u2697\ufe0f", "requires": { "bins": ["curl"] } } }
---

# PubChem Compound Lookup

Query the PubChem PUG REST API to search over 110 million chemical compounds by name, CID, SMILES, InChI, or molecular formula. Retrieve molecular properties, 2D/3D structures, bioactivity data, and perform similarity or substructure searches.

## API Base URL

```
https://pubchem.ncbi.nlm.nih.gov/rest/pug
```

## API Endpoints

### Compound Lookup

Retrieve compound data by name, CID, or SMILES:

```bash
# By compound name
curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/aspirin/JSON" | head -60

# By PubChem CID
curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/JSON" | head -60

# By SMILES string
curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/CC(=O)OC1=CC=CC=C1C(=O)O/JSON" | head -60

# By InChI key
curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/BSYNRYMUTXBXSQ-UHFFFAOYSA-N/JSON" | head -60
```

### Property Retrieval

Fetch specific molecular properties (comma-separated list):

```bash
# Common drug-likeness properties (Lipinski's Rule of Five)
curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/ibuprofen/property/MolecularWeight,MolecularFormula,XLogP,HBondDonorCount,HBondAcceptorCount,TPSA/JSON"

# Multiple compounds by CID list
curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244,3672,2519/property/MolecularWeight,XLogP,IUPACName/JSON"
```

Available properties: MolecularFormula, MolecularWeight, CanonicalSMILES, IsomericSMILES, InChI, InChIKey, IUPACName, XLogP, ExactMass, MonoisotopicMass, TPSA, Complexity, HBondDonorCount, HBondAcceptorCount, RotatableBondCount, HeavyAtomCount, AtomStereoCount, BondStereoCount, Volume3D.

### Similarity Search

Find structurally similar compounds using 2D fingerprint Tanimoto similarity:

```bash
# Find compounds with >=90% similarity to aspirin
curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/smiles/CC(=O)OC1=CC=CC=C1C(=O)O/cids/JSON?Threshold=90"

# Similarity search returning properties
curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/cid/2244/property/MolecularWeight,XLogP,CanonicalSMILES/JSON?Threshold=85&MaxRecords=10"
```

### Substructure Search

Find compounds containing a specific substructure:

```bash
# Find compounds containing a benzene ring with carboxylic acid
curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsubstructure/smiles/c1ccccc1C(=O)O/cids/JSON?MaxRecords=10"
```

### PUG-View (Detailed Annotations)

Retrieve detailed compound records including pharmacology, safety, and literature:

```bash
# Full PUG-View record
curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/2244/JSON" | head -100

# Specific section headings
curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/2244/JSON?heading=Pharmacology+and+Biochemistry" | head -80
```

### Bioactivity Data

Retrieve assay results and biological activity:

```bash
# Get bioassay summary for a compound
curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/assaysummary/JSON" | head -80

# Get assay results with activity outcome
curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/aids/JSON?aids_type=active"
```

## Common Queries

```bash
# Check if a compound exists and get its CID
curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/metformin/cids/JSON"

# Get canonical SMILES for a compound
curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/caffeine/property/CanonicalSMILES,IsomericSMILES/JSON"

# Get 2D structure image URL (PNG)
# https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/PNG

# Get SDF (structure-data file)
curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/aspirin/SDF"
```

## Best Practices

1. Rate limit requests to 5 per second maximum; PubChem may throttle or block excessive usage.
2. Use CID lookups when possible -- name lookups may return multiple matches.
3. For large result sets, use `MaxRecords` and pagination with `cids_type=standardized`.
4. URL-encode SMILES strings that contain special characters (e.g., `#`, `/`, `\`).
5. Prefer property endpoints over full JSON records to reduce response size.
6. Use PUG-View for curated annotations (pharmacology, safety, patents) rather than raw compound data.
7. For bulk operations (>100 compounds), use the PUG REST asynchronous listkey pattern.

## Data Integrity Rule

NEVER fabricate database results from training data. Every protein ID, gene name, compound property, pathway ID, structure detail, and metadata MUST come from an actual API response in this conversation. If the API returns no results, errors, or partial data, report exactly what happened. Do not "fill in" missing data from memory or make up identifiers.
