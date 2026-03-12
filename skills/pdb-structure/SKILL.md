---
name: pdb-structure
description: "Query the RCSB PDB API for protein 3D structures, experimental metadata, and structure files. Use when the user needs crystal or cryo-EM structure data, PDB entries, resolution info, or structure file downloads. NOT for protein sequences/annotations (use UniProt), gene data (use NCBI), or pathway info (use KEGG)."
metadata: { "openclaw": { "emoji": "🧊", "requires": { "bins": ["curl"] } } }
---

# RCSB Protein Data Bank (PDB) API

Access the RCSB PDB to search, retrieve, and download macromolecular 3D structures.
Covers X-ray crystallography, cryo-EM, NMR, and other experimental methods. No authentication required.

## API Endpoints

Data API Base: `https://data.rcsb.org`
Search API Base: `https://search.rcsb.org`
File Downloads: `https://files.rcsb.org`

### GET /rest/v1/core/entry/{pdb_id} -- Structure metadata
```bash
# Get full metadata for a PDB entry (human hemoglobin)
curl -s "https://data.rcsb.org/rest/v1/core/entry/1HBB"

# Get entry metadata for SARS-CoV-2 spike protein structure
curl -s "https://data.rcsb.org/rest/v1/core/entry/6VYB"
```

### POST /rcsbsearch/v2/query -- Advanced search API
```bash
# Search by text (protein name)
curl -s -X POST "https://search.rcsb.org/rcsbsearch/v2/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": {
      "type": "terminal",
      "service": "full_text",
      "parameters": { "value": "insulin receptor" }
    },
    "return_type": "entry",
    "request_options": { "paginate": { "start": 0, "rows": 10 } }
  }'

# Search by organism and resolution
curl -s -X POST "https://search.rcsb.org/rcsbsearch/v2/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": {
      "type": "group",
      "logical_operator": "and",
      "nodes": [
        {
          "type": "terminal",
          "service": "text",
          "parameters": {
            "attribute": "rcsb_entity_source_organism.ncbi_scientific_name",
            "operator": "exact_match",
            "value": "Homo sapiens"
          }
        },
        {
          "type": "terminal",
          "service": "text",
          "parameters": {
            "attribute": "rcsb_entry_info.resolution_combined",
            "operator": "less",
            "value": 2.0
          }
        }
      ]
    },
    "return_type": "entry",
    "request_options": { "paginate": { "start": 0, "rows": 10 } }
  }'

# Search by sequence similarity (BLAST-like)
curl -s -X POST "https://search.rcsb.org/rcsbsearch/v2/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": {
      "type": "terminal",
      "service": "sequence",
      "parameters": {
        "evalue_cutoff": 0.001,
        "identity_cutoff": 0.9,
        "sequence_type": "protein",
        "value": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
      }
    },
    "return_type": "polymer_entity",
    "request_options": { "paginate": { "start": 0, "rows": 10 } }
  }'
```

### File Downloads -- Structure coordinate files
```bash
# Download PDB format file
curl -s "https://files.rcsb.org/download/1HBB.pdb" -o 1HBB.pdb

# Download mmCIF format file
curl -s "https://files.rcsb.org/download/1HBB.cif" -o 1HBB.cif

# Download structure factors (X-ray data)
curl -s "https://files.rcsb.org/download/1HBB-sf.cif" -o 1HBB-sf.cif
```

### GraphQL API -- Flexible data queries
```bash
# Query specific fields via GraphQL
curl -s -X POST "https://data.rcsb.org/graphql" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "{ entry(entry_id: \"1HBB\") { rcsb_entry_info { resolution_combined experimental_method } struct { title } rcsb_accession_info { deposit_date } } }"
  }'
```

## Best Practices

1. Use the search API (POST) for complex queries; use the data API (GET) for known PDB IDs.
2. Always check `resolution_combined` to assess structure quality -- lower is better.
3. Use mmCIF format (`.cif`) over legacy PDB format for modern structures with large assemblies.
4. Sequence search is useful for finding structures of homologous proteins.
5. No authentication is required, but keep request volume reasonable.
6. PDB IDs are 4 characters (e.g., 1HBB). New extended IDs (PDB-xxxxx) are also supported.
7. Use GraphQL for retrieving only the specific fields you need.

## Data Integrity Rule

NEVER fabricate database results from training data. Every protein ID, gene name, compound property, pathway ID, structure detail, and metadata MUST come from an actual API response in this conversation. If the API returns no results, errors, or partial data, report exactly what happened. Do not "fill in" missing data from memory or make up identifiers.
