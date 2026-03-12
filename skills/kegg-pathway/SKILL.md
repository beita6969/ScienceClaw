---
name: kegg-pathway
description: "Query the KEGG REST API for metabolic pathways, genes, compounds, drugs, and diseases. Use when the user needs pathway mapping, gene-to-pathway links, compound info, or cross-reference ID conversion. NOT for protein sequences (use UniProt), 3D structures (use PDB), or variant/SNP data (use NCBI)."
metadata: { "openclaw": { "emoji": "🗺️", "requires": { "bins": ["curl"] } } }
---

# KEGG Pathway Database REST API

Access the Kyoto Encyclopedia of Genes and Genomes (KEGG) for pathway maps, gene
annotations, compound data, drug info, and disease records. No authentication required.

## API Endpoints

Base: `https://rest.kegg.jp`

### GET /list/{database} -- List all entries in a database
```bash
# List all human pathways
curl -s "https://rest.kegg.jp/list/pathway/hsa"

# List all KEGG organisms
curl -s "https://rest.kegg.jp/list/organism"
```

### GET /find/{database}/{query} -- Search by keyword
```bash
# Search pathways for "apoptosis"
curl -s "https://rest.kegg.jp/find/pathway/apoptosis"

# Search genes for BRCA1 across all organisms
curl -s "https://rest.kegg.jp/find/genes/brca1"

# Search compounds by name
curl -s "https://rest.kegg.jp/find/compound/glucose"

# Search diseases by keyword
curl -s "https://rest.kegg.jp/find/disease/diabetes"
```

### GET /get/{entry} -- Retrieve full entry details
```bash
# Get a pathway entry (human apoptosis pathway)
curl -s "https://rest.kegg.jp/get/hsa04210"

# Get a human gene entry
curl -s "https://rest.kegg.jp/get/hsa:7157"

# Get a compound entry (glucose)
curl -s "https://rest.kegg.jp/get/C00031"

# Get a KEGG Orthology entry
curl -s "https://rest.kegg.jp/get/K00001"

# Get pathway map as image
curl -s "https://rest.kegg.jp/get/hsa04210/image" -o apoptosis.png
```

### GET /link/{target}/{source} -- Cross-reference between databases
```bash
# Find all genes in a pathway
curl -s "https://rest.kegg.jp/link/hsa/hsa04210"

# Find pathways associated with a gene
curl -s "https://rest.kegg.jp/link/pathway/hsa:7157"

# Find compounds in a pathway
curl -s "https://rest.kegg.jp/link/compound/hsa00010"

# Find diseases linked to a gene
curl -s "https://rest.kegg.jp/link/disease/hsa:672"

# Link KEGG Orthology to pathways
curl -s "https://rest.kegg.jp/link/pathway/ko:K00001"
```

### GET /conv/{target}/{source} -- Convert between ID systems
```bash
# Convert KEGG gene IDs to NCBI Gene IDs
curl -s "https://rest.kegg.jp/conv/ncbi-geneid/hsa:7157"

# Convert NCBI Gene IDs to KEGG
curl -s "https://rest.kegg.jp/conv/hsa/ncbi-geneid:7157"

# Convert KEGG compound to PubChem
curl -s "https://rest.kegg.jp/conv/pubchem/compound:C00031"

# Convert UniProt to KEGG gene IDs
curl -s "https://rest.kegg.jp/conv/hsa/uniprot:P04637"
```

## Supported Databases

| Database   | Code        | Description                            |
|-----------|-------------|----------------------------------------|
| Pathway   | `pathway`   | Metabolic and signaling pathway maps   |
| Module    | `module`    | Functional units within pathways       |
| KO        | `ko`        | KEGG Orthology (functional orthologs)  |
| Genome    | `genome`    | Organism genomes                       |
| Genes     | `genes`     | Gene entries per organism (e.g., hsa)  |
| Compound  | `compound`  | Small molecules and metabolites        |
| Drug      | `drug`      | Drug and pharmaceutical entries        |
| Disease   | `disease`   | Human disease entries                  |

## Common Patterns

```bash
# Full workflow: find pathways for a gene, then get pathway details
curl -s "https://rest.kegg.jp/link/pathway/hsa:7157"   # Step 1: find pathways
curl -s "https://rest.kegg.jp/get/hsa04115"             # Step 2: get details

# Map between KEGG and external IDs for batch processing
curl -s "https://rest.kegg.jp/conv/ncbi-geneid/hsa"
```

## Organism Codes

Common codes: `hsa` (human), `mmu` (mouse), `rno` (rat), `dme` (fly), `sce` (yeast), `eco` (E. coli).

## Best Practices

1. KEGG returns tab-separated plain text by default -- parse with `cut`, `awk`, or similar.
2. Use `/link` for mapping between databases and `/conv` for external ID conversion.
3. Prefix gene IDs with the organism code (e.g., `hsa:7157` for human TP53).
4. Pathway IDs use organism prefix + number (e.g., `hsa04210`); use `map04210` for reference.
5. No authentication is required, but KEGG limits heavy automated access -- keep requests reasonable.
6. Download pathway images with `/get/{pathway_id}/image` for visual reference.
7. Use `/find` for keyword search and `/list` to enumerate all entries in a database.

## Data Integrity Rule

NEVER fabricate database results from training data. Every protein ID, gene name, compound property, pathway ID, structure detail, and metadata MUST come from an actual API response in this conversation. If the API returns no results, errors, or partial data, report exactly what happened. Do not "fill in" missing data from memory or make up identifiers.
