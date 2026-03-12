---
name: uniprot-protein
description: "Query the UniProt REST API for protein sequences, function annotations, structure info, and cross-references. Use when the user needs protein data, gene-to-protein mapping, functional annotation, or FASTA sequences. NOT for nucleotide sequences (use NCBI), 3D structure files (use PDB), or pathway data (use KEGG)."
metadata: { "openclaw": { "emoji": "🧬", "requires": { "bins": ["curl"] } } }
---

# UniProt Protein Database API

Access the Universal Protein Resource (UniProt) to search and retrieve protein entries,
sequences, functional annotations, and cross-references. No authentication required.

## API Endpoints

Base: `https://rest.uniprot.org`

### GET /uniprotkb/search -- Search protein entries
```bash
# Search for human insulin proteins (reviewed/Swiss-Prot only)
curl -s "https://rest.uniprot.org/uniprotkb/search?query=insulin+AND+organism_id:9606+AND+reviewed:true&format=json&size=5"

# Search by gene name exact match
curl -s "https://rest.uniprot.org/uniprotkb/search?query=gene_exact:TP53+AND+organism_id:9606&format=json"

# Search by EC number (enzyme classification)
curl -s "https://rest.uniprot.org/uniprotkb/search?query=ec:2.7.11.1+AND+reviewed:true&format=json&size=10"

# Search by Gene Ontology term
curl -s "https://rest.uniprot.org/uniprotkb/search?query=go:0006915+AND+organism_id:9606&format=json&size=10"
```

### GET /uniprotkb/{accession} -- Retrieve a single protein entry
```bash
# Get full entry for human p53 (JSON)
curl -s "https://rest.uniprot.org/uniprotkb/P04637?format=json"

# Get entry in TSV with selected fields
curl -s "https://rest.uniprot.org/uniprotkb/search?query=accession:P04637&format=tsv&fields=accession,gene_names,protein_name,organism_name,length,go_p"
```

### GET /uniprotkb/{accession}.fasta -- Download FASTA sequence
```bash
# Get FASTA sequence for human hemoglobin subunit alpha
curl -s "https://rest.uniprot.org/uniprotkb/P69905.fasta"
```

### GET /uniref/search -- Search UniProt Reference Clusters
```bash
# Find UniRef90 clusters for a protein
curl -s "https://rest.uniprot.org/uniref/search?query=uniprot_id:P04637&format=json&size=5"
```

### GET /uniparc/search -- Search UniProt Archive
```bash
# Search UniParc for cross-reference records
curl -s "https://rest.uniprot.org/uniparc/search?query=uniprotkb:P04637&format=json&size=5"
```

## Query Syntax

UniProt supports a rich query language for the `query` parameter:

| Field             | Example                              | Description                        |
|-------------------|--------------------------------------|------------------------------------|
| `gene_exact`      | `gene_exact:BRCA1`                   | Exact gene name match              |
| `organism_id`     | `organism_id:9606`                   | NCBI taxonomy ID (9606 = human)    |
| `ec`              | `ec:3.4.21.5`                        | Enzyme Commission number           |
| `go`              | `go:0006915`                         | Gene Ontology term ID              |
| `keyword`         | `keyword:Phosphoprotein`             | UniProt keyword                    |
| `reviewed`        | `reviewed:true`                      | Swiss-Prot (reviewed) entries only |
| `length`          | `length:[100 TO 500]`               | Sequence length range              |
| `structure_3d`    | `structure_3d:true`                  | Has 3D structure                   |
| `accession`       | `accession:P04637`                   | UniProt accession                  |

Combine with `AND`, `OR`, `NOT`. Use `+` for spaces in URL encoding.

## Common Queries

```bash
# All reviewed human kinases
curl -s "https://rest.uniprot.org/uniprotkb/search?query=keyword:Kinase+AND+organism_id:9606+AND+reviewed:true&format=json&size=25"

# Proteins with disease association
curl -s "https://rest.uniprot.org/uniprotkb/search?query=keyword:Disease+AND+gene_exact:CFTR&format=json"

# Batch retrieve multiple accessions (TSV)
curl -s "https://rest.uniprot.org/uniprotkb/search?query=accession:P04637+OR+accession:P69905+OR+accession:P00533&format=tsv&fields=accession,gene_names,protein_name,length"

# Paginate results using cursor (check Link header for next page)
curl -sI "https://rest.uniprot.org/uniprotkb/search?query=organism_id:9606+AND+reviewed:true&size=25" | grep -i link
```

## Best Practices

1. Always add `reviewed:true` when searching Swiss-Prot (curated) entries to avoid TrEMBL noise.
2. Use `format=tsv&fields=...` for tabular output when you only need specific fields.
3. Use `format=json` for programmatic parsing of full entry data.
4. Paginate with `size` parameter (max 500) and cursor-based pagination via the `Link` header.
5. No API key is needed, but be respectful with rate limits -- avoid rapid-fire batch requests.
6. Common organism IDs: human=9606, mouse=10090, rat=10116, E.coli=83333, yeast=559292.
7. Use `.fasta` suffix for quick sequence retrieval without parsing JSON.

## Data Integrity Rule

NEVER fabricate database results from training data. Every protein ID, gene name, compound property, pathway ID, structure detail, and metadata MUST come from an actual API response in this conversation. If the API returns no results, errors, or partial data, report exactly what happened. Do not "fill in" missing data from memory or make up identifiers.
