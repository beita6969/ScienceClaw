---
name: ncbi-entrez
description: "Query NCBI E-utilities for GenBank sequences, gene info, SNPs, ClinVar variants, and literature links. Use when the user needs nucleotide/protein sequences from GenBank, gene summaries, variant data, or cross-database links. NOT for protein annotations (use UniProt), 3D structures (use PDB), or pathway mapping (use KEGG)."
metadata: { "openclaw": { "emoji": "🔬", "requires": { "bins": ["curl"] } } }
---

# NCBI Entrez E-utilities API

Access NCBI databases (Gene, SNP, ClinVar, Nucleotide, Protein, OMIM) through the
Entrez Programming Utilities. Supports search, fetch, linking, and summary operations.

## API Endpoints

Base: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils`

### Authentication & Rate Limits

Set the `NCBI_API_KEY` environment variable for higher throughput.
- **With API key**: 10 requests/second
- **Without API key**: 3 requests/second

Append `&api_key=$NCBI_API_KEY` to all requests when available.

### esearch.fcgi -- Search a database and return IDs
```bash
# Search for TP53 gene in human
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gene&term=TP53[Gene]+AND+Homo+sapiens[Organism]&retmode=json"

# Search ClinVar for BRCA1 pathogenic variants
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=clinvar&term=BRCA1[gene]+AND+pathogenic[clinical_significance]&retmode=json&retmax=20"

# Search nucleotide database
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=nucleotide&term=SARS-CoV-2[Organism]+AND+complete+genome&retmode=json&retmax=5"
```

### efetch.fcgi -- Retrieve full records by ID
```bash
# Fetch gene record for TP53 (Gene ID: 7157)
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=gene&id=7157&retmode=xml"

# Fetch nucleotide sequence in FASTA format
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nucleotide&id=NM_000546.6&rettype=fasta&retmode=text"

# Fetch SNP record
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=snp&id=rs1042522&retmode=json"

# Fetch ClinVar record in XML
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=clinvar&id=37653&rettype=clinvarset&retmode=xml"
```

### esummary.fcgi -- Retrieve document summaries
```bash
# Get gene summary for TP53
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gene&id=7157&retmode=json"

# Get summaries for multiple SNPs
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=snp&id=rs1042522,rs28897696&retmode=json"
```

### elink.fcgi -- Cross-database linking
```bash
# Find SNPs linked to a gene
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=gene&db=snp&id=7157&retmode=json"

# Link gene to ClinVar entries
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=gene&db=clinvar&id=672&retmode=json"

# Find PubMed articles linked to a gene
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=gene&db=pubmed&id=7157&retmode=json"
```

### einfo.fcgi -- List available databases and fields
```bash
# List all Entrez databases
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi?retmode=json"

# Get searchable fields for the gene database
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi?db=gene&retmode=json"
```

## Supported Databases

| Database     | db= value     | Description                           |
|-------------|---------------|---------------------------------------|
| Gene        | `gene`        | Gene records with summaries           |
| SNP         | `snp`         | Single nucleotide polymorphisms       |
| ClinVar     | `clinvar`     | Clinical variant interpretations      |
| Nucleotide  | `nucleotide`  | GenBank nucleotide sequences          |
| Protein     | `protein`     | GenBank protein sequences             |
| OMIM        | `omim`        | Mendelian inheritance records         |
| PubMed      | `pubmed`      | Biomedical literature                 |

## Common Patterns

```bash
# Two-step search + fetch workflow
# Step 1: Search for IDs
IDS=$(curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gene&term=insulin[Gene]+AND+human[Organism]&retmode=json" | python3 -c "import sys,json; print(','.join(json.load(sys.stdin)['esearchresult']['idlist']))")

# Step 2: Fetch summaries for those IDs
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gene&id=$IDS&retmode=json"

# Search with date range
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=clinvar&term=BRCA1[gene]&datetype=pdat&mindate=2024/01/01&maxdate=2025/12/31&retmode=json"
```

## Best Practices

1. Always include `&api_key=$NCBI_API_KEY` when the env var is set for 10 req/s throughput.
2. Use `retmode=json` for easier parsing; fall back to `retmode=xml` when JSON is unavailable.
3. Use esearch + efetch two-step pattern for bulk retrieval workflows.
4. Respect rate limits strictly -- NCBI will block IPs that exceed them.
5. Use `retmax` to limit result counts (default is 20, max is 10000 for esearch).
6. Use elink to discover cross-database relationships (gene to SNP, gene to literature).
7. Include `[Organism]` qualifiers in search terms for species-specific results.

## Data Integrity Rule

NEVER fabricate database results from training data. Every protein ID, gene name, compound property, pathway ID, structure detail, and metadata MUST come from an actual API response in this conversation. If the API returns no results, errors, or partial data, report exactly what happened. Do not "fill in" missing data from memory or make up identifiers.
