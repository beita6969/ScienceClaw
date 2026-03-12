---
name: pubmed-search
description: "Search PubMed/MEDLINE for biomedical literature via NCBI E-utilities API. Use when: (1) searching medical/biomedical papers, (2) finding clinical studies, (3) querying with MeSH terms, (4) retrieving abstracts by PMID. NOT for: non-biomedical papers (use arxiv-search or semantic-scholar), full-text access (PubMed provides abstracts), or social science literature."
metadata: { "openclaw": { "emoji": "🏥", "requires": { "bins": ["curl"] } } }
---

# PubMed Search

Search PubMed/MEDLINE (36M+ citations) via NCBI E-utilities REST API.

## API Endpoints

Base: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`

### esearch -- Search and get PMIDs

```bash
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=COVID-19+vaccine+efficacy&retmax=10&retmode=json"
```

Parameters: `db=pubmed`, `term=` (URL-encoded query), `retmax=` (default 20, max 10000),
`retstart=` (pagination), `retmode=json`, `sort=relevance|pub_date`,
`mindate=`/`maxdate=` (YYYY/MM/DD), `datetype=pdat`.

### efetch -- Retrieve records by PMID

```bash
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=39142890,39088712&retmode=xml&rettype=abstract"
```

### einfo -- Database metadata

```bash
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi?db=pubmed&retmode=json"
```

## Query Syntax

**Boolean**: `AND`, `OR`, `NOT`, parentheses for grouping.

**Field tags**: `[ti]` title, `[tiab]` title/abstract, `[au]` author, `[mesh]` MeSH heading,
`[majr]` MeSH major topic, `[pt]` publication type, `[dp]` date, `[la]` language, `[jour]` journal.

**MeSH terms**: Standardized vocabulary with automatic explosion to narrower terms.
Use `[mesh:noexp]` for exact heading only. Qualifiers: `/therapy`, `/diagnosis`,
`/epidemiology`, `/genetics`, `/prevention and control`.

Example: `"Breast Neoplasms"[mesh] AND "Drug Therapy"[mesh]`

## Rate Limiting

- Without API key: 3 requests/sec
- With `NCBI_API_KEY`: 10 requests/sec (append `&api_key=${NCBI_API_KEY}`)
- Register at: https://www.ncbi.nlm.nih.gov/account/settings/

## Two-Step Search Pattern

```bash
PMIDS=$(curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=QUERY&retmax=5&retmode=json" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(','.join(d['esearchresult']['idlist']))")
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=${PMIDS}&retmode=xml&rettype=abstract"
```

## Specialty Search Patterns

- **Oncology**: `"Neoplasms"[mesh] AND "immunotherapy"[tiab] AND "Clinical Trial"[pt]`
- **Cardiology**: `"Cardiovascular Diseases"[mesh] AND "meta-analysis"[pt]`
- **Neurology**: `"Alzheimer Disease"[mesh] AND "biomarkers"[tiab] AND 2020:2024[dp]`
- **Infectious Disease**: `"Anti-Bacterial Agents"[mesh] AND "Drug Resistance"[mesh]`
- **Genetics**: `"Genome-Wide Association Study"[pt] AND "Diabetes Mellitus, Type 2"[mesh]`
- **Systematic Reviews**: `"systematic review"[ti] AND "Randomized Controlled Trial"[pt]`

## Best Practices

1. Start broad, narrow with MeSH terms and field tags.
2. Use `retmax=5` for exploration, increase for comprehensive searches.
3. URL-encode queries (spaces as `+`, quotes as `%22`).
4. Check `count` in esearch results before paginating.
5. Document exact query strings for systematic review reproducibility.
6. For systematic reviews, use `"systematic review"[ti]` combined with study type filters.

## Zero-Hallucination Rule

NEVER fabricate results from training data. Every paper title, author, DOI, PMID, citation count, and metadata detail presented to the user MUST come from an actual API response in this conversation. If the API returns no results or partial data, report exactly what was returned. Do not "fill in" missing details from memory.
