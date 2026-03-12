---
name: semantic-scholar
description: "Search Semantic Scholar for academic papers across all disciplines with citation graph analysis. Use when: (1) finding papers across disciplines, (2) citation/reference graph traversal, (3) author search, (4) paper recommendations, (5) large-scale bibliometric analysis. NOT for: specific database access (use pubmed for biomedical, arxiv for preprints)."
metadata: { "openclaw": { "emoji": "🎓", "requires": { "bins": ["curl"] }, "primaryEnv": "S2_API_KEY" } }
---

# Semantic Scholar Search

Search 200M+ papers across all disciplines with citation graph, author profiles,
and recommendations via the Semantic Scholar API.

## Authentication

```bash
curl -s -H "x-api-key: ${S2_API_KEY}" "https://api.semanticscholar.org/graph/v1/paper/search?query=example"
```

## Paper Search

```bash
curl -s "https://api.semanticscholar.org/graph/v1/paper/search?query=machine+learning+drug+discovery&limit=5&fields=title,authors,year,abstract,citationCount,url"
```

Parameters: `query=` (required), `limit=` (max 100), `offset=`, `fields=`,
`year=` (e.g. `2020-2024`), `fieldsOfStudy=` (e.g. `Computer Science`, `Medicine`),
`openAccessPdf`, `minCitationCount=`.

**Bulk search** (up to 10M results, token-based pagination):
```bash
curl -s "https://api.semanticscholar.org/graph/v1/paper/search/bulk?query=CRISPR+gene+editing&fields=title,year,citationCount"
```

## Paper Lookup by Identifier

```bash
# DOI
curl -s "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1038/s41586-021-03819-2?fields=title,abstract,authors,year,citationCount"
# arXiv ID
curl -s "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2301.07041?fields=title,abstract,year"
# PubMed ID
curl -s "https://api.semanticscholar.org/graph/v1/paper/PMID:34265844?fields=title,abstract,year"
```

## Citation Graph

```bash
# Papers that cite a given paper
curl -s "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2301.07041/citations?fields=title,year,citationCount&limit=20"
# Papers cited by a given paper
curl -s "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2301.07041/references?fields=title,year,citationCount&limit=20"
```

## Author Search

```bash
# Search by name
curl -s "https://api.semanticscholar.org/graph/v1/author/search?query=yann+lecun&fields=name,hIndex,paperCount,citationCount"
# Author papers
curl -s "https://api.semanticscholar.org/graph/v1/author/1688681/papers?fields=title,year,citationCount&limit=20"
```

## Recommendations

```bash
# Single-paper
curl -s "https://api.semanticscholar.org/recommendations/v1/papers/forpaper/ARXIV:2301.07041?fields=title,year,citationCount&limit=10"
# Multi-paper (POST)
curl -s -X POST "https://api.semanticscholar.org/recommendations/v1/papers/" \
  -H "Content-Type: application/json" \
  -d '{"positivePaperIds": ["ARXIV:2301.07041", "ARXIV:2302.13971"], "negativePaperIds": []}'
```

## Available Fields

**Paper**: `paperId`, `title`, `abstract`, `year`, `venue`, `citationCount`,
`referenceCount`, `influentialCitationCount`, `isOpenAccess`, `openAccessPdf`,
`fieldsOfStudy`, `publicationTypes`, `publicationDate`, `journal`, `url`,
`externalIds`, `tldr`

**Nested**: `authors.name`, `authors.authorId`, `citations.title`, `references.title`

## Rate Limits

- Without API key: 100 requests per 5 minutes
- With API key: 1/sec sustained, 10/sec burst
- Register free at: https://www.semanticscholar.org/product/api#api-key
- HTTP 429 on rate limit; back off and retry

## Ready-to-Use Query Templates

### Find recent high-impact papers on a topic
```bash
curl -s "https://api.semanticscholar.org/graph/v1/paper/search?\
query=YOUR+TOPIC&limit=10&year=2023-2025&minCitationCount=10&\
fields=title,authors,year,citationCount,influentialCitationCount,tldr,venue,isOpenAccess,openAccessPdf,externalIds" \
| python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'Found {data[\"total\"]} papers')
for i, p in enumerate(data['data']):
    authors = ', '.join(a['name'] for a in (p.get('authors') or [])[:3])
    tldr = (p.get('tldr') or {}).get('text', 'N/A')[:120]
    doi = (p.get('externalIds') or {}).get('DOI', 'N/A')
    print(f'[{i+1}] {p[\"title\"]}')
    print(f'    {authors} ({p.get(\"year\")}) {p.get(\"venue\",\"\")}  Cited:{p[\"citationCount\"]}')
    print(f'    TLDR: {tldr}')
    print(f'    DOI: {doi}')
    print()
"
```

### Build citation graph for a paper
```bash
PAPER_ID="DOI:10.1038/s41586-021-03819-2"  # or ARXIV:2301.07041 or S2 ID
# Who cites this?
curl -s "https://api.semanticscholar.org/graph/v1/paper/${PAPER_ID}/citations?\
fields=title,year,citationCount,venue&limit=20" | python3 -c "
import sys,json; data=json.load(sys.stdin)
print(f'{len(data[\"data\"])} citing papers:')
for c in sorted(data['data'], key=lambda x: x['citingPaper'].get('citationCount',0), reverse=True)[:10]:
    p=c['citingPaper']
    print(f'  [{p.get(\"year\",\"?\")}] {p[\"title\"][:80]} (cited:{p.get(\"citationCount\",0)})')
"
```

### Find open access PDF for a paper
```bash
curl -s "https://api.semanticscholar.org/graph/v1/paper/DOI:10.xxxx/xxxxx?\
fields=title,isOpenAccess,openAccessPdf" | python3 -c "
import sys,json; p=json.load(sys.stdin)
pdf = p.get('openAccessPdf',{})
if pdf: print(f'OA PDF: {pdf[\"url\"]}')
else: print('No open access PDF available')
"
```

## Best Practices

1. Always specify `fields=` to reduce response size.
2. Use `minCitationCount` to filter low-impact results.
3. Use external IDs (DOI, ARXIV, PMID) for precise lookups.
4. Chain citation/reference lookups to build citation graphs.
5. Use `tldr` field for AI-generated paper summaries.
6. Use `influentialCitationCount` to identify high-impact papers.
7. **Prefer Semantic Scholar over CrossRef** for paper discovery searches.
8. Combine with OpenAlex for broader coverage and OA link discovery.
9. **NEVER fabricate results.** Every paper detail must come from an actual API response. If the API returns no results, report that honestly.
10. **Self-check before responding:** Did this paper title, DOI, author list, and citation count ALL come from an S2 API call in this conversation?
