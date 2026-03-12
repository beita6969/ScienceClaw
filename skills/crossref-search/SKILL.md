---
name: crossref-search
description: "Academic metadata search via CrossRef API. Use when: user needs DOI resolution, citation counts, journal metadata, or publisher info. NOT for: full-text access or downloading papers."
metadata: { "openclaw": { "emoji": "🔗", "requires": { "bins": ["curl"] } } }
---

# CrossRef Search

Academic metadata search and DOI resolution via the public CrossRef REST API.

## When to Use

- Resolving a DOI to get full citation metadata
- Searching for papers by title, author, or keywords
- Looking up journal ISSN metadata or publisher info
- Finding citation counts and reference lists
- Retrieving funder information for grants/awards

## When NOT to Use

- Full-text access or downloading papers (use publisher sites)
- Preprint search (use arxiv-search)
- Biomedical literature (use pubmed-search)
- Author profile pages or h-index (use openalex-search)

## DOI Resolution

```bash
curl -s "https://api.crossref.org/works/10.1038/nature12373" | python3 -c "
import sys, json
data = json.load(sys.stdin)['message']
title = data.get('title', [''])[0]
authors = ', '.join(f\"{a.get('given','')} {a.get('family','')}\" for a in data.get('author', []))
journal = data.get('container-title', [''])[0]
cited = data.get('is-referenced-by-count', 0)
print(f'Title: {title}')
print(f'Authors: {authors}')
print(f'Journal: {journal} | Citations: {cited}')
"
```

## Works Search

```bash
# Search by query terms
curl -s "https://api.crossref.org/works?query=machine+learning+protein+folding&rows=5&mailto=user@example.com" | python3 -c "
import sys, json
data = json.load(sys.stdin)['message']
for item in data['items']:
    title = item.get('title', [''])[0]
    doi = item.get('DOI', '')
    cited = item.get('is-referenced-by-count', 0)
    print(f'{title}')
    print(f'  DOI: {doi} | Citations: {cited}')
"

# Filter by date, type, and sort by citations
curl -s "https://api.crossref.org/works?query=CRISPR&filter=from-pub-date:2023-01-01,type:journal-article&rows=10&sort=is-referenced-by-count&order=desc&mailto=user@example.com"

# Search by author
curl -s "https://api.crossref.org/works?query.author=Jennifer+Doudna&rows=10&sort=published&order=desc&mailto=user@example.com"
```

## Journal Lookup

```bash
# Search journals by title
curl -s "https://api.crossref.org/journals?query=nature+biotechnology&rows=5" | python3 -c "
import sys, json
for j in json.load(sys.stdin)['message']['items']:
    print(f\"{j['title']} (ISSN: {', '.join(j.get('ISSN', []))})\")
"

# Get journal metadata by ISSN
curl -s "https://api.crossref.org/journals/0028-0836"

# Recent works from a journal
curl -s "https://api.crossref.org/journals/0028-0836/works?rows=5&sort=published&order=desc"
```

## Funder Search

```bash
curl -s "https://api.crossref.org/funders?query=national+institutes+of+health&rows=5" | python3 -c "
import sys, json
for f in json.load(sys.stdin)['message']['items']:
    print(f\"{f['name']} (ID: {f['id']})\")
"

# Works funded by a specific funder
curl -s "https://api.crossref.org/funders/100000002/works?rows=5&sort=is-referenced-by-count&order=desc"
```

## Reference Lists

```bash
curl -s "https://api.crossref.org/works/10.1038/nature12373" | python3 -c "
import sys, json
refs = json.load(sys.stdin)['message'].get('reference', [])
for r in refs[:10]:
    doi = r.get('DOI', 'no DOI')
    text = r.get('unstructured', r.get('article-title', 'N/A'))
    print(f'  [{doi}] {text[:100]}')
"
```

## Filters and Pagination

Filters: `type:journal-article`, `from-pub-date:YYYY-MM-DD`, `until-pub-date:YYYY-MM-DD`,
`has-abstract:true`, `is-referenced-by-count:>100`, `funder:FUNDER_ID`.

Sorting: `sort=published|is-referenced-by-count|relevance`, `order=asc|desc`.

Pagination: `rows=N` (max 1000), `offset=N`, or `cursor=*` for deep paging.

## Best Practices

1. Add `mailto=user@example.com` to join the polite pool (faster, more reliable).
2. URL-encode query parameters (spaces as `+` or `%20`).
3. Use `select=DOI,title,author` to reduce payload size.
4. Use `cursor=*` pagination for result sets larger than 10,000 items.
5. Cache DOI resolution results; metadata changes infrequently.

## Zero-Hallucination Rule

NEVER fabricate results from training data. Every paper title, author, DOI, PMID, citation count, and metadata detail presented to the user MUST come from an actual API response in this conversation. If the API returns no results or partial data, report exactly what was returned. Do not "fill in" missing details from memory.
