---
name: openalex-search
description: "Open academic metadata via OpenAlex API. Use when: user needs author profiles, institution data, concept mapping, or open citation data. NOT for: full-text search or downloading papers."
metadata: { "openclaw": { "emoji": "📚", "requires": { "bins": ["curl"] } } }
---

# OpenAlex Search Skill

Query the OpenAlex open academic metadata API for works, authors, institutions, and concepts.

## When to Use

- "Find papers by this author"
- "What institution produces the most research in X?"
- "Show citation data for this paper"
- "Map concepts related to this topic"
- "Find open access papers on..."

## When NOT to Use

- Full-text paper access (use semantic-scholar or direct PDF)
- Biomedical-specific search (use pubmed-search)
- Pre-print focused search (use arxiv-search)
- DOI resolution only (use crossref-search)

## Setup

No API key required. Add `mailto` parameter for polite pool (10x faster rate limit).

## API Base

```
https://api.openalex.org
```

## Common Commands

### Search Works

```bash
curl -s "https://api.openalex.org/works?search=CRISPR+gene+editing&per_page=5&mailto=user@example.com"

# Filter by year + open access
curl -s "https://api.openalex.org/works?filter=publication_year:2024,open_access.is_oa:true&search=quantum+computing&per_page=10"

# Get specific work by DOI
curl -s "https://api.openalex.org/works/doi:10.1038/s41586-024-07000-0"
```

### Search Authors

```bash
curl -s "https://api.openalex.org/authors?search=Yoshua+Bengio&per_page=5"

# Get author profile by ID
curl -s "https://api.openalex.org/authors/A5023888391"
```

### Search Institutions

```bash
curl -s "https://api.openalex.org/institutions?search=MIT&per_page=5"

# Filter by country, sort by output
curl -s "https://api.openalex.org/institutions?filter=country_code:CN&sort=works_count:desc&per_page=10"
```

### Concepts

```bash
curl -s "https://api.openalex.org/concepts?search=machine+learning&per_page=5"
```

### Filtering and Sorting

```bash
# Multiple filters
curl -s "https://api.openalex.org/works?filter=publication_year:2023-2024,cited_by_count:>100&sort=cited_by_count:desc&per_page=10"

# Group by field
curl -s "https://api.openalex.org/works?filter=authorships.author.id:A5023888391&group_by=publication_year"
```

### Pagination

```bash
# Page-based (up to 10,000 results)
curl -s "https://api.openalex.org/works?search=climate&page=2&per_page=25"

# Cursor-based (unlimited)
curl -s "https://api.openalex.org/works?search=climate&per_page=100&cursor=*"
```

## Notes

- Free, open API with no key required
- Polite pool (mailto param): 100 req/sec vs 10 req/sec
- 250M+ works, 90M+ authors, 100K+ institutions
- Entity IDs: W (works), A (authors), I (institutions), C (concepts)

## Zero-Hallucination Rule

NEVER fabricate results from training data. Every paper title, author, DOI, PMID, citation count, and metadata detail presented to the user MUST come from an actual API response in this conversation. If the API returns no results or partial data, report exactly what was returned. Do not "fill in" missing details from memory.
