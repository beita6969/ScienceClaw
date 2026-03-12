---
name: ssrn-econpapers
description: "Search social science research papers across SSRN, RePEc/IDEAS, CrossRef, and Semantic Scholar. Use when: (1) finding working papers in economics, finance, law, or political science, (2) searching RePEc for economics papers, (3) querying CrossRef for social science journal articles, (4) retrieving citation metadata for social science research. NOT for: natural sciences (use arxiv-search or pubmed-search), computer science preprints (use arxiv-search), general academic search (use semantic-scholar)."
metadata: { "openclaw": { "emoji": "\U0001F4CA", "requires": { "bins": ["curl"] } } }
---

# Social Science Paper Search

Search working papers and published articles across SSRN, RePEc/IDEAS, CrossRef,
and Semantic Scholar. Covers economics, finance, law, management, political science,
and related social sciences.

## CrossRef API (Structured Metadata)

CrossRef provides the most reliable structured search for published social science:

```bash
# Search social science articles
curl -s "https://api.crossref.org/works?query=behavioral+economics+nudge&filter=subject:social-science&rows=10&sort=relevance" \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
for item in data.get('message', {}).get('items', []):
    title = item.get('title', ['N/A'])[0]
    doi = item.get('DOI', 'N/A')
    year = item.get('published-print', item.get('published-online', {}))
    yr = year.get('date-parts', [[None]])[0][0] if year else 'N/A'
    authors = '; '.join(
        f\"{a.get('family','')}, {a.get('given','')}\"
        for a in item.get('author', [])[:3]
    )
    print(f'[{yr}] {title}')
    print(f'  DOI: {doi}')
    print(f'  Authors: {authors}\n')
"
```

## CrossRef Filtered Queries

```bash
# Filter by date range and subject
curl -s "https://api.crossref.org/works?query=inequality+mobility&filter=from-pub-date:2020-01-01,subject:economics&rows=10&sort=published&order=desc"

# Search within a specific journal (by ISSN)
curl -s "https://api.crossref.org/journals/0002-8282/works?query=monetary+policy&rows=10"

# Lookup by DOI
curl -s "https://api.crossref.org/works/10.1257/aer.20171330"
```

## RePEc IDEAS Search

Search the largest economics-specific paper database:

```bash
# Search IDEAS/RePEc
curl -s "https://ideas.repec.org/cgi-bin/htsearch?q=trade+tariffs+welfare&cmd=Search%21&ul=&m=all&fmt=long&wm=wrd&sp=1&sy=1&dt=range&db=2020-01-01&de=" \
  | python3 -c "
import sys, re
html = sys.stdin.read()
titles = re.findall(r'<dt>\d+\.\s*<a href=\"([^\"]+)\">([^<]+)</a>', html)
for url, title in titles[:10]:
    print(f'{title.strip()}')
    print(f'  https://ideas.repec.org{url}\n')
"
```

## SSRN Search

Search SSRN working papers (HTML scraping, structured API not public):

```bash
# Search SSRN by keyword
curl -s "https://papers.ssrn.com/sol3/results.cfm?RequestTimeout=50000000&txtKey_Words=climate+finance+risk" \
  -H "User-Agent: Mozilla/5.0" \
  | python3 -c "
import sys, re
html = sys.stdin.read()
papers = re.findall(r'<a class=\"title\" href=\"(https://ssrn.com/abstract=\d+)\"[^>]*>([^<]+)</a>', html)
for url, title in papers[:10]:
    print(f'{title.strip()}')
    print(f'  {url}\n')
"
```

## Semantic Scholar (Backup for Social Science)

```bash
# Semantic Scholar API with social science field filter
curl -s "https://api.semanticscholar.org/graph/v1/paper/search?query=causal+inference+social+policy&fieldsOfStudy=Economics,Sociology,Political+Science&fields=title,year,authors,citationCount,externalIds&limit=10" \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
for p in data.get('data', []):
    title = p.get('title', 'N/A')
    year = p.get('year', 'N/A')
    cites = p.get('citationCount', 0)
    authors = ', '.join(a['name'] for a in p.get('authors', [])[:3])
    doi = p.get('externalIds', {}).get('DOI', '')
    print(f'[{year}] ({cites} cites) {title}')
    print(f'  Authors: {authors}')
    if doi:
        print(f'  DOI: {doi}')
    print()
"
```

## Common Social Science Fields (Semantic Scholar)

Use with `fieldsOfStudy` parameter: `Economics`, `Sociology`, `Political Science`,
`Business`, `Psychology`, `Law`, `Geography`.

## Key Journal ISSNs for CrossRef

AER: `0002-8282`, QJE: `0033-5533`, JPE: `0022-3808`, Econometrica: `0012-9682`,
REStud: `0034-6527`, JoF: `0022-1082`, APSR: `0003-0554`, ASR: `0003-1224`.

## Best Practices

1. Start with CrossRef for published articles; it provides structured metadata and DOIs.
2. Use SSRN for recent working papers not yet in journals.
3. Use RePEc/IDEAS for the broadest economics-specific coverage.
4. Semantic Scholar is useful as a fallback with citation counts and field filtering.
5. Always include `&rows=` or `&limit=` to control result count.
6. Add `mailto` parameter to CrossRef for polite pool: `&mailto=you@example.com`.
7. Respect rate limits: CrossRef allows 50 req/sec with mailto, 1 req/sec without.
