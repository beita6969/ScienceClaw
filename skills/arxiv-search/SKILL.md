---
name: arxiv-search
description: "Search arXiv for preprints in physics, math, CS, quantitative biology, quantitative finance, statistics, electrical engineering, economics. Use when: (1) finding preprints by topic, (2) searching by author, (3) browsing arXiv categories, (4) getting paper metadata/abstracts. NOT for: published journal articles (use crossref-search), biomedical (use pubmed-search)."
metadata: { "openclaw": { "emoji": "📄", "requires": { "bins": ["curl"] } } }
---

# arXiv Search

Search arXiv preprints via public API. Covers physics, math, CS, q-bio, q-fin,
statistics, electrical engineering, and economics.

## API Endpoint

```bash
curl -s "http://export.arxiv.org/api/query?search_query=all:transformer+attention&start=0&max_results=5"
```

Parameters: `search_query=` (required), `id_list=` (direct lookup by arXiv ID),
`start=` (pagination offset), `max_results=` (default 10, max 30000),
`sortBy=relevance|lastUpdatedDate|submittedDate`, `sortOrder=ascending|descending`.

## Query Syntax

**Field prefixes**: `ti:` title, `au:` author, `abs:` abstract, `co:` comment,
`jr:` journal ref, `cat:` category, `all:` all fields.

**Boolean**: `AND`, `OR`, `ANDNOT`. Example:
```bash
curl -s "http://export.arxiv.org/api/query?search_query=au:bengio+AND+cat:cs.LG+AND+ti:attention&max_results=10"
```

## Category Codes

**Physics**: `astro-ph` (.CO/.EP/.GA/.HE/.IM/.SR), `cond-mat` (.dis-nn/.mes-hall/.mtrl-sci/.soft/.stat-mech/.str-el/.supr-con), `hep-ex`, `hep-lat`, `hep-ph`, `hep-th`, `quant-ph`, `gr-qc`, `nucl-ex`, `nucl-th`

**CS**: `cs.AI`, `cs.CL` (NLP), `cs.CV`, `cs.LG` (ML), `cs.CR`, `cs.DB`, `cs.DS`, `cs.SE`, `cs.RO`

**Math**: `math.AG`, `math.AP`, `math.CO`, `math.PR`, `math.ST`

**Other**: `q-bio` (.BM/.CB/.GN/.MN/.NC/.PE/.QM/.SC/.TO), `q-fin` (.CP/.EC/.GN/.MF/.PM/.PR/.RM/.ST/.TR), `stat` (.AP/.CO/.ME/.ML/.OT/.TH), `eess` (.AS/.IV/.SP/.SY), `econ` (.EM/.GN/.TH)

## Response Parsing

The API returns Atom XML. Parse with Python:

```bash
curl -s "http://export.arxiv.org/api/query?search_query=ti:large+language+model&max_results=5&sortBy=submittedDate&sortOrder=descending" | python3 -c "
import sys, xml.etree.ElementTree as ET
ns = {'a': 'http://www.w3.org/2005/Atom'}
root = ET.parse(sys.stdin).getroot()
for entry in root.findall('a:entry', ns):
    title = entry.find('a:title', ns).text.strip().replace('\n', ' ')
    aid = entry.find('a:id', ns).text.strip().split('/abs/')[-1]
    pub = entry.find('a:published', ns).text[:10]
    authors = ', '.join(a.find('a:name', ns).text for a in entry.findall('a:author', ns))
    print(f'[{aid}] {pub} | {title}')
    print(f'  Authors: {authors}\n')
"
```

## Direct Lookup and Pagination

```bash
# By ID
curl -s "http://export.arxiv.org/api/query?id_list=2301.07041,2302.13971"

# Pagination
curl -s "http://export.arxiv.org/api/query?search_query=cat:cs.AI&start=0&max_results=25&sortBy=submittedDate&sortOrder=descending"
curl -s "http://export.arxiv.org/api/query?search_query=cat:cs.AI&start=25&max_results=25&sortBy=submittedDate&sortOrder=descending"
```

## Rate Limiting

No official limit, but keep to 1 request per 3 seconds for bulk queries.
For large-scale harvesting, use the OAI-PMH bulk access endpoint instead.

## Best Practices

1. Use `sortBy=submittedDate&sortOrder=descending` for latest papers.
2. Combine `cat:` with keyword searches for targeted results.
3. Check `opensearch:totalResults` in the response for total match count.
4. For PDF access, replace `/abs/` with `/pdf/` in the paper URL.
5. Use `id_list` for direct lookups (faster and more reliable).
6. URL-encode spaces as `+` in query terms.

## Zero-Hallucination Rule

NEVER fabricate results from training data. Every paper title, author, DOI, PMID, citation count, and metadata detail presented to the user MUST come from an actual API response in this conversation. If the API returns no results or partial data, report exactly what was returned. Do not "fill in" missing details from memory.
