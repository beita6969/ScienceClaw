---
name: literature-search
description: "Comprehensive multi-database scientific literature search orchestrating Semantic Scholar, OpenAlex, arXiv, PubMed, and CrossRef. Use when: (1) systematic literature review, (2) finding all relevant papers on a topic, (3) checking state of the art, (4) building comprehensive bibliographies. NOT for: single-database queries (use specific search skills), data analysis (use code-execution)."
metadata: { "openclaw": { "emoji": "🔎" } }
---

# Literature Search (Meta Skill)

Orchestrate comprehensive literature searches across multiple databases.
**Always execute real API calls** — never fabricate results or rely on training data.

## Priority Order of Databases

1. **Semantic Scholar** (PRIMARY) — best relevance ranking, AI TLDR summaries, citation graph
2. **OpenAlex** (PRIMARY) — 250M+ works, powerful filtering, open access URLs
3. **arXiv** — preprints in physics, math, CS, biology, finance, statistics
4. **PubMed** — biomedical and life sciences (NCBI may be unreachable from some networks)
5. **CrossRef** — DOI resolution and metadata only (NOT for search — poor relevance ranking)

**IMPORTANT**: CrossRef search results are poorly ranked by relevance. Never use CrossRef
as the primary search engine. Use it only for DOI-based lookups and metadata enrichment.

## Mandatory Search Protocol

Every literature search MUST follow this protocol:

### Step 1: Semantic Scholar Search (always do this first)

```bash
# Primary search — returns papers ranked by relevance with AI summaries
curl -s "https://api.semanticscholar.org/graph/v1/paper/search?\
query=YOUR+SEARCH+TERMS&limit=10&\
fields=title,authors,year,abstract,citationCount,influentialCitationCount,\
isOpenAccess,openAccessPdf,url,externalIds,tldr,venue,publicationDate"
```

Parse results with:
```bash
| python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'Total: {data[\"total\"]} papers')
for i, p in enumerate(data['data']):
    authors = ', '.join(a['name'] for a in (p.get('authors') or [])[:3])
    if len(p.get('authors') or []) > 3: authors += ' et al.'
    tldr = p.get('tldr', {})
    tldr_text = tldr['text'][:150] if tldr else 'N/A'
    oa = '🔓' if p.get('isOpenAccess') else '🔒'
    doi = (p.get('externalIds') or {}).get('DOI', '')
    print(f'[{i+1}] {p[\"title\"]}')
    print(f'    {authors} ({p.get(\"year\",\"?\")}) — {p.get(\"venue\",\"?\")}')
    print(f'    Cited: {p.get(\"citationCount\",0)} (influential: {p.get(\"influentialCitationCount\",0)}) {oa}')
    print(f'    TLDR: {tldr_text}')
    print(f'    DOI: {doi}')
    print()
"
```

Useful filters:
- `year=2022-2025` — restrict by year range
- `fieldsOfStudy=Computer Science` — filter by discipline
- `minCitationCount=10` — only cited papers

### Step 2: OpenAlex Search (for broader coverage + OA links)

```bash
# Complementary search with powerful filtering
curl -s "https://api.openalex.org/works?\
search=YOUR+SEARCH+TERMS&per_page=10&\
sort=relevance_score:desc&\
select=title,publication_year,cited_by_count,doi,authorships,open_access,\
primary_location,abstract_inverted_index&\
mailto=scienceclaw@openclaw.ai"
```

Useful filters (append to URL as `&filter=`):
- `publication_year:2023-2025` — year range
- `cited_by_count:>50` — minimum citations
- `open_access.is_oa:true` — only open access
- `authorships.author.id:A5023888391` — by author OpenAlex ID
- `concepts.id:C41008148` — by concept (e.g., Computer Science)

### Step 3: Discipline-Specific Database (if relevant)

| Discipline | Additional Database | Skill |
|---|---|---|
| Biomedicine / Clinical | PubMed | `pubmed-search` |
| Physics / CS / Math | arXiv | `arxiv-search` |
| Computer Science | DBLP | `dblp-search` |
| Economics / Social Sci | SSRN/RePEc | `ssrn-econpapers` |

### Step 4: Deduplication and Ranking

Match across databases by DOI (most reliable), then normalized title.
Rank by: Semantic Scholar relevance > citation count > influential citations > recency.

### Step 5: Citation Chaining (for thorough searches)

For top 3-5 seed papers, retrieve their references and citations:
```bash
# Forward citations (who cites this paper)
curl -s "https://api.semanticscholar.org/graph/v1/paper/{paperId}/citations?\
fields=title,year,citationCount,venue&limit=20"

# Backward references (what this paper cites)
curl -s "https://api.semanticscholar.org/graph/v1/paper/{paperId}/references?\
fields=title,year,citationCount,venue&limit=20"
```

### Step 6: Paper Recommendations (for discovery)

```bash
# Find similar papers
curl -s "https://api.semanticscholar.org/recommendations/v1/papers/\
forpaper/{paperId}?fields=title,year,citationCount,tldr&limit=10"
```

## Search Quality Checklist

Before presenting results, verify:
- [ ] At least Semantic Scholar was searched with a real API call
- [ ] Results contain real DOIs/paper IDs (not fabricated)
- [ ] Citation counts are from the API (not estimated)
- [ ] Each paper has a verifiable identifier (DOI, arXiv ID, PMID, or S2 URL)
- [ ] TLDR summaries are from Semantic Scholar (not self-generated)

## Output Format

```
[1] Title
    Authors (Year) — Venue
    Cited: N (influential: M) 🔓/🔒
    TLDR: AI-generated summary from Semantic Scholar
    DOI: 10.xxxx/xxxxx | arXiv: xxxx.xxxxx | PMID: xxxxxxxx
    URL: https://...
```

## Zero-Hallucination Rule (ABSOLUTE)

**Every citation detail must come from a tool result in this conversation.**

- NEVER fabricate or "fill in" paper titles, authors, DOIs, PMIDs, citation counts, or journal names from training data
- NEVER say "a well-known study by X et al." without having searched for it first
- If a search returns 0 results, report that honestly — do not substitute training knowledge
- If a tool returns partial metadata (title but no DOI), report only what the tool returned
- Before presenting any paper, verify: Did a tool in THIS conversation return this information?

## Common Pitfalls to Avoid

1. **DO NOT** use CrossRef `/works?query=` for discovery — its relevance ranking is poor
2. **DO NOT** fabricate paper titles, authors, or DOIs from training knowledge
3. **DO NOT** skip API calls and rely on what you "know" about the literature
4. **DO NOT** present Semantic Scholar TLDRs as your own analysis
5. **ALWAYS** run the actual curl commands and parse real responses
6. **ALWAYS** include at least one verifiable identifier per paper
7. **ALWAYS** self-check: every detail in your response must trace back to a tool result

## Rate Limits

| Database | Without Key | With Key |
|---|---|---|
| Semantic Scholar | 100 req/5 min | 1/sec sustained |
| OpenAlex | 10 req/sec (polite pool with mailto) | Same |
| arXiv | ~1 req/3 sec | Same |
| CrossRef | 1 req/sec | 50 req/sec (with mailto) |
