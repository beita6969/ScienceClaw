---
name: scienceclaw-retrieval
description: "Retrieve scientific information from databases, literature, and knowledge bases. Use when: (1) finding relevant papers, (2) querying scientific databases, (3) cross-referencing findings, (4) building bibliographies, (5) systematic literature search. NOT for: answering questions (use scienceclaw-qa), summarizing (use scienceclaw-summarization), or data analysis (use code-execution skill)."
metadata: { "openclaw": { "emoji": "📚" } }
---

# scienceclaw-retrieval

Retrieve scientific information from databases, literature repositories, and knowledge bases using structured search strategies, relevance ranking, and citation chaining.

## When to Use

- Finding relevant papers on a specific research topic or question
- Querying scientific databases (PubMed, arXiv, Semantic Scholar, CrossRef, OpenAlex)
- Cross-referencing findings across multiple sources and databases
- Building comprehensive bibliographies for a research project or review
- Conducting systematic literature searches with reproducible methodology
- Tracking citation networks to discover related or derivative work
- Locating datasets, code repositories, or supplementary materials linked to publications

## When NOT to Use

- Answering specific scientific questions -- use `scienceclaw-qa`
- Summarizing papers or synthesizing findings -- use `scienceclaw-summarization`
- Running data analysis or computations on retrieved data -- use code-execution skill
- Extracting structured information from paper text -- use `scienceclaw-ie`
- Verifying claims or checking calculations -- use `scienceclaw-verification`

## Multi-Database Search Strategies

**Parallel Search**: For broad discovery, query PubMed, arXiv, Semantic Scholar, and OpenAlex simultaneously, collect results with DOIs and metadata, deduplicate by DOI, apply relevance ranking, then filter by date/type/discipline.

**Sequential Refinement**: For targeted retrieval, start with a broad query to gauge the landscape, analyze initial results for recurring keywords and author clusters, refine with Boolean operators and filters, snowball via citation chaining on top hits, and stop at saturation (new queries return mostly known results).

**Systematic Review Search**: Define PICO/PEO framework, construct Boolean queries with synonyms and controlled vocabulary, document every query/database/date/count for reproducibility, include grey literature (preprints, proceedings, registries), screen via title/abstract then full-text phases, and track numbers through a PRISMA flow diagram.

## Database-Specific Query Syntax

- **PubMed**: MeSH terms via `[MeSH Terms]`, field tags `[tiab]`/`[au]`/`[dp]`, capitalized Boolean operators, `[pt]` for publication type. Example: `"machine learning"[tiab] AND "drug discovery"[tiab] AND "2023"[dp]`
- **arXiv**: Field prefixes `ti:`/`au:`/`abs:`/`cat:`, Boolean AND/OR/ANDNOT, category codes (cs.AI, q-bio.BM), trailing wildcards. Example: `ti:"neural network" AND cat:cs.LG AND au:bengio`
- **Semantic Scholar**: API parameters `query`/`year`/`fieldsOfStudy`/`venue`, field filtering, pagination via `offset`/`limit`, direct lookup by DOI or arXiv ID
- **CrossRef**: `/works?query=` endpoint, filters like `from-pub-date:2023,type:journal-article`, field queries `query.title=`/`query.author=`, sort by `relevance`/`published`/`is-referenced-by-count`
- **OpenAlex**: Entity endpoints `/works`/`/authors`/`/sources`, filters with commas (AND) or pipe (OR), `search=` for full-text, `group_by=` for aggregation, open access filtering via `open_access.is_oa:true`

## Relevance Ranking

Combine multiple scoring signals: textual similarity between query and title/abstract (primary), citation count with recency weighting, publication date, venue quality (impact factor or acceptance rate), author authority (h-index in subfield), and reference overlap with known relevant papers. For active fields, apply time-decayed citation scoring: `adjusted_score = citation_count / (current_year - publication_year + 1)`. Support user-guided re-ranking by marking papers as highly relevant, somewhat relevant, or not relevant, then refine queries using terms from top-marked papers.

## Citation Chaining

- **Forward chaining (cited-by)**: From a seed paper, find all papers that cite it via Semantic Scholar or OpenAlex, filter by date/venue/topic, repeat for new relevant hits (limit depth to 2-3 hops)
- **Backward chaining (references)**: Extract the seed paper's reference list, score references by co-occurrence frequency across your relevant set, identify foundational works
- **Co-citation analysis**: Gather citation neighborhoods of 3-5 seed papers, find papers appearing in multiple neighborhoods as conceptually related candidates
- **Bibliographic coupling**: Find papers sharing high reference overlap with seed papers, indicating they address similar research questions

## Deduplication

- **DOI-based**: Primary key for deduplication; prefer the record with richest metadata when merging
- **Fuzzy title matching**: For records without DOIs, normalize titles (lowercase, strip punctuation/articles), apply Jaccard > 0.85 or edit distance ratio > 0.90, verify by checking author overlap and publication year
- **Preprint-publication linking**: Match arXiv preprints to journal versions via DOI metadata or title matching, prefer published version but retain preprint if it has additional content (appendices, code), flag substantial differences between versions

## Integration with Specialized Skills

- **PubMed**: biomedical and life sciences, MeSH controlled vocabulary, structured abstracts, clinical trial metadata
- **arXiv**: physics, math, CS, quantitative biology preprints, open-access full-text PDFs, new submission monitoring
- **Semantic Scholar**: cross-disciplinary search, citation graph features, TLDR summaries, influential citation filtering
- **CrossRef**: DOI resolution, comprehensive metadata, funding and license data, reference lists, citation ambiguity resolution
- **OpenAlex**: large-scale bibliometrics, trend discovery, open-access links via Unpaywall, concept tagging for topic filtering

## Output Format

### Single Query Result
```
Query: [Search query text]
Database(s): [Databases searched]
Total Results: [Count]
After Deduplication: [Count]

Top Results:
  1. [Title] | [Authors] | [Year] | [Venue]
     DOI: [DOI] | Citations: [Count]
     Relevance: [Score] | Abstract: [First 200 chars...]
```

### Systematic Search Report
```
Search Strategy Report
======================
Research Question: [PICO-formatted question]
Date Executed: [Date]

Database Searches:
  - PubMed: [Query] -> [N results]
  - arXiv: [Query] -> [N results]
  - Semantic Scholar: [Query] -> [N results]

Total Retrieved: [N]
After Deduplication: [N]
After Title/Abstract Screening: [N]
Final Included: [N]

Included Papers:
  [Numbered list with full bibliographic details]
```

## Zero-Hallucination Rule

ALL factual claims, citations, database results, and scientific data presented to the user MUST come from actual tool results (API calls, code execution, web search) in this conversation. NEVER fabricate or "fill in" details from training data. If a tool returns no results or partial data, report exactly what happened.
