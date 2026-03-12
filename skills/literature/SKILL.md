---
name: literature
description: # Literature Search & Review
---

# Literature Search & Review

## Overview
Comprehensive academic literature search and synthesis across 15+ sources.

## Capabilities
- Multi-database parallel search (PubMed, arXiv, bioRxiv, medRxiv, OpenAlex, Semantic Scholar, Crossref, DBLP, CORE, DOAJ, Europe PMC)
- Web search via Agent-Reach (Exa semantic search, Jina Reader for any URL/PDF)
- Social academic search (Twitter/X threads, YouTube talks, GitHub repos)
- Structured literature reviews with citation networks
- Knowledge gap identification
- Hypothesis generation from literature analysis

## Search Strategy
1. **Query optimization**: Short, keyword-based queries (max 7 words). PubMed-friendly syntax.
2. **Multi-source fan-out**: Parallel queries across all sources for maximum coverage.
3. **Deduplication**: By PMID, DOI, then normalized title.
4. **Reflection loop**: Evaluate coverage → identify gaps → generate follow-up queries.

## Citation Rules
- ZERO hallucinated citations. Every reference must come from real search data.
- Unified article schema: source_type, title, authors, year, journal, url, doi, pmid, abstract.
- Numbered references [1], [2], [3] — built only from retrieved articles.

## Best Practices
- Start broad, then narrow. First search finds the landscape; follow-up queries fill gaps.
- Cross-domain search. The breakthrough paper might be in an unexpected field.
- Check preprints AND published papers. Recent findings may only be on bioRxiv/arXiv.
- Verify high-impact claims. Use Semantic Scholar's citation count to identify landmark papers.
