---
name: scientific-retrieval
description: Retrieve and recommend relevant documents from financial, historical, and scientific archives
---

# Scientific Retrieval & Recommendation

## Purpose
Retrieve relevant documents, datasets, and resources from large scientific and domain-specific archives.

## Key Datasets
- **Financial Reports SEC** (JanosAudran/financial-reports-sec): SEC 10-K filings with 20 sections and sentiment labels
- **Historical Newswire** (dell-research-harvard/newswire): Historical news article corpus for digital humanities research

## Protocol
1. **Query analysis** — Parse information need, identify key concepts and constraints
2. **Source selection** — Choose appropriate databases and archives
3. **Search execution** — Multi-strategy search (keyword, semantic, citation-based)
4. **Relevance ranking** — Score and rank results by relevance, authority, recency
5. **Result synthesis** — Organize and present findings with metadata

## Retrieval Domains
- **Financial documents**: SEC filings (10-K, 10-Q, 8-K), earnings calls, analyst reports
- **Historical archives**: Newspapers, government records, digitized manuscripts
- **Scientific literature**: Journal articles, preprints, conference proceedings
- **Patent databases**: USPTO, EPO, WIPO patent documents

## Recommendation Types
- **Similar documents**: Find related papers/reports based on content similarity
- **Citation chain**: Forward/backward citation tracking
- **Cross-domain**: Find analogous work in different disciplines
- **Temporal**: Track how a topic evolves over time

## Rules
- Always report search coverage and potential gaps
- Rank by relevance, not just recency
- Include document metadata (date, source, section, author)
- For financial documents, note the filing period and any restatements
