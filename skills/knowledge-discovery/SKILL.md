---
name: knowledge-discovery
description: Discover patterns, build knowledge graphs, and extract insights from linguistic and historical data
---

# Knowledge Discovery & Graphs

## Purpose
Discover hidden patterns, build knowledge graphs, and extract novel insights from structured and unstructured data.

## Key Datasets
- **WALS** (wals.info): World Atlas of Language Structures — 192 linguistic features across 2,679 languages in CLDF format (CC-BY 4.0)
- **HistWords** (nlp.stanford.edu/projects/histwords): Historical word embeddings tracking semantic change across 4 languages over centuries (.npy/.pkl format)

## Protocol
1. **Data exploration** — Profile data, identify patterns, check distributions
2. **Feature engineering** — Create derived features, temporal features, cross-references
3. **Pattern detection** — Apply clustering, association rules, anomaly detection
4. **Knowledge graph construction** — Build entity-relation graphs from discovered patterns
5. **Insight generation** — Interpret patterns in domain context
6. **Validation** — Verify discoveries against known phenomena

## Discovery Types
- **Linguistic typology**: Cross-linguistic universals, language family features, areal patterns
- **Semantic change**: Word meaning evolution, neologism tracking, conceptual drift
- **Scientific trends**: Emerging research topics, citation patterns, collaboration networks
- **Biomedical discovery**: Drug repurposing candidates, gene-disease associations

## Rules
- Distinguish between correlation and causation in discovered patterns
- Report statistical significance and effect sizes
- Validate against domain expertise and existing literature
- Handle missing data transparently
- For knowledge graphs, use standard ontologies (RDF, OWL) when possible
