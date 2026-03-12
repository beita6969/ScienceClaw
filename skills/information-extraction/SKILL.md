---
name: information-extraction
description: Extract structured entities, relations, and clauses from scientific and legal documents
---

# Information Extraction

## Purpose
Extract structured information (entities, relations, events, clauses) from unstructured scientific and domain-specific text.

## Key Datasets
- **ChemProt** (bigbio/chemprot): Chemical-protein interaction extraction from BioCreative VI; 10 relation types (CPR:3-CPR:9) between chemicals and proteins
- **CUAD** (atticus-project/cuad): Contract Understanding Atticus Dataset; 41 clause types from 510 legal contracts (CC-BY licensed)
- **JNLPBA**: Biomedical named entity recognition (protein, DNA, RNA, cell line, cell type)
- **SciERC**: Scientific entity and relation extraction from AI paper abstracts

## Protocol
1. **Schema definition** — Define target entity types, relation types, and attributes
2. **Preprocessing** — Sentence segmentation, tokenization, abbreviation expansion
3. **Entity recognition** — Identify and classify named entities (NER)
4. **Relation extraction** — Detect relationships between entity pairs (RE)
5. **Normalization** — Map entities to standard ontologies (MeSH, ChEBI, UniProt)
6. **Output structuring** — Format as structured JSON, RDF triples, or knowledge graph

## Extraction Types
- **Chemical-protein interactions**: Substrate, inhibitor, agonist, antagonist, activator
- **Legal clause extraction**: Termination, IP rights, non-compete, indemnification, limitation of liability
- **Gene-disease associations**: Causal, biomarker, therapeutic target
- **Drug-drug interactions**: Synergistic, antagonistic, pharmacokinetic

## Rules
- Report extraction confidence scores for each entity/relation
- Provide span offsets for traceability back to source text
- Normalize entities to standard identifiers (CAS, UniProt ID, etc.)
- Handle nested entities and overlapping relations
- Validate extracted facts against known databases when possible
