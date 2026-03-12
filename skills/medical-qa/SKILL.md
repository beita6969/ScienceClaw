---
name: medical-qa
description: Medical question answering using structured biomedical knowledge bases and clinical datasets
---

# Medical Question Answering

## Purpose
Answer medical and biomedical questions with evidence-based precision using structured datasets and clinical knowledge bases.

## Key Datasets
- **MedQuAD** (abachaa/MedQuAD): 47,457 QA pairs from 12 NIH sources (NCI, GARD, GHR, MedlinePlus, NIDDK, NHLBI, NICHD, NIA, NIAMS, NINDS, NIDA, GARD)
- **PubMedQA** (qiaojin/PubMedQA): Yes/No/Maybe reasoning from PubMed abstracts

## Protocol
1. **Parse the question** — Identify medical entities (diseases, drugs, genes, symptoms)
2. **Source identification** — Match question type to appropriate NIH source
3. **Evidence retrieval** — Search PubMed, clinical guidelines, drug databases
4. **Answer synthesis** — Provide answer with confidence level and citations
5. **Verification** — Cross-reference with at least 2 independent sources

## Question Types
- Disease/condition: Etiology, diagnosis, prognosis, treatment
- Drug/treatment: Mechanism, dosage, side effects, interactions
- Genetic: Gene function, variants, inheritance patterns
- Prevention: Risk factors, screening, lifestyle modifications

## Rules
- Always cite primary sources (PMID, DOI, or guideline reference)
- Distinguish between established evidence and emerging research
- Flag when evidence is limited or conflicting
- Never provide personalized medical advice
- Include confidence level: HIGH (multiple RCTs), MODERATE (observational), LOW (case reports)
