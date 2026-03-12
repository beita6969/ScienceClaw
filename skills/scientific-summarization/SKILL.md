---
name: scientific-summarization
description: Summarize and simplify scientific literature, educational content, and research papers
---

# Scientific Summarization & Simplification

## Purpose
Generate concise, accurate summaries of scientific papers, educational materials, and complex technical documents.

## Key Datasets
- **PubMed Summarization** (ccdv/pubmed-summarization): Article-abstract pairs for biomedical summarization
- **LearningQ** (AngusGLChen/LearningQ): TED-Ed (7K) + Khan Academy (223K) educational QA for learning-oriented summarization

## Protocol
1. **Document analysis** — Identify paper structure (IMRaD, review, case report)
2. **Key claim extraction** — Extract main findings, methods, and conclusions
3. **Audience calibration** — Adjust complexity to target audience (expert, student, public)
4. **Summary generation** — Structured summary with key takeaways
5. **Fidelity check** — Verify no hallucinated claims; all statements traceable to source

## Summary Types
- **Structured abstract**: Background, Methods, Results, Conclusions
- **Lay summary**: Plain-language explanation for non-experts
- **Technical brief**: Key findings and implications for domain experts
- **Educational summary**: Concept-first explanation with learning objectives

## Rules
- Never introduce claims not present in the source material
- Preserve numerical results exactly (p-values, effect sizes, confidence intervals)
- Flag study limitations mentioned by authors
- Distinguish between authors' conclusions and your interpretation
- For educational content, maintain pedagogical structure
