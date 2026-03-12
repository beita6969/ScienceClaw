---
name: scienceclaw-summarization
description: "Summarize scientific papers, datasets, experimental results, and literature reviews. Use when: (1) condensing research papers, (2) creating literature reviews, (3) summarizing experimental findings, (4) meta-analysis synthesis, (5) creating executive summaries of research. NOT for: information extraction (use scienceclaw-ie), full paper retrieval (use scienceclaw-retrieval), or writing new content (use scienceclaw-generation)."
metadata: { "openclaw": { "emoji": "📝" } }
---

# scienceclaw-summarization

Summarize scientific papers, datasets, experimental results, and literature reviews with discipline-aware precision and faithful representation of source material.

## When to Use

- Condensing a full research paper into a structured abstract or brief summary
- Creating literature review summaries across multiple papers on a topic
- Summarizing experimental findings, including methods, results, and conclusions
- Synthesizing results from multiple studies for meta-analysis overviews
- Producing executive summaries of research for non-specialist audiences
- Distilling key takeaways from conference proceedings or preprint batches
- Generating comparative summaries across related studies

## When NOT to Use

- Extracting structured data points, entities, or relations from papers -- use `scienceclaw-ie`
- Retrieving or finding papers from databases -- use `scienceclaw-retrieval`
- Writing original research content, drafts, or manuscripts -- use `scienceclaw-generation`
- Answering specific factual questions about science -- use `scienceclaw-qa`
- Verifying claims or checking statistical validity -- use `scienceclaw-verification`

## Summary Types

### Abstract-Style Summary
A concise summary (150-300 words) that mirrors the structure of a scientific abstract:
1. **Background** -- one to two sentences of context and motivation
2. **Objective** -- the research question or hypothesis
3. **Methods** -- brief description of approach, dataset, or experimental design
4. **Results** -- key quantitative findings with effect sizes and confidence intervals where available
5. **Conclusion** -- main takeaway and implications

### Executive Summary
A high-level overview (300-500 words) aimed at decision-makers or non-specialists:
1. **Problem Statement** -- why this research matters
2. **Approach** -- what was done, in plain language
3. **Key Findings** -- the most impactful results, translated for a general audience
4. **Implications** -- practical significance and next steps
5. **Limitations** -- major caveats or open questions

### Detailed Summary
A thorough walkthrough (500-1500 words) preserving methodological detail:
1. **Introduction and Motivation** -- full context and prior work referenced
2. **Methods and Materials** -- detailed experimental or analytical design
3. **Results** -- comprehensive reporting of all major findings, tables, and figures described
4. **Discussion** -- interpretation, comparison with related work, alternative explanations
5. **Limitations and Future Work** -- weaknesses acknowledged by authors and beyond

### Systematic Review Summary
A structured synthesis across multiple papers:
1. **Search Strategy** -- how papers were identified and selected
2. **Inclusion/Exclusion Criteria** -- what qualified for the review
3. **Study Characteristics** -- table of included studies with key attributes
4. **Synthesized Findings** -- aggregated results, agreement and disagreement across studies
5. **Quality Assessment** -- risk of bias and evidence strength per study
6. **Gaps and Recommendations** -- what remains unanswered

## Discipline-Aware Terminology

Summaries must respect the vocabulary conventions of the source discipline:

- **Biomedical Sciences** -- use MESH terms, standard gene/protein nomenclature, clinical trial phase terminology, CONSORT-aligned reporting
- **Physics and Astronomy** -- preserve unit conventions (SI, CGS), uncertainty notation, standard model terminology
- **Computer Science** -- retain benchmark names, model architecture terms, dataset identifiers, metric abbreviations (F1, BLEU, ROUGE)
- **Chemistry** -- use IUPAC nomenclature, preserve reaction notation, maintain spectroscopic data references
- **Social Sciences** -- respect statistical reporting norms (APA style), effect size conventions, survey methodology terms
- **Earth and Environmental Sciences** -- preserve geospatial references, climate model identifiers, temporal scale descriptors

When summarizing across disciplines (interdisciplinary work), define domain-specific terms on first use and favor the terminology conventions of the primary discipline.

## Citation Handling

### In-Summary Citations
- Preserve author-year citations from the source when referencing specific claims: "(Smith et al., 2024)"
- When summarizing multiple papers, maintain consistent citation format throughout
- Use numbered references [1], [2] when summarizing more than ten sources for readability
- Always attribute quantitative claims to their source study

### Citation Integrity Rules
- Never fabricate citations -- if a referenced work cannot be confirmed, note it as "cited by authors, not independently verified"
- Preserve DOI links when available in the source material
- Flag retracted or corrected papers when encountered during summarization
- Distinguish between primary sources (original research) and secondary sources (reviews, textbooks)

### Reference List
- Append a reference list at the end of systematic review summaries and literature review summaries
- Use a consistent format (preferably matching the source discipline convention)
- Include DOIs where available

## Output Templates

### Single Paper Summary Template
```
Title: [Paper Title]
Authors: [Author List]
Source: [Journal/Preprint Server, Year]
DOI: [DOI if available]

Summary Type: [Abstract-Style | Executive | Detailed]

[Summary content organized by the selected type structure above]

Key Metrics: [Primary quantitative results]
Limitations Noted: [Major caveats]
```

### Multi-Paper Synthesis Template
```
Topic: [Research Topic]
Papers Reviewed: [Count]
Date Range: [Earliest -- Latest publication]

Synthesis Type: [Literature Review | Systematic Review | Meta-Analysis Overview]

[Synthesis content organized by the selected type structure above]

Agreement: [Points of consensus across studies]
Disagreement: [Points of conflict or contradiction]
Gaps: [Unanswered questions identified]

References:
[Numbered reference list]
```

## Quality Criteria

A good scientific summary must satisfy the following:

1. **Faithfulness** -- no claims that are not present in or directly supported by the source material
2. **Completeness** -- all major findings and caveats are represented, not just positive results
3. **Precision** -- quantitative results include exact values, units, confidence intervals, and p-values as reported
4. **Neutrality** -- avoids editorializing; reports what the authors found and claimed
5. **Clarity** -- readable by the target audience without losing scientific rigor
6. **Traceability** -- every claim can be traced back to its source document or section

## Zero-Hallucination Rule

ALL factual claims, citations, database results, and scientific data presented to the user MUST come from actual tool results (API calls, code execution, web search) in this conversation. NEVER fabricate or "fill in" details from training data. If a tool returns no results or partial data, report exactly what happened.
