---
name: paper-analysis
description: Read, summarize, and critically analyze scientific papers. Extract key findings, methodology, limitations, and contributions. Use when user shares a paper (PDF/URL/DOI), asks to summarize a paper, critique methodology, extract data from a paper, compare papers, or do a critical review. Triggers on "summarize this paper", "analyze this study", "what does this paper say", "critique this methodology", "extract findings from".
---

# Paper Analysis

Systematic scientific paper reading, summarization, and critical analysis.

## Paper Acquisition

1. If user provides a DOI: fetch via `https://doi.org/DOI` or Semantic Scholar API
2. If user provides arXiv ID: fetch via `https://arxiv.org/abs/ID`
3. If user provides a URL: use `web_fetch` to extract content
4. If user provides a PDF file: read directly or use summarize skill

## Analysis Framework

### Quick Summary (default)
Provide in ~200 words:
- Research question / objective
- Method (1-2 sentences)
- Key findings (2-3 bullet points)
- Main contribution
- One key limitation

### Deep Analysis
When requested, provide structured analysis:

#### 1. Paper Metadata
- Title, authors, year, journal/venue, DOI
- Citation count (via Semantic Scholar)

#### 2. Research Context
- Problem statement and motivation
- Research gap being addressed
- Theoretical framework

#### 3. Methodology Assessment
- Study design (experimental/observational/computational/theoretical)
- Sample/dataset description
- Variables (independent, dependent, controls)
- Analysis methods
- Reproducibility assessment (data/code availability)

#### 4. Results Evaluation
- Key findings with effect sizes and confidence intervals
- Statistical significance vs practical significance
- Figures and tables interpretation
- Are results consistent with claims?

#### 5. Critical Assessment

Check for:
- **Internal validity**: confounds, selection bias, measurement error
- **External validity**: generalizability, ecological validity
- **Statistical issues**: multiple comparisons, p-hacking, small N
- **Logical issues**: correlation ≠ causation, survivorship bias
- **Reporting issues**: selective reporting, missing negative results
- **Methodological rigor**: appropriate controls, blinding, randomization

#### 6. Contribution & Impact
- Novelty assessment
- Practical implications
- Theoretical implications
- Future directions suggested

## Comparison Mode

When comparing multiple papers:

| Dimension | Paper A | Paper B | Paper C |
|-----------|---------|---------|---------|
| Research Question | | | |
| Method | | | |
| Sample Size | | | |
| Key Finding | | | |
| Limitation | | | |
| Strength | | | |

## Domain-Specific Checklists

### RCT (Randomized Controlled Trial)
- CONSORT checklist compliance
- Randomization method, allocation concealment
- Blinding (single/double/triple)
- ITT vs per-protocol analysis
- Dropout rates and handling

### Observational Studies
- STROBE checklist
- Confounding control methods
- Selection bias assessment

### Machine Learning Papers
- Train/val/test split methodology
- Baseline comparisons
- Ablation studies
- Statistical significance of improvements
- Computational cost reporting
- Code/data availability

### Qualitative Research
- Sampling strategy (purposive, theoretical, snowball)
- Data saturation
- Coding methodology (thematic, grounded theory)
- Reflexivity and positionality
- Member checking / triangulation

## Output Style
- Use academic but accessible language
- Cite specific sections/figures/tables from the paper
- Distinguish between what the paper claims and what the evidence supports
- Flag any red flags clearly but diplomatically
