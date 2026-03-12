---
name: scienceclaw-classification
description: "Classify scientific content by discipline, methodology, topic, and quality. Use when: user asks to categorize papers, methods, or research outputs. NOT for: simple keyword tagging or non-scientific content."
metadata: { "openclaw": { "emoji": "🏷️" } }
---

# Scientific Classification Skill

Classify and categorize scientific content across all disciplines.

## When to Use

- "Classify this paper by discipline"
- "What research methodology does this use?"
- "Categorize these results by topic"
- "Assess the quality tier of this journal/paper"
- Sorting literature by approach or field
- Identifying study design type (RCT, cohort, case-control, etc.)

## When NOT to Use

- Simple keyword extraction (use scienceclaw-ie)
- Full paper summarization (use scienceclaw-summarization)
- Fact verification (use scienceclaw-verification)

## Classification Dimensions

### 1. Discipline Classification
Classify into one of 17+ primary disciplines and subdisciplines:
- **Natural Sciences**: Physics, Chemistry, Biology, Medicine, Materials Science, Astronomy, Earth Science, Environmental Science, Agricultural Science
- **Formal Sciences**: Mathematics, Computer Science
- **Social Sciences**: Economics, Sociology, Psychology, Political Science, Linguistics
- **Humanities**: Philosophy, History, Law

### 2. Methodology Classification
- **Empirical**: experimental, observational, survey, case study
- **Theoretical**: mathematical modeling, simulation, analytical
- **Computational**: data-driven, machine learning, numerical methods
- **Review**: systematic review, meta-analysis, scoping review, narrative review
- **Mixed Methods**: combining qualitative and quantitative

### 3. Study Design Classification
- Randomized controlled trial (RCT)
- Cohort study (prospective/retrospective)
- Case-control study
- Cross-sectional study
- Longitudinal study
- Qualitative study (ethnography, phenomenology, grounded theory)

### 4. Quality Assessment
- **Tier 1**: High-quality evidence (large RCTs, systematic reviews with meta-analysis)
- **Tier 2**: Moderate evidence (cohort studies, well-designed experiments)
- **Tier 3**: Low evidence (case reports, expert opinion, preliminary studies)
- **Tier 4**: Pre-print or non-peer-reviewed

## Output Format

Always structure classification output as:

```
**Discipline**: [Primary] > [Subdiscipline]
**Methodology**: [Type]
**Study Design**: [Design type]
**Quality Tier**: [1-4] — [Justification]
**Key Topics**: [topic1, topic2, ...]
**Confidence**: [High/Medium/Low]
```

## Guidelines

1. Always provide confidence level for classifications
2. Note when content spans multiple disciplines (interdisciplinary)
3. Distinguish between primary and secondary methodologies
4. Consider journal impact factor and peer-review status for quality assessment
5. Flag potential misclassifications or ambiguous cases
6. Use standardized vocabulary (MeSH terms for biomedical, ACM CCS for CS, etc.)
