---
name: systematic-review
description: "Orchestrates a systematic review and meta-analysis workflow following PRISMA 2020 guidelines, from protocol development through multi-database search, screening, data extraction, and evidence synthesis. Use when conducting evidence-based reviews, meta-analyses, or scoping reviews. NOT for single-study analysis or narrative literature surveys."
metadata: { "openclaw": { "emoji": "📋" } }
---

# Systematic Review (Meta Skill)

This meta-skill coordinates a complete systematic review pipeline following
PRISMA 2020 guidelines. It integrates multi-database literature searching,
structured screening, information extraction, quantitative synthesis, and
standardized reporting into a rigorous evidence review workflow by combining
three specialized skills.

## Workflow

### Step 1: Protocol Development

Define the review protocol before conducting any searches:
- Formulate the research question using the PICO framework
  (Population, Intervention, Comparator, Outcome)
- Establish inclusion and exclusion criteria with explicit justification
- Define the search strategy: databases, date range, language restrictions
- Specify outcome measures and effect size metrics
- Pre-register the protocol (PROSPERO or OSF recommended)
- Document any planned sensitivity or subgroup analyses

### Step 2: Multi-Database Systematic Search

Execute comprehensive searches across multiple bibliographic databases:
- **PubMed/MEDLINE**: Biomedical and clinical literature via structured MeSH queries
- **arXiv**: Preprints in quantitative and computational fields
- **Semantic Scholar**: AI-augmented citation graph and full-text search
- **CrossRef**: DOI-based metadata and cross-publisher discovery
- Construct database-specific search strings from the master strategy
- Document exact queries, dates, result counts; deduplicate exports
- Supplement with citation chaining (forward and backward) on key papers

### Step 3: Screening and Eligibility Assessment

Apply a two-stage screening process to identify eligible studies:
- **Title/abstract screening**: Apply inclusion criteria, flag uncertain cases
- **Full-text assessment**: Evaluate against all criteria, document exclusion reasons
- Track inter-rater agreement (Cohen's kappa) if multiple reviewers
- Maintain a log of all screening decisions for the PRISMA flow diagram
- Resolve disagreements through discussion or third-reviewer arbitration

### Step 4: Structured Data Extraction

Extract pre-defined data elements from each included study:
- Study characteristics: design, setting, sample size, follow-up duration
- Population, intervention, comparator: demographics, dosage, duration
- Outcomes and results: endpoints, effect estimates, confidence intervals
- Quality indicators: randomization method, blinding, attrition, funding

Use ScienceClaw information extraction to assist with structured data capture
from PDF full texts, reducing manual effort and transcription errors.

### Step 5: Risk of Bias and Quality Assessment

Evaluate methodological quality of each included study:
- Apply appropriate tools (RoB 2 for RCTs, ROBINS-I for non-randomized, Newcastle-Ottawa)
- Assess each domain: selection, performance, detection, attrition, reporting
- Generate risk-of-bias summary figures (traffic light plots)
- Evaluate overall certainty of evidence using GRADE framework
- Document judgments with supporting quotations from study texts

### Step 6: Meta-Analysis and Evidence Synthesis

Perform quantitative synthesis when studies are sufficiently homogeneous:
- Calculate standardized effect sizes (SMD, OR, RR, HR as appropriate)
- Fit random-effects or fixed-effects meta-analysis models
- Generate forest plots with study-level and pooled estimates
- Assess heterogeneity: I-squared statistic, Cochran's Q test, tau-squared
- Subgroup and sensitivity analyses: leave-one-out, trim-and-fill, funnel plots

### Step 7: PRISMA Reporting and Final Output

Compile the review following PRISMA 2020 reporting standards:
- PRISMA flow diagram with identification, screening, eligibility, inclusion counts
- Characteristics of included studies table
- Risk-of-bias summary and individual study assessments
- Forest plots, funnel plots, and subgroup analysis figures
- Summary of findings table with GRADE certainty ratings
- Complete PRISMA 2020 checklist (Page et al., BMJ 2021;372:n71) cross-referenced
  to report sections

## Integration Points

- **literature-search** -- Multi-database querying, deduplication, citation chaining, export
- **scienceclaw-ie** -- Structured data extraction from PDFs, entity recognition, table parsing
- **paper-writing** -- PRISMA-compliant report generation, figure formatting, reference management

## Output Formats

- **PRISMA flow diagram**: Study counts at each screening stage with exclusion reasons
- **Study characteristics table**: Design, population, intervention, outcomes per study
- **Forest plot**: Effect sizes with CIs, weights, pooled estimate, heterogeneity stats
- **Risk-of-bias table**: Domain-level judgments per study with traffic light visualization
- **Summary of findings**: GRADE-rated evidence table for each outcome
- **Full report**: PRISMA 2020 compliant manuscript with all required sections

## PRISMA 2020 Checklist Reference

This workflow aligns with the PRISMA 2020 statement (Page et al., BMJ
2021;372:n71). The 27-item checklist spans title through other information,
and each workflow step maps to specific checklist items to ensure completeness.

## Best Practices

1. Register the protocol before beginning searches to reduce reporting bias
2. Use at least two independent reviewers for screening and extraction
3. Document every decision point for full transparency and reproducibility
4. Never modify inclusion criteria after seeing search results without justification
5. Report all pre-planned analyses regardless of statistical significance
6. Use GRADE to rate certainty of evidence for each outcome separately
7. Clearly distinguish direct evidence from indirect comparisons
8. Acknowledge limitations in study-level quality and review-level methodology
9. Update the review when substantial new evidence becomes available
10. Make extracted data and analysis code publicly available when possible
