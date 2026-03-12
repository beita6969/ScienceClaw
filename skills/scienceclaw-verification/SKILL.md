---
name: scienceclaw-verification
description: "Verify scientific claims, check calculations, validate experimental designs, and fact-check citations. Use when: (1) checking a claim against evidence, (2) validating statistical analyses, (3) verifying experimental reproducibility claims, (4) fact-checking references, (5) adversarial review of research. NOT for: generating new content (use scienceclaw-generation), simple QA (use scienceclaw-qa)."
metadata: { "openclaw": { "emoji": "✅" } }
---

# scienceclaw-verification

Verify scientific claims, check calculations, validate experimental designs, and fact-check citations with structured, evidence-based assessment workflows.

## When to Use

- Checking whether a specific scientific claim is supported by cited evidence
- Validating statistical analyses, p-values, effect sizes, or confidence intervals
- Verifying that experimental designs are sound and controls are adequate
- Fact-checking references to confirm they exist and support the claims attributed to them
- Conducting adversarial review of a manuscript or research report
- Assessing reproducibility based on reported methods and data availability
- Cross-checking numerical results against raw data or supplementary materials

## When NOT to Use

- Generating new research content, hypotheses, or manuscripts -- use `scienceclaw-generation`
- Answering general scientific questions -- use `scienceclaw-qa`
- Summarizing papers or findings -- use `scienceclaw-summarization`
- Retrieving papers or building bibliographies -- use `scienceclaw-retrieval`
- Extracting structured information from papers -- use `scienceclaw-ie`

## Verification Workflow

Every verification task passes through four stages: Claim Decomposition, Evidence Search, Assessment, and Verdict Synthesis.

### Stage 1: Claim Decomposition

Break the target claim into atomic, independently verifiable sub-claims. For each, identify the core assertion, extract quantitative components (numbers, thresholds, comparisons), note causal vs. correlational language, list cited evidence, and record scope qualifiers.

### Stage 2: Evidence Search

For each sub-claim: check whether the cited reference actually contains the claimed information, seek independent corroboration from other sources, verify raw data if available, review methodology for sufficient detail, and check whether others have replicated the finding. Use `scienceclaw-retrieval` to locate papers and `scienceclaw-ie` to extract specific data points.

### Stage 3: Assessment

Assign one of the following verdicts to each sub-claim:

| Verdict | Definition |
|---|---|
| **SUPPORTED** | Evidence directly and clearly supports the claim |
| **PARTIALLY SUPPORTED** | Evidence supports the claim with caveats or qualifications |
| **UNVERIFIABLE** | Insufficient evidence available to confirm or deny |
| **CONTRADICTED** | Evidence directly contradicts the claim |
| **MISLEADING** | Claim is technically true but presented in a deceptive context |
| **CALCULATION ERROR** | Numerical result does not match when recomputed from available data |

### Stage 4: Verdict Synthesis

Combine sub-claim assessments into an overall verdict with confidence level (high/medium/low), an evidence summary citing key supporting and contradicting sources, an error catalog categorized by severity, and recommendations for corrections or further verification.

## Statistical Checking Patterns

Common statistical errors to check: p-value misinterpretation, unreported multiple comparison corrections (Bonferroni, FDR), underpowered studies claiming null results, missing effect sizes, confidence interval inconsistencies with reported estimates, incorrect degrees of freedom, violated distribution assumptions, and unacknowledged baseline imbalances in randomized trials.

Numerical consistency checks: verify table percentages sum correctly, confirm sub-analysis sample sizes are consistent with total N, validate that means and SDs are plausible for the measurement scale, ensure figures match text, and recompute derived statistics (odds ratios, hazard ratios) from raw counts when available. Apply GRIM tests (means possible given integer data and sample size) and SPRITE tests (summary stats consistent with plausible distributions).

## Adversarial Review Protocol

- **Methodology**: study design appropriateness, inclusion/exclusion criteria, control adequacy, randomization/blinding, confounder identification, measurement instrument validation
- **Results**: numerical consistency, effect size reporting, negative result inclusion, figure accuracy, outlier handling transparency
- **Citations**: reference existence verification (DOI check), claim-attribution accuracy, contrary findings acknowledgment, self-citation proportion
- **Logic**: conclusion-evidence alignment, generalization scope, alternative explanations, causal language justification

## Discipline-Specific Verification Criteria

- **Biomedical**: CONSORT/STROBE/PRISMA compliance, trial registration verification, IRB approval, conflict of interest disclosure, dose-response plausibility
- **Machine Learning**: benchmark version/split verification, hyperparameter tuning leakage, baseline comparison fairness, ablation completeness, reproducibility artifact availability
- **Physics**: unit consistency, dimensional analysis, order-of-magnitude plausibility, conservation law compliance, uncertainty propagation, calibration documentation
- **Social Sciences**: pre-registration verification (OSF), power analysis adequacy, effect size comparison with meta-analyses, demand characteristics controls, replication status of foundational claims

## Output Format

```
Verification Report
===================
Claim: [Original claim text]
Source: [Paper/report reference]
Date Verified: [Date]

Decomposed Sub-Claims:
  1. [Sub-claim] -- [VERDICT] -- [Brief justification]
  2. [Sub-claim] -- [VERDICT] -- [Brief justification]

Overall Verdict: [SUPPORTED | PARTIALLY SUPPORTED | UNVERIFIABLE | CONTRADICTED]
Confidence: [HIGH | MEDIUM | LOW]

Evidence Summary:
  Supporting: [Key supporting evidence with citations]
  Contradicting: [Key contradicting evidence with citations]

Errors Found:
  [Severity: CRITICAL | MAJOR | MINOR] -- [Description]

Recommendations:
  - [Suggested action items]
```

## Zero-Hallucination Rule

ALL factual claims, citations, database results, and scientific data presented to the user MUST come from actual tool results (API calls, code execution, web search) in this conversation. NEVER fabricate or "fill in" details from training data. If a tool returns no results or partial data, report exactly what happened.
