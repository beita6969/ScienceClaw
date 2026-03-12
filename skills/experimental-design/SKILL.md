---
name: experimental-design
description: "Design rigorous scientific experiments with power analysis and controls. Use when: user needs to plan an experiment, calculate sample sizes, or set up controls. NOT for: running experiments or analyzing collected data."
metadata: { "openclaw": { "emoji": "🔧" } }
---

# Experimental Design Skill

Design rigorous, reproducible experiments across scientific disciplines.

## When to Use

- "Design an experiment to test..."
- "How many samples do I need?"
- "What controls should I include?"
- "Help me plan a clinical trial"
- "Is this experimental design valid?"
- Power analysis and sample size calculation

## When NOT to Use

- Running the actual experiment (use code-execution)
- Analyzing collected data (use scipy-analysis + statsmodels-stats)
- Writing up results (use paper-writing)
- Literature review (use literature-search)

## Design Components

### 1. Research Question and Hypotheses
- State clear, testable research question
- Formulate H0 and H1 (see hypothesis-gen skill)
- Define primary and secondary outcomes

### 2. Study Design Selection

| Design | When to Use | Strengths | Weaknesses |
|--------|------------|-----------|------------|
| RCT | Causal inference needed | Gold standard causality | Expensive, ethical limits |
| Factorial | Multiple factors | Tests interactions | Complex analysis |
| Crossover | Within-subject comparison | Reduced variability | Carryover effects |
| Quasi-experimental | Randomization impossible | Practical feasibility | Weaker causality |
| Observational (cohort) | Long-term outcomes | Natural setting | Confounding |
| Case-control | Rare outcomes | Efficient for rare events | Recall bias |

### 3. Power Analysis

```python
# Sample size calculation template (using scipy/statsmodels)
from statsmodels.stats.power import TTestIndPower
analysis = TTestIndPower()
n = analysis.solve_power(
    effect_size=0.5,   # Cohen's d (small=0.2, medium=0.5, large=0.8)
    alpha=0.05,         # Significance level
    power=0.80,         # Statistical power (commonly 0.80 or 0.90)
    ratio=1.0,          # Ratio of group sizes (n2/n1)
    alternative='two-sided'
)
print(f"Required sample size per group: {int(n) + 1}")
```

Key parameters:
- **Effect size**: Expected magnitude of difference
- **Alpha**: Type I error rate (usually 0.05)
- **Power**: 1 - Type II error rate (usually 0.80-0.95)
- **Attrition**: Add 10-20% for expected dropout

### 4. Variable Control

- **Independent variables**: What you manipulate
- **Dependent variables**: What you measure
- **Confounding variables**: What could bias results
- **Control strategies**: Randomization, blocking, matching, blinding

### 5. Randomization

- Simple randomization (coin flip)
- Block randomization (balanced groups)
- Stratified randomization (balance key covariates)
- Cluster randomization (group-level assignment)

### 6. Blinding

- Single-blind: Participants unaware of assignment
- Double-blind: Participants and researchers unaware
- Triple-blind: Including data analysts

## Reproducibility Checklist

- [ ] Protocol pre-registered (OSF, ClinicalTrials.gov, PROSPERO)
- [ ] All materials/reagents specified with catalog numbers
- [ ] Detailed step-by-step procedure written
- [ ] Statistical analysis plan pre-specified
- [ ] Data management plan documented
- [ ] Raw data sharing plan established
- [ ] Code availability ensured
- [ ] Sample size justified with power analysis
- [ ] Randomization method specified
- [ ] Blinding procedures documented
- [ ] Inclusion/exclusion criteria defined
- [ ] Primary endpoint pre-specified

## Output Format

```
## Experimental Design: [Title]

**Research Question**: [Clear question]
**Design Type**: [RCT/Factorial/etc.]

### Participants/Samples
- Population: [target population]
- Inclusion: [criteria]
- Exclusion: [criteria]
- Sample Size: N=[total] ([n] per group) — Power=[X], alpha=[X], effect=[X]

### Groups
- Experimental: [treatment description]
- Control: [control description]
- Blinding: [single/double/triple/none]

### Variables
- IV: [variables]
- DV: [primary + secondary outcomes]
- Controls: [confounds and how addressed]

### Procedure
1. [Step-by-step protocol]

### Analysis Plan
- Primary: [statistical test]
- Secondary: [additional analyses]
- Multiple comparison correction: [method]

### Timeline
- [Phase 1]: [duration]
- [Phase 2]: [duration]

### Ethics
- IRB/IACUC requirements: [details]
- Consent procedure: [details]
```
