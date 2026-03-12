---
name: meta-analysis
description: "Perform quantitative meta-analysis with effect size calculation, forest plots, funnel plots, and heterogeneity assessment. Use when: user asks to combine results from multiple studies, calculate pooled effect sizes, assess publication bias, or create forest/funnel plots. NOT for: systematic review protocol (use systematic-review) or single-study statistics (use statsmodels-stats)."
metadata: { "openclaw": { "emoji": "🌲" } }
---

# Meta-Analysis

Quantitative synthesis of results from multiple studies. Calculates pooled effect sizes, assesses heterogeneity, detects publication bias, and generates forest and funnel plots.

## When to Use

- "Combine these study results into a meta-analysis"
- "Calculate the pooled odds ratio from these trials"
- "Create a forest plot of these effect sizes"
- "Test for publication bias with a funnel plot"
- "What's the heterogeneity (I²) across these studies?"
- "Run a random-effects meta-analysis"

## When NOT to Use

- Designing a systematic review protocol (use systematic-review)
- Searching for studies (use literature-search)
- Single-study statistical analysis (use statsmodels-stats)
- Narrative literature review (use paper-writing)

## Effect Size Types

| Outcome Type | Effect Size | Formula | Use When |
|-------------|-------------|---------|----------|
| Continuous | SMD (Cohen's d / Hedges' g) | $(M_1 - M_2) / S_p$ | Comparing means across studies with different scales |
| Continuous | Mean Difference (MD) | $M_1 - M_2$ | Same outcome measure across all studies |
| Binary | Odds Ratio (OR) | $(a \times d) / (b \times c)$ | Case-control studies, binary outcomes |
| Binary | Risk Ratio (RR) | $(a/(a+b)) / (c/(c+d))$ | Cohort studies, clinical trials |
| Binary | Risk Difference (RD) | $R_1 - R_2$ | Absolute risk reduction |
| Time-to-event | Hazard Ratio (HR) | From Cox model | Survival analysis |
| Correlation | Fisher's z | $0.5 \ln((1+r)/(1-r))$ | Correlation studies |

## Core Analysis with Python

### Random-Effects Meta-Analysis

```python
import numpy as np
from scipy import stats

def meta_analysis_random_effects(effects, variances, study_names=None):
    """
    DerSimonian-Laird random-effects meta-analysis.

    Args:
        effects: array of effect sizes (log-OR, SMD, etc.)
        variances: array of within-study variances
        study_names: optional list of study labels

    Returns:
        dict with pooled estimate, CI, heterogeneity stats
    """
    effects = np.array(effects, dtype=float)
    variances = np.array(variances, dtype=float)
    k = len(effects)

    # Fixed-effect weights
    w_fe = 1.0 / variances
    pooled_fe = np.sum(w_fe * effects) / np.sum(w_fe)

    # Cochran's Q
    Q = np.sum(w_fe * (effects - pooled_fe) ** 2)
    df = k - 1
    p_heterogeneity = 1 - stats.chi2.cdf(Q, df)

    # tau-squared (DerSimonian-Laird)
    C = np.sum(w_fe) - np.sum(w_fe ** 2) / np.sum(w_fe)
    tau2 = max(0, (Q - df) / C)

    # I-squared
    I2 = max(0, (Q - df) / Q * 100) if Q > 0 else 0

    # Random-effects weights
    w_re = 1.0 / (variances + tau2)
    pooled_re = np.sum(w_re * effects) / np.sum(w_re)
    se_pooled = np.sqrt(1.0 / np.sum(w_re))

    ci_lower = pooled_re - 1.96 * se_pooled
    ci_upper = pooled_re + 1.96 * se_pooled
    z = pooled_re / se_pooled
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        'pooled_effect': pooled_re,
        'se': se_pooled,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'z': z,
        'p_value': p_value,
        'tau2': tau2,
        'I2': I2,
        'Q': Q,
        'Q_df': df,
        'Q_p': p_heterogeneity,
        'k': k,
        'model': 'DerSimonian-Laird random-effects'
    }
```

### Forest Plot

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def forest_plot(effects, ci_lower, ci_upper, study_names, pooled, pooled_ci,
                xlabel='Effect Size', title='Forest Plot', output_path='forest_plot.png'):
    """Generate a publication-quality forest plot."""
    k = len(effects)
    fig, ax = plt.subplots(figsize=(8, max(4, k * 0.4 + 2)))

    y_positions = list(range(k, 0, -1))

    # Individual studies
    for i, y in enumerate(y_positions):
        ax.plot(effects[i], y, 'ks', markersize=8)
        ax.plot([ci_lower[i], ci_upper[i]], [y, y], 'k-', linewidth=1.5)

    # Pooled estimate (diamond)
    diamond_y = 0
    diamond_half_h = 0.3
    diamond = plt.Polygon([
        [pooled_ci[0], diamond_y],
        [pooled, diamond_y + diamond_half_h],
        [pooled_ci[1], diamond_y],
        [pooled, diamond_y - diamond_half_h]
    ], closed=True, facecolor='steelblue', edgecolor='black')
    ax.add_patch(diamond)

    # Reference line at null effect
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)

    # Labels
    yticks = y_positions + [diamond_y]
    ylabels = study_names + ['Pooled']
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_ylim(-1, k + 1.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Forest plot saved: {output_path}")
    return fig
```

### Funnel Plot (Publication Bias)

```python
def funnel_plot(effects, se_values, pooled_effect,
                xlabel='Effect Size', output_path='funnel_plot.png'):
    """Generate a funnel plot to assess publication bias."""
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(effects, se_values, c='black', s=30, zorder=3)

    # Pseudo-confidence region
    se_range = np.linspace(0.001, max(se_values) * 1.1, 100)
    ci_low = pooled_effect - 1.96 * se_range
    ci_high = pooled_effect + 1.96 * se_range
    ax.fill_betweenx(se_range, ci_low, ci_high, alpha=0.1, color='gray')
    ax.axvline(pooled_effect, color='red', linestyle='--', linewidth=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Standard Error')
    ax.set_title('Funnel Plot')
    ax.invert_yaxis()  # Convention: smaller SE at top

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Funnel plot saved: {output_path}")
    return fig
```

## Publication Bias Tests

### Egger's Regression Test
```python
def egger_test(effects, se_values):
    """Egger's test for funnel plot asymmetry."""
    precision = 1.0 / np.array(se_values)
    standardized = np.array(effects) / np.array(se_values)
    slope, intercept, r, p, se = stats.linregress(precision, standardized)
    return {'intercept': intercept, 'se': se, 'p_value': p,
            'interpretation': 'Significant asymmetry' if p < 0.10 else 'No significant asymmetry'}
```

### Begg's Rank Correlation Test
```python
def begg_test(effects, variances):
    """Begg-Mazumdar rank correlation test."""
    standardized = effects / np.sqrt(variances)
    tau, p = stats.kendalltau(standardized, variances)
    return {'tau': tau, 'p_value': p}
```

### Trim-and-Fill Method
- Identifies and imputes missing studies from funnel plot asymmetry
- Re-estimates the pooled effect including imputed studies
- Use `statsmodels` or `metafor` (R) for implementation

## Heterogeneity Interpretation

| I² Value | Interpretation |
|----------|---------------|
| 0-25% | Low heterogeneity |
| 25-50% | Moderate heterogeneity |
| 50-75% | Substantial heterogeneity |
| 75-100% | Considerable heterogeneity |

When I² > 50%, investigate sources:
1. **Subgroup analysis**: Split by study design, population, intervention dose
2. **Meta-regression**: Model effect size as function of study-level covariates
3. **Sensitivity analysis**: Leave-one-out, exclude high risk-of-bias studies

## Reporting Standards

Follow PRISMA 2020 for reporting meta-analyses. Include:
1. Number of studies (k) and total participants (N)
2. Pooled effect size with 95% CI
3. Heterogeneity: Q statistic (df, p), I², tau²
4. Model type: fixed-effect vs. random-effects with justification
5. Publication bias assessment results
6. Forest plot and funnel plot as figures

## Best Practices

1. Use random-effects model by default (studies rarely share a true common effect)
2. Always report both Q and I² for heterogeneity
3. Log-transform ORs and RRs before pooling; back-transform for reporting
4. Use Hedges' g rather than Cohen's d for small-sample correction
5. Minimum 5-10 studies for reliable funnel plot interpretation
6. Never fabricate study data or effect sizes

## Zero-Hallucination Rule

- ALL study-level data must come from tool results or user-provided data
- NEVER generate fictional study names, sample sizes, or effect sizes
- If insufficient data for meta-analysis, say so explicitly
