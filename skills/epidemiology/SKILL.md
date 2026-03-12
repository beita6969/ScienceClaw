---
name: epidemiology
description: Performs epidemiological analyses including disease modeling (SIR/SEIR), outbreak investigation, risk factor identification, incidence/prevalence estimation, and causal inference from observational data; trigger when users discuss disease spread, public health data, or population-level health patterns.
---

## When to Trigger

Activate this skill when the user mentions:
- SIR, SEIR, compartmental models, R0, reproduction number
- Outbreak investigation, contact tracing, epidemic curves
- Incidence, prevalence, mortality rates, case-fatality ratio
- Risk factors, odds ratio, relative risk, hazard ratio
- Cohort studies, case-control studies, cross-sectional surveys
- DAGs (directed acyclic graphs), causal inference, confounding
- Vaccine efficacy, herd immunity, attack rate

## Step-by-Step Methodology

1. **Define the epidemiological question** - Specify the disease/condition, population, time period, and geographic scope. Determine if descriptive, analytic, or modeling approach is needed.
2. **Data characterization** - Identify data source (surveillance, registry, survey). Assess case definitions (confirmed, probable, suspected). Check completeness and reporting biases.
3. **Descriptive epidemiology** - Characterize by person (age, sex, demographics), place (geographic distribution, mapping), and time (epidemic curves, secular trends, seasonality).
4. **Measure calculation** - Compute incidence rate (person-time denominator), prevalence (point or period), attack rate, case-fatality ratio. Report with 95% confidence intervals.
5. **Analytic methods** - For causal questions: draw a DAG to identify confounders and colliders. Use appropriate regression (logistic for OR, Poisson/negative binomial for rates, Cox for time-to-event). Apply propensity score methods if needed.
6. **Disease modeling** - Build SIR/SEIR compartmental models. Estimate R0 from early epidemic growth rate or next-generation matrix. Conduct sensitivity analysis on key parameters (transmission rate, recovery rate, latent period).
7. **Interpretation and communication** - Translate findings into public health actions. Present results with absolute and relative measures. Discuss Hills criteria for causation assessment.

## Key Databases and Tools

- **WHO Global Health Observatory** - International health statistics
- **CDC WONDER / MMWR** - US disease surveillance data
- **Our World in Data** - Pandemic and health metrics
- **GBD (Global Burden of Disease)** - Comprehensive disease burden estimates
- **EpiEstim / R0 package** - R0 estimation tools
- **DAGitty** - DAG drawing and analysis

## Output Format

- Epidemic curves with proper time axis (onset date, not report date when possible).
- Measures of association as tables: measure, point estimate, 95% CI, p-value.
- Compartmental model diagrams with parameter definitions and values.
- Geographic maps with rates (not raw counts) and appropriate denominators.

## Quality Checklist

- [ ] Case definition explicitly stated
- [ ] Denominators appropriate (person-time for rates, population for prevalence)
- [ ] Confidence intervals provided for all estimates
- [ ] Confounders identified via DAG and adjusted for
- [ ] Selection bias and information bias discussed
- [ ] Model assumptions stated and sensitivity analysis performed
- [ ] Absolute and relative measures both reported
- [ ] Temporal relationship between exposure and outcome verified
