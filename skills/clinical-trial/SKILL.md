---
name: clinical-trial
description: Designs and analyzes clinical trials including sample size calculation, randomization schemes, endpoint selection, CONSORT reporting, and interim analysis planning; trigger when users ask about RCTs, Phase I-IV trials, or clinical study design.
---

## When to Trigger

Activate this skill when the user mentions:
- Clinical trial design, RCT, randomized controlled trial
- Sample size calculation, power analysis for trials
- CONSORT, STROBE, SPIRIT guidelines
- Phase I, II, III, IV trials
- Primary/secondary endpoints, composite endpoints
- Interim analysis, adaptive trial design, futility
- Blinding, randomization, allocation concealment
- Intention-to-treat (ITT), per-protocol analysis

## Step-by-Step Methodology

1. **Define the research question** - Specify PICO (Population, Intervention, Comparator, Outcome). Determine if superiority, non-inferiority, or equivalence design is appropriate.
2. **Select trial phase and design** - Choose phase (I: safety/dose, II: efficacy signal, III: confirmatory, IV: post-market). Consider parallel, crossover, factorial, or adaptive designs.
3. **Primary endpoint selection** - Define primary outcome (must be clinically meaningful). Specify measurement timing and minimal clinically important difference (MCID).
4. **Sample size calculation** - Specify alpha (typically 0.05, two-sided), power (typically 80-90%), expected effect size, and dropout rate. Use appropriate formula for the endpoint type (continuous, binary, time-to-event).
5. **Randomization and blinding** - Recommend randomization method (simple, block, stratified, minimization). Specify blinding level (open-label, single, double, triple).
6. **Statistical analysis plan** - Pre-specify primary analysis method (t-test, chi-square, log-rank, mixed models). Define interim analysis schedule with alpha-spending function (O'Brien-Fleming, Lan-DeMets).
7. **Reporting** - Follow CONSORT for RCTs, STROBE for observational, SPIRIT for protocols. Include flow diagram, enrollment numbers, and all pre-specified analyses.

## Key Databases and Tools

- **ClinicalTrials.gov** - Trial registration and results
- **Cochrane Library** - Systematic reviews of trials
- **FDA / EMA guidance documents** - Regulatory requirements
- **nQuery / PASS / G*Power** - Sample size software
- **CONSORT / SPIRIT checklists** - Reporting standards

## Output Format

- Sample size as a table: assumptions, formula, per-arm and total N.
- Trial design as a structured summary: phase, design type, arms, blinding, duration.
- CONSORT flow diagram description with all participant numbers.
- Statistical analysis plan with pre-specified primary and secondary analyses.

## Quality Checklist

- [ ] PICO clearly defined
- [ ] Primary endpoint is clinically meaningful and measurable
- [ ] Sample size assumptions explicitly stated with references
- [ ] Dropout rate factored into sample size
- [ ] Randomization and blinding method specified
- [ ] Multiple comparison adjustment for secondary endpoints
- [ ] Interim analysis boundaries pre-specified if applicable
- [ ] ITT and per-protocol populations defined
- [ ] Regulatory guideline alignment noted (FDA/EMA/ICH)
