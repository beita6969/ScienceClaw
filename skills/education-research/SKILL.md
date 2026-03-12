---
name: education-research
description: Supports education research including pedagogical method evaluation, learning analytics, assessment design, curriculum development analysis, and educational technology evaluation; trigger when users discuss teaching effectiveness, learning outcomes, educational interventions, or student performance data.
---

## When to Trigger

Activate this skill when the user mentions:
- Pedagogical methods, teaching strategies, instructional design
- Learning analytics, student performance data, LMS data
- Assessment design, test validity, reliability, item analysis
- Curriculum development, learning objectives, Bloom's taxonomy
- Educational technology, e-learning, blended learning, MOOCs
- Educational interventions, quasi-experimental designs in education
- Student engagement, motivation, self-regulated learning

## Step-by-Step Methodology

1. **Define the research question** - Specify the educational context (K-12, higher education, professional development). Identify the intervention, outcome measures, and comparison conditions. Frame using established educational theory (constructivism, connectivism, cognitive load theory).
2. **Study design** - Select appropriate design: RCT (gold standard but often impractical), quasi-experimental (difference-in-differences, regression discontinuity), or mixed methods. Address common challenges: nested data (students within classrooms), selection bias, contamination between groups.
3. **Assessment development** - Define learning objectives using Bloom's taxonomy (remember, understand, apply, analyze, evaluate, create). Develop assessment items aligned with objectives. Compute reliability (Cronbach's alpha, test-retest, inter-rater). Conduct item analysis (difficulty, discrimination index).
4. **Data collection** - Gather quantitative data (test scores, grades, completion rates, time-on-task from LMS logs) and qualitative data (surveys, interviews, observations, think-alouds). Ensure IRB approval for human subjects research.
5. **Multilevel analysis** - Use hierarchical linear modeling (HLM) to account for nested data structure (students within classrooms within schools). Report ICC (intraclass correlation) to justify multilevel approach. Include relevant covariates (prior achievement, demographics).
6. **Effect size and practical significance** - Report Cohen's d or Hedges' g for group comparisons. Use standards for education research: d = 0.2 (small), 0.4 (medium), 0.6 (large). Translate to months of learning gain for K-12 contexts (What Works Clearinghouse approach).
7. **Evidence synthesis** - Situate findings within existing evidence base. Reference systematic reviews (What Works Clearinghouse, EPPI-Centre, Campbell Collaboration). Discuss generalizability, implementation fidelity, and scalability.

## Key Databases and Tools

- **ERIC (Education Resources Information Center)** - Education literature database
- **What Works Clearinghouse (WWC)** - Evidence reviews of education programs
- **PISA / TIMSS / NAEP** - International and national assessment data
- **Google Scholar** - Cross-disciplinary search
- **R lme4 / HLM software** - Multilevel modeling
- **Canvas/Blackboard APIs** - LMS data extraction

## Output Format

- Study design diagram showing groups, timeline, and measurement points.
- Assessment statistics table: item number, difficulty, discrimination, point-biserial.
- Results table: outcome, groups, means/SDs, effect size (d), 95% CI, p-value.
- Multilevel model: fixed effects, random effects, ICC, variance explained.
- Practical significance translation: effect size to months of learning gain.

## Quality Checklist

- [ ] Learning objectives clearly defined using established taxonomy
- [ ] Assessment items aligned with stated learning objectives
- [ ] Nested data structure handled with appropriate multilevel model
- [ ] Effect sizes reported and interpreted in educationally meaningful terms
- [ ] Implementation fidelity documented (did the intervention happen as planned?)
- [ ] Threats to validity addressed (selection, maturation, testing effects)
- [ ] IRB approval obtained for human subjects research
- [ ] Practical significance distinguished from statistical significance
- [ ] Comparison to existing evidence base (WWC, systematic reviews)
