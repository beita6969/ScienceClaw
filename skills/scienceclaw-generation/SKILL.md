---
name: scienceclaw-generation
description: "Generate scientific hypotheses, experimental designs, and paper drafts. Use when: user asks to propose hypotheses, design experiments, or write scientific content. NOT for: data analysis or literature search."
metadata: { "openclaw": { "emoji": "💡" } }
---

# Scientific Generation Skill

Generate hypotheses, experimental designs, and scientific writing across all disciplines.

## When to Use

- "Propose hypotheses for this research question"
- "Design an experiment to test..."
- "Draft a methods section for..."
- "Generate research questions for this topic"
- "Write an abstract for these findings"
- Planning new research directions

## When NOT to Use

- Running data analysis (use code-execution + scipy-analysis)
- Literature searching (use literature-search)
- Verifying claims (use scienceclaw-verification)
- Pure information extraction (use scienceclaw-ie)

## Generation Types

### 1. Hypothesis Generation
Follow the structured workflow:
1. **Observation**: State the observed phenomenon or gap
2. **Literature Context**: Reference existing knowledge and gaps
3. **Hypothesis Statement**: Formulate as testable H0/H1
4. **Variables**: Identify independent, dependent, and control variables
5. **Predictions**: State specific, measurable predictions
6. **Falsifiability**: Explain what would disprove the hypothesis
7. **Novelty Assessment**: Rate novelty (incremental/moderate/transformative)

Format: "If [independent variable] then [predicted effect on dependent variable] because [mechanism/rationale]"

### 2. Experimental Design
Include all components:
- **Objective**: Clear research question
- **Design Type**: RCT, factorial, quasi-experimental, etc.
- **Sample**: Size calculation (power analysis), selection criteria, randomization
- **Variables**: IV, DV, controls, confounds
- **Protocol**: Step-by-step procedure
- **Analysis Plan**: Statistical tests, significance thresholds
- **Ethics**: IRB/IACUC considerations
- **Reproducibility Checklist**: Materials, data sharing, pre-registration

### 3. Scientific Writing
Support all IMRaD sections:
- **Introduction**: Background, gap, objective, significance
- **Methods**: Detailed, reproducible protocol
- **Results**: Findings with statistical reporting
- **Discussion**: Interpretation, limitations, implications
- **Abstract**: Structured summary (Background, Methods, Results, Conclusions)

### 4. Research Question Generation
From a broad topic, generate:
- Descriptive questions (What/How/When)
- Comparative questions (differences between groups)
- Correlational questions (relationships between variables)
- Causal questions (cause-effect with mechanisms)

## Quality Criteria

All generated content must:
1. Be grounded in existing scientific knowledge
2. Use discipline-appropriate terminology
3. Be specific and testable (for hypotheses)
4. Include feasibility assessment
5. Consider ethical implications
6. Acknowledge limitations and assumptions
7. Cite relevant foundational work when possible

## Citation Format

When referencing prior work in generated content, use:
- Inline: (Author et al., Year) or [DOI]
- Note which citations need verification
- Distinguish confirmed vs. suggested references
