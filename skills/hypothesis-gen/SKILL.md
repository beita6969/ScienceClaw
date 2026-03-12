---
name: hypothesis-gen
description: "Structured hypothesis generation workflow. Use when: user needs to formulate testable scientific hypotheses from observations, gaps, or preliminary data. NOT for: testing hypotheses or running experiments."
metadata: { "openclaw": { "emoji": "🧪" } }
---

# Hypothesis Generation Skill

Structured workflow for generating testable scientific hypotheses.

## When to Use

- "Generate hypotheses for this research question"
- "What could explain this observation?"
- "Propose testable ideas based on this data"
- "Help me formulate H0 and H1"
- "What hypotheses does this gap suggest?"

## When NOT to Use

- Testing/verifying hypotheses (use scienceclaw-verification)
- Designing experiments (use experimental-design)
- Literature searching (use literature-search)
- Writing full papers (use paper-writing)

## Generation Workflow

### Step 1: Observe
Identify the phenomenon, anomaly, or gap:
- What was observed?
- What is unexpected or unexplained?
- What contradicts existing theory?
- What data pattern needs explanation?

### Step 2: Contextualize
Ground in existing literature:
- What do current theories predict?
- What related findings exist?
- Where are the knowledge gaps?
- What alternative explanations exist?

### Step 3: Formulate
State the hypothesis formally:

**Template**: "If [independent variable/condition], then [predicted effect on dependent variable], because [proposed mechanism]."

**Null Hypothesis (H0)**: No effect / no difference / no relationship
**Alternative Hypothesis (H1)**: The predicted effect exists
**Directional**: Specify direction (increase/decrease) when justified

### Step 4: Evaluate
Score each hypothesis on:

| Criterion | Score (1-5) | Description |
|-----------|-------------|-------------|
| **Testability** | _ | Can be experimentally tested? |
| **Falsifiability** | _ | Can be proven wrong? |
| **Novelty** | _ | How new is this idea? |
| **Mechanism** | _ | Is the proposed mechanism plausible? |
| **Feasibility** | _ | Can current methods test it? |
| **Impact** | _ | How significant if confirmed? |

### Step 5: Prioritize
Rank hypotheses by:
- Total evaluation score
- Risk-reward ratio (impact / feasibility)
- Alignment with available resources
- Potential for publication

## Output Format

```
## Hypothesis [N]: [Short title]

**Statement**: If [condition], then [prediction], because [mechanism].
**H0**: [Null hypothesis]
**H1**: [Alternative hypothesis]

**Variables**:
- Independent: [variable]
- Dependent: [variable]
- Controls: [variables to hold constant]

**Evaluation**: Testability=[X] Falsifiability=[X] Novelty=[X] Mechanism=[X] Feasibility=[X] Impact=[X]
**Priority Score**: [Total/30]

**Key References**: [relevant citations]
**Suggested Test**: [brief experimental approach]
```

## Quality Checklist

- [ ] Hypothesis is specific and testable
- [ ] Variables are clearly identified
- [ ] Mechanism is plausible given known science
- [ ] Null hypothesis is properly stated
- [ ] Predictions are measurable
- [ ] At least one path to falsification exists
- [ ] Novel relative to existing literature
- [ ] Ethical considerations addressed
