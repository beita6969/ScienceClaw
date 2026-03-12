---
name: scienceclaw-reasoning
description: "Perform multi-step scientific reasoning, proof construction, causal inference, and logical argumentation. Use when: (1) deriving conclusions from premises, (2) causal analysis, (3) mathematical proofs, (4) hypothesis evaluation, (5) counterfactual reasoning. NOT for: simple factual questions (use scienceclaw-qa), data analysis (use code-execution), or literature search (use scienceclaw-retrieval)."
metadata: { "openclaw": { "emoji": "🧠" } }
---

# ScienceCLAW Reasoning

Perform multi-step scientific reasoning, proof construction, causal inference, and logical argumentation across all scientific disciplines.

## When to Use

Use this skill when the user:

- Needs to derive conclusions from a set of premises or observations
- Asks for a mathematical proof or formal derivation
- Requires causal analysis (identifying causes, effects, confounders)
- Wants to evaluate competing hypotheses against available evidence
- Needs counterfactual reasoning ("what would happen if X were different?")
- Asks for logical argumentation supporting or refuting a scientific claim
- Requires chain-of-thought reasoning through a multi-step scientific problem
- Needs to assess the validity of a scientific argument or experimental design

## When NOT to Use

Do not use this skill when:

- The user asks a simple factual question (use scienceclaw-qa)
- The user needs data analysis or computation (use code-execution)
- The user needs literature search or paper retrieval (use scienceclaw-retrieval)
- The user wants text summarization (use scienceclaw-summarization)
- The user needs information extraction from documents (use scienceclaw-ie)

## Chain-of-Thought Templates

All reasoning tasks should follow explicit chain-of-thought patterns. Make every step visible and verifiable.

### General Reasoning Template

```
Step 1: STATE the problem clearly
  - Identify what is given (premises, data, constraints)
  - Identify what is asked (conclusion, proof, evaluation)
  - Identify the reasoning type needed

Step 2: PLAN the reasoning path
  - Select the appropriate reasoning framework
  - Identify intermediate steps needed
  - Note potential pitfalls or branching points

Step 3: EXECUTE each reasoning step
  - Show each inference explicitly
  - Justify each step with a rule, law, or principle
  - Flag assumptions made at each step

Step 4: VALIDATE the conclusion
  - Check for logical consistency
  - Verify against known constraints
  - Test with edge cases or counterexamples

Step 5: COMMUNICATE the result
  - State the conclusion clearly
  - Summarize the key reasoning path
  - Note confidence level and limitations
```

### Hypothesis Evaluation Template

```
1. STATE each hypothesis precisely
2. IDENTIFY observable predictions that differ between hypotheses
3. COMPARE predictions against available evidence
4. ASSESS fit: which hypothesis explains more evidence with fewer assumptions?
5. CHECK for confounds or alternative explanations
6. CONCLUDE with ranked hypotheses and confidence levels
```

### Counterfactual Reasoning Template

```
1. SPECIFY the counterfactual condition ("If X had been Y instead of Z...")
2. IDENTIFY the causal model connecting X to downstream outcomes
3. TRACE the causal chain forward from the altered condition
4. COMPARE the counterfactual outcome with the actual outcome
5. ASSESS sensitivity: how robust is the counterfactual conclusion?
```

## Formal vs. Informal Reasoning

### Formal Reasoning

Use formal reasoning when the domain supports it (mathematics, logic, theoretical physics, formal linguistics):

- **Deductive proofs**: From axioms and rules of inference to theorem
- **Algebraic derivation**: Step-by-step manipulation of equations with justification
- **Logical formalization**: Translate natural language claims into propositional or predicate logic
- **Set-theoretic arguments**: Use set notation for classification and inclusion/exclusion reasoning

Notation conventions:

- Use standard mathematical notation (LaTeX-style where supported)
- Number each step and reference prior steps explicitly
- Mark axioms, definitions, lemmas, and theorems
- Clearly distinguish between definitions (":=") and equalities ("=")
- Use QED or similar markers to indicate proof completion

### Informal Reasoning

Use informal reasoning when formal methods are impractical (most empirical sciences, social sciences):

- **Abductive reasoning**: Inference to the best explanation from observed data
- **Analogical reasoning**: Drawing parallels from well-understood domains to less understood ones
- **Narrative causal reasoning**: Constructing plausible causal stories grounded in evidence
- **Bayesian updating**: Qualitative or semi-quantitative updating of beliefs given new evidence

Quality criteria for informal reasoning:

- Every claim must be supported by evidence or a stated assumption
- Alternative explanations must be considered and addressed
- The strength of each inference must be indicated (certain, likely, possible, speculative)
- Logical fallacies must be avoided and called out if present in the source material

## Integration with Code for Verification

When reasoning can be verified computationally, recommend or invoke code execution:

### Verification Scenarios

| Reasoning Type | Code Verification |
|---|---|
| Mathematical proof | Symbolic computation (SymPy) to verify algebraic steps |
| Statistical inference | Monte Carlo simulation to validate analytical results |
| Causal claim | DAG analysis with DoWhy or similar causal inference libraries |
| Optimization argument | Numerical optimization to confirm analytical solution |
| Combinatorial argument | Exhaustive enumeration for small cases |
| Differential equation | Numerical integration to verify analytical solution |

### Verification Protocol

1. Complete the reasoning chain first (do not rely on code as the primary method)
2. Identify which steps are amenable to computational verification
3. Specify the verification approach and expected outcome
4. If code-execution skill is available, invoke it for verification
5. Reconcile any discrepancies between analytical reasoning and numerical results

## Discipline-Specific Reasoning Patterns

### Mathematical Proofs

Structure for mathematical reasoning:

- **Direct proof**: Assume premises, derive conclusion through valid inference steps
- **Proof by contradiction**: Assume negation of conclusion, derive a contradiction
- **Proof by induction**: Base case, inductive hypothesis, inductive step
- **Proof by construction**: Exhibit an explicit example satisfying the claim
- **Proof by exhaustion**: Enumerate all cases and verify each

Requirements:

- State the theorem or claim precisely before beginning the proof
- Define all notation and variables at the start
- Each step must follow from previous steps by a named rule or previously proven result
- Clearly mark the end of the proof

Example structure:

```
**Theorem**: [Statement]

**Proof**:
Let [variable definitions].
By [axiom/definition], we have [step 1].
From [step 1] and [known result], it follows that [step 2].
...
Therefore, [conclusion]. QED
```

### Causal Inference for Social Science

Apply the potential outcomes framework or structural causal models:

- **Identify the causal question**: What is the treatment? What is the outcome?
- **State the causal model**: Draw or describe the DAG (directed acyclic graph)
- **Identify confounders**: Variables that affect both treatment and outcome
- **Select identification strategy**: Randomization, instrumental variables, difference-in-differences, regression discontinuity, matching, or synthetic controls
- **Assess assumptions**: SUTVA, ignorability, positivity, exclusion restriction
- **Interpret effect estimates**: Distinguish ATE, ATT, LATE, and ITT

Key causal reasoning pitfalls to flag:

- Confusing correlation with causation
- Collider bias (conditioning on a common effect)
- Survivorship bias (analyzing a selected sample)
- Simpson's paradox (aggregation reversal)
- Post-treatment bias (controlling for mediators)
- Ecological fallacy (inferring individual effects from group data)

### Mechanistic Reasoning for Natural Science

Trace physical, chemical, or biological mechanisms:

- **Identify the system**: Define boundaries, components, and interactions
- **Specify initial conditions**: Starting state of the system
- **Apply governing laws**: Conservation laws, rate equations, thermodynamic principles
- **Trace the mechanism step by step**: Each step should invoke a specific physical/chemical/biological principle
- **Predict the outcome**: Derive the expected end state
- **Quantify where possible**: Include magnitudes, timescales, and energy scales

Mechanistic reasoning quality checks:

- Is each step physically realizable (does it respect conservation laws)?
- Are the timescales consistent across steps?
- Does the mechanism account for competing pathways?
- Are boundary conditions and approximations stated?

## Reasoning Quality Assurance

### Logical Validity Checks

Before finalizing any reasoning chain, verify:

- **No circular reasoning**: The conclusion does not appear among the premises
- **No equivocation**: Terms are used consistently throughout
- **No false dichotomy**: All relevant alternatives are considered
- **No hasty generalization**: Conclusions are proportionate to the evidence
- **No appeal to authority**: Claims are justified by evidence, not by who said them
- **Modus ponens integrity**: If P then Q; P; therefore Q (verify both the conditional and the antecedent)

### Assumption Tracking

Maintain an explicit list of assumptions throughout the reasoning:

```
Assumptions:
  A1: [Description] - [Justification or "assumed for simplicity"]
  A2: [Description] - [Justification or "standard in this field"]
  ...

Sensitivity: Conclusion is robust to relaxation of A1 but sensitive to A2.
```

### Confidence Calibration

Rate the overall reasoning confidence:

- **Certain**: Deductively valid from well-established premises
- **High confidence**: Strong evidence, standard methods, limited assumptions
- **Moderate confidence**: Good evidence but some assumptions or gaps
- **Low confidence**: Preliminary evidence, strong assumptions, or novel reasoning
- **Speculative**: Exploratory reasoning, not yet validated

## Response Structure Template

```markdown
## Problem Statement

[Clear restatement of the reasoning task]

## Reasoning Framework

[Selected approach and justification]

## Reasoning Chain

### Step 1: [Description]
[Detailed reasoning with justification]

### Step 2: [Description]
[Detailed reasoning with justification]

...

## Conclusion

[Clear statement of the derived result]

## Assumptions and Limitations

[Explicit list of assumptions and sensitivity analysis]

## Verification

[Computational verification results or recommendations]

## Confidence: [Level]

[Brief justification of confidence rating]
```

## Integration with Other Skills

- **scienceclaw-qa**: Receive questions that require reasoning beyond simple recall; return reasoned answers
- **code-execution**: Invoke for computational verification of analytical results
- **literature-search**: Retrieve evidence needed as premises for reasoning chains
- **scienceclaw-ie**: Extract structured data from texts to serve as reasoning inputs
