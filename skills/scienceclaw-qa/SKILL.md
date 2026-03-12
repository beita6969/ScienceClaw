---
name: scienceclaw-qa
description: "Answer scientific questions across all disciplines with evidence-based responses and citations. Use when: (1) user asks factual science questions, (2) needs explanation of concepts/theories/methods, (3) multi-step scientific reasoning needed. Covers natural sciences (physics, chemistry, biology, medicine, materials, astronomy, earth science, math, CS) and social sciences (economics, sociology, psychology, political science, linguistics, history, law, philosophy, education). NOT for: opinion-based questions, non-scientific queries, or when code execution is needed (use code-execution skill)."
metadata: { "openclaw": { "emoji": "🔬" } }
---

# ScienceCLAW QA

Answer scientific questions with evidence-based responses, structured methodology, and proper citations across all disciplines.

## When to Use

Use this skill when the user:

- Asks factual questions about any scientific discipline (natural or social sciences)
- Needs explanation of scientific concepts, theories, laws, or methods
- Requests comparison of competing scientific hypotheses or models
- Wants to understand experimental results or published findings
- Needs multi-step scientific reasoning to arrive at an answer
- Asks "how does X work" or "why does X happen" questions rooted in science
- Requests definitions, classifications, or taxonomies of scientific terms
- Needs help interpreting data, graphs, or scientific figures (without code execution)

## When NOT to Use

Do not use this skill when:

- The question is opinion-based or subjective (e.g., "Is physics better than biology?")
- The user needs code execution for data analysis (use the code-execution skill)
- The user needs a literature search or paper retrieval (use the scienceclaw-retrieval skill)
- The question is about non-scientific topics (pop culture, personal advice, etc.)
- The user wants summarization of a specific paper (use scienceclaw-summarization)
- The user needs structured information extraction from text (use scienceclaw-ie)
- The user needs formal proof construction or complex causal inference (use scienceclaw-reasoning)

## Methodology

Follow this four-step process for every scientific QA task:

### Step 1: Identify Discipline and Scope

Determine the primary discipline and any cross-disciplinary dimensions of the question. Consult the discipline taxonomy in `references/discipline-taxonomy.md` for classification guidance.

- Map the question to one or more of the 17 supported disciplines
- Identify the subdiscipline(s) most relevant to the question
- Note if the question spans multiple disciplines (e.g., biophysics, neuroeconomics)
- Determine the depth of answer expected (introductory, intermediate, expert)

### Step 2: Retrieve Evidence

Gather supporting evidence before constructing the answer.

- Use the **literature-search** skill to find relevant primary sources, review articles, and textbook references
- Prioritize peer-reviewed sources and authoritative references
- For rapidly evolving fields, check recency of evidence (prefer sources from the last 5 years unless historical context is needed)
- For contested topics, retrieve evidence from multiple perspectives

### Step 3: Synthesize Answer

Construct the answer by integrating evidence from multiple sources.

- Begin with a direct, concise answer to the question
- Provide supporting explanation with appropriate depth
- Include relevant context (historical development, current consensus, open questions)
- Use discipline-appropriate terminology with definitions for technical terms
- Present numerical values with units and uncertainty where applicable
- Acknowledge limitations, caveats, or areas of active debate

### Step 4: Cite Sources

Provide citations for all factual claims.

- Use inline citations with author-year format: (Author et al., Year)
- Include a references section at the end of the answer
- Prefer DOI links when available
- Distinguish between primary research, review articles, and textbook references
- For well-established facts, cite authoritative textbooks or review articles rather than original papers

## Discipline-Specific QA Patterns

### Natural Science Approach

For physics, chemistry, biology, medicine, materials science, astronomy, earth science, mathematics, and computer science:

- **Quantitative emphasis**: Include equations, numerical values, units, and error bounds
- **Mechanistic explanation**: Describe underlying mechanisms and causal chains
- **Experimental grounding**: Reference key experiments that established the finding
- **Model-based reasoning**: Explain which theoretical models apply and their limitations
- **Scale awareness**: Specify the spatial, temporal, or energy scales at which answers apply

Example pattern for natural science questions:

```
1. State the direct answer with key quantitative values
2. Explain the underlying mechanism or theory
3. Reference the foundational experiment(s) or derivation
4. Note boundary conditions, assumptions, and limitations
5. Cite sources (textbooks for established facts, papers for recent findings)
```

### Social Science Approach

For economics, sociology, psychology, political science, linguistics, history, law, philosophy, and education:

- **Context sensitivity**: Specify cultural, temporal, and geographic scope of findings
- **Effect sizes and significance**: Report statistical effect sizes, not just p-values
- **Methodological transparency**: Note the study designs behind cited evidence (RCT, observational, meta-analysis)
- **Theoretical pluralism**: Present competing theoretical frameworks where relevant
- **Normative vs. positive distinction**: Clearly separate empirical findings from value judgments

Example pattern for social science questions:

```
1. State the current consensus or dominant finding
2. Describe the evidence base (study types, sample sizes, replication status)
3. Present alternative theoretical interpretations
4. Note methodological limitations and external validity concerns
5. Cite meta-analyses and systematic reviews where available
```

## Citation Format Requirements

### Inline Citations

Use author-year format within the text:

- Single author: (Smith, 2023)
- Two authors: (Smith & Jones, 2022)
- Three or more: (Smith et al., 2021)
- Multiple citations: (Smith, 2023; Jones et al., 2022)
- Direct quote: (Smith, 2023, p. 45)

### Reference List Format

```
Author, A. B., & Author, C. D. (Year). Title of the article. Journal Name, Volume(Issue), Pages. https://doi.org/xxxxx

Author, A. B. (Year). Title of the Book (Edition). Publisher.
```

### Source Priority

1. Systematic reviews and meta-analyses
2. Peer-reviewed primary research articles
3. Authoritative textbooks and handbooks
4. Institutional reports (WHO, IPCC, National Academies)
5. Preprints (flag as non-peer-reviewed)

## Multi-Source Evidence Synthesis

When answering questions that require integration of multiple sources:

- **Convergence assessment**: Identify where multiple independent sources agree
- **Discrepancy resolution**: When sources conflict, explain why (different methods, populations, time periods)
- **Evidence weighting**: Give more weight to higher-quality evidence (meta-analyses > single studies, RCTs > observational)
- **Confidence calibration**: Express confidence level in the answer based on evidence quality and consensus
- **Knowledge frontier**: Clearly mark where established knowledge ends and speculation begins

Use confidence markers:

- **Well-established**: Broad scientific consensus, replicated findings
- **Probable**: Strong evidence but some uncertainty or ongoing refinement
- **Emerging**: Recent findings, limited replication, active research area
- **Speculative**: Theoretical predictions or preliminary results, not yet validated

## Cross-Discipline QA Bridging

Many questions span multiple disciplines. Handle these by:

1. **Identify the bridge**: Name the interdisciplinary field if one exists (e.g., bioinformatics, econophysics, cognitive linguistics)
2. **Layer the explanation**: Start with the discipline closest to the user's question, then bring in complementary perspectives
3. **Reconcile terminology**: When different disciplines use different terms for the same concept, note the equivalences
4. **Integrate methods**: Describe how methods from different fields complement each other in addressing the question
5. **Acknowledge gaps**: Note where disciplinary boundaries create blind spots in the available evidence

## Response Structure Template

```markdown
## Answer

[Direct, concise answer to the question]

## Explanation

[Detailed explanation with appropriate depth]

### Key Concepts
- [Concept 1]: [Definition and relevance]
- [Concept 2]: [Definition and relevance]

### Evidence Base
[Summary of supporting evidence with inline citations]

### Limitations and Open Questions
[Caveats, boundary conditions, and areas of active research]

## References
[Formatted reference list]
```

## Integration with Other Skills

- **literature-search**: Always invoke for evidence retrieval before answering non-trivial questions
- **scienceclaw-reasoning**: Delegate to this skill when the question requires formal proof, causal inference, or extended logical argumentation
- **scienceclaw-ie**: Use when the answer requires extracting structured data from scientific texts
- **code-execution**: Redirect when the user needs computational analysis, plotting, or simulation
- **scienceclaw-summarization**: Redirect when the user wants a summary of a specific paper rather than a general QA response

## Zero-Hallucination Rule

ALL factual claims, citations, database results, and scientific data presented to the user MUST come from actual tool results (API calls, code execution, web search) in this conversation. NEVER fabricate or "fill in" details from training data. If a tool returns no results or partial data, report exactly what happened.
