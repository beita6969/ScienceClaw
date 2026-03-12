---
name: scienceclaw-discovery
description: "Identify research gaps, synthesize cross-disciplinary insights, and generate novel hypotheses. Use when: user asks about unexplored areas, cross-field connections, or new research directions. NOT for: routine literature review or data analysis."
metadata: { "openclaw": { "emoji": "🔬" } }
---

# Scientific Discovery Skill

Identify research gaps, synthesize cross-disciplinary knowledge, and facilitate novel scientific discovery.

## When to Use

- "What are the open problems in this field?"
- "Are there connections between X and Y research areas?"
- "What's unexplored in this space?"
- "How could findings from field A apply to field B?"
- "Identify novel research directions"
- Cross-disciplinary brainstorming

## When NOT to Use

- Standard literature search (use literature-search)
- Writing a review paper (use scienceclaw-summarization + paper-writing)
- Routine hypothesis generation (use scienceclaw-generation)
- Fact-checking (use scienceclaw-verification)

## Discovery Modes

### 1. Research Gap Identification
Systematic approach:
1. **Map existing knowledge** — What is well-established?
2. **Identify contradictions** — Where do studies disagree?
3. **Find under-explored areas** — What has limited evidence?
4. **Detect methodology gaps** — What approaches haven't been tried?
5. **Note population/context gaps** — Where is evidence missing?
6. **Assess translational gaps** — What basic science lacks clinical application?

Output format:
```
**Gap**: [Description]
**Evidence**: [What's known vs. unknown]
**Significance**: [Why this matters]
**Feasibility**: [How difficult to address]
**Suggested Approach**: [How to fill the gap]
```

### 2. Cross-Disciplinary Synthesis
Find unexpected connections between fields:
- **Analogical reasoning**: Similar mechanisms in different domains
- **Method transfer**: Applying techniques from one field to another
- **Concept bridging**: Shared theoretical frameworks
- **Data reuse**: Existing datasets applicable to new questions
- **Tool adaptation**: Instruments/software transferable across fields

### 3. Novelty Assessment
Evaluate how novel a research direction is:
- **Incremental**: Small extension of existing work
- **Combinatorial**: New combination of known elements
- **Transformative**: Paradigm-shifting potential
- **Disruptive**: Could change fundamental understanding

Score on axes:
- Novelty (1-5)
- Feasibility (1-5)
- Impact potential (1-5)
- Risk level (1-5)

### 4. Serendipity Engine
Structured approach to unexpected discoveries:
1. Present findings from unrelated fields
2. Identify structural or functional analogies
3. Propose testable connections
4. Evaluate plausibility against known constraints
5. Suggest minimal experiments to validate

## Discovery Workflow

```
Observe anomaly/gap
    → Search across disciplines
    → Identify analogies/connections
    → Formulate novel hypothesis
    → Assess novelty + feasibility
    → Design validation experiment
    → Document for peer review
```

## Quality Criteria

1. **Grounded novelty** — New ideas must build on solid existing knowledge
2. **Cross-validation** — Check proposed connections against multiple sources
3. **Mechanism plausibility** — Proposed links should have a plausible mechanism
4. **Testability** — Discoveries must lead to testable predictions
5. **Ethical consideration** — Flag dual-use or sensitive research directions
6. **Reproducibility** — Ensure discovery process can be documented and repeated

## Anti-Patterns to Avoid

- Superficial analogies without mechanistic basis
- Ignoring negative evidence or contradictions
- Over-claiming novelty for well-known connections
- Proposing untestable or unfalsifiable hypotheses
- Discipline-centric bias (favoring one field over another)

## Knowledge Graph-Aided Discovery

Use **networkx-social** (enhanced with knowledge graph features) to build and analyze research knowledge graphs:

### Building a Research Knowledge Graph
1. Extract entities from literature using **spacy-nlp** (genes, proteins, compounds, diseases)
2. Build graph with entities as nodes and co-occurrence/relations as edges
3. Analyze graph topology for hidden connections

### Graph-Based Discovery Patterns
- **Bridging nodes**: Entities connecting otherwise separate research clusters → potential cross-disciplinary links
- **Structural holes**: Missing edges between closely related but unconnected entities → unexplored interactions
- **Community detection**: Identify research sub-fields and their boundaries
- **Link prediction**: Predict likely future connections (e.g., drug-target, gene-disease)

### Integration with Database Skills
- **UniProt + PDB**: Protein interaction networks, structural similarity graphs
- **KEGG**: Pathway topology analysis, metabolic network gaps
- **ChEMBL + PubChem**: Drug-target interaction networks, polypharmacology
- **Open Targets**: Disease-gene association networks, therapeutic area clustering
- **Wikidata**: Cross-domain entity linking, disambiguation

## AI-Augmented Discovery

Use **transformers-inference** for embedding-based discovery:
- Compute paper/concept embeddings, find unexpected semantic neighbors
- Zero-shot classification of research gaps by novelty/impact
- Use **scikit-learn-ml** for clustering related discoveries

## Zero-Hallucination Rule

ALL factual claims, citations, database results, and scientific data presented to the user MUST come from actual tool results (API calls, code execution, web search) in this conversation. NEVER fabricate or "fill in" details from training data. If a tool returns no results or partial data, report exactly what happened.
