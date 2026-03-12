---
name: research-reflection
description: "Reflect on completed research tasks to improve future performance. Use when: a research task has just been completed and the agent should evaluate its own process, store lessons learned, or retrieve past reflections before starting new work. NOT for: active research execution or data analysis."
metadata: { "openclaw": { "emoji": "🪞" } }
---

# Research Reflection (Reflexion Pattern)

Self-evaluate completed research tasks, store lessons learned, and retrieve relevant past experience to improve future research quality. Based on the Reflexion framework (verbal reinforcement learning without weight updates).

## When to Use

- After completing any substantial research task (literature review, data analysis, systematic review)
- Before starting a new task in a previously explored domain
- When a research approach failed and you need to diagnose why
- "What did I learn from the last CRISPR review?"
- "Before starting, check if we've done similar work before"

## When NOT to Use

- During active research execution (focus on the task)
- For simple factual lookups
- For non-research tasks

## Reflexion Cycle

The cycle runs after each substantial research task:

```
Execute Task → Evaluate Output → Generate Reflection → Store in Memory → Retrieve Next Time
```

### Phase 1: Self-Evaluation

After completing a research task, evaluate against these dimensions:

```markdown
## Post-Task Evaluation

### Completeness (1-5)
- Did I search enough databases? Which ones did I miss?
- Did I follow all 6 Research Depth Phases from SCIENCE.md?
- Are there obvious gaps in my coverage?

### Accuracy (1-5)
- Did every citation come from a tool result?
- Did I cross-verify claims against primary databases?
- Were statistical results correctly reported?

### Efficiency (1-5)
- Did I use the right tools for each subtask?
- Were there redundant searches I could have avoided?
- Did I hit rate limits that slowed me down?

### Depth (1-5)
- Did I go beyond surface-level results?
- Did I trace citation chains?
- Did I identify contradictions and open questions?

### Actionability (1-5)
- Is the output directly useful to the user?
- Are next steps clearly identified?
- Is the methodology reproducible?
```

### Phase 2: Generate Reflection

Produce a structured reflection:

```markdown
## Reflection: [Task Title]
Date: [ISO date]
Domain: [e.g., molecular biology, econometrics]
Task type: [literature review / data analysis / systematic review / ...]

### What went well
- [Specific successful strategies]

### What went poorly
- [Specific failures or inefficiencies]

### Key lessons
- [Actionable insights for future tasks]

### Tools that worked best
- [Tool name]: [Why it was effective for this task]

### Tools that underperformed
- [Tool name]: [Why, and what to use instead]

### Domain-specific insights
- [API endpoints that were useful]
- [Search strategies that worked]
- [Data sources that were reliable/unreliable]

### Score: [total/25]
```

### Phase 3: Store Reflection

Store reflections using the science-memory extension:

```bash
# Store reflection with domain tags
curl -X POST http://localhost:18789/memory/store \
  -H "Content-Type: application/json" \
  -d '{
    "key": "reflection-[task-id]",
    "content": "[reflection markdown]",
    "tags": ["reflection", "domain-tag", "task-type"],
    "metadata": {"score": 18, "date": "2026-03-11"}
  }'
```

### Phase 4: Retrieve Before New Task

Before starting a new research task, retrieve relevant past reflections:

```bash
# Search for reflections in the same domain
curl http://localhost:18789/memory/search?q=domain-tag+task-type&tags=reflection
```

Use retrieved reflections to:
1. Avoid repeating past mistakes
2. Reuse successful search strategies
3. Skip known-unreliable data sources
4. Apply domain-specific insights

## Reflection Templates by Task Type

### Literature Review Reflection
- Databases searched and result counts
- Most effective search terms
- Key papers found vs. expected coverage
- Citation chains that revealed important papers
- Gaps: papers known to exist but not found

### Data Analysis Reflection
- Data sources used and data quality issues
- Analysis methods applied and their appropriateness
- Unexpected findings that required investigation
- Reproducibility: can the analysis be rerun from the report?

### Systematic Review Reflection
- PRISMA compliance level
- Screening efficiency (false positive/negative rate)
- Data extraction completeness
- Meta-analysis method appropriateness

## Continuous Improvement

### Pattern Recognition
Over multiple reflections, identify recurring patterns:
- Which databases consistently perform best for your domain?
- Which search strategies yield the highest-quality results?
- What's your typical bottleneck (rate limits, data quality, synthesis)?

### Strategy Evolution
Based on accumulated reflections:
1. Update personal search templates
2. Maintain a "preferred tools by domain" list
3. Build domain-specific query libraries
4. Track API reliability over time

## Auto-Trigger: Post-Task Reflection

This skill should be triggered automatically after any substantial research task. The agent should:

1. **After task completion**: Generate a reflection without being asked
2. **Save to disk**: Store as `~/clawd/reflections/{domain}/{date}-{topic}.md`
3. **Before new tasks**: Check `~/clawd/reflections/` for relevant past reflections in the same domain

### File-Based Reflection Storage (No External Service Required)

```bash
# Create reflection directory structure
mkdir -p ~/clawd/reflections/{biology,medicine,economics,physics,chemistry,cs,social-science,general}

# Save a reflection after task completion
cat > ~/clawd/reflections/biology/2026-03-11-crispr-review.md << 'EOF'
## Reflection: CRISPR Base Editing Literature Review
Date: 2026-03-11
Domain: biology
Task type: literature review
Score: 20/25

### What went well
- Semantic Scholar returned high-quality results for "CRISPR base editing"
- Citation chain from Komor et al. 2016 revealed all major developments
- Cross-verification with UniProt confirmed protein structures

### What went poorly
- PubMed API was slow (>5s per query), should have used Europe PMC
- Missed Anzalone 2019 prime editing paper initially

### Key lessons
- For CRISPR topics, always include David Liu's lab papers as seed
- UniProt cross-references to PDB save a separate structure search
- Europe PMC is faster than PubMed for programmatic access

### Tools that worked best
- Semantic Scholar: excellent for CRISPR literature, good TLDR summaries
- Jina Reader: successfully extracted full text from Nature papers

### Tools that underperformed
- PubMed: slow API, consider Europe PMC as primary for biomedical
EOF
```

### Retrieving Past Reflections Before New Tasks

```bash
# Search for relevant reflections before starting a new task
# Simple grep-based search (no vector DB needed)
grep -rl "CRISPR\|base editing\|gene editing" ~/clawd/reflections/ 2>/dev/null | head -5

# Read the most recent reflection in a domain
ls -t ~/clawd/reflections/biology/*.md 2>/dev/null | head -3 | xargs head -50
```

### Accumulated Reflection Analysis

After 5+ reflections in a domain, generate a meta-analysis:

```markdown
## Domain Meta-Analysis: Biology (N=8 reflections)

### Best-performing tools
1. Semantic Scholar (avg score 4.2/5 across tasks)
2. UniProt (4.0/5 for protein-related queries)
3. Jina Reader (3.8/5 for full-text extraction)

### Common failure patterns
1. PubMed API timeouts (3/8 tasks affected)
2. Missing recent preprints not yet indexed
3. Rate limits on Semantic Scholar for >50 queries

### Recommended strategy updates
1. Use Europe PMC instead of PubMed for programmatic access
2. Add bioRxiv/medRxiv search for recent preprints
3. Batch Semantic Scholar queries to avoid rate limits
```

## Best Practices

1. Reflect immediately after task completion while details are fresh
2. Be honest about failures -- the goal is improvement, not a perfect score
3. Keep reflections concise (200-400 words)
4. Tag reflections with both domain and task type for easy retrieval
5. Review accumulated reflections quarterly to identify meta-patterns
6. Share useful reflections across the skill-evolution system
7. Always save reflections to disk -- memory is ephemeral, files persist
8. Keep only the last 3 reflections per domain in active context (window management)
9. After 5+ reflections in a domain, generate a meta-analysis to compress accumulated knowledge
