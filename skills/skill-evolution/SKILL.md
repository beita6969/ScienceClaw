---
name: skill-evolution
description: "Track and improve skill effectiveness over time using VOYAGER-style skill library patterns. Use when: analyzing which tools/strategies worked best, proposing skill improvements, or storing reusable research patterns. NOT for: active research tasks or immediate problem-solving."
metadata: { "openclaw": { "emoji": "🧬" } }
---

# Skill Evolution

Track tool and strategy effectiveness across research tasks, propose improvements to existing skills, and maintain a reusable pattern library inspired by VOYAGER's skill library architecture.

## When to Use

- After multiple research sessions to identify improvement opportunities
- "Which search strategies work best for biomedical topics?"
- "Propose improvements to the literature-search skill"
- "Store this successful analysis pattern for reuse"
- When the science-evolution extension triggers post-session analysis

## When NOT to Use

- During active research (focus on the task, reflect afterward)
- For one-off questions that don't need pattern storage
- For simple skill usage (just use the skill directly)

## VOYAGER Skill Library Pattern

Inspired by VOYAGER (Wang et al., 2023), maintain a library of reusable research patterns:

### Pattern Structure

```json
{
  "pattern_id": "lit-review-biomedical-v2",
  "name": "Biomedical Literature Review",
  "domain": ["biology", "medicine"],
  "task_type": "literature_review",
  "version": 2,
  "description": "Optimized search strategy for biomedical topics",
  "steps": [
    "Search Semantic Scholar with MeSH-equivalent terms",
    "Search PubMed via NCBI Entrez with MeSH filters",
    "Cross-reference with ClinicalTrials.gov for ongoing trials",
    "Citation chain top 3 papers (forward + backward)",
    "Verify key genes/proteins in UniProt",
    "Verify drug interactions in ChEMBL"
  ],
  "tools_used": ["semantic-scholar", "ncbi-entrez", "uniprot-protein", "chembl-drug"],
  "success_rate": 0.85,
  "avg_quality_score": 21,
  "times_used": 12,
  "last_used": "2026-03-11",
  "lessons": [
    "PubMed MeSH terms are more precise than free-text for biomedical queries",
    "Always check ClinicalTrials.gov for therapy-related topics",
    "UniProt cross-references to PDB save a separate search step"
  ]
}
```

### Pattern Operations

```bash
# Store a new pattern
curl -X POST http://localhost:18789/evolution/pattern \
  -H "Content-Type: application/json" \
  -d '{"pattern": {...}}'

# Search for patterns matching a new task
curl "http://localhost:18789/evolution/search?domain=biology&task_type=literature_review"

# Update pattern after use (increment times_used, update success_rate)
curl -X PATCH http://localhost:18789/evolution/pattern/lit-review-biomedical-v2 \
  -d '{"success": true, "quality_score": 22}'
```

## Skill Improvement Proposals

### Analysis Framework

After 5+ uses of a skill, analyze its performance:

```markdown
## Skill Analysis: [skill-name]

### Usage Statistics
- Times used: N
- Average quality score: X/25
- Success rate: Y%
- Common failure modes: [list]

### Strengths
- [What the skill does well]

### Weaknesses
- [Where the skill falls short]
- [Missing capabilities]
- [Incorrect or outdated guidance]

### Proposed Changes
1. [Specific change to SKILL.md]
   - Rationale: [why]
   - Expected impact: [improvement area]

2. [Another change]
   - Rationale: [why]
   - Expected impact: [improvement area]

### Priority: [high/medium/low]
```

### Automated Improvement Detection

Track these signals across research sessions:

| Signal | Indicates | Action |
|--------|-----------|--------|
| Repeated tool failures | API endpoint changed or unreliable | Update SKILL.md with workaround |
| Consistent low scores in one dimension | Skill gap in that area | Add guidance for that dimension |
| User corrections | Skill provides wrong guidance | Fix the incorrect guidance |
| New API discovered | Opportunity to expand | Add new tool instructions |
| Cross-domain pattern success | Transferable knowledge | Create cross-domain pattern |

## Cross-Domain Knowledge Transfer

### Identifying Transferable Patterns

Some research patterns work across disciplines:

1. **Citation chain analysis** — Works for any field with citation data
2. **Database cross-verification** — Applicable whenever primary data exists
3. **Effect size reporting** — Standard across quantitative disciplines
4. **PICO framework** — Adaptable beyond medicine (SPIDER for qualitative)
5. **Visualization standards** — Journal figure requirements are similar

### Transfer Process

When a pattern succeeds in domain A, evaluate for domain B:
1. Are the tools available? (e.g., does domain B have equivalent databases?)
2. Are the methods appropriate? (e.g., meta-analysis needs comparable studies)
3. What adaptations are needed? (e.g., different search terms, different databases)
4. Store as a new domain-specific variant

## Evolution Metrics

### Skill Health Dashboard

```markdown
| Skill | Uses | Avg Score | Trend | Health |
|-------|------|-----------|-------|--------|
| literature-search | 45 | 22/25 | +1.2 | Healthy |
| statsmodels-stats | 12 | 18/25 | -0.5 | Needs attention |
| semantic-scholar | 38 | 23/25 | +0.8 | Healthy |
| meta-analysis | 3 | -- | -- | Too few uses |
```

### Trend Analysis
- **Improving**: Score trending up → skill guidance is effective
- **Declining**: Score trending down → investigate (API changes? outdated guidance?)
- **Stable**: No trend → working as expected
- **Insufficient data**: < 5 uses → collect more data before drawing conclusions

## Integration with science-evolution Extension

This skill works with the `science-evolution` extension:
- Extension tracks tool usage and outcomes automatically
- Stores data in `~/.openclaw/science-evolution.db`
- Provides API endpoints for pattern storage and retrieval
- Triggers post-session analysis when enough data accumulates

## Best Practices

1. Don't optimize prematurely — wait for 5+ uses before proposing changes
2. Track both successes and failures for each pattern
3. Version patterns so you can roll back if a change hurts performance
4. Cross-reference reflections (research-reflection) with evolution data
5. Focus improvements on the highest-impact skills first
6. Keep the pattern library curated — remove patterns that are never reused
