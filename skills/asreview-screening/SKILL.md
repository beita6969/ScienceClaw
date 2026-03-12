---
name: asreview-screening
description: "Screen papers for systematic reviews using ASReview active learning. Use when: user has a large set of papers to screen for inclusion/exclusion, wants to prioritize relevant papers, or needs to reduce manual screening workload. NOT for: searching papers (use literature-search) or meta-analysis (use meta-analysis)."
metadata: { "openclaw": { "emoji": "🔍", "requires": { "bins": ["python3"] }, "install": [{ "id": "asreview", "kind": "pip", "package": "asreview" }] } }
---

# ASReview Screening

Use active learning to prioritize and screen papers for systematic reviews, reducing manual workload by up to 95%. ASReview uses machine learning to learn from your screening decisions and prioritize the most likely relevant papers.

## When to Use

- "I have 500 papers to screen for my systematic review"
- "Help me prioritize papers for inclusion"
- "Set up active learning screening for my review"
- "How many papers do I need to screen manually?"

## When NOT to Use

- Searching for papers (use literature-search)
- Performing meta-analysis (use meta-analysis)
- Writing the review (use systematic-review + paper-writing)
- Small sets (< 50 papers) — manual screening is faster

## Setup

### Install ASReview

```bash
pip install asreview asreview-insights asreview-datatools
```

### Prepare Input Data

ASReview accepts RIS, CSV, TSV, or Excel files with at minimum:
- `title`: Paper title
- `abstract`: Paper abstract

Optional but recommended:
- `doi`, `authors`, `year`, `keywords`, `label` (if some are pre-labeled)

### Export from Search Results

```python
# Convert Semantic Scholar / OpenAlex results to ASReview format
import csv

def export_for_asreview(papers: list[dict], output_path: str):
    """Export papers to CSV for ASReview."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'title', 'abstract', 'authors', 'year', 'doi', 'keywords'
        ])
        writer.writeheader()
        for p in papers:
            writer.writerow({
                'title': p.get('title', ''),
                'abstract': p.get('abstract', ''),
                'authors': '; '.join(a.get('name', '') for a in p.get('authors', [])),
                'year': p.get('year', ''),
                'doi': p.get('externalIds', {}).get('DOI', ''),
                'keywords': '; '.join(p.get('fieldsOfStudy', []))
            })
    print(f"Exported {len(papers)} papers to {output_path}")
```

## Screening Workflow

### Step 1: Create ASReview Project

```bash
# Start ASReview LAB (web interface)
asreview lab

# Or use the command-line simulation mode for automated screening
asreview simulate your_papers.csv \
  --state_file output/simulation.asreview \
  --model nb \
  --feature_extraction tfidf \
  --query_strategy max \
  --balance_strategy double \
  --n_prior_included 5 \
  --n_prior_excluded 5
```

### Step 2: Prior Knowledge

Provide seed papers to initialize the model:
- Include 1-5 papers you know are **relevant** (included)
- Include 1-5 papers you know are **irrelevant** (excluded)
- More diverse priors = better initial model

### Step 3: Active Learning Loop

```python
# Programmatic screening with ASReview
from asreview import ASReviewData, ReviewSimulate
from asreview.models import NBModel
from asreview.query_strategies import MaxQuery
from asreview.feature_extraction import Tfidf

# Load data
data = ASReviewData.from_file("papers.csv")

# Configure model
model = NBModel()
query_strategy = MaxQuery()
feature_extraction = Tfidf()

# The model learns from each decision and reprioritizes remaining papers
# In practice, use the web interface (asreview lab) for interactive screening
```

### Step 4: Stopping Criteria

When to stop screening:

| Method | Rule | Conservative? |
|--------|------|--------------|
| Consecutive irrelevant | Stop after N consecutive irrelevant papers | Moderate |
| Percentage | Screen top 10-20% of all papers | Conservative |
| Recall target | Estimate 95% recall reached | Model-dependent |
| ASReview heuristic | Stop when model confidence stabilizes | Built-in |

Recommended: Screen until you've seen at least **50 consecutive irrelevant papers** after finding all known relevant papers.

## Quality Assessment

### Simulation for Validation

If you have a fully labeled dataset, simulate to assess ASReview's performance:

```bash
# Run simulation
asreview simulate labeled_papers.csv \
  --state_file simulation.asreview

# Generate metrics
asreview insights simulation.asreview \
  --output metrics.json

# Key metrics:
# - WSS@95: Work Saved over Sampling at 95% recall
# - RRF@10: Relevant Records Found after screening 10%
# - ATD: Average Time to Discovery
```

### Interpreting Results

| Metric | Good | Excellent |
|--------|------|-----------|
| WSS@95 | > 70% | > 85% |
| RRF@10 | > 40% | > 60% |
| ATD | < 30% of dataset | < 15% of dataset |

## Integration with Systematic Review Workflow

```
literature-search → export results → asreview-screening → filtered papers → systematic-review
```

1. **literature-search**: Multi-database search, deduplication
2. **Export**: Convert to ASReview-compatible format (CSV/RIS)
3. **ASReview screening**: Active learning prioritization
4. **Output**: List of included/excluded papers with reasons
5. **systematic-review**: Data extraction, meta-analysis, PRISMA report

### PRISMA Flow Diagram Numbers

After screening, report:
- Total records identified (from all databases)
- Duplicates removed
- Records screened (title/abstract)
- Records excluded (with reasons)
- Full-text articles assessed
- Studies included in synthesis

## Advanced Features

### Multiple Models

```bash
# Compare model performance
asreview simulate papers.csv --model nb --state_file sim_nb.asreview
asreview simulate papers.csv --model svm --state_file sim_svm.asreview
asreview simulate papers.csv --model logistic --state_file sim_lr.asreview
```

### Deduplication

```bash
# Remove duplicates before screening
asreview data dedup input.csv --output deduped.csv
```

## Best Practices

1. Always provide diverse prior knowledge (relevant + irrelevant examples)
2. Use at least 2-3 relevant and 5-10 irrelevant seed papers
3. Screen conservatively — missing a relevant paper is worse than extra screening
4. Document your stopping criteria and justify in the methods section
5. Run simulation on a subset if possible to estimate recall
6. Export screening decisions for PRISMA flow diagram
7. Never fabricate screening statistics or WSS values

## Zero-Hallucination Rule

- ALL screening statistics must come from actual ASReview output
- NEVER estimate recall without running a proper simulation
- Report exact numbers from the screening log, not approximations
