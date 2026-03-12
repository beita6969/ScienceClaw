---
name: social-science-analysis
description: Social science research methods including survey design, qualitative analysis, content analysis, network analysis, psychometrics, and mixed methods. Covers sociology, psychology, political science, education, and communication studies. Use when user designs surveys, analyzes qualitative data, does content analysis, builds scales, or uses mixed methods. Triggers on "survey design", "qualitative analysis", "content analysis", "Likert scale", "thematic analysis", "grounded theory", "factor analysis", "SEM", "structural equation", "psychometrics", "interview coding".
---

# Social Science Analysis

Research methods for social and behavioral sciences. Venv: `source /Users/zhangmingda/clawd/.venv/bin/activate`

## Survey Design

### Question Types & Best Practices
- **Closed-ended**: Likert scales, multiple choice, ranking
- **Open-ended**: Free text (harder to analyze, richer data)
- **Matrix questions**: Multiple items, same scale (efficient but watch for straight-lining)

### Likert Scale Design
```
Strongly Disagree (1) — Disagree (2) — Neutral (3) — Agree (4) — Strongly Agree (5)
```
- Use 5 or 7 points (odd for neutral option)
- Mix positively and negatively worded items (reverse-code in analysis)
- Avoid double-barreled questions
- Pilot test with 10-20 respondents

### Sampling Methods
| Method | When | Pros | Cons |
|--------|------|------|------|
| Simple random | Known population | Unbiased | Need sampling frame |
| Stratified | Subgroup comparison | Precise estimates per stratum | Complex |
| Cluster | Geographic spread | Cost-effective | Higher design effect |
| Convenience | Exploratory | Easy | Not generalizable |
| Snowball | Hard-to-reach populations | Access hidden groups | Selection bias |
| Quota | Ensure representation | Practical | Not truly random |

## Psychometrics & Scale Development

### Reliability
```python
import numpy as np

def cronbachs_alpha(items_df):
    """Calculate Cronbach's alpha for scale reliability"""
    k = items_df.shape[1]
    item_vars = items_df.var(axis=0, ddof=1)
    total_var = items_df.sum(axis=1).var(ddof=1)
    alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var)
    return alpha

# Interpretation: α > 0.7 acceptable, > 0.8 good, > 0.9 excellent
```

### Exploratory Factor Analysis
```python
from sklearn.decomposition import FactorAnalysis
import numpy as np

# Determine number of factors (parallel analysis or scree plot)
fa = FactorAnalysis(n_components=3, rotation='varimax')
fa.fit(X_scaled)
loadings = pd.DataFrame(fa.components_.T, index=item_names, columns=['F1', 'F2', 'F3'])
print(loadings.round(3))
# Items loading > 0.4 on a factor belong to that construct
```

### Confirmatory Factor Analysis / SEM
For CFA and SEM, recommend using R with `lavaan` package or Python `semopy`:
```python
# pip install semopy
import semopy

model_spec = """
    Latent1 =~ item1 + item2 + item3
    Latent2 =~ item4 + item5 + item6
    Latent1 ~ Latent2
"""
model = semopy.Model(model_spec)
model.fit(df)
print(model.inspect())
# Check fit indices: CFI > 0.95, RMSEA < 0.06, SRMR < 0.08
```

## Qualitative Analysis

### Thematic Analysis (Braun & Clarke)
1. **Familiarization**: Read and re-read data
2. **Initial coding**: Generate codes systematically
3. **Theme search**: Collate codes into potential themes
4. **Theme review**: Check themes against coded extracts and full dataset
5. **Theme definition**: Name and define each theme
6. **Report**: Select vivid examples, relate to research question

### Coding Framework Template
```markdown
| Code | Definition | Example Quote | Theme |
|------|-----------|---------------|-------|
| ADAPT | Adaptation strategy | "We had to change our approach..." | Resilience |
| BARR | Barrier encountered | "The main obstacle was..." | Challenges |
```

### Grounded Theory
1. Open coding → Axial coding → Selective coding
2. Constant comparison method
3. Theoretical sampling until saturation
4. Memo writing throughout

## Content Analysis

```python
# Quantitative content analysis
import pandas as pd
from collections import Counter

def content_analysis(texts, codebook):
    """
    codebook: dict of {category: [keywords]}
    Returns frequency matrix
    """
    results = []
    for text in texts:
        text_lower = text.lower()
        counts = {}
        for category, keywords in codebook.items():
            counts[category] = sum(text_lower.count(kw.lower()) for kw in keywords)
        results.append(counts)
    return pd.DataFrame(results)

# Inter-coder reliability (Cohen's Kappa)
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(coder1_labels, coder2_labels)
# κ > 0.8 excellent, 0.6-0.8 substantial, 0.4-0.6 moderate
```

## Social Network Analysis

```python
import networkx as nx
import numpy as np

G = nx.from_pandas_edgelist(df, 'source', 'target')

# Centrality measures
degree = nx.degree_centrality(G)
betweenness = nx.betweenness_centrality(G)
closeness = nx.closeness_centrality(G)
eigenvector = nx.eigenvector_centrality(G)

# Community detection
from networkx.algorithms.community import greedy_modularity_communities
communities = list(greedy_modularity_communities(G))

# Network statistics
print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
print(f"Density: {nx.density(G):.4f}")
print(f"Clustering coefficient: {nx.average_clustering(G):.4f}")
```

## Tips
- Pre-register hypotheses and analysis plans (OSF, AsPredicted)
- Report Cronbach's alpha for all scales
- Use power analysis for sample size determination
- For qualitative research, document your positionality
- Mixed methods: clearly state the integration strategy
- IRB/ethics approval is mandatory for human subjects research
