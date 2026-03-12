---
name: citation-analysis
description: "Analyze citation networks, compute bibliometric indicators, and identify research fronts. Use when: user asks about citation patterns, h-index, co-authorship networks, research trends, or bibliometric analysis. NOT for: literature searching (use literature-search) or writing papers (use paper-writing)."
metadata: { "openclaw": { "emoji": "🔗" } }
---

# Citation Analysis

Analyze citation networks, compute bibliometric indicators, and identify research fronts using Semantic Scholar, OpenAlex, and CrossRef data.

## When to Use

- "What's the h-index of this author?"
- "Show me the citation network for this paper"
- "Identify the most influential papers in this field"
- "Map the co-authorship network in this area"
- "What are the emerging research fronts in NLP?"
- "Analyze citation trends for CRISPR papers over time"

## When NOT to Use

- Finding papers by topic (use literature-search)
- Reading or summarizing papers (use scienceclaw-summarization)
- Writing papers (use paper-writing)
- Statistical analysis unrelated to citations (use statsmodels-stats)

## Bibliometric Indicators

### Author-Level Metrics

```python
import numpy as np

def h_index(citations: list[int]) -> int:
    """Compute h-index from a list of citation counts."""
    sorted_c = sorted(citations, reverse=True)
    h = 0
    for i, c in enumerate(sorted_c):
        if c >= i + 1:
            h = i + 1
        else:
            break
    return h

def g_index(citations: list[int]) -> int:
    """Compute g-index: largest g such that top g papers have >= g^2 citations."""
    sorted_c = sorted(citations, reverse=True)
    cumsum = np.cumsum(sorted_c)
    g = 0
    for i in range(len(sorted_c)):
        if cumsum[i] >= (i + 1) ** 2:
            g = i + 1
    return g

def i10_index(citations: list[int]) -> int:
    """Number of papers with 10+ citations."""
    return sum(1 for c in citations if c >= 10)
```

### Paper-Level Metrics

- **Citation count**: Raw count from Semantic Scholar / OpenAlex
- **Field-weighted citation impact (FWCI)**: Citations / expected citations in field
- **Percentile rank**: Position relative to same-year, same-field papers
- **Citation velocity**: Citations per year since publication

### Journal-Level Metrics

- **Impact Factor**: Citations in year N to papers published in N-1 and N-2
- **CiteScore**: Citations over 4 years / documents over 4 years
- **h5-index**: h-index of articles published in the last 5 years

## Citation Network Analysis

### Build Citation Graph

```python
import networkx as nx

def build_citation_graph(papers: list[dict]) -> nx.DiGraph:
    """
    Build a directed citation graph.
    Each paper dict should have 'paperId', 'title', 'citations', 'references'.
    Edge direction: citing -> cited.
    """
    G = nx.DiGraph()
    for p in papers:
        G.add_node(p['paperId'], title=p['title'],
                    year=p.get('year'), citations=p.get('citationCount', 0))
        for ref in p.get('references', []):
            if ref.get('paperId'):
                G.add_edge(p['paperId'], ref['paperId'])
        for cit in p.get('citations', []):
            if cit.get('paperId'):
                G.add_edge(cit['paperId'], p['paperId'])
    return G
```

### Key Network Metrics

```python
def analyze_citation_network(G: nx.DiGraph) -> dict:
    """Compute key citation network metrics."""
    results = {}
    results['num_papers'] = G.number_of_nodes()
    results['num_citations'] = G.number_of_edges()
    results['density'] = nx.density(G)

    # Most cited (highest in-degree)
    in_deg = dict(G.in_degree())
    results['most_cited'] = sorted(in_deg.items(), key=lambda x: -x[1])[:10]

    # PageRank (identifies influential papers beyond raw citations)
    pr = nx.pagerank(G)
    results['pagerank_top'] = sorted(pr.items(), key=lambda x: -x[1])[:10]

    # Betweenness centrality (bridge papers connecting subfields)
    bc = nx.betweenness_centrality(G)
    results['bridge_papers'] = sorted(bc.items(), key=lambda x: -x[1])[:10]

    return results
```

### Co-Authorship Network

```python
def build_coauthor_graph(papers: list[dict]) -> nx.Graph:
    """Build undirected co-authorship graph."""
    G = nx.Graph()
    for p in papers:
        authors = [a['name'] for a in p.get('authors', []) if a.get('name')]
        for i, a1 in enumerate(authors):
            G.add_node(a1)
            for a2 in authors[i+1:]:
                if G.has_edge(a1, a2):
                    G[a1][a2]['weight'] += 1
                else:
                    G.add_edge(a1, a2, weight=1)
    return G

def find_communities(G: nx.Graph) -> list:
    """Detect research communities via Louvain."""
    from networkx.algorithms.community import louvain_communities
    return louvain_communities(G, resolution=1.0)
```

## Research Front Detection

### Method: Co-Citation Clustering
1. Identify highly co-cited paper pairs (cited together frequently)
2. Cluster co-cited papers into research fronts
3. Label fronts by common keywords in citing papers

### Method: Citation Burst Detection
1. Track citation counts per year for a set of papers
2. Identify papers with sudden citation increases (Kleinberg burst detection)
3. Papers with recent bursts indicate active research fronts

### Method: Bibliographic Coupling
1. Two papers are coupled if they share references
2. Stronger coupling = more shared references
3. Cluster coupled papers to find parallel research streams

## Visualization

### Citation Trend Plot
```python
import matplotlib.pyplot as plt

def plot_citation_trend(papers: list[dict], output_path: str):
    """Plot citation counts over publication years."""
    years = [p['year'] for p in papers if p.get('year')]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(years, bins=range(min(years), max(years)+2), edgecolor='black')
    ax.set_xlabel('Publication Year')
    ax.set_ylabel('Number of Papers')
    ax.set_title('Publication Trend')
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")
```

## Data Sources

| Source | Endpoint | Free? | Rate Limit |
|--------|----------|-------|------------|
| Semantic Scholar | `/paper/{id}/citations`, `/paper/{id}/references` | Yes | 100/5min (no key) |
| OpenAlex | `/works?cited_by={id}`, `/works?cites={id}` | Yes | 100k/day |
| CrossRef | `/works/{doi}` (reference list) | Yes | Polite pool |

## Best Practices

1. Always use Semantic Scholar paperId or DOI as canonical identifiers
2. Normalize author names (handle variants: "J. Smith" vs "John Smith")
3. Filter self-citations when computing impact metrics
4. Use field-normalized metrics for cross-discipline comparisons
5. Report the date of data collection (citation counts change daily)
6. Visualize networks with node size proportional to citations

## Zero-Hallucination Rule

- NEVER fabricate citation counts, h-indices, or paper metadata
- All bibliometric data must come from tool results in the current session
- If an API returns no data for an author/paper, report the empty result explicitly
