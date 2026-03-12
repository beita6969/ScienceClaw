---
name: networkx-social
description: "Social network and graph analysis via NetworkX. Use when: graph analysis, community detection, centrality measures, network visualization, knowledge graphs, bipartite networks, graph I/O. NOT for: large-scale graph processing (>1M nodes), GPU graph analytics (use cuGraph)."
metadata: { "openclaw": { "emoji": "🕸️", "requires": { "bins": ["python3"] }, "install": [{ "id": "uv-networkx", "kind": "uv", "package": "networkx matplotlib" }] } }
---

# NetworkX Graph Analysis

Graph analysis, community detection, centrality measures, knowledge graphs, bipartite networks, and visualization.

## Graph Creation

```python
import networkx as nx

G = nx.Graph()
G.add_edge('Alice', 'Bob', weight=3)
G.add_edges_from([('Bob', 'Carol'), ('Carol', 'Dave'), ('Alice', 'Dave')])

D = nx.DiGraph()                          # directed graph
G = nx.read_edgelist('network.txt')       # from edge list file
G = nx.from_pandas_edgelist(df, 'source', 'target')  # from DataFrame
```

## Knowledge Graph Construction

```python
# Build a knowledge graph from (subject, predicate, object) triples
KG = nx.DiGraph()
triples = [
    ("Python", "is_a", "Language"), ("Pandas", "depends_on", "Python"),
    ("NumPy", "depends_on", "Python"), ("Pandas", "depends_on", "NumPy"),
]
for subj, pred, obj in triples:
    KG.add_edge(subj, obj, relation=pred)

# Query: all dependencies of Pandas
deps = list(nx.descendants(KG, "Pandas"))

# Subgraph around a node (ego graph)
ego = nx.ego_graph(KG, "Python", radius=2, undirected=True)
```

## Centrality Measures

```python
dc = nx.degree_centrality(G)              # fraction of connected nodes
bc = nx.betweenness_centrality(G)         # shortest-path intermediary
cc = nx.closeness_centrality(G)           # inverse avg distance
ec = nx.eigenvector_centrality(G, max_iter=1000)  # neighbor importance
pr = nx.pagerank(D, alpha=0.85)           # PageRank (directed)

for node, score in sorted(bc.items(), key=lambda x: -x[1])[:5]:
    print(f"{node}: {score:.4f}")
```

## Community Detection

```python
from networkx.algorithms.community import louvain_communities, modularity
from networkx.algorithms.community import label_propagation_communities, greedy_modularity_communities

communities = louvain_communities(G, seed=42)        # modularity optimization
communities = list(label_propagation_communities(G))  # fast, non-deterministic
greedy = list(greedy_modularity_communities(G))       # greedy modularity

mod = modularity(G, communities)
print(f"Modularity: {mod:.4f}")
```

## Shortest Paths and Distances

```python
path = nx.shortest_path(G, source='Alice', target='Dave')
length = nx.shortest_path_length(G, source='Alice', target='Dave')
if nx.is_connected(G):
    diameter = nx.diameter(G)
    avg_path = nx.average_shortest_path_length(G)
```

## Clustering and Connectivity

```python
avg_cc = nx.average_clustering(G)
transitivity = nx.transitivity(G)                     # global clustering
components = list(nx.connected_components(G))
largest_cc = G.subgraph(max(components, key=len)).copy()
core_numbers = nx.core_number(G)
```

## Bipartite Graphs and Projections

```python
from networkx.algorithms import bipartite

B = nx.Graph()
B.add_nodes_from(["u1", "u2", "u3"], bipartite=0)    # users
B.add_nodes_from(["p1", "p2"], bipartite=1)           # products
B.add_edges_from([("u1", "p1"), ("u2", "p1"), ("u2", "p2"), ("u3", "p2")])
users = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}
user_graph = bipartite.projected_graph(B, users)      # shared neighbors become edges
weighted_proj = bipartite.weighted_projected_graph(B, users)
```

## Network Visualization

```python
import matplotlib.pyplot as plt

pos = nx.spring_layout(G, seed=42, k=1.5)
node_sizes = [3000 * dc[n] for n in G.nodes()]
color_map = {n: i for i, comm in enumerate(communities) for n in comm}
nx.draw_networkx(G, pos, node_size=node_sizes,
                 node_color=[color_map.get(n, 0) for n in G.nodes()],
                 cmap=plt.cm.Set3, edge_color='gray', alpha=0.8, font_size=9)
plt.tight_layout()
plt.savefig('network.png', dpi=150)
plt.close()

# Other layouts: nx.circular_layout, nx.kamada_kawai_layout, nx.shell_layout
```

## Graph I/O

```python
# GraphML (XML-based, preserves attributes)
nx.write_graphml(G, 'graph.graphml')
G = nx.read_graphml('graph.graphml')

# GEXF (Gephi format)
nx.write_gexf(G, 'graph.gexf')
G = nx.read_gexf('graph.gexf')

# JSON (node-link format)
from networkx.readwrite import json_graph
import json
data = json_graph.node_link_data(G)
json.dump(data, open('graph.json', 'w'))
G = json_graph.node_link_graph(json.load(open('graph.json')))
```

## Best Practices

1. Use `G.copy()` before destructive operations (node/edge removal).
2. For large graphs, prefer `louvain_communities` over `girvan_newman`.
3. Set `seed` in layout functions for reproducible visualizations.
4. Check `nx.is_connected(G)` before computing diameter or avg path length.
5. For weighted networks, pass `weight='weight'` to centrality functions.
6. For knowledge graphs, use `DiGraph` and store relation types as edge attributes.
7. Export to GraphML or GEXF for interoperability with Gephi and other tools.
