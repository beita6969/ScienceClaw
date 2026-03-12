---
name: visualization
description: Create publication-quality scientific figures and plots using Python (matplotlib, seaborn, plotly). Supports bar charts, scatter plots, heatmaps, box plots, violin plots, survival curves, network graphs, and more. Use when user asks to plot data, create figures, make charts, visualize results, or generate publication-ready graphics. Triggers on "plot", "chart", "figure", "graph", "visualize", "heatmap", "scatter plot", "bar chart", "histogram".
---

# Scientific Visualization

Publication-quality figures with Python. Use venv: `source /Users/zhangmingda/clawd/.venv/bin/activate`

## Style Defaults (journal-ready)

```python
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np

# Publication style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.transparent': True,
})
sns.set_palette("colorblind")  # accessible colors
```

## Common Plot Types

### Distribution
```python
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(data=df, x='value', hue='group', kde=True, ax=ax)
ax.set_xlabel('Value')
ax.set_ylabel('Count')
plt.savefig('dist.png', dpi=300)
```

### Comparison (box + strip)
```python
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=df, x='group', y='value', ax=ax, width=0.5)
sns.stripplot(data=df, x='group', y='value', ax=ax, color='black', alpha=0.3, size=3)
ax.set_ylabel('Measurement (units)')
plt.savefig('comparison.png', dpi=300)
```

### Scatter + Regression
```python
fig, ax = plt.subplots(figsize=(6, 5))
sns.regplot(data=df, x='x', y='y', ax=ax, scatter_kws={'alpha': 0.5})
r, p = stats.pearsonr(df['x'], df['y'])
ax.annotate(f'r = {r:.3f}, p = {p:.3g}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10)
plt.savefig('scatter.png', dpi=300)
```

### Heatmap (correlation / expression)
```python
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax)
plt.savefig('heatmap.png', dpi=300)
```

### Multi-panel Figure
```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# Panel A
axes[0].plot(x, y)
axes[0].set_title('A', loc='left', fontweight='bold')
# Panel B
axes[1].bar(categories, values)
axes[1].set_title('B', loc='left', fontweight='bold')
# Panel C
axes[2].scatter(x2, y2)
axes[2].set_title('C', loc='left', fontweight='bold')
plt.tight_layout()
plt.savefig('figure1.png', dpi=300)
```

### Volcano Plot (genomics)
```python
fig, ax = plt.subplots(figsize=(7, 5))
colors = np.where((df['padj'] < 0.05) & (abs(df['log2FC']) > 1), 'red',
         np.where(df['padj'] < 0.05, 'blue', 'grey'))
ax.scatter(df['log2FC'], -np.log10(df['padj']), c=colors, alpha=0.5, s=10)
ax.axhline(-np.log10(0.05), ls='--', color='grey', lw=0.8)
ax.axvline(-1, ls='--', color='grey', lw=0.8)
ax.axvline(1, ls='--', color='grey', lw=0.8)
ax.set_xlabel('log₂ Fold Change')
ax.set_ylabel('-log₁₀ adjusted p-value')
plt.savefig('volcano.png', dpi=300)
```

### Network Graph
```python
import networkx as nx
G = nx.from_pandas_edgelist(df, 'source', 'target', 'weight')
pos = nx.spring_layout(G, seed=42)
fig, ax = plt.subplots(figsize=(8, 8))
nx.draw_networkx(G, pos, ax=ax, node_size=300, font_size=8, edge_color='grey', alpha=0.7)
plt.savefig('network.png', dpi=300)
```

### Interactive (Plotly)
```python
import plotly.express as px
fig = px.scatter(df, x='x', y='y', color='group', hover_data=['label'],
                 title='Interactive Scatter')
fig.write_html('interactive.html')
fig.write_image('scatter.png', scale=2)  # needs kaleido
```

## Journal Requirements

| Journal | Width (single col) | Width (double col) | Format | Font min |
|---------|-------------------|-------------------|--------|----------|
| Nature | 89mm | 183mm | PDF/EPS/TIFF | 5pt |
| Science | 85mm | 174mm | PDF/EPS | 6pt |
| PNAS | 87mm | 178mm | PDF/EPS/TIFF | 6pt |
| IEEE | 3.5in | 7.16in | PDF/EPS | 8pt |
| Elsevier | 90mm | 190mm | PDF/EPS/TIFF | 6pt |

```python
# Nature single-column figure
fig, ax = plt.subplots(figsize=(3.5, 2.6))  # 89mm ≈ 3.5in
```

## Accessibility
- Use colorblind-safe palettes: `sns.set_palette("colorblind")`
- Add patterns/markers in addition to color
- Ensure sufficient contrast
- Use descriptive axis labels with units
- Include alt text in figure captions

## Tips
- Save as both PNG (for preview) and PDF/SVG (for publication)
- Always label axes with units
- Use consistent color coding across related figures
- Avoid 3D plots unless data is truly 3D
- Minimize chart junk (unnecessary gridlines, borders, decorations)
