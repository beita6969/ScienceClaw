---
name: matplotlib-viz
description: "Scientific visualization via Matplotlib. Use when: user asks for plots, charts, or data visualization. NOT for: interactive dashboards or web-based charts."
metadata: { "openclaw": { "emoji": "📊", "requires": { "bins": ["python3"] }, "install": [{ "id": "uv-matplotlib", "kind": "uv", "package": "matplotlib numpy" }] } }
---

# Matplotlib Visualization

Scientific visualization and publication-quality figures using Matplotlib and NumPy.

## When to Use

- Static plots: line, scatter, bar, histogram, heatmap
- Publication-ready scientific figures
- Multi-panel (subplot) layouts
- Saving figures to PNG, SVG, or PDF
- Annotated or styled plots for presentations and papers

## When NOT to Use

- Interactive dashboards (use Plotly or Dash)
- Web-based charts (use D3.js or Chart.js)
- Real-time streaming visualizations
- Geographic/map plots (use Cartopy or Folium)

## Basic Setup

```python
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving files
import matplotlib.pyplot as plt
import numpy as np
```

## Line and Scatter Plots

```python
x = np.linspace(0, 10, 100)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, np.sin(x), label='sin(x)', linewidth=2)
ax.plot(x, np.cos(x), label='cos(x)', linestyle='--')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Trigonometric Functions')
ax.legend()
fig.savefig('line_plot.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# Scatter with colormap
fig, ax = plt.subplots()
sc = ax.scatter(x_data, y_data, c=color_values, cmap='viridis', s=50, alpha=0.7)
fig.colorbar(sc, ax=ax, label='Magnitude')
fig.savefig('scatter.png', dpi=150, bbox_inches='tight')
plt.close(fig)
```

## Bar Charts and Histograms

```python
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 12, 67]
fig, ax = plt.subplots()
ax.bar(categories, values, color='steelblue', edgecolor='black')
ax.set_ylabel('Count')
fig.savefig('bar_chart.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# Histogram with KDE overlay
fig, ax = plt.subplots()
ax.hist(data, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
ax.set_xlabel('Value')
ax.set_ylabel('Density')
fig.savefig('histogram.png', dpi=150, bbox_inches='tight')
plt.close(fig)
```

## Heatmaps

```python
data_matrix = np.random.rand(10, 10)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(data_matrix, cmap='coolwarm', aspect='auto')
fig.colorbar(im, ax=ax)
ax.set_xticks(range(10))
ax.set_yticks(range(10))
fig.savefig('heatmap.png', dpi=150, bbox_inches='tight')
plt.close(fig)
```

## Subplots and Multi-Panel Figures

```python
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].plot(x, y1)
axes[0, 0].set_title('Panel A')
axes[0, 1].scatter(x, y2, s=10)
axes[0, 1].set_title('Panel B')
axes[1, 0].bar(categories, values)
axes[1, 0].set_title('Panel C')
axes[1, 1].hist(data, bins=20)
axes[1, 1].set_title('Panel D')
fig.tight_layout()
fig.savefig('multi_panel.png', dpi=150, bbox_inches='tight')
plt.close(fig)
```

## Scientific Figure Templates

```python
# Error bars
fig, ax = plt.subplots()
ax.errorbar(x, y_mean, yerr=y_std, fmt='o-', capsize=4, capthick=1.5, label='Experiment')
ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)
fig.savefig('errorbar.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# Box plot
fig, ax = plt.subplots()
bp = ax.boxplot([group1, group2, group3], labels=['Ctrl', 'Treatment A', 'Treatment B'],
                patch_artist=True, showmeans=True)
fig.savefig('boxplot.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# Violin plot
fig, ax = plt.subplots()
vp = ax.violinplot([group1, group2, group3], showmeans=True, showmedians=True)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['Ctrl', 'Treatment A', 'Treatment B'])
fig.savefig('violin.png', dpi=300, bbox_inches='tight')
plt.close(fig)
```

## Saving Figures

```python
fig.savefig('figure.png', dpi=300, bbox_inches='tight')   # raster
fig.savefig('figure.svg', bbox_inches='tight')              # vector (editable)
fig.savefig('figure.pdf', bbox_inches='tight')              # vector (print-ready)
```

## Journal-Quality Figure Standards

**Sizing presets (width x height):**
- single_column: `(8.5/2.54, 7/2.54)` — 8.5 x 7 cm
- one_half_column: `(12/2.54, 9/2.54)` — 12 x 9 cm
- double_column: `(17.5/2.54, 10/2.54)` — 17.5 x 10 cm
- presentation: `(25/2.54, 18/2.54)` — 25 x 18 cm

**Journal color palettes:**
```python
PALETTES = {
    'NPG': ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85"],
    'Lancet': ["#00468B", "#ED0000", "#42B540", "#0099B4", "#925E9F", "#FDAF91", "#AD002A", "#ADB6B6"],
    'JCO': ["#0073C2", "#EFC000", "#868686", "#CD534C", "#7AA6DC", "#003C67", "#8F7700", "#3B3B3B"],
    'NEJM': ["#BC3C29", "#0072B5", "#E18727", "#20854E", "#7876B1", "#6F99AD", "#FFDC91", "#EE4C97"],
}
```

**File naming:** Use descriptive names a human can understand months later:
- `km_survival_thbs2_high_vs_low.png` (not `figure1.png`)
- `volcano_plot_deseq2_tumor_vs_normal.png` (not `plot.png`)
- `forest_plot_meta_analysis.pdf` (not `result.pdf`)

## Best Practices

1. Always use `matplotlib.use('Agg')` before importing `pyplot` for headless environments.
2. Use `fig, ax = plt.subplots()` (OO interface) instead of `plt.plot()` (state machine).
3. Call `plt.close(fig)` after saving to free memory.
4. Use `bbox_inches='tight'` to avoid clipped labels.
5. Set `dpi=300` for publication figures, `dpi=150` for screen.
6. Use colormaps from `matplotlib.colormaps` (avoid jet; prefer viridis, coolwarm).
7. **Never save to `/tmp/`.** Save to the project workspace directory for persistence.
8. Always report the full output path after saving so the user can find the file.
