---
name: scanpy-singlecell
description: "Single-cell RNA-seq analysis with scanpy and anndata. Use when: (1) scRNA-seq preprocessing and QC, (2) clustering and cell type annotation, (3) differential expression analysis, (4) trajectory and pseudotime analysis, (5) UMAP/tSNE visualization. NOT for: bulk RNA-seq differential expression (use DESeq2/edgeR), protein structure analysis (use pdb-structure), or imaging data."
metadata: { "openclaw": { "emoji": "🧬", "requires": { "bins": ["python3"] }, "install": [{ "id": "uv-scanpy", "kind": "uv", "package": "scanpy anndata" }] } }
---

# Scanpy Single-Cell Analysis

Single-cell RNA-seq analysis using scanpy and anndata for preprocessing,
clustering, differential expression, and visualization.

## When to Use

- scRNA-seq data preprocessing and quality control
- Cell clustering and cell type annotation
- Differential expression between cell groups
- Trajectory inference and pseudotime analysis
- Dimensionality reduction visualization (UMAP, tSNE)
- Integration of multiple scRNA-seq datasets

## When NOT to Use

- Bulk RNA-seq differential expression (use DESeq2 or edgeR)
- Protein structure prediction or analysis (use pdb-structure)
- Imaging or spatial transcriptomics without companion tools
- General-purpose statistics (use scipy-analysis)

## Reading Data and Preprocessing

```python
import scanpy as sc

# Read data: h5ad, 10x mtx, or 10x h5
adata = sc.read_h5ad('dataset.h5ad')
adata = sc.read_10x_mtx('filtered_feature_bc_matrix/', var_names='gene_symbols')

# QC filtering
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, inplace=True)
adata = adata[adata.obs.pct_counts_mt < 20, :]

# Normalize, log-transform, select HVGs
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata  # preserve full gene set for DE
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var.highly_variable]
sc.pp.scale(adata, max_value=10)
```

## Dimensionality Reduction and Clustering

```python
# PCA
sc.tl.pca(adata, svd_solver='arpack', n_comps=50)
sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True)

# Neighborhood graph and clustering
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)

# Alternative: Louvain clustering
sc.tl.louvain(adata, resolution=0.8)

# tSNE (alternative to UMAP)
sc.tl.tsne(adata, n_pcs=40)
```

## Differential Expression

```python
# Rank genes per cluster (Wilcoxon is recommended)
sc.tl.rank_genes_groups(adata, groupby='leiden', method='wilcoxon', use_raw=True)
sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False)

# Extract DE results as DataFrame
result = adata.uns['rank_genes_groups']
de_df = sc.get.rank_genes_groups_df(adata, group='0')
de_df_filtered = de_df[(de_df['pvals_adj'] < 0.05) & (de_df['logfoldchanges'].abs() > 1)]

# Compare specific groups
sc.tl.rank_genes_groups(adata, groupby='leiden', groups=['0'], reference='1',
                        method='wilcoxon', use_raw=True)
```

## Visualization

```python
# UMAP colored by cluster
sc.pl.umap(adata, color=['leiden'], frameon=False, save='_clusters.pdf')

# UMAP colored by gene expression
sc.pl.umap(adata, color=['CST3', 'NKG7', 'MS4A1'], frameon=False)

# Dot plot for marker genes across clusters
marker_genes = ['CD3D', 'CD79A', 'CST3', 'NKG7', 'PPBP']
sc.pl.dotplot(adata, marker_genes, groupby='leiden', save='_markers.pdf')

# Stacked violin plot
sc.pl.stacked_violin(adata, marker_genes, groupby='leiden', rotation=90)

# Heatmap of top DE genes
sc.pl.rank_genes_groups_heatmap(adata, n_genes=5, groupby='leiden', show_gene_labels=True)

# Matrix plot
sc.pl.matrixplot(adata, marker_genes, groupby='leiden', standard_scale='var')
```

## Trajectory Analysis

```python
# Diffusion map and pseudotime
sc.tl.diffmap(adata)
sc.tl.dpt(adata, n_dcs=10)

# PAGA (partition-based graph abstraction)
sc.tl.paga(adata, groups='leiden')
sc.pl.paga(adata, plot=True, threshold=0.03)
sc.tl.umap(adata, init_pos='paga')
```

## Best Practices

1. Always start with QC: filter low-quality cells and doublets before analysis.
2. Use `adata.raw` to preserve full gene set for DE testing after subsetting HVGs.
3. Prefer Leiden over Louvain clustering (better modularity optimization).
4. Use Wilcoxon rank-sum for DE; it is robust and non-parametric.
5. Save intermediate results with `adata.write('checkpoint.h5ad')`.
6. Set `sc.settings.figdir` and use `save=` parameter for reproducible figures.
7. Use `sc.logging.print_versions()` to record environment for reproducibility.
8. Adjust `resolution` parameter in Leiden to control cluster granularity.
