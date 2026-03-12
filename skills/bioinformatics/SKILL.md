---
name: bioinformatics
description: Performs bioinformatics analyses including pathway enrichment, gene ontology analysis, protein-protein interaction networks, multi-omics integration, and biological sequence database querying; trigger when users discuss gene sets, biological pathways, functional annotation, or omics data integration.
---

## When to Trigger

Activate this skill when the user mentions:
- Pathway analysis, KEGG, Reactome, WikiPathways
- Gene Ontology (GO) enrichment, biological process, molecular function
- Protein-protein interaction (PPI) networks, STRING, BioGRID
- Multi-omics integration (transcriptomics + proteomics + metabolomics)
- Gene set enrichment analysis (GSEA), over-representation analysis (ORA)
- Sequence databases, UniProt, NCBI, Ensembl queries
- Single-cell RNA-seq analysis, clustering, trajectory inference

## Step-by-Step Methodology

1. **Data preparation** - Standardize gene/protein identifiers (convert to Entrez, Ensembl, or UniProt IDs as needed). Remove duplicates and handle ambiguous mappings. Verify organism and genome build.
2. **Differential analysis** - For transcriptomics: DESeq2 or edgeR (count data), limma-voom (normalized). For proteomics: limma with appropriate normalization. Apply multiple testing correction (BH-FDR). Set thresholds (|log2FC| > 1, padj < 0.05 as defaults, adjustable).
3. **Functional enrichment** - Perform GO enrichment (BP, MF, CC) using clusterProfiler, g:Profiler, or DAVID. Run KEGG/Reactome pathway enrichment. Use GSEA for ranked gene lists (no arbitrary cutoff). Report enriched terms with gene ratio, p-value, adjusted p-value, and gene members.
4. **Network analysis** - Build PPI networks from STRING (confidence > 0.7 for high confidence). Identify hub genes (degree centrality), bottleneck nodes (betweenness centrality), and functional modules (MCODE, Louvain clustering). Overlay expression data on network.
5. **Multi-omics integration** - For paired omics: correlation analysis, canonical correlation (CCA), or MOFA/DIABLO. Map features across omics layers using shared identifiers or known biological connections. Identify convergent pathways.
6. **Single-cell analysis** - QC filtering (genes/cell, UMI/cell, mitochondrial %). Normalization (scran, SCTransform). Dimensionality reduction (PCA, UMAP). Clustering (Leiden, Louvain). Cell type annotation (SingleR, scType, marker genes). Trajectory inference (Monocle3, Slingshot).
7. **Visualization** - Generate volcano plots, heatmaps (with hierarchical clustering), dot plots (enrichment), network diagrams, UMAP/tSNE plots (single-cell), and circos plots (multi-omics).

## Key Databases and Tools

- **Gene Ontology (GO)** - Functional annotations
- **KEGG / Reactome / WikiPathways** - Pathway databases
- **STRING / BioGRID / IntAct** - PPI databases
- **Ensembl / NCBI / UniProt** - Sequence and annotation databases
- **clusterProfiler / g:Profiler / DAVID** - Enrichment tools
- **Seurat / Scanpy** - Single-cell analysis frameworks
- **Cytoscape** - Network visualization

## Output Format

- Enrichment results as tables: term, description, gene ratio, p-value, padj, gene list.
- Volcano plots with labeled significant genes and fold-change thresholds.
- Network figures with node coloring (expression), size (degree), and module highlighting.
- UMAP/tSNE plots with cluster labels and cell type annotations.
- Heatmaps with dendrograms and annotation bars.

## Quality Checklist

- [ ] Gene ID mapping verified (conversion losses reported)
- [ ] Background gene set appropriate for enrichment analysis
- [ ] Multiple testing correction applied (BH-FDR or equivalent)
- [ ] Redundant GO terms handled (semantic similarity, REVIGO)
- [ ] Network confidence threshold specified and justified
- [ ] Single-cell QC thresholds documented
- [ ] Batch effects assessed and corrected if present
- [ ] Results cross-validated across databases or methods
- [ ] Biological interpretation grounded in literature
