---
name: genomics-analysis
description: "Orchestrates a genomics analysis workflow from gene query through expression analysis to pathway enrichment. Use when investigating gene function, analyzing expression data, or performing pathway-level interpretation. NOT for pure protein structure modeling or drug-target interaction analysis."
metadata: { "openclaw": { "emoji": "🧬" } }
---

# Genomics Analysis (Meta Skill)

This meta-skill coordinates a complete genomics analysis pipeline by integrating
gene database queries, sequence analysis, expression profiling, and pathway
enrichment into a unified workflow. It combines three specialized skills to
deliver comprehensive gene-level and systems-level biological insights.

## Workflow

### Step 1: Gene Information Retrieval

Query NCBI Entrez for comprehensive gene details including official nomenclature,
genomic coordinates, transcript variants, and functional annotations. Retrieve
orthologs across model organisms for evolutionary context. Pull known variants
from ClinVar and dbSNP, noting pathogenic or pharmacogenomic associations.
Collect linked references from PubMed for recent literature context.

### Step 2: Sequence Analysis

Use BioPython to perform sequence-level analyses on retrieved gene and protein
sequences:
- Multiple sequence alignment of orthologs to identify conserved regions
- Motif discovery in promoter regions or protein domains
- Domain architecture mapping against Pfam/InterPro signatures
- Codon usage analysis for expression optimization studies
- Variant impact prediction based on conservation scores

### Step 3: Expression Analysis

Apply scanpy for expression data analysis, supporting both single-cell and
bulk RNA-seq workflows:
- For single-cell: quality control, normalization, clustering, marker gene
  identification, cell type annotation
- For bulk: differential expression analysis, volcano plots, heatmaps
- Cross-dataset comparison when multiple conditions are available
- Identification of co-expressed gene modules

### Step 4: Pathway Enrichment and Functional Annotation

Map differentially expressed or co-expressed genes to biological pathways:
- KEGG pathway mapping for metabolic and signaling context
- Gene Ontology enrichment (biological process, molecular function, cellular component)
- Reactome pathway analysis for detailed mechanistic understanding
- Network-based enrichment to identify hub genes and regulatory modules

### Step 5: Integrated Report Generation

Compile findings into a structured report with:
- Gene summary card with key identifiers and annotations
- Sequence conservation highlights and domain maps
- Expression analysis results with statistical summaries
- Enriched pathways ranked by significance
- Key findings synthesis connecting sequence, expression, and pathway data
- Publication-ready figures and supplementary tables

## Integration Points

- **ncbi-entrez** -- Gene records, variant data, orthologs, literature links
- **biopython-bio** -- Sequence alignment, motif search, domain analysis, format conversion
- **scanpy-singlecell** -- Expression quantification, clustering, differential expression, visualization

## Output Formats

- **Gene card**: Symbol, aliases, genomic location, function summary, disease associations
- **Alignment view**: Conserved regions highlighted across orthologs
- **Expression summary**: DE gene lists with fold change, p-values, FDR
- **Pathway table**: Enriched pathways with gene counts, p-values, leading-edge genes
- **Figures**: Heatmaps, volcano plots, UMAP embeddings, pathway diagrams

## Best Practices

1. Start with gene identifiers from a reliable source (NCBI Gene ID or HGNC symbol)
2. Verify gene nomenclature across databases to avoid confusion from aliases
3. Use appropriate normalization for the expression data type (TPM, CPM, SCTransform)
4. Apply multiple testing correction (Benjamini-Hochberg) for all enrichment analyses
5. Set biologically meaningful fold-change thresholds alongside statistical cutoffs
6. Include both up- and down-regulated gene sets in pathway analysis
7. Cross-reference pathway results with known biology to filter spurious enrichments
8. Report effect sizes and confidence intervals, not just p-values
9. Note species differences when translating findings from model organisms
10. Archive intermediate results for reproducibility and downstream re-analysis
