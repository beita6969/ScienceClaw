---
name: genome-analysis
description: Performs genomics analyses including gene expression profiling, BLAST sequence alignment, GWAS interpretation, variant calling, and genome assembly tasks; trigger when the user mentions DNA/RNA sequences, SNPs, gene panels, or comparative genomics.
---

## When to Trigger

Activate this skill when the user mentions any of the following:
- BLAST, sequence alignment, homology search
- Gene expression, RNA-seq, differential expression, DESeq2, edgeR
- GWAS, SNP, variant calling, VCF files
- Genome assembly, annotation, scaffolding
- Phylogenomics, comparative genomics, synteny
- Genotyping, haplotype analysis, linkage disequilibrium

## Step-by-Step Methodology

1. **Clarify the organism and genome build** - Confirm species, reference genome version (e.g., GRCh38 for human, GRCm39 for mouse), and data type (WGS, WES, RNA-seq, microarray).
2. **Data ingestion and QC** - Check raw data quality (FastQC metrics, read depth, coverage). Flag low-quality samples before proceeding.
3. **Alignment / Assembly** - For alignment tasks, specify the aligner (BWA-MEM2, STAR for RNA-seq, minimap2 for long reads). For de novo assembly, recommend assemblers (SPAdes, Flye, hifiasm).
4. **Variant calling / Expression quantification** - Use GATK HaplotypeCaller or DeepVariant for variants; featureCounts or Salmon for transcript quantification.
5. **Statistical analysis** - Apply appropriate multiple-testing correction (Bonferroni, BH-FDR). For GWAS, use mixed models (BOLT-LMM, SAIGE) to handle population structure.
6. **Annotation and interpretation** - Annotate variants with VEP/ANNOVAR; enrich gene lists with GO, KEGG, Reactome pathways.
7. **Visualization** - Generate Manhattan plots (GWAS), volcano plots (DE), circos plots (structural variants), or heatmaps (expression clusters).

## Key Databases and Tools

- **NCBI GenBank / RefSeq** - Reference sequences and annotations
- **Ensembl / UCSC Genome Browser** - Genome browsing and tracks
- **BLAST (NCBI)** - Sequence similarity search
- **UniProt** - Protein function annotation
- **ClinVar / gnomAD** - Clinical variant interpretation
- **KEGG / Reactome / Gene Ontology** - Pathway and functional enrichment
- **GEO / ArrayExpress** - Public expression datasets

## Output Format

- Provide results in structured tables (gene, log2FC, p-value, adjusted p-value).
- Include publication-quality figure descriptions with axis labels and legends.
- Report genome coordinates in standard notation (chr:start-end, 1-based).
- Always state the reference genome build used.

## Quality Checklist

- [ ] Reference genome build explicitly stated
- [ ] Multiple-testing correction applied and method named
- [ ] Sample sizes and statistical power discussed
- [ ] QC metrics reported (mapping rate, duplication rate, coverage)
- [ ] Biological vs. statistical significance distinguished
- [ ] All gene identifiers use standard nomenclature (HGNC symbols for human)
- [ ] Effect sizes reported alongside p-values
- [ ] Reproducibility: exact tool versions and parameters documented
