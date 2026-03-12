---
name: scienceclaw-ie
description: "Extract structured information from scientific texts: entities, relations, data tables, methods, results. Use when: (1) parsing papers for key data, (2) extracting experimental parameters, (3) building knowledge graphs from literature, (4) NER on scientific documents, (5) extracting methods/results sections. NOT for: summarization (use scienceclaw-summarization), full text retrieval (use scienceclaw-retrieval)."
metadata: { "openclaw": { "emoji": "🔍" } }
---

# ScienceCLAW Information Extraction

Extract structured information from scientific texts including named entities, relations, data tables, methods, results, and experimental parameters.

## When to Use

Use this skill when the user:

- Needs to parse a scientific paper for key data points (measurements, parameters, outcomes)
- Wants to extract experimental methods, conditions, or protocols from a paper
- Needs named entity recognition (NER) on scientific documents (chemicals, genes, diseases, etc.)
- Wants to build a knowledge graph or structured database from literature
- Needs to extract relations between scientific entities (drug-target, gene-disease, cause-effect)
- Wants structured tables from unstructured scientific text
- Needs to compare methods or results across multiple papers in a structured format
- Wants to extract metadata from papers (authors, affiliations, funding, conflicts of interest)

## When NOT to Use

Do not use this skill when:

- The user wants a narrative summary of a paper (use scienceclaw-summarization)
- The user needs to find and retrieve papers (use scienceclaw-retrieval)
- The user wants to answer a scientific question (use scienceclaw-qa)
- The user needs multi-step reasoning or proof construction (use scienceclaw-reasoning)
- The user needs code-based data analysis (use code-execution)

## Entity Types by Discipline

### Biomedical Sciences

| Entity Type | Examples | Common Formats |
|---|---|---|
| Gene/Protein | TP53, BRCA1, insulin | Uppercase symbols, UniProt IDs |
| Disease | Type 2 diabetes, glioblastoma | MeSH terms, ICD codes |
| Drug/Chemical | Metformin, aspirin, NaCl | IUPAC names, brand names, CAS numbers |
| Organism | Homo sapiens, E. coli | Binomial nomenclature, NCBI Taxonomy ID |
| Cell Line | HeLa, HEK293T | ATCC identifiers |
| Anatomical Structure | hippocampus, mitochondria | UBERON ontology terms |
| Biological Process | apoptosis, glycolysis | GO terms |
| Dosage | 10 mg/kg, 500 nM | Value + unit patterns |
| Biomarker | HbA1c, PSA, CRP | Abbreviations with reference ranges |

### Chemistry and Materials Science

| Entity Type | Examples | Common Formats |
|---|---|---|
| Chemical Compound | benzene, polyethylene glycol | IUPAC, SMILES, InChI |
| Material | graphene, Ti-6Al-4V, PDMS | Common names, composition formulas |
| Property | melting point, band gap, tensile strength | Property name + value + unit |
| Synthesis Method | sol-gel, CVD, electrospinning | Method name + parameters |
| Characterization Technique | XRD, SEM, NMR, FTIR | Acronyms with parameters |
| Crystal Structure | FCC, BCC, hexagonal | Space group, lattice parameters |
| Catalyst | Pd/C, TiO2, zeolite | Composition + support |

### Physics and Astronomy

| Entity Type | Examples | Common Formats |
|---|---|---|
| Physical Quantity | 3.14 eV, 300 K, 1.5 T | Value + SI/derived unit |
| Particle | electron, muon, Higgs boson | Standard Model names |
| Astronomical Object | NGC 1277, Proxima Centauri | Catalog designations |
| Physical Constant | c, h, G, k_B | Symbol + numerical value + unit |
| Equation/Law | Schrodinger equation, F = ma | Named equations, formulas |
| Experimental Apparatus | LHC, LIGO, HST | Acronyms and facility names |
| Measurement Technique | spectroscopy, interferometry | Technique name + configuration |

### Social Sciences

| Entity Type | Examples | Common Formats |
|---|---|---|
| Study Design | RCT, cohort study, survey | Methodology descriptors |
| Statistical Measure | p = 0.03, r = 0.45, d = 0.8 | Statistic symbol + value |
| Sample | N = 1,200 adults aged 18-65 | Size + demographic descriptors |
| Instrument/Scale | Likert scale, BDI-II, MMSE | Named instruments with citations |
| Effect | income inequality, voter turnout | Outcome variable descriptions |
| Theory/Framework | rational choice, social learning | Named theoretical frameworks |
| Policy/Intervention | minimum wage increase, CBT | Intervention descriptions |
| Geographic Scope | United States, OECD countries | Country/region names |
| Time Period | 2010-2020, post-WWII | Date ranges, era references |

### Computer Science

| Entity Type | Examples | Common Formats |
|---|---|---|
| Algorithm | Adam optimizer, BERT, ResNet-50 | Named algorithms with versions |
| Dataset | ImageNet, MNIST, GLUE | Named benchmarks |
| Metric | F1 = 0.92, accuracy = 95.3%, BLEU = 32.1 | Metric name + value |
| Architecture | transformer, CNN, GAN | Architecture type names |
| Hyperparameter | learning rate = 1e-4, batch size = 32 | Parameter name + value |
| Framework/Library | PyTorch, TensorFlow, scikit-learn | Software names with versions |
| Hardware | A100 GPU, TPU v4 | Hardware specifications |

## Relation Schemas

### Core Relation Types

| Relation | Description | Example |
|---|---|---|
| TREATS | Drug/intervention treats disease | (Metformin, TREATS, Type 2 diabetes) |
| CAUSES | Agent causes effect | (Smoking, CAUSES, Lung cancer) |
| INHIBITS | Entity inhibits another | (Aspirin, INHIBITS, COX-2) |
| ACTIVATES | Entity activates another | (EGF, ACTIVATES, EGFR) |
| PART_OF | Component is part of whole | (Hippocampus, PART_OF, Limbic system) |
| MEASURED_BY | Property measured by technique | (Crystal structure, MEASURED_BY, XRD) |
| SYNTHESIZED_VIA | Material made by method | (Graphene, SYNTHESIZED_VIA, CVD) |
| ASSOCIATED_WITH | Statistical association | (Gene X, ASSOCIATED_WITH, Disease Y) |
| OUTPERFORMS | Method A outperforms B | (BERT, OUTPERFORMS, LSTM on GLUE) |
| DERIVED_FROM | Result derived from data/method | (Estimate, DERIVED_FROM, Meta-analysis) |
| USED_IN | Tool/method used in study | (fMRI, USED_IN, Smith et al. 2023) |
| CONTRADICTS | Finding contradicts another | (Study A, CONTRADICTS, Study B) |
| SUPPORTS | Finding supports another | (Experiment, SUPPORTS, Hypothesis) |

### Relation Attributes

Each extracted relation should include:

- **Source**: The text span or sentence from which the relation was extracted
- **Confidence**: High / Medium / Low based on explicitness of the statement
- **Direction**: Whether the relation is directional or bidirectional
- **Qualifier**: Any conditions or modifiers (e.g., "in vitro", "at high doses", "in Western populations")
- **Evidence type**: Direct statement, implication, or inference

## Output Formats

### JSON Output

For programmatic consumption and knowledge graph construction:

```json
{
  "document": {
    "title": "Paper title",
    "doi": "10.xxxx/xxxxx",
    "authors": ["Author A", "Author B"]
  },
  "entities": [
    {
      "id": "E1",
      "text": "metformin",
      "type": "Drug",
      "normalized": "CHEMBL1431",
      "spans": [{"start": 145, "end": 154}]
    }
  ],
  "relations": [
    {
      "id": "R1",
      "type": "TREATS",
      "subject": "E1",
      "object": "E2",
      "confidence": "high",
      "qualifier": "first-line therapy",
      "source_sentence": "Metformin is the first-line therapy for T2D."
    }
  ],
  "extracted_data": {
    "methods": {},
    "results": {},
    "parameters": {}
  }
}
```

### Markdown Table Output

For human-readable structured summaries:

```markdown
| Parameter | Value | Unit | Conditions | Source (Section) |
|---|---|---|---|---|
| Temperature | 350 | C | Under N2 atmosphere | Methods 2.3 |
| Pressure | 1.5 | atm | At steady state | Methods 2.3 |
| Yield | 87.3 | % | After 24h reaction | Results 3.1 |
```

### Knowledge Graph Triples

For direct ingestion into graph databases:

```
(Metformin)-[TREATS {confidence: high}]->(Type 2 Diabetes)
(Metformin)-[INHIBITS {confidence: high}]->(Hepatic Gluconeogenesis)
(Metformin)-[ACTIVATES {confidence: medium}]->(AMPK)
```

## Extraction Templates

### Experimental Methods Extraction

Extract the following from methods sections:

```yaml
methods:
  study_design: [RCT | cohort | case-control | cross-sectional | in vitro | in vivo | simulation]
  sample:
    size: [N]
    description: [population/material description]
    inclusion_criteria: [list]
    exclusion_criteria: [list]
  intervention:
    name: [intervention name]
    dose: [dose with units]
    duration: [time period]
    route: [administration route]
  control:
    type: [placebo | active comparator | no treatment | baseline]
    description: [control description]
  measurements:
    primary_outcome: [outcome measure]
    secondary_outcomes: [list]
    instruments: [measurement tools/techniques]
    timepoints: [when measured]
  statistical_analysis:
    methods: [t-test, ANOVA, regression, etc.]
    software: [R, SPSS, Python, etc.]
    significance_threshold: [alpha level]
```

### Results Extraction

Extract quantitative results in structured form:

```yaml
results:
  primary_outcome:
    measure: [outcome name]
    treatment_group: [value +/- SD or 95% CI]
    control_group: [value +/- SD or 95% CI]
    effect_size: [value with CI]
    p_value: [value]
    significance: [significant | not significant]
  secondary_outcomes:
    - measure: [name]
      value: [result]
      p_value: [value]
  adverse_events:
    - event: [description]
      frequency: [n/N or percentage]
      severity: [mild | moderate | severe]
  key_figures:
    - figure_number: [Fig. N]
      description: [what is shown]
      key_finding: [main takeaway]
  key_tables:
    - table_number: [Table N]
      description: [what is tabulated]
      key_values: [most important entries]
```

### Paper Metadata Extraction

```yaml
metadata:
  title: [full title]
  authors: [list with affiliations]
  journal: [journal name]
  year: [publication year]
  doi: [DOI]
  keywords: [author keywords]
  abstract_summary: [one-sentence summary]
  study_type: [primary research | review | meta-analysis | case report | editorial]
  funding: [funding sources]
  conflicts: [declared conflicts of interest]
  data_availability: [statement about data sharing]
  code_availability: [statement about code sharing]
```

## Extraction Quality Guidelines

### Accuracy Standards

- Extract only information explicitly stated in the text (do not infer or hallucinate)
- Preserve exact numerical values including units and uncertainty
- Flag ambiguous extractions with a confidence indicator
- When multiple interpretations are possible, extract all with annotations
- Distinguish between author claims and independently verified facts

### Completeness Standards

- Extract all instances of the target entity/relation type in the document
- Cover all relevant sections (abstract, introduction, methods, results, discussion, supplementary)
- Note when information is missing or not reported ("NR" = not reported)
- Flag when the text references supplementary materials not included in the extraction scope

### Normalization Standards

- Normalize entity names to canonical forms where possible (e.g., gene symbols to HGNC)
- Convert units to SI when practical, noting the original units
- Resolve abbreviations to full forms on first occurrence
- Link entities to standard identifiers (DOI, PubChem CID, UniProt ID, etc.) when identifiable

## Multi-Document Extraction

When extracting from multiple papers for comparison:

1. **Align schemas**: Use consistent entity types and relation schemas across all documents
2. **Create comparison tables**: Organize results in parallel columns by study
3. **Flag contradictions**: Highlight where different papers report conflicting values or conclusions
4. **Aggregate where appropriate**: Compute ranges, means, or consensus values across studies
5. **Track provenance**: Every extracted value must be traceable to its source document and section

## Integration with Other Skills

- **scienceclaw-retrieval**: Retrieve papers that serve as input for extraction
- **scienceclaw-qa**: Provide extracted data as evidence for answering questions
- **scienceclaw-reasoning**: Supply extracted premises and data for reasoning chains
- **code-execution**: Post-process extracted data (statistics, visualization, database loading)
- **scienceclaw-summarization**: Extraction provides structured input; summarization provides narrative output

## Zero-Hallucination Rule

ALL factual claims, citations, database results, and scientific data presented to the user MUST come from actual tool results (API calls, code execution, web search) in this conversation. NEVER fabricate or "fill in" details from training data. If a tool returns no results or partial data, report exactly what happened.
