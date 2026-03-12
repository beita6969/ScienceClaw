---
name: data-visualization-expert
description: COPYRIGHT NOTICE
---

<!--
# COPYRIGHT NOTICE
# This file is part of the "Universal Biomedical Skills" project.
# Copyright (c) 2026 MD BABU MIA, PhD <md.babu.mia@mssm.edu>
# All Rights Reserved.
#
# This code is proprietary and confidential.
# Unauthorized copying of this file, via any medium is strictly prohibited.
#
# Provenance: Authenticated by MD BABU MIA

-->

---
name: data-visualization-expert
description: Generate insightful, publication-quality visualizations from complex datasets.
keywords:
  - charts
  - plots
  - analysis
  - pandas
  - matplotlib
  - seaborn
measurable_outcome: Create 3 high-resolution (300dpi) statistical plots (volcano, heatmap, scatter) within 15 minutes.
license: MIT
metadata:
  author: AI Agentic Skills Team
  version: "2.0.0"
compatibility:
  - system: linux, macos
allowed-tools:
  - run_shell_command
  - write_file
  - read_file
---

# Data Visualization Expert

A dedicated skill for transforming raw data (CSV, JSON, Excel) into compelling visual narratives. Specializes in statistical and scientific plotting.

## When to Use
- **Reports:** Summarizing key metrics or KPIs.
- **Exploration:** Initial data analysis (EDA) to find trends/outliers.
- **Publication:** Generating figures for papers or presentations.
- **Comparison:** Comparing models, cohorts, or experimental groups.

## Core Capabilities
1.  **Code Generation:** Creates Python scripts (Matplotlib, Seaborn, Plotly) or R code (ggplot2).
2.  **Style Enforcement:** Adheres to specific journal/company branding (fonts, colors).
3.  **Data Cleaning:** Preprocesses data (handle missing values, normalize) for plotting.
4.  **Artifact Management:** Saves plots as PNG/SVG/PDF files.

## Workflow
1.  **Load Data:** Read input file (`pd.read_csv()`) and inspect columns/types.
2.  **Clean & Transform:** Filter, pivot, or aggregate data as needed.
3.  **Generate Plot:** Write plotting script with strict aesthetic controls.
4.  **Save & Verify:** Execute script, check output file existence/size.

## Example Usage
```bash
# Agent prompt:
"Visualize the distribution of 'Age' vs 'Income' from customers.csv"
# Triggers generation of `plot_age_income.py` using Seaborn scatterplot.
```

## Guardrails
- **Privacy:** Avoid plotting PII (names, emails) directly.
- **Accuracy:** Ensure axes are labeled correctly with units.
- **Readability:** Use appropriate scales (log vs linear) and avoid clutter.


<!-- AUTHOR_SIGNATURE: 9a7f3c2e-MD-BABU-MIA-2026-MSSM-SECURE -->