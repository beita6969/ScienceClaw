---
name: social-science-research
description: "Orchestrates a social science research workflow from literature review through data collection, text analysis, statistical modeling, and report generation. Use when conducting empirical social science research, policy analysis, or mixed-methods studies. NOT for pure natural science analysis or clinical trial data."
metadata: { "openclaw": { "emoji": "📈" } }
---

# Social Science Research (Meta Skill)

This meta-skill coordinates a complete social science research pipeline by
integrating literature discovery, data collection, natural language processing,
statistical analysis, and academic report writing. It combines four specialized
skills to support rigorous empirical research across economics, political
science, sociology, and related disciplines.

## Workflow

### Step 1: Literature Review and Theoretical Framing

Search SSRN and CrossRef for relevant academic papers on the research topic:
- Keyword and author-based searches across working paper repositories
- Citation network exploration to identify seminal and recent contributions
- Extraction of theoretical frameworks, hypotheses, and methodological approaches
- Identification of gaps in existing literature that motivate the study
- Construction of an annotated bibliography with key findings per source

### Step 2: Data Collection and Preparation

Gather quantitative data from established sources:
- **World Bank**: Development indicators, governance metrics, trade data
- **Census/Survey**: Demographic, labor market, household-level data
- **Custom datasets**: User-provided CSV, Excel, or API-sourced data
- Data cleaning: handle missing values, outliers, encoding inconsistencies
- Variable construction: create indices, interaction terms, lagged variables
- Descriptive statistics: summary tables, distributions, correlation matrices

### Step 3: Qualitative Text Analysis

Apply spaCy NLP tools to analyze qualitative or textual data sources:
- Named entity recognition to extract people, organizations, locations
- Document classification by topic, sentiment, or policy domain
- Keyword extraction and frequency analysis across corpora
- Relationship extraction between entities in policy documents
- Coding assistance: map text segments to predefined thematic categories

### Step 4: Statistical Analysis and Causal Inference

Use statsmodels for rigorous quantitative analysis:
- **Regression**: OLS, logistic, Poisson, negative binomial models
- **Panel data**: Fixed effects, random effects, difference-in-differences
- **Causal inference**: Instrumental variables, regression discontinuity, propensity score matching
- **Time series**: ARIMA, VAR, cointegration analysis
- **Robustness**: Alternative specifications, placebo tests, sensitivity analysis
- Report standard errors appropriate to the data structure (clustered, HAC)

### Step 5: Visualization and Figure Preparation

Generate publication-quality figures with matplotlib:
- Coefficient plots with confidence intervals
- Time series trend charts with event markers
- Geographic maps for spatial data (choropleth, point maps)
- Distribution comparisons (kernel density, box plots, violin plots)
- Regression diagnostic plots (residuals, Q-Q, leverage)

### Step 6: Academic Report Generation

Compile findings into a structured academic report:
- Abstract summarizing research question, method, and key findings
- Introduction with literature context and contribution statement
- Data section with source descriptions and summary statistics
- Methodology section with model specifications and identification strategy
- Results with tables, figures, and interpretation
- Discussion of limitations, policy implications, and future directions
- Properly formatted references and appendices

## Integration Points

- **ssrn-econpapers** -- Working paper search, citation discovery, literature mapping
- **world-bank-data** -- Development indicators, cross-country panel data, time series
- **statsmodels-stats** -- Regression analysis, panel methods, causal inference, diagnostics
- **spacy-nlp** -- Entity recognition, text classification, keyword extraction, coding support

## Output Formats

- **Literature table**: Paper title, authors, year, method, key finding, relevance score
- **Data summary**: Variable descriptions, summary statistics, sample sizes
- **Regression table**: Coefficients, standard errors, significance, R-squared, diagnostics
- **Figures**: Publication-ready plots with labeled axes, legends, and annotations
- **Report draft**: Structured academic document with sections and citations

## Best Practices

1. Define the research question and identification strategy before touching data
2. Pre-register hypotheses when conducting confirmatory analysis
3. Use consistent variable definitions across all analyses
4. Report all specifications tested, not only those yielding significant results
5. Cluster standard errors at the appropriate level of treatment assignment
6. Include balance tests and pre-trend checks for quasi-experimental designs
7. Triangulate quantitative findings with qualitative evidence where possible
8. Use multiple imputation or bounds analysis for missing data sensitivity
9. Follow disciplinary reporting standards (APA, AER, APSR as appropriate)
10. Clearly distinguish correlation from causation in all interpretive sections
