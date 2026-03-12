---
name: reproducibility
description: Promotes research reproducibility through guidance on pre-registration, open data/code sharing, replication study design, computational reproducibility, and open science best practices; trigger when users discuss replication, open science, pre-registration, data sharing, or research transparency.
---

## When to Trigger

Activate this skill when the user mentions:
- Reproducibility, replicability, replication crisis
- Pre-registration, registered reports, AsPredicted
- Open data, data sharing, FAIR principles
- Open code, computational reproducibility, Docker, containers
- Open access publishing, preprints, green/gold OA
- Research transparency, open science framework (OSF)
- P-hacking, HARKing, questionable research practices
- Power analysis for replication, replication study design

## Step-by-Step Methodology

1. **Assess current reproducibility state** - Evaluate the research against reproducibility dimensions: methodological (sufficient detail to replicate), computational (code + data + environment = same results), and results reproducibility (independent replication yields consistent findings). Identify specific gaps.
2. **Pre-registration** - Guide pre-registration of hypotheses, methods, and analysis plan BEFORE data collection. Use appropriate platform: OSF Registries, AsPredicted, ClinicalTrials.gov (clinical), or PROSPERO (systematic reviews). Distinguish confirmatory from exploratory analyses.
3. **Data management** - Apply FAIR principles: Findable (persistent identifier, metadata), Accessible (open or controlled access with clear process), Interoperable (standard formats, vocabularies), Reusable (license, provenance). Create data dictionary documenting every variable. Use tidy data formats.
4. **Code and computational environment** - Share analysis code in a public repository (GitHub, GitLab, Zenodo for DOI). Document dependencies with requirements.txt, renv.lock, or conda environment.yml. For full reproducibility: containerize with Docker or use Binder. Include README with execution instructions.
5. **Replication study design** - For direct replication: match original methods as closely as possible. For conceptual replication: test same hypothesis with different methods. Conduct power analysis based on original effect size (use safeguard power: assume smaller effect). Determine sample size for meaningful replication test (use equivalence testing or Bayesian replication factors).
6. **Reporting transparency** - Follow reporting guidelines (CONSORT, STROBE, ARRIVE, PRISMA). Report all pre-specified analyses regardless of results. Clearly label exploratory analyses. Share full materials (stimuli, protocols, instruments) as supplementary files.
7. **Open science practices** - Adopt open science badges (data, materials, pre-registration). Consider registered reports format (peer review before results). Use preprint servers (bioRxiv, medRxiv, arXiv, SSRN). Choose open access publication route.

## Key Platforms and Tools

- **OSF (Open Science Framework)** - Project management and pre-registration
- **AsPredicted** - Streamlined pre-registration
- **Zenodo** - Data and code archival with DOI
- **GitHub / GitLab** - Code version control and sharing
- **Docker / Binder** - Computational environment reproducibility
- **FAIR self-assessment tool** - Data FAIRness evaluation
- **COS (Center for Open Science)** - Reproducibility guidelines

## Output Format

- Reproducibility assessment: checklist of current state vs. best practices.
- Pre-registration template: hypotheses, design, sample, variables, analysis plan.
- Data sharing package: dataset + data dictionary + codebook + license + README.
- Computational reproducibility: repository structure, Dockerfile, execution instructions.
- Replication study protocol: power analysis, design, success criteria (equivalence test bounds or replication Bayes factor thresholds).

## Quality Checklist

- [ ] Pre-registration completed before data collection/analysis
- [ ] Confirmatory and exploratory analyses clearly distinguished
- [ ] Data deposited in trusted repository with persistent identifier (DOI)
- [ ] FAIR principles self-assessment completed
- [ ] Analysis code shared and tested on a clean environment
- [ ] Computational environment documented or containerized
- [ ] All materials sufficient for independent replication
- [ ] Reporting guideline checklist completed
- [ ] License specified for data (CC-BY, CC0) and code (MIT, Apache)
- [ ] Deviations from pre-registration documented and justified
