---
name: scientific-writing
description: Assist with scientific paper writing, LaTeX formatting, abstract drafting, review responses, grant proposals, and academic communication. Use when user asks to write/edit a paper section, draft an abstract, format in LaTeX, respond to reviewer comments, write a grant proposal, or improve academic writing. Triggers on "write abstract", "draft introduction", "LaTeX", "reviewer response", "grant proposal", "improve my writing", "paper draft", "methods section".
---

# Scientific Writing

Academic paper composition, LaTeX formatting, and scholarly communication.

## Paper Structure Templates

### IMRaD (standard empirical paper)
1. **Title**: Concise, informative, includes key variables
2. **Abstract**: Background (1-2 sentences) → Objective → Methods → Results → Conclusion (150-300 words)
3. **Introduction**: Broad context → Narrow focus → Gap → Research question/hypothesis
4. **Methods**: Reproducible detail; subsections by procedure
5. **Results**: Findings without interpretation; tables/figures referenced
6. **Discussion**: Summary → Interpretation → Comparison with literature → Limitations → Implications → Future work
7. **References**: Consistent citation style

### Review / Survey Paper
1. Introduction with scope and search methodology
2. Taxonomy / organizational framework
3. Systematic coverage of subtopics
4. Synthesis and comparison
5. Open problems and future directions

### Grant Proposal (NSF/NIH style)
1. Specific Aims (1 page)
2. Significance
3. Innovation
4. Approach (preliminary data, research plan, timeline)
5. Budget justification

## LaTeX Support

### Common Templates

Article:
```latex
\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage[margin=1in]{geometry}
\usepackage{natbib}

\title{Your Title}
\author{Author Name \\ Institution \\ \texttt{email@example.com}}
\date{\today}

\begin{document}
\maketitle
\begin{abstract}
Your abstract here.
\end{abstract}

\section{Introduction}
\section{Methods}
\section{Results}
\section{Discussion}

\bibliographystyle{plainnat}
\bibliography{references}
\end{document}
```

### Useful LaTeX Packages
- `booktabs` — professional tables
- `siunitx` — SI units and number formatting
- `algorithm2e` — pseudocode
- `tikz/pgfplots` — figures and plots
- `cleveref` — smart cross-references
- `subcaption` — subfigures
- `listings/minted` — code listings

### BibTeX Entry Formats
```bibtex
@article{key,
  author  = {Last, First and Last2, First2},
  title   = {Paper Title},
  journal = {Journal Name},
  year    = {2024},
  volume  = {1},
  pages   = {1--10},
  doi     = {10.xxxx/xxxxx}
}
```

## Reviewer Response Template

```
We thank the reviewer for their constructive feedback. Below we address each comment point by point.

---

**Reviewer Comment 1:** [Quote the comment]

**Response:** [Your response]

**Changes made:** [Describe specific changes with page/line numbers]

---
```

Guidelines for reviewer responses:
- Be respectful and grateful, even for harsh reviews
- Address every point, even minor ones
- Clearly distinguish between changes made and rebuttals
- Provide evidence (new analyses, references) for disagreements
- Reference specific manuscript locations for changes

## Writing Quality Checklist

### Clarity
- One idea per paragraph
- Topic sentence first
- Active voice preferred (but passive OK for methods)
- Avoid jargon without definition
- Short sentences for complex ideas

### Precision
- Quantify claims ("increased by 15%" not "significantly increased")
- Distinguish correlation from causation
- Use hedging appropriately ("suggests" vs "proves")
- Report effect sizes, not just p-values

### Flow
- Logical paragraph transitions
- Consistent terminology throughout
- Forward references for later sections
- Signposting ("First... Second... Finally...")

### Common Issues to Fix
- Dangling modifiers
- Pronoun ambiguity ("this" without referent)
- Nominalization overuse (use verbs, not noun forms)
- Redundancy ("past history", "future plans")
- Weak openings ("It is well known that...")

## Citation Styles
- APA 7th: (Author, Year) — social sciences
- IEEE: [1] numbered — engineering/CS
- Vancouver: (1) numbered — biomedical
- Chicago: footnotes — humanities
- Nature: superscript numbered — natural sciences

When unsure, ask the user for their target journal/conference and adapt accordingly.
