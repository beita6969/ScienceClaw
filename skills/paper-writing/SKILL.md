---
name: paper-writing
description: "Write scientific papers following IMRaD structure with proper citations. Use when: user asks to draft or improve a scientific paper, write specific sections, or format citations. NOT for: non-academic writing or presentation slides."
metadata: { "openclaw": { "emoji": "📝" } }
---

# Paper Writing Skill

Draft, structure, and refine scientific papers across all disciplines.

## When to Use

- "Write the methods section for..."
- "Draft an abstract for these results"
- "Help me structure this paper"
- "Format these citations in APA/IEEE"
- "Improve the introduction of my paper"
- Generating LaTeX-formatted content

## When NOT to Use

- Literature searching (use literature-search)
- Data analysis (use code-execution + scipy-analysis)
- Non-academic writing
- Grant proposals (different structure)

## Paper Structure (IMRaD)

### Title
- Concise, informative, contains key terms
- Avoid abbreviations and jargon
- 10-15 words ideal

### Abstract (150-300 words)
```
Background: [1-2 sentences: context and gap]
Objective: [1 sentence: what this study does]
Methods: [2-3 sentences: approach and data]
Results: [2-3 sentences: key findings with numbers]
Conclusions: [1-2 sentences: implications]
```

### Introduction
1. Broad context (funnel from general to specific)
2. Literature review (key findings, organized thematically)
3. Research gap (what's missing or contradictory)
4. Study objective and hypothesis
5. Brief approach overview
6. Significance statement

### Methods
- Study design and setting
- Participants/samples (inclusion/exclusion criteria)
- Materials and instruments
- Procedure (reproducible detail)
- Data analysis plan
- Ethical approvals

### Results
- Present in order of research questions
- Text + tables + figures (no redundancy)
- Statistical results: test statistic, df, p-value, effect size
- Report exact p-values (not just p < 0.05)

### Discussion
1. Summary of key findings
2. Comparison with prior literature
3. Mechanistic explanation
4. Strengths of the study
5. Limitations and their impact
6. Future directions
7. Conclusions and implications

## Citation Formats

### APA 7th Edition
- In-text: (Author, Year) or Author (Year)
- Two authors: (Smith & Jones, 2024)
- 3+ authors: (Smith et al., 2024)

### IEEE
- Numbered: [1], [2], [3]
- Reference list: [1] A. Author, "Title," Journal, vol. X, no. Y, pp. Z-Z, Year.

### Chicago (Author-Date)
- In-text: (Author Year, page)
- Bibliography: Author. Year. "Title." Journal Vol(Issue): Pages.

### Vancouver/NLM
- Numbered: (1), (2), (3)
- Reference: Author(s). Title. Journal. Year;Vol(Issue):Pages.

## LaTeX Templates

### Article
```latex
\documentclass[12pt]{article}
\usepackage{amsmath,graphicx,natbib}
\title{Title}
\author{Author}
\begin{document}
\maketitle
\begin{abstract}
...
\end{abstract}
\section{Introduction}
...
\bibliographystyle{plainnat}
\bibliography{refs}
\end{document}
```

## Integration with MCP Tools

**Citation Workflow**
1. Search papers via `academic-mcp` (19 databases) or `semantic-scholar-mcp`
2. Export BibTeX/APA/MLA citations directly from Semantic Scholar MCP
3. Manage references with `zotero-mcp` (if Zotero is available)
4. Use `arxiv-latex-mcp` to read LaTeX source for accurate equation citing

**Related Skills**
- For detailed LaTeX formatting → use `latex-writing` skill
- For citation network analysis → use `citation-analysis` skill
- For systematic reviews → use `systematic-review` skill
- For meta-analysis → use `meta-analysis` skill

## Guidelines

1. Use active voice where appropriate
2. Be precise — avoid vague qualifiers ("somewhat", "relatively")
3. Define abbreviations on first use
4. Every claim needs a citation or data reference
5. Tables and figures must be self-contained with captions
6. Follow target journal's style guide
7. Report effect sizes, not just p-values
8. Include CRediT author contributions, data availability, and COI statements
