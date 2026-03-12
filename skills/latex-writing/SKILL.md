---
name: latex-writing
description: "Write and format LaTeX documents for academic journals. Use when: user asks to write LaTeX code, format papers for specific journals (Nature/Science/IEEE/ACM), create equations, tables, or BibTeX entries. NOT for: non-LaTeX writing (use paper-writing), data analysis, or literature search."
metadata: { "openclaw": { "emoji": "📐" } }
---

# LaTeX Academic Writing

Generate publication-ready LaTeX documents, equations, tables, and bibliography entries for academic journals across all disciplines.

## When to Use

- "Write this section in LaTeX for Nature"
- "Format this equation in LaTeX"
- "Create a BibTeX entry for this paper"
- "Generate a LaTeX table from this data"
- "Set up a LaTeX document for IEEE conference"
- "Help me with natbib citation commands"

## When NOT to Use

- General paper writing advice (use paper-writing)
- Literature searching (use literature-search)
- Statistical analysis (use statsmodels-stats or scipy-analysis)
- Figure generation (use matplotlib-viz)

## Journal Templates

### Nature / Nature Communications
```latex
\documentclass[12pt]{article}
\usepackage{times}
\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath,graphicx,hyperref}
\usepackage[numbers,sort&compress]{natbib}

% Nature: Title, Authors, Affiliations, Abstract (150 words max),
% Main text (2500 words), Methods, References (max 30), Figures (max 6)
\title{Title (max 90 characters)}
\author{Author One$^{1,*}$, Author Two$^{2}$}
\date{}
\begin{document}
\maketitle
\begin{abstract}
% 150 words max. No references. No abbreviations.
\end{abstract}
\section*{Introduction}
\section*{Results}
\section*{Discussion}
\section*{Methods}
\bibliographystyle{naturemag}
\bibliography{refs}
\end{document}
```

### IEEE Conference (IEEEtran)
```latex
\documentclass[conference]{IEEEtran}
\usepackage{amsmath,graphicx,cite}
\title{Paper Title}
\author{\IEEEauthorblockN{First Author}
\IEEEauthorblockA{Affiliation\\Email}}
\begin{document}
\maketitle
\begin{abstract}
% 150-200 words
\end{abstract}
\begin{IEEEkeywords}
keyword1, keyword2
\end{IEEEkeywords}
\section{Introduction}
\section{Related Work}
\section{Methodology}
\section{Experiments}
\section{Conclusion}
\bibliographystyle{IEEEtran}
\bibliography{refs}
\end{document}
```

### ACM (acmart)
```latex
\documentclass[sigconf,review]{acmart}
\title{Paper Title}
\author{Name}{Affiliation}{email}{orcid}{}
\begin{document}
\begin{abstract}
\end{abstract}
\begin{CCSXML}
% ACM CCS concepts
\end{CCSXML}
\keywords{keyword1, keyword2}
\maketitle
\section{Introduction}
\bibliographystyle{ACM-Reference-Format}
\bibliography{refs}
\end{document}
```

### PNAS
```latex
\documentclass[9pt,twocolumn,twoside]{pnas-new}
% 6 pages max, abstract 250 words, significance 120 words
```

### Social Science (APA 7th - apa7)
```latex
\documentclass[man,12pt]{apa7}
\usepackage[english]{babel}
\usepackage{csquotes}
\usepackage[style=apa,backend=biber]{biblatex}
\addbibresource{refs.bib}
\title{Title}
\authorsnames{Author Name}
\authorsaffiliations{University}
\abstract{Abstract text}
\keywords{keyword1, keyword2}
\begin{document}
\maketitle
\section{Introduction}
\printbibliography
\end{document}
```

## Equation Formatting

### Common Patterns
```latex
% Inline: $E = mc^2$
% Display:
\begin{equation}
  \hat{\beta} = (X^{\top}X)^{-1}X^{\top}y
  \label{eq:ols}
\end{equation}

% Aligned multi-line:
\begin{align}
  \mathcal{L}(\theta) &= \sum_{i=1}^{n} \log p(x_i | \theta) \\
  \nabla_\theta \mathcal{L} &= \sum_{i=1}^{n} \nabla_\theta \log p(x_i | \theta)
\end{align}

% Cases:
\begin{equation}
  f(x) = \begin{cases}
    x^2 & \text{if } x \geq 0 \\
    -x  & \text{if } x < 0
  \end{cases}
\end{equation}
```

### Discipline-Specific Notation
- **Physics**: `\hbar`, `\nabla`, `\partial`, `\langle \psi | \hat{H} | \psi \rangle`
- **Chemistry**: `\ce{H2O}` (mhchem), `\ch{->}` (chemformula)
- **Biology**: Gene names in italics `\textit{TP53}`, protein names upright
- **Economics**: Expectations `\mathbb{E}[X]`, summations with limits
- **Statistics**: `\sim`, `\mid`, `\perp`, hat/tilde for estimators

## Table Formatting

### Publication-Quality Table
```latex
\usepackage{booktabs}
\begin{table}[htbp]
  \centering
  \caption{Results of regression analysis}
  \label{tab:results}
  \begin{tabular}{lcccc}
    \toprule
    Variable & Coeff. & SE & $t$ & $p$ \\
    \midrule
    Intercept & 2.34 & 0.12 & 19.5 & $<$0.001 \\
    Treatment & 0.87 & 0.15 & 5.8  & $<$0.001 \\
    Age       & 0.02 & 0.01 & 2.1  & 0.036 \\
    \bottomrule
  \end{tabular}
  \begin{tablenotes}
    \small
    \item Note: $N = 500$. Standard errors in parentheses.
  \end{tablenotes}
\end{table}
```

## BibTeX Management

### Entry Types
```bibtex
@article{smith2024method,
  author  = {Smith, John A. and Jones, Mary B.},
  title   = {A New Method for Protein Folding},
  journal = {Nature},
  year    = {2024},
  volume  = {625},
  pages   = {123--130},
  doi     = {10.1038/s41586-024-00001-1}
}

@inproceedings{wang2024transformer,
  author    = {Wang, Lei},
  title     = {Efficient Transformers},
  booktitle = {Proceedings of ICML 2024},
  year      = {2024},
  pages     = {1--10}
}

@book{wooldridge2020econometrics,
  author    = {Wooldridge, Jeffrey M.},
  title     = {Introductory Econometrics},
  publisher = {Cengage},
  year      = {2020},
  edition   = {7th}
}
```

### Citation Packages
| Package | Style | Command | Output |
|---------|-------|---------|--------|
| natbib | Author-year | `\citet{key}` | Smith et al. (2024) |
| natbib | Author-year | `\citep{key}` | (Smith et al., 2024) |
| natbib | Numbered | `\cite{key}` | [1] |
| biblatex-apa | APA 7 | `\textcite{key}` | Smith et al. (2024) |
| biblatex-apa | APA 7 | `\parencite{key}` | (Smith et al., 2024) |

## Pre-Submission Checklist

1. **Formatting**: Page limits, font size, margins per journal guidelines
2. **Abstract**: Word count within limit, no undefined abbreviations
3. **Figures**: Resolution >= 300 DPI, vector format preferred (PDF/EPS)
4. **Tables**: Use `booktabs` (no vertical lines), caption above table
5. **References**: All cited works in bibliography, no orphan entries
6. **Cross-references**: All `\ref{}` and `\eqref{}` resolve correctly
7. **Supplementary**: Separate file if required, referenced from main text
8. **Compile**: Clean compilation with no warnings on `\ref` or `\cite`

## Zero-Hallucination Rule

- NEVER generate fake DOIs, journal names, or citation details
- If converting a paper reference to BibTeX, only include fields verified through tool results
- When unsure of a journal's LaTeX class, say so and suggest checking the journal website
