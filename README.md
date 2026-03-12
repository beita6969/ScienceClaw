<h1 align="center">🔬 ScienceClaw — AI Research Gateway</h1>

<p align="center">
  <strong>Not just another AI assistant. A research colleague that never sleeps.</strong>
</p>

<p align="center">
  <a href="https://github.com/beita6969/ScienceClaw"><img src="https://img.shields.io/badge/Skills-285-blueviolet?style=for-the-badge" alt="285 Skills"></a>
  <a href="https://github.com/beita6969/ScienceClaw"><img src="https://img.shields.io/badge/Databases-25+-green?style=for-the-badge" alt="25+ Databases"></a>
  <a href="https://github.com/beita6969/ScienceClaw"><img src="https://img.shields.io/badge/Zero-Hallucination-red?style=for-the-badge" alt="Zero Hallucination"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="MIT License"></a>
</p>

---

## What is ScienceClaw?

**ScienceClaw** is an AI-powered scientific research gateway built for researchers, scientists, and academics. While general-purpose AI assistants help you set reminders and write emails, ScienceClaw is purpose-built for one thing: **rigorous scientific work**.

It connects to 25+ academic databases, enforces zero-hallucination protocols, and evolves its own capabilities through research iterations — like having a postdoc who gets smarter with every paper they read.

## ScienceClaw vs General AI Assistants

| Capability | General AI Assistant | ScienceClaw |
|---|---|---|
| Literature search | Basic web search | Multi-database protocol (PubMed, Semantic Scholar, OpenAlex, arXiv, CrossRef, DBLP, SSRN...) |
| Data sources | Wikipedia, web pages | 25+ scientific databases with structured API queries |
| Hallucination control | "Try to be accurate" | **Zero-Hallucination Rule** — every claim must cite a source or be explicitly marked as inference |
| Research depth | Single-pass answers | Multi-phase protocol: search → retrieve → cross-reference → verify → synthesize |
| Statistical analysis | Basic math | Full SciPy/statsmodels/scikit-learn pipelines with rigor checks |
| Skill count | ~50 generic | **285 specialized skills** (121 science-specific) |
| Self-improvement | Static | **VOYAGER-style skill evolution** — learns from every research session |
| Output format | Chat text | Structured reports, LaTeX papers, XLSX data, publication-ready figures |

## Core Philosophy

### 🚫 Zero Hallucination

Every factual claim must be backed by a retrievable source. If ScienceClaw can't find evidence, it says so — clearly and without fabrication. No "I think", no "probably", no made-up citations.

### 🔁 Never Stop Early

Built-in persistence protocol with minimum tool-call thresholds by task complexity:
- Quick lookup: 5+ tool calls
- Survey: 30+ tool calls
- Literature review: 60+ tool calls
- Systematic review: 100+ tool calls

An anti-premature-conclusion checklist runs before every final output.

### 🧬 Self-Evolving Skills

Inspired by VOYAGER (the Minecraft agent that builds its own skill library), ScienceClaw tracks which strategies work and creates new skills from successful patterns. After each research session, the `research-reflection` module evaluates what worked, and `skill-evolution` proposes improvements.

## 285 Skills at a Glance

### 🔬 ScienceClaw Core (10)
`scienceclaw-retrieval` · `scienceclaw-qa` · `scienceclaw-verification` · `scienceclaw-discovery` · `scienceclaw-prediction` · `scienceclaw-reasoning` · `scienceclaw-classification` · `scienceclaw-generation` · `scienceclaw-ie` · `scienceclaw-summarization`

### 📚 Literature Search (35)
`pubmed-search` · `semantic-scholar` · `arxiv-search` · `openalex-search` · `crossref-search` · `dblp-search` · `biorxiv-search` · `ssrn-econpapers` · `literature-search` · `deep-research` · `deep-research-swarm` · `multi-search-engine` · `perplexity-search` · `wikipedia-search` · and more...

### 🗄️ Scientific Databases (25)
`uniprot-protein` · `ncbi-entrez` · `kegg-pathway` · `pdb-structure` · `pubchem-compound` · `chembl-drug` · `open-targets` · `wikidata-knowledge` · `clinicaltrials-database` · `clinvar-database` · `ensembl-database` · `gnomad-database` · `geo-database` · `reactome-database` · `string-database` · and more...

### 🧮 Computation & Analysis (26)
`scipy-analysis` · `statsmodels-stats` · `scikit-learn-ml` · `matplotlib-viz` · `sympy-math` · `rdkit-chemistry` · `biopython-bio` · `astropy-astronomy` · `networkx-social` · `geopandas-spatial` · `spacy-nlp` · `nltk-linguistics` · `transformers-inference` · `plotly` · `seaborn` · `polars` · and more...

### ✍️ Research & Writing (18)
`paper-writing` · `latex-writing` · `latex-posters` · `scientific-slides` · `citation-analysis` · `meta-analysis` · `systematic-review` · `grant-writing` · `protocol-writing` · `peer-review` · `review-writing` · `patent-drafting` · `regulatory-drafting` · and more...

### 🧫 Domain-Specific (50+)
`drug-discovery` · `genomics-analysis` · `protein-structure` · `molecular-dynamics` · `scanpy-singlecell` · `phylogenetics` · `epidemiology` · `neuroscience` · `clinical-trial` · `quantum-computing` · `materials-science` · `environmental-science` · `food-science` · `economics-analysis` · `political-science` · `psychology-research` · and more...

### 🧬 Meta & Evolution (6)
`skill-evolution` · `research-reflection` · `skill-creator` · `find-skills` · `knowledge-synthesis` · `knowledge-discovery`

### 🛠️ Utility & Integration (100+)
Messaging channels, document processing, code execution, visualization, and system tools.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                 ScienceClaw Gateway                   │
│              (Research Control Plane)                  │
├──────────────┬───────────────┬───────────────────────┤
│   SCIENCE.md │  Memory Core  │   Skill Evolution     │
│   (629-line  │  (Temporal    │   (VOYAGER-style      │
│   research   │   decay +     │    pattern library)   │
│   protocol)  │   LanceDB)   │                       │
├──────────────┴───────────────┴───────────────────────┤
│                   285 Skills                          │
│  ┌─────────┐ ┌──────────┐ ┌────────┐ ┌───────────┐  │
│  │ Search  │ │ Database │ │ Compute│ │ Writing   │  │
│  │ (35)    │ │ (25)     │ │ (26)   │ │ (18)      │  │
│  └─────────┘ └──────────┘ └────────┘ └───────────┘  │
│  ┌─────────┐ ┌──────────┐ ┌────────┐ ┌───────────┐  │
│  │ Domain  │ │ Core     │ │ Meta   │ │ Utility   │  │
│  │ (50+)   │ │ (10)     │ │ (6)    │ │ (100+)    │  │
│  └─────────┘ └──────────┘ └────────┘ └───────────┘  │
├──────────────────────────────────────────────────────┤
│              MCP Server Integrations                  │
│  academic-mcp · arxiv-mcp · biomcp · chembl-mcp      │
│  semantic-scholar-mcp · zotero-mcp · mcp-research    │
│  arxiv-latex-mcp                                     │
└──────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Node.js 20+ (or Bun)
- macOS / Linux / Windows (WSL2)

### Install

```bash
# Clone the repository
git clone https://github.com/beita6969/ScienceClaw.git
cd ScienceClaw

# Install dependencies
pnpm install   # or npm install / bun install

# Run the onboarding wizard
npx openclaw onboard
```

### Configure for Science

Edit `~/.openclaw/openclaw.json`:

```jsonc
{
  "plugins": {
    "slots": { "memory": "memory-core" },
    "entries": {
      "memory-core": { "enabled": true },
      "memory-lancedb": { "enabled": true }
    }
  },
  "agent": {
    "timeout": 3600  // 1-hour timeout for deep research
  }
}
```

### Add MCP Servers (Optional, Recommended)

For enhanced academic capabilities, add MCP servers to your config:

```jsonc
{
  "mcpServers": {
    "academic-mcp":          { "command": "uvx", "args": ["academic-mcp"] },
    "arxiv-mcp":             { "command": "uvx", "args": ["arxiv-mcp-server"] },
    "biomcp":                { "command": "uvx", "args": ["biomcp", "run"] },
    "chembl-mcp":            { "command": "uvx", "args": ["chembl-mcp"] },
    "semantic-scholar-mcp":  { "command": "uvx", "args": ["semantic-scholar-mcp"] },
    "zotero-mcp":            { "command": "uvx", "args": ["zotero-mcp"] }
  }
}
```

## SCIENCE.md — The Research Protocol

The heart of ScienceClaw is [`SCIENCE.md`](SCIENCE.md) — a 629-line research protocol that governs agent behavior:

| Section | Purpose |
|---------|---------|
| **Persistence Rules** | Agents never stop early; minimum tool-call thresholds enforced |
| **Zero-Hallucination Rule** | Every claim needs a source; no fabricated citations |
| **Research Depth Enforcement** | Multi-phase search with fallback chains |
| **Academic Literature Search** | Structured multi-channel search protocol |
| **Scientific Database Queries** | API templates for 25+ databases |
| **Statistical Rigor Standards** | Effect sizes, confidence intervals, multiple comparison corrections |
| **Visualization Standards** | Publication-quality figures with proper labeling |
| **LaTeX & Academic Writing** | Template management, citation handling |
| **Systematic Review Protocol** | PRISMA-compliant screening workflows |
| **Memory & Learning** | Cross-session knowledge retention |
| **Compaction Guidance** | Context window management for long research sessions |

## How Skill Self-Evolution Works

```
Research Session
       │
       ▼
┌─────────────┐     ┌──────────────────┐
│  Execute     │────▶│  research-       │
│  Research    │     │  reflection      │
│  Task        │     │  (what worked?)  │
└─────────────┘     └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  skill-evolution  │
                    │  (store pattern,  │
                    │   improve skill)  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  skill-creator    │
                    │  (new SKILL.md   │
                    │   if needed)      │
                    └──────────────────┘
                             │
                             ▼
                  Next session starts
                  with improved skills
```

The agent writes new `SKILL.md` files at runtime — no redeployment needed. Skills accumulate like a lab notebook: each successful strategy becomes a reusable template.

## Example Use Cases

### Biomedical Literature Review
> "Find all studies on CRISPR-Cas9 off-target effects in human cell lines published since 2022"

ScienceClaw will search PubMed, Semantic Scholar, OpenAlex, and bioRxiv, cross-reference results, extract key findings with proper citations, and output a structured Markdown report.

### Drug Target Discovery
> "Identify potential drug targets for idiopathic pulmonary fibrosis using protein interaction networks"

Queries UniProt, Open Targets, STRING, KEGG, and ChEMBL, builds interaction networks with NetworkX, and generates publication-ready figures.

### Statistical Meta-Analysis
> "Perform a meta-analysis of RCTs comparing immunotherapy vs chemotherapy in NSCLC"

Searches clinical trial databases, extracts effect sizes, runs random-effects models with proper heterogeneity assessment (I², Q-test), generates forest plots.

### Materials Science Screening
> "Screen perovskite materials with band gap 1.1-1.7 eV for solar cell applications"

Queries Materials Project and PubChem, filters by computed properties, generates comparison tables with DFT-calculated values.

## Memory System

ScienceClaw remembers across sessions:

- **Temporal Decay**: Recent findings are weighted higher, but important discoveries persist
- **LanceDB Backend**: Vector-based semantic memory for efficient retrieval
- **Cross-Session Learning**: Patterns from past research improve future searches
- **Context Pruning**: Smart compaction keeps the most relevant context within token limits

## Project Structure

```
ScienceClaw/
├── SCIENCE.md          # 629-line research protocol (the brain)
├── AGENTS.md           # Agent behavior guidelines
├── skills/             # 285 skill definitions
│   ├── scienceclaw-*/  # Core research skills
│   ├── *-search/       # Literature search skills
│   ├── *-database/     # Database query skills
│   ├── skill-evolution/ # Self-improvement engine
│   └── ...
├── src/                # Core source code (4,832 files)
│   ├── agents/         # Agent orchestration
│   ├── memory/         # Memory management system
│   ├── skills/         # Skill loading & execution
│   └── ...
├── ui/                 # Web-based control interface
├── extensions/         # Plugin extensions
├── apps/               # Application integrations
├── scripts/            # Build & utility scripts
└── docs/               # Documentation
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Priority areas:
- New scientific database integrations
- Domain-specific skill packs (ecology, astronomy, social sciences...)
- Improved search strategies and fallback chains
- Statistical method implementations
- Visualization templates for specific journal formats

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>ScienceClaw</strong> — Because science deserves better than "let me search that for you."
</p>
