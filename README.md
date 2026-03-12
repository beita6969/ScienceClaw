<h1 align="center">🔬 ScienceClaw</h1>

<p align="center">
  <strong>A self-evolving AI research colleague for scientists.</strong>
</p>

<p align="center">
  <a href="https://github.com/beita6969/ScienceClaw"><img src="https://img.shields.io/badge/Skills-285-blueviolet?style=for-the-badge" alt="285 Skills"></a>
  <a href="https://github.com/beita6969/ScienceClaw"><img src="https://img.shields.io/badge/Databases-25+-green?style=for-the-badge" alt="25+ Databases"></a>
  <a href="https://github.com/beita6969/ScienceClaw"><img src="https://img.shields.io/badge/Zero-Hallucination-red?style=for-the-badge" alt="Zero Hallucination"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="MIT License"></a>
</p>

---

## Why ScienceClaw?

General-purpose AI assistants are built for everyone. ScienceClaw is built for **researchers**.

The core idea is simple: an AI that does real scientific work — searching literature, querying databases, running analyses — and **gets better at it the more you use it**. It remembers your research context across sessions, adapts its skills to your field, and never fabricates a citation.

ScienceClaw is built on the [OpenClaw](https://github.com/openclaw/openclaw) engine, but redesigned from the ground up for academic research. Here's what makes it different:

---

## 🧬 Core 1: Self-Evolving Skills

**This is ScienceClaw's most important feature.**

Most AI tools ship with a fixed set of capabilities. ScienceClaw's skills **evolve with you**. Every time you complete a research task, the system learns:

```
  You do research
       │
       ▼
┌─────────────────┐
│ research-        │  ← "What strategies worked? What failed?
│ reflection       │     Which databases had the best results?"
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ skill-evolution  │  ← Stores successful patterns, proposes
│                  │     improvements to existing skills
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ skill-creator    │  ← If no existing skill fits, writes a
│                  │     brand new SKILL.md at runtime
└─────────────────┘
         │
         ▼
  Next session is smarter
```

**What this means in practice:**

- Week 1: You study immunology. ScienceClaw learns that PubMed + Semantic Scholar works best for your queries, that you prefer forest plots over tables, and that you always need PMID + DOI in citations.
- Week 4: The system has created specialized skills for your subfield — optimized search templates, preferred statistical methods, database priority chains tuned to immunology literature.
- Month 3: ScienceClaw handles your domain like a trained research assistant. It knows which databases to hit first, which journals matter, and how you like your output formatted.

This is inspired by [VOYAGER](https://voyager.minedojo.org/) — the Minecraft agent that builds its own skill library. Except here, the skill library serves science.

> **Compared to standard OpenClaw:** OpenClaw ships with ~54 general-purpose skills that don't change. ScienceClaw starts with 285 skills and grows from there — the agent writes new `SKILL.md` files at runtime without any redeployment.

---

## 🧠 Core 2: Research Memory That Persists

Standard AI assistants forget everything when the conversation ends. ScienceClaw doesn't.

**Memory architecture:**

| Layer | What it remembers | How long |
|-------|-------------------|----------|
| **Session Memory** | Current research context, intermediate findings, tool outputs | Current session |
| **Temporal Decay Memory** | Key findings, successful strategies, verified facts | Weeks to months (decays by relevance) |
| **LanceDB Vector Memory** | Semantic embeddings of research patterns and discoveries | Permanent (vector-indexed) |
| **Skill Memory** | Which tools worked for which types of queries | Permanent (encoded in evolved skills) |

**What this enables:**

- "Continue the literature review we started last Tuesday" — it remembers where you left off
- "Use the same search strategy that worked for the BRCA2 project" — it retrieves past patterns
- Cross-session knowledge accumulation — findings from project A can inform project B
- Smart context pruning — when the context window fills up, it preserves statistical results, effect sizes, and key citations while compacting intermediate steps

> **Compared to standard OpenClaw:** OpenClaw has a basic memory plugin. ScienceClaw adds temporal decay weighting, LanceDB vector storage, and cross-session research pattern retrieval — specifically designed for long-running academic work.

---

## ⏱️ Core 3: Built for Long-Duration Research

A real literature review takes hours, not seconds. Most AI tools time out after a few minutes. ScienceClaw is engineered for extended research sessions:

| Capability | Standard OpenClaw | ScienceClaw |
|---|---|---|
| Agent timeout | 600s (10 min) | **3600s (1 hour)** |
| Session persistence | Ends with conversation | Heartbeat keeps sessions alive across interruptions |
| Research depth | Single-pass response | **Multi-phase protocol with mandatory depth thresholds** |
| Minimum effort | No guarantee | Quick=5, Survey=30, Review=60, Systematic=100+ tool calls |
| Early stopping | Common | **Anti-premature-conclusion checklist** blocks shallow answers |
| Context management | Basic truncation | **Smart compaction** preserves key findings when context fills up |

**The persistence protocol enforces real research depth:**

Before ScienceClaw concludes any task, it must verify:
- ✅ Searched at least 3 different databases/sources
- ✅ Retrieved full metadata (not just titles)
- ✅ Cross-referenced findings across sources
- ✅ Checked for contradictory evidence
- ✅ Verified key statistics against primary sources
- ✅ Organized results into a structured output file
- ✅ Met the minimum tool-call threshold for the task type

If any box is unchecked, it **keeps working** instead of giving you a half-baked answer.

> **Compared to standard OpenClaw:** OpenClaw's default 10-minute timeout is fine for sending messages and setting reminders. ScienceClaw's 1-hour sessions with heartbeat monitoring and mandatory depth enforcement are built for real academic research.

---

## 🚫 Core 4: Zero Hallucination

This is the highest-priority rule in the entire system. It's non-negotiable.

**The problem:** General AI assistants routinely fabricate citations — inventing DOIs, making up author names, citing papers that don't exist. In scientific work, this is catastrophic.

**ScienceClaw's approach:**

```
EVERY citation must come from a tool result in the CURRENT conversation.

If a database didn't return it → you can't cite it.
If you're not sure → say "not verified" explicitly.
If you can't find evidence → say so. Don't guess.

No "I think." No "probably." No hallucinated PMIDs.
```

This is enforced at the protocol level in [`SCIENCE.md`](SCIENCE.md) — the 629-line research protocol that governs all agent behavior. It's not a suggestion. It's a hard rule that applies before any other instruction.

> **Compared to standard OpenClaw:** OpenClaw has no special hallucination controls. ScienceClaw's SCIENCE.md protocol treats every factual claim as requiring evidence — the same standard you'd apply to a manuscript under peer review.

---

## 🌍 Core 5: All of Science, Not Just Biomedicine

ScienceClaw covers **natural sciences AND social sciences** across dozens of disciplines:

### Natural Sciences
| Domain | Key Skills & Databases |
|--------|----------------------|
| **Biomedicine** | PubMed, UniProt, KEGG, PDB, ClinicalTrials, gnomAD, scanpy, biopython |
| **Chemistry** | PubChem, ChEMBL, RDKit, drug-discovery, molecular-dynamics |
| **Genomics** | NCBI Entrez, Ensembl, ClinVar, GEO, phylogenetics |
| **Materials Science** | Materials Project, pymatgen, materials-screening |
| **Physics** | astropy, quantum-computing, physics-solver, simulation |
| **Environmental Science** | Copernicus climate data, geospatial analysis, GIS tools |
| **Food Science** | Specialized analysis pipelines |

### Social Sciences
| Domain | Key Skills & Databases |
|--------|----------------------|
| **Economics** | World Bank, SSRN, census data, econometrics |
| **Political Science** | Policy analysis, legislative data |
| **Psychology** | Experimental design, statistical testing, meta-analysis |
| **Linguistics** | spaCy, NLTK, NLP analysis |
| **Education** | Research methodology, assessment analysis |
| **Sociology** | Network analysis, survey methods |

### Cross-Disciplinary Tools
| Category | Capabilities |
|----------|-------------|
| **Statistics** | SciPy, statsmodels, scikit-learn, effect sizes, confidence intervals, multiple comparison corrections |
| **Visualization** | matplotlib, plotly, seaborn, publication-quality figures |
| **Writing** | LaTeX papers, systematic reviews (PRISMA), grant proposals, patent drafting |
| **Mathematics** | SymPy symbolic computation, numerical methods, optimization |

**285 skills total** — and growing, because the self-evolution system creates new ones as you work.

> **Compared to standard OpenClaw:** OpenClaw has no scientific database integrations. No PubMed, no UniProt, no arXiv, no World Bank. ScienceClaw connects to 25+ academic databases with structured API query skills across all major scientific disciplines.

---

## How It All Fits Together

```
┌─────────────────────────────────────────────────────────────┐
│                    ScienceClaw Gateway                       │
├──────────────┬────────────────┬──────────────────────────────┤
│  SCIENCE.md  │  Memory System │  Skill Self-Evolution        │
│  Zero-halluc │  LanceDB +     │  research-reflection →       │
│  Persistence │  Temporal decay │  skill-evolution →           │
│  Depth rules │  Cross-session  │  skill-creator               │
├──────────────┴────────────────┴──────────────────────────────┤
│                       285 Skills                             │
│                                                              │
│  Literature Search (35)  │  Scientific Databases (25)        │
│  Computation (26)        │  Research & Writing (18)          │
│  Domain-Specific (50+)   │  ScienceClaw Core (10)           │
│  Meta & Evolution (6)    │  Utility & Integration (100+)    │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  MCP Servers: academic-mcp · arxiv-mcp · biomcp · chembl    │
│  semantic-scholar-mcp · zotero-mcp · arxiv-latex-mcp        │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/beita6969/ScienceClaw.git
cd ScienceClaw

# Install
pnpm install   # or npm / bun

# Onboard
npx openclaw onboard
```

### Enable Research Features

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
    "timeout": 3600
  }
}
```

### Add Academic MCP Servers (Recommended)

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

---

## Project Structure

```
ScienceClaw/
├── SCIENCE.md              # Research protocol (the brain)
├── skills/                 # 285 skill definitions (and growing)
│   ├── skill-evolution/    # Self-improvement engine
│   ├── research-reflection/# Post-task learning
│   ├── skill-creator/      # Runtime skill generation
│   └── ...
├── src/
│   ├── memory/             # Memory management (temporal decay, LanceDB)
│   ├── agents/             # Agent orchestration & persistence
│   └── skills/             # Skill loading & execution
├── ui/                     # Web control interface
├── extensions/             # Plugin system
└── docs/                   # Documentation
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Priority areas:

- New database integrations (especially social science data sources)
- Domain-specific skill packs
- Statistical method implementations
- Improved search strategies and fallback chains

## License

MIT — see [LICENSE](LICENSE).

---

<p align="center">
  <em>"The best research assistant is one that gets better every time you use it."</em>
</p>
