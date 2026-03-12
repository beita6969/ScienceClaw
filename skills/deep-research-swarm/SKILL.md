---
name: deep-research-swarm
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
name: deep-research-swarm
description: Multi-agent research literature analysis
keywords:
  - research
  - literature
  - swarm
  - multi-agent
  - hypothesis
measurable_outcome: Generates comprehensive literature review with >50 citations in <5 minutes.
license: MIT
metadata:
  author: Biomedical OS Team
  version: "1.0.0"
compatibility:
  - system: Python 3.10+
allowed-tools:
  - run_shell_command
  - read_file
  - google_web_search
---

# DeepResearch Swarm

A coordinated swarm of agents designed to perform deep, parallelized research into biomedical literature, aggregating findings into comprehensive reports.

## When to Use This Skill

*   When you need an exhaustive review of a specific medical topic.
*   When connecting disparate pieces of evidence across thousands of papers.
*   When generating hypotheses based on recent literature.

## Core Capabilities

1.  **Parallel Search**: Querying multiple databases simultaneously.
2.  **Evidence Synthesis**: Combining facts into a coherent narrative.
3.  **Citation Verification**: Ensuring all claims are backed by sources.

## Example Usage

**User**: "Research the latest advancements in mRNA cancer vaccines."

**Agent Action**:
```bash
python3 src/research/agents/agent_coordinator.py --topic "mRNA cancer vaccines" --depth "deep"
```


<!-- AUTHOR_SIGNATURE: 9a7f3c2e-MD-BABU-MIA-2026-MSSM-SECURE -->