---
name: fact-verification
description: Verify scientific claims, political statements, and environmental assertions against evidence
---

# Fact Verification

## Purpose
Systematically verify factual claims using evidence retrieval, source evaluation, and logical reasoning.

## Key Datasets
- **PolitiFact** (Jinyan1/PolitiFact): Political statements rated on 6-level truth scale (True, Mostly True, Half True, Mostly False, False, Pants on Fire)
- **Climate-FEVER** (tdiggelm/climate_fever): Climate claims labeled SUPPORTS/REFUTES/NOT_ENOUGH_INFO with evidence sentences

## Protocol
1. **Claim decomposition** — Break complex claims into atomic verifiable statements
2. **Evidence retrieval** — Search authoritative sources for each sub-claim
3. **Source evaluation** — Assess source credibility and potential bias
4. **Evidence-claim alignment** — Determine if evidence supports, refutes, or is insufficient
5. **Verdict synthesis** — Aggregate sub-verdicts into overall assessment

## Verification Categories
- **Scientific claims**: Published findings, statistical assertions, causal claims
- **Political statements**: Policy claims, historical assertions, statistical citations
- **Environmental claims**: Climate data, pollution metrics, biodiversity assertions
- **Health claims**: Treatment efficacy, risk factors, epidemiological data

## Verdict Scale
- **VERIFIED**: Multiple independent high-quality sources confirm
- **LIKELY TRUE**: Evidence supports but limited independent confirmation
- **MIXED**: Partially true with important caveats or context
- **LIKELY FALSE**: Evidence contradicts but some ambiguity remains
- **FALSE**: Clear evidence contradicts the claim
- **UNVERIFIABLE**: Insufficient evidence to determine

## Rules
- Always cite specific evidence for each verdict
- Distinguish between factual errors and misleading framing
- Check for cherry-picked statistics or out-of-context quotes
- Consider temporal context (claim may have been true when made)
