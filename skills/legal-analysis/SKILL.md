---
name: legal-analysis
description: Analyze legal contracts, extract clauses, and perform legal research with structured frameworks
---

# Legal Analysis

## Purpose
Analyze legal documents, extract key clauses, and perform structured legal research.

## Key Datasets
- **CUAD** (atticus-project/cuad): Contract Understanding Atticus Dataset — 510 contracts, 41 clause types, 13,000+ annotations (CC-BY)
- **CaseHOLD**: Legal case holding identification
- **LegalBench**: Legal reasoning benchmark across 162 tasks

## Clause Types (from CUAD)
Document Name, Parties, Agreement Date, Effective Date, Expiration Date, Renewal Term, Notice Period, Governing Law, Arbitration, Anti-Assignment, Non-Compete, Non-Solicitation, Termination for Convenience, IP Ownership, License Grant, Non-Disparagement, Revenue/Profit Sharing, Change of Control, Audit Rights, Uncapped Liability, Cap on Liability, Liquidated Damages, Most Favored Nation, Exclusivity, Minimum Commitment, Volume Restriction, Insurance, Covenant Not to Sue, Third Party Beneficiary

## Protocol
1. **Document intake** — Parse contract structure, identify parties and key dates
2. **Clause extraction** — Identify and classify clauses by type
3. **Risk analysis** — Flag unusual, missing, or potentially problematic provisions
4. **Comparison** — Compare against standard terms for the contract type
5. **Summary** — Structured output with key terms, obligations, and risks

## Rules
- This is NOT legal advice — always recommend professional legal review
- Identify jurisdiction-specific considerations
- Flag ambiguous language that could be interpreted multiple ways
- Note missing standard clauses that are typically expected
