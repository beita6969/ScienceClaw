---
name: open-targets
description: "Query the Open Targets Platform GraphQL API for gene-drug-disease associations, evidence scores, and therapeutic target validation. Use when the user needs disease associations for a gene, drug evidence for a target, or target prioritization for a disease. NOT for compound property lookup (use pubchem-compound), NOT for bioactivity measurements (use chembl-drug), NOT for protein 3D structures (use pdb-structure)."
metadata: { "openclaw": { "emoji": "\ud83c\udfaf", "requires": { "bins": ["curl"] } } }
---

# Open Targets Platform Lookup

Query the Open Targets Platform GraphQL API to explore gene-drug-disease associations, evidence from genetic studies, known drugs, pathway data, and overall target validation scores.

## API Base URL

```
https://api.platform.opentargets.org/api/v4/graphql
```

All requests use HTTP POST with a JSON body containing `query` and optionally `variables`.

## API Endpoints

### Search Targets by Gene Symbol

```bash
curl -s -X POST https://api.platform.opentargets.org/api/v4/graphql \
  -H "Content-Type: application/json" \
  -d '{
    "query": "query { search(queryString: \"BRAF\", entityNames: [\"target\"], page: {size: 5, index: 0}) { total hits { id name entity description } } }"
  }' | python3 -m json.tool | head -40
```

### Get Target Details

Retrieve detailed information about a specific target by Ensembl gene ID:

```bash
curl -s -X POST https://api.platform.opentargets.org/api/v4/graphql \
  -H "Content-Type: application/json" \
  -d '{
    "query": "query { target(ensemblId: \"ENSG00000157764\") { id approvedSymbol approvedName biotype functionDescriptions subcellularLocations { location } } }"
  }' | python3 -m json.tool
```

### Get Disease Associations for a Target

Find diseases associated with a gene/target, ranked by overall association score:

```bash
curl -s -X POST https://api.platform.opentargets.org/api/v4/graphql \
  -H "Content-Type: application/json" \
  -d '{
    "query": "query { target(ensemblId: \"ENSG00000157764\") { approvedSymbol associatedDiseases(page: {size: 10, index: 0}) { count rows { disease { id name } score datatypeScores { id score } } } } }"
  }' | python3 -m json.tool | head -60
```

### Get Target Associations for a Disease

Find targets associated with a specific disease by EFO ID:

```bash
curl -s -X POST https://api.platform.opentargets.org/api/v4/graphql \
  -H "Content-Type: application/json" \
  -d '{
    "query": "query { disease(efoId: \"EFO_0000616\") { id name associatedTargets(page: {size: 10, index: 0}) { count rows { target { id approvedSymbol } score datatypeScores { id score } } } } }"
  }' | python3 -m json.tool | head -60
```

### Get Drug Evidence for a Target

Retrieve known drugs and clinical evidence for a target-disease pair:

```bash
curl -s -X POST https://api.platform.opentargets.org/api/v4/graphql \
  -H "Content-Type: application/json" \
  -d '{
    "query": "query { target(ensemblId: \"ENSG00000157764\") { approvedSymbol knownDrugs(page: {size: 10, index: 0}) { count rows { drug { id name mechanismsOfAction { rows { mechanismOfAction } } } phase status diseaseFromSource } } } }"
  }' | python3 -m json.tool | head -80
```

### Search Diseases

```bash
curl -s -X POST https://api.platform.opentargets.org/api/v4/graphql \
  -H "Content-Type: application/json" \
  -d '{
    "query": "query { search(queryString: \"melanoma\", entityNames: [\"disease\"], page: {size: 5, index: 0}) { total hits { id name entity description } } }"
  }' | python3 -m json.tool | head -40
```

## Evidence Types

The `datatypeScores` array contains scores for each evidence category:

- **genetic_association** -- GWAS and gene-burden analyses linking gene variants to disease
- **known_drug** -- approved or clinical-stage drugs with established target-disease evidence
- **affected_pathway** -- pathway-level evidence from Reactome and other pathway databases
- **somatic_mutation** -- cancer somatic mutation data from COSMIC, IntOGen, and others
- **literature** -- text-mined co-occurrences from Europe PMC literature
- **rna_expression** -- differential expression data from Expression Atlas
- **animal_model** -- phenotype evidence from mouse model knockouts (MGI, IMPC)

## Common Queries

```bash
# Resolve gene symbol to Ensembl ID
curl -s -X POST https://api.platform.opentargets.org/api/v4/graphql \
  -H "Content-Type: application/json" \
  -d '{"query": "query { search(queryString: \"TP53\", entityNames: [\"target\"], page: {size: 1, index: 0}) { hits { id name } } }"}' | python3 -m json.tool

# Resolve disease name to EFO ID
curl -s -X POST https://api.platform.opentargets.org/api/v4/graphql \
  -H "Content-Type: application/json" \
  -d '{"query": "query { search(queryString: \"breast cancer\", entityNames: [\"disease\"], page: {size: 1, index: 0}) { hits { id name } } }"}' | python3 -m json.tool

# Get drug details by ChEMBL ID
curl -s -X POST https://api.platform.opentargets.org/api/v4/graphql \
  -H "Content-Type: application/json" \
  -d '{"query": "query { drug(chemblId: \"CHEMBL941\") { id name drugType maximumClinicalTrialPhase mechanismsOfAction { rows { mechanismOfAction targets { id approvedSymbol } } } } }"}' | python3 -m json.tool
```

## Best Practices

1. Always resolve gene symbols to Ensembl IDs and disease names to EFO IDs before querying associations.
2. Use `page: {size: N, index: 0}` to control result counts; default pages can be large.
3. Filter by `datatypeScores` to focus on specific evidence types relevant to the research question.
4. Scores range from 0 to 1; values above 0.5 indicate strong association evidence.
5. Combine with ChEMBL skill for detailed bioactivity data on drugs found through Open Targets.
6. The API has no authentication requirement but rate limit to 10 requests per second.
7. Request only the fields you need to reduce response size and latency.

## Data Integrity Rule

NEVER fabricate database results from training data. Every protein ID, gene name, compound property, pathway ID, structure detail, and metadata MUST come from an actual API response in this conversation. If the API returns no results, errors, or partial data, report exactly what happened. Do not "fill in" missing data from memory or make up identifiers.
