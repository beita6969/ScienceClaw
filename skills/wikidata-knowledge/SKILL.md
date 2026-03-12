---
name: wikidata-knowledge
description: "Query Wikidata for structured knowledge using SPARQL and entity search. Use when: (1) finding structured facts about entities (people, places, organizations), (2) querying relationships between entities, (3) cross-referencing external identifiers (Wikipedia, VIAF, GND, ORCID), (4) building knowledge graphs from linked data. NOT for: full-text article content (use Wikipedia API), scientific literature (use semantic-scholar), geospatial data (use OpenStreetMap)."
metadata: { "openclaw": { "emoji": "\U0001F310", "requires": { "bins": ["curl"] } } }
---

# Wikidata SPARQL and Entity Search

Query Wikidata's knowledge graph of 100M+ items using SPARQL and the entity
search API. Covers people, places, organizations, events, scientific concepts,
and their relationships.

## Entity Search API

Find Wikidata entity IDs by label:

```bash
curl -s "https://www.wikidata.org/w/api.php?action=wbsearchentities&search=Marie+Curie&language=en&format=json&limit=5" \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
for r in data.get('search', []):
    qid = r.get('id', 'N/A')
    label = r.get('label', 'N/A')
    desc = r.get('description', '')
    print(f'{qid:12s} {label} - {desc}')
"
```

## SPARQL Endpoint

```
https://query.wikidata.org/sparql?query={SPARQL}&format=json
```

## Basic SPARQL Query via curl

```bash
curl -s -G "https://query.wikidata.org/sparql" \
  --data-urlencode "format=json" \
  --data-urlencode "query=
SELECT ?item ?itemLabel WHERE {
  ?item wdt:P31 wd:Q515 .
  ?item wdt:P17 wd:Q183 .
  ?item wdt:P1082 ?pop .
  FILTER(?pop > 500000)
  SERVICE wikibase:label { bd:serviceParam wikibase:language \"en\" . }
} ORDER BY DESC(?pop) LIMIT 10
" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for r in data['results']['bindings']:
    qid = r['item']['value'].split('/')[-1]
    label = r['itemLabel']['value']
    print(f'{qid:12s} {label}')
"
```

## Common Property Codes

`P31` (instance of), `P279` (subclass of), `P17` (country), `P569` (date of birth),
`P570` (date of death), `P106` (occupation), `P1082` (population), `P625` (coordinates),
`P214` (VIAF ID), `P496` (ORCID iD), `P356` (DOI).

## Common Entity Codes

`Q5` (human), `Q515` (city), `Q6256` (country), `Q3918` (university),
`Q7889` (computer program), `Q11173` (chemical compound), `Q16521` (taxon).

## SPARQL: Find Nobel Prize Winners in Physics

```bash
curl -s -G "https://query.wikidata.org/sparql" \
  --data-urlencode "format=json" \
  --data-urlencode "query=
SELECT ?person ?personLabel ?year WHERE {
  ?person wdt:P166 wd:Q38104 .
  ?person p:P166 ?statement .
  ?statement ps:P166 wd:Q38104 ;
             pq:P585 ?date .
  BIND(YEAR(?date) AS ?year)
  FILTER(?year >= 2020)
  SERVICE wikibase:label { bd:serviceParam wikibase:language \"en\" . }
} ORDER BY DESC(?year)
"
```

## SPARQL: Cross-Reference External IDs

```bash
# Find ORCID and VIAF for a researcher
curl -s -G "https://query.wikidata.org/sparql" \
  --data-urlencode "format=json" \
  --data-urlencode "query=
SELECT ?person ?personLabel ?orcid ?viaf WHERE {
  ?person wdt:P31 wd:Q5 ;
          wdt:P496 ?orcid ;
          wdt:P214 ?viaf ;
          rdfs:label ?name .
  FILTER(CONTAINS(LCASE(?name), 'hinton'))
  FILTER(LANG(?name) = 'en')
  SERVICE wikibase:label { bd:serviceParam wikibase:language \"en\" . }
} LIMIT 5
"
```

## Rate Limits

SPARQL: 1 concurrent query, 60-second timeout. Entity search: standard MediaWiki
rate limits. User-Agent header recommended for all requests.

## Best Practices

1. Always use `SERVICE wikibase:label` to get human-readable labels.
2. Start with entity search to find Q-IDs before writing SPARQL queries.
3. Use `LIMIT` on all queries to avoid timeouts on large result sets.
4. Prefer `wdt:` (direct truthy) over `p:`/`ps:` unless you need qualifiers.
5. For complex queries, test at https://query.wikidata.org/ first.
6. URL-encode SPARQL queries when using curl with `--data-urlencode`.
7. Add `OPTIONAL {}` blocks for properties that may not exist on all entities.

## Data Integrity Rule

NEVER fabricate database results from training data. Every protein ID, gene name, compound property, pathway ID, structure detail, and metadata MUST come from an actual API response in this conversation. If the API returns no results, errors, or partial data, report exactly what happened. Do not "fill in" missing data from memory or make up identifiers.
