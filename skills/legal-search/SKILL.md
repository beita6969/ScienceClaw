---
name: legal-search
description: "Search US and EU case law, court opinions, dockets, and regulations via CourtListener, Harvard Case Law, and EUR-Lex. Use when: (1) finding US court opinions by keyword or citation, (2) searching federal and state dockets, (3) querying EU legislation and case law, (4) looking up cases by date range or jurisdiction. NOT for: current legislation text (use congress.gov), legal advice (never provide), patent search (use USPTO APIs)."
metadata: { "openclaw": { "emoji": "\u2696\uFE0F", "requires": { "bins": ["curl"] } } }
---

# Legal Case Law and Regulation Search

Search US case law via CourtListener and Harvard Case Law Access Project, and EU
law via EUR-Lex. Covers court opinions, dockets, and legislative documents.

## CourtListener API (US Case Law)

Free, open API for US federal and state court opinions and dockets.
Base URL: `https://www.courtlistener.com/api/rest/v4/`

### Search Opinions

```bash
curl -s "https://www.courtlistener.com/api/rest/v4/search/?q=qualified+immunity&type=o&order_by=score+desc" \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
for r in data.get('results', [])[:10]:
    name = r.get('caseName', 'N/A')
    court = r.get('court', 'N/A')
    date = r.get('dateFiled', 'N/A')
    cite = r.get('citation', [r.get('sibling_ids', 'N/A')])
    print(f'[{date}] {name}')
    print(f'  Court: {court}')
    print()
"
```

### Search Dockets

```bash
curl -s "https://www.courtlistener.com/api/rest/v4/search/?q=antitrust+merger&type=r&order_by=score+desc"
```

### Filter by Date Range

```bash
curl -s "https://www.courtlistener.com/api/rest/v4/search/?q=fourth+amendment+digital+privacy&type=o&filed_after=2020-01-01&filed_before=2024-12-31&order_by=dateFiled+desc"
```

### Filter by Court

```bash
# Supreme Court opinions
curl -s "https://www.courtlistener.com/api/rest/v4/search/?q=free+speech&type=o&court=scotus&order_by=dateFiled+desc"

# Specific circuit court
curl -s "https://www.courtlistener.com/api/rest/v4/search/?q=patent+eligibility&type=o&court=cafc&order_by=dateFiled+desc"
```

### Common Court Codes

`scotus` (SCOTUS), `ca1`-`ca11` (Circuit Courts), `cafc` (Federal Circuit),
`cadc` (DC Circuit), `nyd` (SDNY), `cand` (N.D. Cal).

## Harvard Case Law Access Project

Historical US case law (1658-2020). Free API with registration.
Base URL: `https://api.case.law/v1/`

### Search Cases

```bash
curl -s "https://api.case.law/v1/cases/?search=miranda+rights&decision_date_min=2000-01-01&page_size=10" \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
for c in data.get('results', []):
    name = c.get('name_abbreviation', 'N/A')
    date = c.get('decision_date', 'N/A')
    court = c.get('court', {}).get('name', 'N/A')
    cite = c.get('citations', [{}])[0].get('cite', 'N/A') if c.get('citations') else 'N/A'
    print(f'[{date}] {name}')
    print(f'  Citation: {cite}  Court: {court}\n')
"
```

### Search by Citation

```bash
curl -s "https://api.case.law/v1/cases/?cite=410+U.S.+113"
```

### Filter by Jurisdiction

```bash
curl -s "https://api.case.law/v1/cases/?search=eminent+domain&jurisdiction=cal&decision_date_min=2010-01-01&page_size=10"
```

## EUR-Lex (EU Law)

Access EU legislation, case law, and treaties via the EUR-Lex SPARQL endpoint.

### Search EU Legislation via SPARQL

```bash
curl -s -G "https://eur-lex.europa.eu/EURLexWebService" \
  --data-urlencode "query=SELECT ?doc ?title WHERE {
    ?doc a <http://publications.europa.eu/ontology/cdm#regulation> ;
         <http://publications.europa.eu/ontology/cdm#resource_legal_title> ?title .
    FILTER(CONTAINS(LCASE(?title), 'artificial intelligence'))
  } LIMIT 10" \
  --data-urlencode "format=application/sparql-results+json"
```

### Lookup by CELEX Number

```bash
curl -s "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32016R0679" -o gdpr.html
```

## Best Practices

1. Use CourtListener for recent US case law; it has the most up-to-date coverage.
2. Use Harvard Case Law for historical cases and bulk data access.
3. CourtListener requires no API key for basic search; register for higher rate limits.
4. Always include date filters to narrow results for broad legal topics.
5. For citation lookups, normalize to standard reporter format (e.g., "410 U.S. 113").
6. EUR-Lex SPARQL queries can be complex; start with keyword REST search first.
7. Never provide legal advice; present search results as reference material only.
