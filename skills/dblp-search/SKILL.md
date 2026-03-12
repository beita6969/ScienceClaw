---
name: dblp-search
description: "Computer science bibliography via DBLP API. Use when: user asks about CS publications, author publication lists, or venue (conference/journal) metadata. NOT for: non-CS publications or citation counts."
metadata: { "openclaw": { "emoji": "💻", "requires": { "bins": ["curl"] } } }
---

# DBLP Search Skill

Query the DBLP computer science bibliography for publications, authors, and venues.

## When to Use

- "Find papers by this CS researcher"
- "List publications from ICML 2024"
- "Search for papers on transformer architectures"
- "Get author's full publication list"

## When NOT to Use

- Non-CS publications (use semantic-scholar or openalex)
- Citation counts or impact metrics (use semantic-scholar)
- Full-text paper access (use arxiv-search)

## Setup

No API key required. Public REST API.

## API Base

```
https://dblp.org/search/
```

## Common Commands

### Search Publications

```bash
# Search by keyword
curl -s "https://dblp.org/search/publ/api?q=attention+is+all+you+need&format=json&h=5" | python3 -c "
import json,sys
data=json.load(sys.stdin)
for hit in data['result']['hits'].get('hit',[]):
    info=hit['info']
    print(f\"Title: {info.get('title','')}\")
    print(f\"Authors: {', '.join(a.get('text','') for a in info.get('authors',{}).get('author',[]))}\")
    print(f\"Venue: {info.get('venue','')}, Year: {info.get('year','')}\")
    print()
"

# Filter by year
curl -s "https://dblp.org/search/publ/api?q=large+language+model+year%3A2024&format=json&h=10"
```

### Search Authors

```bash
# Find author
curl -s "https://dblp.org/search/author/api?q=Yoshua+Bengio&format=json&h=5"

# Get author's publication list
curl -s "https://dblp.org/pid/b/YoshuaBengio.xml?format=json" | python3 -c "
import json,sys
data=json.load(sys.stdin)
pubs=data.get('dblpperson',{}).get('r',[])[:10]
for p in pubs:
    entry=list(p.values())[0] if isinstance(p,dict) else {}
    print(f\"{entry.get('year','')} - {entry.get('title','')}\")
"
```

### Search Venues

```bash
# Find conference/journal
curl -s "https://dblp.org/search/venue/api?q=NeurIPS&format=json&h=5"
```

### Pagination

```bash
# h = results per page, f = first result index
curl -s "https://dblp.org/search/publ/api?q=graph+neural+network&format=json&h=20&f=0"
curl -s "https://dblp.org/search/publ/api?q=graph+neural+network&format=json&h=20&f=20"
```

## Notes

- Free, no API key needed
- Covers CS conferences, journals, and workshops comprehensively
- Author disambiguation built-in (PID-based)
- Supports XML and JSON output formats
- Rate limit: be respectful, no official limit
- Does NOT include citation counts (use Semantic Scholar for that)
