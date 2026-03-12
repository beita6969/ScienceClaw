---
name: world-bank-data
description: "World Bank Open Data API for development indicators. Use when: user asks about GDP, population, poverty, health, or education statistics by country. NOT for: real-time financial data or stock prices."
metadata: { "openclaw": { "emoji": "🌍", "requires": { "bins": ["curl"] } } }
---

# World Bank Data

Query World Bank Open Data API for development indicators by country and year.

## API Base

```
https://api.worldbank.org/v2/
```

All requests should include `format=json`. Responses return `[metadata, data]`.

## Country Indicators

```bash
# GDP (current US$) for a country
curl -s "https://api.worldbank.org/v2/country/US/indicator/NY.GDP.MKTP.CD?format=json&date=2015:2023&per_page=50"

# Multiple countries (semicolon-separated)
curl -s "https://api.worldbank.org/v2/country/US;CN;IN/indicator/NY.GDP.MKTP.CD?format=json&date=2020:2023"
```

## Common Indicator Codes

| Code | Description |
|------|-------------|
| NY.GDP.MKTP.CD | GDP (current US$) |
| NY.GDP.PCAP.CD | GDP per capita (current US$) |
| SP.POP.TOTL | Total population |
| SP.DYN.LE00.IN | Life expectancy at birth |
| SE.ADT.LITR.ZS | Adult literacy rate (%) |
| SL.UEM.TOTL.ZS | Unemployment (% of labor force) |
| SI.POV.DDAY | Poverty headcount at $2.15/day (%) |
| SH.XPD.CHEX.PC.CD | Health expenditure per capita |
| EN.ATM.CO2E.PC | CO2 emissions (metric tons per capita) |

## Country Metadata

```bash
# All countries
curl -s "https://api.worldbank.org/v2/country?format=json&per_page=300"

# Countries by income level (HIC, UMC, LMC, LIC)
curl -s "https://api.worldbank.org/v2/country?format=json&incomeLevel=LIC&per_page=100"
```

## Topic and Source Listings

```bash
# List all topics (education, health, etc.)
curl -s "https://api.worldbank.org/v2/topic?format=json"

# Indicators under a topic (topic 4 = Education)
curl -s "https://api.worldbank.org/v2/topic/4/indicator?format=json&per_page=100"
```

## Pagination and Date Ranges

```bash
# Pagination: page= and per_page= (max 32500)
curl -s "https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?format=json&date=2023&per_page=300&page=1"

# Date formats: date=2023 (single), date=2015:2023 (range), date=2015;2018;2023 (list)
# Most recent value only
curl -s "https://api.worldbank.org/v2/country/US/indicator/NY.GDP.MKTP.CD?format=json&mrnev=1"
```

## Response Parsing

```bash
curl -s "https://api.worldbank.org/v2/country/US;CN;IN/indicator/NY.GDP.PCAP.CD?format=json&date=2020:2023&per_page=100" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for entry in sorted(data[1], key=lambda x: (x['country']['value'], x['date'])):
    if entry['value'] is not None:
        print(f\"{entry['country']['value']} ({entry['date']}): \${entry['value']:,.0f}\")
"
```

## Best Practices

1. Always append `format=json` to get JSON instead of XML.
2. Use `mrnev=1` for the most recent non-null value.
3. Check `[0]['total']` in the response metadata for total result count.
4. Country codes follow ISO 3166-1 alpha-2 (US, CN, IN, BR, etc.).
5. Use `per_page=300` for country-level queries to get all in one request.
6. Null values indicate missing data for that year; filter them in output.
