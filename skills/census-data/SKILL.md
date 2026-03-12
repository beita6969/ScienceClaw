---
name: census-data
description: "US Census Bureau data via API. Use when: user asks about US demographics, population, housing, or economic data by geography. NOT for: non-US data or real-time statistics."
metadata: { "openclaw": { "emoji": "📋", "requires": { "bins": ["curl"], "env": ["CENSUS_API_KEY"] } } }
---

# US Census Data Skill

Query the US Census Bureau API for demographics, population, and economic data.

## When to Use

- "What's the population of Texas?"
- "Show demographic breakdown by county"
- "Get median household income for California"
- "Compare poverty rates across states"

## When NOT to Use

- Non-US data (use world-bank-data)
- Real-time economic data (use financial APIs)
- Historical pre-1790 data

## Setup

Get a free API key at https://api.census.gov/data/key_signup.html

```bash
export CENSUS_API_KEY=your_key_here
```

## API Base

```
https://api.census.gov/data/
```

## Common Commands

### ACS 5-Year Estimates (Most Used)

```bash
# Total population by state
curl -s "https://api.census.gov/data/2022/acs/acs5?get=NAME,B01001_001E&for=state:*&key=$CENSUS_API_KEY"

# Median household income by state
curl -s "https://api.census.gov/data/2022/acs/acs5?get=NAME,B19013_001E&for=state:*&key=$CENSUS_API_KEY"

# Population by county in California (state FIPS=06)
curl -s "https://api.census.gov/data/2022/acs/acs5?get=NAME,B01001_001E&for=county:*&in=state:06&key=$CENSUS_API_KEY"

# Poverty rate by state
curl -s "https://api.census.gov/data/2022/acs/acs5?get=NAME,B17001_001E,B17001_002E&for=state:*&key=$CENSUS_API_KEY"
```

### Decennial Census

```bash
# 2020 total population by state
curl -s "https://api.census.gov/data/2020/dec/pl?get=NAME,P1_001N&for=state:*&key=$CENSUS_API_KEY"
```

### Common Variables

| Variable | Description |
|----------|-------------|
| B01001_001E | Total population |
| B19013_001E | Median household income |
| B17001_002E | Population below poverty |
| B25077_001E | Median home value |
| B15003_022E | Bachelor's degree holders |
| B02001_002E | White alone population |

## Notes

- Free API key required for most queries
- ACS 5-year estimates are most reliable for small geographies
- FIPS codes identify states (e.g., 06=CA, 36=NY, 48=TX)
- Rate limit: 500 requests/day without key, more with key
- Variables list: https://api.census.gov/data/2022/acs/acs5/variables.html
