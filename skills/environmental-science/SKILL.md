---
name: environmental-science
description: Analyzes environmental and climate data including temperature trends, pollution monitoring, ecological modeling, carbon footprint assessment, and biodiversity metrics; trigger when users discuss climate change, ecosystems, pollutants, or sustainability assessments.
---

## When to Trigger

Activate this skill when the user mentions:
- Climate data, temperature anomalies, CO2 levels, greenhouse gases
- Air/water quality, pollutant concentrations, EPA standards
- Ecological modeling, species distribution, biodiversity indices
- Carbon footprint, life cycle assessment (LCA), emissions inventory
- Remote sensing, satellite imagery for environmental monitoring
- Deforestation, habitat loss, conservation planning
- Ocean acidification, sea level rise, ice sheet dynamics

## Step-by-Step Methodology

1. **Define the environmental question** - Specify the spatial scale (local, regional, global), temporal range, and environmental domain (atmosphere, hydrosphere, lithosphere, biosphere).
2. **Data acquisition** - Identify appropriate datasets: NOAA/NASA for climate, EPA for pollution, GBIF for biodiversity, Copernicus for satellite data. Check data quality, coverage, and temporal resolution.
3. **Exploratory analysis** - Visualize spatial and temporal patterns. Plot time series for trends, anomalies, and seasonal decomposition. Map spatial distributions using appropriate projections.
4. **Statistical modeling** - Apply trend analysis (Mann-Kendall, Sen's slope for non-parametric trends). Use regression models for exposure-response relationships. For ecological data: species distribution models (MaxEnt, random forests), diversity indices (Shannon, Simpson).
5. **Impact assessment** - Quantify environmental impact using standard metrics: carbon equivalent (tCO2e), air quality index (AQI), water quality index (WQI), ecological footprint. Compare against regulatory thresholds (EPA NAAQS, WHO guidelines).
6. **Scenario analysis** - Model future projections under different scenarios (RCP/SSP pathways for climate, land-use change scenarios). Conduct sensitivity analysis on key parameters.
7. **Communication** - Present findings with clear maps, time series, and comparison to baselines. Translate technical results into policy-relevant language.

## Key Databases and Tools

- **NOAA / NASA GISS** - Climate and weather data
- **EPA / EEA** - Pollution and environmental monitoring
- **Copernicus / MODIS** - Satellite remote sensing
- **GBIF** - Global biodiversity occurrence records
- **IPCC AR6** - Climate assessment reports and scenarios
- **Our World in Data** - Environmental statistics

## Output Format

- Time series plots with trend lines, confidence bands, and anomaly baselines.
- Maps with proper projections, color scales, and legends (use diverging colormaps for anomalies).
- Impact metrics in standard units with regulatory threshold comparisons.
- Scenario projections clearly labeled with assumptions.

## Quality Checklist

- [ ] Data source, spatial resolution, and temporal coverage documented
- [ ] Baseline period defined for anomaly calculations
- [ ] Appropriate statistical tests for trend significance
- [ ] Uncertainty quantified and communicated (confidence intervals, ensemble spread)
- [ ] Regulatory standards cited with specific thresholds
- [ ] Map projection appropriate for the geographic extent
- [ ] Seasonal and cyclical patterns separated from long-term trends
- [ ] Limitations of data coverage and model assumptions stated
