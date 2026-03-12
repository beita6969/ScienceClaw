---
name: energy-systems
description: Analyzes energy systems including renewable energy resource assessment, power grid modeling, battery storage optimization, energy efficiency evaluation, and techno-economic analysis of energy technologies; trigger when users discuss solar, wind, grid integration, energy storage, or power system design.
---

## When to Trigger

Activate this skill when the user mentions:
- Solar energy, photovoltaic, wind power, hydropower
- Power grid, load balancing, dispatch optimization
- Battery storage, lithium-ion, energy density, cycling
- Energy efficiency, HVAC, building energy modeling
- Techno-economic analysis, LCOE, payback period
- Electric vehicles, charging infrastructure, V2G
- Hydrogen economy, fuel cells, electrolysis

## Step-by-Step Methodology

1. **Define the energy system scope** - Specify system boundaries: single building, microgrid, regional grid, or national scale. Identify energy sources (solar, wind, fossil, nuclear, hydro), storage technologies, and demand profiles.
2. **Resource assessment** - For solar: analyze irradiance data (GHI, DNI, DHI), calculate capacity factor, account for degradation and soiling. For wind: analyze wind speed distributions (Weibull), compute power curves, assess turbulence intensity. Use TMY (Typical Meteorological Year) data or site-specific measurements.
3. **System modeling** - Size components: panels/turbines (capacity), inverters, batteries (energy and power), converters. Model energy balance: generation - consumption - storage - curtailment = grid exchange. Use hourly or sub-hourly time resolution.
4. **Grid integration** - Analyze grid interconnection requirements: voltage, frequency, power factor. Assess variability and ramping impacts. Model dispatch optimization (merit order, economic dispatch, unit commitment). Evaluate ancillary services potential.
5. **Storage analysis** - Characterize storage technology: energy density (Wh/kg), power density (W/kg), round-trip efficiency, cycle life, calendar life, self-discharge rate. Optimize sizing based on arbitrage value, peak shaving, or reliability requirements.
6. **Economic analysis** - Calculate LCOE (levelized cost of energy) with discount rate, capital costs, O&M, fuel costs, and lifetime. Compute NPV, IRR, and payback period. Include incentives (ITC, PTC, feed-in tariffs). Conduct sensitivity analysis on key assumptions.
7. **Environmental assessment** - Calculate avoided CO2 emissions using grid emission factors. Perform lifecycle emissions analysis (cradle-to-gate). Compare with conventional alternatives.

## Key Databases and Tools

- **NREL (SAM, PVWatts, NSRDB)** - Solar and renewable energy tools
- **Global Wind Atlas** - Wind resource data
- **EIA / IEA** - Energy statistics and projections
- **HOMER Energy** - Microgrid optimization
- **OpenDSS** - Distribution system simulation
- **PyPSA** - Open-source power system analysis

## Output Format

- Resource assessment as tables: annual/monthly capacity factor, energy yield (kWh/kWp).
- System diagram with component sizes, power flows, and energy balance.
- LCOE breakdown: capital, O&M, fuel, financing costs per kWh.
- Economic results as NPV, IRR, payback period with sensitivity tornado chart.
- Time series plots: generation, demand, storage state-of-charge, grid exchange.

## Quality Checklist

- [ ] Resource data source and time resolution specified
- [ ] System losses itemized (inverter, wiring, degradation, soiling, curtailment)
- [ ] Discount rate and financial assumptions documented
- [ ] Storage degradation and replacement costs included in economics
- [ ] Grid emission factor source and year specified
- [ ] Sensitivity analysis covers key uncertainties (resource, cost, discount rate)
- [ ] Units consistent (kW vs. kWh, AC vs. DC clearly distinguished)
- [ ] Comparison with alternatives provided for context
