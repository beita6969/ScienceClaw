---
name: copernicus-climate
description: "Access Copernicus Climate Data Store (CDS) for ERA5 reanalysis, climate projections, and satellite observations. Use when: (1) retrieving historical weather/climate data, (2) downloading ERA5 reanalysis fields, (3) querying climate projections (CMIP), (4) getting satellite-derived climate variables. NOT for: real-time weather forecasts (use weather APIs), ocean biology (use Copernicus Marine), air quality (use CAMS)."
metadata: { "openclaw": { "emoji": "\U0001F30D", "requires": { "bins": ["python3"] } } }
---

# Copernicus Climate Data Store (CDS)

Access ERA5 reanalysis, climate projections, and satellite climate records through
the Copernicus CDS API. Covers global gridded climate data from 1940 to present.

## Prerequisites

Install the CDS API client and configure credentials:

```bash
pip install cdsapi
```

Create `~/.cdsapirc` with your CDS credentials:

```
url: https://cds.climate.copernicus.eu/api
key: <your-uid>:<your-api-key>
```

Register at https://cds.climate.copernicus.eu to obtain credentials.

## API Base URL

```
https://cds.climate.copernicus.eu/api
```

## Basic Python Retrieval Pattern

```python
import cdsapi

c = cdsapi.Client()

c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": "2m_temperature",
        "year": "2023",
        "month": "07",
        "day": "15",
        "time": "12:00",
        "area": [60, -10, 35, 30],  # N, W, S, E bounding box
        "format": "netcdf",
    },
    "era5_temperature.nc",
)
```

## ERA5 Pressure-Level Variables

Retrieve upper-air data on pressure levels:

```python
c.retrieve(
    "reanalysis-era5-pressure-levels",
    {
        "product_type": "reanalysis",
        "variable": ["temperature", "geopotential", "relative_humidity"],
        "pressure_level": ["500", "700", "850", "925"],
        "year": "2023",
        "month": "01",
        "day": "15",
        "time": "12:00",
        "format": "netcdf",
    },
    "era5_pressure_levels.nc",
)
```

## Key Dataset Identifiers

| Dataset ID                              | Description                              |
|-----------------------------------------|------------------------------------------|
| `reanalysis-era5-single-levels`         | Surface and single-level hourly fields   |
| `reanalysis-era5-pressure-levels`       | Upper-air on 37 pressure levels          |
| `reanalysis-era5-single-levels-monthly` | Monthly-averaged surface fields          |
| `reanalysis-era5-land`                  | ERA5-Land (enhanced land, 9 km)          |
| `satellite-sea-level-global`            | Satellite altimetry sea level            |

## Common Variables

**Single level**: `2m_temperature`, `total_precipitation`, `10m_u_component_of_wind`,
`10m_v_component_of_wind`, `mean_sea_level_pressure`, `surface_solar_radiation_downwards`.

**Pressure level**: `temperature`, `geopotential`, `relative_humidity`, `specific_humidity`.

## Processing Downloaded NetCDF

```python
import xarray as xr
ds = xr.open_dataset("era5_temperature.nc")
temp_celsius = ds["t2m"] - 273.15  # Kelvin to Celsius
print(f"Mean temperature: {float(temp_celsius.mean()):.1f} C")
```

## Area Selection (N, W, S, E bounding box)

Global: `[90, -180, -90, 180]`, Europe: `[72, -25, 33, 45]`,
Continental US: `[50, -125, 25, -65]`, East Asia: `[55, 70, 5, 145]`.

## Best Practices

1. Specify the smallest area and fewest variables needed to reduce download time.
2. Use monthly-averaged datasets when daily resolution is not required.
3. Request data in NetCDF format for analysis; GRIB for operational workflows.
4. CDS queues requests; large jobs may take hours. Check status via the web dashboard.
5. ERA5 data is available from 1940 to present with ~5-day latency.
6. For multi-year bulk downloads, split requests by year to avoid timeouts.
7. Install `xarray` and `netCDF4` for reading downloaded files in Python.
