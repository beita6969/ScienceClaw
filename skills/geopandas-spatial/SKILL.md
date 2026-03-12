---
name: geopandas-spatial
description: "Geospatial and climate data analysis with geopandas and xarray. Use when: (1) geographic vector data analysis, (2) climate and weather NetCDF data, (3) spatial joins and overlay operations, (4) map visualization, (5) CRS transformations and area calculations. NOT for: satellite imagery ML (use rasterio/torchgeo), real-time GPS tracking, or interactive web maps (use folium/leaflet directly)."
metadata: { "openclaw": { "emoji": "🗺️", "requires": { "bins": ["python3"] }, "install": [{ "id": "uv-geopandas", "kind": "uv", "package": "geopandas xarray netcdf4 shapely" }] } }
---

# GeoPandas Spatial Analysis

Geospatial data analysis using geopandas for vector data and xarray for
multidimensional climate/weather datasets.

## When to Use

- Loading and analyzing shapefiles, GeoJSON, GeoPackage
- Spatial joins, intersections, buffers, and dissolves
- Climate and weather data from NetCDF files
- CRS transformations and geographic projections
- Map visualization and choropleth maps
- Area, distance, and geometric calculations

## When NOT to Use

- Satellite imagery classification or ML (use rasterio/torchgeo)
- Real-time GPS tracking or routing
- Interactive web map applications (use folium or deck.gl)
- General tabular data without spatial component (use pandas)

## Reading Vector Data

```python
import geopandas as gpd

# Read various vector formats
gdf = gpd.read_file("boundaries.shp")
gdf = gpd.read_file("data.geojson")
gdf = gpd.read_file("database.gpkg", layer="cities")

# Read from URL
gdf = gpd.read_file("https://example.com/regions.geojson")

# Inspect the GeoDataFrame
print(gdf.head())
print(gdf.crs)                    # coordinate reference system
print(gdf.geometry.type.unique()) # geometry types present
print(gdf.total_bounds)           # [minx, miny, maxx, maxy]
```

## CRS Transformations and Projections

```python
# Check and set CRS
print(gdf.crs)                                # e.g., EPSG:4326 (WGS84)
gdf = gdf.set_crs("EPSG:4326")               # assign if missing

# Reproject to a different CRS
gdf_proj = gdf.to_crs("EPSG:3857")           # Web Mercator
gdf_utm = gdf.to_crs("EPSG:32633")           # UTM Zone 33N

# Area calculation (reproject to equal-area CRS first)
gdf_equal = gdf.to_crs("ESRI:54009")         # Mollweide equal-area
gdf_equal["area_km2"] = gdf_equal.geometry.area / 1e6
```

## Spatial Operations

```python
from shapely.geometry import Point, Polygon, box

# Create geometries
point = Point(-73.985, 40.748)
polygon = Polygon([(-74, 40.7), (-74, 40.8), (-73.9, 40.8), (-73.9, 40.7)])
bbox = box(-74.05, 40.68, -73.90, 40.82)

# Spatial joins
joined = gpd.sjoin(points_gdf, polygons_gdf, how="inner", predicate="within")

# Buffer around geometries (in CRS units)
gdf_buffered = gdf.copy()
gdf_buffered["geometry"] = gdf.geometry.buffer(1000)  # 1000m if projected CRS

# Dissolve by attribute (merge geometries)
dissolved = gdf.dissolve(by="region", aggfunc="sum")

# Overlay operations
intersection = gpd.overlay(gdf1, gdf2, how="intersection")
union = gpd.overlay(gdf1, gdf2, how="union")
difference = gpd.overlay(gdf1, gdf2, how="difference")

# Clip to bounding box or polygon
clipped = gpd.clip(gdf, mask=bbox_gdf)

# Nearest join
nearest = gpd.sjoin_nearest(points_gdf, target_gdf, how="left", distance_col="dist_m")
```

## Map Visualization

```python
import matplotlib.pyplot as plt

# Basic plot
ax = gdf.plot(figsize=(12, 8), edgecolor="black", linewidth=0.5)
ax.set_title("Geographic Boundaries")

# Choropleth map
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
gdf.plot(column="population", cmap="YlOrRd", legend=True,
         legend_kwds={"label": "Population"}, ax=ax,
         edgecolor="gray", linewidth=0.3)
ax.set_axis_off()
plt.savefig("choropleth.pdf", bbox_inches="tight")

# Multi-layer map
fig, ax = plt.subplots(figsize=(12, 8))
polygons_gdf.plot(ax=ax, color="lightblue", edgecolor="gray")
points_gdf.plot(ax=ax, color="red", markersize=5)
lines_gdf.plot(ax=ax, color="darkblue", linewidth=1)
plt.savefig("multi_layer.pdf", bbox_inches="tight")
```

## NetCDF and Climate Data with xarray

```python
import xarray as xr
import numpy as np

# Open a single NetCDF file
ds = xr.open_dataset("climate.nc")
print(ds)                                     # dimensions, variables, coords
print(ds.data_vars)                           # available variables

# Open multiple files (e.g., monthly data)
ds = xr.open_mfdataset("data_*.nc", combine="by_coords")

# Select by coordinates
temp = ds["temperature"]
subset = temp.sel(lat=slice(30, 50), lon=slice(-100, -70))
single_time = temp.sel(time="2020-06-15", method="nearest")

# Time series operations
annual_mean = ds.groupby("time.year").mean(dim="time")
monthly_clim = ds.groupby("time.month").mean(dim="time")
seasonal = ds.resample(time="QS-DEC").mean()

# Spatial mean
area_avg = temp.mean(dim=["lat", "lon"])

# Save processed data
ds_subset.to_netcdf("processed_output.nc")
```

## Combining xarray with GeoPandas

```python
# Extract xarray values at point locations
import xarray as xr
import geopandas as gpd

ds = xr.open_dataset("climate.nc")
stations = gpd.read_file("stations.geojson")

# Sample raster values at station coordinates
for idx, row in stations.iterrows():
    val = ds["temperature"].sel(
        lat=row.geometry.y, lon=row.geometry.x, method="nearest"
    ).values
    stations.loc[idx, "temperature"] = float(val)
```

## Best Practices

1. Always check and set CRS before spatial operations; mismatched CRS cause errors.
2. Reproject to an equal-area CRS (e.g., Mollweide) before area calculations.
3. Use projected CRS (meters) for buffer and distance operations, not EPSG:4326.
4. Use `xr.open_mfdataset()` with `chunks=` for large multi-file climate datasets.
5. Close datasets with `ds.close()` or use context managers for large files.
6. Prefer `gpd.sjoin_nearest()` over manual distance loops for point matching.
7. Use `.dissolve()` instead of manual groupby for merging geometries.
8. Write intermediate results to GeoPackage (`.gpkg`) for multi-layer support.
