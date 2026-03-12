---
name: geospatial-analysis
description: Performs geospatial data analysis including GIS operations, spatial statistics, remote sensing image processing, geocoding, and cartographic visualization; trigger when users discuss maps, coordinates, satellite imagery, spatial patterns, or geographic data.
---

## When to Trigger

Activate this skill when the user mentions:
- GIS, geographic information systems, spatial data
- Coordinates, latitude/longitude, projections, CRS
- Spatial statistics, spatial autocorrelation, hotspot analysis
- Remote sensing, satellite imagery, NDVI, land cover classification
- Mapping, cartography, choropleth, heatmaps
- Geocoding, reverse geocoding, routing, network analysis
- Shapefiles, GeoJSON, raster data, vector data

## Step-by-Step Methodology

1. **Data acquisition and format assessment** - Identify data types: vector (points, lines, polygons in shapefile/GeoJSON/GeoPackage) or raster (GeoTIFF, NetCDF). Determine coordinate reference system (CRS). Check for common issues: mixed CRS, topology errors, missing geometries.
2. **Projection and transformation** - Ensure all layers share the same CRS. Use geographic CRS (WGS84/EPSG:4326) for global data, projected CRS (UTM, state plane) for area/distance calculations. Apply appropriate datum transformation.
3. **Spatial operations** - Perform geoprocessing: buffer, intersect, union, clip, dissolve. For point data: spatial joins, nearest neighbor analysis. For raster: reclassification, map algebra, zonal statistics.
4. **Spatial statistics** - Test for spatial autocorrelation (Global Moran's I). Identify clusters and hotspots (Local Moran's I / LISA, Getis-Ord Gi*). For point patterns: kernel density estimation, Ripley's K function. For regression: spatial lag or spatial error models (GWR for non-stationarity).
5. **Remote sensing analysis** - Atmospheric correction and preprocessing. Compute indices (NDVI, NDWI, NDBI). Supervised classification (random forest, SVM) or unsupervised (K-means, ISODATA). Accuracy assessment with confusion matrix and Kappa statistic.
6. **Visualization and cartography** - Create maps with proper elements: title, scale bar, north arrow, legend, data source. Use appropriate color schemes (sequential for magnitude, diverging for deviation, qualitative for categories). Consider colorblind-safe palettes.
7. **Validation** - Verify spatial operations with visual inspection and area/count checks. Cross-validate classification accuracy. Assess edge effects in spatial statistics. Report spatial resolution and positional accuracy.

## Key Databases and Tools

- **OpenStreetMap** - Open geographic data
- **USGS Earth Explorer / Copernicus Open Access Hub** - Satellite imagery
- **Natural Earth** - Public domain map data
- **Census TIGER/Line** - US geographic boundaries
- **QGIS / ArcGIS** - GIS desktop software
- **GeoPandas / Rasterio / Folium** - Python geospatial libraries
- **Google Earth Engine** - Cloud-based remote sensing platform

## Output Format

- Maps with standard cartographic elements (title, legend, scale bar, north arrow, CRS noted).
- Spatial statistics results with test statistic, p-value, and interpretation.
- Classification accuracy as confusion matrix with overall accuracy, Kappa, and per-class metrics.
- Coordinate data in standard formats (decimal degrees for geographic, meters for projected).
- GeoJSON or shapefile outputs for derived spatial data.

## Quality Checklist

- [ ] CRS explicitly stated for all datasets and outputs
- [ ] Projection appropriate for the analysis (equal-area for density, conformal for shape)
- [ ] Spatial resolution and positional accuracy documented
- [ ] Topology errors checked and cleaned
- [ ] Color scheme appropriate for data type and accessible to colorblind viewers
- [ ] Scale bar and north arrow included on all maps
- [ ] Edge effects and modifiable areal unit problem (MAUP) considered
- [ ] Data sources and vintage documented
