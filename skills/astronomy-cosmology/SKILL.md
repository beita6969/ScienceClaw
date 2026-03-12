---
name: astronomy-cosmology
description: Analyzes astronomical observations and cosmological models including telescope data processing, celestial mechanics calculations, stellar evolution, galaxy classification, and cosmological parameter estimation; trigger when users discuss stars, galaxies, exoplanets, dark matter, or the universe's large-scale structure.
---

## When to Trigger

Activate this skill when the user mentions:
- Telescope observations, photometry, spectroscopy, astrometry
- Celestial mechanics, orbital calculations, Kepler's laws
- Stellar evolution, HR diagram, spectral classification
- Galaxy morphology, redshift, distance ladder
- Cosmological models, dark matter, dark energy, CMB
- Exoplanet detection, transit method, radial velocity
- Gravitational waves, black holes, neutron stars

## Step-by-Step Methodology

1. **Define the astronomical question** - Specify the object type (star, galaxy, nebula, exoplanet), observational band (optical, radio, X-ray, IR), and physical quantity of interest (distance, mass, luminosity, composition).
2. **Data acquisition** - Identify relevant surveys and archives: Gaia for astrometry, SDSS for optical spectra/photometry, 2MASS/WISE for IR, Chandra for X-ray. Download data using VO (Virtual Observatory) tools or API queries.
3. **Calibration and reduction** - Apply bias subtraction, flat-fielding, wavelength/flux calibration. For photometry: aperture or PSF fitting. For spectroscopy: sky subtraction, continuum normalization. Report signal-to-noise ratios.
4. **Physical parameter derivation** - Compute distances (parallax, standard candles, redshift-distance relation using appropriate cosmology). Derive masses (Kepler's third law, virial theorem, mass-luminosity relation). Determine compositions from spectral line analysis.
5. **Modeling** - Fit observational data with physical models: stellar atmosphere models (ATLAS, PHOENIX), N-body simulations for dynamics, cosmological models (LCDM, wCDM). Use MCMC or nested sampling for parameter estimation.
6. **Cosmological calculations** - Use standard cosmological parameters (H0, Omega_m, Omega_Lambda). Compute comoving distances, lookback times, luminosity distances. Note current tensions (H0 tension between early and late universe).
7. **Visualization** - Produce standard astronomical plots: HR diagrams, light curves, spectra, sky maps in appropriate coordinate systems (equatorial, galactic). Use logarithmic scales where appropriate.

## Key Databases and Tools

- **NASA/IPAC Extragalactic Database (NED)** - Extragalactic object data
- **SIMBAD / VizieR** - Stellar object data and catalog queries
- **Gaia Archive** - Astrometric and photometric data
- **SDSS SkyServer** - Optical survey data
- **NASA Exoplanet Archive** - Confirmed exoplanet parameters
- **Astropy** - Python astronomy library
- **MAST (STScI)** - Hubble, JWST, and other mission archives

## Output Format

- Coordinates in standard systems: RA/Dec (J2000) or Galactic (l, b).
- Distances with method and uncertainty (parallax, photometric, spectroscopic).
- Physical quantities in CGS or SI with astronomical conventions (solar units, parsecs, magnitudes).
- Spectra with wavelength/frequency axis, flux units, and line identifications.

## Quality Checklist

- [ ] Coordinate system and epoch explicitly stated
- [ ] Distance method and its systematic uncertainties discussed
- [ ] Cosmological parameters (H0, Omega_m) specified when used
- [ ] Photometric system (Vega, AB) identified for magnitudes
- [ ] Extinction/reddening corrections applied where relevant
- [ ] Instrument and survey limitations acknowledged
- [ ] Error propagation through derived quantities
- [ ] Known systematic effects (selection bias, Malmquist bias) addressed
