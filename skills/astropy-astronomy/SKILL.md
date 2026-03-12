---
name: astropy-astronomy
description: "Astronomical computations via Astropy. Use when: user asks about celestial coordinates, FITS files, or cosmological calculations. NOT for: telescope control or real-time observation planning."
metadata: { "openclaw": { "emoji": "\uD83D\uDD2D", "requires": { "bins": ["python3"] }, "install": [{ "id": "uv-astropy", "kind": "uv", "package": "astropy" }] } }
---

# Astropy Astronomy

Astronomical computations using Astropy.

## When to Use

- Celestial coordinate transforms (ICRS, Galactic, AltAz)
- Unit conversions for astronomical quantities
- Reading, writing, or inspecting FITS files
- Cosmological calculations (distances, ages, lookback times)
- Time system conversions (UTC, TAI, TDB, MJD, JD)

## When NOT to Use

- Telescope control or instrument automation
- Real-time observation planning or scheduling
- Image reduction or photometry pipelines (use photutils)
- N-body simulations

## Coordinate Transforms

```python
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u

coord = SkyCoord(ra=10.684*u.deg, dec=41.269*u.deg, frame='icrs')  # M31
coord_str = SkyCoord('00h42m44.3s', '+41d16m09s', frame='icrs')

# ICRS to Galactic
galactic = coord.galactic
print(f"l={galactic.l:.4f}, b={galactic.b:.4f}")

# Angular separation
c1 = SkyCoord(ra=10.684*u.deg, dec=41.269*u.deg)
c2 = SkyCoord(ra=11.0*u.deg, dec=41.5*u.deg)
sep = c1.separation(c2)

# AltAz (horizontal) coordinates
location = EarthLocation(lat=34.05*u.deg, lon=-118.25*u.deg, height=100*u.m)
time = Time('2026-03-15 03:00:00', scale='utc')
altaz = coord.transform_to(AltAz(obstime=time, location=location))
print(f"Alt={altaz.alt:.2f}, Az={altaz.az:.2f}")
```

## Unit Conversions

```python
import astropy.units as u

d = 10 * u.pc
print(d.to(u.lyr))         # parsecs to light-years
wav = 21 * u.cm
freq = wav.to(u.GHz, equivalencies=u.spectral())
wavelength = (13.6 * u.eV).to(u.nm, equivalencies=u.spectral())
```

## FITS File Handling

```python
from astropy.io import fits

with fits.open('image.fits') as hdul:
    hdul.info()
    header = hdul[0].header
    data = hdul[0].data

hdu = fits.PrimaryHDU(data_array)
hdu.header['OBJECT'] = 'M31'
hdu.writeto('output.fits', overwrite=True)
```

## Cosmological Calculations

```python
from astropy.cosmology import Planck18 as cosmo

z = 1.0
d_L = cosmo.luminosity_distance(z)       # luminosity distance
d_A = cosmo.angular_diameter_distance(z) # angular diameter distance
age = cosmo.age(z)                       # age of universe at z
lookback = cosmo.lookback_time(z)        # lookback time
H_z = cosmo.H(z)                        # Hubble parameter at z
```

## Time Conversions

```python
from astropy.time import Time
import astropy.units as u

t = Time('2026-03-15 12:00:00', scale='utc')
print(t.jd, t.mjd, t.unix)   # JD, MJD, Unix
print(t.tai, t.tdb)           # TAI, TDB scales
now = Time.now()
```

## Quick One-liner

```bash
python3 -c "
from astropy.coordinates import SkyCoord; import astropy.units as u
c = SkyCoord(ra=83.633*u.deg, dec=22.014*u.deg)
print(f'Galactic: l={c.galactic.l:.3f}, b={c.galactic.b:.3f}')
"
```

## Best Practices

1. Always attach units to quantities using `astropy.units`.
2. Specify time scale explicitly (`utc`, `tai`, `tdb`).
3. Use `Planck18` as default cosmology unless otherwise specified.
4. Close FITS files or use context managers to prevent resource leaks.
5. Use `SkyCoord` for all coordinate work rather than manual trig.
