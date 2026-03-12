---
name: signal-processing
description: Performs signal processing tasks including spectral analysis (FFT), digital filtering, time-frequency decomposition, noise reduction, and modulation/demodulation; trigger when users discuss waveforms, frequency spectra, filters, or time series in engineering contexts.
---

## When to Trigger

Activate this skill when the user mentions:
- FFT, DFT, spectral analysis, power spectral density
- Digital filters (FIR, IIR), Butterworth, Chebyshev
- Time-frequency analysis, STFT, wavelets, spectrograms
- Signal denoising, SNR, noise floor
- Sampling, Nyquist theorem, aliasing, ADC/DAC
- Modulation (AM, FM, QAM), demodulation, baseband
- Convolution, correlation, matched filtering

## Step-by-Step Methodology

1. **Signal characterization** - Identify signal type (continuous/discrete, deterministic/stochastic, stationary/non-stationary). Determine sampling rate, duration, and bit depth. Check Nyquist criterion (fs > 2*fmax).
2. **Preprocessing** - Remove DC offset (mean subtraction). Apply windowing (Hann, Hamming, Blackman) to reduce spectral leakage. Handle missing data or outliers. Normalize amplitude if needed.
3. **Spectral analysis** - Compute FFT with appropriate zero-padding for frequency resolution. Estimate power spectral density (Welch's method for noise reduction, periodogram for snapshot). Identify dominant frequency components and harmonics.
4. **Filtering** - Design filter based on requirements: passband/stopband frequencies, ripple, attenuation. Choose type: FIR (linear phase, higher order) or IIR (lower order, nonlinear phase). Implement using appropriate method (Parks-McClellan for FIR, bilinear transform for IIR).
5. **Time-frequency analysis** - For non-stationary signals: compute STFT (spectrogram) with appropriate window size trade-off. Apply wavelet transform (CWT for analysis, DWT for decomposition/compression). Select mother wavelet (Morlet for frequency, Daubechies for transients).
6. **Denoising** - Estimate noise characteristics (white, colored, impulsive). Apply appropriate method: spectral subtraction, Wiener filter, wavelet thresholding (soft/hard), or adaptive filtering (LMS, RLS).
7. **Validation** - Verify filter response meets specifications (frequency response, phase response, group delay). Check for artifacts (ringing, Gibbs phenomenon). Compute output SNR improvement.

## Key Databases and Tools

- **SciPy signal** - Python signal processing functions
- **MATLAB Signal Processing Toolbox** - Comprehensive DSP tools
- **GNU Radio** - Software-defined radio framework
- **Librosa** - Audio signal processing
- **PyWavelets** - Wavelet transform library

## Output Format

- Frequency spectra with labeled axes (Hz or normalized frequency, dB or linear magnitude).
- Filter specifications: type, order, cutoff frequencies, passband ripple, stopband attenuation.
- Time-frequency plots (spectrograms) with time, frequency, and magnitude axes.
- SNR values in dB before and after processing.
- Transfer function coefficients (numerator b, denominator a for IIR; taps for FIR).

## Quality Checklist

- [ ] Sampling rate and Nyquist criterion verified
- [ ] Windowing function specified and justified
- [ ] FFT length and frequency resolution stated
- [ ] Filter order and stability verified (all poles inside unit circle for IIR)
- [ ] Phase response considered (linear phase requirement?)
- [ ] Group delay acceptable for application
- [ ] SNR improvement quantified
- [ ] Edge effects and transient responses handled
