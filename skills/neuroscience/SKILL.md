---
name: neuroscience
description: Supports neuroscience research including brain imaging analysis (fMRI, EEG), neural circuit modeling, cognitive experiment design, and neurological disorder investigation; trigger when users discuss brain regions, neural signals, cognitive tasks, or neuroimaging data.
---

## When to Trigger

Activate this skill when the user mentions:
- fMRI, EEG, MEG, PET, MRI brain imaging
- Neural circuits, synaptic transmission, neurotransmitters
- Cognitive experiments, reaction time, psychophysics
- Brain regions, Brodmann areas, connectome
- Neurological disorders (Alzheimer's, Parkinson's, epilepsy)
- Computational neuroscience, spiking neural networks, Hodgkin-Huxley
- Brain-computer interfaces (BCI), neural decoding

## Step-by-Step Methodology

1. **Define the neuroscience question** - Specify level of analysis (molecular, cellular, circuit, systems, cognitive, behavioral). Identify target brain regions or networks.
2. **Experimental design** - For imaging studies: specify modality (fMRI for spatial resolution, EEG for temporal resolution, PET for neurochemistry). Design task paradigm with proper controls, counterbalancing, and trial timing (ISI, ITI).
3. **Data acquisition guidance** - Recommend acquisition parameters: fMRI (TR, voxel size, field strength), EEG (sampling rate, electrode montage, impedance thresholds). Specify preprocessing steps.
4. **Preprocessing** - fMRI: slice timing, motion correction, normalization (MNI/Talairach), smoothing. EEG: filtering (bandpass), artifact rejection (ICA for eye blinks/muscle), re-referencing. Always report each step and parameters.
5. **Analysis** - fMRI: GLM for activation, seed-based or ICA for connectivity, MVPA for decoding. EEG: ERP analysis, time-frequency decomposition, source localization. Computational models: implement and fit biophysical or phenomenological models.
6. **Statistical inference** - Apply appropriate correction for multiple comparisons: cluster-level FWE for fMRI, permutation-based corrections for EEG. Report effect sizes. Use Bayesian approaches when frequentist results are ambiguous.
7. **Interpretation** - Map results to known neuroanatomy (use atlases: AAL, Desikan-Killiany, Schaefer). Discuss findings in context of established theoretical frameworks. Avoid reverse inference pitfalls.

## Key Databases and Tools

- **NeuroSynth / Neuroquery** - Meta-analytic functional maps
- **Allen Brain Atlas** - Gene expression and connectivity
- **OpenNeuro** - Open neuroimaging datasets
- **BrainMap** - Functional neuroimaging database
- **SPM / FSL / AFNI / FreeSurfer** - Neuroimaging analysis software
- **MNE-Python / EEGLAB** - EEG/MEG analysis tools
- **NEURON / Brian2** - Neural simulation environments

## Output Format

- Brain activation maps with MNI coordinates (x, y, z), cluster size, peak t/z-value.
- ERP waveforms with component labels (N1, P3, N400), latency, and amplitude.
- Time-frequency plots with frequency bands labeled (delta, theta, alpha, beta, gamma).
- Computational model parameters with biological interpretation.

## Quality Checklist

- [ ] Brain coordinates in standard space (MNI or Talairach) with atlas labels
- [ ] Multiple comparison correction method specified and justified
- [ ] Sample size adequate for imaging modality (power analysis cited)
- [ ] Preprocessing pipeline fully documented (software version, parameters)
- [ ] Task design includes appropriate controls and counterbalancing
- [ ] Effect sizes reported alongside statistical significance
- [ ] Reverse inference explicitly avoided or qualified
- [ ] Raw data sharing or availability discussed (OpenNeuro, BIDS format)
