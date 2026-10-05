---
name: scienceclaw-benchmark-for51
description: "Use when a task asks to predict a phonon property of inorganic crystals from their structure, for example the frequency of the last peak of the phonon density of states in cm^-1 (Matbench phonons), given structures (lattice, species, fractional coordinates) and a labelled training set; the deliverable is a float vector in input order, judged by mean absolute error. Covers composition/geometry descriptors, log-target tree regressors and pretrained universal interatomic potentials (SevenNet, CHGNet) as phonon feature extractors. Corresponds to FoR51 of the companion ScienceClaw-Eval benchmark."
metadata: { "openclaw": { "emoji": "📊" } }
---

# Phonon frequency from crystal structure

## Task
- Inputs: structures `{formula, lattice (3x3 row vectors, angstrom), species, frac_coords}`; labelled training structures with the target in cm^-1; structures to predict.
- Deliverable: a 1-D float array, one value per structure in input order, in cm^-1 (1 THz = 33.356 cm^-1; convert explicitly if a tool returns THz).

## Quality
Mean absolute error in cm^-1, lower is better. A sensible baseline is 5-nearest-neighbour regression on standardised composition/geometry descriptors.

## Tools
- Descriptors are built in a code node (no scilib function): site-weighted statistics of atomic number, mass, electronegativity, covalent radius, group, period and valence electrons, plus n_sites, volume per atom, density, packing fraction and nearest-neighbour distances.
- `scilib.matphonon_regression.fit_predict(X_train, y_train, X_eval, ...)`: ExtraTrees on the log target, three seeds averaged, clipped to 0..5000 cm^-1 (typed operator `phonon_frequency_regression`); `fit_hist_predict` is a HistGradientBoosting alternative when trees over-smooth (`phonon_frequency_regression_boosted`); `cross_validate(X, y, folds=5)` gives `mae` (`phonon_regression_cv_mae`). `X` must be finite and `y` positive; `return_info=True` adds provenance.
- `scilib.matphonon_mlip.phonon_features(structures, mesh=8, min_len=7.0, disp=0.01, model="sevennet")`: phonopy finite displacements on the unrelaxed structure with a pretrained universal potential (`model` is `sevennet`, `sevennet_mf0_pbe`, `sevennet_mf0_r2scan` or `chgnet`); returns `{"X": (n, 21), "names": [...]}` of frequency statistics and DOS peaks in cm^-1, NaN rows for failed structures (operators `phonon_feature_matrix_mlip`, `phonon_features_mlip`). Check `scilib.matphonon_mlip.available()` and `scienceclaw_tools(operation=status)`; needs `sevenn` or `chgnet`, plus `phonopy` and `torch`; weights ship with the pip packages (asset ids `sevennet`, `chgnet`). About 0.25 s (CHGNet) to 0.9 s (SevenNet) per structure on a GPU. Concatenate these columns with the descriptors and fit the regressor.

## Rules
- Universal potentials soften frequencies systematically: use the MLIP columns as regression inputs, never as predictions.
- Fit only on labelled training rows; impute or drop NaN feature rows with training statistics only.
- SevenNet-l3i5 and CHGNet were trained on Materials Project trajectories (energies, forces, stresses, not phonon frequencies), which can include the task crystals; disclose that overlap.
- Shuffled K-fold is optimistic when new crystals contain elements absent from training; estimate under shift with leave-element-out folds.
- Keep unit and shape; predictions finite and positive. SevenNet on CPU repeats only to about 1e-5 relative, so set `SCIENCECLAW_MLIP_CACHE` for exact replay.
