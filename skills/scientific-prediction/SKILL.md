---
name: scientific-prediction
description: Predict material properties, economic indicators, and scientific outcomes using computational models
---

# Scientific Prediction & Simulation

## Purpose
Predict scientific outcomes, material properties, and time series using computational models and simulation.

## Key Datasets
- **Materials Project** (materials-toolkits/materials-project): 133K+ materials with DFT-computed properties (band gap, formation energy, elastic moduli, etc.)
- **FRED** (fred.stlouisfed.org): Federal Reserve Economic Data — macroeconomic time series (GDP, CPI, unemployment, interest rates)

## Protocol
1. **Problem formulation** — Define target variable, features, and prediction horizon
2. **Data preparation** — Feature engineering, normalization, train/test split
3. **Model selection** — Choose appropriate model class (regression, time series, ML, physics-informed)
4. **Training & validation** — Fit model, cross-validate, tune hyperparameters
5. **Prediction & uncertainty** — Generate predictions with confidence intervals
6. **Evaluation** — Report metrics (RMSE, MAE, R², MAPE) and compare to baselines

## Prediction Domains
- **Materials properties**: Band gap, formation energy, thermal conductivity, hardness
- **Economic forecasting**: GDP growth, inflation, employment, market indices
- **Molecular properties**: Solubility, toxicity, binding affinity, ADMET
- **Climate modeling**: Temperature trends, precipitation patterns, extreme events

## Rules
- Always report prediction uncertainty/confidence intervals
- Compare against meaningful baselines (not just random)
- Validate on held-out data (never evaluate on training data)
- For materials predictions, verify physical plausibility (positive energies, reasonable ranges)
- For economic predictions, note structural breaks and regime changes
