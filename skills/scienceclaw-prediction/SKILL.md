---
name: scienceclaw-prediction
description: "Predict scientific properties, trends, and outcomes. Use when: user asks for property prediction, trend forecasting, or model-based estimation. NOT for: historical data lookup or real-time monitoring."
metadata: { "openclaw": { "emoji": "📈" } }
---

# Scientific Prediction Skill

Predict properties, trends, and outcomes across scientific disciplines.

## When to Use

- "Predict the solubility of this compound"
- "What's the expected trend for..."
- "Estimate the effect size for this intervention"
- "Forecast the trajectory of..."
- "What properties would this material have?"
- Model-based estimation tasks

## When NOT to Use

- Looking up known properties (use literature-search)
- Running actual computations (use code-execution)
- Verifying existing predictions (use scienceclaw-verification)
- Real-time data monitoring or alerts

## Prediction Categories

### 1. Property Prediction
- **Chemistry**: Molecular properties (logP, solubility, toxicity, pKa, boiling point)
- **Materials**: Mechanical (strength, hardness), thermal, electrical properties
- **Biology**: Protein function, binding affinity, gene expression levels
- **Physics**: Material behavior under conditions (temperature, pressure)

### 2. Trend Analysis
- Time-series extrapolation with confidence intervals
- Growth/decay curve fitting (exponential, logistic, polynomial)
- Seasonal pattern identification
- Regime change detection

### 3. Outcome Prediction
- Clinical trial outcome estimation
- Experimental result prediction
- Treatment response probability
- Environmental impact forecasting

### 4. Model-Based Estimation
- QSAR/QSPR (quantitative structure-activity/property relationships)
- Pharmacokinetic modeling (ADME)
- Population dynamics modeling
- Economic indicator forecasting

## Output Format

All predictions must include:

```
**Prediction**: [Value or range]
**Confidence Interval**: [Lower - Upper] at [confidence level]%
**Method**: [Approach used]
**Key Assumptions**: [List]
**Uncertainty Sources**: [List]
**Validation**: [How to verify this prediction]
**Caveats**: [Known limitations]
```

## Guidelines

1. **Always quantify uncertainty** — never provide point estimates without ranges
2. **State assumptions explicitly** — hidden assumptions undermine predictions
3. **Distinguish extrapolation from interpolation** — flag when predicting outside training data range
4. **Consider domain constraints** — physical laws, biological limits, economic boundaries
5. **Recommend validation approaches** — suggest experiments or data to verify predictions
6. **Use appropriate models** — match model complexity to data availability
7. **Flag low-confidence predictions** — be transparent about reliability

## Discipline-Specific Methods

| Domain | Common Methods |
|--------|---------------|
| Chemistry | QSAR, DFT calculations, molecular dynamics |
| Biology | Sequence-based prediction, network analysis |
| Medicine | Cox regression, Kaplan-Meier, NNT/NNH |
| Physics | Theoretical models, scaling laws |
| Economics | Econometric models, agent-based simulation |
| Climate | GCM projections, statistical downscaling |
| Materials | Phase diagrams, computational screening |
| Sociology | Panel data models, social network evolution |

## Computational Prediction Tools

When predictions require computation, integrate with these skills:

### Molecular Property Prediction
- Use **rdkit-chemistry** for descriptor-based QSAR models
- Use **pubchem-compound** to retrieve experimental property values for training data
- Use **scikit-learn-ml** to build/evaluate prediction models

### Materials Property Prediction
- Use **materials-project** to retrieve DFT-computed properties
- Use **pymatgen-materials** for structure-property analysis
- Use **scipy-analysis** for interpolation and regression

### Biological Outcome Prediction
- Use **biopython-bio** for sequence-based feature extraction
- Use **transformers-inference** for protein language models (ESM, ProtTrans)
- Use **scanpy-singlecell** for cell-type and trajectory prediction

### Geospatial/Climate Prediction
- Use **geopandas-spatial** for spatial feature engineering
- Use **copernicus-climate** for historical climate data as training input
- Use **statsmodels-stats** for time-series forecasting models

## Zero-Hallucination Rule

ALL factual claims, citations, database results, and scientific data presented to the user MUST come from actual tool results (API calls, code execution, web search) in this conversation. NEVER fabricate or "fill in" details from training data. If a tool returns no results or partial data, report exactly what happened.
