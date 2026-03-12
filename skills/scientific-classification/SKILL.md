---
name: scientific-classification
description: Classify scientific objects, detect patterns, and categorize data across astronomy, biology, and social sciences
---

# Scientific Classification & Detection

## Purpose
Classify scientific objects and detect patterns using established taxonomies and classification schemes.

## Key Datasets
- **SDSS Stellar Classification** (Allanatrix/Astro): 100K objects from SDSS DR17 — Stars, Galaxies, Quasars with photometric features (u, g, r, i, z magnitudes, redshift)
- **Social Bias Frames** (allenai/social_bias_frames): Allen AI SBIC corpus for detecting implicit social biases in text

## Protocol
1. **Feature extraction** — Identify relevant features for classification task
2. **Taxonomy mapping** — Map to standard classification scheme
3. **Classification** — Apply appropriate classifier with confidence scores
4. **Validation** — Cross-validate against known labeled examples
5. **Edge case analysis** — Flag ambiguous or borderline cases

## Classification Domains
- **Astronomical objects**: Stellar spectral types (OBAFGKM), galaxy morphology (Hubble), AGN types
- **Biological taxonomy**: Species classification, protein families, cell types
- **Chemical compounds**: Functional groups, drug classes, toxicity levels
- **Text classification**: Sentiment, bias detection, topic classification
- **Image classification**: Histopathology, satellite imagery, microscopy

## Rules
- Report classification confidence and alternative labels
- Use domain-standard taxonomies (not ad-hoc categories)
- Handle multi-label and hierarchical classification
- Document decision boundaries and feature importance
