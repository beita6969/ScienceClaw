---
name: computational-pathology-agent
description: COPYRIGHT NOTICE
---

<!--
# COPYRIGHT NOTICE
# This file is part of the "Universal Biomedical Skills" project.
# Copyright (c) 2026 MD BABU MIA, PhD <md.babu.mia@mssm.edu>
# All Rights Reserved.
#
# This code is proprietary and confidential.
# Unauthorized copying of this file, via any medium is strictly prohibited.
#
# Provenance: Authenticated by MD BABU MIA

-->

---
name: computational-pathology-agent
description: Analyze Whole Slide Images (WSI) for digital pathology, including tissue segmentation and feature extraction.
keywords:
  - wsi
  - digital-pathology
  - deep-learning
  - resnet
  - openslide
measurable_outcome: Preprocess and extract tissue patches from a 1GB+ .svs slide within 15 minutes for downstream ML tasks.
license: MIT
metadata:
  author: MD BABU MIA, PhD
  version: "1.0.0"
compatibility:
  - system: python 3.9+
allowed-tools:
  - run_shell_command
  - read_file
  - write_file
---

# Computational Pathology Agent

**Version:** 1.0.0
**Author:** MD BABU MIA, PhD
**Date:** February 2026

## Overview
This agent specializes in the analysis of Whole Slide Images (WSIs) for digital pathology. It leverages Deep Learning models (ResNet, ViT, HoverNet) to perform segmentation, classification, and feature extraction from gigapixel histology images.

## Capabilities
1.  **WSI Handling:** Efficient reading/tiling of .svs, .ndpi, .tiff files (using OpenSlide/TiffSlide).
2.  **Tissue Segmentation:** Separation of tissue from background.
3.  **Patch Extraction:** Automated generation of patches for ML training/inference.
4.  **Nuclei Segmentation:** Integration with StarDist/HoverNet for cellular analysis.
5.  **Feature Extraction:** Generating feature vectors for slide-level clustering.

## Usage
```python
from Skills.Pathology_AI.Computational_Pathology_Agent.wsi_analyzer import WSIAnalyzer

# Initialize
path_agent = WSIAnalyzer(slide_path="./data/biopsy_001.svs")

# Extract tissue patches
path_agent.extract_patches(patch_size=256, level=1)

# Analyze Nuclei (requires model weights)
# path_agent.segment_nuclei()
```

## Requirements
*   openslide-python
*   opencv-python
*   pytorch
*   scikit-image

<!-- AUTHOR_SIGNATURE: 9a7f3c2e-MD-BABU-MIA-2026-MSSM-SECURE -->