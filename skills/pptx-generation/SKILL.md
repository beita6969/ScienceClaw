---
name: pptx-generation
description: # Academic Presentation Generation
---

# Academic Presentation Generation

## Overview
Create publication-quality academic presentations (.pptx) for group meetings, thesis defenses, conference talks, and posters.

## Templates

### Group Meeting (10-15 slides)
1. Title slide (project name, date, presenter)
2. Background / Context (1-2 slides)
3. This Week's Progress (3-5 slides)
4. Results & Figures (2-3 slides)
5. Challenges / Questions (1 slide)
6. Next Steps (1 slide)

### Thesis Defense (30-50 slides)
1. Title (name, committee, date)
2. Outline
3. Background & Significance (5-8 slides, establishing expertise)
4. Specific Aims / Research Questions (1-2 slides)
5. Aim 1: Methods → Results → Summary (8-12 slides)
6. Aim 2: Methods → Results → Summary (8-12 slides)
7. Aim 3 (if applicable)
8. Integrated Discussion (2-3 slides)
9. Conclusions & Impact (1-2 slides)
10. Future Directions (1-2 slides)
11. Acknowledgments
12. Backup slides (for committee questions)

### Conference Talk (12-20 slides, 10-15 min)
1. Title (grab attention)
2. The Problem (why should the audience care?)
3. Key Question (one sentence)
4. Our Approach (why this method?)
5. Key Results (3-5 slides, one finding per slide)
6. So What? (implications, impact)
7. Acknowledgments

## Design Principles
- **One message per slide**: if you need two messages, use two slides
- **Visual hierarchy**: title → key message → supporting data → source
- **Minimal text**: bullet points, not paragraphs. The speaker IS the narrative.
- **Figure-first**: lead with the figure, explain verbally
- **Consistent styling**: same fonts, colors, and layout throughout

## Technical Implementation
- Use python-pptx for programmatic generation
- Template-based: define slide layouts, then populate with content
- Support custom color themes and institutional branding
- Export as .pptx (editable) and .pdf (final)
