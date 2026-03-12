---
name: food-science
description: Analyzes food science topics including nutritional composition, food chemistry, food safety hazard analysis, sensory evaluation design, and food processing optimization; trigger when users discuss nutrients, food additives, HACCP, shelf life, or food product development.
---

## When to Trigger

Activate this skill when the user mentions:
- Nutritional analysis, macronutrients, micronutrients, dietary reference intakes
- Food chemistry, Maillard reaction, emulsification, gelation
- Food safety, HACCP, critical control points, pathogen analysis
- Sensory evaluation, taste panels, hedonic scales
- Food processing, pasteurization, fermentation, preservation
- Shelf life, water activity, food packaging
- Dietary assessment, food frequency questionnaire, 24-hour recall

## Step-by-Step Methodology

1. **Define the food science question** - Specify the food matrix (raw ingredient, processed product, meal). Identify whether the question is about composition, safety, processing, sensory properties, or health effects.
2. **Nutritional analysis** - Query food composition databases (USDA FoodData Central, EFSA). Report per serving and per 100g. Compare against DRIs (Dietary Reference Intakes) or RDAs. Account for bioavailability and cooking losses.
3. **Food chemistry analysis** - Identify key chemical reactions (Maillard browning, lipid oxidation, enzymatic browning, starch gelatinization). Characterize relevant physical chemistry (pH, water activity, emulsion stability, rheology). Relate to quality attributes (color, texture, flavor).
4. **Food safety assessment** - Identify hazards: biological (pathogens: Salmonella, Listeria, E. coli O157:H7), chemical (pesticides, mycotoxins, heavy metals, allergens), physical (foreign objects). Apply HACCP principles: hazard analysis, critical control points, critical limits, monitoring, corrective actions.
5. **Process optimization** - Define processing parameters (temperature, time, pH, pressure). Model thermal processing (D-value, z-value, F0 calculations for sterilization). Optimize for safety while minimizing quality loss. Consider novel technologies (HPP, PEF, UV).
6. **Sensory evaluation** - Design appropriate test: discrimination (triangle, duo-trio), descriptive (QDA, CATA), or affective (hedonic, preference). Determine panel size (trained vs. consumer), number of replicates, and serving conditions. Apply appropriate statistical analysis.
7. **Shelf life estimation** - Monitor quality indicators over time (microbial counts, chemical markers, sensory scores). Model degradation kinetics (zero or first order). Apply accelerated shelf life testing (ASLT) with Arrhenius equation for temperature-dependent reactions.

## Key Databases and Tools

- **USDA FoodData Central** - US food composition database
- **EFSA / Codex Alimentarius** - Food safety standards
- **FDA Food Safety** - US food regulations
- **CompTox (EPA)** - Chemical toxicity data
- **Mintel / Innova Market Insights** - Food product trends

## Output Format

- Nutritional composition tables: nutrient, amount per serving, % DRI.
- HACCP plan as a table: hazard, CCP, critical limit, monitoring, corrective action.
- Thermal processing: D-values, z-values, F0 calculation with target organism.
- Sensory results: panel demographics, test statistics, significance levels.
- Shelf life: degradation curves, estimated shelf life with confidence intervals.

## Quality Checklist

- [ ] Food composition data source and version specified
- [ ] Serving sizes clearly defined and consistent
- [ ] HACCP analysis covers all three hazard categories (biological, chemical, physical)
- [ ] Thermal processing targets the most resistant pathogen of concern
- [ ] Sensory evaluation design includes proper controls and blinding
- [ ] Regulatory framework identified (FDA, EFSA, Codex)
- [ ] Allergen considerations addressed
- [ ] Shelf life conditions (temperature, humidity, packaging) specified
