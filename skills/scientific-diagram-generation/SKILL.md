---
name: scientific-diagram-generation
description: # Scientific Diagram Generation
---

# Scientific Diagram Generation

AI-powered scientific illustration generation using Gemini Image models. Creates publication-quality mechanism diagrams, pathway illustrations, and scientific figures.

## API Configuration

| Parameter | Value |
|-----------|-------|
| **Provider** | Google Gemini via yunwu.ai relay |
| **Model** | `gemini-3.1-flash-image-preview` |
| **Base URL** | `https://yunwu.ai/v1beta/models` |
| **Full Endpoint** | `https://yunwu.ai/v1beta/models/gemini-3.1-flash-image-preview:generateContent` |
| **Auth** | `Authorization: Bearer <LLM_API_KEY>` |
| **API Key env var** | `LLM_API_KEY` (Gemini series key) |
| **Response** | Image in `candidates[].content.parts[].inlineData.data` (base64 PNG) |

## API Call Structure

```bash
curl -X POST "https://yunwu.ai/v1beta/models/gemini-3.1-flash-image-preview:generateContent" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LLM_API_KEY" \
  -d '{
    "contents": [{"role": "user", "parts": [{"text": "YOUR_PROMPT_HERE"}]}],
    "generationConfig": {
      "responseModalities": ["TEXT", "IMAGE"]
    }
  }'
```

## Python Implementation

```python
import httpx, base64

API_KEY = "your-gemini-key"
MODEL = "gemini-3.1-flash-image-preview"
URL = f"https://yunwu.ai/v1beta/models/{MODEL}:generateContent"

async def generate(prompt: str) -> bytes:
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}
    }
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(URL, json=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {API_KEY}"})
        r.raise_for_status()
        for cand in r.json().get("candidates", []):
            for part in cand["content"]["parts"]:
                if "inlineData" in part:
                    return base64.b64decode(part["inlineData"]["data"])
    return b""
```

## Style Presets

### Publication Style (default)
```
Cell/Nature/Science publication style. Realistic cell morphology with smooth
membranes. Activation arrows: solid black. Inhibition: red T-bar. Secretion:
dashed arrow. Proteins as colored ovals. Receptors as Y-shapes on membranes.
Clean, professional, suitable for journal figures.
```

### Vector-Friendly Style
```
Flat vector style with clean outlines and solid color fills. No gradients,
textures, or noise. High contrast. Easy to edit in Adobe Illustrator or Inkscape.
```

### Infographic Style
```
Modern infographic style with grid layout. Rounded rectangles for cells.
Circles for molecules. 3-5 accent colors maximum. Clean geometric shapes.
```

## Core Prompt Rules (append to every prompt)

```
CRITICAL RULES FOR SCIENTIFIC DIAGRAM GENERATION:
1. BIOLOGICAL COMPLETENESS: Name all cell types, receptors, ligands, molecules,
   transcription factors. Include activation, inhibition, binding, phosphorylation,
   secretion, translocation.
2. VISUAL COMPOSITION: Describe spatial layout (top/bottom/left/right). Define
   compartments (membrane, cytoplasm, nucleus, extracellular). Choose layout flow.
3. FONT: Use Arial or clean sans-serif for ALL labels. Never decorative fonts.
4. TEXT LABELS: Title Case for all labels. Never ALL CAPS. Gene/protein
   abbreviations kept as-is (PD-L1, IFN-γ, JAK).
5. BACKGROUND: Pure white #FFFFFF. No gradients, textures, or vignettes.
6. EDITABILITY: Elements clearly separated with sharp edges. No overlapping.
   Easy post-editing.
```

## SketchGraph Schema (for structured bio diagrams)

### BioNode Types
`cell`, `protein`, `receptor`, `mRNA`, `DNA`, `complex`, `small_molecule`, `vesicle`, `exosome`, `antibody`, `other`

### BioEdge Actions
`activate`, `inhibit`, `bind`, `phosphorylate`, `secrete`, `recruit`, `translocate`, `transcribe`, `degrade`, `upregulate`, `downregulate`

### Compartments
`membrane`, `cytoplasm`, `nucleus`, `extracellular_space`, `mitochondria`, `endoplasmic_reticulum`, `golgi`

## Prompt Templates

### Template 1: Mechanism Diagram
```
Generate a scientific mechanism diagram:
[DESCRIPTION OF THE MECHANISM]

STYLE: [publication / vector_friendly / infographic]

[CORE_PROMPT_RULES]

EXACT TEXT LABELS to render (use these exact strings, do NOT change
capitalization): ["Label1", "Label2", ...]
```

### Template 2: Signaling Pathway
```
Generate a signaling pathway diagram showing:
- Ligand: [name] binding to receptor: [name] on [cell type]
- Intracellular cascade: [kinase1] → [kinase2] → [transcription factor]
- Downstream effects: [gene expression changes]
- Compartments: extracellular, membrane, cytoplasm, nucleus

STYLE: publication
[CORE_PROMPT_RULES]
```

### Template 3: Tumor Microenvironment
```
Generate a tumor microenvironment diagram showing:
- Central: tumor cells (irregular shape, dark)
- Surrounding: [immune cells, fibroblasts, endothelial cells]
- Key interactions: [list of interactions with arrow types]
- Secreted factors: [cytokines, chemokines with dashed arrows]

Layout: radial, tumor at center
STYLE: publication
[CORE_PROMPT_RULES]
```

### Template 4: Cell Biology Overview
```
Generate a cell biology diagram showing:
- Cell with organelles: nucleus, mitochondria, ER, Golgi, lysosomes
- Process: [e.g., autophagy, apoptosis, protein trafficking]
- Key molecules at each step: [list]
- Arrows showing process flow

STYLE: publication
[CORE_PROMPT_RULES]
```

### Template 5: Non-Bio Scientific Diagram (GENERAL)
```
Generate a scientific diagram (bypassing SketchGraph, direct prompting):
[DESCRIPTION — can be circuit diagram, chemical reaction scheme,
geological cross-section, physics experiment setup, etc.]

STYLE RULES:
- White background #FFFFFF
- Arial font for all labels
- Title Case for labels
- Clean, professional, publication quality
- No decorative elements
```

## Retry Logic

The API may return 429 (rate limit) or 5xx errors. Recommended retry:
- Attempt 1: immediate
- Attempt 2: wait 15 seconds
- Attempt 3: wait 30 seconds
- Max 3 attempts

## Output

- Format: PNG (default), can request JPEG
- Resolution: ~1024x1024 (model default)
- File size: typically 800KB-1.5MB
- Save to: `~/.scienceclaw/workspace/diagrams/diagram_{uuid}.png`

## Tips for Best Results

1. **Be specific**: "TREM2 receptor on macrophage" > "a receptor on a cell"
2. **Name everything**: Every molecule, cell, and arrow should have a label
3. **Specify compartments**: "extracellular space", "cytoplasm", "nucleus"
4. **Use exact label injection**: Always provide a list of exact text labels
5. **Keep it focused**: One mechanism per diagram, not an entire pathway map
6. **Iterate**: If the first result isn't perfect, refine the prompt and regenerate
