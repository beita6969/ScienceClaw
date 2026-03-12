---
name: patent-analysis
description: Conducts patent landscape analysis including prior art searches, patent claim interpretation, freedom-to-operate assessment, and intellectual property strategy for scientific inventions; trigger when users discuss patents, prior art, IP protection, or technology licensing.
---

## When to Trigger

Activate this skill when the user mentions:
- Patent search, prior art, novelty search
- Patent claims, claim construction, claim interpretation
- Freedom to operate (FTO), infringement analysis
- Patent landscape, technology mapping, IP portfolio
- Provisional patent, utility patent, PCT application
- Licensing, royalties, technology transfer
- Patent classification (CPC, IPC), patent families

## Step-by-Step Methodology

1. **Define the invention** - Clearly articulate the novel technical features. Distinguish the invention from the prior art. Identify the technical problem solved and the solution provided.
2. **Prior art search** - Search patent databases systematically: Google Patents, USPTO (PatFT/AppFT), Espacenet, WIPO PatentScope. Use keyword combinations, CPC/IPC classification codes, and citation analysis. Search non-patent literature (publications, conference papers) as well.
3. **Landscape analysis** - Map the patent landscape: identify key players, filing trends over time, geographic distribution, technology clusters. Use patent classification to categorize technologies. Identify white spaces (underpatented areas).
4. **Claim analysis** - Parse claims into elements. Identify independent vs. dependent claims. Map claim elements to technical features. For FTO: compare each element of relevant claims against the proposed product/method.
5. **Novelty and non-obviousness assessment** - Compare invention against closest prior art references. Identify elements that are novel (not found in any single reference) and non-obvious (not a trivial combination of references to a person of ordinary skill in the art).
6. **IP strategy recommendation** - Advise on filing strategy: provisional vs. utility, domestic vs. PCT, continuation and divisional applications. Consider timing (grace periods, priority dates). Evaluate trade secret vs. patent protection trade-offs.
7. **Documentation** - Prepare invention disclosure with: title, inventors, technical field, background, description of embodiments, and claims outline. Note that legal patent drafting requires a registered patent attorney.

## Key Databases and Tools

- **Google Patents** - Free full-text patent search
- **USPTO PatFT / AppFT** - US Patent and Trademark Office
- **Espacenet** - European Patent Office database
- **WIPO PatentScope** - International PCT applications
- **Lens.org** - Open patent and scholarly search
- **CPC / IPC classification** - Technology classification systems

## Output Format

- Prior art results as a table: patent number, title, filing date, assignee, relevance summary.
- Claim charts: claim element in left column, prior art reference in right column.
- Patent landscape as a technology map or filing trend chart.
- FTO opinion summary: clear / potential risk / likely infringement (with caveats).
- Invention disclosure following standard institutional templates.

## Quality Checklist

- [ ] Multiple patent databases searched (not just one)
- [ ] Both patent and non-patent prior art considered
- [ ] Classification codes (CPC/IPC) used to ensure comprehensive search
- [ ] Claims parsed element by element for accurate analysis
- [ ] Filing dates and priority dates correctly identified
- [ ] Patent family relationships traced (equivalents in other jurisdictions)
- [ ] Disclaimer: analysis is informational, not legal advice
- [ ] Recommendation to consult registered patent attorney for legal opinions
- [ ] Geographic scope of patent rights identified
