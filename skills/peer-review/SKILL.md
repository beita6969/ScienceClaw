---
name: peer-review
description: Conducts structured peer review of scientific manuscripts including methodological evaluation, statistical assessment, clarity analysis, and constructive feedback generation following journal-specific guidelines; trigger when users ask for manuscript critique, reviewer reports, or feedback on research papers.
---

## When to Trigger

Activate this skill when the user mentions:
- Peer review, manuscript review, referee report
- Reviewer comments, constructive criticism, revision feedback
- Journal submission evaluation, editorial assessment
- Methodological critique, statistical review
- Paper strengths and weaknesses analysis
- Response to reviewers, rebuttal letter

## Step-by-Step Methodology

1. **Initial assessment** - Read the full manuscript. Identify the research question, study design, key findings, and claimed conclusions. Assess whether the paper is within scope for the target journal.
2. **Novelty and significance evaluation** - Determine the contribution relative to existing literature. Is the advance incremental or substantial? Are similar results already published? Check for proper citation of prior work.
3. **Methods evaluation** - Assess study design appropriateness for the research question. Check for adequate controls, sample sizes, blinding, and randomization. Verify that methods are described with sufficient detail for replication.
4. **Statistical review** - Verify appropriate statistical tests for data type and design. Check for multiple comparison corrections. Assess effect sizes (not just p-values). Look for signs of p-hacking or selective reporting. Verify that assumptions of statistical tests are met.
5. **Results assessment** - Check that results directly address the stated aims. Verify figures and tables are accurate, well-labeled, and consistent with text. Look for cherry-picking or over-interpretation of data.
6. **Discussion evaluation** - Assess whether conclusions are supported by the data. Check for appropriate caveats and limitations. Evaluate whether alternative interpretations are considered.
7. **Generate structured review** - Organize feedback into: (a) Summary of the paper, (b) Major concerns (issues that must be addressed), (c) Minor concerns (suggestions for improvement), (d) Optional comments (stylistic or presentational). Be specific, constructive, and provide actionable suggestions.

## Review Structure Template

- **Summary**: 3-4 sentences describing what the paper does and finds
- **Major concerns**: Numbered list of significant issues affecting validity or interpretation
- **Minor concerns**: Numbered list of smaller issues and suggestions
- **Questions for authors**: Specific clarifications needed
- **Overall recommendation**: Accept / Minor revision / Major revision / Reject (with justification)

## Output Format

- Structured reviewer report following the template above.
- Each concern should cite specific sections, figures, or page numbers.
- Suggestions should be actionable ("Consider adding X" rather than "This is wrong").
- Separate methodological, statistical, and presentational issues.

## Quality Checklist

- [ ] Full manuscript read before writing review
- [ ] Summary accurately represents the paper's claims
- [ ] Major concerns are truly significant (not stylistic preferences)
- [ ] Statistical methods evaluated for appropriateness
- [ ] Constructive tone maintained throughout (critique the work, not the authors)
- [ ] Specific references to text, figures, and tables provided
- [ ] Alternative approaches suggested where methods are criticized
- [ ] Conflict of interest and bias in review self-assessed
- [ ] Recommendation consistent with the severity of concerns raised
