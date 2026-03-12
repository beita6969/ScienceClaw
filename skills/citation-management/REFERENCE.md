# citation-management — Extended Reference

> This file contains detailed tool tables, examples, and templates extracted from SKILL.md.
> The core workflow is in SKILL.md. Read this file for additional details.

## Integration with Other Skills

### Literature Review Skill

**Citation Management** provides the technical infrastructure for **Literature Review**:

- **Literature Review**: Multi-database systematic search and synthesis
- **Citation Management**: Metadata extraction and validation

**Combined workflow**:
1. Use literature-review for systematic search methodology
2. Use citation-management to extract and validate citations
3. Use literature-review to synthesize findings
4. Use citation-management to ensure bibliography accuracy

### Scientific Writing Skill

**Citation Management** ensures accurate references for **Scientific Writing**:

- Export validated BibTeX for use in LaTeX manuscripts
- Verify citations match publication standards
- Format references according to journal requirements

### Venue Templates Skill

**Citation Management** works with **Venue Templates** for submission-ready manuscripts:

- Different venues require different citation styles
- Generate properly formatted references
- Validate citations meet venue requirements

## Resources

### Bundled Resources

**References** (in `references/`):
- `google_scholar_search.md`: Complete Google Scholar search guide
- `pubmed_search.md`: PubMed and E-utilities API documentation
- `metadata_extraction.md`: Metadata sources and field requirements
- `citation_validation.md`: Validation criteria and quality checks
- `bibtex_formatting.md`: BibTeX entry types and formatting rules

**Scripts** (in `scripts/`):
- `search_google_scholar.py`: Google Scholar search automation
- `search_pubmed.py`: PubMed E-utilities API client
- `extract_metadata.py`: Universal metadata extractor
- `validate_citations.py`: Citation validation and verification
- `format_bibtex.py`: BibTeX formatter and cleaner
- `doi_to_bibtex.py`: Quick DOI to BibTeX converter

**Assets** (in `assets/`):
- `bibtex_template.bib`: Example BibTeX entries for all types
- `citation_checklist.md`: Quality assurance checklist

### External Resources

**Search Engines**:
- Google Scholar: https://scholar.google.com/
- PubMed: https://pubmed.ncbi.nlm.nih.gov/
- PubMed Advanced Search: https://pubmed.ncbi.nlm.nih.gov/advanced/

**Metadata APIs**:
- CrossRef API: https://api.crossref.org/
- PubMed E-utilities: https://www.ncbi.nlm.nih.gov/books/NBK25501/
- arXiv API: https://arxiv.org/help/api/
- DataCite API: https://api.datacite.org/

**Tools and Validators**:
- MeSH Browser: https://meshb.nlm.nih.gov/search
- DOI Resolver: https://doi.org/
- BibTeX Format: http://www.bibtex.org/Format/

**Citation Styles**:
- BibTeX documentation: http://www.bibtex.org/
- LaTeX bibliography management: https://www.overleaf.com/learn/latex/Bibliography_management

## Dependencies

### Required Python Packages

```bash
# Core dependencies
pip install requests  # HTTP requests for APIs
pip install bibtexparser  # BibTeX parsing and formatting
pip install biopython  # PubMed E-utilities access

# Optional (for Google Scholar)
pip install scholarly  # Google Scholar API wrapper
# or
pip install selenium  # For more robust Scholar scraping
```

### Optional Tools

```bash
# For advanced validation
pip install crossref-commons  # Enhanced CrossRef API access
pip install pylatexenc  # LaTeX special character handling
```

## Summary

The citation-management skill provides:

1. **Comprehensive search capabilities** for Google Scholar and PubMed
2. **Automated metadata extraction** from DOI, PMID, arXiv ID, URLs
3. **Citation validation** with DOI verification and completeness checking
4. **BibTeX formatting** with standardization and cleaning tools
5. **Quality assurance** through validation and reporting
6. **Integration** with scientific writing workflow
7. **Reproducibility** through documented search and extraction methods

Use this skill to maintain accurate, complete citations throughout your research and ensure publication-ready bibliographies.

