---
name: academic-literature-search
description: # Academic Literature Search — 学术文献检索与引用管理
---

# Academic Literature Search — 学术文献检索与引用管理

Use this skill when the user asks to search for academic papers, retrieve literature, generate citations, format references, or any task involving PubMed, bioRxiv, arXiv, or academic reference management. Trigger keywords: "搜文献", "检索", "找论文", "参考文献", "引用", "citation", "search papers", "PubMed", "bioRxiv", "arXiv", "GB/T 7714", "PMID", "DOI", "批量引用".

## Core Principles

1. **MCP first, Python second**: PubMed operations → MCP tools (zero code). Python only for arXiv, GB/T 7714 formatting, and citation post-processing.
2. **Code-driven citations**: Citation formatting, validation, deduplication, renumbering — ALL via Python code. NEVER fabricate PMIDs, DOIs, author names, or journal names.
3. **GB/T 7714-2015 sequential numbering**: `[1][2][3]` in-text, references numbered by order of first appearance.
4. **Journal name consistency**: Use **full journal names** throughout (NOT ISO abbreviations). E.g., `Nature Medicine` not `Nat Med`. If MCP returns abbreviated names, expand them; if expansion is uncertain, use the name as returned.

---

## Tool Routing Decision Table

| 操作 | 用什么 | 为什么 |
|------|--------|--------|
| PubMed 关键词搜索 | **MCP** `pubmed_search_articles` | 原生日期/类型过滤/排序，Agent 零代码 |
| PMID 批量获取详情 | **MCP** `pubmed_fetch_contents` | 4种详情级别，一次200个，含 MeSH |
| 相似论文发现 | **MCP** `pubmed_article_connections` (similar) | 直接调用，返回结构化数据 |
| 被引论文发现 | **MCP** `pubmed_article_connections` (citedin) | 同上 |
| 论文参考文献（它引了谁） | **MCP** `pubmed_article_connections` (references) | **MCP 独有**，ELink 不支持 |
| RIS/BibTeX 导出 | **MCP** `pubmed_article_connections` (citation_formats) | 内置格式化 |
| bioRxiv/medRxiv 搜索 | **MCP** `pubmed_search_articles` + journal filter | queryTerm 加 `biorxiv[journal]` |
| arXiv 搜索 | **Python** | MCP 不覆盖 arXiv |
| GB/T 7714-2015 格式化 | **Python** | MCP 无国标格式 |
| 引用后处理/去重/编号 | **Python** | MCP 不覆盖 |

---

## MCP Operations (PubMed — 主力)

### 1. 关键词搜索

```
Tool: pubmed_search_articles
Parameters:
  queryTerm: "large language model bioinformatics"
  maxResults: 20
  sortBy: "relevance"                       ← 或 "pub_date"
  fetchBriefSummaries: 10                   ← 返回前10篇摘要
  dateRange:                                ← 可选
    minDate: "2022"
    maxDate: "2026"
    dateType: "pdat"
  filterByPublicationTypes: ["Review"]      ← 可选
```

**bioRxiv/medRxiv**：queryTerm 加 journal filter：
```
queryTerm: "(large language model agent) AND (biorxiv[journal] OR medrxiv[journal])"
```

### 2. PMID 批量获取详情

```
Tool: pubmed_fetch_contents
Parameters:
  pmids: ["39361263", "38768397", "36869294"]
  detailLevel: "abstract_plus"              ← 推荐，解析后的结构化数据
  includeMeshTerms: true
```

详情级别：`abstract_plus`（推荐）| `citation_data`（轻量）| `full_xml` | `medline_text`

### 3. 相关论文发现

```
Tool: pubmed_article_connections
Parameters:
  sourcePmid: "39361263"
  relationshipType: "pubmed_similar_articles"   ← 或 citedin / references / citation_formats
  maxRelatedResults: 15
```

### 4. 引用格式导出

```
Tool: pubmed_article_connections
Parameters:
  sourcePmid: "39361263"
  relationshipType: "citation_formats"
  citationStyles: ["ris", "bibtex", "apa_string"]
```

---

## Python Operations

### arXiv 搜索

Uses stdlib only (`urllib.request` + `xml.etree.ElementTree`), no external dependencies:

```python
import urllib.request, urllib.parse, xml.etree.ElementTree as ET, re

def _parse_author_name(full_name):
    """Convert 'Shunyu Yao' → 'Yao S' (GB/T 7714: surname first, initials)."""
    parts = full_name.strip().split()
    if not parts:
        return full_name
    if len(parts) == 1:
        return parts[0]
    surname = parts[-1]
    initials = "".join(p[0].upper() for p in parts[:-1])
    return f"{surname} {initials}"


def search_arxiv(query, max_results=10):
    """Search arXiv. Free API, no key needed. Uses only stdlib."""
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}", "start": 0,
        "max_results": max_results, "sortBy": "relevance",
        "sortOrder": "descending"
    })
    url = f"https://export.arxiv.org/api/query?{params}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        xml_text = resp.read().decode("utf-8")

    ns = {"atom": "http://www.w3.org/2005/Atom",
          "arxiv": "http://arxiv.org/schemas/atom"}
    root = ET.fromstring(xml_text)
    articles = []
    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", "", ns) or "").replace("\n", " ").strip()
        abstract = (entry.findtext("atom:summary", "", ns) or "").replace("\n", " ").strip()

        raw_authors = [a.findtext("atom:name", "", ns).strip()
                       for a in entry.findall("atom:author", ns)]
        authors_gbt = [_parse_author_name(a) for a in raw_authors]

        published = entry.findtext("atom:published", "", ns)[:10]
        id_url = entry.findtext("atom:id", "", ns) or ""
        arxiv_id = id_url.split("/abs/")[-1] if "/abs/" in id_url else ""

        doi_e = entry.find("arxiv:doi", ns)
        doi = doi_e.text.strip() if doi_e is not None and doi_e.text else ""

        cat_e = entry.find("arxiv:primary_category", ns)
        cat = cat_e.get("term", "") if cat_e is not None else ""

        articles.append({
            "source": "arxiv", "pmid": "", "doi": doi, "arxiv_id": arxiv_id,
            "title": title, "authors": authors_gbt,
            "journal": f"arXiv:{arxiv_id}",
            "year": published[:4] if published else "",
            "volume": "", "issue": "", "pages": "",
            "abstract": abstract,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "category": cat,
            "venue": "",  # populated manually for published conference papers
        })
    return articles
```

### GB/T 7714-2015 格式化 (完整版)

支持 [J] 期刊、[Z/OL] 预印本、[C] 会议论文、[M] 专著、[D] 学位论文。
所有字段均做 None 安全处理。

```python
def format_gbt7714(article, seq_num):
    """Format one article as GB/T 7714-2015 sequential reference.

    Handles: MCP article dicts (authors as list of dicts, journalInfo as nested dict),
             Python-parsed dicts (flat fields), and arXiv dicts.
    Supports: [J] journal, [Z/OL] preprint, [C] conference, [M] book, [D] dissertation.
    """
    # ── Author parsing (handles 4 input formats) ──
    authors_raw = article.get("authors") or []
    if isinstance(authors_raw, str):
        authors = [a.strip() for a in authors_raw.split(",") if a.strip()]
    elif authors_raw and isinstance(authors_raw[0], dict):
        authors = []
        for a in authors_raw:
            last = a.get("lastName") or ""
            ini = a.get("initials") or (a.get("firstName") or "")[:1]
            if last:
                authors.append(f"{last} {ini}".strip())
    else:
        authors = [str(a) for a in authors_raw if a]

    if len(authors) > 3:
        author_str = ", ".join(authors[:3]) + ", et al"
    elif authors:
        author_str = ", ".join(authors)
    else:
        author_str = "Anonymous"

    title = (article.get("title") or "Untitled").rstrip(".")
    source = article.get("source") or "pubmed"
    doi = (article.get("doi") or "").strip()

    # ── Journal info (MCP nested dict vs flat fields) ──
    journal_info = article.get("journalInfo")
    if journal_info and isinstance(journal_info, dict):
        journal = journal_info.get("title") or journal_info.get("isoAbbreviation") or ""
        volume = journal_info.get("volume") or ""
        issue = journal_info.get("issue") or ""
        pages = journal_info.get("pages") or ""
        pub_date = journal_info.get("publicationDate") or {}
        year = pub_date.get("year", "") if isinstance(pub_date, dict) else str(pub_date)[:4]
    else:
        journal = article.get("journal") or ""
        volume = article.get("volume") or ""
        issue = article.get("issue") or ""
        pages = article.get("pages") or ""
        year = article.get("year") or ""

    venue = article.get("venue") or ""  # for conference papers

    # ── Detect document type ──
    doc_type = article.get("doc_type") or ""  # explicit override
    if not doc_type:
        jl = (journal or "").lower()
        if source in ("biorxiv", "medrxiv") or "biorxiv" in jl or "medrxiv" in jl:
            doc_type = "preprint"
        elif source == "arxiv":
            if venue:
                doc_type = "conference"
            else:
                doc_type = "preprint"
        elif venue:
            doc_type = "conference"
        else:
            doc_type = "journal"

    # ── Format by document type ──
    if doc_type == "preprint":
        platform = "bioRxiv" if "biorxiv" in (journal or "").lower() else \
                   "medRxiv" if "medrxiv" in (journal or "").lower() else \
                   f"arXiv:{article.get('arxiv_id', '')}" if source == "arxiv" else \
                   journal or "Preprint"
        ref = f"[{seq_num}] {author_str}. {title}[Z/OL]. {platform}, {year}"
        if doi:
            ref += f". DOI: {doi}"
        ref += "."

    elif doc_type == "conference":
        # [C]//Conference Name. City: Publisher, Year: Pages.
        # Simplified: if venue is known, use it; otherwise fall back
        ref = f"[{seq_num}] {author_str}. {title}[C]//{venue}"
        if year:
            ref += f", {year}"
        if pages:
            ref += f": {pages}"
        ref += "."
        if doi:
            ref += f" DOI: {doi}."

    elif doc_type == "book":
        publisher = article.get("publisher") or ""
        city = article.get("city") or ""
        ref = f"[{seq_num}] {author_str}. {title}[M]"
        if city or publisher:
            ref += f". {city}: {publisher}" if city else f". {publisher}"
        if year:
            ref += f", {year}"
        ref += "."

    elif doc_type == "dissertation":
        institution = article.get("institution") or ""
        city = article.get("city") or ""
        ref = f"[{seq_num}] {author_str}. {title}[D]"
        if city or institution:
            ref += f". {city}: {institution}" if city else f". {institution}"
        if year:
            ref += f", {year}"
        ref += "."

    else:
        # Default: [J] journal article
        ref = f"[{seq_num}] {author_str}. {title}[J]. {journal}, {year}"
        if volume:
            ref += f", {volume}"
        if issue:
            ref += f"({issue})"
        if pages:
            ref += f": {pages}"
        ref += "."
        if doi:
            ref += f" DOI: {doi}."

    return ref


def format_reference_list(articles):
    """Batch format articles into numbered GB/T 7714 reference list."""
    return "\n".join(format_gbt7714(a, i) for i, a in enumerate(articles, 1))
```

#### GB/T 7714 常见会议论文示例

写综述 §3 时，ReAct 等 ML 会议论文应标注为 [C]：

```
[N] Yao S, Zhao J, Yu D, et al. ReAct: synergizing reasoning and acting in language models[C]//International Conference on Learning Representations (ICLR), 2023.
[N] Wei J, Wang X, Schuurmans D, et al. Chain-of-thought prompting elicits reasoning in large language models[C]//Advances in Neural Information Processing Systems (NeurIPS), 2022.
```

要自动生成这种格式，需要在 article dict 中设置 `venue` 字段：
```python
article["venue"] = "International Conference on Learning Representations (ICLR)"
article["doc_type"] = "conference"
```

对于 arXiv 论文已被会议收录的情况（如 ReAct 先发 arXiv 后收录于 ICLR），应优先使用会议格式 [C]，而非预印本格式 [Z/OL]。Agent 在写综述时需要根据论文的最终发表状态判断使用哪种格式。

### 引用后处理

```python
import re

def process_citations(body, sources):
    """Post-process LLM-written text:
    1. Expand [1,2,3] → [1][2][3]
    2. Collect cited IDs in order of first appearance
    3. Remap to sequential numbering
    4. Remove phantom citations (IDs not in sources)
    """
    # Step 1: Expand multi-citations
    body = re.sub(
        r"\[(\d+(?:\s*,\s*\d+)+)\]",
        lambda m: "".join(f"[{n.strip()}]" for n in m.group(1).split(",")),
        body
    )

    # Step 2: Collect valid cited IDs in order of first appearance
    seen, ordered = set(), []
    for m in re.finditer(r"\[(\d+)\]", body):
        n = int(m.group(1))
        if n not in seen and 1 <= n <= len(sources):
            seen.add(n)
            ordered.append(n)

    # Step 3: Build remap and reference list
    remap = {old: new for new, old in enumerate(ordered, 1)}
    cited_sources = [sources[old - 1] for old in ordered]
    refs = [format_gbt7714(sources[old - 1], new) for old, new in remap.items()]

    # Step 4: Remap in body, remove phantoms
    def _remap(m):
        n = int(m.group(1))
        return f"[{remap[n]}]" if n in remap else ""
    body = re.sub(r"\[(\d+)\]", _remap, body)
    body = re.sub(r"  +", " ", body)

    return body, refs, cited_sources


def deduplicate(articles):
    """Deduplicate by DOI > PMID > normalized title."""
    seen = set()
    unique = []
    for a in articles:
        key = ((a.get("doi") or "").strip() or
               (a.get("pmid") or "").strip() or
               (a.get("title") or "").strip().lower())
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        unique.append(a)
    return unique
```

---

## GB/T 7714-2015 Quick Reference

| 文献类型 | 标识 | 格式模板 |
|---------|------|---------|
| 期刊论文 | [J] | `[N] 作者. 题名[J]. 刊名, 年, 卷(期): 页码. DOI: xxx.` |
| 预印本 | [Z/OL] | `[N] 作者. 题名[Z/OL]. 平台, 年. DOI: xxx.` |
| 会议论文 | [C] | `[N] 作者. 题名[C]//会议名, 年: 页码. DOI: xxx.` |
| 专著 | [M] | `[N] 作者. 书名[M]. 出版地: 出版社, 年.` |
| 学位论文 | [D] | `[N] 作者. 题名[D]. 所在地: 授予单位, 年.` |
| 网络资源 | [EB/OL] | `[N] 作者. 题名[EB/OL]. (发布日期). URL.` |

**规则**：
- 超过 3 位作者：前 3 位 + `, et al`
- 顺序编码制：正文 `[1][2][3]`，参考文献表按引用先后排列
- **刊名用全称**，全文统一（如 `Nature Medicine`，不用 `Nat Med`）
- arXiv 论文若已被会议正式收录，用 [C] 格式，否则用 [Z/OL]

---

## Typical Workflows

### "搜文献" — 关键词检索
1. MCP `pubmed_search_articles`（主力）
2. Python `search_arxiv()`（CS/AI 领域补充）
3. 合并 + `deduplicate()`
4. 展示表格

### "把这些 PMID 转成引用格式"
1. MCP `pubmed_fetch_contents` (pmids=[...], detailLevel="abstract_plus")
2. Python `format_gbt7714()` 逐条格式化
3. 输出 + 写文件

### "这篇论文的引用网络"
1. MCP `pubmed_article_connections` (similar) → 相似论文
2. MCP `pubmed_article_connections` (citedin) → 谁引了它
3. MCP `pubmed_article_connections` (references) → 它引了谁
4. 合并展示

### "bioRxiv 上关于 X 的预印本"
1. MCP `pubmed_search_articles` (queryTerm="X AND biorxiv[journal]")
2. 展示结果

---

## Dependency

All Python code uses ONLY stdlib (`urllib.request`, `xml.etree.ElementTree`, `re`, `json`). **Zero external dependencies**.
