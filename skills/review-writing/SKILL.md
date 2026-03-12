---
name: review-writing
description: # Review Writing — 学术综述逐节写作方法论
---

# Review Writing — 学术综述逐节写作方法论

Use this skill when the user asks to write a literature review, review article, or 综述 based on an outline. Trigger keywords: "写综述", "write review", "综述写作", "按大纲写", "逐节写", "review section", "写第N节". This skill orchestrates the ENTIRE review writing process from outline to finished manuscript.

**This skill calls `academic-literature-search` skill for all search and citation operations. Read that skill first if not already loaded.**

**Tool routing**: PubMed operations → MCP tools (`pubmed_search_articles`, `pubmed_fetch_contents`, `pubmed_article_connections`). arXiv search, GB/T 7714 formatting, citation processing → Python code. See `academic-literature-search` for the complete routing table and code templates.

---

## Architecture: Why Section-by-Section

A full review (12,000–15,000 words, 100–130 references) CANNOT be written in one pass due to context window limits. The correct approach:

```
Outline
  → [Phase 0: Validate & Revise outline]
  → [Phase 1..N: Per-section pipeline]
  → [Final: Assemble full review]

Each section is an independent unit of work:
  Search → Filter → Group → Write → Cite → Save to file
                                              ↓
                                    section_N.md (persisted immediately)

Final assembly reads all section files → cross-section dedup → unified numbering
```

---

## Phase 0: Outline Validation & Revision (大纲验证与修订)

**DO NOT skip this phase.** No outline is perfect before reading the literature.

### Step 0.1: Read the outline

Read the user's outline file. Parse each section's title, sub-topics, and any pre-identified references.

### Step 0.2: Scout search (侦察检索)

For each section, run ONE quick search using the section title/topic as query:

- **MCP** `pubmed_search_articles` (maxResults=10, fetchBriefSummaries=5)
- For CS/AI-heavy sections: also **Python** `search_arxiv()` (max_results=5)
- For sections with known seed papers: **MCP** `pubmed_article_connections` (similar, maxRelatedResults=5)

### Step 0.3: Evaluate and suggest revisions

Based on scout results, produce a validation report:

```markdown
## 大纲验证报告

### §1 [section title]
- 检索命中: PubMed X篇, arXiv Y篇
- 代表性论文:
  - "Paper Title A" (Journal, Year) — [关系:直接相关]
  - "Paper Title B" (Journal, Year) — [关系:方法论参考]
  - "Paper Title C" (Journal, Year) — [关系:最新进展]
- 评估: ✅ 文献充足 / ⚠️ 偏少建议扩展 / 🔴 极少建议合并或调整
- 建议: [specific suggestion if any]

### §2 ...

### 整体建议
- 建议新增: [topic] — 检索发现大量文献但大纲未覆盖
- 建议合并: §X 和 §Y 文献高度重叠
- 建议拆分: §Z 文献过于丰富，建议拆为两节
```

### Step 0.4: User confirms revised outline

Wait for user to confirm or further adjust. **Only proceed to Phase 1 after outline is finalized.**

---

## Pre-Writing: Thesis Reference Ingestion (正文引用复用)

If the user's thesis body already has references (like the `论文正文——第一二部分合并.md`), BEFORE starting Phase 1:

1. Extract all references from the thesis body (PMIDs, DOIs, author-year citations)
2. Fetch their full metadata via **MCP** `pubmed_fetch_contents`
3. Store as a **seed reference pool** — when the same paper appears in review search results, reuse this metadata exactly (ensures consistency between thesis body and review)
4. When writing review sections, if a thesis-body reference is relevant, cite it directly from the seed pool without re-searching

---

## Phase 1–N: Per-Section Writing (逐节写作)

For each section, execute the full pipeline below. **One section per conversation turn.**

### Step 1: Deep Search (深度检索)

Generate 3-5 targeted search queries based on the section's sub-topics. Then:

**For biomedical-heavy sections (§1, §2, §5, §7):**
1. **MCP** `pubmed_search_articles` (maxResults=15, dateRange minDate="2020", fetchBriefSummaries=10)
2. **MCP** `pubmed_search_articles` (queryTerm="X AND biorxiv[journal]", maxResults=5) — 预印本

**For CS/AI-heavy sections (§3, §4, §6):**
1. **MCP** `pubmed_search_articles` (maxResults=10)
2. **Python** `search_arxiv(query, max_results=10)` — ML/AI 会议论文和预印本

**For all sections:**
3. Deduplicate across all queries: Python `deduplicate()`
4. For foundational papers in the outline: **MCP** `pubmed_fetch_contents` (pmids=[...])
5. For expanding from seed papers: **MCP** `pubmed_article_connections` (similar / citedin / references)
6. Check seed reference pool for any already-known relevant papers

**Target candidates per section:**
- Biomedical sections: 15–25 papers
- CS/AI sections: 20–30 papers (wider net because PubMed coverage is sparser)

### Step 2: Filter & Score (筛选评估)

Present search results to the LLM with this instruction:

```
From the following N search results, select the most relevant papers
for writing a review section about [section topic].

SELECTION TARGETS:
- Biomedical sections: select 10-15 papers
- CS/AI sections: select 15-20 papers

SELECTION CRITERIA (in priority order):
1. RELEVANCE to the specific section topic
2. IMPACT: prefer high-impact journals and highly-cited works
3. RECENCY: prefer 2022-2026, but include seminal older papers
4. DIVERSITY: cover different sub-aspects, not just the most popular finding
5. BALANCE: include both supporting evidence and contrasting viewpoints

For each selected paper, assign ONE role:
- FOUNDATIONAL: establishes the field/concept
- EVIDENCE: provides key experimental/computational evidence
- METHOD: introduces an important method/tool
- COMPARISON: enables comparison between approaches
- GAP: identifies limitations or open problems
- MILESTONE: landmark paper (e.g., AlphaFold, GPT-4)

Output format:
[search_index] [ROLE] — one-sentence reason for inclusion
```

### Step 3: Evidence Grouping (证据分组)

Organize selected papers into narrative groups that will drive paragraph structure:

```
Group A: "Historical development / Milestones"
Group B: "Current mainstream approaches / Consensus"
Group C: "Emerging advances / Recent breakthroughs"
Group D: "Methodological comparisons"
Group E: "Limitations, controversies, and open problems"
```

Not every section needs all groups. Choose the groups that fit the section's content.

### Step 4: Write Section (撰写本节)

```
Write a review section about [section topic] using the provided literature.

WRITING RULES:
1. ACADEMIC CHINESE PROSE (学术中文). Scientific terms keep English abbreviations
   on first mention: e.g., 检索增强生成（Retrieval-Augmented Generation, RAG）.
   Subsequent uses can use abbreviation directly.

2. NARRATIVE, NOT LIST. Write flowing paragraphs with logical transitions.
   ❌ "A研究了X[1]。B研究了Y[2]。C研究了Z[3]。"
   ✅ "多项研究从不同角度探讨了这一问题。A等[1]首先通过...揭示了...；
      在此基础上，B等[2]进一步...；然而，C等[3]的研究指出..."

3. CITE BY INDEX. Reference papers using [N] where N is the paper's index in
   the provided source list. Every factual claim MUST have at least one citation.

4. CRITICAL ANALYSIS. Don't just summarize — compare, contrast, evaluate.
   Point out methodological differences, conflicting findings, remaining gaps.
   ❌ "取得了重要进展"、"具有广阔前景"
   ✅ "将检索精度从 70.1% 提升至 80.7%"、"覆盖了 338 个数据库"

5. SECTION STRUCTURE:
   a. 开门点题（1-2句）：本节综述什么主题，为什么重要
   b. 发展脉络（2-3段）：按时间或逻辑组织
   c. 现状分析（1-2段）：主流方法/共识/争议
   d. 批判性评价（1段）：现有工作的局限和不足
   e. 收束引出（1-2句）：指向下一节或研究空白

6. SECTION TRANSITION: The FIRST sentence of this section must logically connect
   to the LAST sentence of the previous section. The LAST sentence must set up
   the next section's topic. [Agent: verify this after writing.]

7. LENGTH: 1,500-2,500 Chinese characters per section.

8. NO FABRICATION. Only cite papers from the provided source list.
   If a fact lacks source support, write "据报道" without citation rather than
   fabricating one. NEVER invent PMIDs, DOIs, or author names.

9. JOURNAL NAMES: Use FULL journal names (Nature Medicine, not Nat Med).
   Keep this consistent across all sections.

SOURCE LIST:
[paste filtered papers with index, title, authors, year, abstract]
```

### Step 5: Post-Write Checks (写后检查)

After the LLM writes the section, perform these checks:

**5a. Citation integration** (code):
- `process_citations()` from `academic-literature-search` skill
- Expand multi-citations, remove phantoms, record actually-cited papers

**5b. Section transition check** (LLM):
- Read the last 2 sentences of the PREVIOUS section file
- Read the first 2 sentences of the current section
- Verify logical connection. If disconnected, suggest revision.

**5c. Citation density check** (code):
```python
# Count paragraphs and citations
paragraphs = [p for p in section_text.split("\n\n") if p.strip()]
for i, p in enumerate(paragraphs):
    cite_count = len(re.findall(r"\[\d+\]", p))
    if cite_count == 0 and len(p) > 100:
        print(f"WARNING: Paragraph {i+1} has no citations ({len(p)} chars)")
```

**5d. Format reference list** (code):
- `format_gbt7714()` for each cited paper
- Verify journal names are full names, not abbreviations

### Step 6: Save to File (保存)

```
output_dir/
  section_1_[short_name].md
  section_2_[short_name].md
  ...
  section_N_[short_name].md
  _metadata.json
```

Each section file:

```markdown
# [Section Number] [Section Title]

[Section text with [N] citations]

---
## 本节参考文献（临时编号）

[1] Author, et al. Title[J]. Journal, Year, Vol(Issue): Pages. DOI: xxx.
[2] ...

---
<!-- metadata
section_index: 1
cited_papers: [
  {"local_index": 1, "pmid": "12345678", "doi": "10.1234/xxx", "title": "...", "source": "pubmed"},
  {"local_index": 2, "pmid": "", "doi": "", "arxiv_id": "2210.03629", "title": "...", "source": "arxiv"},
  ...
]
search_queries: ["query1", "query2", ...]
candidate_count: 25
cited_count: 14
-->
```

`_metadata.json` tracks cross-section state:

```json
{
  "outline_file": "/path/to/综述大纲.md",
  "output_dir": "/path/to/综述输出/",
  "sections_completed": [1, 2, 3],
  "sections_total": 8,
  "all_cited_papers": [
    {"pmid": "12345678", "doi": "...", "title": "...", "first_cited_in_section": 1},
    ...
  ],
  "total_unique_references": 45,
  "seed_reference_pool": [...],
  "last_updated": "2026-02-27T20:30:00"
}
```

---

## Final Phase: Assembly (全文组装)

### Step F1: Read all section files

### Step F2: Cross-section deduplication

Same paper cited in §2 and §5 → ONE reference number. Match by DOI > PMID > normalized title.

### Step F3: Unified sequential numbering

**MUST be done by code**, scanning sections in order:

```python
import re, json

def assemble_review(section_files, output_path):
    """Assemble all sections into final review with unified GB/T 7714 numbering."""
    global_refs = []
    paper_to_global = {}
    global_num = 1
    full_text_parts = []

    for sf in section_files:
        with open(sf) as f:
            content = f.read()

        # Split text from metadata
        text_part = content.split("---\n## 本节参考文献")[0]
        # Load cited papers from metadata comment
        meta_match = re.search(r'<!-- metadata\n(.*?)\n-->', content, re.DOTALL)
        local_papers = []
        if meta_match:
            meta_text = meta_match.group(1)
            cp_match = re.search(r'cited_papers: (\[.*?\])', meta_text, re.DOTALL)
            if cp_match:
                local_papers = json.loads(cp_match.group(1))

        if not local_papers:
            full_text_parts.append(text_part)
            continue

        paper_by_local = {p["local_index"]: p for p in local_papers}

        def remap(m):
            nonlocal global_num
            local_idx = int(m.group(1))
            paper = paper_by_local.get(local_idx)
            if not paper:
                return ""
            key = (paper.get("doi") or paper.get("pmid") or
                   paper.get("title", "").lower())
            if key not in paper_to_global:
                paper_to_global[key] = global_num
                global_refs.append(paper)
                global_num += 1
            return f"[{paper_to_global[key]}]"

        remapped = re.sub(r"\[(\d+)\]", remap, text_part)
        full_text_parts.append(remapped)

    # Build final GB/T 7714 reference list
    from academic_literature_search import format_gbt7714  # conceptual import
    ref_lines = [format_gbt7714(p, i) for i, p in enumerate(global_refs, 1)]

    full_review = "\n\n".join(full_text_parts)
    full_review += "\n\n---\n\n# 参考文献\n\n" + "\n".join(ref_lines)

    with open(output_path, "w") as f:
        f.write(full_review)

    return len(global_refs)
```

### Step F4: Quality Check (质量自检)

Run automated checks and produce a report:

```
质量自检报告
─────────────────────────────────
总节数: N
总唯一参考文献: M
每节平均引用: M/N = X.X (目标: 12-18)
─────────────────────────────────
引用覆盖率: Y% 段落有至少1个引用
最长无引用段: Z 字符 (目标: <500)
─────────────────────────────────
来源分布:
  PubMed 期刊论文 [J]: X篇 (XX%)
  预印本 [Z/OL]: Y篇 (YY%)
  会议论文 [C]: Z篇 (ZZ%)
─────────────────────────────────
年份分布:
  2024-2026: X篇 (XX%)
  2021-2023: Y篇 (YY%)
  2020及以前: Z篇 (ZZ%)
─────────────────────────────────
跨节引用复用: X篇被多节引用
无引用的节: [列表, 应为空]
─────────────────────────────────
节间衔接:
  §1→§2: ✅ / ⚠️ [具体问题]
  §2→§3: ✅ / ⚠️
  ...
─────────────────────────────────
与论文正文引用一致性:
  正文引用在综述中也出现: X/Y篇
  建议补引的正文参考文献: [列表]
```

### Step F5: Write final files

```
output_dir/
  综述_final.md           ← 统一编号的完整综述
  参考文献_final.md       ← 独立的 GB/T 7714 参考文献列表
  quality_report.md       ← 质量自检报告
```

---

## Interaction Protocol

### Starting

User: "按大纲写综述" / "写综述"
1. Read outline file
2. Say: "我先做一轮侦察检索来验证大纲，然后给你修订建议。确认开始？"
3. Execute Phase 0
4. Present validation report (with sample paper titles)
5. Wait for user to confirm

### Per-section

1. "大纲已确认。现在逐节写作，先从第1节开始？"
2. Execute full pipeline for §1
3. Show written section + local references + check results
4. "第1节写完了。需要修改还是继续第2节？"

### User commands (anytime)

| 用户说 | Agent 做 |
|--------|---------|
| "这节重写" | 重新执行当前节的 Step 1-6 |
| "多找几篇关于X的文献" | 追加检索，合入候选池 |
| "这篇一定要引：PMID/DOI" | MCP fetch → 强制纳入当前节 |
| "大纲要改" | 回到 Phase 0，仅重新验证受影响的节 |
| "跳过这节" | 标记 skipped，继续下一节 |
| "组装全文" | 跳到 Final Phase |
| "检查质量" | 对已完成的节运行 Step F4 质量自检 |

---

## Writing Quality Standards (协和博士论文级别)

### Language

- 学术中文，措辞严谨，避免口语化
- 英文术语首次出现标注中文翻译和英文缩写：检索增强生成（Retrieval-Augmented Generation, RAG）
- 后续直接使用缩写
- **刊名全称**，全文统一：`Nature Medicine` 不用 `Nat Med`

### Narrative Structure (per section)

1. **开门点题**（1-2句）：本节综述什么，为什么重要
2. **发展脉络**（2-3段）：按时间或逻辑组织
3. **现状分析**（1-2段）：主流方法/共识/争议
4. **批判性评价**（1段）：局限和不足
5. **收束引出**（1-2句）：引出下一节或研究空白

### Citation Density

- 每个实质性段落至少 2-3 个引用
- 关键结论/数据/数字必须有引用
- 连续超过 3 句无引用 → 检查是否缺引
- 目标：每节 12-18 篇（生物节）/ 15-20 篇（CS/AI 节）

### Forbidden Patterns

- ❌ 罗列式："A研究了X[1]。B研究了Y[2]。C研究了Z[3]。"
- ✅ 叙事式："多项研究从不同角度探讨了这一问题。A等[1]首先通过...揭示了..."
- ❌ 空泛评价："取得了重要进展"、"具有广阔前景"
- ✅ 具体评价："将检索精度从 70.1% 提升至 80.7%"、"覆盖了 338 个数据库"
- ❌ 刊名缩写（任何地方）
- ❌ 在未确认发表状态下将 arXiv 论文标为 [J]

### Conference Paper Handling (§3, §4 重要)

Many key papers in AI/ML sections (ReAct, CoT, Reflexion, etc.) are published at conferences, not journals.

- If the paper is published at a conference (ICLR, NeurIPS, ICML, ACL, etc.): use `[C]` format
- If still only on arXiv without conference acceptance: use `[Z/OL]` format
- Agent MUST check: does this arXiv paper have a published venue? If yes, use `[C]`.

Common venues to check:
- ICLR, NeurIPS, ICML (machine learning)
- ACL, EMNLP, NAACL (NLP)
- SIGIR, CIKM (information retrieval)
- KDD, WWW (data mining / web)
- AAAI, IJCAI (general AI)

---

## Recovery Protocol

If conversation is interrupted:

1. Check for `_metadata.json` in output directory
2. If exists: read it, report which sections are done, offer to continue from next section
3. If not: start fresh from Phase 0

Every section is saved to file immediately — no work is lost on interruption.
