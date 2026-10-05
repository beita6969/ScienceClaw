"""Tools for OCR post-correction of historical text units scored by cMER (HIPE-OCRepair-2026).

Metric (the scorer's own definitions)
norm(text) -> str                              the scorer's normalisation (lower-case, ß->ss, ligature and combining-e mappings, soft hyphen +
                                               newline removed, every non-word character -> space, whitespace collapsed)
char_counts(gold, hypothesis) -> (H, S, D, I)   jiwer character alignment counts of the normalised texts
cmer(gold, hypothesis) -> float                (S+D+I)/(H+S+D+I) of one pair;  edit_rate(a, b) -> float: (S+D+I)/(H+S+D) of b against a
official_weight(test_set) -> float             1/3 for test sets named dta19-l0/l1/l2, 1 for every other test set
score(test_sets, golds, hypotheses) -> dict    weighted cMER-micro of the task: 'weighted_cmer_micro', 'per_test_set', 'counts'
                                               (per test set the counts are summed over the units, the test-set values are then weighted)
Training-data statistics and text rules
noise_profile(train) -> {test_set: dict}       per test set of the ``load_train`` rows: n, ocr_cmer, unit_cmer_p50/p90/max, length_ratio_p05/p95
                                               (gold/OCR normalised length), line_hyphens (OCR lines ending in '-'), hyphen_join_gain
                                               (relative reduction of the summed edit count when ``join_line_hyphens`` is applied to the OCR
                                               text of these pairs)
join_line_hyphens(text) -> str                 an end-of-line '-' directly after a letter and before a lower-case line start becomes the
                                               soft hyphen '¬' (the normalisation then joins the two word parts); this is the convention the
                                               gold texts of the training pairs follow (¬, or an em dash in dta19)
Chunking, prompts, merging
split_chunks(text, max_chars) -> list[str]     contiguous, balanced pieces (''.join(pieces) == text) cut at line ends, sentence ends or spaces
build_prompt(chunk, language, test_set, examples, template) -> str     the prompt text of one llm item (see below)
plan(units, train, max_chars, skip_below, n_examples, join_hyphens, prompt, max_items) -> dict    chunks the units and builds the llm items
clean_reply(reply) -> str | None               a reply without code fence and without a leading 'Corrected text:' label
filter_hunks(original, reply, max_hunk_edits) -> (text, n_taken, n_kept_original)    word-level acceptance of the changes of a reply
merge(plan, outputs, max_edit_rate, len_band, max_hunk_edits, max_edit_rate_by_test_set) -> {'y': [str], 'report': dict}   guarded reassembly

``units`` are rows with ``ocr_text``, ``language``, ``test_set`` (``load_dev_inputs`` / ``load_eval_inputs`` rows; dev and evaluation rows
can be concatenated into one plan); ``train`` are the ``load_train`` rows. Plain Python for ``code`` nodes; the llm call is a separate
``llm`` node (one plan, one llm node, one merge):

    from scilib.ocrfix import plan, merge           # node 1 (code): inputs train, dev_items, items  ->  outputs plan, llm_items
    p = plan(dev_items + items, train)              # p['items']: list of dicts (prompt, chunk, instruction, uid, cid, language, test_set)
    return {"plan": p, "llm_items": p["items"]}     # llm node: items <- llm_items, prompt "{prompt}", config {"parse": "text"}
    # node 3 (code): inputs plan, outputs (the llm node's "outputs")
    r = merge(plan, outputs); y_all = r["y"]        # one string per unit, in the order given to plan(); r['report'] counts rejections
    return {"dev_pred": y_all[:len(dev)], "y": y_all[len(dev):]}

plan: a unit is cut into chunks of about ``max_chars`` (default 1100; chunk sizes from 600 characters to whole units were explored; an
llm reply is limited to 2000 tokens and one chunk is one llm item). ``max_items`` (default 128, the item limit of an llm node): if a plan
has more items, the chunk size is enlarged (up to 2200) until it fits; ``settings`` of the result holds the chunk size used and the item
count. Units of a test set whose training OCR cMER is below ``skip_below`` (default 0.008; None = never skip) get no llm item and keep their
OCR text; it only acts when ``train`` holds pairs of that test set. ``n_examples`` (default 4; 0 to 6 were explored) aligned OCR/gold
excerpts of the same test set, taken from the shortest ``train`` rows, are shown in the prompt. ``join_hyphens`` ('auto', True, False): apply
``join_line_hyphens`` to the final text of a test set; 'auto' = only where ``hyphen_join_gain`` > 0 and there are at least 10 end-of-line
hyphens in the training pairs of that test set. ``prompt`` = None for the built-in instruction (error kinds: wrongly recognised or missing
characters, spurious or missing spaces, accents/umlauts/long s; keep names, numbers, historical spelling, punctuation, line breaks; reply with
the text only) or your own template with the placeholders {chunk} {language} {test_set} {examples}.

merge: for every chunk the reply (cleaned by ``clean_reply``) is checked in three steps; an empty, missing or rejected reply keeps the OCR
chunk. (1) ``len_band`` (default (0.95, 1.05); None = off): len(norm(reply))/len(norm(OCR chunk)) inside the band. (2) ``max_edit_rate``
(default 0.15; None = off; ``max_edit_rate_by_test_set`` = {test_set: cap} overrides per test set): edit_rate(OCR chunk, reply) at most this
value. (3) ``max_hunk_edits`` (default 4; None = off): the reply is compared with the chunk word by word; a changed passage is taken if it
differs from the chunk by at most this many characters after ``norm`` (passages that differ only in whitespace or in the '-'/'¬' at a line
end are taken), otherwise the chunk's passage is kept. Of the changed passages between the OCR and gold texts of the training pairs, 57 % need
1 edit, 78 % at most 2, 87 % at most 3, 91 % at most 4, 96 % at most 6; values from 2 to 10 and None were explored. The unit text is the
sequence of accepted or kept chunks, then ``join_line_hyphens`` where enabled. ``report``: n_units, n_chunks, n_sent, n_rejected,
rejected_by_reason (empty / length / edit_rate), per_test_set {sent, rejected}, hunks_taken, hunks_kept_original, mean_edit_rate of the
accepted chunks, skipped_test_sets. The task's own hard constraint on y is a normalised length within [0.5, 2] x the OCR length of every
unit. Everything is deterministic and needs no files; ``norm``/``cmer`` agree with the scorer used by ``score_dev`` and ``text_mer``.
"""
from __future__ import annotations

import difflib
import math
import re
from collections import Counter
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = ["norm", "char_counts", "cmer", "edit_rate", "official_weight", "score", "noise_profile", "join_line_hyphens",
           "split_chunks", "filter_hunks", "build_prompt", "plan", "merge", "clean_reply"]

DEFAULT_MAX_CHARS = 1100
DEFAULT_SKIP_BELOW = 0.008
DEFAULT_MAX_EDIT_RATE = 0.15
DEFAULT_LEN_BAND = (0.95, 1.05)
DEFAULT_MAX_HUNK_EDITS = 4
DEFAULT_N_EXAMPLES = 4
MAX_LLM_ITEMS = 128   # items an llm node accepts
MAX_CHUNK_CHARS = 2200   # upper end of the chunk size range explored (an llm reply is limited to 2000 tokens)
MIN_HYPHENS = 10   # 'auto' hyphen joining needs at least this many end-of-line hyphens in the training pairs of the test set


# ----------------------------------------------------------------------------------------------- metric
_MAPS = (("ß", "ss"), ("ꝛ", "r"), ("œ", "oe"), ("æ", "ae"), ("aͤ", "ä"), ("oͤ", "ö"), ("uͤ", "ü"))


def norm(text: str) -> str:
    """The scorer's normalisation (see the module docstring)."""
    s = str(text).lower()
    for a, b in _MAPS:
        s = s.replace(a, b)
    s = s.replace("—\n", "").replace("¬\n", "")
    s = re.sub(r"[^\w]", " ", s, flags=re.UNICODE)
    s = re.sub(r"_", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def char_counts(gold: str, hypothesis: str) -> tuple[int, int, int, int]:
    """(hits, substitutions, deletions, insertions) of the character alignment of the normalised texts."""
    from jiwer import process_characters
    ref, hyp = norm(gold), norm(hypothesis)
    if not ref:
        raise ValueError("empty normalised gold text")
    o = process_characters(ref, hyp)
    return int(o.hits), int(o.substitutions), int(o.deletions), int(o.insertions)


def cmer(gold: str, hypothesis: str) -> float:
    """(S+D+I)/(H+S+D+I) of one text pair after the scorer's normalisation."""
    h, s, d, i = char_counts(gold, hypothesis)
    return (s + d + i) / max(1, h + s + d + i)


def edit_rate(a: str, b: str) -> float:
    """Normalised edit distance of ``b`` from ``a``: (S+D+I) / len(norm(a)); 0 for two texts with the same normal form."""
    na, nb = norm(a), norm(b)
    if na == nb:
        return 0.0
    if not na:
        return float(len(nb))
    from jiwer import process_characters
    o = process_characters(na, nb) if nb else None
    if o is None:
        return 1.0
    return (o.substitutions + o.deletions + o.insertions) / len(na)


def official_weight(test_set: str) -> float:
    return 1.0 / 3.0 if str(test_set).startswith("dta19-l") else 1.0


def score(test_sets: Sequence[str], golds: Sequence[str], hypotheses: Sequence[str]) -> dict:
    """Weighted cMER-micro over the test sets present (per-test-set sums of the alignment counts, then the weighted mean)."""
    if not (len(test_sets) == len(golds) == len(hypotheses)):
        raise ValueError("test_sets, golds and hypotheses must have the same length")
    per: dict[str, np.ndarray] = {}
    for ts, r, h in zip(test_sets, golds, hypotheses):
        per.setdefault(ts, np.zeros(4))
        per[ts] += np.asarray(char_counts(r, h), dtype=float)
    cm = {ts: float(v[1:].sum() / max(1.0, v.sum())) for ts, v in per.items()}
    w = {ts: official_weight(ts) for ts in cm}
    tot = sum(w.values())
    return {"weighted_cmer_micro": float(sum(w[t] * cm[t] for t in cm) / tot) if cm else float("nan"),
            "per_test_set": cm, "counts": {ts: [int(x) for x in v] for ts, v in per.items()}}


# ----------------------------------------------------------------------------------------------- layout rule
_LINE_HYPHEN = re.compile(r"(?<=[^\W\d_])-\n(?=[^\W\d_])")


def join_line_hyphens(text: str) -> str:
    """'-' + newline between a letter and a lower-case letter -> '¬' + newline (word parts joined by the normalisation)."""
    return _LINE_HYPHEN.sub(lambda m: "¬\n" if text[m.end():m.end() + 1].islower() else m.group(), text)


def _joined_gain(pairs: list[tuple[str, str]]) -> float:
    """Relative reduction of the summed edit count of all pairs when join_line_hyphens is applied to the OCR text."""
    ea = eb = 0
    for ocr, gold in pairs:
        c = char_counts(gold, ocr)
        ea += c[1] + c[2] + c[3]
        if "-\n" in ocr:
            j = char_counts(gold, join_line_hyphens(ocr))
            eb += j[1] + j[2] + j[3]
        else:
            eb += c[1] + c[2] + c[3]
    return float((ea - eb) / ea) if ea > 0 else 0.0


# ----------------------------------------------------------------------------------------------- profile
def noise_profile(train: Iterable[dict]) -> dict[str, dict]:
    """Per-test-set statistics of the labelled ``load_train`` rows (see the module docstring)."""
    by: dict[str, list[dict]] = {}
    for r in train:
        by.setdefault(str(r["test_set"]), []).append(r)
    out: dict[str, dict] = {}
    for ts, rows in sorted(by.items()):
        tot = np.zeros(4)
        unit, ratio, hy = [], [], 0
        pairs = []
        for r in rows:
            ocr, gold = r["ocr_text"], r["gt_text"]
            if not norm(gold) or not norm(ocr):
                continue
            c = char_counts(gold, ocr)
            tot += np.asarray(c, dtype=float)
            unit.append((c[1] + c[2] + c[3]) / max(1, sum(c)))
            ratio.append(len(norm(gold)) / max(1, len(norm(ocr))))
            hy += len(re.findall(r"-\n", ocr))
            pairs.append((ocr, gold))
        if not pairs:
            continue
        q = lambda v, p: float(np.percentile(v, p))
        out[ts] = {"n": len(pairs), "ocr_cmer": float(tot[1:].sum() / max(1.0, tot.sum())),
                   "unit_cmer_p50": q(unit, 50), "unit_cmer_p90": q(unit, 90), "unit_cmer_max": float(max(unit)),
                   "length_ratio_p05": q(ratio, 5), "length_ratio_p95": q(ratio, 95), "line_hyphens": hy,
                   "hyphen_join_gain": _joined_gain(pairs)}
    return out


# ----------------------------------------------------------------------------------------------- chunking
def _boundaries(text: str) -> list[tuple[int, int]]:
    """Candidate cut positions (index of the first character of the next piece, priority): 3 line end, 2 sentence end, 1 space."""
    out = []
    for m in re.finditer(r"\n+|(?<=[.!?;:])[ \t]+|[ \t]+", text):
        s = m.group()
        pri = 3 if "\n" in s else 2 if text[m.start() - 1:m.start()] in (".", "!", "?", ";", ":") else 1
        out.append((m.end(), pri))
    return out


def split_chunks(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Balanced contiguous pieces of at most about ``max_chars`` characters, cut at the best boundary near each ideal position."""
    n = len(text)
    max_chars = max(50, int(max_chars))
    if n <= max_chars:
        return [text]
    k = math.ceil(n / max_chars)
    target = n / k
    cands = [(p, pri) for p, pri in _boundaries(text) if 0 < p < n]
    cuts, prev = [], 0
    for i in range(1, k):
        ideal = int(round(i * target))
        lo, hi = max(prev + 1, ideal - int(0.3 * target)), min(n - 1, ideal + int(0.3 * target))
        window = [(p, pri) for p, pri in cands if lo <= p <= hi]
        if window:
            p = max(window, key=lambda t: (t[1], -abs(t[0] - ideal)))[0]
        else:
            p = min(max(prev + 1, ideal), n - 1)
        cuts.append(p)
        prev = p
    pieces, a = [], 0
    for p in cuts + [n]:
        if p > a:
            pieces.append(text[a:p])
            a = p
    return pieces


# ----------------------------------------------------------------------------------------------- prompt
_LANG = {"en": "English", "fr": "French", "de": "German"}
_HINT = {
    "en": "Long s (ſ) is often read as f and vice versa.",
    "fr": "Restore accents and cedillas (é è ê à ç ...) wherever the word requires them; 'ä' for 'à' is a typical misreading.",
    "de": "Umlauts (ä ö ü) and ß are often lost or wrongly added; long s (ſ) is often read as f; keep combining small e above a vowel as it is.",
}
INSTRUCTION = (
    "You correct OCR errors in a historical {lang} text (newspaper or book, 17th-20th century, corpus {test_set}). "
    "Reply with the corrected text only, with the same line breaks as the input.\n"
    "Kinds of errors to fix: wrongly recognised characters (e.g. h for b, n for u, c for e, li for h, rn for m); "
    "missing or superfluous characters; a space wrongly inserted inside a word (rejoin the word) or missing between two words (separate them). "
    "{hint}\n"
    "Rules: change only what is clearly an OCR error. Keep the historical spelling and archaic word forms, names, numbers, abbreviations, "
    "punctuation and capitalisation as they are unless they are clearly misread. Do not translate, modernise, summarise, reorder, add or "
    "delete words. Keep a hyphen at the end of a line where it is. If a passage is unreadable or ambiguous, copy it unchanged.\n"
)


def build_prompt(chunk: str, language: str = "en", test_set: str = "", examples: Sequence[tuple[str, str]] = (),
                 template: str | None = None) -> str:
    """Instruction + optional (ocr, corrected) example excerpts + the chunk to correct."""
    lang = _LANG.get(str(language)[:2], str(language))
    ex = ""
    if examples:
        ex = "Examples of corrections in this corpus:\n" + "\n".join(f"OCR: {a}\nCorrected: {b}\n" for a, b in examples)
    if template is not None:
        return template.format(chunk=chunk, language=lang, test_set=test_set, examples=ex)
    head = INSTRUCTION.format(lang=lang, test_set=test_set or "unknown", hint=_HINT.get(str(language)[:2], ""))
    return f"{head}{ex}\nOCR text to correct:\n{chunk}\n\nCorrected text:"


def _aligned_excerpts(ocr: str, gold: str, k: int, target_chars: int = 220) -> list[tuple[str, str]]:
    """Up to k short (ocr, gold) excerpts with a few differences, found by a word-level alignment (single-line text)."""
    if "\n" in ocr or "\n" in gold:
        ol, gl = ocr.split("\n"), gold.split("\n")
        if len(ol) == len(gl):
            pick = [(a, b) for a, b in zip(ol, gl) if a != b and 40 <= len(a) <= 160 and edit_rate(b, a) <= 0.15]
            return pick[:k]
    ow, gw = ocr.split(" "), gold.split(" ")
    sm = difflib.SequenceMatcher(None, ow, gw, autojunk=False)
    ops = sm.get_opcodes()
    out, i = [], 0
    while i < len(ops) and len(out) < k:
        tag, i1, i2, j1, j2 = ops[i]
        if tag == "equal":
            i += 1
            continue
        a0, b0 = max(0, i1 - 6), max(0, j1 - 6)
        a1, b1 = i2, j2
        j = i + 1
        while j < len(ops) and len(" ".join(ow[a0:a1])) < target_chars:
            a1, b1 = ops[j][2], ops[j][4]
            j += 1
        a1, b1 = min(len(ow), a1 + 4), min(len(gw), b1 + 4)
        o, g = " ".join(ow[a0:a1]), " ".join(gw[b0:b1])
        if 40 <= len(o) <= 400 and edit_rate(g, o) <= 0.15:
            out.append((o, g))
        i = j
    return out


def _examples_by_test_set(train: Iterable[dict], n_examples: int) -> dict[str, list[tuple[str, str]]]:
    out: dict[str, list[tuple[str, str]]] = {}
    if n_examples <= 0:
        return out
    by: dict[str, list[dict]] = {}
    for r in train or []:
        by.setdefault(str(r["test_set"]), []).append(r)
    for ts, rows in by.items():
        rows = sorted(rows, key=lambda r: (len(r["ocr_text"]), r["ocr_text"]))
        picked: list[tuple[str, str]] = []
        for r in rows:
            for ex in _aligned_excerpts(r["ocr_text"], r["gt_text"], 1):
                picked.append(ex)
            if len(picked) >= n_examples:
                break
        out[ts] = picked[:n_examples]
    return out


# ----------------------------------------------------------------------------------------------- plan / merge
def _build_items(units: Sequence[dict], prof: dict, ex: dict, max_chars: int, skip_below: float | None, join_hyphens: Any,
                 prompt: str | None) -> tuple[list[dict], list[dict]]:
    items, unit_rows = [], []
    for u, unit in enumerate(units):
        text, ts = str(unit["ocr_text"]), str(unit.get("test_set", ""))
        lang = str(unit.get("language", "en"))
        chunks = split_chunks(text, max_chars)
        p = prof.get(ts)
        skipped = bool(skip_below is not None and p is not None and p["ocr_cmer"] < float(skip_below))
        if join_hyphens == "auto":
            join = bool(p is not None and p["hyphen_join_gain"] > 0 and p["line_hyphens"] >= MIN_HYPHENS)
        else:
            join = bool(join_hyphens)
        unit_rows.append({"test_set": ts, "language": lang, "chunks": chunks, "skipped": skipped, "join": join})
        if skipped:
            continue
        instr = "" if prompt is not None else \
            build_prompt("", lang, ts, ex.get(ts, ()), template=None).split("OCR text to correct:")[0].rstrip()
        for c, chunk in enumerate(chunks):
            if not norm(chunk):
                continue
            items.append({"uid": u, "cid": c, "language": lang, "test_set": ts, "chunk": chunk, "instruction": instr,
                          "prompt": build_prompt(chunk, lang, ts, ex.get(ts, ()), template=prompt)})
    return items, unit_rows


def plan(units: Sequence[dict], train: Iterable[dict] | None = None, max_chars: int = DEFAULT_MAX_CHARS,
         skip_below: float | None = DEFAULT_SKIP_BELOW, n_examples: int = DEFAULT_N_EXAMPLES, join_hyphens: Any = "auto",
         prompt: str | None = None, max_items: int = MAX_LLM_ITEMS) -> dict:
    """Chunk the units and build the llm items (see the module docstring). The returned dict is plain data."""
    train = list(train or [])
    prof = noise_profile(train) if train else {}
    ex = _examples_by_test_set(train, int(n_examples))
    size = int(max_chars)
    items, unit_rows = _build_items(units, prof, ex, size, skip_below, join_hyphens, prompt)
    while len(items) > int(max_items) and size < MAX_CHUNK_CHARS:      # too many items for one llm node: larger chunks
        size = min(MAX_CHUNK_CHARS, int(size * 1.25) + 1)
        items, unit_rows = _build_items(units, prof, ex, size, skip_below, join_hyphens, prompt)
    return {"items": items, "units": unit_rows,
            "settings": {"max_chars": size, "skip_below": skip_below, "n_examples": int(n_examples), "n_items": len(items)},
            "skipped_test_sets": sorted({r["test_set"] for r in unit_rows if r["skipped"]})}


def _norm_edits(a: str, b: str) -> int:
    """Number of character edits (S+D+I) between the normalised forms of two texts."""
    na, nb = norm(a), norm(b)
    if na == nb:
        return 0
    if not na or not nb:
        return len(na) + len(nb)
    from jiwer import process_characters
    o = process_characters(na, nb)
    return int(o.substitutions + o.deletions + o.insertions)


def filter_hunks(original: str, reply: str, max_hunk_edits: int | None = DEFAULT_MAX_HUNK_EDITS) -> tuple[str, int, int]:
    """Word-level diff of ``reply`` against ``original``: a changed passage is taken from the reply if it differs from the original
    by at most ``max_hunk_edits`` characters after normalisation (changes without effect on the normalised text are taken too);
    otherwise the original passage is kept. Returns (text, n_taken, n_kept_original); ``None`` takes the reply as it is."""
    if max_hunk_edits is None:
        return reply, 0, 0
    a = re.findall(r"\s+|\S+", original)
    b = re.findall(r"\s+|\S+", reply)
    out, taken, kept = [], 0, 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        old, new = "".join(a[i1:i2]), "".join(b[j1:j2])
        if tag == "equal":
            out.append(old)
            continue
        e = _norm_edits(old, new)
        if e == 0 and norm(old) == norm(new):
            out.append(new)
        elif e <= max_hunk_edits:
            out.append(new)
            taken += 1
        else:
            out.append(old)
            kept += 1
    return "".join(out), taken, kept


_FENCE = re.compile(r"^```[A-Za-z0-9_+-]*[ \t]*\n(.*?)\n?```\s*$", re.S)
_LABEL = re.compile(r"^\s*(?:corrected(?: text)?|correction|output)\s*:\s*\n?", re.I)


def clean_reply(reply: Any) -> str | None:
    """The reply text without a surrounding code fence or a leading 'Corrected text:' label; None for a non-string."""
    if not isinstance(reply, str):
        return None
    t = reply.strip()
    m = _FENCE.match(t)
    if m:
        t = m.group(1).strip()
    t = _LABEL.sub("", t, count=1).strip()
    return t


def merge(plan_: dict, outputs: Sequence[Any], max_edit_rate: float | None = DEFAULT_MAX_EDIT_RATE,
          len_band: tuple[float, float] | None = DEFAULT_LEN_BAND, max_hunk_edits: int | None = DEFAULT_MAX_HUNK_EDITS,
          max_edit_rate_by_test_set: dict | None = None) -> dict:
    """Guarded reassembly of the unit texts from the llm replies (see the module docstring)."""
    items = plan_["items"]
    if len(outputs) != len(items):
        raise ValueError(f"expected {len(items)} llm outputs (one per plan item), got {len(outputs)}")
    lo, hi = (float(len_band[0]), float(len_band[1])) if len_band is not None else (0.0, 0.0)
    caps = dict(max_edit_rate_by_test_set or {})
    chosen: dict[tuple[int, int], str] = {}
    rej = Counter()
    per_ts: dict[str, dict[str, int]] = {}
    rates = []
    hunks = Counter()
    for it, out in zip(items, outputs):
        ts = it["test_set"]
        cap = caps.get(ts, max_edit_rate)
        st = per_ts.setdefault(ts, {"sent": 0, "rejected": 0})
        st["sent"] += 1
        rep = clean_reply(out)
        chunk = it["chunk"]
        why = None
        if not rep or not norm(rep):
            why = "empty"
        else:
            r = len(norm(rep)) / max(1, len(norm(chunk)))
            if len_band is not None and not (lo <= r <= hi):
                why = "length"
            else:
                er = edit_rate(chunk, rep)
                if cap is not None and er > cap:
                    why = "edit_rate"
                else:
                    rates.append(er)
        if why:
            rej[why] += 1
            st["rejected"] += 1
            continue
        rep, taken, kept = filter_hunks(chunk.strip(), rep, max_hunk_edits)
        hunks["taken"] += taken
        hunks["kept_original"] += kept
        # keep the separator whitespace of the original chunk so that the pieces concatenate as before
        lead = chunk[:len(chunk) - len(chunk.lstrip())]
        trail = chunk[len(chunk.rstrip()):]
        chosen[(it["uid"], it["cid"])] = lead + rep + trail
    y = []
    for u, row in enumerate(plan_["units"]):
        text = "".join(chosen.get((u, c), chunk) for c, chunk in enumerate(row["chunks"]))
        y.append(join_line_hyphens(text) if row["join"] else text)
    report = {"n_units": len(y), "n_chunks": sum(len(r["chunks"]) for r in plan_["units"]), "n_sent": len(items),
              "n_rejected": int(sum(rej.values())), "rejected_by_reason": dict(rej), "per_test_set": per_ts,
              "hunks_taken": hunks["taken"], "hunks_kept_original": hunks["kept_original"],
              "mean_edit_rate": float(np.mean(rates)) if rates else 0.0,
              "skipped_test_sets": list(plan_.get("skipped_test_sets", []))}
    return {"y": y, "report": report}
