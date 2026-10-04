"""scilib.ocrfix: metric parity with the adapter, chunking, prompt plan, guarded merge, noise profile, sandbox import."""
from __future__ import annotations

import pytest

import scilib
from scilib import ocrfix as ox
from scienceclaw.bench.tasks import for43_hipe as m
from scienceclaw.runtime.integrity import scan_code

OCR = ("The Mothei s Dav appeal oiganised by the\nlocal committee was held in the Town\nHall on Mon-\nday last, and the follow ing\n"
       "morning the results were published.")
GOLD = ("The Mother's Day appeal organised by the\nlocal committee was held in the Town\nHall on Mon¬\nday last, and the following\n"
        "morning the results were published.")


def _units():
    return [{"ocr_text": OCR, "language": "en", "test_set": "impresso-snippets/en"},
            {"ocr_text": "Der Käse iſt gut.\nEs war ſchön.", "language": "de", "test_set": "dta19-l1/de"}]


def _train(n: int = 12, test_set: str = "impresso-snippets/en"):
    return [{"ocr_text": OCR + f" Item {i}.", "gt_text": GOLD + f" Item {i}.", "test_set": test_set, "language": "en"} for i in range(n)]


def test_metric_matches_adapter():
    pairs = [(GOLD, OCR), ("Straße Œuvre ﬁn ß", "Strasse Oeuvre fin ss"), ("abc¬\ndef", "abcdef"), ("x y z", "x yz"), ("Ab—\ncd", "abcd"),
             ("daß aͤ oͤ uͤ", "dass ä ö ü")]
    for ref, hyp in pairs:
        assert ox.norm(ref) == m.hipe_norm(ref)
        assert ox.char_counts(ref, hyp) == m.char_counts(ref, hyp)
    ts = ["dta19-l1/de", "dta19-l2/de", "icdar2017/en", "icdar2017/en"]
    refs = ["Das ist gut.", "Es war ſo.", "the old man", "and so on"]
    hyps = ["Das ist gnt.", "Es war so.", "the oid man", "and so on"]
    s = ox.score(ts, refs, hyps)
    counts = [m.char_counts(r, h) for r, h in zip(refs, hyps)]
    ref_score, per = m.weighted_cmer_micro(counts, ts)
    assert s["weighted_cmer_micro"] == pytest.approx(ref_score)
    assert s["per_test_set"] == pytest.approx(per)
    assert ox.official_weight("dta19-l0/de") == pytest.approx(1 / 3) and ox.official_weight("icdar2017/fr") == 1.0
    assert ox.cmer(GOLD, GOLD) == 0.0 and ox.cmer(GOLD, OCR) > 0.0


def test_empty_hypothesis_and_edit_rate():
    assert ox.char_counts("abc", "") == (0, 0, 3, 0)
    assert ox.edit_rate("abc", "") == 1.0
    assert ox.edit_rate("abc def", "abc  def") == 0.0
    assert ox.edit_rate("abcd", "abce") == pytest.approx(0.25)
    assert ox.edit_rate("", "ab") == 2.0


def test_join_line_hyphens_only_letter_hyphen_newline_lowercase():
    assert ox.join_line_hyphens("Mon-\nday") == "Mon¬\nday"
    assert ox.join_line_hyphens("Ost-\nDeutschland") == "Ost-\nDeutschland"      # capital: compound, stays
    assert ox.join_line_hyphens("1914-\n1918") == "1914-\n1918"
    assert ox.join_line_hyphens("a - \nb") == "a - \nb"
    assert ox.norm(ox.join_line_hyphens("Mon-\nday")) == "monday"
    assert ox.cmer(GOLD, ox.join_line_hyphens(OCR)) < ox.cmer(GOLD, OCR)


def test_split_chunks_reassemble_and_respect_size():
    text = "\n".join(f"line number {i} with some words in it." for i in range(120))
    for mc in (200, 1100, 5000):
        parts = ox.split_chunks(text, mc)
        assert "".join(parts) == text
        assert all(len(p) <= 1.35 * mc for p in parts)
        if mc < len(text):
            assert len(parts) >= len(text) // mc
        assert all(p.strip() for p in parts)
    assert ox.split_chunks("short", 100) == ["short"]
    blob = "x" * 5000                                                             # no boundary at all: hard cut
    assert "".join(ox.split_chunks(blob, 1000)) == blob and len(ox.split_chunks(blob, 1000)) == 5
    parts = ox.split_chunks(text, 400)
    assert all(p.endswith("\n") for p in parts[:-1])                              # cuts at line ends when they exist


def test_noise_profile_statistics():
    prof = ox.noise_profile(_train() + _train(6, "dta19-l0/de"))
    p = prof["impresso-snippets/en"]
    assert p["n"] == 12 and 0 < p["ocr_cmer"] < 0.2 and p["line_hyphens"] == 12
    assert p["hyphen_join_gain"] > 0.0
    assert set(prof) == {"impresso-snippets/en", "dta19-l0/de"}
    assert p["unit_cmer_p50"] <= p["unit_cmer_p90"] <= p["unit_cmer_max"]


def test_plan_items_and_identity_merge():
    units, train = _units(), _train()
    p = ox.plan(units, train, max_chars=60, n_examples=2, skip_below=None, join_hyphens=False)
    assert len(p["items"]) == sum(len(r["chunks"]) for r in p["units"]) and len(p["items"]) > 2
    for it in p["items"]:
        assert it["chunk"] in it["prompt"] and "Mothei" in it["prompt"] or it["test_set"] != "impresso-snippets/en" or it["cid"] > 0
        assert set(it) >= {"uid", "cid", "language", "test_set", "chunk", "prompt"}
    r = ox.merge(p, [it["chunk"] for it in p["items"]])
    assert r["y"] == [u["ocr_text"] for u in units]
    assert r["report"]["n_rejected"] == 0 and r["report"]["n_sent"] == len(p["items"])


def test_plan_skips_low_noise_test_sets_and_joins_hyphens_from_training_gain():
    clean = [{"ocr_text": GOLD.replace("¬", ""), "gt_text": GOLD.replace("¬", ""), "test_set": "dta19-l0/de", "language": "de"} for _ in range(4)]
    train = _train() + clean
    units = [{"ocr_text": OCR, "language": "en", "test_set": "impresso-snippets/en"},
             {"ocr_text": "Sauber.\nText.", "language": "de", "test_set": "dta19-l0/de"}]
    p = ox.plan(units, train)
    assert p["skipped_test_sets"] == ["dta19-l0/de"]
    assert {it["uid"] for it in p["items"]} == {0}
    assert p["units"][0]["join"] is True and p["units"][1]["join"] is False       # hyphen gain only where the training pairs show it
    r = ox.merge(p, [it["chunk"] for it in p["items"]])
    assert r["y"][1] == "Sauber.\nText." and "Mon¬\nday" in r["y"][0]
    p2 = ox.plan(units, train, skip_below=None, join_hyphens=False)
    assert {it["uid"] for it in p2["items"]} == {0, 1} and ox.merge(p2, [it["chunk"] for it in p2["items"]])["y"][0] == OCR


def test_merge_guard_rejects_and_falls_back():
    unit = [{"ocr_text": "\n".join(f"the quick brown fox number {i} jumps over" for i in range(30)), "language": "en", "test_set": "icdar2017/en"}]
    p = ox.plan(unit, _train(), max_chars=300, skip_below=None, n_examples=0, join_hyphens=False)
    n = len(p["items"])
    assert n >= 3
    chunks = [it["chunk"] for it in p["items"]]
    outs = list(chunks)
    outs[0] = chunks[0].replace("quick", "quiek", 1)                             # accepted: one character
    outs[1] = "Sure, here is the corrected text:\n" + chunks[1]                  # added preamble: length band
    outs[2] = chunks[2][: len(chunks[2]) // 2]                                   # truncated reply: length band
    r = ox.merge(p, outs)
    assert r["report"]["rejected_by_reason"] == {"length": 2} and r["report"]["n_rejected"] == 2
    pieces = ox.split_chunks(unit[0]["ocr_text"], 300)
    assert r["y"][0].startswith(pieces[0].replace("quick", "quiek", 1))
    assert pieces[1] in r["y"][0] and pieces[2] in r["y"][0]                     # rejected chunks keep the input text
    assert r["report"]["per_test_set"] == {"icdar2017/en": {"sent": n, "rejected": 2}}
    # empty / None replies keep the input; a rewritten chunk is rejected by the edit-rate cap
    outs = [None] * n
    assert ox.merge(p, outs)["y"] == [unit[0]["ocr_text"]]
    assert ox.merge(p, outs)["report"]["rejected_by_reason"] == {"empty": n}
    rewritten = [c.replace("brown", "crimson").replace("jumps", "leaps") for c in chunks]
    r = ox.merge(p, rewritten, len_band=None, max_hunk_edits=None, max_edit_rate=0.15)
    assert r["report"]["rejected_by_reason"].get("edit_rate") == n and r["y"] == [unit[0]["ocr_text"]]
    r = ox.merge(p, rewritten, len_band=None, max_hunk_edits=None, max_edit_rate=None)
    assert r["y"] == ["".join(rewritten)] and r["report"]["n_rejected"] == 0
    with pytest.raises(ValueError):
        ox.merge(p, outs[:-1])


def test_hunk_filter_keeps_small_changes_and_drops_rewrites():
    src = "the tlie old cily wall was bnilt in seventeen hundred"
    rep = "the the old city wall was built in eighteen thousand"
    out, taken, kept = ox.filter_hunks(src, rep, max_hunk_edits=3)
    assert out == "the the old city wall was built in seventeen hundred"
    assert taken == 3 and kept >= 1
    assert ox.filter_hunks(src, rep, None) == (rep, 0, 0)
    assert ox.filter_hunks("a b", "a  b", 0)[0] == "a  b"                        # whitespace-only change is free
    assert ox.filter_hunks("prov-\ning", "proving", 1)[0] == "proving"           # one edit after normalisation ('prov ing' -> 'proving')
    assert ox.filter_hunks("prov-\ning", "proving", 0)[0] == "prov-\ning"


def test_clean_reply_removes_fences_and_label():
    assert ox.clean_reply("```text\nabc def\n```") == "abc def"
    assert ox.clean_reply("Corrected text:\nabc def") == "abc def"
    assert ox.clean_reply(None) is None and ox.clean_reply(3) is None


def test_prompt_contains_instruction_examples_and_chunk():
    t = ox.build_prompt("Some text", "fr", "icdar2017/fr", [("bad", "good")])
    assert "French" in t and "OCR: bad" in t and "Corrected: good" in t and t.rstrip().endswith("Corrected text:") and "Some text" in t
    assert "Some text" in ox.build_prompt("Some text", "xx", "t", [], template="Fix:\n{chunk}") and "{" not in ox.build_prompt("t", "de", "x")


def test_describe_lists_every_public_name():
    text = scilib.describe("ocrfix")
    for name in ox.__all__:
        assert name in text


def test_code_node_may_import_scilib():
    code = "from scilib.ocrfix import plan, merge\n\ndef run(inputs, config):\n    return {}\n"
    assert scan_code(code) == []


def test_plan_enlarges_chunks_to_respect_the_item_limit():
    big = [{"ocr_text": "\n".join(f"the quick brown fox number {i} jumps over the lazy dog" for i in range(900)),
            "language": "en", "test_set": "icdar2017/en"}]
    p = ox.plan(big, _train(), max_chars=300, skip_below=None, n_examples=0, join_hyphens=False)
    assert len(p["items"]) <= 128 and p["settings"]["max_chars"] > 300 and p["settings"]["n_items"] == len(p["items"])
    assert "".join(p["units"][0]["chunks"]) == big[0]["ocr_text"]
    small = ox.plan(big, _train(), max_chars=300, skip_below=None, n_examples=0, join_hyphens=False, max_items=10**6)
    assert len(small["items"]) > 128 and small["settings"]["max_chars"] == 300
    assert ox.plan(_units(), _train())["settings"]["n_examples"] == 4


def test_module_is_deterministic():
    a = ox.plan(_units(), _train(), max_chars=80)
    b = ox.plan(_units(), _train(), max_chars=80)
    assert a == b
