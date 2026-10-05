"""Tools for caption strings scored by sentence chrF++ (AmericasNLP 2026 cultural image captioning).

chrf(hyp, ref) -> float                      sentence chrF++, 0-100: the sacrebleu ``CHRF(word_order=2)`` definition (character
                                             1-6-grams with whitespace removed, word 1-2-grams after splitting a leading or
                                             trailing punctuation mark off each word, beta = 2, orders without n-grams skipped),
                                             computed here without sacrebleu
mean_chrf(hyp, refs, weights=None) -> float  mean (or weighted mean) of chrf(hyp, r) over the strings r in refs
medoid(captions) -> str                      among the distinct captions (sorted), the first one with the highest mean chrf
                                             against the other distinct captions (always one of the given captions)
consensus_caption(captions, max_words=25, vocab_size=300, ...) -> str
                                             greedy word-by-word string; see below
consensus_details(captions, ...) -> dict     the same search with its trace: text, words, per-step mean chrf++, fallback flag
mbr_caption(captions, max_words=40, ...) -> str
                                             greedy selection of whole clauses of the given captions; see below
by_language(rows, key="iso_lang") -> dict    {language code: [captions]} from the ``load_train`` rows (empty captions dropped)
loo_score(captions, strategy, ...) -> dict   leave-one-out / repeated-split estimate of the mean chrf++ that a strategy
                                             reaches on captions it did not see
loo_compare(captions, strategies, ...) -> dict   the same on identical folds for several strategies, with paired differences
loo_lengths(captions, lengths, ...) -> dict  the same for ``consensus_caption`` at several values of ``max_words`` (one search per fold)
fit_predict(train_rows, items, method="consensus", ...) -> list[str]   one string per item, built per language

All functions work on plain strings and use no image. Every function takes the captions of ONE language (chrF++ is
computed on characters and words, so captions of different languages must not be pooled); ``by_language`` and
``fit_predict`` group the ``load_train`` rows by ``iso_lang`` and never combine languages. A language without any
training caption gets the string ``fallback`` (default "a"). Results are deterministic (no randomness except the
seeded fold assignment of ``loo_score`` / ``loo_compare`` / ``loo_lengths``). ``chrf`` matches
``sacrebleu.metrics.CHRF(word_order=2).sentence_score(hyp, [ref]).score`` (checked in the tests).

consensus_caption(captions, max_words, vocab_size, weights, min_captions, max_chars, jackknife)
    Candidate words = the ``vocab_size`` most frequent whitespace-separated words of the given captions (frequency =
    number of captions containing the word, ties broken alphabetically). Start from the empty string; at each step append
    the candidate word that gives the largest (weighted) mean ``chrf`` of the growing string against ALL given captions
    (a word may be appended several times); stop when no word raises that mean, after ``max_words`` words, or when the
    string would exceed ``max_chars`` characters. With ``jackknife=True`` (default) a word that occurs in exactly one of
    the given captions is not credited against that caption: its n-grams count neither as matches nor as hypothesis
    n-grams in the chrf of that caption (they still count for the other captions); ``jackknife=False`` credits every
    word against every caption. The result depends only on the given captions, not on any item or image. With fewer than
    ``min_captions`` (default 2) distinct captions the medoid is returned instead (with one distinct caption, that
    caption). The medoid is also returned when its mean ``chrf`` against the other captions is higher than the search's
    final mean; the medoid is a whole caption, so in that case the result can be longer than ``max_words`` words (it is
    cut at ``max_chars`` characters). Otherwise the output is a list of frequent words chosen for n-gram overlap with the
    given captions; it is not a grammatical sentence.

mbr_caption(captions, max_words, weights, max_chars)
    Candidate units = the clauses of the given captions (a clause ends at . ! ? ... ; : , or a line break; identical
    clauses are one unit). Start from the empty string; at each step append the clause that gives the largest (weighted)
    mean ``chrf`` of the growing string against all given captions, where a clause is not credited against the captions
    that contain it (its n-grams count neither as matches nor as hypothesis n-grams in the chrf of those captions); stop
    when no clause raises the mean, or when ``max_words`` words or ``max_chars`` characters would be exceeded. The result
    is a sequence of clauses that occur in the given captions. The medoid is returned when no clause can be appended and
    when there are fewer than 2 distinct captions.

loo_score(captions, strategy, n_train=None, n_repeats=20, max_folds=30, seed=0)
    ``captions: list of strings of one language, or dict {language: list}. ``strategy``: a function
    ``f(train_captions) -> str`` (or the name "medoid" / "consensus" / "mbr") returning ONE string; that string is scored with
    ``chrf`` against each held-out caption. Duplicate strings count as one caption. ``n_train=None``: leave-one-out, fold
    i trains on all other captions and is scored on caption i (``max_folds`` keeps a seeded random subset of at most that
    many folds per language; ``max_folds=None`` uses every fold). ``n_train=m``: ``n_repeats`` random splits that train
    on m captions and score on all the others (m is clipped to leave at least one test caption); ``n_train`` can be set
    to the number of training captions of the episode to mimic the deployment situation. Returns ``mean`` (over
    languages, each language weighted equally), ``sem`` (standard error over folds; folds of repeated splits overlap, so
    it is optimistic), ``n_folds``, ``n_test`` and ``per_language``. The cost is n_folds calls of ``strategy`` per
    language.

loo_compare(captions, strategies, ...)
    ``strategies``: dict {name: strategy}. Same arguments and folds as ``loo_score``; returns ``scores``
    ({name: {"mean", "sem"}}), ``difference`` ({name: {"mean", "sem", "share_positive"}}: paired per-fold difference to
    the first strategy) and ``per_language``.

loo_lengths(captions, lengths=(10, 15, 20, 25, 30, 40), vocab_size=300, jackknife=True, n_train, n_repeats, max_folds, seed)
    ``loo_compare`` of ``consensus_caption(train, max_words=L, ...)`` for each L in ``lengths``. The search is
    prefix-monotone in ``max_words`` (the string for a smaller L is the first L words of the string for a larger L), so
    each fold runs it once for all lengths; the cost is n_folds searches per language. Strategy names are ``str(L)``.

fit_predict(train_rows, items, method, max_words, vocab_size, fallback)
    ``train_rows``: ``load_train()["train"]`` (dicts with ``iso_lang`` and ``caption``); ``items``: ``load_eval_inputs()
    ["items"]``. method "consensus" builds ``consensus_caption``, "mbr" builds ``mbr_caption`` and "medoid" builds ``medoid``
    from the training captions of each item's language (``max_words=None`` means 25 for "consensus" and 40 for "mbr");
    every item of a language receives the same string, which is non-empty and at most 1000 characters.

Cost on one CPU core: 0.3-1 s per language for ``consensus_caption`` with 12-50 captions and the default arguments.
"""
from __future__ import annotations

import re
import zlib
from collections import Counter
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

__all__ = ["chrf", "mean_chrf", "medoid", "consensus_caption", "consensus_details", "mbr_caption", "by_language", "loo_score",
           "loo_compare", "loo_lengths", "fit_predict"]

CHAR_ORDER, WORD_ORDER, BETA = 6, 2, 2
_ORDERS = CHAR_ORDER + WORD_ORDER
_PUNCTS = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")   # sacrebleu's CHRF._PUNCTS


# ----------------------------------------------------------------------------------------------- chrF++ statistics
def _split_punct(word: str) -> list[str]:
    """Split one punctuation mark off the end (else the start) of a word, as sacrebleu's chrF++ tokenizer does."""
    if len(word) == 1:
        return [word]
    if word[-1] in _PUNCTS:
        return [word[:-1], word[-1]]
    if word[0] in _PUNCTS:
        return [word[0], word[1:]]
    return [word]


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    for w in text.split():
        out.extend(_split_punct(w))
    return out


class _Profile:
    """The 8 n-gram multisets (character orders 1-6, word orders 1-2) of one string and their totals."""

    __slots__ = ("counters", "totals")

    def __init__(self, text: str) -> None:
        chars = "".join(text.split())
        counters = [Counter(chars[i:i + n] for i in range(len(chars) - n + 1)) for n in range(1, CHAR_ORDER + 1)]
        toks = _tokens(text)
        counters += [Counter(" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)) for n in range(1, WORD_ORDER + 1)]
        self.counters = counters
        self.totals = np.array([sum(c.values()) for c in counters], dtype=float)


def _f_from_counts(n_hyp, n_ref, n_match):
    """chrF++ F-score (0-100) from arrays (..., 8) of hypothesis / reference / matching n-gram counts.

    Same arithmetic as sacrebleu (eps_smoothing off): an order counts only if both sides have n-grams of that order;
    precision and recall are averaged over the counted orders, then combined with beta = 2. ``n_hyp`` must already be
    0 for the orders in which the reference has no n-gram (sacrebleu reports no hypothesis n-grams there)."""
    n_hyp, n_ref, n_match = (np.asarray(a, dtype=float) for a in (n_hyp, n_ref, n_match))
    valid = (n_hyp > 0) & (n_ref > 0)
    prec = np.where(n_hyp > 0, n_match / np.maximum(n_hyp, 1.0), 0.0)
    rec = np.where(n_ref > 0, n_match / np.maximum(n_ref, 1.0), 0.0)
    eff = np.maximum(valid.sum(-1), 1)
    ap = (prec * valid).sum(-1) / eff
    ar = (rec * valid).sum(-1) / eff
    f2 = float(BETA) ** 2
    den = f2 * ap + ar
    return np.where(den > 0, 100.0 * (1.0 + f2) * ap * ar / np.where(den > 0, den, 1.0), 0.0)


def _pair_score(h: _Profile, r: _Profile) -> float:
    match = np.empty(_ORDERS)
    for o, (hc, rc) in enumerate(zip(h.counters, r.counters)):
        if len(hc) > len(rc):
            hc, rc = rc, hc                                    # min(count) is symmetric: iterate the smaller counter
        match[o] = sum(min(c, rc[g]) for g, c in hc.items() if g in rc)
    n_hyp = np.where(r.totals > 0, h.totals, 0.0)
    return float(_f_from_counts(n_hyp, r.totals, match))


def chrf(hyp: str, ref: str) -> float:
    """Sentence chrF++ (character 1-6-grams, word 1-2-grams, beta = 2) of ``hyp`` against ``ref``, 0-100."""
    return _pair_score(_Profile(str(hyp)), _Profile(str(ref)))


def _clean(captions: Iterable[str], what: str = "captions") -> list[str]:
    out: list[str] = []
    for c in captions:
        if not isinstance(c, str):
            raise TypeError(f"{what} must be strings, got {type(c).__name__}")
        if c.strip():
            out.append(c.strip())
    return out


def _weights(weights, n: int) -> np.ndarray:
    if weights is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,) or np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError(f"weights must be {n} non-negative numbers with a positive sum")
    return w / w.sum()


def mean_chrf(hyp: str, refs: Sequence[str], weights=None) -> float:
    """Mean sentence chrF++ of ``hyp`` against every string in ``refs`` (optionally weighted), 0-100."""
    refs = list(refs)
    if not refs:
        raise ValueError("refs is empty")
    w = _weights(weights, len(refs))
    h = _Profile(str(hyp))
    return float(sum(wi * _pair_score(h, _Profile(str(r))) for wi, r in zip(w, refs)))


def medoid(captions: Sequence[str]) -> str:
    """The distinct caption (sorted order) with the highest mean chrF++ against the other distinct captions; the first
    one wins ties. With one distinct caption that caption is returned; with none, ValueError."""
    caps = sorted(set(_clean(captions)))
    if not caps:
        raise ValueError("no non-empty caption")
    if len(caps) == 1:
        return caps[0]
    profs = [_Profile(c) for c in caps]
    best, best_s = caps[0], -1.0
    for i, c in enumerate(caps):
        s = float(np.mean([_pair_score(profs[i], profs[j]) for j in range(len(caps)) if j != i]))
        if s > best_s + 1e-12:
            best, best_s = c, s
    return best


# ----------------------------------------------------------------------------------------------- consensus string
class _RefIndex:
    """Reference n-grams of a caption set with an inverted index, for fast incremental chrF++ of a growing string."""

    def __init__(self, captions: Sequence[str]) -> None:
        profs = [_Profile(c) for c in captions]
        self.n = len(profs)
        self.n_ref = np.stack([p.totals for p in profs], axis=1)             # (8, R)
        self._inv: list[dict[str, tuple[list[int], list[int]]]] = [dict() for _ in range(_ORDERS)]
        for j, p in enumerate(profs):
            for o, cnt in enumerate(p.counters):
                inv = self._inv[o]
                for g, c in cnt.items():
                    idx, val = inv.setdefault(g, ([], []))
                    idx.append(j)
                    val.append(c)
        self._vec: list[dict[str, np.ndarray | None]] = [dict() for _ in range(_ORDERS)]

    def counts(self, o: int, g: str):
        """(R,) counts of n-gram g in every reference, or None when no reference contains it."""
        cache = self._vec[o]
        if g in cache:
            return cache[g]
        hit = self._inv[o].get(g)
        if hit is None:
            v = None
        else:
            v = np.zeros(self.n, dtype=np.int32)
            v[hit[0]] = hit[1]
        cache[g] = v
        return v


class _Growing:
    """A string built from units (words or clauses) with incrementally maintained n-gram counts and, for every
    reference, the number of hypothesis n-grams and matching n-grams counted against it."""

    def __init__(self, index: _RefIndex) -> None:
        self.index = index
        self.chars = ""
        self.tokens: list[str] = []
        self.hyp_counts = [Counter() for _ in range(_ORDERS)]
        self.n_hyp = np.zeros((_ORDERS, index.n))
        self.match = np.zeros((_ORDERS, index.n))
        self._ones = np.ones(index.n)

    def _delta(self, unit: str) -> list[list[str]]:
        new_chars = self.chars + "".join(unit.split())
        toks = self.tokens + _tokens(unit)
        out = []
        for n in range(1, CHAR_ORDER + 1):
            out.append([new_chars[i:i + n] for i in range(max(len(self.chars) - n + 1, 0), len(new_chars) - n + 1)])
        for n in range(1, WORD_ORDER + 1):
            out.append([" ".join(toks[i:i + n]) for i in range(max(len(self.tokens) - n + 1, 0), len(toks) - n + 1)])
        return out

    def _delta_match(self, delta: list[list[str]]) -> np.ndarray:
        d = np.zeros((_ORDERS, self.index.n))
        for o, grams in enumerate(delta):
            base, seen = self.hyp_counts[o], {}
            for g in grams:
                c = base.get(g, 0) + seen.get(g, 0) + 1
                seen[g] = seen.get(g, 0) + 1
                v = self.index.counts(o, g)
                if v is not None:
                    d[o] += v >= c
        return d

    def _stats(self, unit: str, skip: Sequence[int]):
        delta = self._delta(unit)
        dm = self._delta_match(delta)
        dn = np.array([len(g) for g in delta], dtype=float)[:, None] * self._ones
        if len(skip):
            dm[:, list(skip)] = 0.0
            dn[:, list(skip)] = 0.0
        return delta, dm, dn

    def score_with(self, unit: str, w: np.ndarray, skip: Sequence[int] = ()) -> float:
        """Weighted mean chrF++ against the references if ``unit`` were appended; the references in ``skip`` receive
        neither matches nor hypothesis n-grams from ``unit``."""
        _, dm, dn = self._stats(unit, skip)
        n_hyp = np.where(self.index.n_ref > 0, self.n_hyp + dn, 0.0)
        return float(_f_from_counts(n_hyp.T, self.index.n_ref.T, (self.match + dm).T) @ w)

    def append(self, unit: str, skip: Sequence[int] = ()) -> None:
        delta, dm, dn = self._stats(unit, skip)
        self.match = self.match + dm
        self.n_hyp = self.n_hyp + dn
        for o, grams in enumerate(delta):
            self.hyp_counts[o].update(grams)
        self.chars += "".join(unit.split())
        self.tokens += _tokens(unit)


def _greedy(caps: Sequence[str], units: Mapping[str, Sequence[int]], w: np.ndarray, max_words: int, max_chars: int,
            max_units: int = 10 ** 6) -> tuple[list[str], list[float]]:
    """Forward selection over ``units`` ({text: reference indices that get no credit for it}); the unit that gives the
    largest weighted mean chrF++ is appended until no unit raises it, ``max_words`` words or ``max_chars`` characters
    would be exceeded, or ``max_units`` units are chosen. Ties go to the alphabetically first unit."""
    grow = _Growing(_RefIndex(caps))
    order = sorted(units)
    chosen: list[str] = []
    steps: list[float] = []
    best, n_words, n_chars = 0.0, 0, 0
    while len(chosen) < max_units:
        cand = [u for u in order if n_words + len(u.split()) <= max_words
                and n_chars + len(u) + (1 if chosen else 0) <= max_chars]
        if not cand:
            break
        sc = np.array([grow.score_with(u, w, units[u]) for u in cand])
        j = int(np.argmax(sc))
        if not sc[j] > best + 1e-9:
            break
        best = float(sc[j])
        grow.append(cand[j], units[cand[j]])
        chosen.append(cand[j])
        steps.append(best)
        n_words += len(cand[j].split())
        n_chars += len(cand[j]) + (1 if len(chosen) > 1 else 0)
    return chosen, steps


def _word_units(caps: Sequence[str], vocab_size: int, jackknife: bool = True) -> dict[str, tuple[int, ...]]:
    """The ``vocab_size`` most frequent words ({word: (i,) if only caption i contains it and ``jackknife`` else ()})."""
    where: dict[str, list[int]] = {}
    for i, c in enumerate(caps):
        for word in set(c.split()):
            where.setdefault(word, []).append(i)
    top = sorted(where.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:max(int(vocab_size), 0)]
    return {word: (tuple(idx) if jackknife and len(idx) == 1 else ()) for word, idx in top}


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    return (cut.rsplit(" ", 1)[0] if " " in cut else cut).strip() or cut


def consensus_details(captions: Sequence[str], max_words: int = 25, vocab_size: int = 300, weights=None,
                      min_captions: int = 2, max_chars: int = 1000, jackknife: bool = True) -> dict:
    """``consensus_caption`` with its trace: ``text``, ``words``, ``step_scores`` (weighted mean chrF++ against the
    captions after each appended word, with the ``jackknife`` rule), ``visible_score`` (the same quantity for ``text``;
    for a fallback, the medoid's mean chrF++ against all captions), ``medoid``, ``medoid_score`` (its mean chrF++ against
    the other captions), ``fallback`` (True when the medoid was returned), ``n_captions``. ``weights``: optional
    per-caption weights (same order as ``captions``)."""
    raw = list(captions)
    if weights is not None and len(weights) != len(raw):
        raise ValueError("weights must have one entry per caption")
    keep = [i for i, c in enumerate(raw) if isinstance(c, str) and c.strip()]
    caps = _clean(raw)
    if not caps:
        raise ValueError("no non-empty caption")
    w = _weights(None if weights is None else np.asarray(weights, dtype=float)[keep], len(caps))
    med = _truncate(medoid(caps), max_chars)
    others = [c for c in sorted(set(caps)) if c != med]
    med_score = mean_chrf(med, others) if others else 100.0
    fallback = {"text": med, "words": med.split(), "step_scores": [], "visible_score": mean_chrf(med, caps, w),
                "medoid": med, "medoid_score": med_score, "fallback": True, "n_captions": len(caps)}
    if len(set(caps)) < max(int(min_captions), 1):
        return fallback
    words, steps = _greedy(caps, _word_units(caps, vocab_size, jackknife), w, int(max_words), int(max_chars))
    if not words or steps[-1] < med_score:
        return {**fallback, "step_scores": steps}
    return {"text": " ".join(words), "words": words, "step_scores": steps, "visible_score": steps[-1], "medoid": med,
            "medoid_score": med_score, "fallback": False, "n_captions": len(caps)}


def consensus_caption(captions: Sequence[str], max_words: int = 25, vocab_size: int = 300, weights=None,
                      min_captions: int = 2, max_chars: int = 1000, jackknife: bool = True) -> str:
    """Greedy forward word selection maximising the mean chrF++ against ``captions`` (see the module docstring)."""
    return consensus_details(captions, max_words, vocab_size, weights, min_captions, max_chars, jackknife)["text"]


_CLAUSE_BREAK = re.compile(r"(?<=[.!?\u2026;:,])\s+|\n+")


def _clause_units(caps: Sequence[str]) -> dict[str, tuple[int, ...]]:
    """{clause: indices of the captions that contain it} (clauses end at . ! ? ... ; : , or a line break)."""
    where: dict[str, list[int]] = {}
    for i, c in enumerate(caps):
        for part in _CLAUSE_BREAK.split(c):
            part = part.strip()
            if part and i not in where.get(part, ()):
                where.setdefault(part, []).append(i)
    return {u: tuple(idx) for u, idx in where.items()}


def mbr_caption(captions: Sequence[str], max_words: int = 40, weights=None, max_chars: int = 1000) -> str:
    """Greedy selection of whole clauses of the given captions maximising the mean chrF++ against the captions (see the
    module docstring). Returns the medoid when no clause can be appended."""
    raw = list(captions)
    if weights is not None and len(weights) != len(raw):
        raise ValueError("weights must have one entry per caption")
    keep = [i for i, c in enumerate(raw) if isinstance(c, str) and c.strip()]
    caps = _clean(raw)
    if not caps:
        raise ValueError("no non-empty caption")
    w = _weights(None if weights is None else np.asarray(weights, dtype=float)[keep], len(caps))
    med = _truncate(medoid(caps), max_chars)
    if len(set(caps)) < 2:
        return med
    chosen, _ = _greedy(caps, _clause_units(caps), w, int(max_words), int(max_chars))
    return " ".join(chosen) if chosen else med


# ----------------------------------------------------------------------------------------------- grouping
def by_language(rows: Iterable[Mapping], key: str = "iso_lang") -> dict[str, list[str]]:
    """{language code: [caption, ...]} from ``load_train`` rows, languages in sorted order, empty captions dropped."""
    out: dict[str, list[str]] = {}
    for r in rows:
        cap = r.get("caption")
        if isinstance(cap, str) and cap.strip():
            out.setdefault(str(r[key]), []).append(cap.strip())
    return {k: out[k] for k in sorted(out)}


# ----------------------------------------------------------------------------------------------- validation folds
_STRATEGIES: dict[str, Callable[[list[str]], str]] = {"medoid": medoid, "consensus": consensus_caption, "mbr": mbr_caption}


def _resolve(strategy) -> Callable[[list[str]], str]:
    if isinstance(strategy, str):
        if strategy not in _STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; use a function or one of {sorted(_STRATEGIES)}")
        return _STRATEGIES[strategy]
    if not callable(strategy):
        raise TypeError("strategy must be a function f(train_captions) -> str or a strategy name")
    return strategy


def _folds(caps: list[str], n_train, n_repeats: int, max_folds, rng) -> list[tuple[list[str], list[str]]]:
    caps = sorted(set(caps))
    n = len(caps)
    if n < 2:
        return []
    if n_train is None:
        order = [int(i) for i in rng.permutation(n)] if max_folds is not None else list(range(n))
        if max_folds is not None:
            order = sorted(order[:max(int(max_folds), 1)])
        return [([c for j, c in enumerate(caps) if j != i], [caps[i]]) for i in order]
    m = int(min(max(int(n_train), 1), n - 1))
    folds = []
    for _ in range(int(n_repeats)):
        perm = rng.permutation(n)
        folds.append(([caps[j] for j in sorted(perm[:m])], [caps[j] for j in sorted(perm[m:])]))
    return folds


def _as_groups(captions) -> dict[str, list[str]]:
    if isinstance(captions, Mapping):
        groups = {str(k): _clean(v) for k, v in captions.items()}
    else:
        groups = {"": _clean(captions)}
    return {k: groups[k] for k in sorted(groups)}


def _sem(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def _evaluate(captions, strategies: Mapping[str, Callable], n_train, n_repeats, max_folds, seed) -> dict:
    """Per language and strategy: fold scores (mean chrF++ of the strategy's string on the fold's held-out captions)."""
    groups = _as_groups(captions)
    per: dict[str, dict] = {}
    for lang, caps in groups.items():
        rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, zlib.crc32(lang.encode("utf-8"))])
        folds = _folds(caps, n_train, n_repeats, max_folds, rng)
        if not folds:
            continue
        scores = {name: [] for name in strategies}
        for train, test in folds:
            for name, fn in strategies.items():
                pred = fn(list(train))
                if not isinstance(pred, str):
                    raise TypeError(f"strategy {name!r} returned {type(pred).__name__}, expected str")
                scores[name].append(mean_chrf(pred, test))
        per[lang] = {name: np.asarray(v) for name, v in scores.items()}
        per[lang]["_n_test"] = float(np.mean([len(t) for _, t in folds]))
    if not per:
        raise ValueError("need at least 2 distinct captions in some language")
    return per


def _pool(values: list[float], sems: list[float]) -> tuple[float, float]:
    mean = float(np.mean(values))
    sem = float(np.sqrt(np.sum(np.square(sems))) / len(sems)) if np.all(np.isfinite(sems)) else float("nan")
    return mean, sem


def loo_score(captions, strategy, n_train: int | None = None, n_repeats: int = 20, max_folds: int | None = 30,
              seed: int = 0) -> dict:
    """Mean chrF++ that ``strategy`` reaches on held-out captions (see the module docstring).

    Returns {"mean", "sem", "n_folds", "n_test", "per_language": {lang: {"mean", "sem", "n_folds"}}}; ``mean`` is the
    equal-weight mean over languages (a single list of captions counts as one language)."""
    fn = _resolve(strategy)
    per = _evaluate(captions, {"s": fn}, n_train, n_repeats, max_folds, seed)
    langs = {k: {"mean": float(v["s"].mean()), "sem": _sem(v["s"]), "n_folds": int(len(v["s"]))} for k, v in per.items()}
    mean, sem = _pool([v["mean"] for v in langs.values()], [v["sem"] for v in langs.values()])
    return {"mean": mean, "sem": sem, "n_folds": int(sum(v["n_folds"] for v in langs.values())),
            "n_test": float(np.mean([v["_n_test"] for v in per.values()])), "per_language": langs}


def loo_compare(captions, strategies: Mapping[str, object], n_train: int | None = None, n_repeats: int = 20,
                max_folds: int | None = 30, seed: int = 0) -> dict:
    """``loo_score`` for several strategies on identical folds, plus paired differences to the first strategy.

    Returns {"scores": {name: {"mean", "sem"}}, "difference": {name: {"mean", "sem", "share_positive"}},
    "per_language": {lang: {name: mean}}, "n_folds"}; the difference is (name minus first strategy) per fold."""
    fns = {str(k): _resolve(v) for k, v in strategies.items()}
    if len(fns) < 1:
        raise ValueError("strategies is empty")
    names = list(fns)
    per = _evaluate(captions, fns, n_train, n_repeats, max_folds, seed)
    scores, diffs = {}, {}
    for name in names:
        m, s = _pool([float(v[name].mean()) for v in per.values()], [_sem(v[name]) for v in per.values()])
        scores[name] = {"mean": m, "sem": s}
        d = {lang: v[name] - v[names[0]] for lang, v in per.items()}
        dm, ds = _pool([float(x.mean()) for x in d.values()], [_sem(x) for x in d.values()])
        diffs[name] = {"mean": dm, "sem": ds, "share_positive": float(np.mean(np.concatenate(list(d.values())) > 0))}
    return {"scores": scores, "difference": diffs,
            "per_language": {lang: {name: float(v[name].mean()) for name in names} for lang, v in per.items()},
            "n_folds": int(sum(len(v[names[0]]) for v in per.values()))}


def loo_lengths(captions, lengths: Sequence[int] = (10, 15, 20, 25, 30, 40), vocab_size: int = 300, jackknife: bool = True,
                n_train: int | None = None, n_repeats: int = 20, max_folds: int | None = 30, seed: int = 0) -> dict:
    """``loo_compare`` of ``consensus_caption(train, max_words=L, vocab_size=vocab_size, jackknife=jackknife)`` for every L
    in ``lengths``, on the folds of ``loo_score``. The greedy search is prefix-monotone in ``max_words`` (the string for
    a smaller L is the first L words of the string for a larger L), so each fold runs the search once for all lengths.
    Returns the ``loo_compare`` dict (strategy names are ``str(L)``; ``difference`` is relative to the smallest L) plus
    ``lengths``."""
    lens = sorted({int(x) for x in lengths if int(x) > 0})
    if not lens:
        raise ValueError("lengths must contain positive integers")
    memo: dict = {}

    def texts(train: list[str]) -> dict[int, str]:
        key = tuple(train)
        if memo.get("key") != key:
            caps = _clean(train)
            med = medoid(caps)
            others = [c for c in sorted(set(caps)) if c != med]
            med_score = mean_chrf(med, others) if others else 100.0
            words, steps = ([], [])
            if len(set(caps)) >= 2:
                words, steps = _greedy(caps, _word_units(caps, vocab_size, jackknife), _weights(None, len(caps)), lens[-1], 1000)
            out = {}
            for n in lens:
                k = min(n, len(words))
                out[n] = med if k == 0 or steps[k - 1] < med_score else " ".join(words[:k])
            memo["key"], memo["texts"] = key, out
        return memo["texts"]

    res = loo_compare(captions, {str(n): (lambda train, n=n: texts(train)[n]) for n in lens}, n_train, n_repeats, max_folds, seed)
    res["lengths"] = lens
    return res


# ----------------------------------------------------------------------------------------------- entry point
def fit_predict(train_rows: Iterable[Mapping], items: Sequence[Mapping], method: str = "consensus",
                max_words: int | None = None, vocab_size: int = 300, fallback: str = "a", key: str = "iso_lang",
                max_chars: int = 1000) -> list[str]:
    """One caption string per item: the ``method`` string ("consensus" | "mbr" | "medoid") built from the training
    captions of the item's own language (rows are matched on ``key``); items of one language share the string.
    ``max_words=None`` means 25 for "consensus" and 40 for "mbr"; ``vocab_size`` applies to "consensus"."""
    if method not in _STRATEGIES:
        raise ValueError(f"unknown method {method!r}; use one of {sorted(_STRATEGIES)}")
    groups = by_language(train_rows, key)
    built: dict[str, str] = {}
    out = []
    for it in items:
        lang = str(it[key])
        if lang not in built:
            caps = groups.get(lang)
            if not caps:
                built[lang] = fallback
            elif method == "medoid":
                built[lang] = _truncate(medoid(caps), max_chars)
            elif method == "mbr":
                built[lang] = mbr_caption(caps, 40 if max_words is None else max_words, max_chars=max_chars)
            else:
                built[lang] = consensus_caption(caps, 25 if max_words is None else max_words, vocab_size, max_chars=max_chars)
        out.append(built[lang])
    return out
