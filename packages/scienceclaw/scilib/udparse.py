"""Tagging and dependency parsing for Universal Dependencies treebanks (numpy / scikit-learn; an optional pretrained French parser).

Sentence format: a dict ``{"words": [{"form": str, ...}, ...]}`` exactly as returned by ``load_train`` (words also carry
``upos``, ``feats``, ``head`` = 1-based head index with 0 = root, and ``deprel``), ``load_dev_inputs`` and
``load_eval_inputs`` (words carry ``id`` and ``form`` only). A plain list of form strings is accepted where no
annotation is needed. A parse is ``{"head": [int] * n_words, "deprel": [str] * n_words}``, the format of the deliverable;
the parsers below always return exactly one root word (head 0), heads in range and no cycle.

UDParser(...).fit(train) -> self         POS tagger + arc scorer + relation labeller, trained on annotated sentences
    .tag(sents) -> list[list[str]]       predicted UPOS per word
    .parse(sents) -> list[parse]         tags, then the tree of highest total arc score (projective Eisner or
                                         Chu-Liu/Edmonds), then one relation per arc
    .arc_scores(sents) -> list[array]    (n_words + 1, n_words) arc-score matrices, row = head (0 = root), column = word - 1
    .oof_upos_accuracy_                  accuracy of the 5-fold out-of-fold tags of the training sentences
fit_predict(train, targets, ...) -> list of parse lists    UDParser(**kw).fit(train), then .parse for every list in targets
fit_predict(train, targets, pretrained=True) -> list of parse lists    parses every list in targets with the pretrained French
                                         parser of ``scilib.udparse_pretrained`` (CamemBERT-large encoder with a deep-biaffine
                                         parser; ``train`` is not used; stored weights, run on a GPU worker when this interpreter
                                         has none; ``scilib.describe("udparse_pretrained")`` gives its training data and cost)
cross_validate(train, n_folds=3, gold_tags=False, ...) -> dict    k-fold estimate on the annotated sentences:
                                                       {"las", "uas", "upos_accuracy", "n_words", "per_fold"}
las_uas(gold, parses) -> dict            {"las", "uas", "n_words"}: share of words whose head (and universal relation,
                                         the part of deprel before ':') match; punctuation included (official CoNLL 2018 LAS)
check_parse(parse, n_words=None) -> list[str]   validity problems (heads outside 0..n, root count != 1, cycle); empty list = valid tree
UPOSTagger(...).fit(train).predict(sents) -> list[list[str]]    the tagger alone (linear SVM on word, affix, shape and
                                         neighbouring-word features)
.jackknife(sents, n_folds=5) -> list[list[str]]    out-of-fold predictions on the sentences it is given
eisner(scores) / chu_liu_edmonds(scores) -> np.ndarray    single-root maximum-scoring tree from a (n + 1, n) arc-score
                                         matrix (row = head, 0 = root; column = word - 1); returns heads of words 1..n
UD_RELATIONS                             the 37 UD v2 relations

UDParser options (defaults in the signature): ``decode`` 'eisner' (projective) or 'cle' (non-projective); ``epochs`` of
the averaged passive-aggressive arc-scorer training; ``n_models`` arc scorers with different shuffles whose score matrices
are averaged; ``jackknife_folds`` (the arc scorer and the labeller are trained on out-of-fold predicted tags; 0 = train on
gold tags); ``tagger_C`` / ``label_C`` SVM regularisation; ``use_morph`` adds predicted VerbForm, Number and Gender
values (from the annotated ``feats``) to the arc and relation features; ``seed``. Everything is deterministic given ``seed``.

Cost (measured on an otherwise idle CPU, 1000 training sentences = 24k words): ``fit`` about 20-30 CPU seconds and up to about
300 MB above the input data; ``parse`` about 1.5 s per 400 sentences; ``cross_validate`` runs ``n_folds`` fits, each on
(n_folds - 1) / n_folds of the sentences. The CPU time of a fit is stored in ``UDParser.fit_seconds_``.
Example: ``parses = fit_predict(train, [dev_sentences, eval_sentences])``; ``las_uas`` compares parses with annotated sentences.
"""
from __future__ import annotations

import re
import time
import warnings
from typing import Any, Sequence

import numpy as np

__all__ = ["UDParser", "UPOSTagger", "fit_predict", "cross_validate", "las_uas", "check_parse", "eisner",
           "chu_liu_edmonds", "UD_RELATIONS"]

UD_RELATIONS = ("acl", "advcl", "advmod", "amod", "appos", "aux", "case", "cc", "ccomp", "clf", "compound", "conj",
                "cop", "csubj", "dep", "det", "discourse", "dislocated", "expl", "fixed", "flat", "goeswith", "iobj",
                "list", "mark", "nmod", "nsubj", "nummod", "obj", "obl", "orphan", "parataxis", "punct",
                "reparandum", "root", "vocative", "xcomp")
_NEG = -1e9
_CACHE_BYTES = 200_000_000        # arc-feature tensors kept between epochs (larger training sets are recomputed)


# ----------------------------------------------------------------------------------------------- sentence access
def _forms(s: Any) -> list[str]:
    if isinstance(s, dict):
        return [str(w["form"]) for w in s["words"]]
    return [str(w) for w in s]


def _column(s: Any, key: str, default: str = "_") -> list[str]:
    return [str(w.get(key, default)) for w in s["words"]]


def _universal(rel: str) -> str:
    return rel.split(":")[0]


def _coerce(p: Any) -> tuple[list[int], list[str]]:
    if isinstance(p, dict):
        return [int(h) for h in p["head"]], [str(r) for r in p["deprel"]]
    return [int(x[0]) for x in p], [str(x[1]) for x in p]


# ----------------------------------------------------------------------------------------------- validity / metric
def check_parse(parse: Any, n_words: int | None = None) -> list[str]:
    """Problems that make a parse an invalid tree (empty list = valid): head outside 0..n, root count != 1, cycle."""
    heads, _ = _coerce(parse)
    n = len(heads)
    issues: list[str] = []
    if n_words is not None and n != n_words:
        issues.append(f"{n} heads for {n_words} words")
    if any(h < 0 or h > n for h in heads):
        return issues + ["HEAD outside the sentence"]
    roots = sum(1 for h in heads if h == 0)
    if roots != 1:
        issues.append(f"{roots} roots")
    for i in range(1, n + 1):
        seen: set[int] = set()
        j = i
        while j != 0:
            if j in seen:
                issues.append("cycle")
                return issues
            seen.add(j)
            j = heads[j - 1]
    return issues


def las_uas(gold: Sequence[Any], parses: Sequence[Any]) -> dict:
    """Attachment scores of ``parses`` against annotated sentences: LAS = share of words whose head and universal relation
    (deprel before ':') equal the gold ones, UAS = share with the gold head; punctuation counted; micro-average."""
    las = uas = tot = 0
    if len(gold) != len(parses):
        raise ValueError(f"{len(gold)} gold sentences but {len(parses)} parses")
    for g, p in zip(gold, parses):
        gh = [int(w["head"]) for w in g["words"]]
        gr = [_universal(str(w["deprel"])) for w in g["words"]]
        h, r = _coerce(p)
        if len(h) != len(gh):
            raise ValueError("parse and gold sentence differ in length")
        for a, b, c, d in zip(gh, gr, h, r):
            if a == c:
                uas += 1
                las += int(b == _universal(d))
        tot += len(gh)
    return {"las": las / max(tot, 1), "uas": uas / max(tot, 1), "n_words": tot}


# ----------------------------------------------------------------------------------------------- tree decoding
def _find_cycle(heads: np.ndarray) -> list[int] | None:
    n = len(heads)
    color = np.zeros(n, dtype=np.int8)
    for s in range(1, n):
        if color[s]:
            continue
        path = []
        v = s
        while v > 0 and color[v] == 0:
            color[v] = 1
            path.append(v)
            v = int(heads[v])
        if v > 0 and color[v] == 1:
            return path[path.index(v):]
        for u in path:
            color[u] = 2
    return None


def _cle(S: np.ndarray) -> np.ndarray:
    """Maximum spanning arborescence rooted at node 0 of the dense matrix S[h, d] (Chu-Liu/Edmonds); heads[0] = -1."""
    N = S.shape[0]
    heads = np.argmax(S, axis=0).astype(np.int64)
    heads[0] = -1
    cyc = _find_cycle(heads)
    if cyc is None:
        return heads
    cyc = np.asarray(cyc)
    in_cyc = np.zeros(N, dtype=bool)
    in_cyc[cyc] = True
    others = np.flatnonzero(~in_cyc)                    # contains node 0
    m = len(others)
    cyc_in = S[heads[cyc], cyc]
    total = cyc_in.sum()
    S2 = np.full((m + 1, m + 1), -np.inf)
    S2[:m, :m] = S[np.ix_(others, others)]
    enter = S[np.ix_(others, cyc)] - cyc_in[None, :]    # (m, |C|): attach cycle node u below outside node v
    enter_arg = np.argmax(enter, axis=1)
    S2[:m, m] = enter[np.arange(m), enter_arg] + total
    leave = S[np.ix_(cyc, others)]                      # (|C|, m)
    leave_arg = np.argmax(leave, axis=0)
    S2[m, :m] = leave[leave_arg, np.arange(m)]
    S2[:, 0] = -np.inf
    h2 = _cle(S2)
    out = heads.copy()
    for j in range(1, m):                               # non-cycle nodes
        hj = int(h2[j])
        out[others[j]] = others[hj] if hj < m else cyc[leave_arg[j]]
    v = int(h2[m])                                      # outside head of the contracted cycle
    out[cyc[enter_arg[v]]] = others[v]
    return out


def _single_root(S: np.ndarray, decode) -> np.ndarray:
    heads = decode(S)
    roots = np.flatnonzero(heads[1:] == 0) + 1
    if len(roots) <= 1:
        return heads
    best, best_val = None, -np.inf
    for r in roots:
        S2 = S.copy()
        S2[0, :] = -np.inf
        S2[0, r] = S[0, r]
        h = decode(S2)
        val = float(S[h[1:], np.arange(1, len(h))].sum())
        if val > best_val:
            best, best_val = h, val
    return best


def chu_liu_edmonds(scores: np.ndarray) -> np.ndarray:
    """Heads (1-based, 0 = root) of words 1..n of the single-root spanning tree that maximises the summed arc scores.
    ``scores[h, d - 1]``: score of the arc head h (0 = root) -> word d, shape (n + 1, n). Non-projective trees allowed."""
    Sc = np.asarray(scores, dtype=np.float64)
    n = Sc.shape[1]
    S = np.full((n + 1, n + 1), -np.inf)
    S[:, 1:] = Sc
    S[np.arange(1, n + 1), np.arange(1, n + 1)] = -np.inf
    return _single_root(S, _cle)[1:].astype(np.int64)


def eisner(scores: np.ndarray) -> np.ndarray:
    """Heads (1-based, 0 = root) of words 1..n of the single-root *projective* tree that maximises the summed arc
    scores (first-order Eisner algorithm). ``scores[h, d - 1]``: score of the arc head h (0 = root) -> word d."""
    Sc = np.asarray(scores, dtype=np.float64)
    n = Sc.shape[1]
    if n == 1:
        return np.zeros(1, dtype=np.int64)
    A = Sc[1:, :]                                        # A[h, d]: word h+1 -> word d+1 (0-based words)
    R = Sc[0, :]
    CR = np.full((n, n), -np.inf)
    CL = np.full((n, n), -np.inf)
    IR = np.full((n, n), -np.inf)
    IL = np.full((n, n), -np.inf)
    idx = np.arange(n)
    CR[idx, idx] = 0.0
    CL[idx, idx] = 0.0
    bI = np.zeros((n, n), dtype=np.int64)
    bCR = np.zeros((n, n), dtype=np.int64)
    bCL = np.zeros((n, n), dtype=np.int64)
    for k in range(1, n):
        i = np.arange(n - k)
        j = i + k
        rr = i[:, None] + np.arange(k)[None, :]
        split = CR[i[:, None], rr] + CL[rr + 1, j[:, None]]
        a = np.argmax(split, axis=1)
        best = split[np.arange(n - k), a]
        bI[i, j] = i + a
        IR[i, j] = best + A[i, j]
        IL[i, j] = best + A[j, i]
        rr2 = i[:, None] + 1 + np.arange(k)[None, :]
        v = IR[i[:, None], rr2] + CR[rr2, j[:, None]]
        a2 = np.argmax(v, axis=1)
        CR[i, j] = v[np.arange(n - k), a2]
        bCR[i, j] = i + 1 + a2
        v = CL[i[:, None], rr] + IL[rr, j[:, None]]
        a3 = np.argmax(v, axis=1)
        CL[i, j] = v[np.arange(n - k), a3]
        bCL[i, j] = i + a3
    top = R + CL[0, :] + CR[:, n - 1]
    root = int(np.argmax(top))
    heads = np.zeros(n, dtype=np.int64)                  # 1-based heads of words 1..n
    heads[root] = 0
    stack = [("L", 0, root), ("R", root, n - 1)]
    while stack:
        kind, i, j = stack.pop()
        if i == j:
            continue
        if kind == "R":                                  # complete span headed at i, extending right
            r = int(bCR[i, j])
            stack.append(("iR", i, r))
            stack.append(("R", r, j))
        elif kind == "L":                                # complete span headed at j, extending left
            r = int(bCL[i, j])
            stack.append(("L", i, r))
            stack.append(("iL", r, j))
        else:                                            # incomplete span: arc between i and j, then two complete halves
            r = int(bI[i, j])
            if kind == "iR":
                heads[j] = i + 1
            else:
                heads[i] = j + 1
            stack.append(("R", i, r))
            stack.append(("L", r + 1, j))
    return heads


def _tree_or_chain(heads: np.ndarray) -> list[int]:
    """Final guard: return ``heads`` if it is a valid single-root tree, else a right-branching chain."""
    h = [int(x) for x in heads]
    if not check_parse({"head": h, "deprel": ["dep"] * len(h)}):
        return h
    n = len(h)
    return [i + 2 if i + 1 < n else 0 for i in range(n)]


# ----------------------------------------------------------------------------------------------- token features
_SHAPE_RE = re.compile(r"[A-ZÀ-ÖØ-ÞŒ]")
_DEFAULT_TAG_C = 0.3


def _word_feats(w: str) -> list[str]:
    lw = w.lower()
    f = ["w=" + lw, "s1=" + lw[-1:], "s2=" + lw[-2:], "s3=" + lw[-3:], "s4=" + lw[-4:], "p1=" + lw[:1],
         "p2=" + lw[:2], "p3=" + lw[:3]]
    if any(c.isdigit() for c in w):
        f.append("digit")
    if w.isupper() and len(w) > 1:
        f.append("allcap")
    elif w[:1].isupper():
        f.append("cap")
    if "-" in w:
        f.append("hyph")
    if "'" in w or "’" in w:
        f.append("apos")
    if not any(c.isalnum() for c in w):
        f.append("punct=" + w[:3])
    if _SHAPE_RE.search(w[1:]) and not w.isupper():
        f.append("innercap")
    return f


class UPOSTagger:
    """Linear-SVM tagger over word / affix / shape features of the word and its two neighbours on each side.

    ``fit(sentences)`` reads ``words[*].upos`` (or another column: ``column='feats'`` values are then reduced with
    ``feat_key``); ``predict`` returns one label per word; ``jackknife`` gives out-of-fold predictions."""

    def __init__(self, C: float = _DEFAULT_TAG_C, column: str = "upos", feat_key: str | None = None, seed: int = 0):
        self.C, self.column, self.feat_key, self.seed = float(C), column, feat_key, int(seed)

    # -- labels
    def _label(self, w: dict) -> str:
        v = str(w.get(self.column, "_"))
        if self.feat_key is None:
            return v
        for kv in v.split("|"):
            if kv.startswith(self.feat_key + "="):
                return kv.split("=", 1)[1]
        return "_"

    # -- features
    def _token_strings(self, forms: list[str], cache: dict) -> list[list[str]]:
        wf = []
        for w in forms:
            f = cache.get(w)
            if f is None:
                f = cache[w] = _word_feats(w)
            wf.append(f)
        n = len(forms)
        lw = [w.lower() for w in forms]
        out = []
        for i in range(n):
            f = list(wf[i])
            f.append("bias")
            if i == 0:
                f.append("first")
            for off in (-2, -1, 1, 2):
                j = i + off
                if 0 <= j < n:
                    f.append(f"w{off}={lw[j]}")
                    if abs(off) == 1:
                        f.append(f"s3_{off}={lw[j][-3:]}")
                        f.append(f"s2_{off}={lw[j][-2:]}")
                        if wf[j] and ("cap" in wf[j]):
                            f.append(f"cap{off}")
                else:
                    f.append(f"w{off}=<pad>")
            if i > 0:
                f.append(f"w-1w0={lw[i - 1]}|{lw[i]}")
            if i + 1 < n:
                f.append(f"w0w+1={lw[i]}|{lw[i + 1]}")
            if i > 0 and i + 1 < n:
                f.append(f"w-1w+1={lw[i - 1]}|{lw[i + 1]}")
            out.append(f)
        return out

    def _matrix(self, sents: Sequence[Any], grow: bool):
        cache = self._cache
        indptr, indices = [0], []
        for s in sents:
            for f in self._token_strings(_forms(s), cache):
                for x in f:
                    j = self.vocab_.get(x)
                    if j is None and grow:
                        j = self.vocab_[x] = len(self.vocab_)
                    if j is not None:
                        indices.append(j)
                indptr.append(len(indices))
        return indptr, indices

    def _csr(self, indptr, indices):
        from scipy.sparse import csr_matrix
        return csr_matrix((np.ones(len(indices), dtype=np.float32), np.asarray(indices, dtype=np.int32),
                           np.asarray(indptr, dtype=np.int64)), shape=(len(indptr) - 1, len(self.vocab_)))

    def _fit_rows(self, X, y):
        from sklearn.svm import LinearSVC
        y = np.asarray(y)
        if len(set(y.tolist())) < 2:
            return None, y[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return LinearSVC(C=self.C, random_state=self.seed, max_iter=300).fit(X, y), None

    def fit(self, sents: Sequence[Any]) -> "UPOSTagger":
        self.vocab_: dict[str, int] = {}
        self._cache: dict[str, list[str]] = {}
        indptr, indices = self._matrix(sents, grow=True)
        y = [self._label(w) for s in sents for w in s["words"]]
        self._X_train_ptr = (indptr, indices)
        self._y_train = y
        self._lens_train = [len(s["words"]) for s in sents]
        self.model_, self._const = self._fit_rows(self._csr(indptr, indices), y)
        return self

    def _predict_X(self, X, model=None, const=None):
        model = self.model_ if model is None else model
        const = self._const if const is None else const
        if model is None:
            return np.asarray([const] * X.shape[0])
        return model.predict(X)

    def predict(self, sents: Sequence[Any]) -> list[list[str]]:
        indptr, indices = self._matrix(sents, grow=False)
        if len(indptr) == 1:                                   # no words at all
            return [[] for _ in sents]
        pred = self._predict_X(self._csr(indptr, indices))
        out, k = [], 0
        for s in sents:
            n = len(_forms(s))
            out.append([str(x) for x in pred[k:k + n]])
            k += n
        return out

    def jackknife(self, sents: Sequence[Any] | None = None, n_folds: int = 5) -> list[list[str]]:
        """Out-of-fold predictions for the sentences the tagger was fitted on (folds of whole sentences, seeded)."""
        ptr, ind = self._X_train_ptr
        X = self._csr(ptr, ind)
        lens = np.asarray(self._lens_train)
        offs = np.concatenate([[0], np.cumsum(lens)])
        y = np.asarray(self._y_train)
        n = len(lens)
        fold = np.random.default_rng(self.seed).permutation(n) % max(n_folds, 2)
        tok_fold = np.repeat(fold, lens)
        pred = np.empty(len(y), dtype=object)
        for f in range(max(n_folds, 2)):
            te = np.flatnonzero(tok_fold == f)
            tr = np.flatnonzero(tok_fold != f)
            if len(te) == 0:
                continue
            m, c = self._fit_rows(X[tr], y[tr])
            pred[te] = self._predict_X(X[te], m, c)
        return [[str(x) for x in pred[offs[i]:offs[i + 1]]] for i in range(n)]


# ----------------------------------------------------------------------------------------------- hashing helpers
_K1 = np.uint64(0x9E3779B97F4A7C15)
_K2 = np.uint64(0x7F4A7C15)
_S29 = np.uint64(29)
_DB = np.array([0, 1, 2, 3, 4, 5, 5, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9], dtype=np.uint64)
_FAM = {"VERB": 1, "AUX": 2, "PUNCT": 3, "CCONJ": 4, "SCONJ": 5, "NOUN": 6, "PROPN": 6, "ADP": 7, "PRON": 8}


def _mix(h, a):
    h = (h ^ (a + _K2)) * _K1
    return h ^ (h >> _S29)


def _hashed(tid: int, *arrs) -> np.ndarray:
    h = np.uint64(tid * 7919 + 13)
    for a in arrs:
        h = _mix(h, a)
    return h


class _Lexicon:
    """Integer ids for lower-cased forms (frequency >= min_count), suffixes and tags; id 0 = unknown."""

    def __init__(self, sents_forms, sents_tags, min_count: int, extra: list | None = None):
        cnt: dict[str, int] = {}
        for fs in sents_forms:
            for f in fs:
                lf = f.lower()
                cnt[lf] = cnt.get(lf, 0) + 1
        self.w = {f: i + 1 for i, f in enumerate(sorted(f for f, c in cnt.items() if c >= min_count))}
        scnt: dict[str, int] = {}
        for lf, c in cnt.items():
            scnt[lf[-3:]] = scnt.get(lf[-3:], 0) + c
        self.s3 = {f: i + 1 for i, f in enumerate(sorted(f for f, c in scnt.items() if c >= 2))}
        s2: dict[str, int] = {}
        for lf, c in cnt.items():
            s2[lf[-2:]] = s2.get(lf[-2:], 0) + c
        self.s2 = {f: i + 1 for i, f in enumerate(sorted(f for f, c in s2.items() if c >= 2))}
        tags = sorted({t for ts in sents_tags for t in ts})
        self.t = {t: i + 4 for i, t in enumerate(tags)}          # 0 unknown, 1 ROOT, 2 BOS, 3 EOS
        self.n_tags = len(tags)
        self.extra = []
        for col in (extra or []):
            vals = sorted({v for vs in col for v in vs})
            self.extra.append({v: i + 4 for i, v in enumerate(vals)})


class _Sent:
    """Integer attribute arrays of one sentence (position 0 = artificial root)."""

    __slots__ = ("n", "w", "t", "s3", "s2", "tm1", "tp1", "tm2", "tp2", "fam", "x", "shape")

    def __init__(self, lex: _Lexicon, forms: list[str], tags: list[str], extras: list[list[str]] | None = None):
        n = len(forms)
        self.n = n
        lf = [f.lower() for f in forms]
        u = np.uint64
        self.w = np.array([0] + [lex.w.get(f, 0) for f in lf], dtype=u)
        self.s3 = np.array([0] + [lex.s3.get(f[-3:], 0) for f in lf], dtype=u)
        self.s2 = np.array([0] + [lex.s2.get(f[-2:], 0) for f in lf], dtype=u)
        t = np.array([1] + [lex.t.get(x, 0) for x in tags], dtype=u)
        self.t = t
        pad = lambda a, k: np.concatenate([np.full(max(-k, 0), 2, u), a[max(k, 0):len(a) + min(k, 0)] if k else a,
                                           np.full(max(k, 0), 3, u)])
        self.tm1, self.tp1 = pad(t, -1), pad(t, 1)
        self.tm2, self.tp2 = pad(t, -2), pad(t, 2)
        fam = np.zeros(n + 1, dtype=np.int64)
        for i, x in enumerate(tags):
            fam[i + 1] = _FAM.get(x, 0)
        self.fam = fam
        self.x = [np.array([1] + [d.get(v, 0) for v in col], dtype=u) for d, col in zip(lex.extra, extras or [])]
        self.shape = np.array([0] + [(1 if f[:1].isupper() else 2 if f.isdigit() else 3 if not any(c.isalnum() for c in f)
                                      else 0) for f in forms], dtype=u)


# ----------------------------------------------------------------------------------------------- arc scorer
class _ArcScorer:
    """Edge-factored linear model over hashed pair features; trained head-selection style with averaged
    passive-aggressive updates (each word must prefer its gold head over every other candidate head by a margin)."""

    def __init__(self, epochs: int = 8, bits: int = 23, C: float = 0.5, margin: float = 1.0, seed: int = 0,
                 use_morph: bool = False):
        self.epochs, self.bits, self.C, self.margin, self.seed = epochs, bits, C, margin, seed
        self.use_morph = use_morph
        self.shift = np.uint64(64 - bits)

    def features(self, a: _Sent) -> np.ndarray:
        n = a.n
        u = np.uint64
        sh = self.shift
        col = lambda v: v[None, 1:]
        row = lambda v: v[:, None]
        pos = np.arange(n + 1)
        dist = pos[None, 1:] - pos[:, None]
        ad = np.abs(dist)
        dr = (dist > 0).astype(np.uint64)
        db = _DB[np.minimum(ad, 30)]
        dd = dr * u(16) + db
        tH, tD = row(a.t), col(a.t)
        wH, wD = row(a.w), col(a.w)
        sH, sD = row(a.s3), col(a.s3)
        s2H, s2D = row(a.s2), col(a.s2)
        tHm, tHp, tDm, tDp = row(a.tm1), row(a.tp1), col(a.tm1), col(a.tp1)
        tHp2, tDp2 = row(a.tp2), col(a.tp2)
        tHm2, tDm2 = row(a.tm2), col(a.tm2)
        T = []
        T.append(_hashed(1, tH, tD, dd))
        T.append(_hashed(2, tH, tD, dr))
        T.append(_hashed(3, wH, wD, dr))
        T.append(_hashed(4, wH, tD, dr))
        T.append(_hashed(5, tH, wD, dr))
        T.append(_hashed(6, wH, tH, tD, dr))
        T.append(_hashed(7, tH, wD, tD, dr))
        T.append(_hashed(8, tH, sD, dr))
        T.append(_hashed(9, sH, tD, dr))
        T.append(_hashed(10, wH, tH, dr, db))
        T.append(_hashed(11, wD, tD, dr, db))
        T.append(_hashed(12, tH, dd))
        T.append(_hashed(13, tD, dd))
        T.append(_hashed(14, wH, dd))
        T.append(_hashed(15, wD, dd))
        T.append(_hashed(16, tH, tHp, tD, dr))
        T.append(_hashed(17, tHm, tH, tD, dr))
        T.append(_hashed(18, tH, tD, tDm, dr))
        T.append(_hashed(19, tH, tD, tDp, dr))
        T.append(_hashed(20, tH, tHp, tDm, tD))
        T.append(_hashed(21, tHm, tH, tDm, tD))
        T.append(_hashed(22, tH, tHp, tD, tDp))
        T.append(_hashed(23, tHm, tH, tD, tDp))
        T.append(_hashed(24, tHm, tH, tHp, tD, dr))
        T.append(_hashed(25, tH, tDm, tD, tDp, dr))
        T.append(_hashed(26, tH, tHp, tHp2, tD, dr))
        T.append(_hashed(27, tH, tD, tDp, tDp2, dr))
        T.append(_hashed(28, tH, tD, tHm2, dr))
        T.append(_hashed(29, tH, tD, tDm2, dr))
        # material between head and dependent
        lo, hi = np.minimum(pos[:, None], pos[None, 1:]), np.maximum(pos[:, None], pos[None, 1:])
        hi1 = np.maximum(hi - 1, 0)
        for c in range(1, 9):
            cum = np.cumsum(a.fam == c)
            cnt = np.clip(cum[hi1] - cum[lo], 0, 2).astype(np.uint64)
            T.append(_hashed(30 + c, tH, tD, dr, cnt))
        # first / last position of the dependent, verbs before / after the dependent
        edge = np.zeros(n + 1, dtype=np.uint64)
        edge[1] = 1
        edge[n] += 2
        vb = np.cumsum((a.fam == 1) | (a.fam == 2))
        before = np.clip(np.concatenate([[0], vb[:-1]]), 0, 2).astype(np.uint64)
        after = np.clip(vb[-1] - vb, 0, 2).astype(np.uint64)
        T.append(_hashed(39, tH, tD, col(edge), dr))
        T.append(_hashed(40, tH, tD, dr, col(before), col(after)))
        T.append(_hashed(41, tH, tD, dr, row(before), row(after)))
        T.append(_hashed(42, sH, sD, dr))
        T.append(_hashed(43, s2H, tD, dr))
        if self.use_morph and a.x:
            xs = a.x
            mH = _hashed(0, *[row(v) for v in xs])
            mD = _hashed(0, *[col(v) for v in xs])
            T.append(_hashed(44, mH, mD, tH, tD, dr))
            T.append(_hashed(45, mH, mD, dd))
        else:
            T.append(_hashed(44, tH, tD, col(a.shape), dr))
            T.append(_hashed(45, tH, tD, dd, col(edge)))
        F = np.empty((len(T), n + 1, n), dtype=np.int32)
        for k, h in enumerate(T):
            F[k] = (h >> sh).astype(np.int32)
        return F

    def _mask_self(self, S: np.ndarray) -> np.ndarray:
        n = S.shape[1]
        S[np.arange(1, n + 1), np.arange(n)] = _NEG
        return S

    def fit(self, sents: list[_Sent], heads: list[np.ndarray]) -> "_ArcScorer":
        rng = np.random.default_rng(self.seed)
        M = 1 << self.bits
        W = np.zeros(M, dtype=np.float32)
        U = np.zeros(M, dtype=np.float32)
        c = 1.0
        order = np.arange(len(sents))
        cache: dict[int, np.ndarray] = {}                 # feature tensors reused across epochs (bounded memory)
        cached_bytes = 0
        for ep in range(self.epochs):
            rng.shuffle(order)
            for i in order:
                a, g = sents[i], heads[i]
                n = a.n
                F = cache.get(int(i))
                if F is None:
                    F = self.features(a)
                    if cached_bytes + F.nbytes <= _CACHE_BYTES:
                        cache[int(i)] = F
                        cached_bytes += F.nbytes
                S = W[F].sum(axis=0)
                d = np.arange(n)
                Sg = S[g, d].copy()
                S = self._mask_self(S) + np.float32(self.margin)
                S[g, d] -= np.float32(self.margin)
                p = np.argmax(S, axis=0)
                hinge = S[p, d] - Sg
                bad = np.flatnonzero((p != g) & (hinge > 0))
                c += 1.0
                if len(bad) == 0:
                    continue
                ig, ip = F[:, g[bad], bad], F[:, p[bad], bad]
                nz = (ig != ip).sum(axis=0) * 2.0
                tau = np.minimum(self.C, hinge[bad] / np.maximum(nz, 1.0)).astype(np.float32)
                tau[nz == 0] = 0.0
                T = F.shape[0]
                tt = np.broadcast_to(tau, (T, len(bad))).ravel()
                for idx, sign in ((ig.ravel(), 1.0), (ip.ravel(), -1.0)):
                    np.add.at(W, idx, sign * tt)
                    np.add.at(U, idx, sign * tt * np.float32(c))
        self.W_ = W - U / np.float32(c)
        return self

    def scores(self, a: _Sent) -> np.ndarray:
        F = self.features(a)
        return self._mask_self(self.W_[F].sum(axis=0).astype(np.float64))


# ----------------------------------------------------------------------------------------------- relation labeller
class _Labeller:
    """Linear SVM over hashed features of (head, dependent, context, dependent's children) for one arc."""

    def __init__(self, C: float = 0.3, bits: int = 24, seed: int = 0, use_morph: bool = False):
        self.C, self.bits, self.seed, self.use_morph = C, bits, seed, use_morph
        self.shift = np.uint64(64 - bits)

    def features(self, a: _Sent, heads: np.ndarray, n_tags: int) -> np.ndarray:
        n = a.n
        u = np.uint64
        sh = self.shift
        D = np.arange(1, n + 1)
        H = np.asarray(heads, dtype=np.int64)
        pos = D
        dist = D - H
        dr = (dist > 0).astype(np.uint64)
        db = _DB[np.minimum(np.abs(dist), 30)]
        dd = dr * u(16) + db
        tH, tD, wH, wD = a.t[H], a.t[D], a.w[H], a.w[D]
        sD, sH, s2D = a.s3[D], a.s3[H], a.s2[D]
        tDm, tDp, tDp2, tDm2 = a.tm1[D], a.tp1[D], a.tp2[D], a.tm2[D]
        tHm, tHp = a.tm1[H], a.tp1[H]
        # children of every word: tag histogram, first/last child tag, sibling tags
        ch = np.zeros((n + 1, n_tags + 4), dtype=np.int64)
        np.add.at(ch, (H, a.t[D].astype(np.int64)), 1)
        lch = np.zeros(n + 1, dtype=np.uint64)
        rch = np.zeros(n + 1, dtype=np.uint64)
        nch = np.zeros(n + 1, dtype=np.uint64)
        prev_sib = np.zeros(n, dtype=np.uint64)
        next_sib = np.zeros(n, dtype=np.uint64)
        sib_idx = np.zeros(n, dtype=np.uint64)
        by_head: dict[int, list[int]] = {}
        for d in range(1, n + 1):
            by_head.setdefault(int(H[d - 1]), []).append(d)
        for h, ds in by_head.items():
            nch[h] = min(len(ds), 4)
            lefts = [d for d in ds if d < h]
            rights = [d for d in ds if d > h]
            if lefts:
                lch[h] = a.t[lefts[0]]
            if rights:
                rch[h] = a.t[rights[-1]]
            for grp in (lefts, rights):
                for k, d in enumerate(grp):
                    prev_sib[d - 1] = a.t[grp[k - 1]] if k > 0 else 0
                    next_sib[d - 1] = a.t[grp[k + 1]] if k + 1 < len(grp) else 0
                    sib_idx[d - 1] = min(k, 3)
        hh = np.zeros(n + 1, dtype=np.int64)
        hh[1:] = H
        gp = a.t[hh[H]]                                         # tag of the head's head
        lc_d, rc_d, nc_d = lch[D], rch[D], nch[D]
        # between counts of the head-dependent span
        lo, hi = np.minimum(H, D), np.maximum(H, D)
        hi1 = np.maximum(hi - 1, 0)
        T = []
        T.append(_hashed(1, tD))
        T.append(_hashed(2, wD, tD))
        T.append(_hashed(3, tH))
        T.append(_hashed(4, wH, tH))
        T.append(_hashed(5, tH, tD))
        T.append(_hashed(6, wH, tD))
        T.append(_hashed(7, tH, wD))
        T.append(_hashed(8, wH, wD))
        T.append(_hashed(9, tD, dd))
        T.append(_hashed(10, tH, tD, dr))
        T.append(_hashed(11, tH, tD, dd))
        T.append(_hashed(12, tDm, tD, tDp))
        T.append(_hashed(13, tD, tDp))
        T.append(_hashed(14, tDm, tD))
        T.append(_hashed(15, tH, tD, tDm))
        T.append(_hashed(16, tH, tD, tDp))
        T.append(_hashed(17, sD, tH))
        T.append(_hashed(18, sD, tD))
        T.append(_hashed(19, s2D, tD, tH))
        T.append(_hashed(20, sH, tD))
        T.append(_hashed(21, wD, tD, dr))
        T.append(_hashed(22, wD, tH, dr))
        T.append(_hashed(23, tD, tH, lc_d, rc_d))
        T.append(_hashed(24, tD, tH, nc_d))
        T.append(_hashed(25, tD, prev_sib, dr))
        T.append(_hashed(26, tD, next_sib, dr))
        T.append(_hashed(27, tD, tH, prev_sib, next_sib))
        T.append(_hashed(28, tD, tH, gp))
        T.append(_hashed(29, tD, sib_idx, dr))
        T.append(_hashed(30, tD, tH, tDm, tDp))
        T.append(_hashed(31, tD, tH, tHm, tHp))
        T.append(_hashed(32, wD, tH, tD, tDp))
        T.append(_hashed(33, a.shape[D], tD, tH))
        T.append(_hashed(34, tD, (D == 1).astype(np.uint64) + u(2) * (D == n).astype(np.uint64), dr))
        for c in (1, 2, 3, 4, 8):
            cum = np.cumsum(a.fam == c)
            cnt = np.clip(cum[hi1] - cum[lo], 0, 2).astype(np.uint64)
            T.append(_hashed(40 + c, tH, tD, dr, cnt))
        for k in range(n_tags):
            T.append(_hashed(50, tD, u(k), np.minimum(ch[D, k + 4], 2).astype(np.uint64)))
        for k in range(n_tags):
            T.append(_hashed(51, tD, tH, u(k), np.minimum(ch[D, k + 4], 1).astype(np.uint64)))
        if self.use_morph and a.x:
            mH = _hashed(0, *[v[H] for v in a.x])
            mD = _hashed(0, *[v[D] for v in a.x])
            T.append(_hashed(60, mD, tD, tH))
            T.append(_hashed(61, mH, mD, tD, tH))
            T.append(_hashed(62, mD, tD, dr, db))
        F = np.empty((len(T), n), dtype=np.int64)
        for k, h in enumerate(T):
            F[k] = (h >> sh).astype(np.int64)
        return F

    def fit(self, sents: list[_Sent], heads: list[np.ndarray], rels: list[list[str]], n_tags: int) -> "_Labeller":
        from sklearn.svm import LinearSVC
        self.n_tags = n_tags
        cols = [self.features(a, h, n_tags).T for a, h in zip(sents, heads)]
        Fm = np.concatenate(cols, axis=0)                       # (n_arcs, T)
        y = np.asarray([r for rs in rels for r in rs])
        self.classes_ = np.asarray(sorted(set(y.tolist())))
        self.cols_ = np.unique(Fm)
        X = self._csr(Fm)
        if len(self.classes_) == 1:
            self.model_ = None
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model_ = LinearSVC(C=self.C, random_state=self.seed, max_iter=200).fit(X, y)
        return self

    def _csr(self, Fm: np.ndarray):
        from scipy.sparse import csr_matrix
        pos = np.searchsorted(self.cols_, Fm)
        pos_c = np.minimum(pos, len(self.cols_) - 1)
        ok = self.cols_[pos_c] == Fm
        T = Fm.shape[1]
        indptr = np.concatenate([[0], np.cumsum(ok.sum(axis=1))])
        return csr_matrix((np.ones(int(ok.sum()), dtype=np.float32), pos_c[ok].astype(np.int32), indptr),
                          shape=(Fm.shape[0], len(self.cols_)))

    def predict(self, a: _Sent, heads: np.ndarray) -> list[str]:
        n = a.n
        if self.model_ is None:
            lab = [str(self.classes_[0])] * n
        else:
            X = self._csr(self.features(a, heads, self.n_tags).T)
            D = self.model_.decision_function(X)
            if D.ndim == 1:
                D = np.stack([-D, D], axis=1)
            classes = list(self.model_.classes_)
            is_root = np.asarray(heads) == 0
            if "root" in classes:
                r = classes.index("root")
                Dm = D.copy()
                Dm[:, r] = -np.inf
                lab = [classes[int(np.argmax(Dm[i]))] if not is_root[i] else "root" for i in range(n)]
                if len(classes) == 1:
                    lab = ["root" if x else "dep" for x in is_root]
            else:
                lab = [classes[int(np.argmax(D[i]))] for i in range(n)]
        return ["root" if h == 0 else (l if l != "root" else "dep") for h, l in zip(heads, lab)]


# ----------------------------------------------------------------------------------------------- pipeline
_MORPH_KEYS = ("VerbForm", "Number", "Gender")


class UDParser:
    """POS tagger + edge-factored arc scorer + relation labeller (see the module docstring for the options)."""

    def __init__(self, decode: str = "eisner", epochs: int = 8, n_models: int = 1, jackknife_folds: int = 5,
                 tagger_C: float = _DEFAULT_TAG_C, label_C: float = 0.3, arc_C: float = 0.5, min_count: int = 2,
                 use_morph: bool = False, seed: int = 0):
        if decode not in ("eisner", "cle"):
            raise ValueError("decode must be 'eisner' or 'cle'")
        self.decode, self.epochs, self.n_models = decode, int(epochs), int(n_models)
        self.jackknife_folds, self.tagger_C, self.label_C, self.arc_C = int(jackknife_folds), tagger_C, label_C, arc_C
        self.min_count, self.use_morph, self.seed = int(min_count), bool(use_morph), int(seed)

    # -- training
    def fit(self, train: Sequence[dict]) -> "UDParser":
        t0 = time.process_time()
        train = [s for s in train if len(s["words"]) > 0]
        if not train:
            raise ValueError("no annotated training sentences")
        self.tagger_ = UPOSTagger(self.tagger_C, "upos", seed=self.seed).fit(train)
        gold_tags = [_column(s, "upos") for s in train]
        if self.jackknife_folds and self.jackknife_folds > 1:
            tags = self.tagger_.jackknife(n_folds=self.jackknife_folds)
        else:
            tags = gold_tags
        self.oof_upos_accuracy_ = float(np.mean([a == b for x, y in zip(tags, gold_tags) for a, b in zip(x, y)]))
        self.morph_taggers_ = []
        extras = None
        if self.use_morph:
            gold_x, oof_x = [], []
            for key in _MORPH_KEYS:
                tg = UPOSTagger(self.tagger_C, "feats", feat_key=key, seed=self.seed).fit(train)
                self.morph_taggers_.append(tg)
                gold_x.append([[tg._label(w) for w in s["words"]] for s in train])
                oof_x.append(tg.jackknife(n_folds=self.jackknife_folds) if self.jackknife_folds > 1 else gold_x[-1])
            extras = oof_x
        forms = [_forms(s) for s in train]
        heads = [np.asarray([int(w["head"]) for w in s["words"]], dtype=np.int64) for s in train]
        rels = [[_universal(str(w["deprel"])) for w in s["words"]] for s in train]
        self.lex_ = _Lexicon(forms, tags + gold_tags, self.min_count, extras)
        sents = [_Sent(self.lex_, f, t, [c[i] for c in extras] if extras else None)
                 for i, (f, t) in enumerate(zip(forms, tags))]
        self.scorers_ = [_ArcScorer(self.epochs, C=self.arc_C, seed=self.seed + k, use_morph=self.use_morph)
                         .fit(sents, heads) for k in range(self.n_models)]
        self.labeller_ = _Labeller(self.label_C, seed=self.seed, use_morph=self.use_morph).fit(
            sents, heads, rels, self.lex_.n_tags)
        self.fit_seconds_ = time.process_time() - t0
        return self

    # -- inference
    def tag(self, sents: Sequence[Any]) -> list[list[str]]:
        return self.tagger_.predict(sents)

    def _sents(self, sents: Sequence[Any], tags: list[list[str]] | None):
        tags = self.tag(sents) if tags is None else tags
        extras = None
        if self.use_morph:
            extras = [self.morph_taggers_[k].predict(sents) for k in range(len(_MORPH_KEYS))]
        return [_Sent(self.lex_, _forms(s), t, [c[i] for c in extras] if extras else None)
                for i, (s, t) in enumerate(zip(sents, tags))]

    def arc_scores(self, sents: Sequence[Any], tags: list[list[str]] | None = None) -> list[np.ndarray]:
        out = []
        for a in self._sents(sents, tags):
            S = np.mean([sc.scores(a) for sc in self.scorers_], axis=0)
            out.append(S)
        return out

    def parse(self, sents: Sequence[Any], tags: list[list[str]] | None = None) -> list[dict]:
        sa = self._sents(sents, tags)
        out = []
        for a in sa:
            if a.n == 0:
                out.append({"head": [], "deprel": []})
                continue
            S = np.mean([sc.scores(a) for sc in self.scorers_], axis=0)
            if a.n == 1:
                heads = [0]
            else:
                heads = _tree_or_chain((eisner if self.decode == "eisner" else chu_liu_edmonds)(S))
            rels = self.labeller_.predict(a, np.asarray(heads))
            out.append({"head": heads, "deprel": rels})
        return out


def fit_predict(train: Sequence[dict], targets: Sequence[Sequence[Any]], pretrained: bool = False,
                **kwargs) -> list[list[dict]]:
    """``UDParser(**kwargs).fit(train)`` and one ``parse`` call per element of ``targets`` (each a list of sentences).
    With ``pretrained=True`` every target list is parsed by ``scilib.udparse_pretrained.parse_gold_tokens`` instead (French
    only; ``train`` and ``kwargs`` are not used); the output has the same ``{"head", "deprel"}`` format."""
    if pretrained:
        from . import udparse_pretrained as _upp
        if not _upp.available():
            raise RuntimeError("fit_predict(pretrained=True): the pretrained parser is not available here")
        return [[{"head": p["head"], "deprel": p["deprel"]} for p in _upp.parse_gold_tokens(t)] for t in targets]
    model = UDParser(**kwargs).fit(train)
    return [model.parse(t) for t in targets]


def cross_validate(train: Sequence[dict], n_folds: int = 3, gold_tags: bool = False, seed: int = 0,
                   max_sentences: int | None = None, **kwargs) -> dict:
    """k-fold cross-validation of ``UDParser(**kwargs)`` over annotated sentences (folds of whole sentences, seeded).
    With ``gold_tags=True`` the parser is applied to the gold UPOS tags (measures the parser alone)."""
    train = list(train)
    if max_sentences:
        train = train[:max_sentences]
    n = len(train)
    fold = np.random.default_rng(seed).permutation(n) % n_folds
    per, tot_l, tot_u, tot_n, tag_ok, tag_n = [], 0.0, 0.0, 0, 0, 0
    for f in range(n_folds):
        tr = [train[i] for i in range(n) if fold[i] != f]
        te = [train[i] for i in range(n) if fold[i] == f]
        model = UDParser(seed=seed, **kwargs).fit(tr)
        tags = [_column(s, "upos") for s in te] if gold_tags else None
        pred = model.parse(te, tags)
        sc = las_uas(te, pred)
        pt = model.tag(te)
        ok = sum(a == b for x, y in zip(pt, [_column(s, "upos") for s in te]) for a, b in zip(x, y))
        nw = sc["n_words"]
        per.append({"las": sc["las"], "uas": sc["uas"], "upos_accuracy": ok / nw, "n_words": nw})
        tot_l += sc["las"] * nw
        tot_u += sc["uas"] * nw
        tot_n += nw
        tag_ok += ok
    return {"las": tot_l / tot_n, "uas": tot_u / tot_n, "upos_accuracy": tag_ok / tot_n, "n_words": tot_n, "per_fold": per}
