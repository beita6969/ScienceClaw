"""Tools for evidence identification in non-disclosure agreements (ContractNLI).

fit_predict(train_documents, train_annotations, hypotheses, items, documents, seed, n_jobs, **kw) -> list[dict]
    one call: fit ``ContractModel`` on the training contracts, predict ``[{'label', 'span_scores'}]`` for ``items`` (same order)
cross_validate(train_documents, train_annotations, hypotheses, n_folds, seed, **kw) -> dict
    contract-grouped k-fold of the whole pipeline on the annotated training pairs: map, p_at_r80, nli_binary_accuracy,
    per_hypothesis_map, aps, n_pairs
ContractModel(C, n_folds, context, char_weight, stage2, use_not_mentioned, hyp_id, seed, n_jobs, n_estimators, plm, emb_C)
    .fit(train_documents, train_annotations, hypotheses) -> self;  .predict(items, documents) -> list[dict];
    .predict_scores(items, documents) -> list[np.ndarray] (raw span scores);  .stage1_oof_map_ -> {hypothesis_key: mean AP}
LabelModel(C, min_minority, top_k, rel_threshold)   Entailment / Contradiction from evidence text, one model per hypothesis
Featurizer(word_ngrams, char_ngrams, min_df, ...)   .fit(documents) / .transform(documents) -> dict of sparse span features;
    .query_scores(feats, hypothesis_text) -> dict of per-span similarity to a hypothesis (cos_uni, cos_char, bm25, overlap, idf_cov)
tokenize(text, stopwords)                            lower-case word/number tokens with a light suffix stemmer
structure_features(span_texts) -> (n, 14) array      columns ``STRUCT_COLUMNS`` (length, case, digits, heading/list markers, negations, ...)
gold_vector(n_spans, evidence_spans) -> 0/1 array    span indicator vector of one pair
average_precision(gold, scores) -> float             per-pair AP (sklearn ``average_precision_score``), as the task metric uses it
precision_at_recall(gold, scores, recall) -> float   official precision at the given recall for one pair (NaN without gold spans)
evidence_map(gold_list, score_list) -> float         mean per-pair AP
evaluate_predictions(predictions, gold_list, labels) -> dict   map, p_at_r80 (mean over pairs with gold spans), nli_binary_accuracy, aps
annotation_pairs(train_documents, train_annotations) -> (items, gold_list, labels)   labelled pairs of the training annotations
PLM_GROUPS                                           names accepted by ``ContractModel(plm=...)``

The functions are plain Python, called inside a ``code`` node (they are not tool nodes); the ``load_*`` tool outputs arrive as node inputs::

    from scilib.contracts import fit_predict
    y = fit_predict(train_documents, train_annotations, hypotheses, items, documents)     # items / documents: dev or eval

Inputs are the rows of the task's tools: ``documents`` / ``train_documents`` = list of dicts with ``doc_id`` and ``span_texts``
(or ``text`` + character ``spans``); ``train_annotations`` = dicts with ``doc_id``, ``hypothesis_key``, ``label``
('Entailment' | 'Contradiction' | 'NotMentioned') and ``evidence_spans`` (span indices); ``hypotheses`` = dict key -> dict with 'hypothesis' (text) and 'short_description';
``items`` = dicts with ``doc_id``, ``hypothesis_key`` and optionally ``hypothesis`` and ``n_spans`` (checked against the document).
The prediction for an item is ``{'label': 'Entailment' | 'Contradiction', 'span_scores': [float in [0, 1]] * n_spans}``.

ContractModel, in two stages, for the pairs annotated Entailment / Contradiction with at least one evidence span:
 1. per hypothesis, a class-balanced logistic regression on word 1-2-gram TF-IDF of a span plus half-weighted TF-IDF of the previous
    and next span; scores of the training pairs are out-of-fold over contract-grouped folds (``n_folds``);
 2. one LightGBM classifier over all hypotheses (span-level, ``n_estimators`` trees) on features of every span of a pair: similarity
    to the hypothesis text (unigram / character TF-IDF cosine, BM25, term overlap, each also as within-contract rank and gap to the
    contract's maximum), structure features and position, stage-1 score with rank and the scores of the 1-2 neighbouring spans,
    similarity to the training evidence spans of that hypothesis, hypothesis index.
A hypothesis without training pairs is scored by a second LightGBM that uses only the hypothesis-independent features.
``plm`` (default False; True or a subset of ``PLM_GROUPS`` = rerank, nli, cos, emb_lr, emb_knn) adds features from ``scilib.textenc`` (needs
``scilib.textenc.available()``): rerank = cross-encoder relevance logit of (hypothesis, span) with within-contract rank, gap to the maximum
and the neighbouring spans' values; nli = log-probabilities of contradiction / entailment / neutral of (span, hypothesis) and the log-sum
of the first two, ranked; cos = cosine of the sentence embeddings of span and hypothesis (ranked, with neighbours); emb_lr = out-of-fold
score of a per-hypothesis logistic regression (``emb_C``) on the span embedding, ranked, with neighbours; emb_knn = maximum and mean of the
three largest cosines to the training evidence spans of the hypothesis (other contracts only). Every span goes through the textenc models
once per fit / prediction call (hypotheses x spans pairs per cross-encoder; 40 contracts and 17 hypotheses = about 54,000 pairs each).
Without ``plm`` nothing from textenc is called.
The label of a pair is decided by ``LabelModel`` from the text of the top-scoring span(s): a per-hypothesis logistic regression on
word 1-3-grams and negation / condition cue counts; a hypothesis whose training pairs have (almost) one label always gets that label.
Stage-1 fits use scipy sparse matrices and liblinear; a fit on 40 contracts (about 3000 spans, 17 hypotheses) takes about 20 s and a
prediction for 16 pairs about 2 s with 2 threads on an idle CPU. Everything is deterministic given ``seed``.
"""
from __future__ import annotations

import collections
import re

import numpy as np

__all__ = ["fit_predict", "cross_validate", "ContractModel", "LabelModel", "Featurizer", "tokenize", "structure_features",
           "STRUCT_COLUMNS", "gold_vector", "average_precision", "precision_at_recall", "evidence_map", "evaluate_predictions",
           "annotation_pairs", "PLM_GROUPS"]

# ------------------------------------------------------------------------------------------------ metrics
LABELS = ("Entailment", "Contradiction")


def gold_vector(n_spans, evidence_spans) -> np.ndarray:
    y = np.zeros(int(n_spans), dtype=int)
    idx = [int(i) for i in evidence_spans]
    if idx:
        y[idx] = 1
    return y


def average_precision(gold, scores) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(np.asarray(gold), np.asarray(scores, dtype=float)))


def precision_at_recall(gold, scores, recall: float = 0.8) -> float:
    from sklearn.metrics import precision_score
    y_true = np.asarray(gold)
    y_prob = np.asarray(scores, dtype=float)
    if len(y_true) == 0 or np.sum(y_true) == 0:
        return float("nan")
    threshs = np.sort(np.unique(y_prob))[::-1]
    y_preds = y_prob[None, :] >= threshs[:, None]
    recalls = np.logical_and(y_true[None, :], y_preds).sum(axis=1) / np.sum(y_true)
    thresh = threshs[np.where(recalls >= recall)[0][0]]
    return float(precision_score(y_true, y_prob >= thresh, zero_division=0.0))


# ------------------------------------------------------------------------------------------------ text
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+")
_SUFFIXES = ("ations", "ation", "ingly", "ments", "ment", "ings", "ing", "ies", "ied", "ers", "ed", "es", "ly", "s")
STOPWORDS = frozenset(
    "a an the of to in on for by with and or any all some shall may not be is are was were that this which from as at "
    "its it their such other than under upon each has have had been if then there these those herein hereof "
    "thereof hereby hereto will would".split())


def stem(w: str) -> str:
    if len(w) > 5:
        for suf in _SUFFIXES:
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                return w[: len(w) - len(suf)]
    return w


def tokenize(text: str, stopwords: bool = False) -> list[str]:
    toks = [stem(w) for w in _WORD.findall(text.lower())]
    if stopwords:
        toks = [t for t in toks if t not in STOPWORDS and len(t) > 2]
    return toks


def _tok_all(text: str) -> list[str]:
    return tokenize(text, False)


def _tok_content(text: str) -> list[str]:
    return tokenize(text, True)


# ------------------------------------------------------------------------------------------------ span structure
_NEG = re.compile(r"\b(?:not|no|never|nor|neither|without|except|unless|excluding|prohibit\w*|cannot)\b", re.I)
_MOD = re.compile(r"\b(?:shall|must|will|may|might|can|could|should)\b", re.I)
_LIST = re.compile(r"^\s*(?:\(?[a-zA-Z0-9]{1,4}[\).]|[-*•])\s")
_CONF = re.compile(r"confidential", re.I)
_DEF = re.compile(r"\bmeans?\b|\bincludes?\b", re.I)
STRUCT_COLUMNS = ("log_chars", "n_words", "upper_ratio", "digit_ratio", "heading_like", "ends_colon", "list_marker",
                  "n_negations", "n_modals", "n_confidential", "ends_semicolon_comma", "starts_lowercase",
                  "n_commas_semicolons", "definition_like")


def structure_features(span_texts) -> np.ndarray:
    """(n, 14) array with the columns ``STRUCT_COLUMNS`` for a list of span texts (no context)."""
    n = len(span_texts)
    F = np.zeros((n, len(STRUCT_COLUMNS)))
    for i, t in enumerate(span_texts):
        s = t.strip()
        nt = max(len(s.split()), 1)
        letters = [c for c in s if c.isalpha()]
        F[i] = (np.log1p(len(s)), nt,
                (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0,
                sum(c.isdigit() for c in s) / max(len(s), 1),
                float(nt <= 10 and not s.endswith((".", ";"))), float(s.endswith(":")), float(bool(_LIST.match(t))),
                len(_NEG.findall(s)), len(_MOD.findall(s)), len(_CONF.findall(s)),
                float(s.endswith((";", ","))), float(s[:1].islower()), s.count(",") + s.count(";"),
                float(bool(_DEF.search(s))))
    return F


# ------------------------------------------------------------------------------------------------ inputs
def _doc_texts(d: dict) -> list[str]:
    if not isinstance(d, dict) or "doc_id" not in d:
        raise ValueError("every document must be a dict with a 'doc_id' key (rows of the load_* document tables)")
    if "span_texts" in d:
        return [str(t) for t in d["span_texts"]]
    if "text" in d and "spans" in d:
        return [d["text"][int(a):int(b)] for a, b in d["spans"]]
    raise ValueError(f"document {d['doc_id']!r} needs 'span_texts' (list of span strings) or 'text' + 'spans'")


class _Corpus:
    """Stacked spans of a set of documents: global row index = offset[doc_id] + span index."""

    def __init__(self, documents):
        self.doc_ids: list[str] = []
        self.offset: dict[str, int] = {}
        self.n_spans: dict[str, int] = {}
        texts: list[str] = []
        for d in documents:
            did = str(d["doc_id"])
            if did in self.offset:
                continue
            t = _doc_texts(d)
            self.doc_ids.append(did)
            self.offset[did] = len(texts)
            self.n_spans[did] = len(t)
            texts.extend(t)
        self.texts = texts
        self.n = len(texts)
        self.doc_index = np.concatenate([np.full(self.n_spans[d], k) for k, d in enumerate(self.doc_ids)]) \
            if self.doc_ids else np.zeros(0, int)
        self.pos_in_doc = np.concatenate([np.arange(self.n_spans[d]) for d in self.doc_ids]) if self.doc_ids \
            else np.zeros(0, int)
        self.prev_row = np.where(self.pos_in_doc > 0, np.arange(self.n) - 1, self.n)      # self.n = "zero row"
        last = np.concatenate([np.full(self.n_spans[d], self.n_spans[d] - 1) for d in self.doc_ids]) \
            if self.doc_ids else np.zeros(0, int)
        self.next_row = np.where(self.pos_in_doc < last, np.arange(self.n) + 1, self.n)

    def rows(self, doc_id: str) -> np.ndarray:
        o = self.offset[doc_id]
        return np.arange(o, o + self.n_spans[doc_id])


# ------------------------------------------------------------------------------------------------ features
def _shift(X, idx, n):
    """Rows of sparse X selected by idx, where idx == n selects an all-zero row."""
    import scipy.sparse as sp
    Xz = sp.vstack([X, sp.csr_matrix((1, X.shape[1]))], format="csr")
    return Xz[idx]


def _hyp_text(h) -> str:
    if isinstance(h, dict):
        return str(h.get("hypothesis", ""))
    return str(h)


class Featurizer:
    """Vocabulary / IDF statistics fitted on the spans of a corpus, then applied to any corpus."""

    def __init__(self, word_ngrams=(1, 2), char_ngrams=(3, 5), min_df=2, max_char_features=60000,
                 bm25_k1=1.5, bm25_b=0.75):
        self.word_ngrams, self.char_ngrams, self.min_df = tuple(word_ngrams), tuple(char_ngrams), int(min_df)
        self.max_char_features, self.k1, self.b = int(max_char_features), float(bm25_k1), float(bm25_b)

    def fit(self, documents) -> "Featurizer":
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        corpus = documents if isinstance(documents, _Corpus) else _Corpus(documents)
        t = corpus.texts
        self.word_ = TfidfVectorizer(tokenizer=_tok_all, lowercase=False, token_pattern=None, ngram_range=self.word_ngrams,
                                     min_df=self.min_df, sublinear_tf=True, dtype=np.float32).fit(t)
        self.char_ = TfidfVectorizer(analyzer="char_wb", ngram_range=self.char_ngrams, min_df=3, sublinear_tf=True,
                                     max_features=self.max_char_features, dtype=np.float32).fit(t)
        self.uni_ = TfidfVectorizer(tokenizer=_tok_content, lowercase=False, token_pattern=None, min_df=1,
                                    sublinear_tf=True, dtype=np.float32).fit(t)
        self.count_ = CountVectorizer(tokenizer=_tok_content, lowercase=False, token_pattern=None, dtype=np.float32).fit(t)
        C = self.count_.transform(t)
        df = np.asarray((C > 0).sum(0)).ravel()
        N = C.shape[0]
        self.idf_ = np.log(1.0 + (N - df + 0.5) / (df + 0.5))
        self.avgdl_ = max(float(C.sum()) / max(N, 1), 1.0)
        return self

    def transform(self, documents) -> dict:
        corpus = documents if isinstance(documents, _Corpus) else _Corpus(documents)
        t = corpus.texts
        C = self.count_.transform(t).tocsr()
        return {"word": self.word_.transform(t).tocsr(), "char": self.char_.transform(t).tocsr(),
                "uni": self.uni_.transform(t).tocsr(), "count": C,
                "len": np.asarray(C.sum(1)).ravel(), "struct": structure_features(t)}

    def query_scores(self, feats: dict, hypothesis: str) -> dict:
        """Similarity of every span of a transformed corpus to one hypothesis text."""
        q_uni = self.uni_.transform([hypothesis])
        q_char = self.char_.transform([hypothesis])
        cos_uni = np.asarray((feats["uni"] @ q_uni.T).todense()).ravel()
        cos_char = np.asarray((feats["char"] @ q_char.T).todense()).ravel()
        q_cnt = self.count_.transform([hypothesis])
        terms = q_cnt.indices
        C = feats["count"]
        if len(terms):
            tf = C[:, terms].toarray()
            norm = self.k1 * (1 - self.b + self.b * feats["len"] / self.avgdl_)
            bm25 = ((tf * (self.k1 + 1)) / (tf + norm[:, None] + 1e-9) * self.idf_[terms][None, :]).sum(1)
            overlap = (tf > 0).sum(1) / len(terms)
            idf_cov = ((tf > 0) * self.idf_[terms][None, :]).sum(1) / max(self.idf_[terms].sum(), 1e-9)
        else:
            bm25 = np.zeros(C.shape[0]); overlap = bm25; idf_cov = bm25
        return {"cos_uni": cos_uni, "cos_char": cos_char, "bm25": bm25, "overlap": overlap, "idf_cov": idf_cov}


# ------------------------------------------------------------------------------------------------ model
def _rank_pct(v: np.ndarray) -> np.ndarray:
    """0 for the highest value of the vector, 1 for the lowest (NaN-safe)."""
    v = np.nan_to_num(np.asarray(v, dtype=float), nan=-1e9)
    r = np.argsort(np.argsort(-v, kind="stable"), kind="stable")
    return r / max(len(v) - 1, 1)


def _norm_annotations(train_annotations, min_evidence=1):
    out = []
    for a in train_annotations:
        lab = a.get("label")
        ev = [int(i) for i in (a.get("evidence_spans") or [])]
        out.append({"doc_id": str(a["doc_id"]), "key": str(a["hypothesis_key"]), "label": lab, "evidence": ev})
    return out


PLM_GROUPS = ("rerank", "nli", "cos", "emb_lr", "emb_knn")


def _plm_groups(plm) -> tuple:
    """``plm`` = False / True (all groups) / an iterable of names from ``PLM_GROUPS``."""
    if plm is None or plm is False:
        return ()
    if plm is True:
        return PLM_GROUPS
    g = tuple(plm)
    bad = [x for x in g if x not in PLM_GROUPS]
    if bad:
        raise ValueError(f"plm groups must be among {PLM_GROUPS}, got {bad}")
    return g


class _PlmTables:
    """Pretrained-model scores of every span of a corpus against a list of hypothesis texts (``scilib.textenc``)."""

    def __init__(self, corpus, hyp_texts, groups):
        from . import textenc
        self.hyp_col = {t: j for j, t in enumerate(hyp_texts)}
        self.rr = self.nli = self.cos = self.emb = None
        if "rerank" in groups:
            self.rr = textenc.relevance(corpus.texts, hyp_texts)
        if "nli" in groups:
            self.nli = textenc.nli(corpus.texts, hyp_texts)
        if "cos" in groups or "emb_lr" in groups or "emb_knn" in groups:
            self.emb = textenc.embed(corpus.texts)
            if "cos" in groups:
                self.cos = self.emb @ textenc.embed(hyp_texts, query=True).T

    def block(self, rows, text):
        """dict of per-span arrays of one hypothesis text for the given corpus rows."""
        j = self.hyp_col[text]
        out = {}
        if self.rr is not None:
            out["rr"] = self.rr[rows, j]
        if self.nli is not None:
            out["nli"] = self.nli[rows, j]
        if self.cos is not None:
            out["cos"] = self.cos[rows, j]
        return out


class ContractModel:
    """Two-stage evidence scorer + label classifier (see the module docstring)."""

    def __init__(self, C: float = 10.0, n_folds: int = 5, context: float = 0.5, char_weight: float = 0.0,
                 stage2: str = "lgbm", use_not_mentioned: bool = False, hyp_id: bool = True, seed: int = 0,
                 n_jobs: int = 2, n_estimators: int = 200, plm=False, emb_C: float = 10.0):
        self.plm = _plm_groups(plm)
        self.emb_C = float(emb_C)
        self.C, self.n_folds, self.context, self.char_weight = float(C), int(n_folds), float(context), float(char_weight)
        self.stage2, self.use_not_mentioned, self.hyp_id = stage2, bool(use_not_mentioned), bool(hyp_id)
        self.seed, self.n_jobs, self.n_estimators = int(seed), int(n_jobs), int(n_estimators)

    # ---------------------------------------------------------------- helpers
    def _design(self, corpus, F):
        import scipy.sparse as sp
        blocks = [F["word"]]
        if self.context:
            blocks += [self.context * _shift(F["word"], corpus.prev_row, corpus.n),
                       self.context * _shift(F["word"], corpus.next_row, corpus.n)]
        if self.char_weight:
            blocks.append(self.char_weight * F["char"])
        return sp.hstack(blocks, format="csr")

    def _fit_lr(self, X, y):
        from sklearn.linear_model import LogisticRegression
        if y.sum() == 0 or y.sum() == len(y):
            return None
        return LogisticRegression(C=self.C, class_weight="balanced", solver="liblinear").fit(X, y)

    @staticmethod
    def _lr_score(m, X):
        return m.decision_function(X) if m is not None else np.full(X.shape[0], np.nan)

    def _hyp_index(self, key):
        return self.hyp_ids_.get(key, -1)

    def _pair_block(self, corpus, F, rows, key, s1, qs, knn, pl=None, e1=None, eknn=None):
        """Feature matrix (len(rows), n_feat) of one (document, hypothesis) pair; ``rows`` = the document's rows."""
        st = F["struct"][rows]
        n = len(rows)
        cols, names = [], []

        def add(name, v):
            cols.append(np.asarray(v, dtype=float).reshape(n)); names.append(name)

        def sh(v, k):                       # value of the span k positions before (k>0) / after (k<0), NaN outside
            v = np.asarray(v, dtype=float)
            out = np.full(n, np.nan)
            if k > 0 and n > k:
                out[k:] = v[:-k]
            elif k < 0 and n > -k:
                out[:k] = v[-k:]
            elif k == 0:
                out = v.copy()
            return out

        def add_ranked(name, v):
            v = np.asarray(v, dtype=float)
            top = np.nanmax(v) if np.isfinite(v).any() else 0.0
            add(name, v); add(name + "_rank", _rank_pct(v)); add(name + "_dmax", v - top)

        for nm in ("cos_uni", "cos_char", "bm25", "overlap", "idf_cov"):
            add_ranked(nm, qs[nm][rows])
        add("cos_uni_prev", sh(qs["cos_uni"][rows], 1)); add("cos_uni_next", sh(qs["cos_uni"][rows], -1))
        for j, nm in enumerate(STRUCT_COLUMNS):
            add(nm, st[:, j])
        for j in (4, 5, 6, 10):
            add("prev_" + STRUCT_COLUMNS[j], sh(st[:, j], 1)); add("next_" + STRUCT_COLUMNS[j], sh(st[:, j], -1))
        add("rel_pos", np.arange(n) / max(n - 1, 1))
        add("pos", np.log1p(np.arange(n)))
        add("log_n_spans", np.full(n, np.log(max(n, 1))))
        if "rerank" in self.plm:
            add_ranked("rr", pl["rr"]); add("rr_prev", sh(pl["rr"], 1)); add("rr_next", sh(pl["rr"], -1))
        if "nli" in self.plm:
            lp = pl["nli"]
            add("nli_con", lp[:, 0]); add("nli_ent", lp[:, 1]); add("nli_neu", lp[:, 2])
            add_ranked("nli_ev", np.logaddexp(lp[:, 0], lp[:, 1]))
        if "cos" in self.plm:
            add_ranked("emb_cos", pl["cos"]); add("emb_cos_prev", sh(pl["cos"], 1)); add("emb_cos_next", sh(pl["cos"], -1))
        n_free = len(names)
        if s1 is not None:
            add_ranked("s1", s1)
            for k in (1, -1, 2, -2):
                add(f"s1_sh{k}", sh(s1, k))
        if s1 is not None:
            nan = np.full(n, np.nan)
            add_ranked("knn_max", knn[0] if knn is not None else nan); add("knn_top3", knn[1] if knn is not None else nan)
        if s1 is not None and "emb_lr" in self.plm:
            nan = np.full(n, np.nan)
            e = e1 if e1 is not None else nan
            add_ranked("e1", e); add("e1_prev", sh(e, 1)); add("e1_next", sh(e, -1))
        if s1 is not None and "emb_knn" in self.plm:
            nan = np.full(n, np.nan)
            add_ranked("eknn_max", eknn[0] if eknn is not None else nan); add("eknn_top3", eknn[1] if eknn is not None else nan)
        if self.hyp_id and s1 is not None:
            add("hyp_id", np.full(n, self._hyp_index(key)))
        self.n_free_ = n_free
        self.feature_names_ = names
        return np.column_stack(cols)

    def _knn(self, X_query, key, exclude_docs=None):
        """max and top-3 mean cosine similarity of query rows to the training evidence spans of ``key``."""
        E = self.evi_.get(key)
        if E is None:
            return None
        rows, docs = E
        keep = np.ones(len(rows), bool) if exclude_docs is None else ~np.isin(docs, list(exclude_docs))
        if keep.sum() == 0:
            return None
        S = (X_query @ self.F_["uni"][rows[keep]].T).toarray()
        top = -np.sort(-S, axis=1)[:, :3]
        return S.max(1), top.mean(1)

    def _eknn(self, E_query, key, exclude_docs=None):
        """max and top-3 mean cosine of query span embeddings to the training evidence spans of ``key`` (sentence-encoder space)."""
        ev = self.evi_.get(key)
        if ev is None:
            return None
        rows, docs = ev
        keep = np.ones(len(rows), bool) if exclude_docs is None else ~np.isin(docs, list(exclude_docs))
        if keep.sum() == 0:
            return None
        S = E_query @ self.plm_emb_[rows[keep]].T
        top = -np.sort(-S, axis=1)[:, :3]
        return S.max(1), top.mean(1)

    def _fit_emb_lr(self, E, y):
        from sklearn.linear_model import LogisticRegression
        if y.sum() == 0 or y.sum() == len(y):
            return None
        return LogisticRegression(C=self.emb_C, class_weight="balanced", max_iter=300).fit(E, y)

    # ---------------------------------------------------------------- fit
    def fit(self, train_documents, train_annotations, hypotheses):
        docs = list(train_documents)
        if not docs:
            raise ValueError("train_documents is empty")
        corpus = _Corpus(docs)
        self.corpus_ = corpus
        self.feat_ = Featurizer().fit(corpus)
        F = self.feat_.transform(corpus)
        self.F_ = F
        X1 = self._design(corpus, F)
        self.X1_ = X1
        hyp_text = {str(k): _hyp_text(v) for k, v in dict(hypotheses).items()}
        self.hyp_text_ = hyp_text
        self.hyp_ids_ = {k: i for i, k in enumerate(sorted(hyp_text))}
        ann = [a for a in _norm_annotations(train_annotations) if a["doc_id"] in corpus.offset]
        pairs = [a for a in ann if a["label"] in LABELS and a["evidence"]]
        if not pairs:
            raise ValueError("no annotated Entailment/Contradiction pairs with evidence spans found in train_annotations "
                             "for the given train_documents")
        for a in pairs:
            n_doc = corpus.n_spans[a["doc_id"]]
            if min(a["evidence"]) < 0 or max(a["evidence"]) >= n_doc:
                raise ValueError(f"evidence span index out of range for document {a['doc_id']!r} ({n_doc} spans)")
        self.pairs_ = pairs
        # evidence rows per hypothesis (for kNN features)
        ev = collections.defaultdict(lambda: ([], []))
        for a in pairs:
            r = corpus.offset[a["doc_id"]] + np.array(a["evidence"], dtype=int)
            ev[a["key"]][0].extend(r.tolist()); ev[a["key"]][1].extend([a["doc_id"]] * len(r))
        self.evi_ = {k: (np.array(v[0]), np.array(v[1])) for k, v in ev.items()}
        # folds by document
        ids = sorted(corpus.doc_ids)
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(ids))
        k_f = max(2, min(self.n_folds, len(ids)))
        fold_of = {ids[j]: int(i % k_f) for i, j in enumerate(perm)}
        self.fold_of_ = fold_of
        by_h = collections.defaultdict(list)
        for a in ann:
            if a["label"] in LABELS and a["evidence"]:
                by_h[a["key"]].append(a)
            elif a["label"] == "NotMentioned" and self.use_not_mentioned:
                by_h[a["key"]].append({**a, "evidence": []})
        self.by_h_ = by_h

        def train_rows(key, exclude_fold=None):
            rr, yy = [], []
            for a in by_h[key]:
                if exclude_fold is not None and fold_of[a["doc_id"]] == exclude_fold:
                    continue
                r = corpus.rows(a["doc_id"])
                rr.append(r); yy.append(gold_vector(len(r), a["evidence"]))
            if not rr:
                return None, None
            return np.concatenate(rr), np.concatenate(yy)

        self.plm_tabs_, self.plm_emb_ = None, None
        if self.plm:
            texts = sorted(set(hyp_text.values()) | {hyp_text.get(a["key"], "") for a in pairs})
            self.plm_tabs_ = _PlmTables(corpus, texts, self.plm)
            self.plm_emb_ = self.plm_tabs_.emb
        # stage 1: per-hypothesis linear models (final + out-of-fold scores of the training pairs)
        self.lr_ = {}
        self.elr_ = {}
        oof = {}                                   # (doc, key) -> scores
        oof_e = {}
        for key in by_h:
            r, y = train_rows(key)
            if r is not None:
                self.lr_[key] = self._fit_lr(X1[r], y)
                if "emb_lr" in self.plm:
                    self.elr_[key] = self._fit_emb_lr(self.plm_emb_[r], y)
            for f in range(k_f):
                tgt = [a for a in pairs if a["key"] == key and fold_of[a["doc_id"]] == f]
                if not tgt:
                    continue
                r, y = train_rows(key, f)
                m = self._fit_lr(X1[r], y) if r is not None else None
                me = self._fit_emb_lr(self.plm_emb_[r], y) if (r is not None and "emb_lr" in self.plm) else None
                for a in tgt:
                    rr_ = corpus.rows(a["doc_id"])
                    oof[(a["doc_id"], key)] = self._lr_score(m, X1[rr_])
                    if "emb_lr" in self.plm:
                        oof_e[(a["doc_id"], key)] = me.decision_function(self.plm_emb_[rr_]) if me is not None else None
        self.oof_s1_ = oof
        by_key_ap = collections.defaultdict(list)
        for a in pairs:
            o = oof[(a["doc_id"], a["key"])]
            if np.isfinite(o).all():
                by_key_ap[a["key"]].append(average_precision(gold_vector(len(o), a["evidence"]), o))
        self.stage1_oof_map_ = {k: float(np.mean(v)) for k, v in sorted(by_key_ap.items())}
        # stage 2 training matrix
        qs_cache = {}
        Xs, ys, Xs_free, groups = [], [], [], []
        for a in pairs:
            key = a["key"]
            if key not in qs_cache:
                qs_cache[key] = self.feat_.query_scores(F, hyp_text.get(key, ""))
            rows = corpus.rows(a["doc_id"])
            s1 = oof[(a["doc_id"], key)]
            knn = self._knn(F["uni"][rows], key, exclude_docs=[d for d in set(self.evi_.get(key, ([], []))[1])
                                                               if fold_of[d] == fold_of[a["doc_id"]]]) if key in self.evi_ else None
            pl = self.plm_tabs_.block(rows, hyp_text.get(key, "")) if self.plm else None
            eknn = None
            if "emb_knn" in self.plm and key in self.evi_:
                eknn = self._eknn(self.plm_emb_[rows], key, exclude_docs=[d for d in set(self.evi_[key][1])
                                                                          if fold_of[d] == fold_of[a["doc_id"]]])
            Xs.append(self._pair_block(corpus, F, rows, key, s1, qs_cache[key], knn, pl, oof_e.get((a["doc_id"], key)), eknn))
            ys.append(gold_vector(len(rows), a["evidence"]))
            groups.append(np.full(len(rows), len(groups)))
        self.qs_train_ = qs_cache
        ev_texts = [" ".join(corpus.texts[corpus.offset[a["doc_id"]] + i] for i in a["evidence"]) for a in pairs]
        self.label_model_ = LabelModel().fit(ev_texts, [a["key"] for a in pairs], [a["label"] for a in pairs])
        self.n_full_ = Xs[0].shape[1] if Xs else 0
        if not Xs:
            self.m2_ = None
            return self
        X = np.vstack(Xs); y = np.concatenate(ys)
        self.m2_ = self._fit_stage2(X, y, None)
        self.m2_free_ = self._fit_stage2(X[:, : self.n_free_], y, None)
        return self

    def _fit_stage2(self, X, y, groups):
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=self.n_estimators, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                               n_jobs=self.n_jobs, random_state=self.seed, verbose=-1, deterministic=True,
                               force_row_wise=True)
        m.fit(X, y)
        return m

    # ---------------------------------------------------------------- predict
    def _check_items(self, items, corpus):
        for i, it in enumerate(items):
            if "hypothesis_key" not in it or "doc_id" not in it:
                raise ValueError(f"item {i} needs 'doc_id' and 'hypothesis_key' (rows of the load_* item tables)")
            did = str(it["doc_id"])
            if did not in corpus.offset:
                raise ValueError(f"item {i}: doc_id {did!r} is not among the documents passed to predict")
            if it.get("n_spans") is not None and int(it["n_spans"]) != corpus.n_spans[did]:
                raise ValueError(f"item {i}: n_spans={it['n_spans']} but document {did!r} has {corpus.n_spans[did]} spans")
            key = str(it["hypothesis_key"])
            if not it.get("hypothesis") and key not in self.hyp_text_:
                raise ValueError(f"item {i}: hypothesis_key {key!r} is unknown and the item has no 'hypothesis' text")

    def predict_scores(self, items, documents):
        if getattr(self, "m2_", None) is None and not hasattr(self, "lr_"):
            raise RuntimeError("call fit(...) before predict")
        corpus = _Corpus(documents)
        items = list(items)
        self._check_items(items, corpus)
        F = self.feat_.transform(corpus)
        X1 = self._design(corpus, F)
        qs_cache = {}
        out = []
        item_text = [_hyp_text(it.get("hypothesis")) if it.get("hypothesis") else self.hyp_text_.get(str(it["hypothesis_key"]), "")
                     for it in items]
        tabs = _PlmTables(corpus, sorted(set(item_text)), self.plm) if (self.plm and items) else None
        for it, text in zip(items, item_text):
            key, did = str(it["hypothesis_key"]), str(it["doc_id"])
            ck = (key, text)
            if ck not in qs_cache:
                qs_cache[ck] = self.feat_.query_scores(F, text)
            rows = corpus.rows(did)
            m1 = self.lr_.get(key)
            if m1 is not None and self.m2_ is not None:
                s1 = self._lr_score(m1, X1[rows])
                knn = self._knn(F["uni"][rows], key)
                pl = tabs.block(rows, text) if tabs is not None else None
                e1 = None
                if "emb_lr" in self.plm and self.elr_.get(key) is not None:
                    e1 = self.elr_[key].decision_function(tabs.emb[rows])
                eknn = self._eknn(tabs.emb[rows], key) if "emb_knn" in self.plm else None
                Xp = self._pair_block(corpus, F, rows, key, s1, qs_cache[ck], knn, pl, e1, eknn)
                sc = self.m2_.predict_proba(Xp)[:, 1]
            else:
                pl = tabs.block(rows, text) if tabs is not None else None
                Xp = self._pair_block(corpus, F, rows, key, None, qs_cache[ck], None, pl)
                sc = self.m2_free_.predict_proba(Xp)[:, 1] if getattr(self, "m2_free_", None) is not None \
                    else qs_cache[ck]["overlap"][rows]
            out.append(sc)
        return out

    def predict(self, items, documents):
        scores = self.predict_scores(items, documents)
        texts = {str(d["doc_id"]): _doc_texts(d) for d in documents}
        out = []
        lm = getattr(self, "label_model_", None)
        for it, s in zip(items, scores):
            lab = "Entailment"
            if lm is not None and len(s):
                order = np.argsort(-s)
                top = [j for j in order[: lm.top_k] if s[j] >= lm.rel_threshold * s[order[0]]]
                lab = lm.predict(str(it["hypothesis_key"]), " ".join(texts[str(it["doc_id"])][j] for j in sorted(top)))
            out.append({"label": lab, "span_scores": [float(v) for v in np.clip(s, 0.0, 1.0)]})
        return out


# ------------------------------------------------------------------------------------------------ labels
_LABEL_CUES = ("not", "no", "never", "without", "except", "unless", "other than", "whether or not", "regardless",
               "provided", "however", "notwithstanding", "only", "solely", "any", "all", "may", "shall")


def _cue_counts(text: str) -> list[float]:
    t = " " + text.lower() + " "
    return [float(len(re.findall(r"\b" + re.escape(c) + r"\b", t))) for c in _LABEL_CUES]


class LabelModel:
    """Entailment vs Contradiction from the text of the evidence spans, one logistic regression per hypothesis
    (word 1-3-gram TF-IDF + cue-word counts); a hypothesis whose training pairs have a single label (or fewer than
    ``min_minority`` of the rarer label) always gets its majority label."""

    def __init__(self, C: float = 3.0, min_minority: int = 3, top_k: int = 2, rel_threshold: float = 0.5):
        self.C, self.min_minority, self.top_k, self.rel_threshold = float(C), int(min_minority), int(top_k), float(rel_threshold)

    def fit(self, evidence_texts, keys, labels):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        import scipy.sparse as sp
        self.prior_ = collections.Counter(labels)
        self.by_key_ = {}
        idx = collections.defaultdict(list)
        for i, k in enumerate(keys):
            idx[k].append(i)
        self.majority_ = {}
        self.models_ = {}
        for k, ii in idx.items():
            y = np.array([1 if labels[i] == "Contradiction" else 0 for i in ii])
            self.majority_[k] = "Contradiction" if y.sum() * 2 > len(y) else "Entailment"
            if min(y.sum(), len(y) - y.sum()) < self.min_minority:
                continue
            tx = [evidence_texts[i] for i in ii]
            vec = TfidfVectorizer(tokenizer=_tok_all, lowercase=False, token_pattern=None, ngram_range=(1, 3), min_df=1,
                                  sublinear_tf=True).fit(tx)
            X = sp.hstack([vec.transform(tx), sp.csr_matrix(np.log1p([_cue_counts(t) for t in tx]))], format="csr")
            self.models_[k] = (vec, LogisticRegression(C=self.C, class_weight="balanced", solver="liblinear").fit(X, y))
        return self

    def predict_proba_contradiction(self, key, text) -> float:
        import scipy.sparse as sp
        if key in self.models_:
            vec, m = self.models_[key]
            X = sp.hstack([vec.transform([text]), sp.csr_matrix(np.log1p([_cue_counts(text)]))], format="csr")
            return float(m.predict_proba(X)[0, 1])
        maj = self.majority_.get(key)
        if maj is None:
            maj = "Contradiction" if self.prior_["Contradiction"] > self.prior_["Entailment"] else "Entailment"
        return 1.0 if maj == "Contradiction" else 0.0

    def predict(self, key, text) -> str:
        return "Contradiction" if self.predict_proba_contradiction(key, text) > 0.5 else "Entailment"


# ------------------------------------------------------------------------------------------------ evaluation helpers
def evidence_map(gold_list, score_list) -> float:
    """Mean over pairs of ``average_precision(gold, scores)`` (the per-pair aggregation of the task metric)."""
    return float(np.mean([average_precision(g, s) for g, s in zip(gold_list, score_list)]))


def evaluate_predictions(predictions, gold_list, labels=None) -> dict:
    """Metrics of ``predictions`` ([{'label', 'span_scores'}]) against ``gold_list`` (0/1 span vectors, one per pair) and
    optionally ``labels`` ('Entailment' | 'Contradiction'): ``map``, ``p_at_r80`` (mean official precision at 80 % recall,
    pairs without gold spans skipped), ``nli_binary_accuracy`` (only when ``labels`` is given), ``aps`` (per pair)."""
    if len(predictions) != len(gold_list):
        raise ValueError(f"{len(predictions)} predictions for {len(gold_list)} gold vectors")
    aps = [average_precision(g, p["span_scores"]) for g, p in zip(gold_list, predictions)]
    pr = [precision_at_recall(g, p["span_scores"], 0.8) for g, p in zip(gold_list, predictions)]
    out = {"map": float(np.mean(aps)), "p_at_r80": float(np.nanmean(pr)), "aps": aps}
    if labels is not None:
        out["nli_binary_accuracy"] = float(np.mean([p.get("label") == l for p, l in zip(predictions, labels)]))
    return out


def annotation_pairs(train_documents, train_annotations):
    """Labelled (document, hypothesis) pairs of the training annotations that have a gold Entailment / Contradiction label
    and at least one evidence span: returns ``(items, gold_list, labels)`` with ``items`` = dicts (doc_id, hypothesis_key,
    n_spans), ``gold_list`` = 0/1 span vectors, ``labels`` = label strings."""
    n = {str(d["doc_id"]): len(_doc_texts(d)) for d in train_documents}
    items, gold, labels = [], [], []
    for a in _norm_annotations(train_annotations):
        if a["label"] in LABELS and a["evidence"] and a["doc_id"] in n:
            items.append({"doc_id": a["doc_id"], "hypothesis_key": a["key"], "n_spans": n[a["doc_id"]]})
            gold.append(gold_vector(n[a["doc_id"]], a["evidence"])); labels.append(a["label"])
    return items, gold, labels


def cross_validate(train_documents, train_annotations, hypotheses, n_folds: int = 3, seed: int = 0, **model_kwargs) -> dict:
    """Document-grouped cross-validation of the whole pipeline on the annotated training contracts: every fold is fitted
    on the other contracts and predicts the labelled pairs of the held-out ones. Returns ``evaluate_predictions`` output
    plus ``n_pairs`` and ``per_hypothesis_map``."""
    items, gold, labels = annotation_pairs(train_documents, train_annotations)
    docs = {str(d["doc_id"]): d for d in train_documents}
    ids = sorted({it["doc_id"] for it in items})
    if len(ids) < 2:
        raise ValueError("cross_validate needs labelled pairs in at least 2 contracts")
    k = max(2, min(int(n_folds), len(ids)))
    perm = np.random.default_rng(seed).permutation(len(ids))
    fold = {ids[j]: i % k for i, j in enumerate(perm)}
    preds = [None] * len(items)
    for f in range(k):
        tr_docs = [d for i, d in docs.items() if fold.get(i, -1) != f]
        tr_ids = {str(d["doc_id"]) for d in tr_docs}
        ann = [a for a in train_annotations if str(a["doc_id"]) in tr_ids]
        te = [j for j, it in enumerate(items) if fold[it["doc_id"]] == f]
        model = ContractModel(seed=seed, **model_kwargs).fit(tr_docs, ann, hypotheses)
        out = model.predict([items[j] for j in te], [docs[i] for i in sorted({items[j]["doc_id"] for j in te})])
        for j, p in zip(te, out):
            preds[j] = p
    res = evaluate_predictions(preds, gold, labels)
    by_h = collections.defaultdict(list)
    for it, ap in zip(items, res["aps"]):
        by_h[it["hypothesis_key"]].append(ap)
    res["per_hypothesis_map"] = {k_: float(np.mean(v)) for k_, v in sorted(by_h.items())}
    res["n_pairs"] = len(items)
    return res


def fit_predict(train_documents, train_annotations, hypotheses, items, documents, seed: int = 0, n_jobs: int = 2,
                **model_kwargs) -> list[dict]:
    """``ContractModel(...).fit(...).predict(items, documents)``: one ``{'label', 'span_scores'}`` dict per item."""
    return ContractModel(seed=seed, n_jobs=n_jobs, **model_kwargs).fit(
        train_documents, train_annotations, hypotheses).predict(items, documents)
