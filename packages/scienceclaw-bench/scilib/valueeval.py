"""Tools for multi-label human-value detection in arguments (SemEval-2023 Task 4 / Touche23-ValueEval).

official_f1(y_true, y_pred) -> float            official metric of the task (see ``official_f1_report``)
official_f1_report(y_true, y_pred) -> dict      f1, macro precision, macro recall, mean per-category F1, #scored categories
argument_text(df) -> np.ndarray[str]            stance token + premise (the text the sparse features are built from)
TextFeatures(word_ngrams, char_ngrams, ...)     .fit(df) / .transform(df) -> scipy CSR: word and character TF-IDF blocks
ovr_logistic_scores(X, Y, X_new, C, ...)        one logistic regression per label -> (n_new, L) probabilities
grouped_oof_scores(X, Y, groups, ...)           out-of-fold probabilities, folds never split a group
column_calibration(scores, Y) -> (L, 2) array   per-label Platt map p = sigmoid(a + b * logit(score)) fitted on (out-of-fold) scores
calibrate(scores, params) -> (n, L) array       apply ``column_calibration`` parameters
expected_f1_decision(probs, ...) -> (n, L) 0/1  per label, how many of the top-ranked rows get a 1, chosen to maximise the
                                                mean official F1 over label matrices drawn from Bernoulli(probs)
episode_f1(Y, probs, k, ...) -> float           mean official F1 of ``expected_f1_decision`` on random k-row subsets of (Y, probs)
episode_threshold(Y, S, k, ...)                 global threshold maximising the mean official F1 on random k-item subsets
fit_predict(train_df, Y, targets, ...)          the pieces above chained: features -> grouped OOF -> calibration -> decision
                                                -> list with one 0/1 matrix per target table (unpacks like ``a, b = ...``)

``df`` are pandas tables with columns ``conclusion``, ``stance`` ('in favor of' | 'against') and ``premise``.

Decision step. The official F1 is the harmonic mean of the macro-averaged precision P and macro-averaged recall R over
the labels that have at least one gold positive among the rows being scored; a label without a predicted positive has
precision 0 and recall 0. P and R are averaged separately, so a label may be marked for every row (recall 1) while another
is marked for a few top-ranked rows (higher precision); ``expected_f1_decision`` chooses this count per label (0..n rows)
by coordinate ascent on the mean official F1 over ``n_draws`` gold matrices drawn independently from ``probs``. The rows
passed in one call are scored together, so the count depends on the number of rows (n = len(table)).

``fit_predict(train_df, Y, targets, n_folds=5, C=0.3, k=16, seed=0, n_jobs=2, groups=None, decision="expected_f1")``
  groups: array aligned with train_df rows that folds never split (default: the ``conclusion`` column).
  decision "expected_f1": Platt calibration fitted on the out-of-fold scores, then ``expected_f1_decision`` per target table.
  decision "threshold": one global cut-off from ``episode_threshold(Y, oof, k)`` applied to the scores.
  Returns ``Predictions``, a list of (len(target), L) int 0/1 matrices, one per target, with attributes ``scores`` and
  ``probs`` (lists of (len(target), L) float arrays), ``counts`` (list of per-label marked-row counts, or None for
  "threshold"), ``threshold`` (float, None for "expected_f1") and ``oof_f1`` (mean official F1 of the decision procedure
  on random k-row subsets of the out-of-fold rows of train_df).
"""
from __future__ import annotations

import numpy as np

__all__ = ["official_f1", "official_f1_report", "argument_text", "TextFeatures", "ovr_logistic_scores",
           "grouped_oof_scores", "column_calibration", "calibrate", "expected_f1_decision", "episode_f1",
           "episode_threshold", "Predictions", "fit_predict"]


def official_f1_report(y_true, y_pred) -> dict:
    """ValueEval'23 evaluator: precision_c = TP/(TP+FP) (0 without predicted positives), recall_c = TP/(TP+FN) for every
    category with at least one gold positive; f1 = harmonic mean of the macro-averaged precision P and recall R (0 if
    P + R = 0). Categories without a gold positive are skipped. NaN f1 when no category has a gold positive."""
    t = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    precs, recs, f1s = [], [], []
    for c in range(t.shape[1]):
        rel = int(t[:, c].sum())
        if rel == 0:
            continue
        pos = int(p[:, c].sum())
        tp = int(((p[:, c] == 1) & (t[:, c] == 1)).sum())
        pr, rc = (tp / pos if pos else 0.0), tp / rel
        precs.append(pr)
        recs.append(rc)
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    if not precs:
        return {"f1": float("nan"), "precision": float("nan"), "recall": float("nan"),
                "mean_category_f1": float("nan"), "n_categories": 0}
    P, R = float(np.mean(precs)), float(np.mean(recs))
    return {"f1": (2 * P * R / (P + R)) if P + R else 0.0, "precision": P, "recall": R,
            "mean_category_f1": float(np.mean(f1s)), "n_categories": len(precs)}


def official_f1(y_true, y_pred) -> float:
    """The task metric: ``official_f1_report(...)["f1"]``."""
    return float(official_f1_report(y_true, y_pred)["f1"])


def argument_text(df) -> np.ndarray:
    """Stance as one token ('in_favor_of' / 'against') followed by the premise."""
    return (df["stance"].astype(str).str.replace(" ", "_") + " " + df["premise"].astype(str)).values


class TextFeatures:
    """Sparse TF-IDF features: word n-grams of ``argument_text`` and character n-grams of the premise."""

    def __init__(self, word_ngrams=(1, 2), char_ngrams=(2, 5), min_df_word=2, min_df_char=3, max_char_features=150000):
        self.word_ngrams, self.char_ngrams = tuple(word_ngrams), tuple(char_ngrams)
        self.min_df_word, self.min_df_char, self.max_char_features = min_df_word, min_df_char, max_char_features

    def fit(self, df) -> "TextFeatures":
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.w_ = TfidfVectorizer(ngram_range=self.word_ngrams, min_df=self.min_df_word, sublinear_tf=True).fit(
            argument_text(df))
        self.c_ = TfidfVectorizer(analyzer="char_wb", ngram_range=self.char_ngrams, min_df=self.min_df_char,
                                  sublinear_tf=True, max_features=self.max_char_features).fit(df["premise"].astype(str).values)
        return self

    def transform(self, df):
        from scipy.sparse import hstack
        return hstack([self.w_.transform(argument_text(df)), self.c_.transform(df["premise"].astype(str).values)]).tocsr()

    def fit_transform(self, df):
        return self.fit(df).transform(df)


def _fit_one(X, y, X_new, C, class_weight):
    from sklearn.linear_model import LogisticRegression
    if y.min() == y.max():                        # a label without both classes in the training rows
        return np.full(X_new.shape[0], float(y[0]))
    m = LogisticRegression(C=C, class_weight=class_weight, max_iter=300).fit(X, y)
    return m.predict_proba(X_new)[:, 1]


def ovr_logistic_scores(X, Y, X_new, C: float = 0.3, class_weight="balanced", n_jobs: int = 2) -> np.ndarray:
    """(n_new, L) probabilities from L independent binary logistic regressions (column c of ``Y`` = label c)."""
    from joblib import Parallel, delayed
    Y = np.asarray(Y, dtype=int)
    cols = Parallel(n_jobs=n_jobs)(delayed(_fit_one)(X, Y[:, c], X_new, C, class_weight) for c in range(Y.shape[1]))
    return np.stack(cols, 1)


def grouped_oof_scores(X, Y, groups, n_folds: int = 5, C: float = 0.3, class_weight="balanced", n_jobs: int = 2) -> np.ndarray:
    """Out-of-fold ``ovr_logistic_scores``; ``groups`` (e.g. the conclusion text) never straddle a train/test fold."""
    from sklearn.model_selection import GroupKFold
    Y = np.asarray(Y, dtype=int)
    oof = np.zeros(Y.shape, dtype=float)
    for a, b in GroupKFold(n_folds).split(X, groups=np.asarray(groups)):
        oof[b] = ovr_logistic_scores(X[a], Y[a], X[b], C, class_weight, n_jobs)
    return oof


def episode_threshold(Y, S, k: int = 16, grid=None, n_sub: int = 400, seed: int = 0) -> float:
    """The global cut-off t for ``S >= t`` with the largest mean ``official_f1`` over ``n_sub`` random subsets of ``k``
    rows (``k`` = number of items scored at once; the metric is computed per subset of that size, not on the pool)."""
    Y = np.asarray(Y, dtype=int)
    S = np.asarray(S, dtype=float)
    grid = np.arange(0.05, 0.9, 0.025) if grid is None else np.asarray(grid, dtype=float)
    rng = np.random.default_rng(seed)
    k = min(int(k), len(Y))
    subs = [rng.choice(len(Y), k, replace=False) for _ in range(n_sub)]
    mean_f1 = [np.nanmean([official_f1(Y[i], (S[i] >= t).astype(int)) for i in subs]) for t in grid]
    return float(grid[int(np.argmax(mean_f1))])


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def column_calibration(scores, Y, C: float = 100.0) -> np.ndarray:
    """(L, 2) array of Platt parameters (a_c, b_c): p = sigmoid(a_c + b_c * logit(score)), one weakly regularised
    logistic fit per label on ``scores`` (n, L) probabilities against the 0/1 matrix ``Y``. A label with a single class
    in ``Y`` gets a constant map (its clipped prevalence)."""
    from sklearn.linear_model import LogisticRegression
    S = np.asarray(scores, dtype=float)
    Y = np.asarray(Y, dtype=int)
    out = np.zeros((Y.shape[1], 2))
    z = _logit(S)
    for c in range(Y.shape[1]):
        y = Y[:, c]
        if y.min() == y.max():
            out[c] = (float(_logit(np.clip(y.mean(), 1e-3, 1 - 1e-3))), 0.0)
            continue
        m = LogisticRegression(C=C, max_iter=300).fit(z[:, [c]], y)
        out[c] = (float(m.intercept_[0]), float(m.coef_[0, 0]))
    return out


def calibrate(scores, params) -> np.ndarray:
    """Probabilities sigmoid(a_c + b_c * logit(score)) for ``scores`` (n, L) with ``column_calibration`` parameters."""
    prm = np.asarray(params, dtype=float)
    z = prm[None, :, 0] + prm[None, :, 1] * _logit(scores)
    return 1.0 / (1.0 + np.exp(-z))


def expected_f1_decision(probs, n_draws: int = 300, seed: int = 0, n_sweeps: int = 4, return_counts: bool = False,
                         max_cells: int = 2_000_000):
    """0/1 matrix for the (n, L) probability matrix ``probs`` (rows scored together, columns = labels).

    Column c is marked for its m_c highest-probability rows (ties: earlier row first), m_c in 0..n. The vector m is
    chosen by coordinate ascent (``n_sweeps`` passes over the columns, starting from m_c = max(1, n // 3)) to maximise the
    mean ``official_f1`` over ``n_draws`` gold matrices drawn with independent Bernoulli(probs) entries (labels without a
    drawn positive are skipped as in the metric; ``n_draws`` shrinks when n * L * n_draws would exceed ``max_cells``).
    Deterministic given ``seed``. Returns the matrix, or ``(matrix, m)`` with ``return_counts=True``."""
    P = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    n, L = P.shape
    M = int(max(20, min(int(n_draws), max_cells // max(1, (n + 1) * L))))
    rng = np.random.default_rng(seed)
    G = (rng.random((M, n, L)) < P[None]).astype(np.float32)
    order = np.argsort(-P, axis=0, kind="stable")                            # (n, L) best row first
    Gs = np.take_along_axis(G, order[None], axis=1)
    cum = np.concatenate([np.zeros((M, 1, L), np.float32), np.cumsum(Gs, axis=1, dtype=np.float32)], axis=1)  # (M, n+1, L)
    rel = cum[:, n, :]                                                       # gold positives per draw and label
    scored = (rel > 0).astype(np.float32)
    n_sc = np.maximum(scored.sum(1), 1.0)                                    # (M,)
    ms = np.maximum(np.arange(n + 1), 1).astype(np.float32)
    prec = cum / ms[None, :, None] * scored[:, None, :]                      # precision of "top-m rows" (0 for m = 0)
    rec = cum / np.maximum(rel, 1.0)[:, None, :] * scored[:, None, :]
    cols = np.arange(L)
    m = np.full(L, max(1, n // 3), dtype=int)

    def f1_of(p_sum, r_sum):
        d = n_sc.reshape((-1,) + (1,) * (p_sum.ndim - 1))
        p, r = p_sum / d, r_sum / d
        return np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)

    best = f1_of(prec[:, m, cols].sum(1), rec[:, m, cols].sum(1)).mean()
    for _ in range(int(n_sweeps)):
        changed = False
        for c in range(L):
            p_rest = prec[:, m, cols].sum(1) - prec[:, m[c], c]
            r_rest = rec[:, m, cols].sum(1) - rec[:, m[c], c]
            v = f1_of(p_rest[:, None] + prec[:, :, c], r_rest[:, None] + rec[:, :, c]).mean(0)   # (n+1,) over m_c
            j = int(np.argmax(v))
            if j != m[c] and v[j] > best + 1e-9:
                m[c], best, changed = j, float(v[j]), True
        if not changed:
            break
    out = np.zeros((n, L), dtype=int)
    for c in range(L):
        out[order[:m[c], c], c] = 1
    return (out, m) if return_counts else out


def episode_f1(Y, probs, k: int = 16, n_sub: int = 100, seed: int = 0, n_draws: int = 300) -> float:
    """Mean ``official_f1`` of ``expected_f1_decision`` (applied to each subset separately) over ``n_sub`` random subsets
    of ``k`` rows of the (n, L) matrices ``Y`` (0/1) and ``probs``."""
    Y = np.asarray(Y, dtype=int)
    P = np.asarray(probs, dtype=float)
    rng = np.random.default_rng(seed)
    k = min(int(k), len(Y))
    vals = []
    for _ in range(int(n_sub)):
        i = rng.choice(len(Y), k, replace=False)
        vals.append(official_f1(Y[i], expected_f1_decision(P[i], n_draws=n_draws, seed=seed)))
    return float(np.nanmean(vals))


class Predictions(list):
    """List of 0/1 matrices (one per target table of ``fit_predict``) that also carries ``scores``, ``probs``, ``counts``,
    ``threshold`` and ``oof_f1`` (see the module docstring)."""

    def __init__(self, mats, **info):
        super().__init__(mats)
        self.__dict__.update(info)


def fit_predict(train, Y, targets, n_folds: int = 5, C: float = 0.3, k: int = 16, seed: int = 0, n_jobs: int = 2,
                groups=None, decision: str = "expected_f1") -> Predictions:
    """Features -> grouped out-of-fold scores -> decision -> one 0/1 matrix per table of ``targets`` (module docstring).

    ``train``: DataFrame with conclusion/stance/premise; ``Y``: (n, L) 0/1; ``targets``: list of such DataFrames;
    ``groups``: rows that must stay in one fold (default ``train["conclusion"]``)."""
    if decision not in ("expected_f1", "threshold"):
        raise ValueError(f"decision must be 'expected_f1' or 'threshold', got {decision!r}")
    Y = np.asarray(Y, dtype=int)
    grp = train["conclusion"].values if groups is None else np.asarray(groups)
    feats = TextFeatures().fit(train)
    X = feats.transform(train)
    oof = grouped_oof_scores(X, Y, grp, n_folds, C, n_jobs=n_jobs)
    scores = [ovr_logistic_scores(X, Y, feats.transform(t), C, n_jobs=n_jobs) for t in targets]
    if decision == "threshold":
        thr = episode_threshold(Y, oof, k=k, seed=seed)
        rng = np.random.default_rng(seed)
        kk = min(int(k), len(Y))
        oof_f1 = float(np.nanmean([official_f1(Y[i], (oof[i] >= thr).astype(int))
                                   for i in (rng.choice(len(Y), kk, replace=False) for _ in range(100))]))
        return Predictions([(s >= thr).astype(int) for s in scores], scores=scores, probs=scores, counts=None,
                           threshold=thr, oof_f1=oof_f1)
    params = column_calibration(oof, Y)
    probs = [calibrate(s, params) for s in scores]
    decided = [expected_f1_decision(p, seed=seed, return_counts=True) for p in probs]
    return Predictions([d[0] for d in decided], scores=scores, probs=probs, counts=[d[1] for d in decided],
                       threshold=None, oof_f1=episode_f1(Y, calibrate(oof, params), k=k, seed=seed))
