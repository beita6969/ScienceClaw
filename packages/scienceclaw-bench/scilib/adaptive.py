"""Adaptive testing with a one-dimensional item response model: item curves, ability posterior, question selection, prediction.

Array conventions (all arrays have one row per student and one column per question, Q columns):
  answers   (n_train, Q) int    training matrix: 1 = answered correctly, 0 = answered incorrectly, -1 = not answered
  can_query (n, Q) bool         cells that may be revealed;  targets (n, Q) bool  cells whose correctness must be predicted
  revealed  (n, Q) int          -1 = NOT revealed, 0 = revealed and answered incorrectly, 1 = revealed and answered correctly
                                (so ``revealed >= 0`` is the revealed mask; ``revealed > 0`` is only the correct ones).
                                Also valid input: a list/tuple of such arrays (one per earlier ``query_answers`` call; a cell counts as
                                revealed when any of them has a value >= 0), or None / an empty list for "nothing revealed yet".
                                ``revealed_values`` (chosen option 1-4) is a different array and is not valid input here.
  selections (n, k) int         question ids per student (-1 = none): the ``selections`` port of ``query_answers``.
A ``query_answers`` call returns ``revealed`` for that call's cells only (-1 everywhere else): the reveals of earlier calls are not
repeated in it, so with several calls keep every returned array (a list) and pass all of them.
``query_answers`` is a tool node, so a code node cannot call it: a query round is code node (selections) -> ``query_answers`` node.

Model: P(student s answers question q correctly) = sigmoid(a_q * (theta_s - b_q)) with an ability theta_s ~ N(0, 1), discrimination a_q > 0 and
difficulty b_q. The ability posterior of a student is evaluated on a grid of 81 points in [-4, 4] (``model['grid']``).

fit_item_curves(answers, iterations=30) -> model (dict of numpy arrays)
    Marginal maximum likelihood (EM on the ability grid, standard-normal prior on theta) for a_q, b_q from the observed cells of
    ``answers``; about 5-10 s for a 4,854 x 948 matrix on 2 BLAS threads, deterministic (no random numbers). A question without any answer gets
    a = 1, b = 0. model keys: 'grid' (81,), 'prior' (81,), 'curves' (Q, 81) = P(correct | theta_grid), 'discrimination' (Q,),
    'difficulty' (Q,), 'n_answers' (Q,), 'majority' (Q,) int (1 if more correct than incorrect training answers else 0),
    'loglik' (marginal log-likelihood per iteration).
ability_posterior(model, revealed) -> (n, 81)
    Posterior weights of theta on the grid given the revealed cells (rows sum to 1; a student with nothing revealed gets the prior).
predict_proba(model, revealed) -> (n, Q)
    Posterior-predictive probability of a correct answer for every cell.
select_queries(model, can_query, revealed=None, k=10, budget=10, method='batch') -> (n, k) int
    ``selections`` for ``query_answers``: per student up to k question ids among the cells with can_query True and not yet revealed, -1
    padding; k is capped so that (revealed cells + selected) <= ``budget`` per student. Selection uses the posterior given ``revealed``.
    method 'bald': the k cells with the largest mutual information between the answer and theta, I = H(E[p_q]) - E[H(p_q)] (expectations
    over the posterior); 'batch': greedy joint selection, each next cell maximises the mutual information between its answer and theta given
    the cells already chosen in this call (joint distribution over their answers), which avoids near-duplicate cells within one call. Both
    are deterministic: the result depends only on the argument arrays (ties -> smaller question id), and for the same ``revealed`` the
    list for k is a prefix of the list for a larger k. ``revealed`` after a further round is a different input, so changing the number of
    rounds or their sizes changes which cells the later rounds ask for.
predict(model, revealed, targets) -> (n, Q) int
    1 where predict_proba > 0.5 else 0 on cells with targets True, -1 elsewhere (the layout of the required deliverable).

merge_revealed(revealed, shape=None) -> (n, Q) int
    The single -1 / 0 / 1 array of a list of ``revealed`` arrays (-1 where no array has a value >= 0).

``model`` in select_queries / predict / predict_proba / ability_posterior is either the dict returned by fit_item_curves or the training
answer matrix itself (fitted with the default settings; the fit is cached per process, keyed by the array contents).

Call pattern (``answers`` from load_train, ``can_query`` / ``targets`` from load_eval_inputs, ``rev1`` / ``rev2`` the ``revealed`` ports
of two query_answers nodes):  selections = select_queries(answers, can_query, [], 5);  selections2 = select_queries(answers, can_query, [rev1], 5);
y = predict(answers, [rev1, rev2], targets).

Distinct cells charged by ``query_answers`` are counted over every executed call of the whole session (also of removed or replaced
nodes), so a selection that differs from an earlier one for the same student adds new cells to the total.
"""
from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["fit_item_curves", "ability_posterior", "predict_proba", "select_queries", "predict", "merge_revealed"]

_GRID = np.linspace(-4.0, 4.0, 81)
_EPS = 1e-4
_CACHE: dict = {}


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _check_answers(answers) -> np.ndarray:
    a = np.asarray(answers)
    if a.ndim != 2:
        raise ValueError(f"answers must be a 2-D array (students x questions), got shape {a.shape}")
    if not np.all(np.isin(a, (-1, 0, 1))):
        raise ValueError("answers must contain only -1 (not answered), 0 (incorrect) and 1 (correct)")
    return a


def fit_item_curves(answers, iterations: int = 30) -> dict:
    """Item curves of the one-dimensional two-parameter model by marginal maximum likelihood (see the module docstring)."""
    from scipy import sparse

    A = _check_answers(answers)
    n, Q = A.shape
    grid = _GRID
    G = grid.size
    logprior = -0.5 * grid ** 2
    logprior -= np.log(np.exp(logprior).sum())
    prior = np.exp(logprior)

    rows, cols = np.nonzero(A >= 0)
    ones = (A[rows, cols] == 1).astype(np.float64)
    C1 = sparse.csr_matrix((ones, (rows, cols)), shape=(n, Q))                      # correct cells
    Mm = sparse.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, Q))       # answered cells
    C1T, MT = C1.T.tocsr(), Mm.T.tocsr()
    n_ans = np.asarray(Mm.sum(axis=0)).ravel()
    n_cor = np.asarray(C1.sum(axis=0)).ravel()

    # start: a = 1, difficulty from the question's proportion correct
    a = np.ones(Q)
    b = -np.log((n_cor + 1.0) / (n_ans - n_cor + 1.0))
    a_prior, lam_a, lam_c = 1.0, 0.5, 0.01
    loglik: list[float] = []
    for _ in range(int(iterations)):
        c = a * b
        P = np.clip(_sigmoid(a[:, None] * grid[None, :] - c[:, None]), _EPS, 1 - _EPS)      # (Q, G)
        LP, LQ = np.log(P), np.log1p(-P)
        ll = C1 @ (LP - LQ) + Mm @ LQ + logprior[None, :]                                    # (n, G)
        mx = ll.max(axis=1, keepdims=True)
        post = np.exp(ll - mx)
        norm = post.sum(axis=1, keepdims=True)
        post /= norm
        loglik.append(float((np.log(norm[:, 0]) + mx[:, 0]).sum()))
        R = C1T @ post                                                                      # (Q, G) expected correct counts
        N = MT @ post                                                                       # (Q, G) expected answer counts
        # one Newton step per question on the expected complete-data log-likelihood in (a, c), z = a * theta - c
        res = R - N * P
        w = N * P * (1 - P)
        ga = res @ grid - lam_a * (a - a_prior)
        gc = -res.sum(axis=1) - lam_c * c
        haa = w @ grid ** 2 + lam_a
        hac = -(w @ grid)
        hcc = w.sum(axis=1) + lam_c
        det = haa * hcc - hac ** 2
        da = (hcc * ga - hac * gc) / det
        dc = (haa * gc - hac * ga) / det
        a = np.clip(a + np.clip(da, -0.5, 0.5), 0.05, 6.0)
        c = c + np.clip(dc, -1.0, 1.0)
        b = c / a
        unseen = n_ans == 0
        a[unseen], b[unseen] = 1.0, 0.0

    c = a * b
    curves = np.clip(_sigmoid(a[:, None] * grid[None, :] - c[:, None]), _EPS, 1 - _EPS)
    return {"grid": grid.copy(), "prior": prior, "curves": curves, "discrimination": a, "difficulty": b,
            "n_answers": n_ans.astype(np.int64), "majority": (n_cor > n_ans - n_cor).astype(np.int64),
            "loglik": np.array(loglik)}


def _model(model) -> dict:
    """The dict of fit_item_curves, fitting (and caching per process) when the training matrix was passed."""
    if isinstance(model, dict):
        if "curves" not in model or "prior" not in model:
            raise ValueError("model dict must come from fit_item_curves (keys 'curves' and 'prior')")
        return model
    A = np.ascontiguousarray(_check_answers(model), dtype=np.int8)
    key = (A.shape, hashlib.sha1(A.tobytes()).hexdigest())
    if key not in _CACHE:
        if len(_CACHE) >= 2:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = fit_item_curves(A)
    return _CACHE[key]


def merge_revealed(revealed, shape=None) -> np.ndarray:
    """One (n, Q) int array with -1 where no call revealed the cell, else the revealed 0 / 1 (input: array, list of arrays or None)."""
    if revealed is None or (isinstance(revealed, (list, tuple)) and len(revealed) == 0):
        if shape is None:
            raise ValueError("nothing revealed and no shape given")
        return np.full(shape, -1, dtype=np.int64)
    arrs = [np.asarray(r) for r in revealed] if isinstance(revealed, (list, tuple)) else [np.asarray(revealed)]
    if isinstance(revealed, np.ndarray) and revealed.ndim == 3:
        arrs = [np.asarray(r) for r in revealed]
    out = np.full(arrs[0].shape, -1, dtype=np.int64)
    for r in arrs:
        if r.shape != out.shape:
            raise ValueError(f"revealed arrays differ in shape: {r.shape} vs {out.shape}")
        if not np.all(np.isin(r, (-1, 0, 1))):
            raise ValueError("revealed must contain only -1 (not revealed), 0 (incorrect) and 1 (correct); "
                             "revealed_values (options 1-4) is not accepted")
        out = np.where((r >= 0) & (out < 0), r, out)
    return out


def ability_posterior(model, revealed) -> np.ndarray:
    m = _model(model)
    rev = merge_revealed(revealed, None if not isinstance(revealed, np.ndarray) else revealed.shape)
    if rev.shape[1] != m["curves"].shape[0]:
        raise ValueError(f"revealed has {rev.shape[1]} columns, the model {m['curves'].shape[0]} questions")
    lp, lq = np.log(m["curves"]), np.log1p(-m["curves"])
    ll = (rev == 1).astype(np.float64) @ lp + (rev == 0).astype(np.float64) @ lq + np.log(m["prior"])[None, :]
    ll -= ll.max(axis=1, keepdims=True)
    w = np.exp(ll)
    return w / w.sum(axis=1, keepdims=True)


def predict_proba(model, revealed) -> np.ndarray:
    m = _model(model)
    return np.clip(ability_posterior(m, revealed) @ m["curves"].T, 1e-6, 1 - 1e-6)


def _entropy(p):
    return -(p * np.log(p) + (1 - p) * np.log1p(-p))


def _bald_scores(m, w):
    p = np.clip(w @ m["curves"].T, 1e-6, 1 - 1e-6)
    return _entropy(p) - w @ _entropy(m["curves"]).T


def _greedy_batch(curves, ent, w, ok_idx, k):
    """Greedy joint-information selection for one student; returns question ids (<= k) among ``ok_idx``."""
    Pc = curves[ok_idx]                                     # (m, G)
    cond = ent[ok_idx] @ w                                  # (m,) E_theta H(answer)
    J = w[None, :]                                          # (2^s, G): weight of theta_g and each outcome of the chosen cells
    chosen: list[int] = []
    taken = np.zeros(ok_idx.size, dtype=bool)
    for _ in range(min(k, ok_idx.size)):
        A1 = J @ Pc.T                                       # (2^s, m) P(outcome o, candidate correct)
        A0 = J.sum(axis=1, keepdims=True) - A1
        h = -(np.where(A1 > 0, A1 * np.log(np.maximum(A1, 1e-300)), 0.0).sum(axis=0)
              + np.where(A0 > 0, A0 * np.log(np.maximum(A0, 1e-300)), 0.0).sum(axis=0))
        score = np.where(taken, -np.inf, h - cond)
        j = int(np.argmax(score))                           # first maximum -> smaller question id
        chosen.append(j)
        taken[j] = True
        J = np.concatenate([J * Pc[j][None, :], J * (1 - Pc[j])[None, :]], axis=0)
    return ok_idx[np.array(chosen, dtype=np.int64)]


def select_queries(model, can_query, revealed=None, k: int = 10, budget: int = 10, method: str = "batch") -> np.ndarray:
    """(n, k) int ``selections`` for ``query_answers`` (see the module docstring)."""
    m = _model(model)
    cq = np.asarray(can_query).astype(bool)
    if cq.ndim != 2 or cq.shape[1] != m["curves"].shape[0]:
        raise ValueError(f"can_query must have shape (n, {m['curves'].shape[0]}), got {cq.shape}")
    if method not in ("bald", "batch"):
        raise ValueError("method must be 'bald' or 'batch'")
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")
    n, Q = cq.shape
    rev = merge_revealed(revealed, cq.shape)
    if rev.shape != cq.shape:
        raise ValueError(f"revealed has shape {rev.shape}, can_query {cq.shape}")
    w = ability_posterior(m, rev)
    used = (rev >= 0).sum(axis=1)
    out = np.full((n, k), -1, dtype=np.int64)
    ok = cq & (rev < 0)
    ent = _entropy(m["curves"])
    scores = _bald_scores(m, w) if method == "bald" else None
    for i in range(n):
        kk = min(k, max(0, int(budget) - int(used[i])))
        idx = np.flatnonzero(ok[i])
        if kk == 0 or idx.size == 0:
            continue
        if method == "bald":
            order = np.argsort(-scores[i, idx], kind="stable")[:kk]
            pick = idx[order]
        else:
            pick = _greedy_batch(m["curves"], ent, w[i], idx, kk)
        out[i, :pick.size] = pick
    return out


def predict(model, revealed, targets) -> np.ndarray:
    """(n, Q) int: 0/1 on cells with targets True (1 where predict_proba > 0.5), -1 elsewhere."""
    t = np.asarray(targets).astype(bool)
    p = predict_proba(model, merge_revealed(revealed, t.shape))
    if p.shape != t.shape:
        raise ValueError(f"targets has shape {t.shape}, predictions {p.shape}")
    return np.where(t, (p > 0.5).astype(np.int64), -1)
