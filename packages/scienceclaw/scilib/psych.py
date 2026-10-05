"""Tools for predicting the next choice of a participant from the participant's own trial history (Psych-201).

Sessions are text in which every response is written ``<<KEY>>`` and the outcome (points, reward, correct/incorrect,
counterfactual amounts) follows on the same line. All functions work on the dicts/strings the tools return and read no
files. ``item`` = {"study", "history", "options"[, "options_text"]} as returned by load_eval_inputs / load_dev_inputs;
``session`` = {"study", "text", "responses": [{"pos", "response", "options"}, ...]} as returned by load_train.

Two kinds of predictors are available and can be mixed: models fitted to the participant's own earlier responses and
outcomes (``fit_predict`` methods; they do not read the instruction text or the content of the current trial), and the
answers of an ``llm`` node to prompts made by ``build_prompts`` (the language model reads the instruction text and the
current trial as well as the earlier trials); ``fit_predict(..., llm_answers=...)`` combines the two.

Metric helpers
  mode_prediction(history, options) -> str   most frequent earlier response among ``options`` (ties: most recent;
                                                  none: options[0])
  accuracy(y_true, y_pred, options=None) -> float  fraction of equal entries; an entry outside ``options[i]`` is wrong
  summarize(y_true, y_pred, histories, options) -> dict   accuracy, mode_accuracy (of mode_prediction) and n_items

Parsing and features
  parse_trials(text, options=None, trial_options=None) -> Trials   one record per ``<<KEY>>`` marker in ``text``: key,
        legal options, outcome (first number after the marker, "win X ... lose Y" net, categorical reward/correct words
        -> 1/0), counterfactual amounts ("would have received X ... pressed W", "Forgone Outcome = X"); ``.arrays()``
        gives choice index, availability mask, reward, forgone-reward matrix for the response after the text.
  trial_features(trials) -> (names, F, context, avail, choice)   F[t, k, j]: feature j of candidate option k before
        trial t (t = 0..T, row T is the response to predict; only markers before t are used): last / second-last
        choice indicator, choice frequency, exponentially weighted choice kernels (rates 0.1 / 0.3), delta-rule
        Q-values (learning rate 0.15 / 0.5, rewards centred and scaled by the running mean / sd of the first 30
        outcomes, counterfactual amounts update the unchosen option), last-choice indicator split by previous outcome
        above / below the running mean, log trials since chosen; ``context`` columns are per trial (log position,
        repeat rate so far, win-stay / lose-shift rate so far, number of options).
  make_pseudo_items(sessions, per_session=4, seed=0) -> list of {"study","history","options","target","group","round"}
        cut points inside COMPLETE visible sessions (>= 2 earlier responses, cut before character 16000), for scoring
        methods on data with known responses; ``group`` = session index (use it for grouped splits).

Models (each is fitted on the item's own earlier responses; ``method=`` names below)
  "mode"       participant mode                   "last"     repeat the previous response
  "kernel"     softmax over choice-frequency and recency-weighted choice kernels + last-choice bias (perseveration)
  "qlearn"     softmax over delta-rule Q-values (learning rate chosen on a grid of 7 values) + last choice + choice
               kernel: Q-learning with softmax, inverse temperature and perseveration
  "wsls"       stay bias separated by previous outcome above/below the running mean (win-stay / lose-shift, soft)
               -- "kernel", "qlearn", "wsls": conditional-logit maximum a posteriori fits (scipy L-BFGS, L2 = 0.3 towards
               the population prior below), one fit per item, at most the last 400 earlier responses
  "personal"   probability average of "kernel", "qlearn", "wsls" weighted by exp(-BIC/2) on the item's history
  "gbdt"       LightGBM on per-candidate history features, fitted on ``train_sessions`` and the histories of all given
               items (rows = trials of one text; the candidate with the largest score wins); the participant mode
               when fewer than 200 rows are available
  "auto"       0.5 * "personal" + 0.5 * "gbdt" probabilities

Entry points
  fit_predict(items, train_sessions=None, method="auto", extra_items=None, seed=0, return_proba=False,
              population_prior=True, llm_answers=None, llm_weight=LLM_WEIGHT) -> list of predicted keys (one legal key
        per item, same order); with return_proba=True also a list of {key: probability}. ``train_sessions``
        (load_train dicts) and the histories of ``items`` / ``extra_items`` (e.g. the load_dev_inputs items) feed
        "gbdt" and the population prior of the
        likelihood models (per-study mean coefficients when the study has >= 4 texts, otherwise the mean over all
        texts; ``population_prior=False`` shrinks towards zero instead). Items are independent: adding items changes
        only the fitting rows of "gbdt" and the prior. Ties resolve to the participant mode. ``llm_answers`` = list of
        answer lists (each one key or None per item, e.g. the output of parse_answers for one prompt style): the
        probabilities of ``method`` become (1 - llm_weight) * p + llm_weight * share, share[k] = fraction of the item's
        valid answers that name key k; an item without a valid answer keeps p; llm_weight in [0, 1]
        (LLM_WEIGHT = 0.35 by default). ``method="all"`` returns {name: keys} for every model, and with
        ``llm_answers`` also "llm" (majority of the answers, ties -> participant mode) and "auto_llm".
  cross_validate(sessions, method="auto", n_splits=4, per_session=4, seed=0, extra_methods=()) -> dict
        out-of-fold scores of a method on pseudo-items of COMPLETE sessions, folds grouped by session (a participant
        never appears in both fitting and scoring; each call scores one cut per session); returns accuracy,
        mode_accuracy, n_items and per-study accuracy (a dict per method when ``extra_methods`` is given).

Language-model answers
  PROMPT_STYLES = ("transcript", "long", "model")
  build_prompts(items, style="transcript", train_sessions=None, extra_items=None, seed=0) -> list[str]   one text
        prompt per item: the first 1200 characters of the history (task instructions), the latest trials (last
        9000 characters; "long": 14000) with the ``<<KEY>>`` markers, the legal keys and the question which key the
        participant presses next, "answer with the key only". Style "model" adds one sentence with the probabilities
        that fit_predict(method="personal") gives the legal keys (``train_sessions`` / ``extra_items`` as there).
  parse_answers(texts, items) -> list   key named by each reply text (None when empty or when no single legal key can
        be read): the reply itself after stripping quotes / brackets / '<<>>' / end punctuation, else the last
        '<<KEY>>' in it, else the only legal key occurring in it as a separate token.
  Shapes around an llm node. The template ``{item}`` inserts a string item unchanged, so an llm node with template
  ``{item}``, ``config.parse = "text"`` and a small ``config.max_tokens`` whose input port ``items`` receives a list of
  strings (the list from build_prompts, or the concatenation of the prompt lists of several item lists, e.g. the
  evaluation items followed by the load_dev_inputs items) gives one reply per string on its output port ``outputs``
  (None for a failed request), in the same order; parse_answers takes that list as ``texts`` (slice it to separate the
  item lists) and its result is one element of ``llm_answers`` of fit_predict. Code nodes cannot call the language
  model. An llm node takes at most the llm-item limit stated in the task budget, and identical prompts give identical
  replies (temperature 0, cached); a different prompt style gives a different answer list.
  Example wiring (ports are named as in the comments; the tools' output ports are load_eval_inputs.items,
  load_dev_inputs.dev_items, load_train.sessions):
    code node "prompts"  (inputs eval_items, dev_items; output prompts)
        def run(inputs, config):
            from scilib import psych
            return {"prompts": psych.build_prompts(inputs["eval_items"] + inputs["dev_items"], style="transcript")}
    llm node "answer"    (input items <- prompts; template "{item}"; config {"parse": "text", "max_tokens": 8}; output outputs)
    code node "combine"  (inputs eval_items, dev_items, train_sessions, replies <- answer.outputs; outputs y, dev_pred)
        def run(inputs, config):
            from scilib import psych
            items = inputs["eval_items"] + inputs["dev_items"]
            answers = psych.parse_answers(inputs["replies"], items)
            keys = psych.fit_predict(items, inputs["train_sessions"], "auto", llm_answers=[answers])
            n = len(inputs["eval_items"])
            return {"y": keys[:n], "dev_pred": keys[n:]}

Determinism: seeded; CPU only (LightGBM n_jobs=2); on the order of 0.05 s of CPU per item for the likelihood models
plus a few seconds for the prior and "gbdt" fit.
"""
from __future__ import annotations

import math
import re
import warnings

import numpy as np

__all__ = ["mode_prediction", "accuracy", "summarize", "parse_trials", "trial_features", "make_pseudo_items",
           "fit_predict", "cross_validate", "build_prompts", "parse_answers", "Trials", "MODELS", "PROMPT_STYLES"]

MODELS = ("mode", "last", "kernel", "qlearn", "wsls", "personal", "gbdt", "auto")
LLM_WEIGHT = 0.35                                    # default weight of the language-model answers in fit_predict
_MARK = re.compile(r"<<([^<>]*)>>")
_NUM = r"[-+]?\d+(?:\.\d+)?"
_NUM_RE = re.compile(_NUM)
_DYNAMIC = (
    re.compile(r"You can choose between option ([A-Z]) and option ([A-Z])\. You press $"),
    re.compile(r"You can choose between machines ([A-Z]) and ([A-Z])\. You press $"),
    re.compile(r"You encounter stimuli ([A-Z]), ([A-Z])\. You press $"),
)
_WOULD = re.compile(r"would have (?:received|gotten|earned|won|got)\s*(" + _NUM + r")([^.]*)", re.I)
_WINLOSE = re.compile(r"\bwin\s*(" + _NUM + r")\D{0,12}?\band\s+lose\s*(" + _NUM + r")", re.I)
_LOSE = re.compile(r"\blose\s*(" + _NUM + r")", re.I)
_OBT = re.compile(r"Obtained Outcome\s*=\s*(" + _NUM + r")", re.I)
_FORG = re.compile(r"Forgone Outcome\s*=\s*(" + _NUM + r")", re.I)
_KEYTOK = re.compile(r"\b([A-Z])\b")
N_SCALE = 30                     # rewards of the first N_SCALE trials define the centre / scale of the reward units


# ============================================================================================ metric helpers
def mode_prediction(history: str, options) -> str:
    """Most frequent earlier response of the participant among ``options`` (ties -> most recent; none -> options[0])."""
    options = list(options)
    prev = [r for r in _MARK.findall(history) if r in options]
    if not prev:
        return options[0]
    counts = {o: prev.count(o) for o in options}
    best = max(counts.values())
    tied = {o for o, c in counts.items() if c == best}
    for r in reversed(prev):
        if r in tied:
            return r
    return options[0]


def accuracy(y_true, y_pred, options=None) -> float:
    """Fraction of items with ``y_pred[i] == y_true[i]``; with ``options`` (list of key lists) a prediction outside
    ``options[i]`` counts as wrong. NaN for an empty list."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"accuracy: {len(y_true)} targets but {len(y_pred)} predictions")
    if not len(y_true):
        return float("nan")
    ok = []
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        ok.append(bool(p == t and (options is None or p in options[i])))
    return float(np.mean(ok))


def summarize(y_true, y_pred, histories, options) -> dict:
    """Accuracy of ``y_pred`` and accuracy of :func:`mode_prediction` on the same items."""
    ref = [mode_prediction(h, o) for h, o in zip(histories, options)]
    acc, racc = accuracy(y_true, y_pred, options), accuracy(y_true, ref, options)
    return {"accuracy": acc, "mode_accuracy": racc, "n_items": len(y_true)}


# ============================================================================================ parsing
class Trials:
    """Parsed response markers of one text. ``keys``: sorted response alphabet; per marker ``choice`` (index into
    ``keys``), ``options`` (list of key lists), ``reward`` (float or nan), ``forgone`` ({key: amount}); ``target_options``
    = legal keys of the response after the text (None if unknown)."""

    def __init__(self, choices, options, rewards, forgone, target_options):
        alphabet = set(choices) | {k for o in options for k in o} | set(target_options or [])
        self.keys = sorted(alphabet)
        self._ix = {k: i for i, k in enumerate(self.keys)}
        self.choice_keys = list(choices)
        self.options = [list(o) for o in options]
        self.reward = np.asarray(rewards, dtype=float)
        self.forgone = list(forgone)
        self.target_options = None if target_options is None else list(target_options)

    def __len__(self) -> int:
        return len(self.choice_keys)

    def arrays(self):
        """(choice[T] int, avail[T+1, K] bool, reward[T], forgone[T, K]) for the T markers and the response after them."""
        T, K = len(self), len(self.keys)
        choice = np.array([self._ix[k] for k in self.choice_keys], dtype=int)
        avail = np.zeros((T + 1, K), dtype=bool)
        for t, o in enumerate(self.options):
            avail[t, [self._ix[k] for k in o]] = True
        to = self.target_options or (self.options[-1] if T else self.keys)
        avail[T, [self._ix[k] for k in to]] = True
        forg = np.full((T, K), np.nan)
        for t, d in enumerate(self.forgone):
            for k, v in d.items():
                if k in self._ix:
                    forg[t, self._ix[k]] = v
        return choice, avail, self.reward.copy(), forg


def _outcome(seg: str, key: str, options: list[str]):
    """(reward, {forgone key: amount}) from the text following a response marker on the same line."""
    forg: dict[str, float] = {}
    obtained = seg
    m = _WOULD.search(seg)
    if m:
        obtained = seg[: m.start()]
        for w in _WOULD.finditer(seg):
            v, tail = float(w.group(1)), w.group(2)
            k = re.search(r"(?:pressed|press|chosen|chose|choose|option|machine)\s+'?([A-Z])\b", tail)
            others = [o for o in options if o != key]
            if k and k.group(1) in options and k.group(1) != key:
                forg[k.group(1)] = v
            elif len(others) == 1:
                forg[others[0]] = v
    fo = _FORG.search(seg)
    if fo and len([o for o in options if o != key]) == 1:
        forg[[o for o in options if o != key][0]] = float(fo.group(1))
    rew = float("nan")
    ob = _OBT.search(obtained)
    wl = _WINLOSE.search(obtained)
    if ob:
        rew = float(ob.group(1))
    elif wl:
        rew = float(wl.group(1)) - float(wl.group(2))
    else:
        lo = _LOSE.search(obtained)
        nums = _NUM_RE.findall(obtained)
        if lo:
            rew = -abs(float(lo.group(1)))
        elif nums:
            rew = float(nums[0])
        else:
            low = obtained.lower()
            if "reward" in low:
                rew = 0.0 if re.search(r"\bno\b|\bnot\b", low) else 1.0
            else:
                toks = [t for t in _KEYTOK.findall(obtained) if t in options]
                if len(toks) == 1 and re.search(r"correct|win|wins|right", low):
                    rew = 1.0 if toks[0] == key else 0.0
    return rew, forg


def parse_trials(text: str, options=None, trial_options=None) -> Trials:
    """One record per ``<<KEY>>`` marker in ``text``.

    ``options``: legal keys of the response that follows ``text`` (also the default legal keys of earlier markers).
    ``trial_options``: optional legal keys of the markers, a list (one entry per marker) or a dict {character offset of
    '<<': keys} (e.g. ``{r["pos"]: r["options"] for r in session["responses"]}``). Otherwise the dynamic pair lines ("You can choose between option K and option W. You
    press") give per-marker options and everything else falls back to ``options``."""
    options = None if options is None else list(options)
    marks = list(_MARK.finditer(text))
    ch, opts, rews, forgs = [], [], [], []
    for i, mk in enumerate(marks):
        key = mk.group(1)
        line = text[text.rfind("\n", 0, mk.start()) + 1: mk.start()]
        o = None
        given = None
        if isinstance(trial_options, dict):
            given = trial_options.get(mk.start())
        elif trial_options is not None and i < len(trial_options):
            given = trial_options[i]
        if given:
            o = list(given)
        else:
            for pat in _DYNAMIC:
                d = pat.search(line)
                if d:
                    o = list(d.groups())
                    break
        if o is None:
            o = list(options) if options else [key]
        if key not in o:
            o = o + [key]
        stop = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        seg = text[mk.end():stop]
        nl = seg.find("\n")
        seg = seg if nl < 0 else seg[:nl]
        r, f = _outcome(seg, key, o)
        ch.append(key)
        opts.append(o)
        rews.append(r)
        forgs.append(f)
    return Trials(ch, opts, rews, forgs, options)


# ============================================================================================ features
_ALPHAS = (0.15, 0.5)
_RATES = (0.1, 0.3)
_FEATURES = (["last", "last2", "freq"] + [f"ck{r}" for r in _RATES] + [f"q{a}" for a in _ALPHAS]
             + ["last_win", "last_lose", "since"])
_CONTEXT = ["log_pos", "repeat_rate", "win_stay", "lose_shift", "n_options"]


def _causal_scale(reward: np.ndarray):
    """Per-trial centre / scale (m[t], s[t]) from the observed rewards of trials 0..t, using at most the first N_SCALE
    trials (task units differ by orders of magnitude); s = 1 while the rewards seen so far do not vary."""
    T = len(reward)
    m, s = np.zeros(T), np.ones(T)
    vals: list[float] = []
    cm, cs = 0.0, 1.0
    for t in range(T):
        if t < N_SCALE and np.isfinite(reward[t]):
            vals.append(float(reward[t]))
            cm = float(np.mean(vals))
            sd = float(np.std(vals))
            cs = sd if (len(vals) >= 2 and sd > 1e-9) else 1.0
        m[t], s[t] = cm, cs
    return m, s


def trial_features(trials: Trials, alphas=_ALPHAS, rates=_RATES):
    """Returns ``(names, F, context, avail, choice)``: ``F[t, k, j]`` (t = 0..T, k over ``trials.keys``) uses only
    markers before t; ``context[t, :]`` per-trial columns (:data:`_CONTEXT`); ``choice[t]`` index of the response at t."""
    choice, avail, rew, forg = trials.arrays()
    T, K = len(choice), len(trials.keys)
    cm, cs = _causal_scale(rew)
    na, nr = len(alphas), len(rates)
    F = np.zeros((T + 1, K, 3 + nr + na + 3))
    ctx = np.zeros((T + 1, len(_CONTEXT)))
    Q = np.zeros((na, K))
    CK = np.zeros((nr, K))
    cnt = np.zeros(K)
    since = np.zeros(K)
    a_arr = np.asarray(alphas)[:, None]
    r_arr = np.asarray(rates)[:, None]
    run_sum, run_n = 0.0, 0
    stay = shift = win_stay_n = win_stay_d = lose_shift_n = lose_shift_d = 0.0
    rep_n = rep_d = 0.0
    prev_win = 0.5
    for t in range(T + 1):
        last = np.zeros(K)
        last2 = np.zeros(K)
        if t >= 1:
            last[choice[t - 1]] = 1.0
        if t >= 2:
            last2[choice[t - 2]] = 1.0
        freq = cnt / max(t, 1)
        F[t, :, 0], F[t, :, 1], F[t, :, 2] = last, last2, freq
        F[t, :, 3:3 + nr] = CK.T
        F[t, :, 3 + nr:3 + nr + na] = Q.T
        F[t, :, 3 + nr + na] = last * prev_win
        F[t, :, 4 + nr + na] = last * (1.0 - prev_win)
        F[t, :, 5 + nr + na] = np.log1p(np.minimum(since, 20.0))
        ctx[t] = [math.log(t + 1), rep_n / rep_d if rep_d else 0.5, win_stay_n / win_stay_d if win_stay_d else 0.5,
                  lose_shift_n / lose_shift_d if lose_shift_d else 0.5, float(avail[t].sum())]
        if t == T:
            break
        c = choice[t]
        if t >= 1:
            rep_d += 1
            rep_n += float(c == choice[t - 1])
            if prev_win != 0.5:
                if prev_win == 1.0:
                    win_stay_d += 1
                    win_stay_n += float(c == choice[t - 1])
                else:
                    lose_shift_d += 1
                    lose_shift_n += float(c != choice[t - 1])
        # updates with the outcome of trial t
        CK += r_arr * (np.eye(K)[c][None, :] - CK)
        cnt[c] += 1
        since += 1
        since[c] = 0
        if np.isfinite(rew[t]):
            Q[:, c] += a_arr[:, 0] * ((rew[t] - cm[t]) / cs[t] - Q[:, c])
            prev_win = 1.0 if (run_n and rew[t] > run_sum / run_n + 1e-12) else (
                0.0 if run_n and rew[t] < run_sum / run_n - 1e-12 else 0.5)
            run_sum += rew[t]
            run_n += 1
        else:
            prev_win = 0.5
        for k in np.flatnonzero(np.isfinite(forg[t])):
            if k != c:
                Q[:, k] += a_arr[:, 0] * ((forg[t, k] - cm[t]) / cs[t] - Q[:, k])
    names = list(_FEATURES) if (alphas, rates) == (_ALPHAS, _RATES) else (
        ["last", "last2", "freq"] + [f"ck{x}" for x in rates] + [f"q{x}" for x in alphas]
        + ["last_win", "last_lose", "since"])
    return names, F, ctx, avail, choice


# ============================================================================================ personal likelihood models
_GRID = (0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9)
_MAX_TRIALS = 400
L2 = 0.3


def _tail(trials: Trials, n: int) -> Trials:
    """The last ``n`` markers of ``trials`` (same alphabet)."""
    if len(trials) <= n:
        return trials
    out = Trials(trials.choice_keys[-n:], trials.options[-n:], trials.reward[-n:], trials.forgone[-n:],
                 trials.target_options)
    return out


def _clogit(X, choice, avail, l2, mu, maxiter=60):
    """Conditional-logit maximum a posteriori fit: utility = X @ theta; returns (theta, negative log-likelihood)."""
    from scipy.optimize import minimize
    T, K, P = X.shape
    idx = np.arange(T)
    xc = X[idx, choice]

    def fun(th):
        U = np.where(avail, X @ th, -1e9)
        m = U.max(1)
        lse = m + np.log(np.exp(U - m[:, None]).sum(1))
        nll = float(-(U[idx, choice] - lse).sum())
        Pm = np.exp(U - lse[:, None])
        g = -(xc - (Pm[:, :, None] * X).sum(1)).sum(0)
        d = th - mu
        return nll + l2 * float(d @ d), g + 2 * l2 * d

    r = minimize(fun, mu.copy(), jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    nll = fun(r.x)[0] - l2 * float((r.x - mu) @ (r.x - mu))
    return r.x, nll


_SPECS = {                                        # model -> (column-name list builder over alpha)
    "kernel": lambda a: ["last", "last2", "freq", "ck0.1", "ck0.3"],
    "qlearn": lambda a: [f"q{a}", "last", "ck0.3"],
    "wsls": lambda a: ["last_win", "last_lose", "freq"],
}


def _personal(trials: Trials, prior=None, l2=L2, which=None):
    """Fit the likelihood models on ``trials`` (earlier responses) and predict the response after them.

    Returns {model: (probs over trials.keys, bic)}; a model whose history is too short returns uniform probabilities."""
    t2 = _tail(trials, _MAX_TRIALS)
    names, F, ctx, avail, choice = trial_features(t2, alphas=_GRID, rates=_RATES)
    T = len(choice)
    K = len(t2.keys)
    out = {}
    for model, spec in _SPECS.items():
        if which is not None and model not in which:
            continue
        grid = _GRID if model == "qlearn" else (None,)
        best = None
        for a in grid:
            cols = [names.index(c) for c in spec(a)]
            X = F[:T][:, :, cols]
            if T < 3:
                th, nll = np.zeros(len(cols)), T * math.log(max(2, K))
            else:
                mu = np.zeros(len(cols)) if prior is None else prior.get((model, a), np.zeros(len(cols)))
                th, nll = _clogit(X, choice, avail[:T], l2, mu)
            obj = nll + l2 * float(np.sum((th - mu) ** 2)) if T >= 3 else nll
            if best is None or obj < best[0]:
                best = (obj, th, cols, nll)
        _, th, cols, nll = best
        u = np.where(avail[T], F[T][:, cols] @ th, -np.inf)
        p = np.exp(u - u.max())
        p = p / p.sum()
        out[model] = (p, 2 * nll + len(cols) * math.log(max(T, 2)))
    return out


def _fit_prior(texts, l2=1.0, per_text=40, seed=0, min_study=4, studies=None):
    """Population means of the coefficients of the likelihood models: pooled conditional-logit fits over the earlier
    responses of many texts, per study when the study has at least ``min_study`` texts and over all texts otherwise.
    ``texts``: list of (study, Trials); ``studies``: restrict the per-study fits to these studies (default all).
    Returns {study or None: {(model, alpha): theta}}."""
    rng = np.random.default_rng(seed)
    rows = []                                                  # (study, F rows, choice, avail)
    for study, tr in texts:
        t2 = _tail(tr, _MAX_TRIALS)
        if len(t2) < 6:
            continue
        names, F, ctx, avail, choice = trial_features(t2, alphas=_GRID, rates=_RATES)
        T = len(choice)
        ts = np.arange(2, T)
        if len(ts) > per_text:
            ts = np.sort(rng.choice(ts, per_text, replace=False))
        rows.append((study, F[ts], choice[ts], avail[ts]))
    if not rows:
        return {}
    K = max(r[1].shape[1] for r in rows)
    nf = rows[0][1].shape[2]

    def stack(sel):
        F = np.zeros((sum(len(r[2]) for r in sel), K, nf))
        av = np.zeros((F.shape[0], K), dtype=bool)
        ch = np.zeros(F.shape[0], dtype=int)
        i = 0
        for _, f, c, a in sel:
            n, k = len(c), f.shape[1]
            F[i:i + n, :k], av[i:i + n, :k], ch[i:i + n] = f, a, c
            i += n
        return F, ch, av

    def fit(sel):
        F, ch, av = stack(sel)
        out = {}
        for model, spec in _SPECS.items():
            for a in (_GRID if model == "qlearn" else (None,)):
                cols = [names.index(c) for c in spec(a)]
                out[(model, a)] = _clogit(F[:, :, cols], ch, av, l2, np.zeros(len(cols)), maxiter=60)[0]
        return out

    priors = {None: fit(rows)}
    for st in sorted({r[0] for r in rows if r[0] is not None and (studies is None or r[0] in studies)}):
        sel = [r for r in rows if r[0] == st]
        if len(sel) >= min_study:
            priors[st] = fit(sel)
    return priors


# ============================================================================================ pseudo-items and GBDT
MAX_HISTORY_CHARS = 16000


def make_pseudo_items(sessions, per_session: int = 4, seed: int = 0):
    """Items with known responses cut out of COMPLETE sessions (``load_train`` dicts): the response at a random marker with
    >= 2 earlier responses and '<<' before character 16000 becomes the target, the text before it the history.
    Items of round r are the r-th cut of every session; ``group`` = index of the session."""
    rng = np.random.default_rng(seed)
    items = []
    for gi, s in enumerate(sessions):
        elig = [r for i, r in enumerate(s["responses"]) if i >= 2 and int(r["pos"]) <= MAX_HISTORY_CHARS]
        if not elig:
            continue
        pick = rng.permutation(len(elig))[:per_session]
        for rnd, j in enumerate(pick):
            r = elig[int(j)]
            items.append({"study": s["study"], "history": s["text"][: int(r["pos"])], "options": list(r["options"]),
                          "options_text": ", ".join(r["options"]), "target": r["response"], "group": gi, "round": rnd})
    return items


def _session_trials(s) -> Trials:
    """Trials of a ``load_train`` session (complete text; the response after the last marker is unknown)."""
    to = {int(r["pos"]): list(r["options"]) for r in s.get("responses", [])}
    last = s["responses"][-1]["options"] if s.get("responses") else None
    return parse_trials(s["text"], options=last, trial_options=to)


def _item_trials(it) -> Trials:
    return parse_trials(it["history"], options=it["options"])


def _rows(trials: Trials, rng, max_rows: int, t_from: int = 2, include_target: bool = False):
    """Candidate rows of one text: features (n, n_features + n_context), labels, trial-group ids, weights."""
    names, F, ctx, avail, choice = trial_features(trials)
    T = len(choice)
    ts = np.arange(t_from, T)
    if len(ts) > max_rows:
        ts = np.sort(rng.choice(ts, max_rows, replace=False))
    X, y, g = [], [], []
    for t in ts:
        ks = np.flatnonzero(avail[t])
        if len(ks) < 2:
            continue
        for k in ks:
            X.append(np.concatenate([F[t, k], ctx[t]]))
            y.append(float(choice[t] == k))
            g.append(t)
    return (np.asarray(X).reshape(-1, F.shape[2] + ctx.shape[1]), np.asarray(y), np.asarray(g))


class _Gbdt:
    """LightGBM on per-candidate history features; prediction = candidate probabilities normalised within the trial."""

    def __init__(self, seed=0, rows_per_session=60):
        self.seed, self.rows_per_session, self.model = seed, rows_per_session, None

    def fit(self, trial_list):
        import lightgbm as lgb
        rng = np.random.default_rng(self.seed)
        Xs, ys, ws = [], [], []
        for tr in trial_list:
            X, y, g = _rows(tr, rng, self.rows_per_session)
            if len(y):
                Xs.append(X)
                ys.append(y)
                ws.append(np.full(len(y), 1.0 / max(1, len(np.unique(g)))))     # equal total weight per text
        if not Xs or len(np.concatenate(ys)) < 200:
            return self
        X, y, w = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)
        w = w / w.mean()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=8, min_child_samples=40,
                                            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                                            n_jobs=2, random_state=self.seed, verbose=-1).fit(X, y, sample_weight=w)
        return self

    def predict(self, trials: Trials):
        names, F, ctx, avail, choice = trial_features(trials)
        T = len(choice)
        ks = np.flatnonzero(avail[T])
        if self.model is None:
            return None
        X = np.concatenate([F[T][ks], np.repeat(ctx[T][None, :], len(ks), 0)], 1)
        s = self.model.predict_proba(X)[:, 1]
        p = np.zeros(len(trials.keys))
        p[ks] = s / max(s.sum(), 1e-12)
        return p


# ============================================================================================ language-model prompts
PROMPT_STYLES = ("transcript", "long", "model")
_PROMPT_HEAD = 1200                                   # characters of the task instructions kept in a prompt
_PROMPT_TAIL = {"transcript": 9000, "long": 14000, "model": 9000}       # characters of the latest trials kept


def _clip_transcript(history: str, head: int, tail: int) -> str:
    """The first ``head`` characters (task instructions) and the last ``tail`` characters of ``history`` (cut at a line
    start), joined by a '[...]' line; the whole text when it is shorter."""
    if len(history) <= head + tail:
        return history
    rest = history[-tail:]
    nl = rest.find("\n")
    if 0 <= nl < 400:
        rest = rest[nl + 1:]
    return history[:head] + "\n[...]\n" + rest


def build_prompts(items, style: str = "transcript", train_sessions=None, extra_items=None, seed: int = 0) -> list[str]:
    """One text prompt per item for an ``llm`` node (template ``{item}``): the task instructions and the latest trials of
    the transcript (``item["history"]``, at most about 10k / 15k characters, far below the per-request limit), the legal
    keys and the question which key the participant presses next. Styles (:data:`PROMPT_STYLES`): "transcript" (last
    9000 characters of the history), "long" (last 14000), "model" (as "transcript" plus one sentence with the
    probabilities that ``fit_predict(method="personal")`` assigns to the legal keys; ``train_sessions`` / ``extra_items``
    are passed on to that fit). Replies are turned into keys by :func:`parse_answers`."""
    if style not in PROMPT_STYLES:
        raise ValueError(f"unknown prompt style {style!r}; choose from {PROMPT_STYLES}")
    items = list(items)
    probas = None
    if style == "model" and items:
        _, probas = fit_predict(items, train_sessions, "personal", extra_items=extra_items, seed=seed, return_proba=True)
    out = []
    for i, it in enumerate(items):
        opts = [str(o) for o in it["options"]]
        text = _clip_transcript(str(it["history"]), _PROMPT_HEAD, _PROMPT_TAIL[style])
        hint = ""
        if probas is not None:
            hint = ("A statistical model fitted to this participant's earlier responses assigns these probabilities to the "
                    "next response: " + ", ".join(f"{k} {probas[i][k]:.2f}" for k in it["options"]) + ".\n\n")
        out.append(
            "Below is the transcript of one participant in a psychology experiment: the task instructions followed by the "
            "participant's trials. Each response of the participant is written <<KEY>>. The transcript stops right before "
            f"the next response.\n\n{text}\n\n{hint}"
            f"Which key does the participant press next? Legal keys: {', '.join(opts)}. Use the content of the current trial "
            "(the last line) together with how this participant responded in earlier trials. Answer with the key only.")
    return out


def parse_answers(texts, items) -> list:
    """Key named by every reply text of an ``llm`` node, one entry per item (None for an empty / missing reply or when no
    single legal key can be read): the reply itself (quotes, brackets, '<<>>' and end punctuation stripped, an upper-case
    letter accepted for a lower-case reply), else the last '<<KEY>>' in it, else the only legal key that occurs in it as a
    separate token."""
    texts, items = list(texts), list(items)
    if len(texts) != len(items):
        raise ValueError(f"parse_answers: {len(texts)} replies for {len(items)} items")
    out = []
    for t, it in zip(texts, items):
        opts = [str(o) for o in it["options"]]
        ans = None
        if isinstance(t, str) and t.strip():
            norm = t.strip().strip("`'\".:;!,()[]{}<>* \n\t")
            marks = [m for m in _MARK.findall(t) if m in opts]
            hits = [o for o in opts if re.search(r"(?<![A-Za-z0-9])" + re.escape(o) + r"(?![A-Za-z0-9])", t)]
            if norm in opts:
                ans = norm
            elif len(norm) == 1 and norm.upper() in opts:
                ans = norm.upper()
            elif marks:
                ans = marks[-1]
            elif len(hits) == 1:
                ans = hits[0]
        out.append(ans)
    return out


def _llm_share(llm_answers, items):
    """Per item {key: share of the valid answers naming it} (None when no answer is a legal key)."""
    lists = [list(a) for a in llm_answers]
    for a in lists:
        if len(a) != len(items):
            raise ValueError(f"llm_answers: a list has {len(a)} entries for {len(items)} items")
    out = []
    for i, it in enumerate(items):
        opts = [str(o) for o in it["options"]]
        valid = [a[i] for a in lists if a[i] in opts]
        out.append({k: valid.count(k) / len(valid) for k in opts} if valid else None)
    return out


# ============================================================================================ entry points
def _predict_one(tr: Trials, options, method: str, gb, prior=None):
    """Probabilities over ``options`` for one item and the per-model probabilities (dict)."""
    options = list(options)
    ix = {k: i for i, k in enumerate(tr.keys)}
    n = len(options)
    hist = tr.choice_keys
    parts = {}
    ref = None
    if hist:
        cnt = {o: hist.count(o) for o in options}
        best = max(cnt.values())
        tied = {o for o, c in cnt.items() if c == best}
        ref = next((h for h in reversed(hist) if h in tied), options[0]) if best else options[0]
    ref = ref or options[0]
    parts["mode"] = np.array([1.0 if o == ref else 0.0 for o in options])
    parts["last"] = np.array([1.0 if (hist and o == hist[-1]) else 0.0 for o in options])
    if not parts["last"].any():
        parts["last"] = parts["mode"]
    if method in ("kernel", "qlearn", "wsls", "personal", "auto", "all"):
        per = _personal(tr, prior, which=None if method in ("personal", "auto", "all") else (method,))
        for m, (p, bic) in per.items():
            v = np.array([p[ix[o]] if o in ix else 0.0 for o in options])
            parts[m] = v / max(v.sum(), 1e-12)
            parts[m + "_bic"] = bic
    if gb is not None:
        p = gb.predict(tr)
        if p is not None:
            v = np.array([p[ix[o]] if o in ix else 0.0 for o in options])
            parts["gbdt"] = v / max(v.sum(), 1e-12)
    if method == "all" and "gbdt" not in parts:
        parts["gbdt"] = parts["mode"]
    if method in ("personal", "auto", "all"):
        names = [m for m in ("kernel", "qlearn", "wsls") if m in parts]
        bic = np.array([parts[m + "_bic"] for m in names])
        w = np.exp(-(bic - bic.min()) / 2)
        w = w / w.sum()
        parts["personal"] = sum(wi * parts[m] for wi, m in zip(w, names))
        parts["auto"] = 0.5 * parts["personal"] + 0.5 * parts["gbdt"] if "gbdt" in parts else parts["personal"]
    p = parts.get(method if method != "all" else "auto", parts.get("mode"))
    if method == "gbdt" and "gbdt" not in parts:
        p = parts["mode"]
    return p, parts


def fit_predict(items, train_sessions=None, method: str = "auto", extra_items=None, seed: int = 0,
                return_proba: bool = False, population_prior: bool = True, llm_answers=None,
                llm_weight: float = LLM_WEIGHT):
    """Predicted response key for every item (same order; always one of ``item["options"]``).

    ``items``: dicts with "history" and "options". ``train_sessions``: ``load_train`` sessions (used by "gbdt"/"auto").
    ``extra_items``: further items whose histories are added to the fitting rows of "gbdt"/"auto" (e.g. the
    load_dev_inputs items). With ``return_proba=True`` returns ``(keys, [{key: probability}, ...])``.
    ``llm_answers``: optional list of answer lists (one per prompt style / llm run, each a list of one key or None per
    item, e.g. from :func:`parse_answers`); the probabilities of ``method`` are replaced by
    ``(1 - llm_weight) * p + llm_weight * share`` where share[k] is the fraction of the item's valid answers that name k
    (items without a valid answer keep p). With ``method="all"`` the result also has the entries "llm" (majority of the
    answers, ties -> participant mode) and "auto_llm" (that blend of "auto")."""
    if method not in MODELS + ("all",):
        raise ValueError(f"unknown method {method!r}; choose from {MODELS}")
    trials = [_item_trials(it) for it in items]
    if not 0.0 <= llm_weight <= 1.0:
        raise ValueError(f"llm_weight must be in [0, 1], got {llm_weight}")
    shares = _llm_share(llm_answers, items) if llm_answers else None
    priors = {}
    if population_prior and method in ("kernel", "qlearn", "wsls", "personal", "auto", "all"):
        texts = ([(s.get("study"), _session_trials(s)) for s in (train_sessions or [])]
                 + [(x.get("study"), t) for x, t in zip(items, trials)]
                 + [(x.get("study"), _item_trials(x)) for x in (extra_items or [])])
        priors = _fit_prior(texts, seed=seed, studies={x.get("study") for x in items})
    gb = None
    if method in ("gbdt", "auto", "all"):
        pool = [_session_trials(s) for s in (train_sessions or [])] + trials + [_item_trials(x) for x in (extra_items or [])]
        gb = _Gbdt(seed).fit(pool)
    keys, probas, allkeys = [], [], {}
    for i, (it, tr) in enumerate(zip(items, trials)):
        options = list(it["options"])
        pri = priors.get(it.get("study"), priors.get(None)) if priors else None
        p, parts = _predict_one(tr, options, method, gb, pri)
        ref = parts["mode"]

        def mix(v):
            sh = None if shares is None else shares[i]
            return v if sh is None else (1.0 - llm_weight) * v + llm_weight * np.array([sh[str(o)] for o in options])
        p = mix(p)
        j = int(np.argmax(p + 1e-6 * ref))                      # ties -> the participant's most frequent response
        keys.append(options[j])
        probas.append({o: float(x) for o, x in zip(options, p)})
        if method == "all":
            for m, v in parts.items():
                if not m.endswith("_bic"):
                    allkeys.setdefault(m, []).append(options[int(np.argmax(v + 1e-6 * ref))])
            if shares is not None:
                sh = shares[i]
                lv = np.array([sh[str(o)] for o in options]) if sh is not None else ref
                allkeys.setdefault("llm", []).append(options[int(np.argmax(lv + 1e-6 * ref))])
                allkeys.setdefault("auto_llm", []).append(options[int(np.argmax(mix(parts["auto"]) + 1e-6 * ref))])
    if method == "all":
        return allkeys
    return (keys, probas) if return_proba else keys


def cross_validate(sessions, method: str = "auto", n_splits: int = 4, per_session: int = 4, seed: int = 0,
                   extra_methods=(), **fit_kwargs) -> dict:
    """Out-of-fold scores on pseudo-items of complete ``sessions``; sessions (participants) are split into folds so that
    fitting rows (training sessions and item histories) never come from a session that is scored in the same call.
    Each call scores one cut per session; returns accuracy, mode_accuracy, n_items and per-study accuracy
    (plus the same for every method in ``extra_methods``)."""
    pseudo = make_pseudo_items(sessions, per_session, seed)
    rng = np.random.default_rng(seed)
    fold_of = rng.permutation(len(sessions)) % n_splits
    methods = [method] + [m for m in extra_methods if m != method]
    res = {m: {"y": [], "t": [], "h": [], "o": [], "s": []} for m in methods}
    single = len(methods) == 1
    for f in range(n_splits):
        train = [s for i, s in enumerate(sessions) if fold_of[i] != f]
        for rnd in range(per_session):
            its = [x for x in pseudo if fold_of[x["group"]] == f and x["round"] == rnd]
            if not its:
                continue
            preds = {methods[0]: fit_predict(its, train, methods[0], seed=seed, **fit_kwargs)} if single else fit_predict(
                its, train, "all", seed=seed, **fit_kwargs)
            for m in methods:
                pred = preds[m]
                r = res[m]
                r["y"] += pred
                r["t"] += [x["target"] for x in its]
                r["h"] += [x["history"] for x in its]
                r["o"] += [x["options"] for x in its]
                r["s"] += [x["study"] for x in its]
    out = {}
    for m in methods:
        r = res[m]
        d = summarize(r["t"], r["y"], r["h"], r["o"])
        by = {}
        for st in sorted(set(r["s"])):
            ii = [i for i, s in enumerate(r["s"]) if s == st]
            by[st] = accuracy([r["t"][i] for i in ii], [r["y"][i] for i in ii])
        d["by_study"] = by
        out[m] = d
    return out[method] if not extra_methods else out
