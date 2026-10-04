"""Retrieve(D_t; S_r) and Retrieve(D_t; O_r): lexical top-k retrieval of Skills and Operators (paper Eq. 6).

The retriever ranks the components of an :class:`~scienceclaw.core.program.AgentProgram` against the
*episode query* — the objective, task type, tags, discipline and required-output type of the episode —
with a small pure-Python Okapi BM25 over ``Skill.search_text()`` / ``OperatorSpec.search_text()``, plus a
tag / applicability bonus:

* discipline match (skill tag or operator ``contract.applicability["disciplines"]`` / tags),
* task-type match (same sources, ``applicability["task_types"]`` for operators),
* episode-tag overlap,
* input-type overlap (operators only): the port types the episode's tools produce (plus the required
  output type) against ``applicability["input_types"]`` or, if absent, the operator's input port types.

Applicability adds to the score, and a component learned in one discipline stays retrievable for another
when it is *relevant* to the episode (cross-discipline transfer is part of the benchmark): a relevance
floor (:class:`RetrievalWeights` ``floor``; DESIGN decision 8) keeps only components with a metadata match
(discipline, task type or an episode tag among the component's tags / applicability) or a strong lexical
match (BM25 covering at least ``min_lexical_frac`` of the query's idf mass). Without the floor every
component sharing one token with the query would enter every episode's slice (BM25 is positive for any
shared token), which made every gated candidate change the slice of practically every val episode and
defeated lazy re-validation. Components must also have a strictly positive total score. Ranking is
deterministic: score descending (rounded to 1e-9), then id, then version. :meth:`Retriever.slice_hash`
hashes the ordered list of retrieved version ids (plus a content digest per component) and is what lazy
re-validation compares.

The retriever is read-only after construction and therefore safe to share between threads.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Iterable, Sequence, TypeVar

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .operators import OperatorSpec
    from .program import AgentProgram
    from .skills import Skill

__all__ = [
    "Retriever",
    "BM25Index",
    "RetrievalWeights",
    "tokenize",
    "episode_query",
    "episode_input_types",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Small English stop-word list (kept short on purpose: scientific vocabulary is the signal).
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in into is it its of on or that the this to was were will with
    each per via using use used than then there these those their them they which who whom what when where why how
    all any both can may must not no only other some such so very our your you we i""".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens without stop words or 1-character tokens.

    ``snake_case`` / ``kebab-case`` identifiers are split into their parts ("binary_classification" ->
    ["binary", "classification"]); "FoR34" -> ["for34"].
    """
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) > 1 and t not in _STOPWORDS]


def _norm(value: Any) -> str:
    return str(value).strip().lower()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    return [value]


T = TypeVar("T")


class BM25Index(Generic[T]):
    """Okapi BM25 over a fixed list of documents (pure Python, deterministic).

    idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))  (non-negative Lucene variant)
    score(q, d) = sum_{t in unique(q)} idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))

    Query terms are de-duplicated and iterated in sorted order so that floating-point sums do not depend on
    hash-randomized set iteration order.
    """

    def __init__(self, items: Sequence[T], texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        if len(items) != len(texts):
            raise ValueError("items and texts must have the same length")
        self.items: list[T] = list(items)
        self.k1 = float(k1)
        self.b = float(b)
        self._tfs: list[Counter[str]] = [Counter(tokenize(t)) for t in texts]
        self._lens: list[int] = [sum(c.values()) for c in self._tfs]
        n = len(self._tfs)
        self._avgdl: float = (sum(self._lens) / n) if n else 0.0
        df: Counter[str] = Counter()
        for c in self._tfs:
            df.update(c.keys())
        self._idf: dict[str, float] = {t: math.log(1.0 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def __len__(self) -> int:
        return len(self.items)

    def query_mass(self, query_tokens: Iterable[str]) -> float:
        """Sum of the idf of the unique query terms (unseen terms count with df=0): the score of a
        hypothetical average-length document that contains every query term once. Used to express a
        document's BM25 score as a scale-free coverage of the query."""
        n = len(self._tfs)
        unseen = math.log(1.0 + (n + 0.5) / 0.5)
        return sum(self._idf.get(t, unseen) for t in sorted(set(query_tokens)))

    def scores(self, query_tokens: Iterable[str]) -> list[float]:
        terms = sorted(set(query_tokens))
        out: list[float] = []
        for tf, dl in zip(self._tfs, self._lens):
            s = 0.0
            norm = self.k1 * (1.0 - self.b + self.b * (dl / self._avgdl if self._avgdl > 0 else 0.0))
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    s += self._idf.get(t, 0.0) * f * (self.k1 + 1.0) / (f + norm)
            out.append(s)
        return out


@dataclass(frozen=True)
class RetrievalWeights:
    """Bonus weights added to the BM25 score (all configurable; defaults are small relative to BM25)."""
    discipline: float = 1.0
    task_type: float = 1.0
    tag: float = 0.5            # per overlapping episode tag
    tag_cap: float = 1.5        # cap on the total tag bonus
    input_type: float = 0.5     # operators: overlap between episode data types and operator input types
    min_score: float = 0.0      # components need a total score strictly above this to be retrieved
    floor: bool = True          # relevance floor: metadata match OR lexical coverage >= min_lexical_frac
    min_lexical_frac: float = 0.15   # BM25 / query idf mass needed by a component without a metadata match


def episode_query(episode: Any) -> str:
    """Query text for an episode: objective + task_type + tags + discipline + required-output type."""
    req = getattr(episode, "required_output", None)
    parts = [
        str(getattr(episode, "objective", "") or ""),
        str(getattr(episode, "task_type", "") or ""),
        " ".join(str(t) for t in (getattr(episode, "tags", None) or [])),
        str(getattr(episode, "discipline", "") or ""),
        str(getattr(req, "type", "") or ""),
    ]
    return " ".join(p for p in parts if p)


def episode_input_types(episode: Any) -> set[str]:
    """Port types of the data the episode's tools produce, plus the required-output type ("any" excluded)."""
    types: set[str] = set()
    for tool in getattr(episode, "tools", None) or []:
        for sch in (getattr(tool, "outputs", None) or {}).values():
            t = getattr(sch, "type", None)
            if t:
                types.add(_norm(t))
    req = getattr(episode, "required_output", None)
    if getattr(req, "type", None):
        types.add(_norm(req.type))
    types.discard("any")
    return types


def _component_digest(obj: Any) -> str:
    try:
        blob = json.dumps(obj.to_dict(), sort_keys=True, default=str)
    except Exception as ex:  # a component that cannot be serialized still gets a stable, explicit digest
        blob = f"unserializable:{type(obj).__name__}:{getattr(obj, 'version_id', '')}:{type(ex).__name__}"
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


class Retriever:
    """Top-k retrieval of Skills and Operators of one program for an episode (read-only, thread-safe)."""

    def __init__(self, program: "AgentProgram", weights: RetrievalWeights | None = None,
                 k1: float = 1.5, b: float = 0.75) -> None:
        self.program = program
        self.weights = weights or RetrievalWeights()
        skills = sorted(program.skills.values(), key=lambda s: (s.id, s.version))
        ops = sorted(program.operators.values(), key=lambda o: (o.id, o.version))
        self._skill_index: BM25Index["Skill"] = BM25Index(skills, [s.search_text() for s in skills], k1, b)
        self._op_index: BM25Index["OperatorSpec"] = BM25Index(ops, [o.search_text() for o in ops], k1, b)

    # ------------------------------------------------------------------ bonus
    def _skill_meta(self, skill: "Skill", episode: Any) -> float:
        """Discipline / task-type / tag part of the bonus (> 0 iff the skill's metadata matches the episode)."""
        w = self.weights
        tags = {_norm(t) for t in (skill.tags or [])}
        bonus = 0.0
        if getattr(episode, "discipline", None) and _norm(episode.discipline) in tags:
            bonus += w.discipline
        if getattr(episode, "task_type", None) and _norm(episode.task_type) in tags:
            bonus += w.task_type
        ep_tags = {_norm(t) for t in (getattr(episode, "tags", None) or [])}
        bonus += min(w.tag_cap, w.tag * len(ep_tags & tags))
        return bonus

    def _skill_bonus(self, skill: "Skill", episode: Any) -> float:
        return self._skill_meta(skill, episode)

    def _op_meta(self, op: "OperatorSpec", episode: Any) -> float:
        """Discipline / task-type / tag part of the bonus (> 0 iff the operator's metadata matches the episode)."""
        w = self.weights
        app = dict(getattr(op.contract, "applicability", {}) or {})
        tags = {_norm(t) for t in (op.tags or [])}
        disciplines = {_norm(d) for d in _as_list(app.get("disciplines"))} | tags
        task_types = {_norm(t) for t in _as_list(app.get("task_types"))} | tags
        bonus = 0.0
        if getattr(episode, "discipline", None) and _norm(episode.discipline) in disciplines:
            bonus += w.discipline
        if getattr(episode, "task_type", None) and _norm(episode.task_type) in task_types:
            bonus += w.task_type
        ep_tags = {_norm(t) for t in (getattr(episode, "tags", None) or [])}
        bonus += min(w.tag_cap, w.tag * len(ep_tags & tags))
        return bonus

    def _op_bonus(self, op: "OperatorSpec", episode: Any) -> float:
        w = self.weights
        app = dict(getattr(op.contract, "applicability", {}) or {})
        bonus = self._op_meta(op, episode)
        declared = {_norm(t) for t in _as_list(app.get("input_types"))}
        if not declared:
            declared = {_norm(p.type) for p in op.inputs.values()}
        declared.discard("any")
        if declared & episode_input_types(episode):
            bonus += w.input_type
        return bonus

    # ---------------------------------------------------------------- ranking
    def _rank(self, index: BM25Index, bonus_fn, meta_fn, episode: Any) -> list[tuple[float, Any]]:
        if len(index) == 0:
            return []
        w = self.weights
        q = tokenize(episode_query(episode))
        base = index.scores(q)
        mass = index.query_mass(q) if w.floor else 0.0
        scored = []
        for lex, item in zip(base, index.items):
            total = round(lex + bonus_fn(item, episode), 9)
            if total <= w.min_score:
                continue
            if w.floor and meta_fn(item, episode) <= 0.0 and not (mass > 0.0 and lex / mass >= w.min_lexical_frac):
                continue    # relevance floor: no discipline / task-type / tag match and no strong lexical match
            scored.append((total, item))
        scored.sort(key=lambda p: (-p[0], p[1].id, p[1].version))
        return scored

    def score_skills(self, episode: Any) -> list[tuple[float, "Skill"]]:
        """All retrievable skills with their scores, best first (for diagnostics and tests)."""
        return self._rank(self._skill_index, self._skill_bonus, self._skill_meta, episode)

    def score_operators(self, episode: Any) -> list[tuple[float, "OperatorSpec"]]:
        """All retrievable operators with their scores, best first (for diagnostics and tests)."""
        return self._rank(self._op_index, self._op_bonus, self._op_meta, episode)

    def skills(self, episode: Any, k: int) -> list["Skill"]:
        """RS_{t,r}: top-k Skills for the episode."""
        if k <= 0:
            return []
        return [s for _, s in self.score_skills(episode)[:k]]

    def operators(self, episode: Any, k: int) -> list["OperatorSpec"]:
        """RO_{t,r}: top-k Operators for the episode."""
        if k <= 0:
            return []
        return [o for _, o in self.score_operators(episode)[:k]]

    def slice_hash(self, episode: Any, k_skills: int, k_ops: int) -> str:
        """sha256 of the ordered retrieved version ids (each with a content digest).

        Two programs produce the same hash for an episode iff the policy would see the same retrieved
        components in the same order, which is what lazy re-validation needs (DESIGN decision 8).
        """
        payload = {
            "skills": [f"{s.version_id}#{_component_digest(s)}" for s in self.skills(episode, k_skills)],
            "operators": [f"{o.version_id}#{_component_digest(o)}" for o in self.operators(episode, k_ops)],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
