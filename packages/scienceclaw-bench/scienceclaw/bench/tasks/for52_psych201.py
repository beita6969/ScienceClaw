"""FoR52 Psychology — Psych-201 discrete sequential choice prediction, metric micro accuracy (max).

Data: Hugging Face ``marcelbinz/Psych-201-discrete`` at commit 060062064d00766ea0b5666c73268329d0f556b6
(Apache-2.0; all four parquet shards, 131,834 participant sessions from 103 studies) as downloaded and verified by
the data team under ``<DATA_ROOT>/for52-psych201-discrete/reconstructed_v1/population`` with its row index
``population-index-and-exclusions.parquet``. Each row is one participant session written as natural-language text
in which every human response is enclosed in ``<<`` ``>>``.

* **Item** = one response marker of one participant session: input = the complete session text before the
  marker's ``<<`` (instructions + all earlier trials with the participant's own earlier responses and outcomes,
  i.e. teacher-forced history) and the finite set of legal response keys; target = the participant's actual
  response (a single uppercase key). Legal keys are parsed only from text before the marker with the data team's
  explicit option templates (rule ``strict_single_key_complete_group_v1``: dynamic "You can choose between ..."
  lines, current lottery descriptions, and study-specific fixed key maps stated in the instructions).
* **Units / leakage.** One session (the first row) per participant group (study + participant) and exactly one
  target marker per session, fixed by a seed-independent hash among the session's eligible markers (at least two
  earlier responses; history <= 16,000 characters). A participant therefore contributes at most one item, and no
  item's history can contain another item's target.
* **Pools.** Studies with >= 16 qualified sessions are ranked by ``sha256``; every third study is an **OOD study**
  (held-out experiment paradigms; ``lineage["ood_kind"] = "proxy_within_dataset"`` — same corpus, new studies).
  In IID studies the sessions of each study are ranked by hash: the first 6 non-flagged sessions are visible
  **train** sessions, the next 2 **dev** sessions, the rest are cut into src / val / id (55/15/30 %). Groups
  flagged ``in_eval`` / ``is_psych101_test`` upstream are never used as visible data.
* **Visible data (D_E).** ``load_train``: 32 complete training sessions of IID studies (other participants; text
  truncated at 24,000 characters) with their parsed responses; ``load_dev_inputs`` / ``score_dev``: 16 dev items
  (IID dev sessions) and their accuracy; ``load_eval_inputs``: the 16 items (study id, history, options).
* **Metric (D_V).** Micro accuracy = correct predictions / items (the data-team scorer ``score_response``: a
  prediction outside the legal keys is a failure; here it is additionally a hard-constraint violation).
* **Reference baseline** (deterministic, visible input only): the participant's most frequent earlier response
  among the current legal keys (ties -> most recent; no earlier response -> first listed key).
  **Acceptance:** ``accuracy >= reference + 0.05``.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import scilib

from ...core.schema import PortSchema
from ...core.trace import Trace
from ..task import Budget, EvalResult, Episode, ToolSpec
from ._adapter_utils_for36_46_49_52 import (
    PARTITION_SEED, Lazy, as_str_list, c_allowed_labels, c_str_list, cache_dir, check_split, draw_blocks,
    episode_id, episode_rng, hash_rank, norm_score, partition_ids, receipt_summary, resolve_data_root, sha_hex,
)

CODE = "FoR52"
FAMILY = "Social & behavior"
DATASET_DIR = "for52-psych201-discrete"
POP_SUBDIR = "reconstructed_v1"
MARKER = re.compile(r"<<([^<>]*)>>")
CAND_PER_STUDY = 72          # hash-ranked candidate sessions parsed per study
KEEP_PER_STUDY = 48          # qualified sessions kept per study
MIN_UNITS_STUDY = 16
N_TRAIN_PER_STUDY = 6
N_DEV_PER_STUDY = 2
IID_EVAL_FRACTIONS = {"src": 0.55, "val": 0.15, "id": 0.30}
MIN_PRIOR_RESPONSES = 2
MAX_HISTORY_CHARS = 16_000
MAX_TRAIN_CHARS = 24_000
N_TRAIN_SESSIONS = 32
N_DEV_ITEMS = 48           # = all dev sessions of the IID studies (a 16-item dev accuracy has a standard error of ~0.12)
LLM_BUDGET_DEV_ITEMS = 16  # the LLM-item budget is 3 x (eval items + 16), independent of the dev size
ACCEPT_MARGIN = 0.05
_CACHE_VERSION = "v1"

# A fixed, visible-data route for tool-on runs.  The route deliberately uses the
# history-only Q-learning choice model already shipped in ``scilib.psych``: it
# does not download or fit a new external model, and its only fitting rows are
# the participant histories plus the visible ``load_train`` sessions.  Keeping
# the method and seed here makes the route auditable instead of inheriting a
# mutable library default.  This is a prediction tool, not a scorer/reference
# change.
PSYCH_TOOL_METHOD = "qlearn"
PSYCH_TOOL_SEED = 0
PSYCH_TOOL_POPULATION_PRIOR = True
PSYCH_TOOL_CONFIG = {
    "method": PSYCH_TOOL_METHOD,
    "seed": PSYCH_TOOL_SEED,
    "population_prior": PSYCH_TOOL_POPULATION_PRIOR,
    "data": "visible load_train sessions and label-free load_eval_inputs items",
}

# A second, still label-free route for optimization batches.  It chooses among
# the shipped history models by grouped out-of-fold accuracy on the visible
# participant sessions only, then refits the selected model on those sessions.
# It never reads evaluation targets and does not alter the fixed qlearn route
# used by the existing formal manifest.
PSYCH_ADAPTIVE_METHODS = ("qlearn", "personal", "wsls")

# A split-agnostic route for mixed IID/OOD tool calls.  ``load_train`` carries
# the study identifiers of the visible IID population.  When an evaluation
# study is represented in that payload, the participant-specific q-learning
# route has the strongest visible support; for an unseen study, the pooled
# history GBDT is the available population fallback.  The switch uses only
# study identifiers and the two label-free payloads, never a split name or a
# target.  Keep this separate from the formal fixed route so old manifests and
# their required-tool lineage remain unchanged.
PSYCH_DOMAIN_SEEN_METHOD = "qlearn"
PSYCH_DOMAIN_UNSEEN_METHOD = "gbdt"


# ================================================================================================ option parsing
_DYNAMIC = (
    ("two_options", re.compile(r"You can choose between option ([A-Z]) and option ([A-Z])\. You press $")),
    ("two_machines", re.compile(r"You can choose between machines ([A-Z]) and ([A-Z])\. You press $")),
    ("two_stimuli", re.compile(r"You encounter stimuli ([A-Z]), ([A-Z])\. You press $")),
)
_LOTTERY_STUDIES = {"peterson2021using", "rutledge2023happiness", "wulff2018description", "frey2017mpl",
                    "olschewski2025optimal"}
_FIXED = {
    "binz2022heuristics": r"two aliens, labelled ([A-Z]) and ([A-Z])\.",
    "badham2017deficits": r"belongs to the ([A-Z]) or ([A-Z]) category",
    "steingroever2015data": r"four decks of cards labeled ([A-Z]), ([A-Z]), ([A-Z]), and ([A-Z])\.",
    "somerville2017charting": r"two slot machines, labeled ([A-Z]) and ([A-Z])\.",
    "feng2021dynamics": r"two slot machines, labeled ([A-Z]) and ([A-Z])\.",
    "wilson2014humans": r"two slot machines, labeled ([A-Z]) and ([A-Z])\.",
    "waltz2020differential": r"two slot machines, labeled ([A-Z]) and ([A-Z])\.",
    "dubois2022value": r"three apple trees, labeled ([A-Z]), ([A-Z]), and ([A-Z])\.",
    "gershman2020reward": r"The three responses available are ([A-Z]), ([A-Z]), and ([A-Z])\.",
    "guenther2023grammaticality": r"Press the ([A-Z]) key on your keyboard if the sentence is grammatically correct, "
                                  r"and the ([A-Z]) key if it is not grammatically correct\.",
    "enkavi2019adaptivenback": r"press ([A-Z]), otherwise press ([A-Z])\.",
    "enkavi2019recentprobes": r"If you think it was, you have to press ([A-Z])\. If you think it was not, press ([A-Z])\.",
    "dezfouli2019": r"choose between two options: ([A-Z]) and ([A-Z])\.",
    "breslav2022shuffle": r"two decks of face-down cards labeled ([A-Z]) and ([A-Z])\.",
    "nasioulas2024feedback": r"by pressing ([A-Z]) or ([A-Z]), for choosing the Left or the Right option",
    "ruggeri2022globalizability": r"multiple choices between two options ([A-Z]) and ([A-Z])\.",
    "thoma2025problearn": r"tapping on either house '([A-Z])' or house '([A-Z])'",
    "guenther2020TS": r"press the ([A-Z]) key on your keyboard\. If you feel that a word does not make sense, press "
                      r"the ([A-Z]) key\.",
}
_FIXED_RE = {k: re.compile(v) for k, v in _FIXED.items()}


def legal_options(text: str, pos: int, study: str, intro: str) -> tuple[list[str], str]:
    """Legal keys for the response starting at ``text[pos:]`` ('<<'), using only ``text[:pos]``.

    Port of the data team's ``psych201_strict_adapter.legal_options`` (explicit templates only; the response and
    all later text are never read). ``intro`` = text before the session's first ``<<``.
    """
    line = text[text.rfind("\n", 0, pos) + 1:pos]
    for name, pat in _DYNAMIC:
        m = pat.search(line)
        if m and len(set(m.groups())) == 2:
            return list(m.groups()), name
    if study == "pirrone_2018_dots":
        m = re.search(r"Press ([A-Z]) for [^\n]+ dots and ([A-Z]) for [^\n]+ dots\. You press $", line)
        if m and len(set(m.groups())) == 2:
            return list(m.groups()), "two_dot_arrays"
    if study == "marshall_2022_brightness" and line.endswith("You press "):
        keys = re.findall(r"brightest press ([A-Z])\.", line)
        if len(keys) == 3 and len(set(keys)) == 3:
            return keys, "three_brightness_options"
    if study == "hebart2023things" and line.endswith("You press "):
        keys = re.findall(r"(?:^|, |and )([A-Z]): ", line)
        if len(keys) == 3 and len(set(keys)) == 3:
            return keys, "three_odd_objects"
    if study in _LOTTERY_STUDIES:
        seg = text[text.rfind(">>", 0, pos) + 2 if text.rfind(">>", 0, pos) >= 0 else 0:pos]
        keys = list(dict.fromkeys(re.findall(r"(?:Option|Lottery) ([A-Z])(?::| delivers| offers)", seg)))
        if len(keys) == 2 and re.search(r"You (?:press|choose|chose lottery) $", seg):
            return keys, "two_current_lotteries"
    if study in _FIXED_RE:
        m = _FIXED_RE[study].search(intro)
        if m and len(set(m.groups())) == len(m.groups()):
            return list(m.groups()), "fixed_" + study
    if study == "frey2017risk":
        a = re.search(r"pump up the balloon by pressing ([A-Z])", intro)
        b = re.search(r"stop pumping up the balloon by pressing ([A-Z])", intro)
        if a and b and a[1] != b[1]:
            return [a[1], b[1]], "balloon_pump_collect"
    return [], "unsupported"


def parse_markers(text: str, study: str, limit_chars: int | None = None) -> list[dict]:
    """All response markers (up to ``limit_chars``): start, target, options, valid flag."""
    intro = text.split("<<", 1)[0]
    out = []
    for i, m in enumerate(MARKER.finditer(text)):
        if limit_chars is not None and m.start() > limit_chars:
            break
        opts, rule = legal_options(text, m.start(), study, intro)
        tgt = m.group(1)
        valid = bool(re.fullmatch(r"[A-Z]", tgt)) and len(opts) >= 2 and tgt in opts
        out.append({"index": i, "start": m.start(), "end": m.end(), "target": tgt, "options": opts, "rule": rule,
                    "valid": valid})
    return out


def previous_responses(history: str) -> list[str]:
    return MARKER.findall(history)


def reference_prediction(history: str, options: list[str]) -> str:
    """Most frequent earlier response of this participant among the legal keys (ties -> most recent)."""
    prev = [r for r in previous_responses(history) if r in options]
    if not prev:
        return options[0]
    counts = {o: prev.count(o) for o in options}
    best = max(counts.values())
    tied = {o for o, c in counts.items() if c == best}
    for r in reversed(prev):
        if r in tied:
            return r
    return options[0]


def visible_study_prediction(train_sessions: list[Any], study: str, options: list[str], history: str = "") -> str:
    """Predict from response frequencies of visible *other participants* in one study.

    This is a trusted-side, same-metric reference for pooled reports.  It only consumes the sessions returned by
    ``load_train`` and the current item's legal options; it never reads the current participant's target or any
    evaluation label.  If the visible study rows contain no usable responses, the existing participant-history
    reference is used as a deterministic fallback.
    """
    counts = {str(o): 0 for o in options}
    for session in train_sessions or []:
        if getattr(session, "study", None) != study:
            continue
        for response in getattr(session, "responses", ()) or ():
            key = str(response.get("response", "")) if isinstance(response, dict) else ""
            if key in counts:
                counts[key] += 1
    best = max(counts.values(), default=0)
    if best <= 0:
        return reference_prediction(history, options) if history else (options[0] if options else "")
    # Preserve the item's option order for ties; this is independent of session/order iteration.
    return next(o for o in options if counts[str(o)] == best)


def _validate_psych_inputs(train_sessions: Any, eval_items: Any) -> tuple[list[dict], list[dict]]:
    """Normalize and validate the two label-free tool payloads shared by predictor routes."""
    if not isinstance(train_sessions, list) or not isinstance(eval_items, list):
        raise TypeError("train_sessions and eval_items must be lists")
    train = []
    for i, row in enumerate(train_sessions):
        if not isinstance(row, dict):
            raise TypeError(f"train_sessions[{i}] must be a dict")
        extra = sorted(set(row) - {"study", "text", "responses"})
        missing = sorted({"study", "text", "responses"} - set(row))
        if extra or missing:
            raise ValueError(f"train_sessions[{i}] schema mismatch; missing={missing}, forbidden={extra}")
        if not isinstance(row["responses"], list):
            raise TypeError(f"train_sessions[{i}].responses must be a list")
        responses = []
        for j, response in enumerate(row["responses"]):
            if not isinstance(response, dict):
                raise TypeError(f"train_sessions[{i}].responses[{j}] must be a dict")
            rextra = sorted(set(response) - {"pos", "response", "options"})
            rmissing = sorted({"pos", "response", "options"} - set(response))
            if rextra or rmissing:
                raise ValueError(f"train_sessions[{i}].responses[{j}] schema mismatch; missing={rmissing}, forbidden={rextra}")
            responses.append({"pos": response["pos"], "response": response["response"],
                              "options": list(response["options"])})
        train.append({"study": str(row["study"]), "text": str(row["text"]), "responses": responses})

    items = []
    for i, row in enumerate(eval_items):
        if not isinstance(row, dict):
            raise TypeError(f"eval_items[{i}] must be a dict")
        required = {"study", "history", "options", "options_text"}
        extra = sorted(set(row) - required)
        missing = sorted(required - set(row))
        if extra or missing:
            raise ValueError(f"eval_items[{i}] schema mismatch; missing={missing}, forbidden={extra}")
        options = list(row["options"])
        if len(options) < 2 or any(not isinstance(o, str) for o in options):
            raise ValueError(f"eval_items[{i}].options must contain at least two string keys")
        items.append({"study": str(row["study"]), "history": str(row["history"]),
                      "options": options, "options_text": str(row["options_text"])})
    return train, items


def _fixed_fit_predict(train_sessions: Any, eval_items: Any) -> tuple[list[str], dict[str, Any]]:
    """Run the fixed visible-data psych route used by the agent-facing ToolSpec.

    The two inputs are exactly the payloads returned by ``load_train`` and
    ``load_eval_inputs``.  The validation is intentionally strict so a caller
    cannot smuggle targets or other evaluation columns into this route.  The
    q-learning likelihood fit is the existing CPU-only history model; no new
    model weights are trained or downloaded by the tool.
    """
    train, items = _validate_psych_inputs(train_sessions, eval_items)

    pred = scilib.psych.fit_predict(items, train, method=PSYCH_TOOL_METHOD,
                                    seed=PSYCH_TOOL_SEED,
                                    population_prior=PSYCH_TOOL_POPULATION_PRIOR)
    if not isinstance(pred, list) or len(pred) != len(items):
        raise ValueError("fixed psych route returned the wrong number of predictions")
    if any(p not in it["options"] for p, it in zip(pred, items)):
        raise ValueError("fixed psych route returned an illegal response key")
    info = dict(PSYCH_TOOL_CONFIG)
    info.update({"n_train_sessions": len(train), "n_eval_items": len(items),
                 "method_impl": "scilib.psych.fit_predict"})
    return pred, info


def _adaptive_fit_predict(train_sessions: Any, eval_items: Any) -> tuple[list[str], dict[str, Any]]:
    """Select a shipped history model by grouped OOF on visible sessions, then predict eval items.

    The selection uses only complete visible training sessions.  In particular, no evaluation item target,
    scorer output, or hidden field is accepted by this helper.  The selected model is deterministic for the
    fixed seed and candidate order, and its name/scores are returned as provenance for route audits.
    """
    # Reuse the strict schema normalization of the fixed route before fitting.
    normalized_train, normalized_items = _validate_psych_inputs(train_sessions, eval_items)
    cv = scilib.psych.cross_validate(
        normalized_train, method=PSYCH_ADAPTIVE_METHODS[0],
        extra_methods=PSYCH_ADAPTIVE_METHODS[1:], n_splits=4, per_session=4, seed=PSYCH_TOOL_SEED,
    )
    scores = {m: float(cv[m]["accuracy"]) for m in PSYCH_ADAPTIVE_METHODS}
    # Stable order resolves equal OOF scores without inspecting evaluation data.
    selected = max(PSYCH_ADAPTIVE_METHODS, key=lambda m: (scores[m], -PSYCH_ADAPTIVE_METHODS.index(m)))
    pred = scilib.psych.fit_predict(normalized_items, normalized_train, method=selected,
                                    seed=PSYCH_TOOL_SEED, population_prior=PSYCH_TOOL_POPULATION_PRIOR)
    if not isinstance(pred, list) or len(pred) != len(normalized_items):
        raise ValueError("adaptive psych route returned the wrong number of predictions")
    if any(p not in it["options"] for p, it in zip(pred, normalized_items)):
        raise ValueError("adaptive psych route returned an illegal response key")
    info = {"methods": list(PSYCH_ADAPTIVE_METHODS), "selected_method": selected,
            "cv_accuracy": scores, "cv_n_splits": 4, "cv_per_session": 4,
            "seed": PSYCH_TOOL_SEED, "population_prior": PSYCH_TOOL_POPULATION_PRIOR,
            "data": "visible load_train sessions only", "method_impl": "scilib.psych.cross_validate+fit_predict"}
    return pred, info


def _domain_adaptive_fit_predict(train_sessions: Any, eval_items: Any) -> tuple[list[str], dict[str, Any]]:
    """Use a participant model for visible studies and a pooled model for unseen studies.

    The route is deliberately keyed by the study IDs present in ``load_train`` rather than by a caller-provided
    ``split`` flag.  This keeps the decision available to an agent while preserving the held-out-study boundary:
    an OOD study is absent from the visible training payload and therefore uses the population GBDT; an IID study
    uses the q-learning history model.  ``fit_predict(method="all")`` fits every shipped candidate from the same
    label-free payload, so the switch does not introduce an extra fit or a target-bearing side channel.
    """
    normalized_train, normalized_items = _validate_psych_inputs(train_sessions, eval_items)
    all_pred = scilib.psych.fit_predict(
        normalized_items, normalized_train, method="all", seed=PSYCH_TOOL_SEED,
        population_prior=PSYCH_TOOL_POPULATION_PRIOR,
    )
    if not isinstance(all_pred, dict) or PSYCH_DOMAIN_SEEN_METHOD not in all_pred:
        raise ValueError("domain-adaptive psych route did not return qlearn predictions")
    if PSYCH_DOMAIN_UNSEEN_METHOD not in all_pred:
        raise ValueError("domain-adaptive psych route did not return gbdt predictions")
    seen = {str(s["study"]) for s in normalized_train}
    pred = [
        (all_pred[PSYCH_DOMAIN_SEEN_METHOD][i]
         if str(item["study"]) in seen else all_pred[PSYCH_DOMAIN_UNSEEN_METHOD][i])
        for i, item in enumerate(normalized_items)
    ]
    if not isinstance(pred, list) or len(pred) != len(normalized_items):
        raise ValueError("domain-adaptive psych route returned the wrong number of predictions")
    if any(p not in it["options"] for p, it in zip(pred, normalized_items)):
        raise ValueError("domain-adaptive psych route returned an illegal response key")
    n_seen = sum(str(item["study"]) in seen for item in normalized_items)
    info = {
        "seen_method": PSYCH_DOMAIN_SEEN_METHOD,
        "unseen_method": PSYCH_DOMAIN_UNSEEN_METHOD,
        "n_visible_studies": len(seen),
        "n_seen_items": n_seen,
        "n_unseen_items": len(normalized_items) - n_seen,
        "seed": PSYCH_TOOL_SEED,
        "population_prior": PSYCH_TOOL_POPULATION_PRIOR,
        "data": "visible load_train sessions and label-free load_eval_inputs items",
        "method_impl": "scilib.psych.fit_predict(method=all)+visible-study switch",
    }
    return pred, info


# ================================================================================================ index
def _truncate_at_line(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text.rfind("\n", 0, limit)
    return text[: cut if cut > 0 else limit]


def build_index(root: Path, log: Any = None) -> dict:
    """Parse hash-ranked candidate sessions of every study and fix one target marker per qualified session."""
    import pyarrow.parquet as pq

    base = root / DATASET_DIR / POP_SUBDIR
    idx = pq.read_table(base / "population-index-and-exclusions.parquet",
                        columns=["source_shard", "row_in_shard", "study", "participant", "in_eval",
                                 "is_psych101_test", "group_id"]).to_pandas()
    idx["flag"] = idx["in_eval"].astype(bool) | idx["is_psych101_test"].astype(bool)
    flag_by_group = idx.groupby("group_id")["flag"].any().to_dict()
    first = idx.sort_values(["source_shard", "row_in_shard"]).drop_duplicates("group_id", keep="first")
    cands: list[dict] = []
    for study, g in first.groupby("study"):
        ranked = hash_rank(g["group_id"].tolist(), f"{CODE}|{PARTITION_SEED}|cand|{study}")[:CAND_PER_STUDY]
        sub = g.set_index("group_id").loc[ranked]
        for gid, r in sub.iterrows():
            cands.append({"group_id": gid, "study": study, "shard": r["source_shard"], "row": int(r["row_in_shard"]),
                          "participant": str(r["participant"]), "flagged": bool(flag_by_group.get(gid, False))})
    need: dict[str, dict[int, list[dict]]] = {}
    for c in cands:
        need.setdefault(c["shard"], {}).setdefault(c["row"], []).append(c)
    for shard, rows in sorted(need.items()):
        pf = pq.ParquetFile(base / "population" / shard)
        starts = np.cumsum([0] + [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)])
        by_rg: dict[int, list[int]] = {}
        for row in rows:
            by_rg.setdefault(int(np.searchsorted(starts, row, side="right") - 1), []).append(row)
        for rg, rws in sorted(by_rg.items()):
            col = pf.read_row_group(rg, columns=["text"]).column("text")
            for row in rws:
                txt = col[row - int(starts[rg])].as_py()
                for c in rows[row]:
                    c["text"] = txt
        if log:
            log(f"read {shard}")
    sessions: list[dict] = []
    per_study: dict[str, int] = {}
    for c in cands:
        text = c.pop("text", None)
        if not text or per_study.get(c["study"], 0) >= KEEP_PER_STUDY:
            continue
        marks = parse_markers(text, c["study"], limit_chars=max(MAX_HISTORY_CHARS, MAX_TRAIN_CHARS))
        eligible = [mk for mk in marks if mk["valid"] and mk["index"] >= MIN_PRIOR_RESPONSES
                    and mk["start"] <= MAX_HISTORY_CHARS]
        if sum(mk["valid"] for mk in marks) < 3 or not eligible:
            continue
        tgt = eligible[int(sha_hex(CODE, PARTITION_SEED, "target", c["group_id"])[:12], 16) % len(eligible)]
        kept = _truncate_at_line(text, MAX_TRAIN_CHARS)
        if len(kept) < tgt["end"]:
            kept = text[: tgt["end"]]
        c.update({"text": kept, "n_chars_full": len(text), "target_start": tgt["start"], "target": tgt["target"],
                  "target_index": tgt["index"], "options": tgt["options"], "rule": tgt["rule"],
                  "responses": [{"pos": mk["start"], "response": mk["target"], "options": mk["options"]}
                                for mk in marks if mk["valid"] and mk["end"] <= len(kept)]})
        sessions.append(c)
        per_study[c["study"]] = per_study.get(c["study"], 0) + 1
    return {"version": _CACHE_VERSION, "sessions": sessions,
            "params": {"cand_per_study": CAND_PER_STUDY, "keep_per_study": KEEP_PER_STUDY,
                       "max_history_chars": MAX_HISTORY_CHARS, "max_train_chars": MAX_TRAIN_CHARS,
                       "min_prior_responses": MIN_PRIOR_RESPONSES, "partition_seed": PARTITION_SEED}}


def load_index(root: Path) -> dict:
    f = cache_dir(CODE) / f"index_{_CACHE_VERSION}.json.gz"
    if f.exists():
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    data = build_index(root)
    tmp = f.with_name(f.name + f".tmp{os.getpid()}-{threading.get_ident()}")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(data, fh)
    tmp.replace(f)
    return data


@dataclass
class _Session:
    uid: str
    study: str
    group_id: str
    flagged: bool
    text: str
    target_start: int
    target: str
    options: list[str]
    responses: list[dict]

    @property
    def history(self) -> str:
        return self.text[: self.target_start]

    def item(self) -> dict:
        return {"study": self.study, "history": self.history, "options": list(self.options),
                "options_text": ", ".join(self.options)}


@dataclass
class _PsychData:
    sessions: dict[str, _Session]
    pools: dict[str, list[str]]           # train / dev / src / val / id / ood -> uids
    iid_studies: list[str]
    ood_studies: list[str]
    receipt: dict


def _load(root: Path) -> _PsychData:
    index = load_index(root)
    sessions = {}
    for s in index["sessions"]:
        uid = f"psych201/{s['study']}/{s['group_id'][:16]}"
        sessions[uid] = _Session(uid=uid, study=s["study"], group_id=s["group_id"], flagged=bool(s["flagged"]),
                                 text=s["text"], target_start=int(s["target_start"]), target=s["target"],
                                 options=list(s["options"]), responses=list(s["responses"]))
    by_study: dict[str, list[str]] = {}
    for uid, s in sessions.items():
        by_study.setdefault(s.study, []).append(uid)
    supported = sorted(st for st, u in by_study.items() if len(u) >= MIN_UNITS_STUDY)
    ranked = hash_rank(supported, f"{CODE}|{PARTITION_SEED}|studies")
    ood_st = sorted(ranked[::3])
    iid_st = sorted(set(supported) - set(ood_st))
    pools: dict[str, list[str]] = {k: [] for k in ("train", "dev", "src", "val", "id", "ood")}
    for st in ood_st:
        pools["ood"].extend(hash_rank(by_study[st], f"{CODE}|{PARTITION_SEED}|ood|{st}"))
    for st in iid_st:
        order = hash_rank(by_study[st], f"{CODE}|{PARTITION_SEED}|roles|{st}")
        clean = [u for u in order if not sessions[u].flagged]
        train = clean[:N_TRAIN_PER_STUDY]
        dev = clean[N_TRAIN_PER_STUDY:N_TRAIN_PER_STUDY + N_DEV_PER_STUDY]
        rest = [u for u in order if u not in set(train) | set(dev)]
        pools["train"].extend(train)
        pools["dev"].extend(dev)
        for k, v in partition_ids(rest, IID_EVAL_FRACTIONS, f"{CODE}|{PARTITION_SEED}|eval|{st}").items():
            pools[k].extend(v)
    receipt = receipt_summary(root / DATASET_DIR / "receipt.json")
    return _PsychData(sessions=sessions, pools=pools, iid_studies=iid_st, ood_studies=ood_st, receipt=receipt)


# ================================================================================================ adapter
class Psych201Adapter:
    """Psych-201 next-response prediction adapter (``bench.task.TaskAdapter`` protocol)."""

    discipline = CODE
    name = "Psych-201 discrete (sequential choice prediction)"
    family = FAMILY
    metric = "micro accuracy"
    direction = "max"
    task_type = "behavior_prediction"

    def __init__(self, data_root: str | os.PathLike | None = None, accept_margin: float = ACCEPT_MARGIN,
                 n_train_sessions: int = N_TRAIN_SESSIONS, n_dev: int = N_DEV_ITEMS,
                 budget: Budget | None = None) -> None:
        self.root = resolve_data_root(data_root)
        self.accept_margin = float(accept_margin)
        self.n_train_sessions = int(n_train_sessions)
        self.n_dev = int(n_dev)
        self.budget = budget
        self._data = Lazy(lambda: _load(self.root))

    # ------------------------------------------------------------------------------------------ protocol
    def available(self) -> tuple[bool, str]:
        base = self.root / DATASET_DIR / POP_SUBDIR
        cached = (cache_dir(CODE) / f"index_{_CACHE_VERSION}.json.gz").exists()
        if not cached:
            if not (base / "population-index-and-exclusions.parquet").exists():
                return False, f"missing {base / 'population-index-and-exclusions.parquet'}"
            shards = sorted((base / "population").glob("train-*-of-00004.parquet"))
            if len(shards) != 4:
                return False, f"expected 4 complete Psych-201 shards in {base / 'population'}, found {len(shards)}"
            try:
                import pyarrow  # noqa: F401
            except ImportError:
                return False, "pyarrow is required to read the Psych-201 parquet shards"
        try:
            data = self._data.get()
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as ex:
            return False, f"FoR52 data unreadable: {type(ex).__name__}: {ex}"
        sizes = {k: len(v) for k, v in data.pools.items()}
        return True, (f"{len(data.sessions)} qualified sessions; {len(data.iid_studies)} IID / "
                      f"{len(data.ood_studies)} OOD studies; pools {sizes}")

    def build_episodes(self, split: str, n: int, seed: int, items_per_episode: int = 16) -> list[Episode]:
        check_split(split)
        if items_per_episode < 1:
            raise ValueError("items_per_episode must be >= 1")
        data = self._data.get()
        rng = episode_rng(CODE, split, seed)
        blocks = draw_blocks({"all": data.pools[split]}, {"all": items_per_episode}, int(n), rng, f"{CODE}/{split}")
        return [self._episode(data, split, int(seed), k, ids) for k, ids in enumerate(blocks)]

    def pooled_metric(self, per_episode: list[dict]) -> float | None:
        """Micro accuracy over all items (an invalid output counts as all-wrong, as in the data-team scorer)."""
        correct: list[int] = []
        for p in per_episode:
            if not p:
                continue
            t = list(p.get("y_true") or [])
            pred = p.get("y_pred")
            if pred is None:
                correct.extend([0] * len(t))
                continue
            if len(pred) != len(t):
                raise ValueError("FoR52 pooled payload: y_pred and y_true differ in length")
            correct.extend(int(a == b) for a, b in zip(pred, t))
        return float(np.mean(correct)) if correct else None

    def pooled_diagnostics(self, per_episode: list[dict]) -> dict[str, Any]:
        """Compare the submitted predictions with visible-data references on one pooled accuracy scale.

        ``y_ref`` is the existing participant-history reference.  New ``y_study_ref`` values are the
        study-conditioned baseline computed from visible training sessions only (see
        :func:`visible_study_prediction`).  Both are trusted-side diagnostics: they do not change the primary
        evaluator, acceptance, or the inputs available to the policy.  Missing/invalid submitted vectors retain
        the official all-wrong convention used by :meth:`pooled_metric`.
        """
        pooled_correct: list[int] = []
        pooled_ref_correct: list[int] = []
        pooled_study_correct: list[int] = []
        episode_scores: list[float] = []
        episode_refs: list[float] = []
        episode_study_refs: list[float] = []
        n_study_ref = 0
        n_items = 0
        for payload in per_episode:
            if not payload:
                continue
            truth = payload.get("y_true")
            if truth is None:
                raise ValueError("FoR52 diagnostics need y_true")
            truth = list(truth)
            n = len(truth)
            pred = payload.get("y_pred")
            if pred is not None and len(pred) != n:
                raise ValueError("FoR52 diagnostics y_pred and y_true differ in length")
            ref = payload.get("y_ref")
            if ref is not None and len(ref) != n:
                raise ValueError("FoR52 diagnostics y_ref and y_true differ in length")
            study_ref = payload.get("y_study_ref")
            if study_ref is not None and len(study_ref) != n:
                raise ValueError("FoR52 diagnostics y_study_ref and y_true differ in length")
            item_correct = [int(pred is not None and pred[i] == truth[i]) for i in range(n)]
            ref_correct = [int(ref is not None and ref[i] == truth[i]) for i in range(n)]
            pooled_correct.extend(item_correct)
            pooled_ref_correct.extend(ref_correct)
            if study_ref is not None:
                pooled_study_correct.extend(int(study_ref[i] == truth[i]) for i in range(n))
                n_study_ref += 1
            episode_scores.append(float(np.mean(item_correct)) if n else 0.0)
            episode_refs.append(float(np.mean(ref_correct)) if n else 0.0)
            if study_ref is not None:
                episode_study_refs.append(float(np.mean([study_ref[i] == truth[i] for i in range(n)]))
                                            if n else 0.0)
            n_items += n
        if not pooled_correct:
            return {"n_episodes": 0, "n_items": 0, "pooled_accuracy": None,
                    "pooled_reference_accuracy": None, "pooled_study_reference_accuracy": None,
                    "mean_episode_accuracy": None, "mean_episode_reference_accuracy": None,
                    "mean_episode_study_reference_accuracy": None, "diagnostic_only": True}
        pooled = float(np.mean(pooled_correct))
        pooled_ref = float(np.mean(pooled_ref_correct))
        return {
            "n_episodes": len(episode_scores),
            "n_items": n_items,
            "pooled_accuracy": pooled,
            "pooled_reference_accuracy": pooled_ref,
            "pooled_study_reference_accuracy": (float(np.mean(pooled_study_correct))
                                                  if pooled_study_correct else None),
            "mean_episode_accuracy": float(np.mean(episode_scores)),
            "mean_episode_reference_accuracy": float(np.mean(episode_refs)),
            "mean_episode_study_reference_accuracy": (float(np.mean(episode_study_refs))
                                                        if episode_study_refs else None),
            "study_reference_episodes": n_study_ref,
            "diagnostic_only": True,
        }

    def full_split_reference(self, split: str) -> dict[str, Any]:
        """Score visible-data references over every item in one split.

        Episode references are based on the episode's sampled training sessions, so their scores are not directly
        comparable with a full-pool result.  This trusted-side helper uses the complete visible ``train`` pool for
        the study-conditioned reference and the item's own history for the participant-mode reference.  It never
        enters an episode, tool output, primary score, or acceptance decision; the targets are read here only for
        the audit aggregate.
        """
        check_split(split)
        data = self._data.get()
        ids = list(data.pools[split])
        if not ids:
            return {
                "split": split,
                "n_items": 0,
                "n_studies": 0,
                "participant_history_accuracy": None,
                "study_conditioned_accuracy": None,
                "visible_train_sessions": len(data.pools["train"]),
                "visible_train_studies": len({data.sessions[u].study for u in data.pools["train"]}),
                "diagnostic_only": True,
            }
        train = [data.sessions[u] for u in data.pools["train"]]
        items = [data.sessions[u] for u in ids]
        truth = [s.target for s in items]
        history_pred = [reference_prediction(s.history, s.options) for s in items]
        study_pred = [visible_study_prediction(train, s.study, s.options, s.history) for s in items]
        return {
            "split": split,
            "n_items": len(items),
            "n_studies": len({s.study for s in items}),
            "participant_history_accuracy": float(np.mean([p == t for p, t in zip(history_pred, truth)])),
            "study_conditioned_accuracy": float(np.mean([p == t for p, t in zip(study_pred, truth)])),
            "visible_train_sessions": len(train),
            "visible_train_studies": len({s.study for s in train}),
            "diagnostic_only": True,
        }

    # ------------------------------------------------------------------------------------------ episodes
    def _train_sessions(self, data: _PsychData, rng: np.random.Generator, item_studies: list[str]) -> list[_Session]:
        by_study: dict[str, list[str]] = {}
        for u in data.pools["train"]:
            by_study.setdefault(data.sessions[u].study, []).append(u)
        order = {st: [us[j] for j in rng.permutation(len(us))] for st, us in sorted(by_study.items())}
        chosen: list[str] = []
        for st in sorted(set(item_studies)):
            if st in order and order[st]:
                chosen.append(order[st].pop(0))
        studies = [st for st in sorted(order)]
        studies = [studies[j] for j in rng.permutation(len(studies))]
        while len(chosen) < self.n_train_sessions and any(order.values()):
            for st in studies:
                if order[st] and len(chosen) < self.n_train_sessions:
                    chosen.append(order[st].pop(0))
        return [data.sessions[u] for u in chosen]

    def _episode(self, data: _PsychData, split: str, seed: int, k: int, uids: list[str]) -> Episode:
        items = [data.sessions[u] for u in uids]
        n = len(items)
        pool = "ood" if split == "ood" else "iid"
        erng = episode_rng(CODE, split, seed, k, "visible")
        train = self._train_sessions(data, erng, [s.study for s in items] if pool == "iid" else [])
        dev_pool = data.pools["dev"]
        dev = [data.sessions[dev_pool[j]] for j in sorted(erng.permutation(len(dev_pool))[: self.n_dev])]
        y_true = [s.target for s in items]
        options = [list(s.options) for s in items]
        ref_pred = [reference_prediction(s.history, s.options) for s in items]
        ref_acc = float(np.mean([a == b for a, b in zip(ref_pred, y_true)]))
        study_ref_pred = [visible_study_prediction(train, s.study, s.options, s.history) for s in items]
        study_ref_acc = float(np.mean([a == b for a, b in zip(study_ref_pred, y_true)]))
        dev_true = [s.target for s in dev]
        dev_ref = [reference_prediction(s.history, s.options) for s in dev]
        dev_ref_acc = float(np.mean([a == b for a, b in zip(dev_ref, dev_true)])) if dev else 0.0
        eval_items = [s.item() for s in items]
        dev_items = [s.item() for s in dev]
        nd = len(dev)
        train_payload = [{"study": s.study, "text": s.text,
                          "responses": [{"pos": int(r["pos"]), "response": r["response"], "options": list(r["options"])}
                                        for r in s.responses]} for s in train]

        def load_train(inputs: dict, config: dict) -> dict:
            return {"sessions": [dict(x, responses=[dict(r) for r in x["responses"]]) for x in train_payload]}

        def load_dev_inputs(inputs: dict, config: dict) -> dict:
            return {"dev_items": [dict(x, options=list(x["options"])) for x in dev_items]}

        def load_eval_inputs(inputs: dict, config: dict) -> dict:
            return {"items": [dict(x, options=list(x["options"])) for x in eval_items]}

        def fixed_predict(inputs: dict, config: dict) -> dict:
            pred, info = _fixed_fit_predict(inputs.get("train_sessions"), inputs.get("eval_items"))
            return {"y": pred, "provenance": info}

        def adaptive_predict(inputs: dict, config: dict) -> dict:
            pred, info = _adaptive_fit_predict(inputs.get("train_sessions"), inputs.get("eval_items"))
            return {"y": pred, "provenance": info}

        def domain_adaptive_predict(inputs: dict, config: dict) -> dict:
            pred, info = _domain_adaptive_fit_predict(inputs.get("train_sessions"), inputs.get("eval_items"))
            return {"y": pred, "provenance": info}

        def score_dev(inputs: dict, config: dict) -> dict:
            v, why = as_str_list(inputs.get("dev_pred"), nd)
            if v is None:
                raise ValueError(f"score_dev: dev_pred must be a list of {nd} keys ({why})")
            valid = [p in s.options for p, s in zip(v, dev)]
            acc = float(np.mean([p == t for p, t in zip(v, dev_true)]))
            return {"dev_accuracy": acc, "dev_reference_accuracy": dev_ref_acc,
                    "dev_invalid": int(len(valid) - sum(valid)), "n_dev": nd}

        tools = [
            ToolSpec("load_train", f"{len(train)} complete participant sessions of the visible training studies "
                                   "(other participants; text truncated at a line boundary). Each dict: 'study', "
                                   "'text' (responses written as <<KEY>>), 'responses' (list of {pos: character "
                                   "offset of '<<', response, options}).", {},
                     {"sessions": PortSchema("list", (len(train),), dtype="dict")}, load_train),
            ToolSpec("load_dev_inputs", f"{nd} dev items from held-out training-pool participants (same format as "
                                        "load_eval_inputs, but different participants; targets withheld, see "
                                        "score_dev).", {},
                     {"dev_items": PortSchema("list", (nd,), dtype="dict")}, load_dev_inputs),
            ToolSpec("score_dev", f"Accuracy of dev_pred and the accuracy of the reference predictor on the same items. "
                                  f"dev_pred[i] is the predicted key for dev_items[i] of load_dev_inputs ({nd} keys, "
                                  "same order); predictions for the evaluation items (load_eval_inputs) are scored "
                                  "against the wrong participants.",
                     {"dev_pred": PortSchema("list", (nd,), dtype="str")},
                     {"dev_accuracy": PortSchema("number", unit="1"), "dev_reference_accuracy": PortSchema("number", unit="1"),
                      "dev_invalid": PortSchema("number", dtype="int"), "n_dev": PortSchema("number", dtype="int")},
                     score_dev),
            ToolSpec("load_eval_inputs", f"The {n} evaluation items, in output order. Each dict: 'study' (experiment "
                                         "id), 'history' (session text up to the response to predict), 'options' "
                                         "(legal response keys), 'options_text'.", {},
                     {"items": PortSchema("list", (n,), dtype="dict")}, load_eval_inputs),
            ToolSpec("psych_fixed_predict", "Fixed visible-data Psych route. Wire train_sessions from load_train.sessions "
                                         "and eval_items from load_eval_inputs.items. It runs the shipped qlearn history "
                                         "model with seed=0 and the visible population prior; it downloads no weights, "
                                         "does not receive targets, and returns legal y plus fixed provenance.",
                     {"train_sessions": PortSchema("list", (len(train),), dtype="dict"),
                      "eval_items": PortSchema("list", (n,), dtype="dict")},
                     {"y": PortSchema("list", (n,), dtype="str"), "provenance": PortSchema("dict")}, fixed_predict),
            ToolSpec("psych_adaptive_predict", "Adaptive visible-data Psych route. Wire train_sessions from load_train.sessions "
                                         "and eval_items from load_eval_inputs.items. It selects one of the shipped "
                                         "qlearn, personal, or wsls history models by grouped out-of-fold accuracy "
                                         "on visible training sessions only, refits that model, and returns legal y "
                                         "plus selection provenance. It never receives evaluation targets; wire y "
                                         "directly to submit.y.",
                     {"train_sessions": PortSchema("list", (len(train),), dtype="dict"),
                      "eval_items": PortSchema("list", (n,), dtype="dict")},
                     {"y": PortSchema("list", (n,), dtype="str"), "provenance": PortSchema("dict")}, adaptive_predict),
            ToolSpec("psych_domain_adaptive_predict", "Domain-adaptive visible-data Psych route. Wire train_sessions from "
                     "load_train.sessions and eval_items from load_eval_inputs.items. It uses the qlearn history model "
                     "for studies represented in the visible training sessions and the pooled gbdt history model for "
                     "unseen study IDs. The switch uses only visible study identifiers; it never receives targets or "
                     "a split flag. Wire y directly to submit.y.",
                     {"train_sessions": PortSchema("list", (len(train),), dtype="dict"),
                      "eval_items": PortSchema("list", (n,), dtype="dict")},
                     {"y": PortSchema("list", (n,), dtype="str"), "provenance": PortSchema("dict")},
                     domain_adaptive_predict),
        ]
        constraints = [
            c_str_list(n, "one response key per item"),
            c_allowed_labels(n, options, description="every y[i] is one of the legal keys options[i] of item i"),
        ]

        def evaluate(yv: Any, trace: Trace | None) -> EvalResult:
            payload: dict[str, Any] = {"item_ids": list(uids), "y_true": list(y_true), "y_pred": None,
                                       "y_ref": list(ref_pred), "y_study_ref": list(study_ref_pred),
                                       "studies": [s.study for s in items]}
            v, why = as_str_list(yv, n)
            if v is None:
                return EvalResult(metrics={"reference_accuracy": ref_acc,
                                           "study_reference_accuracy": study_ref_acc}, primary=0.0,
                                  direction="max", accepted=False, details={"reference": ref_acc,
                                  "study_reference": study_ref_acc, "norm_score": 0.0,
                                                           "pooled_payload": payload, "invalid": why})
            acc = float(np.mean([a == b for a, b in zip(v, y_true)]))
            n_invalid = int(sum(p not in o for p, o in zip(v, options)))
            payload["y_pred"] = list(v)
            return EvalResult(
                metrics={"accuracy": acc, "reference_accuracy": ref_acc,
                         "study_reference_accuracy": study_ref_acc, "n_invalid": float(n_invalid)},
                primary=acc, direction="max", accepted=bool(acc >= ref_acc + self.accept_margin - 1e-12),
                details={"reference": ref_acc, "study_reference": study_ref_acc,
                         "norm_score": norm_score(acc, ref_acc, "max"), "pooled_payload": payload})

        objective = (
            f"Predict the next choice of human participants in psychology experiments. Each of the {n} items is one "
            "participant session from the Psych-201 corpus, written as natural language: the experiment "
            "instructions followed by the trials so far, where every earlier response of that participant is "
            "enclosed in << >> (e.g. 'You press <<J>>.'), cut right before the response to predict. Each item "
            "lists the legal response keys. Deliverable y: a list of exactly "
            f"{n} strings, y[i] = the key the participant actually pressed at that point for item i, one of "
            "options[i], in the order of load_eval_inputs. Metric: micro accuracy (fraction of items where y[i] "
            "equals the participant's recorded response). Training sessions of other participants, a dev slice "
            f"and the evaluation items are provided by the tools: load_dev_inputs returns {nd} further items "
            "(other participants, same format as the evaluation items, targets withheld) and score_dev scores "
            "predictions made for those items, in that order. For a fixed no-new-weight route, wire "
            "load_train.sessions and load_eval_inputs.items to psych_fixed_predict; its qlearn y output is "
            "submit-ready and provenance fixes seed=0 plus the visible population prior. For an optimization "
            "route, psych_adaptive_predict performs grouped out-of-fold selection among shipped qlearn, personal "
            "and wsls models using visible training sessions only, then returns submit-ready y with provenance. "
            "For mixed IID and held-out-study pools, psych_domain_adaptive_predict selects qlearn for study IDs "
            "present in load_train and pooled gbdt for unseen IDs, using only the visible payloads. "
            "Wire the tool's y directly to submit.y and finish immediately; do not replace a successful tool "
            "output with a later code node, manual heuristic, or an untracked alternative.\n"
            + scilib.describe("psych")
        )
        return Episode(
            id=episode_id(CODE, split, seed, k), discipline=CODE, family=FAMILY, split=split,
            task_type=self.task_type, objective=objective,
            required_output=PortSchema("list", (n,), dtype="str", description="one legal response key per item"),
            tools=tools, constraints=constraints,
            budget=self.budget or Budget(max_steps=12, max_policy_tokens=200_000, max_wall_s=1800.0,
                                         max_node_s=300.0,
                                         max_llm_items=3 * (n + LLM_BUDGET_DEV_ITEMS)),
            lineage={"dataset": "marcelbinz/Psych-201-discrete", "version": "060062064d00766ea0b5666c73268329d0f556b6",
                     "pool": pool, "ood_kind": "proxy_within_dataset" if pool == "ood" else None,
                     "ood_shift": "held-out studies (experiment paradigms not in the IID/training studies)"
                     if pool == "ood" else None,
                     "item_ids": list(uids), "studies": [s.study for s in items],
                     "train_session_ids": [s.uid for s in train], "dev_item_ids": [s.uid for s in dev],
                     "seed": seed, "index": k, "partition_seed": PARTITION_SEED,
                     "option_rule": "data-team strict_single_key_complete_group_v1 templates (item-level)",
                     "receipt": data.receipt},
            acceptance=f"accuracy >= reference accuracy (participant's most frequent earlier response) + "
                       f"{self.accept_margin:g}",
            tolerance={"rtol": 0.0, "atol": 0.0},
            tags=["psychology", "behavior", "choice_prediction", "text", "sequential", "llm"],
            metric=self.metric, direction=self.direction, n_items=n,
            _evaluate=evaluate, _dev_evaluate=None,
        )


Adapter = Psych201Adapter
