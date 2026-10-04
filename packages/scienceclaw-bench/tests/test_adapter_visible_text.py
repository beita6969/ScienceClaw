"""Review finding F5 (adapter half): what the policy reads about a forecasting episode never spells out the
reference method or the acceptance rule.

The objective, every tool / port description (``public_view()["tools"]``), the visible constraint texts, the
required-output description, the tags and (for the method words only) the hidden ``acceptance`` string of the
FoR33/35/37/38/41 episodes are scanned for the reference recipes (persistence, seasonal naive, climatology,
no-change / random walk, previous-day ...), for a "Success criterion" sentence and for the acceptance factor. The
reference *value* and recipe live only in ``EvalResult.details`` and docs/tasks/*.md; the generic Acceptance section
of the prompt is rendered by the prompt builder, not by the adapters.
"""
from __future__ import annotations

import json
import re
import sys
from importlib import import_module
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scienceclaw.bench.splits import split_seed  # noqa: E402

ADAPTERS = {
    "FoR33": "for33_buildingsbench",
    "FoR35": "for35_tourism",
    "FoR37": "for37_weatherbench",
    "FoR38": "for38_worldbank",
    "FoR41": "for41_neon",
}

# reference recipes: forbidden in everything the policy can read AND in the (hidden) acceptance string
RECIPE = [
    r"persistence", r"seasonal[- ]?naive", r"\bsnaive\b", r"\bnaive\b", r"climatolog", r"no[- ]change",
    r"random[- ]walk", r"previous[- ]day", r"last[- ]value",
    r"ŷ[_\w\[\], ]*=\s*(value|x_|field|context|the last)",
]
# acceptance wording / factor: forbidden in the policy-visible text only (the hidden string may state the factor)
ACCEPTANCE = [
    r"Success criterion", r"success criterion", r"accepted iff", r"\b0\.9\d?\s*(x|×)",
    r"(x|×)\s*(the\s+)?(score|RMSE|MASE|CRPS|NRMSE|sMAPE)\s+of\s+the\s+reference",
    r"reference\s+(recipe|forecast\s+ŷ|forecast\s+is|method)",
]
_RECIPE_RE = [re.compile(p, re.I) for p in RECIPE]
_ACCEPT_RE = [re.compile(p) for p in ACCEPTANCE]


def visible_text(ep) -> str:
    """Everything the policy can read about the episode (public view incl. tool/port descriptions, tags)."""
    return "\n".join([json.dumps(ep.public_view(), default=str, ensure_ascii=False), ep.objective,
                      *[t.config_doc for t in ep.tools], json.dumps(ep.required_output.to_dict(), default=str)])


def scan_recipe(text: str) -> list[str]:
    return [rx.pattern for rx in _RECIPE_RE if rx.search(text)]


def scan_acceptance(text: str) -> list[str]:
    return [rx.pattern for rx in _ACCEPT_RE if rx.search(text)]


def test_scanner_flags_reference_recipes() -> None:
    bad = ["the reference is previous-day persistence", "seasonal naive forecast", "smoothed day-of-year climatology",
           "ŷ_h = value in 2021 (no-change forecast)", "random walk", "a naive forecast"]
    for s in bad:
        assert scan_recipe(s), s
    for s in ["Success criterion: score <= 0.95 x the score of the reference forecast", "accepted iff RMSE <= 0.96 x",
              "0.97 x the reference", "score <= 0.95 x the RMSE of the reference"]:
        assert scan_acceptance(s), s
    good = ["Score (lower is better): mean over items of sMAPE (%)",
            "score_dev returns the same statistic of the reference forecast.",
            "Deliverable y: a float array of shape (16, 4)"]
    for s in good:
        assert not scan_recipe(s) and not scan_acceptance(s), s


@pytest.fixture(scope="module", params=sorted(ADAPTERS))
def episode(request):
    code = request.param
    mod = import_module(f"scienceclaw.bench.tasks.{ADAPTERS[code]}")
    ad = mod.Adapter()
    ok, why = ad.available()
    if not ok:
        pytest.skip(f"{code} data unavailable: {why}")
    return code, ad.build_episodes("val", 1, split_seed(20260928, code, "val"), items_per_episode=16)[0]


def test_visible_text_does_not_reveal_reference_method(episode) -> None:
    code, ep = episode
    text = visible_text(ep)
    assert not scan_recipe(text), f"{code}: visible text matches {scan_recipe(text)}"
    assert not scan_acceptance(text), f"{code}: visible text matches {scan_acceptance(text)}"
    assert "Deliverable" in ep.objective and re.search(r"[Ss]core", ep.objective)   # what to submit and how it is scored
    # the objective does not talk about the reference forecast at all ("reference date" of FoR41 is data, not method)
    assert not re.search(r"reference(?![_ ]date)", ep.objective, re.I), f"{code}: objective mentions the reference"


def test_acceptance_string_is_method_neutral(episode) -> None:
    code, ep = episode
    assert ep.acceptance and "reference" in ep.acceptance
    assert not scan_recipe(ep.acceptance), (code, ep.acceptance)


# ---- second guard: the other adapters name no reference baseline / organiser recipe, and the scilib docs state no margin
NAMED_BASELINE = [
    r"organi[sz]er\s+(starter|baseline)", r"starter", r"majority[- ]label", r"mixture[- ]as[- ]estimate", r"\btrivial\b",
    r"(is|are) the reference", r"uncorrected OCR text is",
]
OTHER_ADAPTERS = {
    "FoR34": "for34_molhiv", "FoR36": "for36_musdb", "FoR39": "for39_eedi", "FoR43": "for43_hipe",
    "FoR49": "for49_smt", "FoR51": "for51_matbench", "FoR52": "for52_psych201",
}


@pytest.fixture(scope="module", params=sorted(OTHER_ADAPTERS))
def other_episode(request):
    code = request.param
    ad = import_module(f"scienceclaw.bench.tasks.{OTHER_ADAPTERS[code]}").Adapter()
    ok, why = ad.available()
    if not ok:
        pytest.skip(f"{code} data unavailable: {why}")
    return code, ad.build_episodes("val", 1, split_seed(20260928, code, "val"))[0]


def test_other_adapters_name_no_reference_baseline(other_episode) -> None:
    code, ep = other_episode
    text = visible_text(ep)
    hits = [p for p in NAMED_BASELINE if re.search(p, text, re.I)]
    assert not hits, f"{code}: visible text matches {hits}"
    assert not scan_acceptance(text), f"{code}: visible text matches {scan_acceptance(text)}"


def test_psych_library_doc_states_no_reference_or_margin() -> None:
    import scilib
    doc = scilib.describe("psych")
    assert not re.search(r"reference|margin|baseline", doc, re.I)
