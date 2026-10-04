"""Uniform disclosure: what the policy may read about an episode must not name the reference predictor.

The generic acceptance section of the system prompt says only that the task metric has to beat the task's reference
predictor by a fixed margin. The adapters' objective strings must not contradict that by naming the reference method
or carrying a 'Success criterion ... reference recipe' sentence.
"""
from __future__ import annotations

import pytest

from scienceclaw.bench.registry import available_adapters

FORBIDDEN = ("seasonal naive", "climatology", "success criterion")


def test_no_adapter_public_view_names_the_reference_recipe() -> None:
    adapters = available_adapters()
    if not adapters:
        pytest.skip("no task adapter is available in this environment")
    offenders: list[str] = []
    for code, adapter in sorted(adapters.items()):
        for ep in adapter.build_episodes("src", 1, 0, 4):
            view = ep.public_view()
            text = (str(view["objective"]) + "\n" + "\n".join(str(t) for t in view["tools"])).lower()
            offenders += [f"{code}: {phrase!r}" for phrase in FORBIDDEN if phrase in text]
    assert not offenders, "reference recipe disclosed in the public view: " + "; ".join(offenders)
