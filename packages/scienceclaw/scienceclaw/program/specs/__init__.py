"""Specifications of the typed library operators, one module per tool family.

A module contributes a list ``SPECS`` of dicts with the keys ``id``, ``tool`` (the ``module.function`` it wraps), ``description``,
``inputs`` / ``outputs`` (port schemas), ``code`` (the body of ``run``, returning a dict keyed by output port), ``pre`` / ``post``
(contract entries checked by the executor) and ``tags`` (retrieval metadata).
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import Any


def all_specs() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
        if info.name.startswith("_"):
            continue
        for spec in importlib.import_module(f"{__name__}.{info.name}").SPECS:
            if spec["id"] in seen:
                raise ValueError(f"duplicate library operator id {spec['id']!r} (module {info.name})")
            seen.add(spec["id"])
            out.append(spec)
    return out
