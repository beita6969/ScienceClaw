"""Helpers shared by the operator specification modules."""
from __future__ import annotations

from scienceclaw.core.schema import PortSchema


def p(type_: str, desc: str, shape: tuple | None = None, unit: str | None = None, dtype: str | None = None) -> PortSchema:
    """A port schema: type, optional shape and unit, and a one-line description of what the value is."""
    return PortSchema(type=type_, shape=shape, unit=unit, dtype=dtype, description=desc)
