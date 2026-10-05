"""Evolution ablation variants (DESIGN section 6, "Variants").

=================  =====================================================================================
variant            what an evolution instance contributes to the program
=================  =====================================================================================
``frozen``         nothing (the source stream is still solved, for comparable cost accounting)
``workflow_only``  the whole replay-verified G+ stored as one retrievable "workflow exemplar" Skill
``skill_only``     only the Skill candidate(s) of Eq. 11
``operator_only``  only the boundary-replay-verified Operator candidates of Eq. 12
``unlinked``       Skill and Operator candidates, gated *independently* (two bundles)
``full``           the linked Skill-Operator bundle B_i = (dS_i, dO_i), gated atomically (the method)
=================  =====================================================================================
"""
from __future__ import annotations

from ..core.program import Bundle

VARIANTS: tuple[str, ...] = ("frozen", "workflow_only", "skill_only", "operator_only", "unlinked", "full")


def check_variant(variant: str) -> str:
    """Return ``variant`` or raise ``ValueError`` if it is not one of :data:`VARIANTS`."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown evolution variant {variant!r}; expected one of {VARIANTS}")
    return variant


def wants_components(variant: str) -> tuple[bool, bool]:
    """(build Skill candidates?, build Operator candidates?) for an abstraction variant.

    ``workflow_only`` and ``frozen`` build neither (they do not abstract the repair).
    """
    check_variant(variant)
    return variant in ("full", "skill_only", "unlinked"), variant in ("full", "operator_only", "unlinked")


def bundles_for_variant(bundle: Bundle, variant: str) -> list[Bundle]:
    """Split a built bundle into the bundles that are gated (each one is applied and validated atomically).

    * ``full`` / ``workflow_only``: the bundle itself (one atomic, linked update);
    * ``skill_only``: only its Skills; ``operator_only``: only its Operators;
    * ``unlinked``: two bundles (Skills, then Operators), gated independently;
    * ``frozen``: nothing.

    Empty bundles are dropped, so the result may be ``[]``.
    """
    check_variant(variant)
    meta = dict(bundle.meta)
    if variant == "frozen":
        return []
    if variant in ("full", "workflow_only"):
        out = [bundle]
    elif variant == "skill_only":
        out = [Bundle(list(bundle.skills), [], {**meta, "part": "skills"})]
    elif variant == "operator_only":
        out = [Bundle([], list(bundle.operators), {**meta, "part": "operators"})]
    else:  # unlinked
        out = [Bundle(list(bundle.skills), [], {**meta, "part": "skills", "linked": False}),
               Bundle([], list(bundle.operators), {**meta, "part": "operators", "linked": False})]
    return [b for b in out if not b.is_empty()]
