"""Regression tests for fixes made during integration (lead-owned)."""
from scienceclaw.core.schema import PortSchema, compat
from scienceclaw.evolution.operator_abstraction import generalize_boundary_shapes


def test_operator_boundary_shapes_are_generalized():
    ins = {"fit__X_train": PortSchema("array", (48, 2), dtype="float"),
           "fit__X_eval": PortSchema("array", (24, 2)), "fit__k": PortSchema("number")}
    outs = {"fit__y": PortSchema("array", (24,), unit="1")}
    gi, go = generalize_boundary_shapes(ins, outs)
    assert gi["fit__X_eval"].shape == ("s0", "s1") and gi["fit__X_train"].shape == ("s2", "s1")
    assert go["fit__y"].shape == ("s0",) and go["fit__y"].unit == "1"
    assert gi["fit__k"].shape is None and gi["fit__X_train"].dtype == "float"
    # the generalized operator accepts inputs of another episode size (the verbatim schema does not)
    other = PortSchema("array", (60, 2))
    assert not compat(other, ins["fit__X_train"])[0]
    assert compat(other, gi["fit__X_train"])[0]
    # symbolic dims are kept
    gi2, _ = generalize_boundary_shapes({"a": PortSchema("array", ("n", 3))}, {})
    assert gi2["a"].shape == ("n", "s0")
