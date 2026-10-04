"""Retriever tests: BM25 ranking, tag/applicability bonus, deterministic ties, slice_hash stability."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))

from test_agent_support import make_episode  # noqa: E402

from scienceclaw.core.graph import Node, WorkflowGraph  # noqa: E402
from scienceclaw.core.operators import Contract, OperatorSpec  # noqa: E402
from scienceclaw.core.program import AgentProgram, Bundle  # noqa: E402
from scienceclaw.core.retrieval import BM25Index, Retriever, RetrievalWeights, episode_query, tokenize  # noqa: E402
from scienceclaw.core.schema import PortSchema  # noqa: E402
from scienceclaw.core.skills import Skill  # noqa: E402


def _op(oid: str, desc: str, app: dict | None = None, in_type: str = "array", tags: list[str] | None = None) -> OperatorSpec:
    body = WorkflowGraph({"c": Node("c", "code", code="def run(i, c):\n    return {'y': i['x']}\n",
                                    inputs={"x": PortSchema(in_type)}, outputs={"y": PortSchema("array")})})
    return OperatorSpec(id=oid, version=1, name=oid, description=desc, body=body, inputs={"x": PortSchema(in_type)},
                        outputs={"y": PortSchema("array")}, input_map={"x": [("c", "x")]}, output_map={"y": ("c", "y")},
                        contract=Contract(applicability=app or {}), tags=list(tags or []))


def _program() -> AgentProgram:
    skills = [
        Skill("forecast", 1, "Seasonal forecasting", "Fit seasonal models to monthly series and forecast horizons.",
              tags=["forecasting"]),
        Skill("scale", 1, "Scaling regression", "Predict the scaled quantity for each x value by regression.",
              tags=["regression"]),
        Skill("poetry", 1, "Sonnets", "Meter and rhyme schemes.", tags=["literature"]),
    ]
    ops = [_op("fit_scale", "regression of scaled quantity", in_type="list"),
           _op("seasonal", "seasonal decomposition of series")]
    return AgentProgram({s.id: s for s in skills}, {o.id: o for o in ops})


def test_tokenize() -> None:
    assert tokenize("Binary_classification of FoR34 molecules, a/b") == ["binary", "classification", "for34", "molecules"]
    assert tokenize("") == []


def test_query_contains_episode_fields() -> None:
    q = episode_query(make_episode())
    for part in ("Predict the scaled quantity", "regression", "toy", "scaling", "FoR49", "list"):
        assert part in q


def test_skill_ranking_and_irrelevant_excluded() -> None:
    r = Retriever(_program())
    ids = [s.id for s in r.skills(make_episode(), 4)]
    assert ids[0] == "scale" and "poetry" not in ids
    assert r.skills(make_episode(), 0) == [] and len(r.skills(make_episode(), 1)) == 1


def test_operator_ranking_uses_input_types() -> None:
    r = Retriever(_program())
    ops = r.operators(make_episode(), 6)
    assert ops[0].id == "fit_scale"          # lexical match + input type "list" produced by the episode tool


def test_applicability_bonus_breaks_equal_text() -> None:
    a = _op("a_plain", "identical description text")
    b = _op("b_app", "identical description text", app={"disciplines": ["FoR49"], "task_types": ["regression"]})
    prog = AgentProgram({}, {"a_plain": a, "b_app": b})
    scored = Retriever(prog).score_operators(make_episode())
    assert [o.id for _, o in scored][0] == "b_app"


def test_ties_broken_by_id() -> None:
    s1 = Skill("zeta", 1, "Scaling regression", "x values", tags=["regression"])
    s2 = Skill("alpha", 1, "Scaling regression", "x values", tags=["regression"])
    prog = AgentProgram({"zeta": s1, "alpha": s2})
    assert [s.id for s in Retriever(prog).skills(make_episode(), 2)] == ["alpha", "zeta"]


def test_discipline_tag_bonus() -> None:
    s1 = Skill("a", 1, "Scaling regression", "x values", tags=["regression"])
    s2 = Skill("b", 1, "Scaling regression", "x values", tags=["regression", "FoR49"])
    prog = AgentProgram({"a": s1, "b": s2})
    assert [s.id for s in Retriever(prog).skills(make_episode(), 2)] == ["b", "a"]


def test_empty_program() -> None:
    r = Retriever(AgentProgram())
    assert r.skills(make_episode(), 4) == [] and r.operators(make_episode(), 6) == []
    assert r.slice_hash(make_episode(), 4, 6) == Retriever(AgentProgram()).slice_hash(make_episode(), 4, 6)


def test_bm25_idf_prefers_rare_terms() -> None:
    idx = BM25Index(["d1", "d2", "d3"], ["common rare", "common", "common"])
    s = idx.scores(["rare", "common"])
    assert s[0] > s[1] == s[2] > 0


def test_slice_hash_stability_and_sensitivity() -> None:
    ep = make_episode()
    prog = _program()
    h = Retriever(prog).slice_hash(ep, 4, 6)
    assert len(h) == 64 and h == Retriever(_program()).slice_hash(ep, 4, 6)
    # an irrelevant new skill (not retrieved) leaves the slice unchanged
    irrelevant = Skill("knitting", 1, "Knitting", "Purl stitches.", tags=["craft"])
    prog2, _ = prog.apply(Bundle(skills=[irrelevant]))
    assert Retriever(prog2).slice_hash(ep, 4, 6) == h
    # a revision of a retrieved skill changes it (version id and content)
    rev = Skill("scale", 1, "Scaling regression", "Predict the scaled quantity for each x value; revised.", tags=["regression"])
    prog3, omega = prog.apply(Bundle(skills=[rev]))
    assert omega == ["skill:scale@v2"]
    assert Retriever(prog3).slice_hash(ep, 4, 6) != h
    # k only matters through what is actually retrieved (only "scale" is relevant here)
    assert Retriever(prog).slice_hash(ep, 1, 6) == h
    assert Retriever(prog).slice_hash(ep, 0, 6) != h


def test_ranking_independent_of_hash_seed() -> None:
    """BM25 sums over sorted query terms, so scores do not depend on PYTHONHASHSEED."""
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from test_retrieval_bm25 import _program\n"
        "from test_agent_support import make_episode\n"
        "from scienceclaw.core.retrieval import Retriever\n"
        "r = Retriever(_program())\n"
        "print([(repr(s), o.id) for s, o in r.score_operators(make_episode())],"
        " [(repr(s), k.id) for s, k in r.score_skills(make_episode())], r.slice_hash(make_episode(), 4, 6))\n"
    ) % str(Path(__file__).parent)
    outs = set()
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        outs.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout)
    assert len(outs) == 1


# ------------------------------------------------------------------------------------ relevance floor
def _mol_episode(eid: str = "mol") -> SimpleNamespace:
    """A different discipline / task type than the toy regression episode."""
    return SimpleNamespace(id=eid, discipline="FoR34", task_type="binary_classification",
                           objective="Classify molecules from SMILES strings and predict the probability of each label.",
                           tags=["molecules", "smiles"], required_output=PortSchema("list"), tools=[])


def _mol_skill(sid: str = "mol-clf", body: str = "Predict the probability of each label from the SMILES string of a molecule.") -> Skill:
    return Skill(sid, 1, "Molecule classifier", body, tags=["FoR34", "binary_classification", "molecules"])


def test_irrelevant_discipline_component_is_not_retrieved() -> None:
    """One shared common word ('predict') must not put a FoR34 skill into a FoR49 slice."""
    ep = make_episode()
    prog = AgentProgram({"mol-clf": _mol_skill()})
    assert Retriever(prog).skills(ep, 4) == []
    assert Retriever(prog).skills(_mol_episode(), 4)[0].id == "mol-clf"        # relevant where it was learned
    # the old behaviour (floor off) retrieved it for every episode sharing a single token
    assert [s.id for s in Retriever(prog, RetrievalWeights(floor=False)).skills(ep, 4)] == ["mol-clf"]


def test_relevance_floor_lets_metadata_matches_and_strong_lexical_matches_through() -> None:
    ep = make_episode()
    tagged = Skill("t", 1, "Other title", "unrelated words", tags=["regression"])          # task-type match
    tag_only = Skill("g", 1, "Other title", "unrelated words", tags=["scaling"])            # episode-tag overlap
    strong = Skill("s", 1, "Scaled quantity regression",
                   "Predict the scaled quantity y for each visible x value with a regression on the toy scaling list.")
    weak = Skill("w", 1, "Unrelated", "predict something else entirely", tags=[])
    prog = AgentProgram({s.id: s for s in (tagged, tag_only, strong, weak)})
    assert {s.id for s in Retriever(prog).skills(ep, 10)} == {"t", "g", "s"}


def test_slice_hash_of_unrelated_episodes_is_unchanged_by_a_new_component() -> None:
    """Lazy re-validation reuse: a FoR34 skill changes the slice of FoR34 episodes only."""
    toy, mol_a, mol_b = make_episode(), _mol_episode("mol-a"), _mol_episode("mol-b")
    base = _program()
    grown, _ = base.apply(Bundle(skills=[_mol_skill()]))
    r0, r1 = Retriever(base), Retriever(grown)
    assert r0.slice_hash(toy, 4, 6) == r1.slice_hash(toy, 4, 6)
    assert r0.slice_hash(mol_a, 4, 6) != r1.slice_hash(mol_a, 4, 6)
    assert r0.slice_hash(mol_b, 4, 6) != r1.slice_hash(mol_b, 4, 6)
    # a second irrelevant skill (different discipline again) does not disturb the toy slice either
    other = Skill("law", 1, "Statute lookup", "Cite the governing statute for each legal claim.",
                  tags=["FoR48", "legal_reasoning"])
    grown2, _ = grown.apply(Bundle(skills=[other]))
    assert Retriever(grown2).slice_hash(toy, 4, 6) == r0.slice_hash(toy, 4, 6)


def test_floor_applies_to_operators_too() -> None:
    ep = make_episode()
    mol = _op("mol_feat", "featurize the molecule list", tags=["FoR34"], in_type="list")
    app = _op("regress_fn", "unrelated text", app={"task_types": ["regression"]})
    prog = AgentProgram({}, {"mol_feat": mol, "regress_fn": app})
    ids = [o.id for o in Retriever(prog).operators(ep, 6)]
    assert "mol_feat" not in ids and "regress_fn" in ids
