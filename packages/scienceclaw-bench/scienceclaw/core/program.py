"""Editable agent program A_r = (𝒮_r, 𝒪_r), bundles B_i and atomic application Apply(A_r; B_i)."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .operators import OperatorSpec
from .skills import Skill


@dataclass
class Bundle:
    """B_i = (Δ𝒮_i, Δ𝒪_i): linked strategy change and executable capability from one evolution instance.

    Skills/operators whose id already exists in the program are revisions (version is bumped on apply);
    new ids are additions.
    """
    skills: list[Skill] = field(default_factory=list)
    operators: list[OperatorSpec] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.skills and not self.operators

    def to_dict(self) -> dict:
        return {"skills": [s.to_dict() for s in self.skills], "operators": [o.to_dict() for o in self.operators],
                "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict) -> "Bundle":
        return cls([Skill.from_dict(s) for s in d.get("skills", [])],
                   [OperatorSpec.from_dict(o) for o in d.get("operators", [])], dict(d.get("meta", {})))


class AgentProgram:
    def __init__(self, skills: dict[str, Skill] | None = None, operators: dict[str, OperatorSpec] | None = None,
                 version: str = "A0", parent: str | None = None, lineage: list[dict] | None = None) -> None:
        self.skills: dict[str, Skill] = dict(skills or {})
        self.operators: dict[str, OperatorSpec] = dict(operators or {})
        self.version = version
        self.parent = parent
        self.lineage: list[dict] = list(lineage or [])

    # ----------------------------------------------------------------- update
    def apply(self, bundle: Bundle, new_version: str | None = None) -> tuple["AgentProgram", list[str]]:
        """Atomic application Ã_i = Apply(A_r; B_i). Returns (new program, ω = new version ids)."""
        skills = copy.deepcopy(self.skills)
        ops = copy.deepcopy(self.operators)
        omega: list[str] = []
        for s in bundle.skills:
            s = copy.deepcopy(s)
            s.version = skills[s.id].version + 1 if s.id in skills else max(1, s.version)
            skills[s.id] = s
            omega.append(s.version_id)
        for o in bundle.operators:
            o = copy.deepcopy(o)
            o.version = ops[o.id].version + 1 if o.id in ops else max(1, o.version)
            ops[o.id] = o
            omega.append(o.version_id)
        ver = new_version or f"{self.version}+{hashlib.sha256(json.dumps(omega).encode()).hexdigest()[:6]}"
        entry = {"from": self.version, "to": ver, "omega": omega, "meta": bundle.meta}
        return AgentProgram(skills, ops, ver, self.version, self.lineage + [entry]), omega

    def version_ids(self) -> set[str]:
        return {s.version_id for s in self.skills.values()} | {o.version_id for o in self.operators.values()}

    def restricted(self, skills: bool = True, operators: bool = True) -> "AgentProgram":
        return AgentProgram(self.skills if skills else {}, self.operators if operators else {},
                            self.version, self.parent, self.lineage)

    # ------------------------------------------------------------ fingerprint
    def to_dict(self) -> dict:
        return {
            "version": self.version, "parent": self.parent, "lineage": self.lineage,
            "skills": {k: v.to_dict() for k, v in sorted(self.skills.items())},
            "operators": {k: v.to_dict() for k, v in sorted(self.operators.items())},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentProgram":
        return cls({k: Skill.from_dict(v) for k, v in d.get("skills", {}).items()},
                   {k: OperatorSpec.from_dict(v) for k, v in d.get("operators", {}).items()},
                   d.get("version", "A0"), d.get("parent"), d.get("lineage", []))

    def fingerprint(self) -> str:
        blob = json.dumps({"skills": {k: v.to_dict() for k, v in sorted(self.skills.items())},
                           "operators": {k: v.to_dict() for k, v in sorted(self.operators.items())}},
                          sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "program.json").write_text(json.dumps(self.to_dict(), indent=1, default=str))
        md = [f"# Program {self.version} (parent {self.parent})", f"skills: {len(self.skills)}, operators: {len(self.operators)}", ""]
        md += [s.render() + "\n" for s in self.skills.values()]
        md += [o.render() + "\n" for o in self.operators.values()]
        (p / "program.md").write_text("\n".join(md))

    @classmethod
    def load(cls, path: str | Path) -> "AgentProgram":
        return cls.from_dict(json.loads((Path(path) / "program.json").read_text()))

    def summary(self) -> dict:
        return {"version": self.version, "n_skills": len(self.skills), "n_operators": len(self.operators),
                "fingerprint": self.fingerprint()}
