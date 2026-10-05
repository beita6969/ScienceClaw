"""The initial agent program of the gateway: Skills loaded from ``SKILL.md`` files plus the typed library operators.

OpenClaw skills are Markdown files with a YAML front matter. The same files are the strategy-level half of the editable
program A = (S, O): they are parsed into :class:`scienceclaw.core.skills.Skill` records so that the canvas orchestration
retrieves them per task with the BM25/metadata retriever, exactly as the engine does for any task, and so that evolved
Skills are written back as versioned records of the same program.
"""
from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Iterable

import yaml

from scienceclaw.core.program import AgentProgram
from scienceclaw.core.skills import Skill

DEFAULT_PATTERNS = ("scienceclaw-*",)
_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.S)


def default_skills_dir() -> Path | None:
    """``$SCIENCECLAW_SKILLS_DIR``, else the ``skills/`` directory of the repository checkout."""
    env = os.environ.get("SCIENCECLAW_SKILLS_DIR")
    if env:
        return Path(env).expanduser()
    repo = Path(__file__).resolve().parents[4] / "skills"
    return repo if repo.is_dir() else None


def parse_skill_md(path: Path) -> Skill:
    text = path.read_text(encoding="utf-8")
    meta: dict = {}
    m = _FRONT.match(text)
    if m:
        try:
            meta = yaml.safe_load(m.group(1)) or {}
        except yaml.YAMLError:
            meta = {}
        text = text[m.end():]
    sid = path.parent.name
    title = str(meta.get("name") or sid)
    description = " ".join(str(meta.get("description") or "").split())
    body = (f"When to use: {description}\n\n" if description else "") + text.strip()
    tags = _tags(sid, description)
    return Skill(id=sid, version=1, title=title, body=body, tags=tags,
                 provenance={"source": f"skills/{sid}/SKILL.md", "kind": "seed"}, created="seed")


def _tags(sid: str, description: str) -> list[str]:
    return list(dict.fromkeys(t for t in re.split(r"[-_]+", sid) if t))


def load_skills(skills_dir: str | Path | None = None, patterns: Iterable[str] = DEFAULT_PATTERNS) -> dict[str, Skill]:
    root = Path(skills_dir) if skills_dir else default_skills_dir()
    if root is None or not root.is_dir():
        return {}
    out: dict[str, Skill] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if any(fnmatch.fnmatch(d.name, pat) for pat in patterns) and (d / "SKILL.md").is_file():
            sk = parse_skill_md(d / "SKILL.md")
            out[sk.id] = sk
    return out


def seed_program(skills_dir: str | Path | None = None, patterns: Iterable[str] = DEFAULT_PATTERNS, *,
                 operators: bool = True) -> AgentProgram:
    """A_0 of the gateway: the seed Skills and, unless ``operators`` is False, the typed library operators."""
    from scienceclaw.program.library_ops import library_operators
    return AgentProgram(skills=load_skills(skills_dir, patterns), operators=library_operators() if operators else {},
                        version="A0")
