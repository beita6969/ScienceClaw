"""Versioned, rollback-able storage of the agent program.

A program snapshot is a directory ``<root>/snapshots/<version>/`` written by :meth:`AgentProgram.save`; ``<root>/HEAD`` names
the active one. Promotions append a JSON receipt to ``<root>/receipts/``, so every change of the program can be traced to the
evidence that justified it and undone with :meth:`ProgramStore.rollback`.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from scienceclaw.core.program import AgentProgram


def default_root() -> Path:
    return Path(os.environ.get("SCIENCECLAW_PROGRAM_DIR") or Path.home() / ".scienceclaw" / "program").expanduser()


class ProgramStore:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root).expanduser() if root else default_root()

    # ------------------------------------------------------------------ read
    @property
    def head_file(self) -> Path:
        return self.root / "HEAD"

    def head(self) -> str | None:
        return self.head_file.read_text().strip() if self.head_file.is_file() else None

    def versions(self) -> list[str]:
        d = self.root / "snapshots"
        return sorted(p.name for p in d.iterdir()) if d.is_dir() else []

    def load(self, version: str | None = None) -> AgentProgram | None:
        v = version or self.head()
        return AgentProgram.load(self.root / "snapshots" / v) if v and (self.root / "snapshots" / v).is_dir() else None

    def history(self) -> list[dict[str, Any]]:
        d = self.root / "receipts"
        return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.is_dir() else []

    # ----------------------------------------------------------------- write
    def commit(self, program: AgentProgram, receipt: dict[str, Any] | None = None, *, activate: bool = True) -> str:
        """Store a snapshot (immutable per version id), optionally make it the head and record the receipt."""
        v = program.version
        target = self.root / "snapshots" / v
        if target.exists():
            existing = AgentProgram.load(target)
            if existing.fingerprint() != program.fingerprint():
                raise ValueError(f"program version {v!r} already exists with different content")
        else:
            program.save(target)
        if receipt is not None:
            rd = self.root / "receipts"
            rd.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%dT%H%M%S")
            (rd / f"{stamp}-{v}.json").write_text(json.dumps({"version": v, "time": stamp, **receipt}, indent=1, default=str))
        if activate:
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.head_file.with_suffix(".tmp")
            tmp.write_text(v)
            os.replace(tmp, self.head_file)
        return v

    def rollback(self, version: str) -> str:
        if version not in self.versions():
            raise KeyError(f"no snapshot {version!r}; have {self.versions()}")
        tmp = self.head_file.with_suffix(".tmp")
        tmp.write_text(version)
        os.replace(tmp, self.head_file)
        return version

    def open(self, **seed_kw: Any) -> AgentProgram:
        """The active program; on first use the seed program is created and stored as the first snapshot."""
        prog = self.load()
        if prog is None:
            from scienceclaw.program.seed import seed_program
            prog = seed_program(**seed_kw)
            self.commit(prog, {"event": "seed", "skills": len(prog.skills), "operators": len(prog.operators)})
        return prog
