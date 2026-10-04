"""Versioned, rollback-able storage of the agent program.

A program snapshot is a directory ``<root>/snapshots/<version>/`` written by :meth:`AgentProgram.save`; ``<root>/HEAD`` names
the active one. Promotions append a JSON receipt to ``<root>/receipts/``, so every change of the program can be traced to the
evidence that justified it and undone with :meth:`ProgramStore.rollback`.

Writers take an exclusive lock on ``<root>/.lock``. A snapshot is written to a scratch directory, checked by loading it back and
only then renamed into place, so a crash never leaves a half-written version; ``HEAD`` is replaced atomically.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

from scienceclaw.core.program import AgentProgram

try:
    import fcntl
except ImportError:                     # not POSIX: single-writer deployments only
    fcntl = None                        # type: ignore[assignment]


class StaleHead(RuntimeError):
    """The active program is not the one the caller derived its change from."""


def default_root() -> Path:
    return Path(os.environ.get("SCIENCECLAW_PROGRAM_DIR") or Path.home() / ".scienceclaw" / "program").expanduser()


class ProgramStore:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root).expanduser() if root else default_root()

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / ".lock", "w") as fh:
            if fcntl is not None:
                fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(fh, fcntl.LOCK_UN)

    # ------------------------------------------------------------------ read
    @property
    def head_file(self) -> Path:
        return self.root / "HEAD"

    def head(self) -> str | None:
        return self.head_file.read_text().strip() if self.head_file.is_file() else None

    def versions(self) -> list[str]:
        d = self.root / "snapshots"
        return sorted(p.name for p in d.iterdir() if not p.name.startswith(".")) if d.is_dir() else []

    def load(self, version: str | None = None) -> AgentProgram | None:
        v = version or self.head()
        return AgentProgram.load(self.root / "snapshots" / v) if v and (self.root / "snapshots" / v).is_dir() else None

    def history(self) -> list[dict[str, Any]]:
        d = self.root / "receipts"
        return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.is_dir() else []

    # ----------------------------------------------------------------- write
    def _receipt(self, version: str, receipt: dict[str, Any]) -> None:
        rd = self.root / "receipts"
        rd.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        seq = sum(1 for _ in rd.glob("*.json")) + 1
        (rd / f"{seq:05d}-{stamp}-{version}.json").write_text(
            json.dumps({"version": version, "time": stamp, **receipt}, indent=1, default=str))

    def _set_head(self, version: str) -> None:
        tmp = self.head_file.with_suffix(".tmp")
        tmp.write_text(version)
        os.replace(tmp, self.head_file)

    def _store_snapshot(self, program: AgentProgram) -> None:
        target = self.root / "snapshots" / program.version
        if target.exists():
            if AgentProgram.load(target).fingerprint() != program.fingerprint():
                raise ValueError(f"program version {program.version!r} already exists with different content")
            return
        scratch = self.root / "snapshots" / f".tmp-{program.version}-{uuid.uuid4().hex[:8]}"
        try:
            program.save(scratch)
            if AgentProgram.load(scratch).fingerprint() != program.fingerprint():
                raise OSError(f"snapshot {program.version!r} did not read back identically")
            os.rename(scratch, target)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def commit(self, program: AgentProgram, receipt: dict[str, Any] | None = None, *, activate: bool = True,
               expected_parent: str | None = None) -> str:
        """Store a snapshot (immutable per version id), optionally make it the head and record the receipt.

        With ``expected_parent`` the commit fails with :class:`StaleHead` unless that version is still the head, so two
        promotions derived from the same program cannot silently overwrite each other.
        """
        with self._locked():
            if expected_parent is not None and self.head() != expected_parent:
                raise StaleHead(f"the active program is {self.head()!r}, not {expected_parent!r}")
            self._store_snapshot(program)
            if receipt is not None:
                self._receipt(program.version, receipt)
            if activate:
                self._set_head(program.version)
        return program.version

    def rollback(self, version: str) -> str:
        """Make an earlier snapshot the head again (after checking that it loads) and record the change."""
        with self._locked():
            if version not in self.versions():
                raise KeyError(f"no snapshot {version!r}; have {self.versions()}")
            AgentProgram.load(self.root / "snapshots" / version)
            before = self.head()
            self._set_head(version)
            self._receipt(version, {"event": "rollback", "from": before, "to": version})
        return version

    def open(self, **seed_kw: Any) -> AgentProgram:
        """The active program; on first use the seed program is created and stored as the first snapshot."""
        head = self.head()
        if head is not None:
            prog = self.load(head)
            if prog is None:
                raise RuntimeError(f"HEAD names {head!r} but {self.root / 'snapshots' / head} is missing; restore it or "
                                   f"roll back to one of {self.versions()}")
            return prog
        from scienceclaw.program.seed import seed_program
        prog = seed_program(**seed_kw)
        self.commit(prog, {"event": "seed", "skills": len(prog.skills), "operators": len(prog.operators)})
        return prog
