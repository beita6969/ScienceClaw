"""Evolve A_0 -> A_R over the source stream D_src (paper Eq. 2-3; DESIGN §7 receipts).

``run_stream(cfg)`` creates ``<runs_root>/<name>-<YYYYmmdd-HHMMSS>/`` containing

* ``config.yaml``   — the full RunConfig (every default made explicit)
* ``splits.json``   — the frozen split manifest (episode ids, lineage, seeds, sha256)
* ``run.json``      — status, timings, snapshot versions/fingerprints and the provenance receipt (git commit +
  dirty flag + diff hash, config hash, package versions, dataset manifest hashes, LLM config, cache path/salt) of
  the first segment (``provenance``) and of every resume segment (``resumes[].provenance``)
* ``programs/A_r/`` — snapshots (written by the Evolver; any missing snapshot is saved here from its return value)
* ``usage.jsonl``   — append-only ledger, one record per process segment and phase (start/end time, wall time,
  ``llm.usage()`` delta, provenance). A resume adds its own record instead of overwriting (M6)
* ``usage.json``    — the evolution-phase sum of that ledger (``llm`` total / by role / by tag, ``wall_s``, number of
  segments; ``unclosed`` lists segments of killed processes whose spend is not recorded = lower bound)
* whatever the Evolver writes (``episodes/``, ``candidates.jsonl``, ``metrics.jsonl`` ...)

The LLM client is created from ``cfg.llm`` unless one is injected (tests / smoke use ``llm.fake.FakeLLM``).
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import RunConfig
from ..core.program import AgentProgram
from .provenance import UsageLedger, code_version, provenance, read_ledger, sum_ledger  # noqa: F401 (re-exported)

SNAPSHOT_RE = re.compile(r"^A_?(\d+)$")
EVOLVE_PHASES = ("evolve",)


def make_run_dir(cfg: RunConfig, now: datetime | None = None) -> Path:
    """``<runs_root>/<name>-<timestamp>`` (suffix ``-2``, ``-3``... if it already exists)."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    safe = re.sub(r"[^A-Za-z0-9_.\-]+", "_", cfg.name) or "run"
    base = Path(cfg.runs_root).expanduser() / f"{safe}-{stamp}"
    cand, i = base, 1
    while cand.exists():
        i += 1
        cand = base.with_name(f"{base.name}-{i}")
    cand.mkdir(parents=True)
    return cand


def snapshot_dirs(run_dir: str | Path) -> dict[int, Path]:
    """round r -> programs/A_r directory (accepts ``A_r`` and ``Ar`` names)."""
    out: dict[int, Path] = {}
    pdir = Path(run_dir) / "programs"
    if pdir.is_dir():
        for d in pdir.iterdir():
            m = SNAPSHOT_RE.match(d.name)
            if m and (d / "program.json").exists():
                out.setdefault(int(m.group(1)), d)
    return dict(sorted(out.items()))


def save_snapshots(run_dir: str | Path, snapshots: list[AgentProgram]) -> dict[int, Path]:
    """Persist snapshots A_0..A_R not already written by the Evolver (as ``programs/A_r``)."""
    have = snapshot_dirs(run_dir)
    for r, prog in enumerate(snapshots):
        if r not in have:
            d = Path(run_dir) / "programs" / f"A_{r}"
            prog.save(d)
    return snapshot_dirs(run_dir)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=1, sort_keys=False, default=str))


def run_stream(cfg: RunConfig | None, adapters: dict | None = None, llm: Any = None,
               program0: AgentProgram | None = None, resume_dir: str | Path | None = None) -> Path:
    """Run the evolution stream and return the run directory.

    With ``resume_dir`` the existing run is continued: its own ``config.yaml`` is used (``cfg`` is
    ignored), the split manifest is rebuilt and must hash to the recorded one, and the Evolver skips
    source episodes already recorded in ``stream.jsonl``.
    """
    from ..bench.splits import SplitPlan, load_adapters
    from ..config import load_config

    t0 = time.time()
    if resume_dir is not None:
        run_dir = Path(resume_dir)
        cfg = load_config(run_dir / "config.yaml")
        status = json.loads((run_dir / "run.json").read_text())
        status.setdefault("resumes", []).append({"at": datetime.now().isoformat(timespec="seconds"),
                                                 "code": code_version()})
        status["status"] = "running"
        _import_legacy_usage(run_dir)
    else:
        assert cfg is not None
        run_dir = make_run_dir(cfg)
        cfg.dump(run_dir / "config.yaml")
        status = {"name": cfg.name, "run_dir": str(run_dir), "status": "running",
                  "started": datetime.now().isoformat(timespec="seconds"), "code": code_version(),
                  "llm_injected": llm is not None,
                  "note": "rebuild experiment; never back-fills numbers of the submitted paper"}
    _write_json(run_dir / "run.json", status)
    if adapters is None:
        adapters = load_adapters(cfg.bench)
    plan = SplitPlan.build(cfg.bench, adapters)
    if resume_dir is not None:
        old = json.loads((run_dir / "splits.json").read_text()).get("sha256")
        new_sha = plan.manifest()["sha256"]
        if old != new_sha:
            raise RuntimeError(f"cannot resume {run_dir}: split manifest changed ({old} -> {new_sha})")
        manifest = plan.manifest()
    else:
        manifest = plan.save(run_dir / "splits.json")
    status.update(disciplines=plan.order, split_sha256=manifest["sha256"], n_source=len(plan.source_stream()),
                  split_warnings=plan.warnings)
    _write_json(run_dir / "run.json", status)
    if llm is None:
        from ..llm.client import LLMClient
        llm = LLMClient(cfg.llm)
    from ..evolution.evolver import Evolver

    # provenance receipt of THIS segment (m3+m4): first segment -> status["provenance"], resume -> resumes[-1]
    prov = provenance(cfg, "evolve", llm=llm, extra={"split_sha256": manifest["sha256"],
                                                     "resume": resume_dir is not None})
    if resume_dir is not None:
        status["resumes"][-1]["provenance"] = prov
    else:
        status["provenance"] = prov
    _write_json(run_dir / "run.json", status)

    try:
        # M6: cost/wall of every segment is appended to usage.jsonl; usage.json / run.json wall_s are its sums
        with UsageLedger(run_dir, "evolve", llm, prov, extra={"resume": resume_dir is not None}):
            snapshots = Evolver(cfg, llm, plan, run_dir).run(program0 or AgentProgram())
    except BaseException as ex:
        tot = _write_usage(run_dir)
        status.update(status="failed", error=f"{type(ex).__name__}: {ex}", wall_s=tot["wall_s"],
                      wall_s_last_segment=time.time() - t0)
        _write_json(run_dir / "run.json", status)
        raise
    dirs = save_snapshots(run_dir, list(snapshots or []))
    tot = _write_usage(run_dir)
    status.update(status="done", finished=datetime.now().isoformat(timespec="seconds"), wall_s=tot["wall_s"],
                  wall_s_last_segment=time.time() - t0, segments=tot["segments"],
                  snapshots={f"A_{r}": {"dir": str(d.relative_to(run_dir)),
                                        **AgentProgram.load(d).summary()} for r, d in dirs.items()})
    _write_json(run_dir / "run.json", status)
    return run_dir


def _import_legacy_usage(run_dir: Path) -> bool:
    """A run started before the ledger existed has only ``usage.json`` (overwritten by every resume): carry its
    numbers over as one ``legacy`` ledger record so the next ``usage.json`` adds to them instead of replacing them."""
    led, old = run_dir / UsageLedger.FILE, run_dir / "usage.json"
    if led.exists() or not old.exists():
        return False
    try:
        d = json.loads(old.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    rec = {"segment": "legacy", "phase": "evolve", "status": "legacy", "llm": d.get("llm") or {},
           "wall_s": float(d.get("wall_s") or 0.0),
           "note": "imported from a pre-ledger usage.json; segments overwritten by earlier resumes are not recoverable"}
    with led.open("a") as f:
        f.write(json.dumps(rec, default=str, sort_keys=True) + "\n")
    return True


def _write_usage(run_dir: Path) -> dict:
    """Rewrite ``usage.json`` as the sum of the evolution-phase ledger records; returns the sum."""
    tot = sum_ledger([r for r in read_ledger(run_dir / UsageLedger.FILE) if r.get("phase") in EVOLVE_PHASES])
    _write_json(run_dir / "usage.json", {"llm": tot["llm"], "wall_s": tot["wall_s"], "segments": tot["segments"],
                                         "unclosed": tot["unclosed"], "failed": tot["failed"],
                                         "source": UsageLedger.FILE})
    return tot
