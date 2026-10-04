#!/usr/bin/env python3
"""Small JSON protocol for the native ScienceClaw benchmark plugin.

The TypeScript gateway calls this file with one JSON request on stdin and reads
one JSON response from stdout.  It exposes the benchmark catalog, task
availability, and report generation for existing runs; formal hidden-split
evaluation remains an explicit server-side operation.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    """Convert non-finite floats so stdout remains standards-compliant JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def emit(payload: dict[str, Any]) -> int:
    print(json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True,
                     default=str, allow_nan=False), flush=True)
    return 0 if payload.get("status") == "ok" else 1


def request() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        raise ValueError("empty request")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("request must be a JSON object")
    if value.get("schema", 1) != 1:
        raise ValueError("unsupported request schema")
    return value


def main() -> int:
    try:
        req = request()
        op = str(req.get("op", "")).strip()
        repo_root = Path(__file__).resolve().parent
        run_root = Path(str(req.get("run_root") or repo_root / "runs")).expanduser().resolve()
        run_root.mkdir(parents=True, exist_ok=True)
        data_root = req.get("data_root")
        if data_root:
            os.environ["SCIENCECLAW_DATA_ROOT"] = str(Path(str(data_root)).expanduser().resolve())
        model_root = req.get("model_root")
        if model_root:
            os.environ["SCIENCECLAW_MODELS"] = str(Path(str(model_root)).expanduser().resolve())

        if op == "catalog":
            import re
            from scienceclaw.bench.registry import DISCIPLINES

            task_dir = repo_root / "scienceclaw" / "bench" / "tasks"
            tool_refs: set[str] = set()
            # Only registered FoR adapters contribute tool refs.  ToolSpec calls
            # may span lines, so whitespace after the opening parenthesis is allowed.
            for discipline in DISCIPLINES:
                source = task_dir / f"{discipline.module}.py"
                if not source.is_file():
                    continue
                text = source.read_text(encoding="utf-8")
                tool_refs.update(re.findall(r'ToolSpec\s*\(\s*"([A-Za-z0-9_]+)"', text))
            configs = sorted(p.name for p in (repo_root / "configs").iterdir() if p.is_file())
            scripts = sorted(str(p.relative_to(repo_root)) for p in (repo_root / "scripts").rglob("*") if p.is_file() and p.suffix in {".py", ".sh", ".sbatch"})
            return emit({"schema": 1, "status": "ok", "op": op,
                         "disciplines": [d.__dict__ for d in DISCIPLINES],
                         "tool_refs": sorted(tool_refs), "configs": configs, "scripts": scripts})

        if op == "list_tasks":
            from scienceclaw.config import BenchConfig
            from scienceclaw.bench.splits import adapter_status

            cfg = BenchConfig()
            if data_root:
                cfg.data_root = str(Path(str(data_root)).expanduser().resolve())
            rows = adapter_status(cfg)
            return emit({"schema": 1, "status": "ok", "op": op, "tasks": rows})

        if op == "report":
            from scienceclaw.experiments.report import build_report

            run_id = str(req.get("run_id", "")).strip()
            if not run_id or Path(run_id).name != run_id:
                raise ValueError("run_id must be a single directory name")
            run_dir = (run_root / run_id).resolve()
            if run_dir.parent != run_root or not run_dir.is_dir():
                raise ValueError("run_id is outside the configured run root or does not exist")
            report_path = build_report(run_dir)
            return emit({"schema": 1, "status": "ok", "op": op,
                         "run_id": run_id, "report": str(report_path)})

        raise ValueError("op must be one of: catalog, list_tasks, report")
    except Exception as exc:  # fail closed; the gateway receives structured error text
        return emit({"schema": 1, "status": "error", "error": str(exc),
                     "type": type(exc).__name__})


if __name__ == "__main__":
    raise SystemExit(main())
