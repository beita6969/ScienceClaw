"""First-install setup: every tool the library wraps is installed, staged and verified before real work starts.

``setup`` does three things and records the outcome in ``$SCIENCECLAW_HOME/setup.json``:

1. Python packages: when a library module cannot import something it needs, the ``[all]`` extra of this package is installed
   into the running interpreter.
2. Pretrained weights and upstream sources: every asset of ``tools/weights.json`` (except optional ones) is staged under the
   model root with the registry's own commands, then verified (files present, hashes where recorded).
3. A report: the availability of every library module, so a remaining gap is named instead of discovered later.

The engine refuses ``canvas.open`` and the evolution methods until the recorded setup is complete (:func:`require`); with
``SCIENCECLAW_AUTO_SETUP=1`` the first such call starts the setup in the background instead of only failing.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from scienceclaw import tools
from scienceclaw.tools import weights as W

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("full", "light")
LIGHT_MAX_GB = 1.5          # the light profile leaves out assets larger than this
MARGIN = 1.3                # free space required: estimated downloads x MARGIN


class SetupRequired(RuntimeError):
    """The tools are not fully installed yet."""


def home() -> Path:
    return Path(os.environ.get("SCIENCECLAW_HOME") or Path.home() / ".scienceclaw").expanduser()


def state_path() -> Path:
    return home() / "setup.json"


def read_state() -> dict[str, Any] | None:
    try:
        return json.loads(state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_state(state: dict[str, Any]) -> None:
    p = state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, p)


# ----------------------------------------------------------------------------------------------------- selection
def select(profile: str = "full", only: list[str] | None = None, skip: list[str] | None = None,
           with_optional: bool = False) -> list[W.WeightAsset]:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {PROFILES}")
    out = []
    for a in W.load():
        if a.restricted or (a.optional and not with_optional):
            continue
        if only and a.id not in only:
            continue
        if skip and a.id in skip:
            continue
        if profile == "light" and a.approx_gb > LIGHT_MAX_GB and not only:
            continue
        out.append(a)
    return out


def missing_assets(assets: list[W.WeightAsset]) -> list[str]:
    """Assets of the selection that are not staged (or whose Python dependencies cannot be imported)."""
    return [a.id for a in assets if a.kind != "manual" and not W.status(a)["present"]]


def missing_python() -> list[str]:
    """Library modules that cannot import a package they need (not a missing-weights or no-GPU condition)."""
    return [r["module"] for r in tools.status_table() if "missing packages" in str(r["reason"])]


def free_gb(path: Path) -> float:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1e9


# ------------------------------------------------------------------------------------------------------- steps
def _env(root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["SCIENCECLAW_MODELS"] = str(root)
    env["HF_HOME"] = str(root / ".hf-cache")
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return env


def _run(script: str, env: dict[str, str], log: Path, timeout: float = 4 * 3600) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as fh:
        fh.write(f"\n$ {script}\n".encode())
        try:
            return subprocess.run(["bash", "-ec", script], env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout,
                                  cwd=str(PACKAGE_ROOT)).returncode
        except subprocess.TimeoutExpired:
            fh.write(b"\n[timed out]\n")
            return 124


def install_python(log: Path) -> dict[str, Any]:
    """Install the ``[all]`` extra into the running interpreter (editable checkout or installed distribution)."""
    target = f"{PACKAGE_ROOT}[all]" if (PACKAGE_ROOT / "pyproject.toml").is_file() else "scienceclaw[all]"
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", target]
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n".encode())
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=4 * 3600).returncode
    return {"target": target, "ok": rc == 0}


def stage_asset(a: W.WeightAsset, root: Path, logs: Path) -> dict[str, Any]:
    t0 = time.monotonic()
    if a.kind == "manual":
        return {"id": a.id, "status": "manual", "note": a.note}
    script = "\n".join(ln for ln in W.plan([a.id], root) if ln.strip() and not ln.startswith("#"))
    env = _env(root)
    rc = 1
    for _ in range(2):                                   # one retry: downloads fail transiently
        rc = _run(script, env, logs / f"{a.id}.log")
        if rc == 0 and W.status(a)["present"]:
            break
    shutil.rmtree(root / ".hf-cache", ignore_errors=True)
    ver = W.verify(a)
    ok = rc == 0 and W.status(a)["present"] and (ver["ok"] or not ver["checked"])
    return {"id": a.id, "status": "ok" if ok else "failed", "returncode": rc, "hash": ver, "seconds": round(time.monotonic() - t0),
            "log": str(logs / f"{a.id}.log")}


# --------------------------------------------------------------------------------------------------------- run
def run(profile: str = "full", only: list[str] | None = None, skip: list[str] | None = None, with_optional: bool = False,
        check_only: bool = False, progress: Callable[[dict], None] | None = None) -> dict[str, Any]:
    """Install and verify everything the selection needs; returns (and records) the setup state."""
    say = progress or (lambda _p: None)
    root = W.model_root()
    logs = home() / "setup"
    assets = select(profile, only, skip, with_optional)
    todo = missing_assets(assets)
    need_pip = bool(missing_python())
    gb = sum(a.approx_gb for a in assets if a.id in todo)
    state: dict[str, Any] = {"profile": profile, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "model_root": str(root),
                             "assets": [a.id for a in assets], "only": list(only or []), "to_stage": todo, "download_gb": round(gb, 1),
                             "free_gb": round(free_gb(root), 1), "python": {"missing_modules": missing_python()},
                             "results": [], "complete": False, "running": not check_only}
    if check_only:
        state["complete"] = not todo and not need_pip
        return state
    if gb * MARGIN > free_gb(root):
        state.update(running=False, error=f"not enough free space under {root}: about {gb:.1f} GB to download, "
                                          f"{free_gb(root):.1f} GB free (use --profile light or --skip to reduce it)")
        _write_state(state)
        return state
    _write_state(state)
    if need_pip:
        say({"step": "python", "message": "installing the Python packages of the tool library"})
        state["python"]["install"] = install_python(logs / "pip.log")
        state["python"]["missing_modules"] = missing_python()
        _write_state(state)
    for i, a in enumerate(a for a in assets if a.id in todo):
        say({"step": "weights", "asset": a.id, "index": i + 1, "total": len(todo), "approx_gb": a.approx_gb})
        state["results"].append(stage_asset(a, root, logs))
        state["current"] = a.id
        _write_state(state)
    still = missing_assets(assets)
    state.update(running=False, missing_assets=still, finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                 python={**state["python"], "missing_modules": missing_python()},
                 modules=[r for r in tools.status_table() if not r["available"]])
    state["partial"] = bool(only or skip)
    state["complete"] = not still and not state["python"]["missing_modules"]
    state.pop("current", None)
    _write_state(state)
    return state


# -------------------------------------------------------------------------------------------------- the gate
def status() -> dict[str, Any]:
    """Whether the recorded setup is complete and still true on disk (cheap enough to run on every gated call)."""
    st = read_state()
    if st is None:
        return {"complete": False, "state": "not set up", "detail": "run `python -m scienceclaw.cli setup`"}
    if st.get("running"):
        return {"complete": False, "state": "running", "detail": st.get("current") or "starting", "profile": st.get("profile")}
    if not st.get("complete"):
        return {"complete": False, "state": "incomplete", "detail": st.get("error") or
                f"missing assets {st.get('missing_assets')}, modules without packages {st.get('python', {}).get('missing_modules')}",
                "profile": st.get("profile")}
    if st.get("partial") and os.environ.get("SCIENCECLAW_ALLOW_PARTIAL_SETUP") != "1":
        return {"complete": False, "state": "partial", "profile": st.get("profile"),
                "detail": "only a subset of the tools was installed (--only/--skip); run `setup` without them, or set "
                          "SCIENCECLAW_ALLOW_PARTIAL_SETUP=1 to accept it"}
    assets = select(st.get("profile", "full"), st.get("only") or None, None)
    lost = missing_assets([a for a in assets if a.id in st.get("assets", [])])
    if lost:
        return {"complete": False, "state": "incomplete", "detail": f"staged assets are gone: {lost}", "profile": st.get("profile")}
    return {"complete": True, "state": "ready", "profile": st.get("profile")}


_job: dict[str, Any] = {"thread": None}


def start_background(profile: str | None = None) -> dict[str, Any]:
    """Run the setup in a background thread (one at a time); returns the current status."""
    th = _job["thread"]
    if th is not None and th.is_alive():
        return status()
    profile = profile or os.environ.get("SCIENCECLAW_SETUP_PROFILE", "full")
    th = threading.Thread(target=run, kwargs={"profile": profile}, name="scienceclaw-setup", daemon=True)
    _job["thread"] = th
    th.start()
    return {"complete": False, "state": "running", "detail": "starting", "profile": profile}


def require(auto: bool | None = None) -> None:
    """Raise :class:`SetupRequired` unless the tools are fully installed; with ``auto`` (default: $SCIENCECLAW_AUTO_SETUP) a
    missing or incomplete setup is started in the background first."""
    if os.environ.get("SCIENCECLAW_SKIP_SETUP_CHECK") == "1":       # development and tests only
        return
    st = status()
    if st["complete"]:
        return
    if auto is None:
        auto = os.environ.get("SCIENCECLAW_AUTO_SETUP") == "1"
    if auto and st["state"] != "running":
        st = start_background()
    hint = ("the installation of the tool library is running in the background; try again when it has finished "
            "(scienceclaw_tools operation=setup shows the progress)" if st["state"] == "running" else
            "the tool library is not installed yet: run `python -m scienceclaw.cli setup` (or enable autoSetup in the plugin "
            "configuration)")
    raise SetupRequired(f"ScienceClaw setup is {st['state']} ({st.get('detail')}); {hint}")
