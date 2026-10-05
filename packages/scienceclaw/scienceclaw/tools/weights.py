"""Registry of the pretrained model weights the tool library can use.

``weights.json`` is the single source of truth: where each checkpoint comes from, which sub-directory of the model root it
lives in, which files prove it is complete, which ``scilib`` modules read it and which Python packages they need. Nothing
here downloads anything: :func:`status` inspects the local model root and :func:`plan` prints the commands that stage a
checkpoint, so weights always arrive by an explicit, reviewable step.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_DATA = Path(__file__).with_name("weights.json")


@dataclass(frozen=True)
class WeightAsset:
    id: str
    title: str
    kind: str                      # huggingface | url | pip | manual
    source: str | None
    subdir: str | None             # relative to the model root (None for pip packages)
    check: tuple[str, ...]         # globs, relative to subdir, that must all match
    used_by: tuple[str, ...]       # scilib modules
    requires: tuple[str, ...]      # python packages (import names)
    license: str | None = None
    files: tuple[str, ...] = ()    # url assets that consist of several files, relative to `source`
    sha256: str | None = None
    bytes: int | None = None
    env: str | None = None         # environment variable that overrides the location
    restricted: bool = False
    note: str | None = None
    flatten: bool = False          # multi-file url assets: store every file directly in `subdir`
    post: tuple[str, ...] = ()     # shell commands run after the download; "{dir}" is the asset directory
    allow_bin: bool = False        # huggingface assets: keep *.bin files (the repo ships no safetensors)
    include: tuple[str, ...] = ()  # huggingface assets: download only these files (patterns) of a multi-checkpoint repo
    extra: dict[str, Any] = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "kind": self.kind, "source": self.source, "subdir": self.subdir,
                "check": list(self.check), "used_by": list(self.used_by), "requires": list(self.requires),
                "license": self.license, "sha256": self.sha256, "env": self.env, "restricted": self.restricted,
                "note": self.note}


# Framework-specific duplicates of the PyTorch weights that are never needed.
_HF_SKIP = ["*.msgpack", "*.h5", "*.ot", "flax_*", "tf_*", "onnx/*", "*.onnx", "openvino/*"]


def model_root() -> Path:
    """The directory weights are read from: ``$SCIENCECLAW_MODELS`` or ``~/.cache/scienceclaw/models``."""
    return Path(os.environ.get("SCIENCECLAW_MODELS") or Path.home() / ".cache" / "scienceclaw" / "models")


def load() -> list[WeightAsset]:
    doc = json.loads(_DATA.read_text(encoding="utf-8"))
    out = []
    for a in doc["assets"]:
        known = {"id", "title", "kind", "source", "subdir", "check", "used_by", "requires", "license", "files", "sha256",
                 "bytes", "env", "restricted", "note", "flatten", "post", "allow_bin", "include"}
        out.append(WeightAsset(
            id=a["id"], title=a["title"], kind=a["kind"], source=a.get("source"), subdir=a.get("subdir"),
            check=tuple(a.get("check", ())), used_by=tuple(a.get("used_by", ())), requires=tuple(a.get("requires", ())),
            license=a.get("license"), files=tuple(a.get("files", ())), sha256=a.get("sha256"), bytes=a.get("bytes"),
            env=a.get("env"), restricted=bool(a.get("restricted", False)), note=a.get("note"),
            flatten=bool(a.get("flatten", False)), post=tuple(a.get("post", ())), allow_bin=bool(a.get("allow_bin", False)), include=tuple(a.get("include", ())),
            extra={k: v for k, v in a.items() if k not in known}))
    return out


def get(asset_id: str) -> WeightAsset:
    for a in load():
        if a.id == asset_id:
            return a
    raise KeyError(f"unknown weight asset {asset_id!r}; known: {[a.id for a in load()]}")


def _have(pkg: str) -> bool:
    try:
        return importlib.util.find_spec(pkg) is not None
    except (ImportError, ValueError):
        return False


def _dir_of(a: WeightAsset) -> Path | None:
    if a.env and os.environ.get(a.env):
        p = Path(os.environ[a.env]).expanduser()
        return p if p.is_dir() else p.parent
    return model_root() / a.subdir if a.subdir else None


def status(asset: WeightAsset | str) -> dict[str, Any]:
    """Is this asset staged, and can its Python dependencies be imported?"""
    a = get(asset) if isinstance(asset, str) else asset
    missing_pkgs = [p for p in a.requires if not _have(p)]
    base = _dir_of(a)
    if a.kind == "pip":
        present, missing, path = not missing_pkgs, [], None
    elif base is None or not base.exists():
        present, missing, path = False, list(a.check) or ["<directory>"], str(base) if base else None
    else:
        missing = [g for g in a.check if not any(any(base.glob(alt)) for alt in g.split("|"))]
        present, path = not missing, str(base)
    size = None
    if present and base is not None and base.is_dir():
        size = sum(f.stat().st_size for f in base.rglob("*") if f.is_file())
    return {"id": a.id, "title": a.title, "present": present, "path": path, "missing_files": missing,
            "missing_packages": missing_pkgs, "ready": present and not missing_pkgs, "restricted": a.restricted,
            "used_by": list(a.used_by), "bytes": size}


def status_all() -> list[dict[str, Any]]:
    return [status(a) for a in load()]


def verify(asset: WeightAsset | str) -> dict[str, Any]:
    """Check the sha256 of a single-file asset against the registry (assets without a recorded hash report ``checked: false``)."""
    a = get(asset) if isinstance(asset, str) else asset
    st = status(a)
    if not st["present"] or not a.sha256 or len(a.check) != 1:
        return {"id": a.id, "checked": False, "ok": st["present"] if not a.sha256 else False,
                "reason": "not staged" if not st["present"] else "no recorded hash"}
    base = _dir_of(a)
    target = next(iter(base.glob(a.check[0])))
    h = hashlib.sha256()
    with target.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {"id": a.id, "checked": True, "ok": h.hexdigest() == a.sha256, "sha256": h.hexdigest(), "expected": a.sha256}


def plan(ids: list[str] | None = None, root: str | Path | None = None) -> list[str]:
    """Shell commands that stage the requested assets (all of them when ``ids`` is None). Restricted assets are skipped."""
    dest = Path(root).expanduser() if root else model_root()
    q = shlex.quote
    cmds: list[str] = [f"mkdir -p {q(str(dest))}"]
    for a in load():
        if (ids and a.id not in ids) or a.restricted:
            continue
        cmds.append(f"# {a.id}: {a.title}" + (f" ({a.license})" if a.license else ""))
        target = dest / a.subdir if a.subdir else None
        if a.kind == "huggingface":
            skip = _HF_SKIP + ([] if a.allow_bin else ["*.bin"])
            cmds.append(f"hf download {q(a.source)} --local-dir {q(str(target))}"
                        + (f" --revision {q(a.extra.get('revision', ''))}" if a.extra.get("revision") else "")
                        + "".join(f" --include {q(x)}" for x in a.include)
                        + ("" if a.include else "".join(f" --exclude {q(x)}" for x in skip)))
        elif a.kind == "url" and a.files:
            for f in a.files:
                out = target / (Path(f).name if a.flatten else f)
                cmds.append(f"mkdir -p {q(str(out.parent))} && curl -fL --retry 5 -C - -o {q(str(out))} {q(a.source.rstrip('/') + '/' + f)}")
        elif a.kind == "url":
            name = a.check[0] if a.check else Path(a.source).name
            out = target / name
            cmds.append(f"mkdir -p {q(str(target))} && curl -fL --retry 5 -C - -o {q(str(out))} {q(a.source)}")
        elif a.kind == "pip":
            cmds.append(f"pip install {q(a.source)}")
        else:
            cmds.append(f"# stage manually into {target}" + (f" -- {a.note}" if a.note else ""))
        for cmd in a.post:
            cmds.append(cmd.replace("{dir}", q(str(target))))
        if a.sha256 and len(a.check) == 1 and target is not None:
            cmds.append(f"echo {q(a.sha256 + '  ' + str(target / a.check[0]))} | sha256sum -c -")
    return cmds
