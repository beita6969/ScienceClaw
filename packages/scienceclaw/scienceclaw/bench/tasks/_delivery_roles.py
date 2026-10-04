"""Resolver for the data team's versioned deliveries (``<data_root>/<dataset>/reconstructed_vN``) and role files.

Private helper (the leading underscore keeps it out of the registry) of the adapters that read a delivery whose
episode roles are frozen in JSONL files: FoR30 (PhenoBench) and FoR33 (BuildingsBench).

Layout of a delivery (``DATASETS.md`` of the data team)::

    <delivery_root>/                      # = <data_root>/..   (``data_root`` is ``<delivery_root>/datasets``)
        configs/data-delivery-v1/catalog.json            # which reconstructed_vN is current, which role files bind it
        configs/episode-designs/episodes/<...>/{source,val,id,ood}.jsonl     # one row per episode unit
        datasets/<dataset>/reconstructed_vN/             # the data itself

* The **catalog** (newest ``configs/data-delivery-v*/catalog.json``) names the current data directory
  (``data_root`` of the entry of a FoR code, e.g. ``.../for33-buildingsbench/reconstructed_v2``) and the role files
  with their expected sha256. It stores absolute paths of the machine that produced it
  (``catalog["root"]``); they are re-anchored under the delivery root that is actually being read, so that a
  relocated copy of the delivery works and tests can point at a temporary copy.
* Role rows are also absolute-path bearing; :meth:`Delivery.anchor` re-anchors them the same way.
* Nothing here writes to the data directories.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROLE_KEYS = ("source", "val", "id", "ood")
_VERSION_RE = re.compile(r"^reconstructed_v(\d+)$")
_SHA_CACHE: dict[tuple[str, int, int], str] = {}
_SHA_LOCK = threading.Lock()


class DeliveryError(ValueError):
    """The delivery catalog / role files are missing, unreadable, inconsistent or fail their sha256."""


def file_sha256(path: Path) -> str:
    """sha256 of a file (memoised per process on path, size and mtime)."""
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    with _SHA_LOCK:
        hit = _SHA_CACHE.get(key)
    if hit is not None:
        return hit
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    with _SHA_LOCK:
        _SHA_CACHE[key] = digest
    return digest


def version_number(name: str) -> int | None:
    m = _VERSION_RE.match(name)
    return int(m.group(1)) if m else None


def newest_version_dir(dataset_dir: Path) -> Path | None:
    """The ``reconstructed_v<highest N>`` sub-directory of ``dataset_dir`` (None if there is none)."""
    if not dataset_dir.is_dir():
        return None
    cands = [(version_number(p.name), p) for p in dataset_dir.iterdir() if p.is_dir()]
    cands = [(v, p) for v, p in cands if v is not None]
    return max(cands, key=lambda t: t[0])[1] if cands else None


def find_catalog(delivery_root: Path) -> Path:
    """Newest ``configs/data-delivery-v*/catalog.json`` below ``delivery_root``."""
    cfg = delivery_root / "configs"
    cands = sorted(cfg.glob("data-delivery-v*/catalog.json"),
                   key=lambda p: int(re.sub(r"\D", "", p.parent.name) or 0)) if cfg.is_dir() else []
    if not cands:
        raise DeliveryError(f"missing data-delivery catalog {cfg / 'data-delivery-v*' / 'catalog.json'}")
    return cands[-1]


@dataclass
class Delivery:
    """One FoR code's delivery: current data directory, re-anchored role files, parsed catalog entry."""
    code: str
    delivery_root: Path
    catalog_path: Path
    catalog_root: str
    entry: dict[str, Any]
    base: Path                                  # <data_root>/<dataset>/reconstructed_vN
    role_files: dict[str, Path]
    role_sha256: dict[str, str]
    newer_dirs: list[str] = field(default_factory=list)     # reconstructed_vM (M > N) present on disk but not current
    _rows: dict[str, list[dict]] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def version(self) -> str:
        return self.base.name

    def anchor(self, p: str | Path) -> Path:
        """Re-anchor a path of the catalog / role rows under this delivery root."""
        s = str(p)
        if self.catalog_root and (s == self.catalog_root or s.startswith(self.catalog_root.rstrip("/") + "/")):
            return self.delivery_root / s[len(self.catalog_root):].lstrip("/")
        path = Path(s)
        return path if path.is_absolute() else self.delivery_root / path

    def missing(self) -> list[str]:
        out = [f"missing role file {p}" for p in self.role_files.values() if not p.is_file()]
        if not self.base.is_dir():
            out.append(f"missing data directory {self.base}")
        return out

    def rows(self, role: str) -> list[dict]:
        """Parsed JSONL rows of a role (``source``/``val``/``id``/``ood``); sha256 checked against the catalog."""
        with self._lock:
            if role not in self._rows:
                path = self.role_files[role]
                if not path.is_file():
                    raise DeliveryError(f"missing role file {path}")
                want = self.role_sha256.get(role)
                if want and file_sha256(path) != want:
                    raise DeliveryError(f"{path.name}: sha256 differs from the catalog ({want[:12]}...)")
                out: list[dict] = []
                for ln, line in enumerate(path.read_text().splitlines(), 1):
                    if not line.strip():
                        continue
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError as ex:
                        raise DeliveryError(f"{path.name}:{ln}: invalid JSON ({ex})") from ex
                self._rows[role] = out
            return self._rows[role]


def resolve_delivery(code: str, data_root: str | Path, delivery_root: str | Path | None = None) -> Delivery:
    """Resolve ``code`` (e.g. ``"FoR33"``) against the catalog found next to ``data_root``.

    ``data_root`` is the directory that holds the ``for*-*`` dataset directories (the adapters' data root);
    ``delivery_root`` defaults to its parent (which holds ``configs/`` and ``datasets/``). The current data
    directory is the catalog's ``data_root`` for the code, i.e. the newest ``reconstructed_vN`` the data team
    has bound to the role files.
    """
    data_root = Path(data_root)
    droot = Path(delivery_root) if delivery_root else data_root.parent
    cat_path = find_catalog(droot)
    try:
        catalog = json.loads(cat_path.read_text())
    except (OSError, json.JSONDecodeError) as ex:
        raise DeliveryError(f"{cat_path}: unreadable catalog ({type(ex).__name__}: {ex})") from ex
    entries = [e for e in catalog.get("datasets", []) if e.get("for_code") == code]
    if not entries:
        raise DeliveryError(f"{cat_path.name}: no entry for {code}")
    entry = entries[0]
    croot = str(catalog.get("root", ""))
    d = Delivery(code=code, delivery_root=droot, catalog_path=cat_path, catalog_root=croot, entry=entry,
                 base=Path(), role_files={}, role_sha256={})
    if not entry.get("data_root"):
        raise DeliveryError(f"{code}: catalog entry has no data_root")
    cat_base = d.anchor(entry["data_root"])
    # the adapters' data_root wins over the anchored location when both name the same dataset directory
    base = data_root / cat_base.parent.name / cat_base.name
    if version_number(base.name) is None:
        raise DeliveryError(f"{code}: catalog data_root {entry['data_root']} is not a reconstructed_vN directory")
    d.base = base
    roles = entry.get("partition_role_files") or {}
    for r in ROLE_KEYS:
        if r not in roles or not roles[r].get("path"):
            raise DeliveryError(f"{code}: catalog entry lists no role file for {r!r}")
        d.role_files[r] = d.anchor(roles[r]["path"])
        d.role_sha256[r] = str(roles[r].get("expected_sha256") or roles[r].get("sha256") or "")
    cur = version_number(base.name)
    d.newer_dirs = sorted(p.name for p in base.parent.glob("reconstructed_v*")
                          if (version_number(p.name) or -1) > (cur or 0))
    return d
