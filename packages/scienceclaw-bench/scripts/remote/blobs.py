"""Content-addressed transfer of large array arguments between the broker (Mac) and the worker (GPU host).

The uplink to the GPU host is slow, so an array argument that the host has already received is not sent again. A large
``{"__nd__": base64 npy}`` argument is cut into chunks along its first axis (whole rows, about 1 MB per chunk); each chunk is
stored on the host as ``<sha256 of its npy bytes>.npy`` and the request carries ``{"__rows__": [hash, ...]}`` plus only the
chunks the host does not have. Nothing about the answer changes: the worker rebuilds exactly the same array, and the spool key
that makes replays reproducible is computed by the sandbox from the full arguments.
"""
from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path

import numpy as np

MIN_B64 = 262_144          # arrays with a shorter base64 text travel inline
CHUNK_BYTES = 1_000_000


def npy_bytes(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(a), allow_pickle=False)
    return buf.getvalue()


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()[:32]


def chunks(a: np.ndarray) -> list[tuple[str, bytes]]:
    """(hash, npy bytes) of the row chunks of ``a`` (ndim >= 2)."""
    row = max(1, a[0].nbytes)
    step = max(1, CHUNK_BYTES // row)
    out = []
    for i in range(0, a.shape[0], step):
        raw = npy_bytes(a[i:i + step])
        out.append((digest(raw), raw))
    return out


def pack(obj, known: set[str], blobs: dict[str, bytes]):
    """Replace big array arguments by row-chunk references; chunks not in ``known`` are added to ``blobs``."""
    if isinstance(obj, dict):
        if set(obj) == {"__nd__"} and len(obj["__nd__"]) >= MIN_B64:
            a = np.load(io.BytesIO(base64.b64decode(obj["__nd__"])), allow_pickle=False)
            if a.ndim >= 2 and a.shape[0] >= 1 and a.dtype != object:
                hs = []
                for h, raw in chunks(a):
                    hs.append(h)
                    if h not in known:
                        blobs[h] = raw
                return {"__rows__": hs}
            return obj
        return {k: pack(v, known, blobs) for k, v in obj.items()}
    if isinstance(obj, list):
        return [pack(v, known, blobs) for v in obj]
    return obj


def store(root: Path, blobs: dict[str, str]) -> None:
    """Write base64 chunk payloads into ``root`` after checking each against its name."""
    root.mkdir(parents=True, exist_ok=True)
    for h, b64 in blobs.items():
        raw = base64.b64decode(b64)
        if digest(raw) != h:
            raise ValueError(f"blob {h} does not match its content")
        tmp = root / f".{h}.tmp"
        tmp.write_bytes(raw)
        tmp.replace(root / f"{h}.npy")


class MissingBlobs(Exception):
    def __init__(self, missing: list[str]):
        super().__init__(f"{len(missing)} array chunks are not on this host")
        self.missing = missing


def unpack(obj, root: Path):
    """Inverse of :func:`pack`: rebuild arrays from stored chunks (returns the object with ``{"__nd__"}`` arrays as ndarrays)."""
    if isinstance(obj, dict):
        if set(obj) == {"__rows__"}:
            parts, missing = [], []
            for h in obj["__rows__"]:
                f = root / f"{h}.npy"
                if not f.is_file():
                    missing.append(h)
                    continue
                raw = f.read_bytes()
                if digest(raw) != h:
                    missing.append(h)
                    continue
                parts.append(np.load(io.BytesIO(raw), allow_pickle=False))
            if missing:
                raise MissingBlobs(missing)
            return np.concatenate(parts, axis=0)
        return {k: unpack(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [unpack(v, root) for v in obj]
    return obj
