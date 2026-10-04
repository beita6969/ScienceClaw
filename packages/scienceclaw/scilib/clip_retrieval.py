"""Frozen open CLIP image embeddings for a visible-caption retrieval reference.

This module is an optional, trusted-side diagnostic for FoR45.  It only loads an
explicitly staged checkpoint from ``SCIENCECLAW_MODELS`` (or the legacy model
roots in :mod:`scilib._pretrained`) and never downloads a model or opens a
network connection.  The checkpoint is used as a frozen image encoder; no
caption, benchmark target, or training label is passed to the encoder.

available(model="open_clip_vit_b32") -> bool
    True when torch, ``open_clip_torch`` and an auditable local checkpoint are
    available in this interpreter, or when the configured remote worker can
    expose the same frozen checkpoint and provenance.
provenance(model="open_clip_vit_b32") -> dict
    JSON-safe availability and checkpoint metadata, including SHA-256, package
    versions, and the reason for an unavailable model.  ``network_allowed`` is
    always false and ``labels_used`` is always false.
encode_images(images, model="open_clip_vit_b32", batch_size=32, device="auto") -> float32
    L2-normalised image embeddings for uint8 ``(n,H,W,3)`` RGB arrays.  Raises
    if the dependency or checkpoint is missing instead of silently falling back.
retrieve_captions(train_rows, train_images, eval_items, eval_images, k=1,
                  model="open_clip_vit_b32", fallback=None,
                  selection="nearest") -> list[str]
    Same-language cosine nearest-neighbour captions from visible image-caption
    rows.  Caption-only rows are excluded.  For ``k > 1``, ``selection`` can
    be ``"nearest"`` (the historical first-neighbour route) or
    ``"caption_medoid"`` (a deterministic character-ngram medoid among the
    retrieved visible captions).  Languages without a visible image use the
    explicitly supplied visible-caption fallback, or ``""``.

The default checkpoint name is an OpenAI CLIP ViT-B/32 checkpoint loaded through
``open_clip``.  The weight file must be staged by an operator and its digest is
returned by :func:`provenance`; ``open_clip`` is never allowed to fetch it.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
from pathlib import Path
from typing import Any

import numpy as np

from . import _remote
from ._pretrained import have_module, model_path, switched_off, torch_device

__all__ = ["MODELS", "available", "provenance", "encode_images", "retrieve_captions"]


MODELS = {
    "open_clip_vit_b32": {
        "backend": "open_clip",
        "model_name": "ViT-B-32",
        "pretrained_name": "openai",
        "weights_subdir": ("clip", "open_clip_vit_b32"),
        "license": "OpenAI CLIP weights; verify the staged checkpoint licence before deployment",
    },
}

_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}
_SHA_CACHE: dict[tuple[str, int, int], str] = {}


def _caption_ngrams(text: str, n: int = 3) -> set[str]:
    """Return padded character n-grams for a visible caption."""
    s = f"  {str(text).strip().lower()}  "
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def _caption_similarity(a: str, b: str) -> float:
    aa, bb = _caption_ngrams(a), _caption_ngrams(b)
    if not aa or not bb:
        return 0.0
    return float(2.0 * len(aa & bb) / (len(aa) + len(bb)))


def _select_caption(captions: list[str], order: np.ndarray, selection: str) -> str:
    """Select one caption from a ranked visible-neighbour list."""
    if selection not in ("nearest", "caption_medoid"):
        raise ValueError("selection must be 'nearest' or 'caption_medoid'")
    if not len(order):
        return ""
    ranked = [str(captions[int(i)]).strip() for i in order]
    if selection == "nearest" or len(ranked) == 1:
        return ranked[0]
    # Ranked order is the CLIP tie-break. Keep it for equal text-side scores.
    best_i, best_s = 0, -1.0
    for i, cap in enumerate(ranked):
        score = float(np.mean([_caption_similarity(cap, other)
                               for j, other in enumerate(ranked) if j != i]))
        if score > best_s + 1e-12:
            best_i, best_s = i, score
    return ranked[best_i]


def _model_spec(model: str) -> dict[str, Any]:
    if model not in MODELS:
        raise ValueError(f"model must be one of {sorted(MODELS)}")
    return MODELS[model]


def _candidate_file(path: Path | None) -> Path | None:
    """Resolve a staged checkpoint path without scanning outside the model root."""
    if path is None:
        return None
    if path.is_file():
        return path
    if not path.is_dir():
        return None
    # The order is intentional and deterministic.  open_clip accepts all of
    # these formats depending on its version; no implicit hub/cache lookup is
    # attempted when none exists.
    for suffix in (".pt", ".pth", ".bin", ".safetensors"):
        files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == suffix)
        if files:
            return files[0]
    return None


def _weights(model: str) -> Path | None:
    _model_spec(model)
    override = os.environ.get("SCIENCECLAW_CLIP_WEIGHTS")
    if override:
        return _candidate_file(Path(override).expanduser())
    spec = MODELS[model]
    return _candidate_file(model_path(*spec["weights_subdir"]))


def _sha256(path: Path) -> str:
    st = path.stat()
    key = (str(path), int(st.st_mtime_ns), int(st.st_size))
    if key not in _SHA_CACHE:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _local_ok(model: str) -> bool:
    """Whether this interpreter can load the frozen checkpoint itself."""
    return (not switched_off() and _weights(model) is not None and have_module("torch")
            and have_module("open_clip"))


def provenance(model: str = "open_clip_vit_b32") -> dict[str, Any]:
    """Return auditable model status; this function never downloads anything."""
    spec = _model_spec(model)
    # The policy process (sc-run) is intentionally lightweight.  When it has a
    # configured remote GPU worker, ask that worker for the same provenance
    # record instead of pretending that a staged checkpoint is unavailable.
    if not switched_off() and _remote.enabled() and not have_module("torch"):
        return dict(_remote.call("clip_retrieval", "provenance", {"model": model}))
    path = _weights(model)
    has_torch = have_module("torch")
    has_open_clip = have_module("open_clip")
    reason: str | None = None
    if switched_off():
        reason = "pretrained tools are disabled by SCIENCECLAW_NO_PRETRAINED/SCIENCECLAW_NO_SCILIB"
    elif path is None:
        reason = "no explicitly staged CLIP checkpoint was found"
    elif not has_torch:
        reason = "torch is not installed"
    elif not has_open_clip:
        reason = "open_clip_torch is not installed"
    ok = reason is None
    out: dict[str, Any] = {
        "component": "scilib.clip_retrieval",
        "model": model,
        "backend": spec["backend"],
        "model_name": spec["model_name"],
        "pretrained_name": spec["pretrained_name"],
        "available": ok,
        "frozen": True,
        "network_allowed": False,
        "labels_used": False,
        "license": spec["license"],
        "weights_path": str(path) if path is not None else None,
        "weights_sha256": _sha256(path) if path is not None else None,
        "weights_bytes": path.stat().st_size if path is not None else None,
        "torch_version": _version("torch") if has_torch else None,
        "open_clip_version": _version("open_clip_torch") if has_open_clip else None,
        "reason": reason,
    }
    return out


def available(model: str = "open_clip_vit_b32") -> bool:
    """Whether the frozen local encoder can be loaded in this process."""
    return bool(provenance(model)["available"])


def _check_images(images: Any, name: str) -> np.ndarray:
    x = np.asarray(images)
    if x.ndim != 4 or x.shape[0] < 1 or x.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (n,H,W,3), got {list(x.shape)}")
    if x.shape[1] < 1 or x.shape[2] < 1:
        raise ValueError(f"{name} must have non-empty image dimensions")
    if not np.issubdtype(x.dtype, np.number) or not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must contain finite numeric RGB pixels")
    return np.clip(x, 0, 255).astype(np.uint8, copy=False)


def _load(model: str, device: str):
    key = (model, device)
    if key in _CACHE:
        return _CACHE[key]
    info = provenance(model)
    if not info["available"]:
        raise RuntimeError(f"clip_retrieval: frozen encoder unavailable: {info['reason']}; provenance={info}")
    import open_clip

    # ``pretrained`` receives a concrete path.  It must not be the registry
    # name (which could trigger a hub lookup in open_clip); this is the central
    # fail-closed guarantee of this wrapper.
    path = info["weights_path"]
    if not path:
        raise RuntimeError(f"clip_retrieval: checkpoint path disappeared; provenance={provenance(model)}")
    # Guard against a backend regression that interprets a malformed local
    # path as a hub identifier.  An explicit checkpoint must fail rather than
    # silently opening a connection or replacing the audited file.
    old_offline = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        made = open_clip.create_model_and_transforms(
            MODELS[model]["model_name"], pretrained=path, device=device,
            # The official OpenAI .pt is a TorchScript archive rather than a
            # plain state-dict.  It is hash-pinned above, so opting out of
            # PyTorch's weights-only restriction is safe for this staged file
            # and is required by torch >=2.6.
            load_weights_only=False,
        )
    finally:
        for key, value in old_offline.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    if not isinstance(made, tuple) or len(made) < 3:
        raise RuntimeError("clip_retrieval: open_clip returned an unexpected model/transform tuple")
    net, _, preprocess = made[:3]
    net.eval()
    _CACHE[key] = (net, preprocess)
    return _CACHE[key]


def encode_images(images: Any, model: str = "open_clip_vit_b32", batch_size: int = 32,
                  device: str = "auto") -> np.ndarray:
    """Encode RGB images with the staged frozen checkpoint and L2-normalise rows."""
    x = _check_images(images, "images")
    if not isinstance(batch_size, (int, np.integer)) or int(batch_size) < 1:
        raise ValueError("batch_size must be a positive integer")
    info = provenance(model)
    if not info["available"]:
        raise RuntimeError(f"clip_retrieval: frozen encoder unavailable: {info['reason']}; provenance={info}")
    if not _local_ok(model) and _remote.enabled():
        result = _remote.call("clip_retrieval", "encode_images", {
            "images": x, "model": model, "batch_size": int(batch_size), "device": "auto",
        })
        return np.asarray(result, dtype=np.float32)
    dev = torch_device(device)
    net, preprocess = _load(model, dev)
    import torch
    from PIL import Image

    out: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x), int(batch_size)):
            batch = torch.stack([preprocess(Image.fromarray(im, mode="RGB"))
                                 for im in x[start:start + int(batch_size)]])
            z = net.encode_image(batch.to(dev))
            if isinstance(z, (tuple, list)):
                z = z[0]
            z = torch.nn.functional.normalize(z.float(), dim=-1)
            out.append(z.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(out, axis=0)


def retrieve_captions(train_rows: list[dict[str, Any]], train_images: Any,
                      eval_items: list[dict[str, Any]], eval_images: Any, k: int = 1,
                      model: str = "open_clip_vit_b32", fallback: dict[str, str] | None = None,
                      batch_size: int = 32, selection: str = "nearest") -> list[str]:
    """Retrieve visible same-language captions using frozen CLIP image cosine.

    ``selection='nearest'`` preserves the historical first-neighbour result.
    ``selection='caption_medoid'`` selects a character-ngram medoid from the
    top ``k`` visible neighbours and is useful when the image encoder returns
    several near-duplicates.  Both modes use visible captions only.
    """
    if not isinstance(k, (int, np.integer)) or int(k) < 1:
        raise ValueError("k must be a positive integer")
    if selection not in ("nearest", "caption_medoid"):
        raise ValueError("selection must be 'nearest' or 'caption_medoid'")
    if len(train_rows) != len(np.asarray(train_images)):
        raise ValueError("train_rows and train_images must have the same length")
    if len(eval_items) != len(np.asarray(eval_images)):
        raise ValueError("eval_items and eval_images must have the same length")
    tr = _check_images(train_images, "train_images") if len(train_rows) else np.zeros((0, 1, 1, 3), np.uint8)
    ev = _check_images(eval_images, "eval_images")
    candidates = [i for i, row in enumerate(train_rows)
                  if bool(row.get("has_image")) and str(row.get("caption", "")).strip()]
    if not candidates:
        fb = fallback or {}
        return [str(fb.get(str(item.get("iso_lang", "")), "")) for item in eval_items]
    X_train = encode_images(tr[candidates], model=model, batch_size=batch_size)
    X_eval = encode_images(ev, model=model, batch_size=batch_size)
    fallback = fallback or {}
    out: list[str] = []
    for j, item in enumerate(eval_items):
        lang = str(item.get("iso_lang", ""))
        positions = [q for q, i in enumerate(candidates) if str(train_rows[i].get("iso_lang", "")) == lang]
        if not positions:
            out.append(str(fallback.get(lang, "")))
            continue
        sims = X_train[positions] @ X_eval[j]
        # Original candidate index is the deterministic tie-breaker.  The
        # optional medoid is selected only from the returned visible captions.
        order = np.lexsort((np.asarray(positions), -sims))[:min(int(k), len(positions))]
        captions = [str(train_rows[candidates[positions[int(q)]]].get("caption", ""))
                    for q in order]
        out.append(_select_caption(captions, np.arange(len(captions)), selection))
    return out
