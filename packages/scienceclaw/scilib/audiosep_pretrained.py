"""Pretrained neural stereo music source separation (Demucs / Hybrid Transformer Demucs, weights stored locally, no network).

Array layout as in ``scilib.audiosep``: mixtures ``(k, n, 2)`` float, linear PCM, ``sample_rate`` Hz (default 22050); estimates
``(k, 4, n, 2)`` with the targets in the order ``TARGETS = (vocals, drums, bass, other)``.

available(model="htdemucs") -> bool
    True when torch, the ``demucs`` package and the requested stored weights can be loaded in this interpreter, or when a
    remote GPU worker is attached (the call then runs on that worker and returns the same kind of result; a call with the
    same arguments returns the stored answer, and arrays the worker has not received before are transferred first at about
    1 MB per 5-10 s).
separate_pretrained(mixtures, model="htdemucs", sample_rate=22050, shifts=0, overlap=0.25, device="auto") -> float32 estimates
    One excerpt at a time: resample to the model's 44.1 kHz, normalise as the Demucs command line does (subtract the mean of
    the channel-mean signal, divide by its standard deviation, undo afterwards), run the network on the whole excerpt
    (``shifts`` random-shift averaging passes, ``overlap`` between internal segments), resample back to ``sample_rate`` and
    crop / zero-pad to the input length. The four estimates are on the amplitude scale of the mixture. ``device`` 'auto' uses
    a GPU when one is visible. Deterministic on CPU for ``shifts=0``.

Models: the default ``htdemucs`` is Rouard, Massa, Defossez, "Hybrid Transformers for Music Source Separation", ICASSP
2023 (MIT licence), trained on MUSDB18-HQ plus 800 further songs. The staged bundle may also expose the official
``htdemucs_ft`` fine-tuned bag and ``mdx_extra`` MDX bag from the Demucs release. These are frozen
checkpoints only; no task labels or fitting are used. Excerpts taken from MUSDB18 training tracks (which can include
the visible training and dev excerpts of a task) were seen in training and their scores are higher than for unseen songs.
The estimates can be combined with, or corrected by, the trained separators of ``scilib.audiosep``.
Cost: about 10-20 s for 16 six-second excerpts on a GPU (plus 1-3 s to load the weights in every process); on CPU several
seconds per excerpt.
"""
from __future__ import annotations

import math

import numpy as np

from . import _remote
from ._pretrained import have_module, model_path, set_cpu_threads, switched_off, torch_device

__all__ = ["SR", "TARGETS", "MODELS", "available", "available_models", "separate_pretrained"]

SR = 22_050
TARGETS = ("vocals", "drums", "bass", "other")
# Official frozen model/bag names shipped by Demucs.  Keeping this explicit
# prevents a task from silently selecting a model that is not staged.
MODELS = ("htdemucs", "htdemucs_ft", "mdx_extra")
_cache: dict = {}


def _weights_dir(model: str = "htdemucs"):
    d = model_path("demucs")
    return d if d is not None and (d / f"{model}.yaml").is_file() else None


def _local_ok(model: str = "htdemucs") -> bool:
    return not switched_off() and have_module("torch") and have_module("demucs") and _weights_dir(model) is not None


def available(model: str = "htdemucs") -> bool:
    """Whether the requested frozen Demucs model is staged locally or via a worker."""
    if not isinstance(model, str) or model not in MODELS:
        return False
    return _local_ok(model) or _remote.enabled()


def available_models() -> tuple[str, ...]:
    """Names of frozen Demucs bundles staged on this host (no downloads)."""
    if not available():
        return ()
    if _local_ok():
        d = _weights_dir()
        assert d is not None
        return tuple(name for name in MODELS if (d / f"{name}.yaml").is_file())
    # The worker advertises one immutable model root; the exact file check is
    # performed when it loads the requested bundle.
    return MODELS


def _load(name: str, device: str):
    key = (name, device)
    if key not in _cache:
        from demucs.pretrained import get_model
        m = _get_model_unpickled(get_model, name)
        m.to(device)
        m.eval()
        _cache[key] = m
    return _cache[key]


def _get_model_unpickled(get_model, name: str):
    """Load the stored checkpoint (a pickled package whose hash demucs checks) with ``weights_only=False`` for this call only:
    since torch 2.6 the default refuses the model class it contains."""
    try:
        import torch
    except ImportError:
        return get_model(name, repo=_weights_dir(name))
    orig = torch.load

    def load(*a, **k):
        k.setdefault("weights_only", False)
        return orig(*a, **k)

    torch.load = load
    try:
        return get_model(name, repo=_weights_dir(name))
    finally:
        torch.load = orig


def _run(model, wav: np.ndarray, shifts: int, overlap: float, device: str) -> np.ndarray:
    """(2, T) float32 array at the model's rate -> (S, 2, T) float32 estimates in ``model.sources`` order."""
    import torch
    from demucs.apply import apply_model
    with torch.no_grad():
        est = apply_model(model, torch.from_numpy(wav)[None], shifts=int(shifts), split=True, overlap=float(overlap),
                          device=device, progress=False)[0]
    return est.cpu().numpy()


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Polyphase resampling of an array (..., samples) from ``src`` to ``dst`` Hz."""
    if src == dst:
        return x
    from scipy.signal import resample_poly
    g = math.gcd(int(src), int(dst))
    return resample_poly(x, int(dst) // g, int(src) // g, axis=-1)


def _fit_length(x: np.ndarray, n: int) -> np.ndarray:
    if x.shape[-1] >= n:
        return x[..., :n]
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, n - x.shape[-1])])


def separate_pretrained(mixtures, model: str = "htdemucs", sample_rate: int = SR, shifts: int = 0, overlap: float = 0.25,
                        device: str = "auto") -> np.ndarray:
    x = np.asarray(mixtures, dtype=np.float32)
    if x.ndim != 3 or x.shape[2] != 2:
        raise ValueError(f"mixtures must have shape (k, n, 2), got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("mixtures must contain only finite values")
    try:
        rate = int(sample_rate)
    except (TypeError, ValueError):
        raise ValueError("sample_rate must be a positive integer") from None
    if rate != sample_rate or rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    try:
        n_shifts = int(shifts)
    except (TypeError, ValueError):
        raise ValueError("shifts must be a non-negative integer") from None
    if n_shifts != shifts or n_shifts < 0:
        raise ValueError("shifts must be a non-negative integer")
    try:
        ov = float(overlap)
    except (TypeError, ValueError):
        raise ValueError("overlap must be in [0, 1)") from None
    if not np.isfinite(ov) or ov < 0.0 or ov >= 1.0:
        raise ValueError("overlap must be in [0, 1)")
    if not isinstance(model, str) or model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    # Do not initialize a heavyweight model or require the remote bridge for an
    # empty batch.  This is useful to adapters that preserve an empty split and
    # keeps the no-op path deterministic in CPU-only environments.
    if x.shape[0] == 0:
        return np.zeros((0, 4, x.shape[1], 2), np.float32)
    # Keep the default call compatible with lightweight test doubles that
    # monkeypatch the legacy zero-argument readiness probe.
    local_ok = _local_ok() if model == "htdemucs" else _local_ok(model)
    if not local_ok:
        if not _remote.enabled():
            raise RuntimeError("separate_pretrained: torch, demucs or the stored htdemucs weights are not available here")
        est = _remote.call("audiosep_pretrained", "separate_pretrained",
                           {"mixtures": x, "model": model, "sample_rate": rate, "shifts": n_shifts,
                            "overlap": ov, "device": "auto"})
        return np.asarray(est, dtype=np.float32)
    dev = torch_device(device)
    if dev == "cpu":
        set_cpu_threads(4)
    net = _load(model, dev)
    native = int(net.samplerate)
    order = [list(net.sources).index(t) for t in TARGETS]
    k, n, _ = x.shape
    out = np.zeros((k, 4, n, 2), np.float32)
    for i in range(k):
        wav = _resample(x[i].T, rate, native).astype(np.float32)                 # (2, T)
        ref = wav.mean(0)
        mu, sd = float(ref.mean()), float(ref.std())
        if not np.isfinite(sd) or sd < 1e-8:
            continue                                                             # silent excerpt: zero estimates
        est = _run(net, ((wav - mu) / sd).astype(np.float32), n_shifts, ov, dev) * sd + mu  # (S, 2, T)
        est = _fit_length(_resample(est, native, rate), n)
        for j, s in enumerate(order):
            out[i, j] = est[s].T
    return out
