"""Quantile forecasts of univariate series from two pretrained time-series models: Chronos-2 (``amazon/chronos-2``, 120 M parameters)
and Chronos-Bolt-base (``amazon/chronos-bolt-base``, 205 M parameters). Running them needs torch, the ``chronos`` package, the weights and
a GPU; when this interpreter cannot run them and a remote GPU worker is configured, the calls run on the worker's GPU host and
return the same values (answers are stored, so an identical call returns the same result). Neither model is trained or adapted on the
series it is given; the series only serve as context.

available() -> bool
    True when torch, the chronos package and the Chronos-2 weights are available here, or a remote GPU worker is configured.
forecast(histories, horizon, quantiles=(0.1, 0.5, 0.9), model="chronos_2", context_length=None, lengths=None) -> float32 array (n, horizon, len(quantiles))
    ``histories``: a list of 1-D float sequences (oldest value first; lengths may differ; NaN = missing observation, at least 3 observed
    values per series); or, when ``lengths`` is given, a 2-D array whose row i holds series i in its last ``lengths[i]`` entries (the
    entries before them are padding and are ignored). Returns, for every series, the predicted quantile of every level in ``quantiles``
    (each strictly between 0 and 1, increasing) for the ``horizon`` steps that follow the end of its history. The series are forecast independently
    (no information is shared between the rows of one call). ``context_length`` = number of most recent values the model sees (None =
    all values up to the model's own limit: Chronos-2 8192, Chronos-Bolt 2048). ``horizon`` is at most 1024 for Chronos-2 and at most 64
    for Chronos-Bolt.
    model "chronos_2":       21 learned quantile levels 0.01, 0.05, 0.1 ... 0.9, 0.95, 0.99 (levels in between are interpolated).
    model "chronos_bolt":    9 learned quantile levels 0.1 ... 0.9 (levels outside this range are clipped to the nearest learned level).
    The model does not know the calendar: a seasonal period is not an input and has to be inferred from the series itself. Values are
    returned in the units of the input.
MODELS                                                  dict name -> (subdirectory of the model root, max horizon, max context)

Training data: Chronos-2 is trained on a large collection of public and synthetic time series (its model card lists the Chronos
Datasets and GIFT-Eval pretraining corpora, plus synthetic series); Chronos-Bolt is trained on a mixture of nearly 100 billion
observations drawn from public Chronos Datasets and synthetic series. These corpora contain well-known public forecasting data sets (for
example the Monash archive, electricity and traffic loads, weather and energy series). Whether the series of this benchmark, or of its
data sources, including the values it holds out as targets, appear among them is not known and cannot be ruled out.

Cost (one shared H800-class GPU, weights loaded, float32; measured): Chronos-2 forecasts 256 series of 120 values for 24 steps in
0.07 s, 256 series of 300 values in 0.13 s, 64 series of 1461 values in 0.05 s; Chronos-Bolt takes 0.07 s, 0.08 s and 0.05 s for the same
three calls. A remote call adds a start-up of several seconds (connection, about 11 s for loading the Chronos-2 weights on the first call
of a worker) and its answer is stored.
"""
from __future__ import annotations

import numpy as np

from . import _remote
from ._pretrained import have_module, model_path, switched_off

__all__ = ["available", "forecast", "MODELS"]

MODELS = {"chronos_2": ("tsfm/chronos_2", 1024, 8192), "chronos_bolt": ("tsfm/chronos_bolt_base", 64, 2048)}
_CACHE: dict = {}


def _dir(model: str):
    return model_path(*MODELS[model][0].split("/")) if model in MODELS else None


def _local_ok(model: str = "chronos_2") -> bool:
    d = _dir(model)
    return (not switched_off() and have_module("torch") and have_module("chronos") and d is not None
            and any(d.glob("*.safetensors")))


def available() -> bool:
    return _local_ok("chronos_2") or _remote.enabled()


def _check(histories, horizon, quantiles, model, context_length, lengths):
    if model not in MODELS:
        raise ValueError(f"model must be one of {list(MODELS)}")
    h = int(horizon)
    if h < 1 or h > MODELS[model][1]:
        raise ValueError(f"horizon must be between 1 and {MODELS[model][1]} for model {model!r}")
    q = [float(v) for v in quantiles]
    if not q or not all(0.0 < v < 1.0 for v in q) or any(b <= a for a, b in zip(q, q[1:])):
        raise ValueError("quantiles must be strictly increasing levels strictly between 0 and 1")
    if context_length is not None and int(context_length) < 3:
        raise ValueError("context_length must be at least 3")
    if len(histories) < 1:
        raise ValueError("histories must hold at least one series")
    if lengths is not None:
        x = np.asarray(histories, dtype=np.float64)
        ln = np.asarray(lengths).astype(np.int64).reshape(-1)
        if x.ndim != 2 or ln.shape[0] != x.shape[0] or ln.min() < 1 or ln.max() > x.shape[1]:
            raise ValueError("with lengths, histories must be a 2-D array and lengths hold one value per row, each between 1 and its width")
        histories = [x[i, x.shape[1] - int(ln[i]):] for i in range(x.shape[0])]
    rows = []
    for k, s in enumerate(histories):
        a = np.asarray(s, dtype=np.float64).reshape(-1)
        if a.size < 3 or int(np.isfinite(a).sum()) < 3:
            raise ValueError(f"series {k} needs at least 3 observed values")
        if np.isinf(a).any():
            raise ValueError(f"series {k} contains infinite values")
        rows.append(a)
    return rows, h, tuple(q), None if context_length is None else int(context_length)


def _pad(rows):
    width = max(r.size for r in rows)
    out = np.full((len(rows), width), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        out[i, width - r.size:] = r
    return out, np.asarray([r.size for r in rows], dtype=np.int64)


def _load(model: str):
    if model not in _CACHE:
        import torch
        from chronos import BaseChronosPipeline
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _CACHE[model] = BaseChronosPipeline.from_pretrained(str(_dir(model)), device_map=dev, dtype=torch.float32)
    return _CACHE[model]


def _forecast_local(rows, horizon, quantiles, model, context_length) -> np.ndarray:
    import torch
    pipe = _load(model)
    cap = MODELS[model][2]
    ctx = cap if context_length is None else min(int(context_length), cap)
    inputs = [r[-ctx:].astype(np.float32) for r in rows]
    out = np.empty((len(rows), horizon, len(quantiles)), dtype=np.float32)
    step = 64
    with torch.inference_mode():
        for i in range(0, len(inputs), step):
            part = inputs[i:i + step]
            if model == "chronos_2":
                qs, _ = pipe.predict_quantiles([x[None, :] for x in part], prediction_length=horizon, quantile_levels=list(quantiles))
                q = np.stack([t[0].float().cpu().numpy() for t in qs])
            else:
                lv = [min(max(v, 0.1), 0.9) for v in quantiles]
                qt, _ = pipe.predict_quantiles([torch.tensor(x) for x in part], prediction_length=horizon, quantile_levels=lv)
                q = qt.float().cpu().numpy()
            out[i:i + len(part)] = q
    return out


def forecast(histories, horizon: int, quantiles=(0.1, 0.5, 0.9), model: str = "chronos_2", context_length=None, lengths=None) -> np.ndarray:
    rows, h, q, ctx = _check(histories, horizon, quantiles, model, context_length, lengths)
    if not _local_ok(model) and _remote.enabled():
        padded, lens = _pad(rows)
        res = _remote.call("tsfm", "forecast", {"histories": padded, "lengths": lens, "horizon": h, "quantiles": list(q), "model": model,
                                                "context_length": ctx})
        return np.asarray(res, dtype=np.float32)
    if _local_ok(model):
        return _forecast_local(rows, h, q, model, ctx)
    raise RuntimeError("forecast: neither local weights nor a remote GPU worker are available")
