"""Clip embeddings of audio waveforms from two pretrained audio models: the Audio Spectrogram Transformer fine-tuned on AudioSet (AST,
``MIT/ast-finetuned-audioset-10-10-0.4593``) and the audio tower of CLAP (HTS-AT, ``laion/clap-htsat-unfused``). Computing them needs torch,
the weights and a GPU; when this interpreter cannot run them and a remote GPU worker is configured, the calls run on the worker's GPU host
and return the same values (answers are stored, so an identical call returns the same result). Neither model saw the labels of this
benchmark; their training corpora are described under "Training data" below.

available() -> bool
    True when torch, transformers and the AST weights are available here, or a remote GPU worker is configured.
embed(waveforms, lengths=None, sample_rate=16000.0, model="ast_audioset", kind="pooled") -> float32 array (n_clips, d)
    One vector per row of ``waveforms`` (n_clips, n_samples; float, linear amplitude in [-1, 1]; ``lengths`` = true number of samples of each
    row, default all of them; rows may be zero-padded). A clip is resampled to the model's rate (AST 16 kHz, CLAP 48 kHz; ``sample_rate`` must
    be a whole number of Hz) and cut into windows of at most 10.24 s (AST) / 10 s (CLAP): one window when the clip is shorter, otherwise the
    first and the last window. The vector is the mean over the windows of the model output named by ``kind``; it is not normalised.
    model "ast_audioset":  kind "pooled" (768; mean of the class and distillation tokens after the final layer norm),
                           kind "logits" (527 AudioSet sound-event logits).
    model "clap_htsat":    kind "pooled" (768; pooled output of the audio encoder),
                           kind "proj" (512; the audio embedding of the joint audio-text space).
MODELS                                                dict name -> (subdirectory of the model root, model rate in Hz, window in samples, kinds)

Training data: AST is a Vision-Transformer-style model initialised from an ImageNet-trained DeiT and fine-tuned on AudioSet (about two
million 10-second YouTube clips labelled with 527 sound-event classes). The CLAP audio tower is trained contrastively with a text tower on
LAION-Audio-630K (633,526 audio-text pairs from several public sources); the model card of this checkpoint lists no further datasets. Audio
of this benchmark may or may not be part of the public corpora behind their pre-training; that cannot be ruled out.

Cost on one shared H800-class GPU (AST in half precision, CLAP in single precision; measured on 728 clips of 10-12 s at 16 kHz, 32 clips
per call, weights already loaded): about 120-140 clips per second for AST and about 28 clips per second for CLAP. A remote call adds a
start-up of several seconds (connection, loading the weights) and its answer is stored.
"""
from __future__ import annotations

import math

import numpy as np

from . import _remote
from ._pretrained import have_module, model_path, switched_off

__all__ = ["available", "embed", "MODELS"]

MODELS = {"ast_audioset": ("audioenc/ast_audioset", 16000, int(10.24 * 16000), ("pooled", "logits")),
          "clap_htsat": ("audioenc/clap_htsat_unfused", 48000, 10 * 48000, ("pooled", "proj"))}
_CACHE: dict = {}


def _dir(model: str):
    return model_path(*MODELS[model][0].split("/")) if model in MODELS else None


def _local_ok(model: str) -> bool:
    d = _dir(model)
    return (not switched_off() and have_module("torch") and have_module("transformers") and d is not None
            and (any(d.glob("*.safetensors")) or any(d.glob("*.bin"))))


def available() -> bool:
    return _local_ok("ast_audioset") or _remote.enabled()


def _check(waveforms, lengths, sample_rate, model, kind):
    if model not in MODELS:
        raise ValueError(f"model must be one of {list(MODELS)}")
    if kind not in MODELS[model][3]:
        raise ValueError(f"kind must be one of {list(MODELS[model][3])} for model {model!r}")
    x = np.asarray(waveforms, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("waveforms must be a non-empty 2-D array (n_clips, n_samples)")
    if not np.all(np.isfinite(x)):
        raise ValueError("waveforms contain non-finite values")
    n = x.shape[0]
    if lengths is None:
        ln = np.full(n, x.shape[1], dtype=np.int64)
    else:
        ln = np.asarray(lengths).astype(np.int64).reshape(-1)
        if ln.shape[0] != n or ln.min() < 1 or ln.max() > x.shape[1]:
            raise ValueError("lengths must hold one value per row, each between 1 and n_samples")
    sr = float(sample_rate)
    if not (sr > 0 and sr == int(sr)):
        raise ValueError("sample_rate must be a positive whole number of Hz")
    return np.ascontiguousarray(x), ln, int(sr)


def _resample(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return x
    from scipy.signal import resample_poly
    g = math.gcd(sr, target)
    return resample_poly(x, target // g, sr // g).astype(np.float32)


def _windows(x: np.ndarray, win: int) -> list[np.ndarray]:
    return [x] if len(x) <= win else [x[:win], x[-win:]]


def _load(model: str):
    if model not in _CACHE:
        import torch
        d = str(_dir(model))
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if model == "ast_audioset":
            from transformers import ASTFeatureExtractor, ASTForAudioClassification
            fe = ASTFeatureExtractor.from_pretrained(d)
            net = ASTForAudioClassification.from_pretrained(d).eval()
            if dev == "cuda":
                net = net.half()
        else:
            from transformers import ClapFeatureExtractor, ClapModel
            fe = ClapFeatureExtractor.from_pretrained(d)
            net = ClapModel.from_pretrained(d).eval()
        _CACHE[model] = (fe, net.to(dev), dev)
    return _CACHE[model]


def _embed_local(x: np.ndarray, ln: np.ndarray, sr: int, model: str, kind: str) -> np.ndarray:
    import torch
    fe, net, dev = _load(model)
    _, rate, win, _ = MODELS[model]
    out = []
    with torch.inference_mode():
        for i in range(x.shape[0]):
            clip = _resample(x[i, :ln[i]], sr, rate)
            wins = _windows(clip, win)
            if model == "ast_audioset":
                inp = fe(wins, sampling_rate=rate, return_tensors="pt")["input_values"].to(dev)
                if dev == "cuda":
                    inp = inp.half()
                o = net(inp, output_hidden_states=True)
                if kind == "logits":
                    v = o.logits.float().mean(0)
                else:
                    last = net.audio_spectrogram_transformer.layernorm(o.hidden_states[-1]).float()
                    v = ((last[:, 0] + last[:, 1]) / 2).mean(0)
            else:
                inp = fe(wins, sampling_rate=rate, return_tensors="pt")["input_features"].to(dev)
                o = net.audio_model(input_features=inp)
                pooled = o.pooler_output
                v = (net.audio_projection(pooled) if kind == "proj" else pooled).float().mean(0)
            out.append(v.cpu().numpy())
    return np.stack(out).astype(np.float32)


def embed(waveforms, lengths=None, sample_rate: float = 16000.0, model: str = "ast_audioset", kind: str = "pooled") -> np.ndarray:
    x, ln, sr = _check(waveforms, lengths, sample_rate, model, kind)
    if not _local_ok(model) and _remote.enabled():
        res = _remote.call("audioenc", "embed", {"waveforms": x, "lengths": ln, "sample_rate": float(sr), "model": model, "kind": kind})
        return np.asarray(res, dtype=np.float32)
    if _local_ok(model):
        return _embed_local(x, ln, sr, model, kind)
    raise RuntimeError("embed: neither local weights nor a remote GPU worker are available")
