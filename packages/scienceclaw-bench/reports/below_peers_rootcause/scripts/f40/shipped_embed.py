"""Embeddings of the delivery's clips with the shipped ``scilib.audioenc`` (run on the GPU host, local weights), with timing.

usage: python shipped_embed.py <manifest.json> <clips dir> <out.npz> [max_clips]
Per clip: int16 wav -> float32 / 32768 -> audioenc.embed (AST pooled + logits, CLAP pooled + proj), one call per 32 clips.
"""
import json
import sys
import time

import numpy as np
from scipy.io import wavfile

from scilib import audioenc

man, cdir, out = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
if len(sys.argv) > 4:
    man = man[:int(sys.argv[4])]
assert audioenc.available()
clips = []
for m in man:
    sr, x = wavfile.read(f"{cdir}/{m['id']}")
    assert sr == 16000 and x.ndim == 1
    clips.append(x.astype(np.float32) / 32768.0)
res = {}
t_all = {}
for model, kind in (("ast_audioset", "pooled"), ("ast_audioset", "logits"), ("clap_htsat", "pooled"), ("clap_htsat", "proj")):
    vs = []
    audioenc.embed(np.stack([clips[0]]), model=model, kind=kind)   # load weights outside the timing
    t0 = time.time()
    for i in range(0, len(clips), 32):
        ch = clips[i:i + 32]
        n = max(len(c) for c in ch)
        w = np.zeros((len(ch), n), dtype=np.float32)
        for r, c in enumerate(ch):
            w[r, :len(c)] = c
        vs.append(audioenc.embed(w, lengths=[len(c) for c in ch], sample_rate=16000.0, model=model, kind=kind))
    dt = time.time() - t0
    res[f"{model}:{kind}"] = np.concatenate(vs)
    t_all[f"{model}:{kind}"] = dt
    print(model, kind, res[f"{model}:{kind}"].shape, f"{dt:.1f} s", f"{len(clips) / dt:.1f} clips/s", flush=True)
np.savez(out, ids=np.array([m["public_id"] for m in man]), **{k.replace(":", "__"): v for k, v in res.items()})
print("saved", out, flush=True)
