"""CLAP (HTSAT, unfused) clip embeddings, run on the GPU host.
usage: python embed_clap.py <manifest.json> <clips dir> <clap model dir> <out.npz>
Per clip: audio resampled 16 -> 48 kHz, windows of at most 10 s (start and end window when the clip is longer, so nothing is randomly cropped);
``proj`` = the 512-d projected audio embedding, ``pooled`` = the 768-d pooled output of the audio encoder; each averaged over the windows."""
import json
import sys
import time

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from transformers import ClapFeatureExtractor, ClapModel

man, cdir, mdir, out = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3], sys.argv[4]
fe = ClapFeatureExtractor.from_pretrained(mdir)
net = ClapModel.from_pretrained(mdir).eval().cuda()
WIN = 10 * 48000
res = {"proj": [], "pooled": []}
t0 = time.time()
with torch.inference_mode():
    for i, m in enumerate(man):
        sr, x = wavfile.read(f"{cdir}/{m['id']}")
        assert sr == 16000
        x = resample_poly(x.astype(np.float32) / 32768.0, 3, 1).astype(np.float32)
        wins = [x] if len(x) <= WIN else [x[:WIN], x[-WIN:]]
        inp = fe(wins, sampling_rate=48000, return_tensors="pt")["input_features"].cuda()
        o = net.audio_model(input_features=inp)
        res["pooled"].append(o.pooler_output.float().mean(0).cpu().numpy())
        res["proj"].append(net.audio_projection(o.pooler_output).float().mean(0).cpu().numpy())
        if i % 100 == 0:
            print(i, round(time.time() - t0), "s", flush=True)
np.savez(out, ids=np.array([m["public_id"] for m in man]), **{k: np.stack(v).astype(np.float32) for k, v in res.items()})
print("saved", out, round(time.time() - t0), "s", flush=True)
