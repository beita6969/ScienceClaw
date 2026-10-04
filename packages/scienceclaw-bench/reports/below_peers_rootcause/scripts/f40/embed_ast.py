"""AST (AudioSet fine-tuned) clip embeddings of the delivery's clips, run on the GPU host.

usage: python embed_ast.py <manifest.json> <clips dir> <ast model dir> <out.npz>
Per clip (16 kHz mono): 10.24 s windows (start and end window when the clip is longer), hidden states of layers 4, 8 and 12 averaged over
patch tokens, the pooled output (mean of the CLS and distillation tokens) and the 527 AudioSet logits, averaged over the windows.
"""
import json
import sys
import time

import numpy as np
import torch
from scipy.io import wavfile
from transformers import ASTFeatureExtractor, ASTForAudioClassification

man, cdir, mdir, out = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3], sys.argv[4]
fe = ASTFeatureExtractor.from_pretrained(mdir)
net = ASTForAudioClassification.from_pretrained(mdir).eval().cuda().half()
WIN = int(10.24 * 16000)
LAYERS = (4, 8, 12)
res = {"layer%d" % l: [] for l in LAYERS}
res["pooled"], res["logits"] = [], []
t0 = time.time()
with torch.inference_mode():
    for i, m in enumerate(man):
        sr, x = wavfile.read(f"{cdir}/{m['id']}")
        assert sr == 16000 and x.ndim == 1
        x = x.astype(np.float32) / 32768.0
        wins = [x] if len(x) <= WIN else [x[:WIN], x[-WIN:]]
        inp = fe(wins, sampling_rate=16000, return_tensors="pt")["input_values"].cuda().half()
        o = net(inp, output_hidden_states=True)
        hs = o.hidden_states
        for l in LAYERS:
            res["layer%d" % l].append(hs[l][:, 2:].float().mean(1).mean(0).cpu().numpy())
        last = net.audio_spectrogram_transformer.layernorm(hs[-1]).float()
        res["pooled"].append(((last[:, 0] + last[:, 1]) / 2).mean(0).cpu().numpy())
        res["logits"].append(o.logits.float().mean(0).cpu().numpy())
        if i % 100 == 0:
            print(i, round(time.time() - t0), "s", flush=True)
np.savez(out, ids=np.array([m["public_id"] for m in man]), **{k: np.stack(v).astype(np.float32) for k, v in res.items()})
print("saved", out, round(time.time() - t0), "s", flush=True)
