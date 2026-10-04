"""Compare the chunk hashes the bridge computes for the adapter's own waveforms (every probe and support clip) with the server-side map.
usage: python check_blobs.py <map.json>"""
import json
import sys

import numpy as np

import common as C
from scienceclaw.bench.tasks import for40_dcase as D
from scripts.remote import blobs

mp = json.load(open(sys.argv[1]))
design = D.load_design(C.ROOT)
bad = n = 0
for c in design.clips():
    w = D.read_clip(c)
    ch = blobs.chunks(np.asarray(w, dtype=np.float32)[None, :])
    n += 1
    if len(ch) != 1 or mp.get(c.sha256) != ch[0][0]:
        bad += 1
print("clips", n, "mismatches", bad)
