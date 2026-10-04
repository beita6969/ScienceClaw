"""Create, on the GPU host, the array chunks the remote bridge would otherwise upload for the delivery's clips.

usage: python prestage_blobs.py <manifest.json> <clips dir> <blobs dir> <out map.json>
A clip row as the sandbox sends it is a float32 array of shape (1, n_samples) (int16 / 32768); the bridge stores it as
``<sha256[:32] of its npy bytes>.npy`` (scripts/remote/blobs.py). The map written is {wav sha256: chunk hash}.
"""
import hashlib
import io
import json
import os
import sys

import numpy as np
from scipy.io import wavfile

man, cdir, bdir, out = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3], sys.argv[4]
res, new = {}, 0
for m in man:
    raw = open(f"{cdir}/{m['id']}", "rb").read()
    assert hashlib.sha256(raw).hexdigest() == m["sha256"]
    sr, x = wavfile.read(io.BytesIO(raw))
    assert sr == 16000 and x.dtype == np.int16 and x.ndim == 1
    row = (x.astype(np.float32) / 32768.0)[None, :]
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(row), allow_pickle=False)
    b = buf.getvalue()
    h = hashlib.sha256(b).hexdigest()[:32]
    p = f"{bdir}/{h}.npy"
    if not os.path.exists(p):
        with open(p + ".tmp", "wb") as f:
            f.write(b)
        os.replace(p + ".tmp", p)
        new += 1
    res[m["sha256"]] = h
json.dump(res, open(out, "w"))
print("clips", len(man), "new blobs", new, "distinct", len(set(res.values())))
