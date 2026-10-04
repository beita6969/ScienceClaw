"""Fetch the manifest's zip members from the public Zenodo archives by HTTP range requests and verify each against its recorded sha256/crc32.

usage: python fetch_clips.py <manifest.json> <out dir> [threads]      (files land at <out dir>/<member name>)
"""
import hashlib
import json
import struct
import sys
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

man, out, nth = json.load(open(sys.argv[1])), Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 6


def get(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    for k in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:
            err = e
            time.sleep(2 + 3 * k)
    raise err


def one(m):
    dst = out / m["id"]
    if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest() == m["sha256"]:
        return "have"
    raw = get(m["url"], m["offset"], m["offset"] + 30 + 400 + m["csize"])
    assert raw[:4] == b"PK\x03\x04", m["id"]
    n, e = struct.unpack("<HH", raw[26:30])
    body = raw[30 + n + e:30 + n + e + m["csize"]]
    assert len(body) == m["csize"], (m["id"], len(body), m["csize"])
    data = zlib.decompress(body, -15) if m["method"] == 8 else body
    assert len(data) == m["usize"] and "%08x" % (zlib.crc32(data) & 0xFFFFFFFF) == m["crc32"], m["id"]
    assert hashlib.sha256(data).hexdigest() == m["sha256"], m["id"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    return "new"


t0 = time.time()
res = {"have": 0, "new": 0}
with ThreadPoolExecutor(nth) as ex:
    for i, r in enumerate(ex.map(one, man)):
        res[r] += 1
        if i % 50 == 0:
            print(i, res, round(time.time() - t0), "s", flush=True)
print("DONE", res, round(time.time() - t0), "s", flush=True)
