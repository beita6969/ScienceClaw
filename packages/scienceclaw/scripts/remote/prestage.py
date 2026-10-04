"""Send the large input arrays of an episode to the GPU host ahead of the run, so the first remote call does not wait for them.

python scripts/remote/prestage.py --config configs/probe_strong.yaml --disciplines FoR36 --split id --n 1 \
    --keys mixtures,dev_mixtures --host gpu-host --root /srv/scienceclaw

The arrays (or lists of arrays, each element on its own) are the outputs of the episode's own no-input tools (what a code node would
receive); they are stored on the host as content-addressed row chunks (``blobs.py``). Only latency changes: the request that later references a chunk is the same request.
"""
from __future__ import annotations

import argparse
import subprocess
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import blobs  # noqa: E402
from broker import SSH, known_blobs  # noqa: E402
from scienceclaw.bench.splits import SplitPlan, load_adapters  # noqa: E402
from scienceclaw.config import load_config  # noqa: E402


def episode_arrays(ep, keys: set[str]) -> dict[str, np.ndarray]:
    out = {}
    for spec in ep.tools:
        if spec.inputs:
            continue
        res = spec.fn({}, {})
        for k, v in res.items():
            if k not in keys:
                continue
            for j, a in enumerate(v if isinstance(v, (list, tuple)) else [v]):
                # arrays whose base64 text is shorter than blobs.MIN_B64 travel inline in the request; nothing to stage
                if isinstance(a, np.ndarray) and a.ndim >= 2 and len(blobs.npy_bytes(a)) * 4 // 3 >= blobs.MIN_B64:
                    out[f"{spec.name}.{k}" + (f"[{j}]" if isinstance(v, (list, tuple)) else "")] = a
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--disciplines", required=True)
    ap.add_argument("--split", default="id")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--keys", default="mixtures,dev_mixtures")
    ap.add_argument("--host", default=os.environ.get("SCIENCECLAW_SSH_HOST"), help="ssh alias of the GPU host (default: $SCIENCECLAW_SSH_HOST)")
    ap.add_argument("--root", default=os.environ.get("SCIENCECLAW_REMOTE_ROOT"), help="working directory on the GPU host (default: $SCIENCECLAW_REMOTE_ROOT)")
    a = ap.parse_args()
    if not a.host or not a.root:
        ap.error("--host/--root (or SCIENCECLAW_SSH_HOST/SCIENCECLAW_REMOTE_ROOT) are required")

    cfg = load_config(a.config)
    cfg.bench.disciplines = a.disciplines.split(",")
    cfg.bench.rounds = max(cfg.bench.rounds, a.skip + a.n)
    adapters = load_adapters(cfg.bench)
    plan = SplitPlan.build(cfg.bench, adapters)
    known = known_blobs(a)
    print(f"{len(known)} chunks on the host", flush=True)
    for code in plan.order:
        for ep in plan.episodes[a.split].get(code, [])[a.skip:a.skip + a.n]:
            for name, arr in episode_arrays(ep, set(a.keys.split(","))).items():
                cs = blobs.chunks(arr)
                new = [(h, r) for h, r in cs if h not in known]
                print(f"{ep.id} {name} {arr.shape} {arr.dtype}: {len(cs)} chunks, {len(new)} to send "
                      f"({sum(len(r) for _, r in new) / 1e6:.1f} MB)", flush=True)
                if not new:
                    continue
                with tempfile.TemporaryDirectory() as td:
                    for h, r in new:
                        (Path(td) / f".{h}.part").write_bytes(r)
                    tar = subprocess.Popen(["tar", "czf", "-", "-C", td, "."], stdout=subprocess.PIPE)
                    # each chunk is written as .<hash>.part, then renamed to <hash>.npy once the whole tar has arrived
                    remote = (f"mkdir -p {a.root}/blobs && cd {a.root}/blobs && tar xzf - && "
                              "for f in .*.part; do [ -e \"$f\" ] && h=${f#.} && mv \"$f\" \"${h%.part}.npy\"; done; true")
                    p = subprocess.run(SSH + [a.host, remote], stdin=tar.stdout, capture_output=True)
                    tar.stdout.close()
                    tar.wait()
                    if p.returncode != 0:
                        raise SystemExit(f"upload failed: {p.stderr.decode()[-300:]}")
                known |= {h for h, _ in new}
    print("done", flush=True)


if __name__ == "__main__":
    main()
