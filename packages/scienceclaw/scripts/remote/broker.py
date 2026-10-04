"""Serve the scilib remote spool: run each request on the GPU host over ssh and write the response next to it.

python scripts/remote/broker.py --spool cache/remote_spool --host gpu-host \
    --root /srv/scienceclaw --gpu 2

On a SLURM cluster the worker runs inside a running allocation (``--srun-job <jobid>``, ``--host <login-alias>``, ``--code-dir``, ``--python`` ...).
The ssh route is the user's own alias (key and jump host live in ~/.ssh/config); nothing but the gzip-JSON request goes over it.
One request runs per GPU at a time (shared GPU host): ``--gpu 2`` serves requests one by one, ``--gpu 2,3,4`` runs up to three requests
at once, each on its own GPU. Answers are kept in the spool, so an identical request is not run twice.
Large array arguments are sent as content-addressed row chunks (``blobs.py``): a chunk the host already holds is not sent again.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import gzip
import json
import os
import queue
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import blobs  # noqa: E402
from scilib import _remote  # noqa: E402

SSH = ["ssh", "-o", "ControlMaster=auto", "-o", "ControlPersist=10m", "-o", "ControlPath=/tmp/sc-ssh-%C",
       "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=20"]


def _ssh(a) -> list[str]:
    """ssh command prefix; ``--ssh-socket`` reuses an existing ControlMaster socket (no new handshake per request)."""
    sock = getattr(a, "ssh_socket", None)
    if sock and Path(sock).expanduser().exists():
        return ["ssh", "-S", str(Path(sock).expanduser()), "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
    return SSH


def known_blobs(a) -> set[str]:
    ls = f"ls {a.blobs or a.root + '/blobs'} 2>/dev/null"
    p = subprocess.run(["bash", "-c", ls] if getattr(a, "local", False) else _ssh(a) + [a.host, ls], capture_output=True, timeout=120)
    return {n[:-4] for n in p.stdout.decode().split() if n.endswith(".npy") and len(n) == 36}


def with_blobs(payload: dict, known: set[str]) -> tuple[bytes, set[str]]:
    """Gzipped request in which big arrays are references plus the chunks not in ``known``; also the chunk hashes it uses."""
    new: dict[str, bytes] = {}
    args = blobs.pack(payload["args"], known, new)
    used = {h for h in _refs(args)}
    out = dict(payload, args=args)
    if new:
        out["blobs"] = {h: base64.b64encode(r).decode() for h, r in new.items()}
    return gzip.compress(json.dumps(out).encode(), 6), used


def _refs(obj):
    if isinstance(obj, dict):
        if set(obj) == {"__rows__"}:
            yield from obj["__rows__"]
        else:
            for v in obj.values():
                yield from _refs(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _refs(v)


def run_remote(a, raw: bytes, known: set[str] | None = None) -> dict:
    """Run one request (``raw`` = the JSON request) on the GPU host; ``known`` = chunk hashes the host has (updated)."""
    known = set() if known is None else known
    payload = json.loads(raw)
    for _ in range(3):
        body, used = with_blobs(payload, known)
        out = _run_ssh(a, body)
        if out.get("ok"):
            known |= used
        elif out.get("missing"):
            known -= set(out["missing"])
            continue
        return out
    return {"ok": False, "transient": True, "error": "array chunks kept going missing on the GPU host"}


def _run_ssh(a, raw: bytes) -> dict:
    code = a.code_dir or f"{a.root}/code"
    py = a.python or f"{a.root}/env/bin/python"
    models = a.models or f"{a.root}/models"
    # The optional frozen CLIP dependency bundle lives beside the remote
    # models under the broker root.  Keeping it on the worker's import path
    # avoids installing a second Torch/CUDA stack into sc-run.
    extra = f":{a.root}/clipdeps" if a.root else ""
    remote = (f"cd {code} && export PYTHONNOUSERSITE=1 PYTHONPATH={code}{extra} CUDA_VISIBLE_DEVICES={a.gpu} SCIENCECLAW_MODELS={models} "
              f"HF_HOME={a.hf_home or models + '/hf'} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True "
              f"SCIENCECLAW_MLIP_CACHE={a.mlip_cache or a.root + '/cache/mlip'} "
              f"SCIENCECLAW_REMOTE_OUT={a.out_dir or a.root + '/remote_models'} SCIENCECLAW_REMOTE_BLOBS={a.blobs or a.root + '/blobs'} "
              f"&& exec nice -n 5 {py} scripts/remote/worker.py")
    if a.srun_job:
        remote = (f"srun --jobid={a.srun_job} --overlap -N1 -n1 -c{a.cpus} --gres=gpu:a100:4 --export=ALL "
                  f"bash -c {shlex.quote(remote)}")
    cmd = ["bash", "-c", remote] if a.local else _ssh(a) + [a.host, remote]
    last = "no attempt"
    for attempt in range(a.retries):
        try:
            p = subprocess.run(cmd, input=raw, capture_output=True, timeout=a.timeout)
        except subprocess.TimeoutExpired:
            last = f"timed out after {a.timeout:.0f} s"
            break
        if p.returncode == 0 and p.stdout:
            try:
                return json.loads(gzip.decompress(p.stdout))
            except Exception as e:  # noqa: BLE001
                last = f"unreadable response ({e})"
        else:
            last = f"ssh exit {p.returncode}: {p.stderr.decode(errors='replace')[-300:]}"
        time.sleep(10 * (attempt + 1))
    return {"ok": False, "transient": True, "error": last}


def serve(a) -> None:
    spool = Path(a.spool)
    spool.mkdir(parents=True, exist_ok=True)
    gpus = [g.strip() for g in str(a.gpu).split(",") if g.strip()]
    slots: "queue.Queue[str]" = queue.Queue()
    for g in gpus:
        slots.put(g)
    pool = cf.ThreadPoolExecutor(len(gpus))
    inflight: set[str] = set()
    try:
        known = known_blobs(a)
    except Exception as e:  # noqa: BLE001
        print(f"could not list the chunks on the host ({e}); sending arrays in full when unsure", flush=True)
        known = set()
    print(f"{len(known)} array chunks already on the host", flush=True)

    def work(req: Path, key: str) -> None:
        t0 = time.time()
        try:
            raw = gzip.decompress(req.read_bytes())
            try:
                known.update(known_blobs(a))                # chunks staged since the last request (scripts/remote/prestage.py)
            except Exception:  # noqa: BLE001
                pass
            gpu = slots.get()
            try:
                out = run_remote(argparse.Namespace(**{**vars(a), "gpu": gpu}), raw, known)
            finally:
                slots.put(gpu)
        except Exception as e:  # noqa: BLE001
            out = {"ok": False, "transient": True, "error": f"broker: {e}"}
        _remote._write_atomic(spool / f"resp-{key}.json.gz", gzip.compress(json.dumps(out).encode(), 6))
        req.unlink(missing_ok=True)
        inflight.discard(key)
        print(f"{time.strftime('%H:%M:%S')} {key} ok={out.get('ok')} {time.time() - t0:.0f}s {out.get('error', '')[:200]}", flush=True)

    print(f"serving {spool} -> {a.host} gpu(s) {','.join(gpus)}", flush=True)
    while True:
        for req in sorted(spool.glob("req-*.json.gz")):
            key = req.name[4:-8]
            if key in inflight:
                continue
            if (spool / f"resp-{key}.json.gz").exists():
                req.unlink(missing_ok=True)
                continue
            inflight.add(key)
            pool.submit(work, req, key)
        time.sleep(0.5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spool", default=str(REPO / "cache" / "remote_spool"))
    ap.add_argument("--host", default=os.environ.get("SCIENCECLAW_SSH_HOST"), help="ssh alias of the GPU host (default: $SCIENCECLAW_SSH_HOST)")
    ap.add_argument("--root", default=os.environ.get("SCIENCECLAW_REMOTE_ROOT"), help="working directory on the GPU host (default: $SCIENCECLAW_REMOTE_ROOT)")
    ap.add_argument("--gpu", default="2", help="one GPU index, or a comma list: one request per GPU runs concurrently")
    ap.add_argument("--timeout", type=float, default=1500.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--srun-job", default=None, help="run the worker inside this running SLURM allocation (srun --overlap)")
    ap.add_argument("--cpus", type=int, default=8)
    ap.add_argument("--local", action="store_true", help="run the worker with a local shell (the broker itself runs on the GPU host / inside the allocation)")
    ap.add_argument("--ssh-socket", default=None, help="existing ssh ControlMaster socket to reuse")
    for name in ("code-dir", "python", "models", "hf-home", "mlip-cache", "out-dir", "blobs"):
        ap.add_argument("--" + name, default=None)
    args = ap.parse_args()
    if not args.root:
        ap.error("--root or SCIENCECLAW_REMOTE_ROOT is required")
    if not args.local and not args.host:
        ap.error("--host or SCIENCECLAW_SSH_HOST is required unless --local is given")
    serve(args)
