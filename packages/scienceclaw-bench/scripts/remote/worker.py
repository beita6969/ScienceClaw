"""GPU-host side of the remote scilib bridge: read one gzip-JSON request on stdin, run it, write one gzip-JSON response on stdout.

Only whitelisted (module, function) pairs run. ``model=`` arguments of the
Stanza parser, the SAM selector and the hippocampus U-Net must point inside
the worker output directory
(``fit_sam_selector`` / ``fit_unet`` store the fitted model there and return its path). Large array arguments arrive as chunk references
into ``$SCIENCECLAW_REMOTE_BLOBS`` (see ``blobs.py``); chunks that came with the request are stored there first.
"""
from __future__ import annotations

import gzip
import importlib
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
_EXTRA = Path(__file__).resolve().parents[3] / "sv_pkgs"      # packages installed beside the environment (SevenNet); searched last
if _EXTRA.is_dir():
    sys.path.append(str(_EXTRA))

import blobs  # noqa: E402
from scilib import _remote  # noqa: E402

ALLOWED = {("udparse_pretrained", "parse_gold_tokens"), ("matphonon_mlip", "phonon_features"),
           ("audiosep_pretrained", "separate_pretrained"), ("scnet_pretrained", "separate_pretrained"), ("phenoseg_sam", "fit_sam_selector"), ("phenoseg_sam", "predict_sam_panoptic"),
           ("hippo_unet", "fit_unet"), ("hippo_unet", "predict_unet"), ("hippo_unet", "predict_proba"), ("hippo_unet", "fit_predict"), ("proteinplm", "site_logprobs"),
           ("textenc", "embed"), ("textenc", "relevance"), ("textenc", "nli"), ("audioenc", "embed"), ("tsfm", "forecast"),
           ("phenoseg_m2f", "predict_panoptic"), ("phenoseg_hapt", "predict_panoptic")}
ALLOWED |= {("clip_retrieval", "provenance"), ("clip_retrieval", "encode_images")}
OUT_ROOT = Path(os.environ.get("SCIENCECLAW_REMOTE_OUT", "remote_models")).resolve()
BLOB_ROOT = Path(os.environ.get("SCIENCECLAW_REMOTE_BLOBS", "remote_blobs")).resolve()


def main() -> None:
    real_out = sys.stdout.buffer
    sys.stdout = sys.stderr
    raw = gzip.decompress(sys.stdin.buffer.read())
    payload = json.loads(raw)
    try:
        module, fn = payload["module"], payload["fn"]
        if (module, fn) not in ALLOWED:
            raise PermissionError(f"{module}.{fn} is not served remotely")
        if payload.get("blobs"):
            blobs.store(BLOB_ROOT, payload["blobs"])
        args = _remote.decode(blobs.unpack(payload["args"], BLOB_ROOT))
        key = _remote.request_key(module, fn, args)[0]
        if module == "udparse_pretrained" and args.get("model"):
            m = Path(args["model"]).resolve()
            if OUT_ROOT not in m.parents or not m.is_file():
                raise PermissionError("model must be a pre-existing file under the worker output directory")
            args["model"] = str(m)
        mod = importlib.import_module(f"scilib.{module}")
        if fn in ("predict_sam_panoptic", "predict_unet", "predict_proba"):
            m = Path(str(args.get("model"))).resolve()
            if OUT_ROOT not in m.parents or not m.is_file():
                raise PermissionError("model must be a path returned by " + ("fit_sam_selector" if module == "phenoseg_sam" else "fit_unet"))
            args["model"] = mod.load_selector(str(m)) if module == "phenoseg_sam" else mod.load_model(str(m))
        res = getattr(mod, fn)(**args)
        if fn in ("fit_sam_selector", "fit_unet"):
            (OUT_ROOT / key).mkdir(parents=True, exist_ok=True)
            path = OUT_ROOT / key / ("selector.pkl" if fn == "fit_sam_selector" else "unet.pt")
            if fn == "fit_sam_selector":
                mod.save_selector(res, str(path))
            else:
                mod.save_model(res, str(path))
            res = {"path": str(path), "params": res.params, "n_train": res.n_train, "fit_s": res.fit_s}
        out = {"ok": True, "result": _remote.encode(res)}
    except blobs.MissingBlobs as e:
        out = {"ok": False, "transient": True, "missing": e.missing, "error": str(e)}
    except Exception as e:  # noqa: BLE001
        out = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
    real_out.write(gzip.compress(json.dumps(out).encode(), 6))
    real_out.flush()


if __name__ == "__main__":
    main()
