#!/usr/bin/env python3
"""Read-only P3 checkpoint smoke tests; never loads ScienceClaw task data."""
from __future__ import annotations
import argparse, importlib.util, json, os, pathlib, sys, time

def fail(msg: str) -> None:
    print(json.dumps({"status":"blocked","reason":msg}, sort_keys=True))
    raise SystemExit(2)

def beats(path: pathlib.Path) -> dict:
    import torch
    source = os.environ.get("BEATS_SOURCE_ROOT", "")
    if not source:
        fail("BEATS_SOURCE_ROOT is unset; refusing to guess an upstream implementation")
    modpath = pathlib.Path(source) / "BEATs.py"
    if not modpath.is_file():
        fail(f"missing upstream BEATs.py at {modpath}")
    # The pinned upstream file uses sibling imports (backbone/modules).
    sys.path.insert(0, str(modpath.parent))
    spec = importlib.util.spec_from_file_location("p3_upstream_beats", modpath)
    if spec is None or spec.loader is None:
        fail("cannot import upstream BEATs.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    if not all(hasattr(mod, x) for x in ("BEATs", "BEATsConfig")):
        fail("upstream BEATs.py lacks BEATs/BEATsConfig")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "cfg" not in ckpt or "model" not in ckpt:
        fail("checkpoint does not have cfg/model structure")
    model = mod.BEATs(mod.BEATsConfig(ckpt["cfg"]))
    model.load_state_dict(ckpt["model"], strict=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    wav = torch.randn(1, 16000, device=device)
    with torch.inference_mode():
        out = model.extract_features(wav)[0]
    shape = list(out.shape) if hasattr(out, "shape") else None
    if not shape or not all(int(x) > 0 for x in shape):
        fail(f"BEATs returned invalid shape {shape}")
    return {"status":"ok", "mode":"beats", "device":device, "output_shape":shape,
            "checkpoint_bytes":path.stat().st_size}

def buildings(path: pathlib.Path) -> dict:
    # Use only the pinned upstream class supplied by the caller.  The random input
    # exercises the checkpoint contract and never touches ScienceClaw task data.
    import torch
    try:
        import tomllib
    except ModuleNotFoundError:
        fail("Python 3.11 tomllib is required for the official config")
    source = os.environ.get("BUILDINGS_SOURCE_ROOT", "")
    if not source:
        fail("BUILDINGS_SOURCE_ROOT is unset; official model/preprocessing code is required")
    # BuildingsBench 1.1.0 still imports the Python 3.10 name ``tomli``.
    # Python 3.11's stdlib parser is API-compatible for this read-only smoke.
    if "tomli" not in sys.modules:
        sys.modules["tomli"] = tomllib
    root = pathlib.Path(source)
    package = root / "buildings_bench"
    if not package.is_dir():
        fail(f"missing BuildingsBench source at {package}")
    sys.path.insert(0, str(root))
    try:
        from buildings_bench.models.transformers import LoadForecastingTransformer
    except Exception as exc:
        fail(f"cannot import pinned BuildingsBench Transformer: {type(exc).__name__}: {exc}")
    cfg_path = package / "configs" / "TransformerWithGaussian-L.toml"
    if not cfg_path.is_file():
        fail(f"missing official config {cfg_path}")
    cfg = tomllib.loads(cfg_path.read_text())
    model_cfg = dict(cfg["model"])
    model = LoadForecastingTransformer(**model_cfg)
    model.load_from_checkpoint(str(path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    batch, total = 1, int(model.context_len + model.pred_len)
    x = {
        "latitude": torch.zeros(batch, total, 1, device=device),
        "longitude": torch.zeros(batch, total, 1, device=device),
        "building_type": torch.zeros(batch, total, 1, dtype=torch.long, device=device),
        "day_of_year": torch.zeros(batch, total, 1, device=device),
        "day_of_week": torch.zeros(batch, total, 1, device=device),
        "hour_of_day": torch.zeros(batch, total, 1, device=device),
        "load": torch.zeros(batch, total, 1, device=device),
    }
    with torch.inference_mode():
        out = model(x)
    shape = list(out.shape)
    if shape != [batch, int(model.pred_len), 2]:
        fail(f"unexpected Gaussian Transformer output shape {shape}")
    return {"status": "ok", "mode": "buildings", "device": device,
            "output_shape": shape, "checkpoint_bytes": path.stat().st_size,
            "upstream_config": str(cfg_path)}

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("mode",choices=("beats","buildings")); ap.add_argument("--checkpoint",required=True)
    a=ap.parse_args(); p=pathlib.Path(a.checkpoint)
    if not p.is_file(): fail(f"checkpoint missing: {p}")
    t=time.time(); result = beats(p) if a.mode=="beats" else buildings(p)
    result["elapsed_s"]=round(time.time()-t,3); print(json.dumps(result,sort_keys=True))
if __name__ == "__main__": main()
