"""FoR37 Earth sciences — WeatherBench 2 global 2 m temperature forecasting (latitude-weighted RMSE, K).

Data (read-only): ``<data_root>/for37-weatherbench2/era5_2m_temperature_2018-2020_6h_64x32.zarr`` — ERA5
``2m_temperature`` from the official WeatherBench 2 store
``gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr`` (time indices
86200-90584), complete calendar years 2018-2020, 6-hourly, 64 longitudes x 32 latitudes (5.625°, equiangular,
conservative regridding), dims (time, longitude, latitude), float32 kelvin. The Zarr v2 chunks are
blosc/lz4-compressed; they are decoded with ``numcodecs`` when installed, otherwise with the pure-Python
blosc1+LZ4 block decoder in this module, verified against the receipt (shape, all finite, min/max, first value)
and cached as ``.npy`` under ``<repo>/cache/tasks/FoR37/``.

Item = one forecast initialisation time t0: input the four global fields at t0-18 h, t0-12 h, t0-6 h, t0;
target the global field at t0 + 24 h (lead time 24 h).

Initialisation times follow one global pattern t0 = 2019-01-01 00 UTC + 60 h * j + {0 h, 12 h}. With this spacing no
item's target time lies inside any item's input window (verified at load time), so no evaluation label is ever
visible, in any episode.

Pools / splits (time-disjoint; the partition is fixed and never depends on the seed). Within an episode no two
initialisation times are closer than ``MIN_SPACING_H = 120 h``: another item's context fields are
observations of a nearby item's target, so episodes are drawn from "lanes" (see :func:`lane_pools`):

* per year the 146 pattern slots are cut into ranges of 32 slots separated by 4 unused guard slots (4 complete ranges
  of 64 items per year, the 2-slot tail is dropped); a range holds 4 lanes (slot parity x init hour) of 16 items,
  consecutive lane items exactly 120 h apart; an episode of 16 items is one lane (capacity 4 per range);
* IID pool: the four 2019 ranges are hash-ordered with the partition seed: 1 -> ``val``, 1 -> ``id``, 2 -> ``src``
  (capacities val 4, id 4, src 8 episodes);
* OOD pool: the four 2020 ranges (WeatherBench 2's official test year, one year further from the visible training
  period; capacity 16) -> ``ood_kind = "proxy_within_dataset"`` (temporal shift; no second dataset is available);
* visible training data: all 6-hourly fields 2018-01-01 00 UTC ... 2018-11-30 18 UTC; dev: 16 December-2018
  initialisations of the same pattern (inputs via ``load_dev``, targets held by ``score_dev``).

Metric (lower is better): WeatherBench 2 RMSE — per initialisation ``sqrt(mean_{lon,lat} w(lat) (ŷ - y)^2)`` with
area weights ``w ∝ sin(lat + Δ/2) - sin(lat - Δ/2)`` normalised to mean 1, averaged over initialisations (the WB2
paper's time-mean of per-time RMSE; ``sqrt`` after averaging is reported as a secondary metric).
Reference: 24 h persistence ``ŷ = field(t0)``. Acceptance: RMSE <= 0.96 x reference (calibrated: a 2018-climatology +
0.7 x anomaly-persistence forecast passes some but not all episodes, persistence never; docs/tasks/FoR37.md).
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import scilib

from ...core.schema import PortSchema
from ...core.trace import Trace
from ..task import EvalResult, Episode, ToolSpec
from ._forecast_common import (ForecastAdapterBase, PoolItem, c_declared_unit, c_finite, c_range, c_shape,
                               common_lineage, default_budget, episode_id, finish_eval, hash_order, invalid_eval,
                               task_cache_dir, time_spacing_conflict, to_float_array, unwrap_payloads,
                               wb2_lat_weights)

CODE = "FoR37"
ZARR_NAME = "era5_2m_temperature_2018-2020_6h_64x32.zarr"
T_ORIGIN = np.datetime64("2018-01-01T00", "h")
STEP_H = 6
CONTEXT_OFFSETS_H = (-18, -12, -6, 0)
LEAD_H = 24
PATTERN_PERIOD_H = 60
PATTERN_OFFSETS_H = (0, 12)
TRAIN_END = np.datetime64("2018-11-30T18", "h")
DEV_START = np.datetime64("2018-12-01T00", "h")
DEV_TARGET_MAX = np.datetime64("2018-12-30T18", "h")
N_DEV = 16
# Draw-time spacing. Within one episode the initialisation times are >= MIN_SPACING_H apart, because an item
# whose init is a few days after another's leaks that item's target (its context fields are near-future observations
# of it: oracle-fitted gain over anomaly persistence 19.5 % at a 12 h gap, 49 % at 48 h, 28 % at 60 h, 15 % at 72 h,
# 3.7 % at 120 h, 0.9 % at 240 h; earlier neighbours <= 0.4 %). The 2019 / 2020 pattern slots (60 h apart) are cut
# into ranges of RANGE_SLOTS usable slots separated by GUARD_SLOTS unused ones; a range holds
# SPACING_SLOTS x len(PATTERN_OFFSETS_H) "lanes" (slot parity x init hour) of LANE_ITEMS items, consecutive lane items
# exactly MIN_SPACING_H apart; an episode of <= LANE_ITEMS items is drawn from one lane.
SPACING_SLOTS = 2
MIN_SPACING_H = SPACING_SLOTS * PATTERN_PERIOD_H
RANGE_SLOTS = 32
GUARD_SLOTS = 4
LANE_ITEMS = RANGE_SLOTS // SPACING_SLOTS
VAL_RANGES, ID_RANGES = 1, 1
ACCEPT_MARGIN = 0.04
UNIT = "K"
T_MIN_K, T_MAX_K = 150.0, 350.0

DATASET = "WeatherBench 2 ERA5 2m_temperature, 6-hourly 64x32 equiangular conservative (2018-2020 subset)"
VERSION = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr, time 86200-90584"
SOURCE = "https://weatherbench2.readthedocs.io/en/latest/data-guide.html"
LICENSE = "ERA5: Copernicus licence (C3S); WeatherBench 2 data: CC BY 4.0"


# --------------------------------------------------------------------------------------------- zarr decoding
def lz4_block_decompress(src: bytes, out_size: int) -> bytes:
    """Decode one raw LZ4 block (no frame header) of known decompressed size."""
    dst = bytearray()
    i, n = 0, len(src)
    while i < n:
        token = src[i]
        i += 1
        lit = token >> 4
        if lit == 15:
            while True:
                b = src[i]
                i += 1
                lit += b
                if b != 255:
                    break
        if lit:
            dst += src[i:i + lit]
            i += lit
        if i >= n:
            break
        off = src[i] | (src[i + 1] << 8)
        i += 2
        if off == 0:
            raise ValueError("corrupt LZ4 block: zero offset")
        ml = token & 15
        if ml == 15:
            while True:
                b = src[i]
                i += 1
                ml += b
                if b != 255:
                    break
        ml += 4
        start = len(dst) - off
        if start < 0:
            raise ValueError("corrupt LZ4 block: offset before start")
        if off >= ml:
            dst += dst[start:start + ml]
        else:
            seg = bytes(dst[start:])
            reps, rem = divmod(ml, off)
            dst += seg * reps + seg[:rem]
    if len(dst) != out_size:
        raise ValueError(f"LZ4 size mismatch: {len(dst)} != {out_size}")
    return bytes(dst)


def blosc_decompress(buf: bytes) -> bytes:
    """Decode a blosc1 frame (compressor LZ4, optional byte shuffle) in pure Python."""
    if len(buf) < 16:
        raise ValueError("blosc frame too short")
    flags, typesize = buf[2], buf[3]
    nbytes, blocksize, cbytes = struct.unpack_from("<III", buf, 4)
    if cbytes != len(buf):
        raise ValueError(f"blosc cbytes {cbytes} != frame length {len(buf)}")
    if flags & 0x02:                                   # memcpyed
        return bytes(buf[16:16 + nbytes])
    if (flags & 0xE0) >> 5 != 1:
        raise ValueError(f"unsupported blosc compressor code {(flags & 0xE0) >> 5} (only lz4)")
    if flags & 0x04:
        raise ValueError("bit-shuffle not supported")
    dont_split = bool(flags & 0x10)
    nblocks = (nbytes + blocksize - 1) // blocksize
    bstarts = struct.unpack_from(f"<{nblocks}i", buf, 16)
    out = bytearray()
    for bi in range(nblocks):
        leftover = bi == nblocks - 1 and nbytes % blocksize != 0
        bsize = nbytes % blocksize if leftover else blocksize
        nsplits = typesize if (not dont_split and not leftover and typesize <= 16 and bsize // typesize >= 128) else 1
        neblock = bsize // nsplits
        pos = bstarts[bi]
        parts = []
        for _ in range(nsplits):
            csize = struct.unpack_from("<i", buf, pos)[0]
            pos += 4
            chunk = buf[pos:pos + csize]
            pos += csize
            parts.append(bytes(chunk) if csize == neblock else lz4_block_decompress(chunk, neblock))
        block = b"".join(parts)
        if flags & 0x01 and typesize > 1:
            nel = bsize // typesize
            main = np.frombuffer(block[:nel * typesize], dtype=np.uint8).reshape(typesize, nel).T.tobytes()
            block = main + block[nel * typesize:]
        out += block
    return bytes(out)


def _decode_chunk(raw: bytes) -> bytes:
    try:
        import numcodecs  # optional fast path
    except ImportError:
        return blosc_decompress(raw)
    return bytes(numcodecs.Blosc().decode(raw))


def read_zarr_array(path: Path) -> np.ndarray:
    """Read a Zarr v2 array chunked along axis 0 only (blosc/lz4, C order)."""
    meta = json.loads((path / ".zarray").read_text())
    if meta.get("zarr_format") != 2 or meta.get("order") != "C" or meta.get("filters"):
        raise ValueError(f"{path.name}: unsupported zarr metadata {meta}")
    comp = meta.get("compressor") or {}
    if comp.get("id") != "blosc":
        raise ValueError(f"{path.name}: unsupported compressor {comp}")
    shape, chunks = tuple(meta["shape"]), tuple(meta["chunks"])
    if tuple(chunks[1:]) != tuple(shape[1:]):
        raise ValueError(f"{path.name}: only axis-0 chunking supported, chunks {chunks} shape {shape}")
    dtype = np.dtype(meta["dtype"])
    nchunks = -(-shape[0] // chunks[0])
    parts = []
    for c in range(nchunks):
        key = ".".join([str(c)] + ["0"] * (len(shape) - 1))
        raw = (path / key).read_bytes()
        arr = np.frombuffer(_decode_chunk(raw), dtype=dtype)
        parts.append(arr.reshape(chunks))
    return np.concatenate(parts, axis=0)[:shape[0]]


def _source_fingerprint(zdir: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(zdir.rglob("*")):
        if p.is_file():
            st = p.stat()
            h.update(f"{p.relative_to(zdir)}|{st.st_size}|{int(st.st_mtime)}\n".encode())
    return h.hexdigest()


@dataclass
class WBData:
    t2m: np.ndarray            # (T, 64, 32) float32 K
    times: np.ndarray          # (T,) datetime64[h]
    lat: np.ndarray            # (32,)
    lon: np.ndarray            # (64,)
    items: dict[str, np.datetime64]   # item id -> t0
    pools: dict[str, list[str]]
    groups: dict[str, str]
    dev_t0: list[np.datetime64]


def _tidx(t: np.datetime64) -> int:
    d = int((t - T_ORIGIN).astype(int))
    if d % STEP_H:
        raise ValueError(f"{t} is not on the 6-hour grid")
    return d // STEP_H


def _pattern(start: np.datetime64, t0_max: np.datetime64) -> list[np.datetime64]:
    out = []
    j = 0
    while True:
        base = start + np.timedelta64(PATTERN_PERIOD_H * j, "h")
        if base > t0_max:
            return out
        for off in PATTERN_OFFSETS_H:
            t = base + np.timedelta64(off, "h")
            if t <= t0_max:
                out.append(t)
        j += 1


def load_weatherbench(root: Path, partition_seed: int, cache_root: Any = None) -> WBData:
    base = root / "for37-weatherbench2"
    zdir = base / ZARR_NAME
    fp = _source_fingerprint(zdir)
    cdir = task_cache_dir(CODE, cache_root, create=False)
    npy, meta_p = cdir / "era5_t2m_2018_2020_6h_64x32.npy", cdir / "era5_t2m_2018_2020_6h_64x32.json"
    t2m = None
    if npy.is_file() and meta_p.is_file():
        meta = json.loads(meta_p.read_text())
        if meta.get("source_fingerprint") == fp:
            t2m = np.load(npy, allow_pickle=False)
    if t2m is None:
        t2m = read_zarr_array(zdir / "2m_temperature").astype(np.float32)
        try:
            cdir.mkdir(parents=True, exist_ok=True)
            tmp = npy.with_suffix(".tmp.npy")
            np.save(tmp, t2m)
            os.replace(tmp, npy)
            meta_p.write_text(json.dumps({"source_fingerprint": fp, "source": str(zdir), "shape": list(t2m.shape),
                                          "note": "decoded copy of the read-only WeatherBench2 subset"}, indent=1))
        except OSError as ex:     # the cache is an optimisation only; decoding again next time is correct
            warnings.warn(f"FoR37: could not write the decoded cache to {cdir}: {ex}", RuntimeWarning, stacklevel=2)
    lat = read_zarr_array(zdir / "latitude").astype(float)
    lon = read_zarr_array(zdir / "longitude").astype(float)
    hours = read_zarr_array(zdir / "time").astype(np.int64)
    times = np.datetime64("1959-01-01T00", "h") + hours.astype("timedelta64[h]")
    receipt = json.loads((base / "receipt.json").read_text())["validation"]
    if list(t2m.shape) != list(receipt["shape"]) or not np.all(np.isfinite(t2m)):
        raise ValueError(f"decoded t2m shape {t2m.shape} / finiteness differs from the receipt")
    if not (abs(float(t2m.min()) - receipt["minimum_kelvin"]) < 1e-3 and abs(float(t2m.max()) - receipt["maximum_kelvin"]) < 1e-3
            and abs(float(t2m[0, 0, 0]) - receipt["sample_t0_lon0_lat0_kelvin"]) < 1e-4):
        raise ValueError("decoded t2m values differ from the receipt (min/max/first value)")
    if times[0] != T_ORIGIN or not np.all(np.diff(times).astype(int) == STEP_H) or lat.shape != (32,) or lon.shape != (64,):
        raise ValueError("unexpected time/latitude/longitude coordinates")
    # items
    t_last = times[-1]
    all_t0 = _pattern(np.datetime64("2019-01-01T00", "h"), t_last - np.timedelta64(LEAD_H, "h"))
    ctx_times = {t0 + np.timedelta64(o, "h") for t0 in all_t0 for o in CONTEXT_OFFSETS_H}
    targets = {t0 + np.timedelta64(LEAD_H, "h") for t0 in all_t0}
    dev_t0 = _pattern(DEV_START, DEV_TARGET_MAX - np.timedelta64(LEAD_H, "h"))[:N_DEV]
    dev_ctx = {t0 + np.timedelta64(o, "h") for t0 in dev_t0 for o in CONTEXT_OFFSETS_H}
    dev_tg = {t0 + np.timedelta64(LEAD_H, "h") for t0 in dev_t0}
    if targets & ctx_times or targets & dev_ctx or dev_tg & (ctx_times | dev_ctx) or max(dev_tg) >= min(ctx_times) \
            or min(dev_tg) <= TRAIN_END or min(targets) <= TRAIN_END:
        raise ValueError("item pattern leaks targets into visible inputs")
    items = {f"era5_t2m:init={str(t)}:00:lead={LEAD_H}h": t for t in all_t0}
    pools, groups = lane_pools(items, partition_seed)
    return WBData(t2m=t2m, times=times, lat=lat, lon=lon, items=items, pools=pools, groups=groups, dev_t0=dev_t0)


def lane_pools(items: dict[str, np.datetime64], partition_seed: int) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Arrange the initialisations into ranges and lanes and split the ranges into ``src/val/id/ood``.

    Per calendar year the pattern slots (``60 h`` apart, two init hours each) are numbered from the year start; slot
    ``q`` belongs to range ``q // (RANGE_SLOTS + GUARD_SLOTS)`` at position ``q % ...``; guard positions and incomplete
    tail ranges are unused. A lane = (year, range, slot parity, init hour); its ``LANE_ITEMS`` items are exactly
    ``MIN_SPACING_H`` apart. 2019 ranges are hash-ordered with the partition seed: ``VAL_RANGES`` -> val,
    ``ID_RANGES`` -> id, the rest -> src; every 2020 range -> ood. Returns ``(pools, groups)`` (item id lists,
    item id -> lane name).
    """
    span = RANGE_SLOTS + GUARD_SLOTS
    lanes: dict[tuple[str, int, int, int], list[str]] = {}
    for i, t in items.items():
        year = str(t)[:4]
        hrs = int((t - np.datetime64(f"{year}-01-01T00", "h")).astype(int))
        q, off = divmod(hrs, PATTERN_PERIOD_H)
        if off not in PATTERN_OFFSETS_H:
            raise ValueError(f"{i}: init hour offset {off} h is not on the item pattern in {year}")
        r, pos = divmod(q, span)
        if pos >= RANGE_SLOTS:
            continue                                                     # guard slot
        lanes.setdefault((year, r, pos % SPACING_SLOTS, off), []).append(i)
    per_range: dict[tuple[str, int], list[int]] = {}
    for (year, r, _, _), ids in lanes.items():
        per_range.setdefault((year, r), []).append(len(ids))
    n_lanes = SPACING_SLOTS * len(PATTERN_OFFSETS_H)
    complete = {k for k, sizes in per_range.items()                      # incomplete tail ranges are dropped
                if len(sizes) == n_lanes and all(s == LANE_ITEMS for s in sizes)}
    rng_names: dict[str, list[str]] = {}
    groups: dict[str, str] = {}
    for (year, r, par, off), ids in sorted(lanes.items()):
        if (year, r) not in complete:
            continue
        name = f"{year}-r{r}-p{par}-o{off:02d}"
        for i in ids:
            groups[i] = name
        rng_names.setdefault(f"{year}-range{r}", []).extend(ids)
    iid = [n for n in rng_names if n.startswith("2019-")]
    order = hash_order(iid, f"{CODE}|ranges|{partition_seed}")
    pools = {"val": [i for n in order[:VAL_RANGES] for i in rng_names[n]],
             "id": [i for n in order[VAL_RANGES:VAL_RANGES + ID_RANGES] for i in rng_names[n]],
             "src": [i for n in order[VAL_RANGES + ID_RANGES:] for i in rng_names[n]],
             "ood": [i for n, ids in rng_names.items() if n.startswith("2020-") for i in ids]}
    return pools, groups


def min_gap_h(t0s: list[np.datetime64]) -> int | None:
    """Smallest gap in hours between two initialisation times of the list (None for fewer than two)."""
    h = sorted(int((t - T_ORIGIN).astype(int)) for t in t0s)
    return min((b - a for a, b in zip(h, h[1:])), default=None)


# --------------------------------------------------------------------------------------------- metric
def weighted_rmse(pred: np.ndarray, obs: np.ndarray, w_lat: np.ndarray) -> np.ndarray:
    """Per-item WB2 RMSE for arrays (n, lon, lat): sqrt(mean_{lon,lat} w(lat) (pred - obs)^2)."""
    return np.sqrt(np.mean((np.asarray(pred, float) - np.asarray(obs, float)) ** 2 * w_lat[None, None, :], axis=(1, 2)))


class Adapter(ForecastAdapterBase):
    """ScienceClaw-Eval FoR37 adapter (WeatherBench 2 ERA5 2 m temperature, 24 h lead)."""

    code = CODE
    name = "WeatherBench2"
    task_type = "spatiotemporal_forecasting"

    def _check_files(self) -> list[str]:
        base = self.data_root / "for37-weatherbench2"
        zdir = base / ZARR_NAME
        missing = [f"missing {p}" for p in (base / "receipt.json", zdir / "2m_temperature" / ".zarray",
                                            zdir / "time" / "0", zdir / "latitude" / "0", zdir / "longitude" / "0")
                   if not p.is_file()]
        if missing:
            return missing
        meta = json.loads((zdir / "2m_temperature" / ".zarray").read_text())
        n = -(-meta["shape"][0] // meta["chunks"][0])
        absent = [k for k in range(n) if not (zdir / "2m_temperature" / f"{k}.0.0").is_file()]
        return [f"missing 2m_temperature chunks {absent[:5]}{'...' if len(absent) > 5 else ''}"] if absent else []

    def data(self) -> WBData:
        return self._memo.get("data", lambda: load_weatherbench(self.data_root, self.partition_seed, self.cache_root))

    def _split_pools(self) -> dict[str, list[PoolItem]]:
        d = self.data()
        hours = {i: int((t - T_ORIGIN).astype(int)) for i, t in d.items.items()}      # init time, h since T_ORIGIN
        return {k: [PoolItem(i, d.groups[i], (hours[i],)) for i in ids] for k, ids in d.pools.items()}

    def _draw_options(self, split: str) -> dict:
        """Every episode is drawn from one lane and no two items are closer than MIN_SPACING_H."""
        return {"lanes": True, "conflict": time_spacing_conflict(lambda it: it.meta[0], MIN_SPACING_H)}

    # ----------------------------------------------------------------------------------------------
    def _make_episode(self, split: str, k: int, items: list[PoolItem], seed: int, reused: bool) -> Episode:
        d = self.data()
        n = len(items)
        ids = [it.id for it in items]
        t0s = [d.items[i] for i in ids]
        nlon, nlat = d.t2m.shape[1], d.t2m.shape[2]
        w_lat = wb2_lat_weights(d.lat)

        def ctx_of(t0s_: list[np.datetime64]) -> np.ndarray:
            return np.stack([np.stack([d.t2m[_tidx(t + np.timedelta64(o, "h"))] for o in CONTEXT_OFFSETS_H])
                             for t in t0s_]).astype(np.float64)

        def tgt_of(t0s_: list[np.datetime64]) -> np.ndarray:
            return np.stack([d.t2m[_tidx(t + np.timedelta64(LEAD_H, "h"))] for t in t0s_]).astype(np.float64)

        ctx, tgt = ctx_of(t0s), tgt_of(t0s)
        ref_pred = ctx[:, -1]
        ref_rmse = weighted_rmse(ref_pred, tgt, w_lat)
        ref_score = float(ref_rmse.mean())
        ref_payload = {"rmse": ref_rmse.tolist(), "ids": ids}
        pool = "ood" if split == "ood" else "iid"
        iso = lambda t: str(t) + ":00"  # noqa: E731
        init_iso = [iso(t) for t in t0s]
        valid_iso = [iso(t + np.timedelta64(LEAD_H, "h")) for t in t0s]

        # ------------------------------------------------------------------ D_E tools
        tr_end = _tidx(TRAIN_END) + 1
        train_times = [iso(t) for t in d.times[:tr_end]]
        dev_ctx, dev_tgt = ctx_of(d.dev_t0), tgt_of(d.dev_t0)
        dev_ref = float(weighted_rmse(dev_ctx[:, -1], dev_tgt, w_lat).mean())
        n_dev = len(d.dev_t0)
        dev_init = [iso(t) for t in d.dev_t0]
        dev_valid = [iso(t + np.timedelta64(LEAD_H, "h")) for t in d.dev_t0]
        lat, lon = d.lat.copy(), d.lon.copy()
        t2m = d.t2m

        def grid() -> dict:
            return {"latitude": lat.copy(), "longitude": lon.copy(), "lat_weights": w_lat.copy()}

        def load_train(inputs: dict, config: dict) -> dict:
            return {"t2m": np.array(t2m[:tr_end], dtype=np.float32), "time": list(train_times), **grid()}

        def load_dev(inputs: dict, config: dict) -> dict:
            return {"context": dev_ctx.copy(), "init_time": list(dev_init), "valid_time": list(dev_valid)}

        def score_dev(inputs: dict, config: dict) -> dict:
            pred = to_float_array(inputs.get("pred"))
            if pred.shape != (n_dev, nlon, nlat):
                raise ValueError(f"pred must have shape ({n_dev}, {nlon}, {nlat}), got {tuple(pred.shape)}")
            if not np.all(np.isfinite(pred)):
                raise ValueError("pred contains non-finite values")
            per = weighted_rmse(pred, dev_tgt, w_lat)
            s = float(per.mean())
            return {"score": s, "report": {"metric": "latitude-weighted RMSE (K) on the dev initialisations "
                                                     "(lower is better)", "score": s, "reference_score": dev_ref,
                                           "per_item": per.tolist(), "n_items": n_dev}}

        def load_eval_inputs(inputs: dict, config: dict) -> dict:
            return {"context": ctx.copy(), "init_time": list(init_iso), "valid_time": list(valid_iso),
                    "context_offsets_h": list(CONTEXT_OFFSETS_H), **grid()}

        field_desc = "2 m temperature fields, dims (…, longitude, latitude) as in the WeatherBench 2 store"
        tools = [
            ToolSpec("load_train",
                     f"Visible training data: ERA5 2 m temperature every 6 h from 2018-01-01 00 UTC to "
                     f"2018-11-30 18 UTC ({tr_end} fields; t2m[t, i, j] at time[t], longitude[i], latitude[j]), "
                     "plus the grid (longitude °E 0..354.375, latitude °N ascending) and the metric's area weights.",
                     {}, {"t2m": PortSchema("array", (tr_end, nlon, nlat), UNIT, "float", field_desc),
                          "time": PortSchema("list", (tr_end,), None, "str", "ISO UTC times"),
                          "latitude": PortSchema("array", (nlat,), "deg", "float"),
                          "longitude": PortSchema("array", (nlon,), "deg", "float"),
                          "lat_weights": PortSchema("array", (nlat,), "1", "float", "area weights, mean 1")},
                     load_train),
            ToolSpec("load_dev",
                     f"Dev inputs: {n_dev} December-2018 initialisations in the same format as load_eval_inputs "
                     "(context fields at init-18 h, -12 h, -6 h, 0 h); score_dev holds their +24 h fields.",
                     {}, {"context": PortSchema("array", (n_dev, len(CONTEXT_OFFSETS_H), nlon, nlat), UNIT, "float",
                                                field_desc),
                          "init_time": PortSchema("list", (n_dev,), None, "str"),
                          "valid_time": PortSchema("list", (n_dev,), None, "str")},
                     load_dev),
            ToolSpec("score_dev",
                     "Scores a forecast of the dev initialisations: pred[i] is the forecast for load_dev's context[i] and "
                     "init_time[i] (a forecast made from load_eval_inputs is a forecast of other times and is scored "
                     "against the dev targets as such); returns the mean latitude-weighted RMSE in K (lower is better) "
                     "and, in report, the same statistic of the reference forecast. load_dev has the layout of "
                     "load_eval_inputs and the final evaluation applies the same per-initialisation latitude-weighted "
                     "RMSE, so a function (context, init_time) -> forecast can be applied to both.",
                     {"pred": PortSchema("array", (n_dev, nlon, nlat), UNIT, "float",
                                         "forecast for the load_dev initialisations, same layout as y")},
                     {"score": PortSchema("number", None, UNIT, "float"), "report": PortSchema("dict")},
                     score_dev),
            ToolSpec("load_eval_inputs",
                     f"Evaluation inputs (no targets) for the {n} items in item order: context[i, k] = global 2 m "
                     "temperature field at init_time[i] + context_offsets_h[k] (offsets -18, -12, -6, 0 h); "
                     "valid_time[i] = init_time[i] + 24 h; grid coordinates and metric area weights.",
                     {}, {"context": PortSchema("array", (n, len(CONTEXT_OFFSETS_H), nlon, nlat), UNIT, "float",
                                                field_desc),
                          "init_time": PortSchema("list", (n,), None, "str"),
                          "valid_time": PortSchema("list", (n,), None, "str"),
                          "context_offsets_h": PortSchema("list", (len(CONTEXT_OFFSETS_H),), "h", "int"),
                          "latitude": PortSchema("array", (nlat,), "deg", "float"),
                          "longitude": PortSchema("array", (nlon,), "deg", "float"),
                          "lat_weights": PortSchema("array", (nlat,), "1", "float")},
                     load_eval_inputs),
        ]

        # ------------------------------------------------------------------ D_V
        constraints = [c_shape((n, nlon, nlat)), c_finite(),
                       c_range(T_MIN_K, T_MAX_K, UNIT, desc=f"all values within [{T_MIN_K:g}, {T_MAX_K:g}] K "
                                                           "(physically plausible near-surface temperatures in kelvin)"),
                       c_declared_unit(UNIT)]

        def evaluate(y: Any, trace: Trace | None) -> EvalResult:
            common = {"reference": ref_score, "direction": "min", "margin": ACCEPT_MARGIN,
                      "reference_payload": ref_payload}
            base_m = {"reference_rmse_K": ref_score}
            try:
                arr = to_float_array(y)
            except ValueError as ex:
                return invalid_eval(str(ex), metrics=base_m, **common)
            if arr.shape != (n, nlon, nlat):
                return invalid_eval(f"shape {tuple(arr.shape)} != ({n}, {nlon}, {nlat})", metrics=base_m, **common)
            if not np.all(np.isfinite(arr)):
                return invalid_eval("non-finite predictions", metrics=base_m, **common)
            per = weighted_rmse(arr, tgt, w_lat)
            mse = per ** 2
            bias = float(np.mean((arr - tgt) * w_lat[None, None, :]))
            metrics = {"rmse_K": float(per.mean()), "rmse_sqrt_after_mean_K": float(np.sqrt(mse.mean())),
                       "bias_K": bias, **base_m}
            return finish_eval(primary=float(per.mean()), metrics=metrics,
                               payload={"rmse": per.tolist(), "ids": ids}, **common)

        objective = (
            "Earth sciences / numerical weather prediction: 24-hour forecasts of the global 2 m air temperature field "
            "(ERA5 reanalysis on the WeatherBench 2 64 x 32 equiangular grid, 5.625°). "
            f"There are {n} evaluation items; each is one initialisation time t0 = init_time[i] (UTC). Input of item "
            "i (tool load_eval_inputs): context[i], the four global 2 m temperature fields (K) at t0-18 h, t0-12 h, "
            "t0-6 h and t0, each of shape (64 longitudes, 32 latitudes). Target of item i: the global 2 m "
            "temperature field at valid_time[i] = t0 + 24 h. Visible training data covers January-November 2018; the "
            "evaluation initialisations are in a later period.\n"
            f"Deliverable y: a float array of shape ({n}, {nlon}, {nlat}) in kelvin; y[i, a, b] is the forecast for "
            "item i (order of load_eval_inputs) at longitude index a and latitude index b of the given grid. Values "
            "must be finite and within [150, 350] K.\n"
            "Score (lower is better): WeatherBench 2 RMSE = mean over items of sqrt(mean over grid points of "
            "lat_weights[b] * (y - observed)^2), lat_weights = normalised cell-area weights.\n"
            + scilib.describe("weather")
        )
        lineage = common_lineage(
            dataset=DATASET, version=VERSION, source=SOURCE, license_=LICENSE, pool=pool,
            ood_kind="proxy_within_dataset" if pool == "ood" else None, items=items, seed=seed,
            partition_seed=self.partition_seed, k=k, reused=reused,
            split_rule=("time-disjoint: init pattern 2019-01-01 00 UTC + 60 h j + {0, 12 h}; per year, ranges of "
                        f"{RANGE_SLOTS} slots ({GUARD_SLOTS} guard slots between ranges) with {SPACING_SLOTS * len(PATTERN_OFFSETS_H)} "
                        f"lanes of {LANE_ITEMS} inits {MIN_SPACING_H} h apart; an episode is one lane (no two inits "
                        f"closer than {MIN_SPACING_H} h); 2019 ranges hash-ordered: {VAL_RANGES} val, {ID_RANGES} id, "
                        "rest src; 2020 ranges = OOD (temporal shift, WB2 test year)"),
            visible="load_train: ERA5 t2m 2018-01-01 00 UTC..2018-11-30 18 UTC; load_dev: 16 Dec-2018 inits; "
                    "per-item 4 context fields",
            extra={"lead_h": LEAD_H, "context_offsets_h": list(CONTEXT_OFFSETS_H), "init_times": init_iso,
                   "min_spacing_h": MIN_SPACING_H, "min_item_spacing_h": min_gap_h(t0s)})
        return Episode(
            id=episode_id(CODE, split, seed, k), discipline=CODE, family=self.family, split=split,
            task_type=self.task_type, objective=objective,
            required_output=PortSchema("array", (n, nlon, nlat), UNIT, "float",
                                       "24 h forecast of the global 2 m temperature field per initialisation"),
            tools=tools, constraints=constraints, budget=default_budget(max_node_s=300.0, max_llm_items=2 * n),
            lineage=lineage,
            acceptance=(f"accepted iff latitude-weighted RMSE <= {1 - ACCEPT_MARGIN:.2f} x the reference RMSE on the "
                        "same items (reference recipe and value only in EvalResult.details / docs, never shown to the "
                        "policy)"),
            tolerance={"rtol": 1e-5, "atol": 1e-4},
            tags=[CODE, self.family, "weather", "forecasting", "gridded", "spatiotemporal", "ERA5", "2m-temperature",
                  "WeatherBench2", UNIT],
            metric=self.metric, direction="min", n_items=n, _evaluate=evaluate, _dev_evaluate=None)

    # ----------------------------------------------------------------------------------------------
    def pooled_metric(self, per_episode: list[dict]) -> float | None:
        """Mean over all initialisations of the per-initialisation latitude-weighted RMSE (K)."""
        vals = [float(v) for p in unwrap_payloads(per_episode) for v in p.get("rmse", [])]
        return float(np.mean(vals)) if vals else None
