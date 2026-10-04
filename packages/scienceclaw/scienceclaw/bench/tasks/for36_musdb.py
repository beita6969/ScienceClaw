"""FoR36 Creative arts and writing — MUSDB18 four-stem music source separation, metric mean target-median SDR (dB).

Status with the data present on 2026-09-28: **available** on the data team's ``full_v1`` (all 150 official MUSDB18
tracks, AAC STEMS). Split sizes, in tracks -> 6-s excerpts: src 64 -> 128, val 14 -> 70, id 50 -> 150, ood 0 (none),
visible train 16 -> 64 and dev 6 -> 24 (both from the 22 ``reserve`` tracks).

* **Roles** come from ``<DATA_ROOT>/for36-musdb18/full_v1/roles/*.jsonl`` (verified against the sha256 and counts of
  ``sampling-manifest.json``): ``source`` (64 official-train tracks) -> ``src``; ``validation`` (the 14 official
  package validation tracks) -> ``val``; ``ID`` (the 50 official test tracks) -> ``id``. ``OOD`` is **empty** (no
  independent collection was supplied and the data team refuses to relabel a random split as OOD), so
  ``build_episodes("ood", ...)`` returns ``[]`` (``SplitPlan`` records its "adapter returned 0 of N" warning).
  ``reserve`` (22 train tracks the data team left unused) is *not* evaluated: it is cut by track (sha256, partition
  seed 20260928) into 16 visible-training and 6 visible-dev tracks, so the visible data D_E is track-disjoint from
  every evaluated item (owner-vetoable choice; the alternative is to enlarge the src / val pools).
* **Item** = one 6-second stereo excerpt of a track: mixture -> the four official targets (vocals, drums, bass,
  other). Every track contributes a fixed, hash-chosen set of non-overlapping 6-s slots (src 2, val 5, id 3, train
  4, dev 4 per track). Lineage unit = track (the roles are track-disjoint).
* **Decoding.** The stems are AAC multi-stream MP4 (stream 0 mixture, 1 drums, 2 bass, 3 other, 4 vocals; 44.1 kHz
  stereo). The main environment has no audio library, so decoding shells out to ffmpeg: the data team's pinned
  binary (``loader-config.json``: path + sha256), else ``$SCIENCECLAW_FFMPEG`` / ``ffmpeg`` on PATH /
  ``imageio_ffmpeg``. Streams are decoded to float32 at the native rate; with the pinned binary the decoded PCM is
  checked against the role file's ``decoded_pcm_sha256`` (bit-exact). The excerpt is then **decimated to 22,050 Hz**
  with ``scipy.signal.resample_poly(x, 1, 2)`` (documented deviation from 44.1 kHz; CPU / memory reduction).
  Excerpts are cached per track in ``cache/tasks/FoR36/`` (about 2.3 GB when every track has been touched).
* **Visible data (D_E).** ``load_train``: 8 training excerpts (mixture + the four stems + ``track_ids``, one opaque
  label per excerpt, equal labels = same track, so a cross-validation can keep the excerpts of a track together);
  ``load_dev_inputs`` /
  ``score_dev``: 4 dev mixtures and the mean target-median SDR of dev estimates; ``load_eval_inputs``: the 16
  evaluation mixtures, shape (16, n_samples, 2), sample rate 22,050 Hz.
* **Metric (D_V).** BSSEval v4 SDR as computed by ``museval`` (``mode="v4"``, image-based decomposition,
  1-second windows and hops): per window ``SDR = 10 log10(sum s^2 / sum (s_hat - s)^2)`` over both channels
  (in the v4 image decomposition ``e_spat + e_interf + e_artif = s_hat - s``, so SDR does not depend on the 512-tap
  distortion filters); a window is skipped (NaN) when any reference source or any estimated source is exactly
  silent in it. Item score per target = median over windows; episode primary = mean over the four targets of the
  median over items ("mean target-median SDR"; SiSEC 2018 aggregation). Pooled metric = the same over all items.
* **Reference baseline:** the SiSEC "MIX" anchor (the mixture used as the estimate of every target).
  **Acceptance:** ``primary >= reference + 3 dB``. ``norm_score = 10^((primary - reference)/10)`` clipped to
  [0, 10] (power-ratio analogue of primary/reference for a dB metric). Note the AAC mixture is not exactly the sum
  of the AAC stems (the data team's limitation notes), so the four stems do not sum to the mixture bit-exactly.
* **Hard constraints:** y has shape (16, 4, n_samples, 2); all values finite; no estimated target is exactly silent
  over a whole 1-second window (such windows would otherwise be skipped by the museval rule).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import zipfile
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import scilib
from ...core.schema import PortSchema
from ...core.trace import Trace
from ..task import Budget, ConstraintSpec, EvalResult, Episode, ToolSpec
from ._adapter_utils_for36_46_49_52 import (
    PARTITION_SEED, Lazy, cache_dir, check_split, draw_blocks, episode_id, episode_rng, file_sha256, hash_rank,
    norm_score_db, partition_ids, read_json, receipt_summary, resolve_data_root, sha_hex,
)

CODE = "FoR36"
FAMILY = "Humanities & law"
DATASET_DIR = "for36-musdb18"
DATA_VERSION = "full_v1"
TARGETS = ("vocals", "drums", "bass", "other")
STEM_INDEX = {"mixture": 0, "drums": 1, "bass": 2, "other": 3, "vocals": 4}      # official stem ids (mus.yaml)
NATIVE_SR = 44_100
SR = 22_050
DECIM = NATIVE_SR // SR
EXCERPT_S = 6.0
WIN_S = 1.0
ACCEPT_MARGIN_DB = 3.0
N_TRAIN = 8
N_DEV = 4
ROLE_POOL = {"source": "src", "validation": "val", "ID": "id"}      # data-team role file -> evaluated pool
EXCERPTS_PER_TRACK = {"src": 2, "val": 5, "id": 3, "train": 4, "dev": 4}
RESERVE_SPLIT = {"train": 16, "dev": 6}                             # tracks of the 22 reserve tracks
REQUIRED_TRACKS = {"src": 64, "val": 14, "id": 50, "ood": 0, "train": 16, "dev": 6}
_CACHE_VERSION = "v1"
_RESAMPLE_PAD = 4096            # native samples of context around an excerpt (the anti-alias FIR is 41 taps)
_MEM_TRACKS = 16                # decoded tracks kept in memory (LRU)


# ================================================================================================ metric
def framewise_sdr(ref: np.ndarray, est: np.ndarray, win: int, hop: int | None = None) -> np.ndarray:
    """BSSEval-v4 (museval ``mode="v4"``) framewise SDR.

    ``ref`` / ``est``: (nsrc, nsampl, nchan). Returns (nsrc, nwin) with NaN for windows in which any reference or
    any estimated source is all-zero (museval ``_any_source_silent``) and +inf for a perfect estimate.
    """
    ref = np.asarray(ref, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)
    if ref.shape != est.shape or ref.ndim != 3:
        raise ValueError(f"reference {ref.shape} and estimate {est.shape} must both be (nsrc, nsampl, nchan)")
    hop = hop or win
    nsrc, nsampl, _ = ref.shape
    nwin = int(np.floor((nsampl - win + hop) / hop))
    out = np.full((nsrc, max(nwin, 0)), np.nan)
    for t in range(max(nwin, 0)):
        sl = slice(t * hop, t * hop + win)
        r, e = ref[:, sl], est[:, sl]
        if np.any(np.all(r.sum(axis=2) == 0, axis=1)) or np.any(np.all(e.sum(axis=2) == 0, axis=1)):
            continue
        num = np.sum(r ** 2, axis=(1, 2))
        den = np.sum((e - r) ** 2, axis=(1, 2))
        with np.errstate(divide="ignore"):
            out[:, t] = np.where(den == 0, np.inf, 10.0 * np.log10(num / np.where(den == 0, 1.0, den)))
    return out


def item_scores(ref: np.ndarray, est: np.ndarray, sr: int = SR) -> np.ndarray:
    """Per-target median over 1-s windows (NaN windows ignored); NaN if a target has no valid window."""
    f = framewise_sdr(ref, est, int(WIN_S * sr))
    res = np.full(f.shape[0], np.nan)
    for j in range(f.shape[0]):
        v = f[j][~np.isnan(f[j])]
        if v.size:
            res[j] = float(np.median(v))
    return res


def aggregate(per_item: np.ndarray) -> float | None:
    """Mean over targets of the median over items (items x targets; NaN entries ignored)."""
    per_item = np.asarray(per_item, dtype=float)
    if per_item.size == 0:
        return None
    meds = []
    for j in range(per_item.shape[1]):
        v = per_item[:, j][~np.isnan(per_item[:, j])]
        if v.size == 0:
            return None
        meds.append(float(np.median(v)))
    return float(np.mean(meds))


# ================================================================================================ audio io
def _ffmpeg_candidates(base: Path) -> tuple[list[str], str | None]:
    """(candidate binaries in order of preference, pinned sha256 of the data team's binary)."""
    cands = [os.environ.get("SCIENCECLAW_FFMPEG", "")]
    pinned: str | None = None
    try:
        cfg = read_json(base / "loader-config.json")
        cands.append(str(cfg.get("ffmpeg", "")))
        pinned = cfg.get("ffmpeg_sha256") or None
    except (OSError, ValueError, AttributeError):
        pass
    cands.append(shutil.which("ffmpeg") or "")
    try:
        import imageio_ffmpeg  # type: ignore
        cands.append(imageio_ffmpeg.get_ffmpeg_exe())
    except (ImportError, RuntimeError):
        pass
    return cands, pinned


def find_ffmpeg(base: Path) -> tuple[str | None, str | None]:
    """First executable candidate (``$SCIENCECLAW_FFMPEG`` > data-team binary > PATH > imageio_ffmpeg) and the pinned
    sha256 recorded by the data team (``None`` if there is no ``loader-config.json``)."""
    cands, pinned = _ffmpeg_candidates(base)
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c, pinned
    return None, pinned


def decode_stream(ffmpeg: str, path: Path, index: int, expect_sha256: str | None = None) -> np.ndarray:
    """One audio stream of ``path`` as float32 PCM (n, 2) at the native rate; optionally check the PCM sha256."""
    cmd = [ffmpeg, "-nostdin", "-v", "error", "-xerror", "-err_detect", "explode", "-i", str(path),
           "-map", f"0:a:{index}", "-c:a", "pcm_f32le", "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True, timeout=600)
    if out.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path.name} stream {index}: {out.stderr.decode(errors='replace')[-300:]}")
    if expect_sha256 and hashlib.sha256(out.stdout).hexdigest() != expect_sha256:
        raise RuntimeError(f"decoded PCM of {path.name} stream {index} does not match the data team's "
                           f"decoded_pcm_sha256 {expect_sha256[:12]}...")
    if len(out.stdout) % 8:
        raise RuntimeError(f"{path.name} stream {index}: decoded byte count {len(out.stdout)} is not a stereo f32 stream")
    return np.frombuffer(out.stdout, dtype="<f4").reshape(-1, 2)


# ================================================================================================ pools
@dataclass(frozen=True)
class _Track:
    track_id: str                                   # "musdb18/<official split>/<title>"
    pool: str                                       # src | val | id | train | dev
    role: str                                       # data-team role: source | validation | ID | reserve
    path: Path
    duration_s: float
    streams: dict                                   # name -> (audio stream index, expected decoded-PCM sha256)
    slots: tuple                                    # 6-s slot numbers (excerpts) drawn from this track


@dataclass(frozen=True)
class _Excerpt:
    item_id: str                                    # "<track_id>@<start_s>"
    track: _Track
    k: int                                          # index into track.slots


@dataclass
class _MusData:
    tracks: dict[str, _Track]
    pools: dict[str, list[str]]                     # src / val / id / ood / train / dev -> excerpt ids
    excerpts: dict[str, _Excerpt]
    receipt: dict
    role_sha256: dict[str, str]
    track_counts: dict[str, int]


def _slots_of(track_id: str, dur_s: float, want: int) -> tuple[int, ...]:
    nslots = int(max(dur_s - 0.1, 0.0) // EXCERPT_S)            # small safety margin against rounded durations
    ranked = hash_rank([str(i) for i in range(nslots)], f"{CODE}|{PARTITION_SEED}|slots|{track_id}")[:want]
    return tuple(sorted(int(s) for s in ranked))


def _read_role(base: Path, role: str, manifest: dict) -> list[dict]:
    entry = manifest["roles"][role]
    f = base / entry["path"]
    if file_sha256(f) != entry["sha256"]:
        raise ValueError(f"{f} does not match the sha256 recorded in sampling-manifest.json")
    recs = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(recs) != int(entry["count"]):
        raise ValueError(f"{f}: {len(recs)} records, manifest says {entry['count']}")
    return recs


def _load(root: Path) -> _MusData:
    base = root / DATASET_DIR / DATA_VERSION
    manifest = read_json(base / "sampling-manifest.json")
    recs = {role: _read_role(base, role, manifest) for role in ("source", "validation", "ID", "reserve", "OOD")}
    pool_of: dict[str, str] = {}
    for role, pool in ROLE_POOL.items():
        pool_of.update({r["id"]: pool for r in recs[role]})
    reserve_ids = [r["id"] for r in recs["reserve"]]
    for pool, ids in partition_ids(reserve_ids, RESERVE_SPLIT, f"{CODE}|{PARTITION_SEED}|reserve").items():
        pool_of.update({i: pool for i in ids})
    all_recs = {r["id"]: (role, r) for role, rs in recs.items() for r in rs}
    if len(all_recs) != sum(len(rs) for rs in recs.values()):
        raise ValueError("a track appears in more than one role (roles must be disjoint)")
    tracks: dict[str, _Track] = {}
    pools: dict[str, list[str]] = {k: [] for k in ("src", "val", "id", "ood", "train", "dev")}
    excerpts: dict[str, _Excerpt] = {}
    for tid in sorted(all_recs):
        role, r = all_recs[tid]
        if role == "OOD":                                       # empty in full_v1; never silently repurposed
            continue
        pool = pool_of[tid]
        streams = {"mixture": (int(r["input"]["index"]), r["input"]["decoded_pcm_sha256"])}
        streams.update({t: (int(r["labels"][t]["index"]), r["labels"][t]["decoded_pcm_sha256"]) for t in TARGETS})
        dur = float(r["duration_seconds"])
        tr = _Track(tid, pool, role, base / r["path"], dur, streams, _slots_of(tid, dur, EXCERPTS_PER_TRACK[pool]))
        tracks[tid] = tr
        for k, s in enumerate(tr.slots):
            ex = _Excerpt(f"{tid}@{s * EXCERPT_S:g}", tr, k)
            excerpts[ex.item_id] = ex
            pools[pool].append(ex.item_id)
    counts = {p: sum(t.pool == p for t in tracks.values()) for p in pools}
    return _MusData(tracks=tracks, pools=pools, excerpts=excerpts,
                    receipt=receipt_summary(base / "full-receipt.json"),
                    role_sha256={r: manifest["roles"][r]["sha256"] for r in manifest["roles"]}, track_counts=counts)


class _ExcerptStore:
    """Decoded 22.05 kHz excerpts per track: memory LRU + ``cache/tasks/FoR36/excerpts_v1_*/<key>.npz``."""

    def __init__(self, ffmpeg: str | None, pinned_sha256: str | None) -> None:
        self.ffmpeg = ffmpeg
        self._pinned = pinned_sha256
        self._verify = Lazy(self._binary_is_pinned)
        self._lock = threading.Lock()
        self._mem: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._dir = cache_dir(CODE) / f"excerpts_{_CACHE_VERSION}_{SR}_{EXCERPT_S:g}s"
        self._dir.mkdir(parents=True, exist_ok=True)

    def _binary_is_pinned(self) -> bool:
        return bool(self.ffmpeg and self._pinned and file_sha256(self.ffmpeg) == self._pinned)

    @property
    def verified(self) -> bool:
        """True when decoding uses the data team's pinned ffmpeg, so decoded PCM is checked bit-exactly."""
        return self._verify.get()

    def get(self, tr: _Track) -> tuple[np.ndarray, np.ndarray]:
        """(mixtures (k, n, 2), stems (k, 4, n, 2)) float32 of the excerpts ``tr.slots`` of the track."""
        with self._lock:
            hit = self._mem.get(tr.track_id)
            if hit is not None:
                self._mem.move_to_end(tr.track_id)
                return hit
            n_len = int(EXCERPT_S * SR)
            tag = "v" if self.verified else "u"
            f = self._dir / f"{sha_hex(CODE, _CACHE_VERSION, tr.track_id, tr.slots, SR, EXCERPT_S)[:24]}_{tag}.npz"
            val: tuple[np.ndarray, np.ndarray] | None = None
            if f.exists():
                try:
                    with np.load(f) as z:
                        mix, stems = z["mix"], z["stems"]
                    if mix.shape == (len(tr.slots), n_len, 2) and stems.shape == (len(tr.slots), 4, n_len, 2):
                        val = (mix, stems)
                except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                    val = None                                  # corrupt / partial file: rebuild below
            if val is None:
                val = self._decode(tr)
                tmp = f.with_name(f.stem + f".tmp{os.getpid()}-{threading.get_ident()}.npz")
                np.savez(tmp, mix=val[0], stems=val[1])
                tmp.replace(f)
            self._mem[tr.track_id] = val
            while len(self._mem) > _MEM_TRACKS:
                self._mem.popitem(last=False)
            return val

    def _stream_excerpts(self, tr: _Track, name: str, check: bool) -> np.ndarray:
        """Excerpts (k, n, 2) float32 at ``SR`` of one stream: native decode, optional hash check, decimation."""
        from scipy.signal import resample_poly
        n_len = int(EXCERPT_S * SR)
        idx, sha = tr.streams[name]
        x = decode_stream(self.ffmpeg, tr.path, idx, sha if check else None)
        out = np.empty((len(tr.slots), n_len, 2), dtype=np.float32)
        for j, s in enumerate(tr.slots):
            a = s * n_len * DECIM
            b = a + n_len * DECIM
            if b > x.shape[0]:
                raise ValueError(f"{tr.track_id}@{s * EXCERPT_S:g}: excerpt ends at native sample {b} but the "
                                 f"{name} stream has {x.shape[0]}")
            lo, hi = max(a - _RESAMPLE_PAD, 0), min(b + _RESAMPLE_PAD, x.shape[0])
            y = resample_poly(x[lo:hi], 1, DECIM, axis=0)
            out[j] = y[(a - lo) // DECIM:(a - lo) // DECIM + n_len]
        return out

    def _decode(self, tr: _Track) -> tuple[np.ndarray, np.ndarray]:
        if not self.ffmpeg:
            raise FileNotFoundError("ffmpeg not found (set SCIENCECLAW_FFMPEG)")
        check = self.verified
        names = ("mixture", *TARGETS)
        with ThreadPoolExecutor(max_workers=len(names)) as pool:            # five ffmpeg processes in parallel
            parts = list(pool.map(lambda nm: self._stream_excerpts(tr, nm, check), names))
        return parts[0], np.stack(parts[1:], axis=1)


# ================================================================================================ adapter
class MusdbAdapter:
    """MUSDB18 four-stem separation adapter (``bench.task.TaskAdapter`` protocol)."""

    discipline = CODE
    name = "MUSDB18 four-stem separation"
    family = FAMILY
    metric = "mean target-median SDR (dB)"
    direction = "max"
    task_type = "source_separation"

    def __init__(self, data_root: str | os.PathLike | None = None, accept_margin_db: float = ACCEPT_MARGIN_DB,
                 budget: Budget | None = None, ffmpeg: str | os.PathLike | None = None) -> None:
        self.root = resolve_data_root(data_root)
        self.base = self.root / DATASET_DIR / DATA_VERSION
        self.accept_margin_db = float(accept_margin_db)
        self.budget = budget
        found, self._pinned_sha256 = find_ffmpeg(self.base)
        self._ffmpeg = str(ffmpeg) if ffmpeg else found
        self._data = Lazy(lambda: _load(self.root))
        self._store = Lazy(lambda: _ExcerptStore(self._ffmpeg, self._pinned_sha256))

    # ------------------------------------------------------------------------------------------ protocol
    def available(self) -> tuple[bool, str]:
        if not (self.base / "sampling-manifest.json").is_file():
            return False, (f"FoR36 full dataset not found: {self.base} lacks sampling-manifest.json (expected the "
                           "data team's full_v1: loader-config.json, sampling-manifest.json, roles/*.jsonl, original/)")
        if not self._ffmpeg:
            return False, ("ffmpeg not found (needed to decode the AAC .stem.mp4 files; set SCIENCECLAW_FFMPEG or "
                           "install the binary named in full_v1/loader-config.json)")
        try:
            probe = subprocess.run([self._ffmpeg, "-version"], capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as ex:
            return False, f"ffmpeg {self._ffmpeg} cannot be executed: {type(ex).__name__}: {ex}"
        if probe.returncode != 0:
            return False, f"ffmpeg {self._ffmpeg} exits with status {probe.returncode} on -version"
        try:
            data = self._data.get()
        except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as ex:
            return False, f"FoR36 data unreadable: {type(ex).__name__}: {ex}"
        missing = [t.track_id for t in data.tracks.values() if not t.path.is_file()]
        if missing:
            return False, f"{len(missing)} of {len(data.tracks)} track files missing under {self.base} (first: {missing[0]})"
        short = {k: (data.track_counts.get(k, 0), need) for k, need in REQUIRED_TRACKS.items()
                 if data.track_counts.get(k, 0) < need}
        if short:
            return False, f"FoR36 pools below the expected full_v1 sizes (tracks: have, need): {short}"
        sizes = {k: len(v) for k, v in data.pools.items()}
        return True, (f"MUSDB18 full_v1: tracks {data.track_counts}, 6-s excerpts {sizes} (ood empty: no independent "
                      f"collection); ffmpeg {'pinned (decoded PCM verified)' if self._store.get().verified else 'unpinned'}")

    def build_episodes(self, split: str, n: int, seed: int, items_per_episode: int = 16) -> list[Episode]:
        check_split(split)
        if split == "ood":                       # full_v1 has no OOD collection (roles/OOD.jsonl is empty)
            return []
        data = self._data.get()
        rng = episode_rng(CODE, split, seed)
        blocks = draw_blocks({"all": data.pools[split]}, {"all": items_per_episode}, int(n), rng, f"{CODE}/{split}",
                             cycle=(split == "src"))
        return [self._episode(data, split, int(seed), k, ids) for k, ids in enumerate(blocks)]

    def pooled_metric(self, per_episode: list[dict]) -> float | None:
        """Mean target-median SDR over all items (invalid outputs contribute the reference item scores)."""
        rows: list[list[float]] = []
        for p in per_episode:
            if not p:
                continue
            sc = p.get("item_sdr") if p.get("item_sdr") is not None else p.get("ref_item_sdr")
            if sc is None:
                continue
            rows.extend([[np.nan if v is None else float(v) for v in r] for r in sc])
        return aggregate(np.asarray(rows, dtype=float)) if rows else None

    # ------------------------------------------------------------------------------------------ episodes
    def _clip(self, ex: _Excerpt) -> tuple[np.ndarray, np.ndarray]:
        """(mixture (n, 2), stems (4, n, 2)) copies of one excerpt."""
        mix, stems = self._store.get().get(ex.track)
        return mix[ex.k].copy(), stems[ex.k].copy()

    def _episode(self, data: _MusData, split: str, seed: int, k: int, ids: list[str]) -> Episode:
        n = len(ids)
        n_len = int(EXCERPT_S * SR)
        win = int(WIN_S * SR)
        erng = episode_rng(CODE, split, seed, k, "visible")
        tr_pool, dv_pool = data.pools["train"], data.pools["dev"]
        train_ids = [tr_pool[j] for j in sorted(erng.permutation(len(tr_pool))[:N_TRAIN])]
        dev_ids = [dv_pool[j] for j in sorted(erng.permutation(len(dv_pool))[:N_DEV])]
        ex = [data.excerpts[i] for i in ids]
        nt, nd = len(train_ids), len(dev_ids)

        def load_train(inputs: dict, config: dict) -> dict:
            clips = [self._clip(data.excerpts[i]) for i in train_ids]
            tracks = sorted({data.excerpts[i].track.track_id for i in train_ids})
            return {"mixtures": np.stack([c[0] for c in clips]) if clips else np.zeros((0, n_len, 2), np.float32),
                    "stems": np.stack([c[1] for c in clips]) if clips else np.zeros((0, 4, n_len, 2), np.float32),
                    "track_ids": [f"track{tracks.index(data.excerpts[i].track.track_id)}" for i in train_ids],
                    "sample_rate": SR}

        def load_dev_inputs(inputs: dict, config: dict) -> dict:
            clips = [self._clip(data.excerpts[i]) for i in dev_ids]
            return {"dev_mixtures": np.stack([c[0] for c in clips]) if clips else np.zeros((0, n_len, 2), np.float32),
                    "sample_rate": SR}

        def score_dev(inputs: dict, config: dict) -> dict:
            est, why = _as_estimates(inputs.get("dev_estimates"), nd, n_len)
            if est is None:
                raise ValueError(f"score_dev: {why}")
            clips = [self._clip(data.excerpts[i]) for i in dev_ids]
            per = np.array([item_scores(c[1], est[j]) for j, c in enumerate(clips)])
            ref = np.array([item_scores(c[1], np.repeat(c[0][None], 4, axis=0)) for c in clips])
            return {"dev_sdr": aggregate(per), "dev_reference_sdr": aggregate(ref),
                    "dev_target_median_sdr": {t: float(np.nanmedian(per[:, j])) if np.any(~np.isnan(per[:, j])) else None
                                              for j, t in enumerate(TARGETS)}, "n_dev": nd}

        def load_eval_inputs(inputs: dict, config: dict) -> dict:
            return {"mixtures": np.stack([self._clip(e)[0] for e in ex]), "sample_rate": SR}

        def separate_htdemucs(inputs: dict, config: dict) -> dict:
            """Run the staged htdemucs checkpoint on mixtures only; target stems never enter this call."""
            from scilib import audiosep_pretrained

            mixtures = inputs.get("mixtures")
            if mixtures is None:
                raise ValueError("separate_htdemucs needs mixtures from load_eval_inputs or load_dev_inputs")
            cfg = dict(config or {})
            # The fine-tuned official bag is the strongest staged route. Keep
            # the required formal tool name, but make an omitted model select
            # the same route so an agent cannot silently fall back to the
            # weaker baseline after a one-step tool call.
            model = str(cfg.get("model", "htdemucs_ft"))
            shifts = int(cfg.get("shifts", 0))
            overlap = float(cfg.get("overlap", 0.25))
            est = audiosep_pretrained.separate_pretrained(mixtures, model=model, sample_rate=SR,
                                                           shifts=shifts, overlap=overlap, device="auto")
            return {"estimates": np.asarray(est, dtype=np.float32)}

        def separate_htdemucs_ft(inputs: dict, config: dict) -> dict:
            """Run the official fine-tuned four-model HTDemucs ensemble on mixtures only."""
            from scilib import audiosep_pretrained
            mixtures = inputs.get("mixtures")
            if mixtures is None:
                raise ValueError("separate_htdemucs_ft needs mixtures from load_eval_inputs or load_dev_inputs")
            est = audiosep_pretrained.separate_pretrained(mixtures, model="htdemucs_ft", sample_rate=SR,
                                                           shifts=0, overlap=0.25, device="auto")
            return {"estimates": np.asarray(est, dtype=np.float32)}

        def separate_scnet(inputs: dict, config: dict) -> dict:
            """Run the public frozen four-source MIMO-SCNet route on mixtures only."""
            from scilib import scnet_pretrained
            mixtures = inputs.get("mixtures")
            if mixtures is None:
                raise ValueError("separate_scnet needs mixtures from load_eval_inputs or load_dev_inputs")
            cfg = dict(config or {})
            est = scnet_pretrained.separate_pretrained(
                mixtures, model="mimo_scnet_small", sample_rate=SR, device="auto",
                iterations=int(cfg.get("iterations", 2)), batch_size=int(cfg.get("batch_size", 2)),
            )
            return {"estimates": np.asarray(est, dtype=np.float32)}

        arr = lambda shape, d: PortSchema("array", shape, unit="1", dtype="float32", description=d)  # noqa: E731
        tools = [
            ToolSpec("load_train", f"{nt} visible training excerpts: stereo mixtures (m, samples, 2) and their four "
                                   "stems (m, 4, samples, 2) in target order vocals, drums, bass, other; linear PCM "
                                   f"amplitude at {SR} Hz; track_ids: one label per excerpt, equal labels = excerpts of the "
                                   "same track.", {},
                     {"mixtures": arr((nt, n_len, 2), "mixtures"), "stems": arr((nt, 4, n_len, 2), "stems"),
                      "track_ids": PortSchema("list", (nt,), description="track label per training excerpt"),
                      "sample_rate": PortSchema("number", unit="Hz", dtype="int")}, load_train),
            ToolSpec("load_dev_inputs", f"{nd} dev mixtures (stems withheld, see score_dev).", {},
                     {"dev_mixtures": arr((nd, n_len, 2), "mixtures"),
                      "sample_rate": PortSchema("number", unit="Hz", dtype="int")}, load_dev_inputs),
            ToolSpec("score_dev", "Mean target-median SDR (dB) of dev_estimates (nd, 4, samples, 2) with the "
                                  "evaluation metric, plus the reference SDR.",
                     {"dev_estimates": arr((nd, 4, n_len, 2), "estimated stems")},
                     {"dev_sdr": PortSchema("number", unit="dB"), "dev_reference_sdr": PortSchema("number", unit="dB"),
                      "dev_target_median_sdr": PortSchema("dict", unit="dB"), "n_dev": PortSchema("number", dtype="int")},
                     score_dev),
            ToolSpec("load_eval_inputs", f"The {n} evaluation mixtures, (n, samples, 2), linear PCM at {SR} Hz, in "
                                         "output order.", {},
                     {"mixtures": arr((n, n_len, 2), "mixtures"), "sample_rate": PortSchema("number", unit="Hz", dtype="int")},
                     load_eval_inputs),
            ToolSpec("separate_htdemucs", "Run a staged frozen Demucs checkpoint on supplied mixtures. The tool sees "
                                          "mixtures only and returns estimates in the required vocals/drums/bass/other "
                                          "order. Config model may be htdemucs_ft (default fine-tuned bag), htdemucs "
                                          "(baseline), or mdx_extra (official MDX bag); all weights are local and no "
                                          "task fitting is performed. The checkpoint provenance and any MUSDB18-train "
                                          "overlap remain disclosed.",
                     {"mixtures": arr((n, n_len, 2), "mixture-only stereo PCM")},
                     {"estimates": arr((n, 4, n_len, 2), "estimated sources in target order")},
                     separate_htdemucs,
                     config_doc="model=htdemucs_ft|htdemucs|mdx_extra (default htdemucs_ft); shifts=0; overlap=0.25; uses staged local/remote weights."),
        ]
        try:
            from scilib import audiosep_pretrained
            ft_available = bool(audiosep_pretrained.available("htdemucs_ft"))
        except (ImportError, RuntimeError, OSError):
            ft_available = False
        if ft_available:
            tools.append(ToolSpec("separate_htdemucs_ft",
                                  "Run the official fine-tuned four-model HTDemucs ensemble on mixture-only input; "
                                  "returns vocals/drums/bass/other estimates in the required order.",
                                  {"mixtures": arr((n, n_len, 2), "mixture-only stereo PCM")},
                                  {"estimates": arr((n, 4, n_len, 2), "estimated sources in target order")},
                                  separate_htdemucs_ft,
                                  config_doc="model=htdemucs_ft; shifts=0; overlap=0.25; frozen official weights."))
        try:
            from scilib import scnet_pretrained
            scnet_available = bool(scnet_pretrained.available("mimo_scnet_small"))
        except (ImportError, RuntimeError, OSError):
            scnet_available = False
        if scnet_available:
            tools.append(ToolSpec("separate_scnet",
                                  "Run the public frozen MIMO-SCNet four-source model on mixture-only input; "
                                  "returns vocals/drums/bass/other estimates in the required order. This is an "
                                  "engineering candidate selected on visible dev and uses no task fitting.",
                                  {"mixtures": arr((n, n_len, 2), "mixture-only stereo PCM")},
                                  {"estimates": arr((n, 4, n_len, 2), "estimated sources in target order")},
                                  separate_scnet,
                                  config_doc="model=mimo_scnet_small; iterations=1|2; batch_size>=1; frozen public weights."))

        def c_shape(y: Any, trace: Trace | None) -> tuple[bool, str]:
            est, why = _as_estimates(y, n, n_len)
            return est is not None, (why or f"array of shape ({n}, 4, {n_len}, 2)")

        def c_finite(y: Any, trace: Trace | None) -> tuple[bool, str]:
            est, why = _as_estimates(y, n, n_len)
            if est is None:
                return False, why
            bad = int((~np.isfinite(est)).sum())
            return bad == 0, "all samples finite" if bad == 0 else f"{bad} non-finite samples"

        def c_silent(y: Any, trace: Trace | None) -> tuple[bool, str]:
            est, why = _as_estimates(y, n, n_len)
            if est is None:
                return False, why
            nw = n_len // win
            fr = est[:, :, : nw * win].reshape(n, 4, nw, win, 2)
            silent = np.all(fr.sum(axis=4) == 0, axis=3)
            cnt = int(silent.sum())
            return cnt == 0, "no silent estimate window" if cnt == 0 else f"{cnt} (item, target, 1-s window) silent"

        constraints = [
            ConstraintSpec("output_shape", f"y is a float array of shape ({n}, 4, {n_len}, 2): item, target (vocals, "
                                           "drums, bass, other), sample, channel", c_shape),
            ConstraintSpec("finite", "every sample of y is finite", c_finite),
            ConstraintSpec("no_silent_window", "no estimated target is exactly zero over a whole 1-second window",
                           c_silent),
        ]

        def reference_items() -> np.ndarray:
            rows = []
            for e in ex:                         # clips are not retained: an Episode may outlive many others
                m, s = self._clip(e)
                rows.append(item_scores(s, np.repeat(m[None], 4, axis=0)))
            return np.array(rows)

        ref_cache = Lazy(reference_items)

        def evaluate(yv: Any, trace: Trace | None) -> EvalResult:
            ref_items = ref_cache.get()
            ref = aggregate(ref_items)
            payload: dict[str, Any] = {"item_ids": list(ids), "item_sdr": None,
                                       "ref_item_sdr": _nan_to_none(ref_items)}
            est, why = _as_estimates(yv, n, n_len)
            if est is not None and not np.all(np.isfinite(est)):
                est, why = None, "non-finite samples"
            if est is None:
                return EvalResult(metrics={"reference_sdr": ref if ref is not None else float("nan")}, primary=None,
                                  direction="max", accepted=False,
                                  details={"reference": ref, "norm_score": 0.0, "pooled_payload": payload,
                                           "invalid": why})
            per = np.array([item_scores(self._clip(e)[1], est[j]) for j, e in enumerate(ex)])
            score = aggregate(per)
            payload["item_sdr"] = _nan_to_none(per)
            metrics = {"sdr": score if score is not None else float("nan"),
                       "reference_sdr": ref if ref is not None else float("nan")}
            for j, t in enumerate(TARGETS):
                v = per[:, j][~np.isnan(per[:, j])]
                metrics[f"sdr_{t}"] = float(np.median(v)) if v.size else float("nan")
            ok = score is not None and ref is not None and score >= ref + self.accept_margin_db
            return EvalResult(metrics=metrics, primary=score, direction="max", accepted=bool(ok),
                              details={"reference": ref, "norm_score": norm_score_db(score, ref),
                                       "pooled_payload": payload})

        objective = (
            f"Separate {n} stereo music excerpts ({EXCERPT_S:g} s, {SR} Hz, linear PCM) into four sources: vocals, "
            "drums, bass and other (all remaining instruments); the four sources sum approximately to the mixture. "
            f"Deliverable y: a float array of shape ({n}, 4, {n_len}, 2) = (item, target in the order "
            "vocals/drums/bass/other, sample, channel), on the same amplitude scale and time axis as the mixtures of "
            "load_eval_inputs. Metric: BSSEval-v4 SDR in dB (museval v4: 1-second windows, per-target median over "
            "windows, median over items, mean over the four targets); higher is better. Visible training excerpts "
            "with their stems, dev mixtures scored by score_dev and the evaluation mixtures are provided by the tools. "
            "The separate_scnet tool is a mixture-only frozen MIMO-SCNet candidate when staged; choose it first on "
            "visible dev if available, then use separate_htdemucs with model=htdemucs_ft as the Demucs fallback. The "
            "separate_htdemucs tool is a mixture-only frozen Demucs SOTA checkpoint path; choose model=htdemucs_ft "
            "for the fine-tuned ensemble when its staged weights are available; htdemucs / mdx_extra are explicit "
            "fallbacks. The explicit separate_htdemucs_ft tool is an engineering alias; for formal runs call "
            "separate_htdemucs with config model=htdemucs_ft so the required formal tool reference is recorded. Its MUSDB18 train-data overlap is "
            "reported as model provenance and does not change the primary scorer. In a tool-on run, call "
            "separate_htdemucs for the evaluation mixtures before considering the classical SoftMask route; the "
            "formal tool-on evidence must show the htdemucs task-tool call. The tool returns the required estimate "
            "array under its `estimates` output port; submit that array directly as y, without replacing it with a "
            "SoftMask estimate. "
            + scilib.describe("audiosep") + scilib.describe_extra("audiosep_pretrained")
        )
        return Episode(
            id=episode_id(CODE, split, seed, k), discipline=CODE, family=FAMILY, split=split,
            task_type=self.task_type, objective=objective,
            required_output=PortSchema("array", (n, 4, n_len, 2), unit="1", dtype="float32",
                                       description="estimated stems (item, target, sample, channel)"),
            tools=tools, constraints=constraints,
            budget=self.budget or Budget(max_steps=12, max_policy_tokens=200_000, max_wall_s=2400.0,
                                         max_node_s=600.0, max_llm_items=n),
            lineage={"dataset": "MUSDB18 (full_v1, AAC STEMS)", "pool": "iid", "ood_kind": None,
                     "role": {"src": "source", "val": "validation", "id": "ID"}[split],
                     "item_ids": list(ids), "tracks": sorted({e.track.track_id for e in ex}),
                     "train_item_ids": train_ids, "dev_item_ids": dev_ids, "sample_rate": SR,
                     "excerpt_s": EXCERPT_S, "seed": seed, "index": k, "partition_seed": PARTITION_SEED,
                     "visible_source": "reserve tracks (16 train / 6 dev), track-disjoint from every evaluated item",
                     "decode_verified": self._store.get().verified, "role_sha256": dict(data.role_sha256),
                     "receipt": data.receipt},
            acceptance=f"mean target-median SDR >= reference (mixture as estimate) + {self.accept_margin_db:g} dB",
            tolerance={"rtol": 1e-4, "atol": 1e-6},
            tags=["audio", "music", "source_separation", "signal_processing"],
            metric=self.metric, direction=self.direction, n_items=n,
            _evaluate=evaluate, _dev_evaluate=None,
        )


def _as_estimates(y: Any, n: int, n_len: int) -> tuple[np.ndarray | None, str]:
    if y is None:
        return None, "no output"
    try:
        est = np.asarray(y, dtype=np.float64)
    except (TypeError, ValueError) as ex:
        return None, f"output is not numeric: {type(ex).__name__}: {ex}"
    if est.shape != (n, 4, n_len, 2):
        return None, f"output has shape {list(est.shape)}, required {[n, 4, n_len, 2]}"
    return est, ""


def _nan_to_none(a: np.ndarray) -> list[list[float | None]]:
    return [[None if np.isnan(v) else float(v) for v in row] for row in np.asarray(a, dtype=float)]


Adapter = MusdbAdapter
