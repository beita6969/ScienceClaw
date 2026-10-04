"""FoR40 Engineering — DCASE 2024 Task 2 first-shot unsupervised anomalous sound detection (ASD),
metric: official DCASE score (max).

Data (data team delivery ``reconstructed_v3``, design ``FoR40.v3``; natural role counts source 210 / validation 210 /
ID 140 / source_fit_support 168 / OOD 0): 16 kHz mono 16-bit PCM WAV clips (10 s, 12 s for ToyCar / ToyTrain) of the seven
development machine types of the official Zenodo record 10902294 (ToyCar, ToyTrain, bearing, fan, gearbox, slider,
valve; one section each). Everything is read from the frozen role files, nothing is discovered or re-partitioned:

* ``roles/source.json`` / ``roles/val.json`` / ``roles/id.json`` = the ``src`` / ``val`` / ``id`` splits. Each file holds
  seven **native cohorts** (one per machine type, both domains, both classes): 30 probe clips for src and val
  (10 source-normal, 10 source-anomaly, 5 target-normal, 5 target-anomaly) and 20 for id (5 per stratum). The three role
  files are clip- and PCM-disjoint (re-checked at load time), so ``src`` / ``val`` / ``id`` episodes are item-disjoint
  for **any** combination of seeds. ``partition_seed`` only salts the episode draws, never the partition.
* ``source-normal-support.json`` = ``source_fit_support``: 168 official *source-domain normal training* clips, 24 per
  machine type. They are the visible training data (``load_train``) of every episode of that machine type in every
  split, i.e. what a detector may fit on. They are disjoint (files and PCM) from all 560 probe clips, are not
  evaluation items (not in ``lineage["item_ids"]``, listed in ``lineage["train_item_ids"]``) and contain no
  target-domain clip: target-domain behaviour has to be inferred from the shift itself, as in the challenge.
* **OOD is empty.** Both official domains occur inside source and validation, and no unseen-machine / cross-dataset pool
  was delivered, so ``build_episodes("ood", ...)`` returns ``[]`` (``SplitPlan`` records the shortfall as a warning,
  ``pooled_scores`` of an empty split is None) and ``available()`` requires no OOD capacity. No proxy OOD is invented.

* **Item** = one probe clip of one machine; the system outputs an anomaly score (higher = more anomalous). Item ids are
  the opaque one-time public handles of the role files (``dcase2024t2/v3/<machine>/<public_id>``), never the
  label-bearing official file names.
* **Episode** = ``items_per_episode`` probe clips of ONE machine type (the official score is per machine type and
  section): ``items/4`` from each (domain, condition) stratum of that machine's native cohort of the split, plus the
  machine's 24 normal source-domain training clips (with the official attribute strings) as visible data. The domain of
  the probe clips is not given (as in the challenge). A 16-clip episode takes 4 of the 5 target-normal / target-anomaly
  clips of a cohort, so each cohort supports ONE such episode per split (src 7, val 7, id 7 episodes: the paper plan
  7 / 2 / 4 fits); ``items_per_episode = 8`` gives two per cohort, values above 20 are impossible. Machines are visited
  round-robin in a seeded order and clips are drawn without replacement (prefix-stable in ``n``).
* **Metric** (official evaluator ``dcase2024_task2_evaluator.py``): for a machine section, AUC(source) over source
  normals + all anomalies, AUC(target) over target normals + all anomalies, pAUC = ``roc_auc_score(...,
  max_fpr=0.1)`` over all clips; official score = harmonic mean of the three (values floored at float eps). The
  pooled metric groups the items of all episodes by machine and takes the harmonic mean over machines of the three
  values, as the official evaluator does over machine sections (for the complete native cohorts it equals the data
  team's ``runtime/dcase_native_v3`` primary; the tests reproduce those receipts).
* **Reference** (deterministic): 1-nearest-neighbour distance of the clip's standardized log-mel statistics (per-mel
  mean and std over frames; 128 mels, n_fft 1024, hop 512) to the machine's 24 training clips. **Acceptance:** official
  score >= reference + margin (default 0.02). No visible dev signal: in the first-shot setting only normal training
  clips exist (``_dev_evaluate`` is None, no ``score_dev`` tool).
* **Hard constraints:** 1-D vector of length ``items``; finite.

Every WAV is checked against the sha256 / frame count of its role record when it is read (fail closed).
``available()`` checks the role files against the frozen sha256 values, the uniqueness / disjointness invariants and
that every referenced file exists with the recorded size, then that the local cohorts support the default plan
(src 7, val 2, id 4 episodes of 16 clips); it needs numpy / scipy only (no audio library). The ``path`` fields of the
records are absolute paths of the data team's machine and are re-based below the configured data root.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import scilib

from ...core.schema import PortSchema
from ...core.trace import Trace
from ..task import Budget, ConstraintSpec, EvalResult, Episode, ToolSpec
from ._adapter_utils_for34_39_40_44_51 import (
    PARTITION_SEED, Lazy, PoolExhausted, as_float_vector, c_finite, c_vector, check_split, episode_id, episode_rng,
    norm_score, read_json, receipt_info, resolve_data_root,
)

CODE = "FoR40"
FAMILY = "Engineering & computing"
DATASET_DIR = "for40-dcase2024-task2"
DATA_VERSION = "reconstructed_v3"
DESIGN_ID = "FoR40.v3"
DEV_MACHINES = ("ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve")
STRATA = (("source", 0), ("source", 1), ("target", 0), ("target", 1))       # (domain, label) ; label 1 = anomaly
ROLE_FILES = {"src": "roles/source.json", "val": "roles/val.json", "id": "roles/id.json"}   # adapter split -> role file
SUPPORT_FILE = "source-normal-support.json"
FROZEN_SHA256 = {                                                             # catalog / FoR40.v3.json (frozen design)
    "roles/source.json": "cd337ac8016fee714bcc88d46c40fe4d8e1874fb6d202da0172634e61d21d916",
    "roles/val.json": "889c44989d75cdf274a5b8eff2ea1bbfd49170de6a63188482f9d627ed8cb219",
    "roles/id.json": "9f9b25e106ef18362f1e2d1e1512e57796becb6deaef8a990e331cb25257347f",
    SUPPORT_FILE: "1a21f239088e7946df41285d43a28031740cc56f1145541a857d047b67689443",
}
MAX_FPR = 0.1
EPS = float(np.finfo(float).eps)
DEFAULT_PLAN = {"src": 7, "val": 2, "id": 4}          # paper plan minus OOD (empty: no formal OOD in the v3 delivery)
DEFAULT_ITEMS = 16
MEL = {"n_fft": 1024, "hop": 512, "n_mels": 128}

_DEV_NAME = re.compile(r"^section_(\d+)_(source|target)_(train|test)_(normal|anomaly)_(\d+)(?:_(.*))?\.wav$")
_ROOT_MARK = DATASET_DIR + "/"


class DCASEDataError(ValueError):
    """The delivered DCASE files violate the contract the adapter relies on (fail closed)."""


@dataclass(frozen=True)
class Clip:
    item_id: str          # opaque: dcase2024t2/v3/<machine>/<public_id>
    machine: str
    section: str
    domain: str           # "source" | "target"
    label: int            # 0 normal, 1 anomaly
    attributes: str       # official operating-condition string (only shown for the normal training clips)
    path: Path
    sha256: str
    pcm_sha256: str
    nbytes: int
    frames: int
    sample_rate: int
    official_id: str      # label-bearing official file id; trusted side only, never shown to the policy


@dataclass
class _Cohort:
    native_id: str
    machine: str
    payload_sha256: str
    clips: dict[str, Clip]
    strata: dict[tuple[str, int], list[str]]      # sorted item ids per stratum


@dataclass
class _Design:
    cohorts: dict[str, dict[str, _Cohort]] = field(default_factory=dict)      # split -> machine -> cohort
    support: dict[str, list[Clip]] = field(default_factory=dict)              # machine -> normal source training clips
    file_sha256: dict[str, str] = field(default_factory=dict)                 # role / support file -> sha256

    def clips(self) -> list[Clip]:
        out = [c for d in self.cohorts.values() for co in d.values() for c in co.clips.values()]
        return out + [c for cs in self.support.values() for c in cs]


# ----------------------------------------------------------------------------------------------- audio
def parse_wav(data: bytes) -> tuple[int, np.ndarray]:
    """(sample_rate, float32 waveform in [-1, 1]) of WAV bytes via scipy.io.wavfile (16-bit PCM, first channel)."""
    from scipy.io import wavfile
    sr, x = wavfile.read(io.BytesIO(data))
    if x.ndim > 1:
        x = x[:, 0]
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    else:
        x = x.astype(np.float32)
    return int(sr), x


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    """(sample_rate, float32 waveform in [-1, 1]) of a WAV file (see :func:`parse_wav`)."""
    return parse_wav(Path(path).read_bytes())


def read_clip(clip: Clip) -> np.ndarray:
    """The clip's waveform, verified against the sha256 / frame count / sample rate of its role record."""
    raw = clip.path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != clip.sha256:
        raise DCASEDataError(f"{clip.machine}: WAV content changed since the frozen role file was written "
                             f"(sha256 mismatch for handle {clip.item_id.rsplit('/', 1)[-1][:12]})")
    sr, x = parse_wav(raw)
    if sr != clip.sample_rate or x.size != clip.frames:
        raise DCASEDataError(f"{clip.machine}: WAV shape ({sr} Hz, {x.size} samples) differs from the role record "
                             f"({clip.sample_rate} Hz, {clip.frames} samples)")
    return x


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    """Triangular HTK-mel filterbank (n_mels, n_fft//2 + 1), area-normalized (Slaney style)."""
    fmax = fmax or sr / 2.0
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)   # noqa: E731
    mel2hz = lambda m: 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)  # noqa: E731
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    fb = np.zeros((n_mels, freqs.size))
    for i in range(n_mels):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        up = (freqs - lo) / max(c - lo, 1e-9)
        down = (hi - freqs) / max(hi - c, 1e-9)
        fb[i] = np.maximum(0.0, np.minimum(up, down)) * (2.0 / max(hi - lo, 1e-9))
    return fb


def log_mel(wave: np.ndarray, sr: int, n_fft: int = 1024, hop: int = 512, n_mels: int = 128) -> np.ndarray:
    """Log-mel power spectrogram in dB, shape (frames, n_mels); Hann window, centred frames (reflect padding)."""
    w = np.asarray(wave, dtype=np.float64)
    pad = n_fft // 2
    w = np.pad(w, pad, mode="reflect") if w.size > pad else np.pad(w, pad)
    n_frames = 1 + (w.size - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    spec = np.abs(np.fft.rfft(w[idx] * np.hanning(n_fft)[None, :], axis=1)) ** 2
    mel = spec @ _mel_filterbank(sr, n_fft, n_mels).T
    return (10.0 * np.log10(np.maximum(mel, 1e-10))).astype(np.float32)


def official_scores(y_true: np.ndarray, domain: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """AUC(source), AUC(target), pAUC and their harmonic mean for one machine section (official evaluator rules)."""
    from scipy.stats import hmean
    from sklearn.metrics import roc_auc_score
    y_true, domain, score = np.asarray(y_true), np.asarray(domain), np.asarray(score, dtype=float)
    out = {}
    for name, d in (("auc_source", "source"), ("auc_target", "target")):
        sel = (domain == d) | (y_true != 0)
        out[name] = float(roc_auc_score(y_true[sel], score[sel]))
    out["pauc"] = float(roc_auc_score(y_true, score, max_fpr=MAX_FPR))
    out["official_score"] = float(hmean(np.maximum([out["auc_source"], out["auc_target"], out["pauc"]], EPS)))
    return out


# ----------------------------------------------------------------------------------------------- data loading
def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clip(root: Path, rec: dict, machine: str, where: str, train: bool) -> Clip:
    """Validated :class:`Clip` of one role / support record (``root`` = the dataset directory)."""
    try:
        official = str(rec["id"])
        public = str(rec["public_id"])
        rec_machine, section = str(rec["machine"]), str(rec["section"])
        domain, label = str(rec["domain"]), int(rec["label"])
        raw_path = str(rec["path"])
        sha, pcm = str(rec["sha256"]), str(rec["pcm_sha256"])
        nbytes, frames, sr, channels = int(rec["bytes"]), int(rec["frames"]), int(rec["sample_rate"]), int(rec["channels"])
    except (KeyError, TypeError, ValueError) as ex:
        raise DCASEDataError(f"{where}: malformed clip record ({type(ex).__name__}: {ex})") from ex
    if rec_machine != machine:
        raise DCASEDataError(f"{where}: clip of machine {rec_machine!r} inside the {machine!r} cohort")
    if domain not in ("source", "target") or label not in (0, 1) or channels != 1 or sr != 16000:
        raise DCASEDataError(f"{where}: unsupported record (domain {domain!r}, label {label}, {channels} channels, {sr} Hz)")
    cond = rec.get("condition")
    if cond is not None and (cond == "anomaly") != (label == 1):
        raise DCASEDataError(f"{where}: condition {cond!r} contradicts label {label}")
    m = _DEV_NAME.match(official.rsplit("/", 1)[-1])
    if not m:
        raise DCASEDataError(f"{where}: clip id is not an official development file name")
    _, fdom, fsplit, fcond, _, attrs = m.groups()
    if fdom != domain or (fcond == "anomaly") != (label == 1) or fsplit != ("train" if train else "test"):
        raise DCASEDataError(f"{where}: record fields contradict the official file name")
    if _ROOT_MARK not in raw_path:
        raise DCASEDataError(f"{where}: record path is not below a {DATASET_DIR} directory")
    rel = Path(raw_path.split(_ROOT_MARK, 1)[1])
    if rel.is_absolute() or ".." in rel.parts:
        raise DCASEDataError(f"{where}: suspicious path in record")
    return Clip(item_id=f"dcase2024t2/v3/{machine}/{public}", machine=machine, section=section, domain=domain,
                label=label, attributes=attrs or "", path=root / rel, sha256=sha, pcm_sha256=pcm, nbytes=nbytes,
                frames=frames, sample_rate=sr, official_id=official)


def load_design(root: Path, verify_frozen: bool = True) -> _Design:
    """Parse and validate the v3 role files below ``root`` (= ``<data root>/for40-dcase2024-task2``)."""
    base = root / DATA_VERSION
    if not base.is_dir():
        raise DCASEDataError(f"missing {base}")
    design = _Design()
    for rel in (*ROLE_FILES.values(), SUPPORT_FILE):
        f = base / rel
        if not f.is_file():
            raise DCASEDataError(f"missing {f}")
        design.file_sha256[rel] = _sha256_file(f)
        if verify_frozen and design.file_sha256[rel] != FROZEN_SHA256[rel]:
            raise DCASEDataError(f"{rel} differs from the frozen {DESIGN_ID} design (sha256 "
                                 f"{design.file_sha256[rel][:12]} vs {FROZEN_SHA256[rel][:12]})")
    for split, rel in ROLE_FILES.items():
        cohorts: dict[str, _Cohort] = {}
        for entry in read_json(base / rel):
            try:
                machine, native = str(entry["machine"]), str(entry["episode_id"])
                payload, recs = str(entry["payload_sha256"]), list(entry["probe_records"])
                declared = dict(entry.get("stratum_counts") or {})
            except (KeyError, TypeError, ValueError) as ex:
                raise DCASEDataError(f"{rel}: malformed cohort ({type(ex).__name__}: {ex})") from ex
            if machine in cohorts:
                raise DCASEDataError(f"{rel}: machine {machine} occurs in two cohorts")
            clips: dict[str, Clip] = {}
            for rec in recs:
                c = _clip(root, rec, machine, f"{rel}:{native}", train=False)
                if c.item_id in clips:
                    raise DCASEDataError(f"{rel}: duplicate clip handle in {native}")
                clips[c.item_id] = c
            strata = {st: sorted(i for i, c in clips.items() if (c.domain, c.label) == st) for st in STRATA}
            if declared and {f"{d}|{y}": len(v) for (d, y), v in strata.items()} != declared:
                raise DCASEDataError(f"{rel}: stratum counts of {native} differ from the file's own declaration")
            cohorts[machine] = _Cohort(native, machine, payload, clips, strata)
        design.cohorts[split] = cohorts
    support: dict[str, list[Clip]] = {}
    for rec in read_json(base / SUPPORT_FILE):
        machine = str(rec.get("machine"))
        c = _clip(root, rec, machine, SUPPORT_FILE, train=True)
        if c.domain != "source" or c.label != 0:
            raise DCASEDataError(f"{SUPPORT_FILE} holds a clip that is not a source-domain normal one")
        support.setdefault(machine, []).append(c)
    design.support = {m: sorted(cs, key=lambda c: c.official_id) for m, cs in sorted(support.items())}
    # ---- invariants: unique handles / files / PCM everywhere, hence src / val / id probes and the support are disjoint
    everything = design.clips()
    for what, key in (("item handles", lambda c: c.item_id), ("PCM payloads", lambda c: c.pcm_sha256),
                      ("file sha256 values", lambda c: c.sha256), ("official ids", lambda c: c.official_id)):
        keys = [key(c) for c in everything]
        if len(set(keys)) != len(keys):
            raise DCASEDataError(f"{what} are not unique across src / val / id / source_fit_support")
    for split, cohorts in design.cohorts.items():
        missing = sorted(set(cohorts) - set(design.support))
        if missing:
            raise DCASEDataError(f"{split}: machines {missing} have no source_fit_support clips")
    return design


# ----------------------------------------------------------------------------------------------- adapter
class DCASE2024Task2Adapter:
    """TaskAdapter for FoR40 (see module docstring)."""

    discipline = CODE
    name = "dcase2024-task2"
    family = FAMILY
    metric = "official DCASE score"
    direction = "max"
    task_type = "anomalous_sound_detection"

    def __init__(self, data_root: str | None = None, partition_seed: int = PARTITION_SEED, margin: float = 0.02,
                 max_train_clips: int = 128, required_plan: dict[str, int] | None = None,
                 required_items: int = DEFAULT_ITEMS, budget: Budget | None = None, verify_frozen: bool = True,
                 **_: Any) -> None:
        self.root = resolve_data_root(data_root) / DATASET_DIR
        self.partition_seed = int(partition_seed)      # salt of the episode draws (the role partition itself is frozen)
        self.margin = float(margin)
        self.max_train_clips = int(max_train_clips)
        self.required_plan = dict(DEFAULT_PLAN if required_plan is None else required_plan)
        self.required_items = int(required_items)
        self.budget = budget
        self.verify_frozen = bool(verify_frozen)
        self._design: Lazy[_Design] = Lazy(lambda: load_design(self.root, self.verify_frozen))

    # ------------------------------------------------------------ protocol
    def available(self) -> tuple[bool, str]:
        if not self.root.exists():
            return False, f"missing {self.root}"
        try:
            design = self._design.get()
        except (DCASEDataError, OSError, json.JSONDecodeError) as ex:
            return False, f"DCASE 2024 Task 2 {DATA_VERSION} unusable: {ex}"
        clips = design.clips()
        bad = [c.item_id for c in clips if not c.path.is_file() or c.path.stat().st_size != c.nbytes]
        if bad:
            return False, (f"DCASE 2024 Task 2 {DATA_VERSION}: {len(bad)} of {len(clips)} referenced WAV files are "
                           f"missing or have the wrong size (first: {bad[0]})")
        caps = {s: self.capacity(s, self.required_items) for s in self.required_plan}
        short = {s: (caps[s], need) for s, need in self.required_plan.items() if caps[s] < need}
        if short:
            detail = "; ".join(f"{s}: {c} of {need} episodes" for s, (c, need) in short.items())
            return False, (f"DCASE 2024 Task 2 {DATA_VERSION} too small for the plan with {self.required_items} "
                           f"clips/episode (single machine, {self.required_items // 4} per domain x condition stratum): "
                           f"{detail}. Probe clips per split/machine/stratum: {self.stratum_counts()}")
        return True, (f"DCASE 2024 Task 2 {DATA_VERSION} at {self.root}: capacity {caps} episodes of "
                      f"{self.required_items} clips; OOD pool empty by design (no formal OOD delivered)")

    def build_episodes(self, split: str, n: int, seed: int, items_per_episode: int = 16) -> list[Episode]:
        check_split(split)
        if items_per_episode < 4 or items_per_episode % 4:
            raise ValueError("DCASE episodes need items_per_episode as a positive multiple of 4 (4 strata)")
        if split not in ROLE_FILES:                    # "ood": no unseen-machine / cross-dataset pool exists in v3
            return []
        plan = self._plan(split, items_per_episode, seed)
        if int(n) > len(plan):
            raise PoolExhausted(f"{CODE}/{split}: requested {n} episodes but the local clips support {len(plan)} "
                                f"(probe clips per machine and stratum {self.stratum_counts().get(split)}, "
                                f"{items_per_episode // 4} per stratum and episode)")
        return [self._episode(split, k, int(seed), machine, items) for k, (machine, items) in enumerate(plan[:int(n)])]

    def pooled_metric(self, per_episode: list[dict]) -> float | None:
        """Official aggregation: per machine AUC(source), AUC(target), pAUC on its pooled items; harmonic mean."""
        from scipy.stats import hmean
        by_machine: dict[str, dict[str, list]] = {}
        for p in per_episode:
            if not p:
                continue
            s = p.get("y_score")
            if s is None:
                s = p.get("y_ref")
            t, d = list(p.get("y_true") or []), list(p.get("domain") or [])
            if s is None or not (len(s) == len(t) == len(d)):
                raise ValueError("FoR40 pooled payload needs y_true, domain and y_score (or y_ref) of equal length")
            g = by_machine.setdefault(str(p.get("machine")), {"t": [], "d": [], "s": []})
            g["t"].extend(int(v) for v in t)
            g["d"].extend(str(v) for v in d)
            g["s"].extend(float(v) for v in s)
        vals: list[float] = []
        for g in by_machine.values():
            sc = official_scores(np.array(g["t"]), np.array(g["d"]), np.array(g["s"]))
            vals.extend([sc["auc_source"], sc["auc_target"], sc["pauc"]])
        return float(hmean(np.maximum(vals, EPS))) if vals else None

    def pooled_diagnostics(self, per_episode: list[dict]) -> dict[str, Any]:
        """Return sample-size and reference diagnostics without changing the primary/acceptance rule.

        ``pooled_metric`` remains the sole pooled primary.  The extra fields make it explicit how many clips and
        machine sections support a number and compare the historical 1-NN reference with the stronger k-NN reference.
        They are trusted-side diagnostics; labels are never included in the returned object.
        """
        by_machine: dict[str, dict[str, list]] = {}
        n_episodes = 0
        for p in per_episode:
            if not p:
                continue
            n_episodes += 1
            t, d = list(p.get("y_true") or []), list(p.get("domain") or [])
            s = p.get("y_score") if p.get("y_score") is not None else p.get("y_ref")
            sr = p.get("y_strong_ref")
            if s is None or sr is None or not (len(s) == len(sr) == len(t) == len(d)):
                raise ValueError("FoR40 diagnostics need y_true, domain, y_score/y_ref and y_strong_ref")
            ref = p.get("y_ref")
            if ref is None or len(ref) != len(t):
                raise ValueError("FoR40 diagnostics need y_ref with the same length as y_true")
            g = by_machine.setdefault(str(p.get("machine")), {"t": [], "d": [], "s": [], "ref": [], "sr": []})
            g["t"].extend(int(v) for v in t); g["d"].extend(str(v) for v in d)
            g["s"].extend(float(v) for v in s); g["ref"].extend(float(v) for v in ref); g["sr"].extend(float(v) for v in sr)
        primary_vals, ref_vals, strong_vals = [], [], []
        counts = {}
        from scipy.stats import hmean
        for machine, g in sorted(by_machine.items()):
            counts[machine] = len(g["t"])
            for key, out in (("s", primary_vals), ("ref", ref_vals), ("sr", strong_vals)):
                sc = official_scores(np.asarray(g["t"]), np.asarray(g["d"]), np.asarray(g[key]))
                out.extend([sc["auc_source"], sc["auc_target"], sc["pauc"]])
        return {"n_episodes": n_episodes, "n_items": int(sum(counts.values())),
                "n_machines": len(counts), "items_per_machine": counts,
                "pooled_primary": float(hmean(np.maximum(primary_vals, EPS))) if primary_vals else None,
                "pooled_reference": float(hmean(np.maximum(ref_vals, EPS))) if ref_vals else None,
                "pooled_strong_reference": float(hmean(np.maximum(strong_vals, EPS))) if strong_vals else None}

    def full_cohort_diagnostics(self, split: str) -> dict[str, Any]:
        """Evaluate the two label-blind references on every native clip in a split for comparability reports."""
        if split not in ROLE_FILES:
            raise ValueError(f"unknown DCASE split {split!r}")
        d = self._design.get()
        payloads = []
        for machine, cohort in sorted(d.cohorts[split].items()):
            clips = [cohort.clips[i] for i in sorted(cohort.clips)]
            train = d.support[machine][: self.max_train_clips]
            ta = (16000, np.asarray([read_clip(c) for c in train], dtype=object),
                  np.asarray([len(read_clip(c)) for c in train], dtype=np.int64))
            ea = (16000, np.asarray([read_clip(c) for c in clips], dtype=object),
                  np.asarray([len(read_clip(c)) for c in clips], dtype=np.int64))
            # Ragged arrays are normalized through the same episode loader semantics.
            def pack(xs):
                L = np.asarray([len(x) for x in xs], dtype=np.int64); W = np.zeros((len(xs), int(L.max())))
                for i, x in enumerate(xs): W[i, :len(x)] = x
                return 16000, W, L
            ta, ea = pack(ta[1].tolist()), pack(ea[1].tolist())
            old = self._reference(ta, ea, np.asarray([c.label for c in clips]), np.asarray([c.domain for c in clips]))
            strong = self._strong_reference(ta, ea, np.asarray([c.label for c in clips]), np.asarray([c.domain for c in clips]))
            payloads.append({"machine": machine, "y_true": [c.label for c in clips], "domain": [c.domain for c in clips],
                             "y_ref": old["scores"], "y_strong_ref": strong["scores"], "y_score": strong["scores"]})
        return self.pooled_diagnostics(payloads)

    # ------------------------------------------------------------ pools / capacity
    def _pools(self, split: str) -> dict[str, dict[tuple[str, int], list[str]]]:
        """machine -> stratum -> sorted probe item ids of the split (empty for "ood")."""
        if split not in ROLE_FILES:
            return {}
        return {m: co.strata for m, co in sorted(self._design.get().cohorts[split].items())}

    def stratum_counts(self) -> dict[str, dict[str, list[int]]]:
        """split -> machine -> probe clip counts per STRATA order (source-normal, source-anomaly, target-normal,
        target-anomaly); the "ood" entry is empty."""
        return {s: {m: [len(v[st]) for st in STRATA] for m, v in self._pools(s).items()} for s in (*ROLE_FILES, "ood")}

    def capacity(self, split: str, items_per_episode: int) -> int:
        q = items_per_episode // 4
        pools = self._pools(split)
        return int(sum(min(len(v[st]) // q for st in STRATA) for v in pools.values())) if q > 0 else 0

    def verify_files(self) -> int:
        """Re-hash every referenced WAV against its role record (raises on mismatch); returns the number checked."""
        clips = self._design.get().clips()
        for c in clips:
            read_clip(c)
        return len(clips)

    def _plan(self, split: str, items: int, seed: int) -> list[tuple[str, list[str]]]:
        """Deterministic, prefix-stable list of (machine, item ids): machines round-robin in a seeded order."""
        q = items // 4
        rng = episode_rng(CODE, split, seed, self.partition_seed)
        pools = self._pools(split)
        machines = sorted(pools)
        order = [machines[i] for i in rng.permutation(len(machines))]
        perms = {m: {st: [pools[m][st][j] for j in rng.permutation(len(pools[m][st]))] for st in STRATA} for m in order}
        cap = {m: min(len(perms[m][st]) // q for st in STRATA) for m in order}
        plan: list[tuple[str, list[str]]] = []
        used = {m: 0 for m in order}
        while True:
            progressed = False
            for m in order:
                j = used[m]
                if j < cap[m]:
                    ids = [i for st in STRATA for i in perms[m][st][j * q:(j + 1) * q]]
                    plan.append((m, [ids[t] for t in rng.permutation(len(ids))]))
                    used[m] += 1
                    progressed = True
            if not progressed:
                return plan

    # ------------------------------------------------------------ episode
    def _episode(self, split: str, k: int, seed: int, machine: str, items: list[str]) -> Episode:
        design = self._design.get()
        cohort = design.cohorts[split][machine]
        eid = episode_id(CODE, split, seed, k)
        n = len(items)
        clips = [cohort.clips[i] for i in items]
        train = design.support[machine][: self.max_train_clips]
        y_true = np.array([c.label for c in clips], dtype=int)
        domain = np.array([c.domain for c in clips])

        def _load(cs: list[Clip]) -> tuple[int, np.ndarray, np.ndarray]:
            waves, srs = [], set()
            for c in cs:
                srs.add(c.sample_rate)
                waves.append(read_clip(c))
            if len(srs) != 1:
                raise ValueError(f"mixed sample rates {srs} for machine {machine}")
            lengths = np.array([w.size for w in waves], dtype=np.int64)
            W = np.zeros((len(waves), int(lengths.max())), dtype=np.float32)
            for r, w in enumerate(waves):
                W[r, :w.size] = w
            return srs.pop(), W, lengths

        eval_audio = Lazy(lambda: _load(clips))
        train_audio = Lazy(lambda: _load(train))
        ref = Lazy(lambda: self._reference(train_audio.get(), eval_audio.get(), y_true, domain))
        strong_ref = Lazy(lambda: self._strong_reference(train_audio.get(), eval_audio.get(), y_true, domain))

        def load_train(inputs: dict, config: dict) -> dict:
            sr, W, L = train_audio.get()
            return {"waveforms": W.copy(), "lengths": L.copy(), "domains": [c.domain for c in train],
                    "attributes": [c.attributes for c in train], "sample_rate": float(sr)}

        def load_eval_inputs(inputs: dict, config: dict) -> dict:
            sr, W, L = eval_audio.get()
            return {"waveforms": W.copy(), "lengths": L.copy(), "sample_rate": float(sr)}

        def log_mel_tool(inputs: dict, config: dict) -> dict:
            W = np.asarray(inputs.get("waveforms"), dtype=np.float32)
            if W.ndim != 2:
                raise ValueError("waveforms must be a 2-D array (n_clips, n_samples)")
            sr = float(config.get("sample_rate", 16000.0))
            if not 1000.0 <= sr <= 192000.0:
                raise ValueError("sample_rate must be in [1000, 192000] Hz")
            n_fft, hop = int(config.get("n_fft", MEL["n_fft"])), int(config.get("hop", MEL["hop"]))
            n_mels = int(config.get("n_mels", MEL["n_mels"]))
            if not (64 <= n_fft <= 8192 and 16 <= hop <= n_fft and 8 <= n_mels <= 256):
                raise ValueError("need 64 <= n_fft <= 8192, 16 <= hop <= n_fft, 8 <= n_mels <= 256")
            S = np.stack([log_mel(w, int(sr), n_fft, hop, n_mels) for w in W])
            return {"log_mel": S, "frame_rate": float(sr / hop)}

        def audio_embedding_tool(inputs: dict, config: dict) -> dict:
            """Optional frozen AST/CLAP features; the default route remains log-mel."""
            from scilib import audioenc
            W = np.asarray(inputs.get("waveforms"), dtype=np.float32)
            if W.ndim != 2:
                raise ValueError("waveforms must be a 2-D array (n_clips, n_samples)")
            L = np.asarray(inputs.get("lengths"), dtype=np.int64).reshape(-1)
            if L.size != W.shape[0] or L.size < 1 or L.min() < 1 or L.max() > W.shape[1]:
                raise ValueError("lengths must hold one value per waveform row within the padded width")
            sr = float(inputs.get("sample_rate", config.get("sample_rate", 16000.0)))
            model = str(config.get("model", "ast_audioset"))
            kind = str(config.get("kind", "pooled"))
            if model not in audioenc.MODELS:
                raise ValueError(f"model must be one of {sorted(audioenc.MODELS)}")
            if kind not in audioenc.MODELS[model][3]:
                raise ValueError(f"kind must be one of {audioenc.MODELS[model][3]} for {model!r}")
            E = np.asarray(audioenc.embed(W, L, sr, model=model, kind=kind), dtype=np.float32)
            if E.ndim != 2 or E.shape[0] != W.shape[0] or not np.all(np.isfinite(E)):
                raise ValueError("audio encoder returned an invalid embedding matrix")
            return {"embeddings": E, "model": model, "kind": kind}

        n_tr = len(train)
        tools = [
            ToolSpec("load_train", f"{n_tr} normal source-domain training clips of the machine type (no anomalies, no "
                     "target-domain clip): waveforms (zero-padded rows, float in [-1, 1]), lengths (samples), domains "
                     "(all 'source'), attributes (official operating-condition string per clip), sample_rate (Hz).",
                     {}, {"waveforms": PortSchema("array", (n_tr, "T"), dtype="float"),
                          "lengths": PortSchema("array", (n_tr,), dtype="int"),
                          "domains": PortSchema("list", (n_tr,), dtype="str"),
                          "attributes": PortSchema("list", (n_tr,), dtype="str"),
                          "sample_rate": PortSchema("number", unit="Hz")},
                     load_train),
            ToolSpec("load_eval_inputs", f"The {n} evaluation test clips of the same machine type (condition and "
                     "domain hidden), in the order y must follow.",
                     {}, {"waveforms": PortSchema("array", (n, "T2"), dtype="float"),
                          "lengths": PortSchema("array", (n,), dtype="int"),
                          "sample_rate": PortSchema("number", unit="Hz")},
                     load_eval_inputs),
            ToolSpec("log_mel_spectrogram", "Log-mel power spectrogram in dB per clip (Hann window, centred frames).",
                     {"waveforms": PortSchema("array", ("n", "T3"), dtype="float")},
                     {"log_mel": PortSchema("array", ("n", "frames", "n_mels"), unit="dB", dtype="float"),
                      "frame_rate": PortSchema("number", unit="Hz")},
                     log_mel_tool, config_doc="{n_fft: int (1024), hop: int (512), n_mels: int (128), sample_rate: Hz (16000)}"),
            ToolSpec("audio_embedding", "Optional frozen pooled AST or CLAP audio embeddings. Use only when the configured "
                     "local/remote audio encoder is available; this is additive and does not replace log-mel.",
                     {"waveforms": PortSchema("array", ("n", "T3"), dtype="float"),
                      "lengths": PortSchema("array", ("n",), dtype="int"),
                      "sample_rate": PortSchema("number", unit="Hz")},
                     {"embeddings": PortSchema("array", ("n", "d"), dtype="float"),
                      "model": PortSchema("text"), "kind": PortSchema("text")},
                     audio_embedding_tool,
                     config_doc="{model: 'ast_audioset'|'clap_htsat', kind: 'pooled'|'logits'|'proj', sample_rate: Hz (16000)}"),
        ]
        constraints: list[ConstraintSpec] = [c_vector(n, "one anomaly score per evaluation clip"), c_finite(n)]

        def evaluate(yv: Any, trace: Trace | None) -> EvalResult:
            r = ref.get()
            sr = strong_ref.get()
            payload = {"item_ids": list(items), "machine": machine, "y_true": y_true.tolist(),
                       "domain": domain.tolist(), "y_score": None, "y_ref": r["scores"],
                       "y_strong_ref": sr["scores"]}
            base = {"reference": r["official_score"], "reference_name": "1-NN log-mel statistics distance",
                    "margin": self.margin, "n_items": n, "machine": machine}
            arr, why = as_float_vector(yv, n)
            if arr is not None and not np.all(np.isfinite(arr)):
                arr, why = None, "non-finite values"
            if arr is None:
                return EvalResult(metrics={"reference_official_score": r["official_score"],
                                           "strong_reference_official_score": sr["official_score"]}, primary=None,
                                  direction="max", accepted=False,
                                  details={**base, "norm_score": 0.0, "pooled_payload": payload, "invalid": why})
            sc = official_scores(y_true, domain, arr)
            payload["y_score"] = arr.tolist()
            return EvalResult(metrics={**sc, "reference_official_score": r["official_score"],
                                       "strong_reference_official_score": sr["official_score"]},
                              primary=sc["official_score"], direction="max",
                              accepted=bool(sc["official_score"] >= r["official_score"] + self.margin),
                              details={**base, "norm_score": norm_score(sc["official_score"], r["official_score"], "max"),
                                       "pooled_payload": payload})

        objective = (
            "Engineering - first-shot unsupervised anomalous sound detection for machine condition monitoring "
            f"(DCASE 2024 Task 2). Machine type: {machine}. Each evaluation item is a single-channel recording of this "
            "machine; some recordings contain anomalous sounds (the machine is damaged or malfunctioning), the others "
            "are normal. Recordings come from two domains: source (the operating condition of the training clips) and "
            "target (a shifted operating condition for which no training clip exists); the domain of evaluation clips "
            "is not given.\n"
            f"Visible data: load_train returns {n_tr} normal source-domain training clips of the same machine type with "
            "their attribute strings (no anomalous and no target-domain training clips exist); load_eval_inputs returns "
            "the evaluation clips; log_mel_spectrogram computes log-mel spectrograms. The optional audio_embedding "
            "tool exposes frozen AST/CLAP features when a configured encoder backend is available.\n"
            f"Deliverable y: a 1-D float array of length {n}; y[i] is the anomaly score of the i-th clip returned by "
            "load_eval_inputs (higher = more likely anomalous). Evaluation metric: official DCASE 2024 Task 2 score = "
            "harmonic mean of AUC(source), AUC(target) and pAUC (FPR <= 0.1).\n"
            + scilib.describe("anomsound") + scilib.describe_extra("audioenc")
        )
        lineage = {
            "dataset": "DCASE 2024 Task 2",
            "version": f"development 10902294; data delivery {DATA_VERSION}, design {DESIGN_ID}; evaluator commit b5d17f63",
            "source_url": "https://dcase.community/challenge2024/task-first-shot-unsupervised-anomalous-sound-detection-"
                          "for-machine-condition-monitoring",
            "license": "CC-BY-NC-SA-4.0 (Zenodo record 10902294)",
            "receipt": receipt_info(self.root / "receipt.json"),
            "delivery_receipt": receipt_info(self.root / DATA_VERSION / "download-receipt.json",
                                             keys=("status", "plan_sha256")),
            "pool": "iid", "role": {"src": "source", "val": "val", "id": "id"}[split], "machine": machine,
            "section": clips[0].section, "native_cohort": cohort.native_id,
            "native_cohort_payload_sha256": cohort.payload_sha256,
            "role_file_sha256": {ROLE_FILES[split]: design.file_sha256[ROLE_FILES[split]],
                                 SUPPORT_FILE: design.file_sha256[SUPPORT_FILE]},
            "ood_kind": None, "ood_shift": None,
            "split": split, "split_seed": seed, "partition_seed": self.partition_seed, "index": k,
            "item_ids": list(items), "n_items": n, "train_item_ids": [c.item_id for c in train],
            "train_role": "source_fit_support (source-domain normal clips; shared by every episode of the machine)",
            "frozen_role_files": True, "historical_ids_recovered": False,
        }
        return Episode(
            id=eid, discipline=CODE, family=FAMILY, split=split, task_type=self.task_type, objective=objective,
            required_output=PortSchema("array", (n,), unit="1", dtype="float",
                                       description="anomaly score per evaluation clip, load_eval_inputs order"),
            tools=tools, constraints=constraints,
            budget=self.budget or Budget(max_steps=12, max_policy_tokens=200_000, max_wall_s=1800.0, max_node_s=300.0,
                                         max_llm_items=2 * n),
            lineage=lineage,
            acceptance=(f"official DCASE score >= reference + {self.margin:g}; reference = 1-NN distance of standardized "
                        "log-mel mean/std statistics to the machine's training clips"),
            tolerance={"rtol": 1e-5, "atol": 1e-7},
            tags=[CODE, "engineering", "acoustics", "audio", "anomaly-detection", "machine-condition-monitoring",
                  "domain-shift", "AUC", "pAUC", "dcase", machine],
            metric=self.metric, direction=self.direction, n_items=n,
            _evaluate=evaluate, _dev_evaluate=None,
        )

    @staticmethod
    def _reference(train_audio: tuple, eval_audio: tuple, y_true: np.ndarray, domain: np.ndarray) -> dict:
        def stats(sr: int, W: np.ndarray, L: np.ndarray) -> np.ndarray:
            rows = []
            for w, n in zip(W, L):
                S = log_mel(w[:n], sr, **MEL)
                rows.append(np.concatenate([S.mean(axis=0), S.std(axis=0)]))
            return np.array(rows)

        Ft, Fe = stats(*train_audio), stats(*eval_audio)
        mu, sd = Ft.mean(axis=0), Ft.std(axis=0)
        sd[sd < 1e-6] = 1.0
        Zt, Ze = (Ft - mu) / sd, (Fe - mu) / sd
        d = np.sqrt(((Ze[:, None, :] - Zt[None, :, :]) ** 2).sum(axis=-1)).min(axis=1)
        sc = official_scores(y_true, domain, d)
        return {"scores": d.tolist(), "official_score": sc["official_score"]}

    @staticmethod
    def _strong_reference(train_audio: tuple, eval_audio: tuple, y_true: np.ndarray, domain: np.ndarray) -> dict:
        """Label-blind k-NN log-mel reference; fit uses source-normal support only."""
        def stats(sr: int, W: np.ndarray, L: np.ndarray) -> np.ndarray:
            rows = []
            for w, n in zip(W, L):
                S = log_mel(np.asarray(w[:n]), sr, **MEL)
                rows.append(np.concatenate([S.mean(axis=0), S.std(axis=0)]))
            return np.asarray(rows)
        Ft, Fe = stats(*train_audio), stats(*eval_audio)
        mu, sd = Ft.mean(axis=0), np.maximum(Ft.std(axis=0), 1e-6)
        d = np.sqrt((( (Fe - mu) / sd)[:, None, :] - ((Ft - mu) / sd)[None, :, :]) ** 2).sum(axis=-1)
        k = min(5, d.shape[1])
        scores = np.mean(np.partition(d, k - 1, axis=1)[:, :k], axis=1)
        sc = official_scores(y_true, domain, scores)
        return {"scores": scores.tolist(), "official_score": sc["official_score"], "k": k}


Adapter = DCASE2024Task2Adapter

__all__ = ["Adapter", "DCASE2024Task2Adapter", "DCASEDataError", "official_scores", "log_mel", "read_wav", "read_clip",
           "load_design"]
