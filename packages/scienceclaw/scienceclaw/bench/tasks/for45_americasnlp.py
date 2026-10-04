"""FoR45 Indigenous studies — AmericasNLP 2026 cultural image captioning, metric mean sentence chrF++ (max).

Data: the AmericasNLP 2026 shared-task repository at commit ``7ff73013`` (https://github.com/AmericasNLP/americasnlp2026,
CC BY-NC 4.0), downloaded by the data team under ``<DATA_ROOT>/for45-americasnlp-2026``. Only the official
``dev`` partition has public target captions (5 languages x 50 images); the official test JSONL files carry no
captions and the pilot file (20 Wixárika items) has captions but no downloaded images. Images are resolved by
file name inside ``reconstructed_v1/upstream/data/dev/<lang>/images`` (200 images) or ``upstream/data/dev/<lang>/
images`` (first 8 per language); 210 of the 250 dev rows have a local image (the remaining 40 are caption-only).

* **Item** = one image (a 64x64 RGB thumbnail, centre-cropped square) + its language / ISO code / culture;
  hidden label = the target caption written in that Indigenous language. Item id = ``americasnlp/<iso>/<id>``.
* **Pools (proxy OOD).** No second Indigenous-language captioning dataset exists locally, so the OOD shift is
  a *held-out language family inside the same dataset*: IID = Bribri (Chibchan), Guaraní (Tupian) and Yucatec Maya
  (Mayan); OOD = Orizaba/Central Nahuatl (the task's "surprise language") and Wixárika — both Uto-Aztecan and never
  seen in any src/val/id episode (``lineage["ood_kind"] = "proxy_within_dataset"``).
* **Splits.** Per language, the image rows are partitioned once (``partition_seed``) at the level of image
  content (sha256 of the image file; duplicate images stay together): IID languages -> src / val / id / train
  = 19 / 6 / 11 / 6 of 42 rows; OOD languages -> ood / train = 40 / 60 %. Caption-only rows (and, for Wixárika,
  the 20 pilot captions) are visible training data only.
* **Episode size.** The labelled population is 250 captions, which cannot supply the historical 16-item episodes
  for 7 src + 2 val + 4 id + 4 ood episodes with disjoint visible data. Episodes therefore have
  ``min(items_per_episode, max_items)`` items (default ``max_items = 8``), balanced over the pool's languages;
  the ID and OOD sets hold 32 items each instead of the historical 64 (documented deviation).
* **Visible data (D_E).** ``load_train`` returns the pool's training rows (captions + thumbnails where available);
  ``load_dev_inputs`` / ``score_dev`` hold out ``n_dev`` image rows of the training partition (their captions
  are withheld; the episode's ``_dev_evaluate`` is None); ``image_features`` computes simple CPU image
  descriptors (colour histograms, grey thumbnail, HSV/edge statistics), and ``image_knn_reference`` retrieves
  a caption from a same-language visible training image. The latter is a small, deterministic image-conditioned
  reference/tool route, not a hidden-label shortcut.
* **Metric (D_V).** Mean over items of sentence-level chrF++ = ``sacrebleu.metrics.CHRF(word_order=2)
  .sentence_score(hyp, [ref]).score`` (character 6-grams + word 2-grams, beta = 2; 0-100 scale) — exactly the
  shared task's official ``baseline/eval.py`` at the pinned commit (per-caption chrF++ then the mean; the
  official script strips each generated caption, so predictions are ``.strip()``-ed as well).
* **Reference baseline** (deterministic): for every item predict the *medoid caption* of its language in the
  episode's visible training rows (the training caption with the highest mean sentence chrF++ against the other
  training captions of that language). **Acceptance:** ``mean chrF++ >= reference + margin`` (default 1.0
  chrF++ point).
* **Diagnostics (details only, no effect on the score or acceptance).** Number and ratio of distinct captions,
  distinct captions per language, ``image_blind`` (one caption per language for every language present) and the
  mean caption length in words, plus a trusted-side score for the visible-image nearest-neighbour reference:
  an image-independent per-language string can reach the level of published systems on chrF++ (beta = 2 rewards
  recall), so a chrF++ score is only evidence of image grounding when ``image_blind`` is false. The image
  nearest-neighbour score is diagnostic and never changes the official reference or acceptance.
* **Hard constraints:** list of ``n_items`` strings; every caption non-empty and at most ``max_caption_chars``
  (default 1000) characters.

Limitations: the policy/executor LLM of this code base is text-only, so image content is only available through
the pixel thumbnails and ``image_features``; this is stated in the objective.
"""
from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import scilib
from scilib import clip_retrieval

from ...core.schema import PortSchema
from ...core.trace import Trace
from ..task import Budget, ConstraintSpec, EvalResult, Episode, ToolSpec
from ._adapter_utils_for43_45_47_48_50 import (
    PARTITION_SEED, Lazy, PoolExhausted, as_list, c_list_length, cache_dir, check_split, draw_stratified,
    episode_id, episode_rng, ids_digest, norm_score, partition_groups, read_jsonl, receipt_info, resolve_data_root,
    stable_int,
)

CODE = "FoR45"
FAMILY = "Humanities & law"
DATASET_DIR = "for45-americasnlp-2026"
LANG_DIRS = ("bribri", "guarani", "maya", "nahuatl", "wixarika")
IID_LANGS = ("bribri", "guarani", "maya")
OOD_LANGS = ("nahuatl", "wixarika")
IID_PARTS = {"src": 19, "val": 6, "id": 11, "train": 6}          # relative sizes (of 42 image rows per IID language)
OOD_PARTS = {"ood": 0.40, "train": 0.60}
THUMB = 64
_CACHE_VERSION = "v1"


def chrf_pp(hyp: str, ref: str) -> float:
    """Sentence-level chrF++ (sacrebleu CHRF, char_order 6, word_order 2, beta 2) on the 0-100 scale."""
    return float(_chrf().sentence_score(hyp, [ref]).score)


_CHRF_LOCAL = threading.local()


def _chrf() -> Any:
    """Per-thread sacrebleu ``CHRF(word_order=2)`` scorer (evaluators may run concurrently)."""
    obj = getattr(_CHRF_LOCAL, "chrf", None)
    if obj is None:
        from sacrebleu.metrics import CHRF
        obj = CHRF(word_order=2)
        _CHRF_LOCAL.chrf = obj
    return obj


# ----------------------------------------------------------------------------------------------- data
@dataclass(frozen=True)
class _Row:
    uid: str
    lang_dir: str
    language: str
    iso: str
    culture: str
    caption: str
    image_path: str | None     # None for caption-only rows
    image_sha: str | None
    source: str                # "dev" | "pilot"


def _find_image(root: Path, lang: str, filename: str) -> Path | None:
    base = Path(filename).name
    for d in (root / "reconstructed_v1" / "upstream" / "data" / "dev" / lang / "images",
              root / "upstream" / "data" / "dev" / lang / "images"):
        p = d / base
        if p.is_file():
            return p
    return None


def _annotation_file(root: Path, lang: str) -> Path | None:
    for d in (root / "reconstructed_v1" / "upstream" / "data" / "dev", root / "upstream" / "data" / "dev"):
        p = d / lang / f"{lang}.jsonl"
        if p.is_file():
            return p
    return None


def _load_rows(root: Path) -> dict[str, _Row]:
    rows: dict[str, _Row] = {}
    for lang in LANG_DIRS:
        ann = _annotation_file(root, lang)
        if ann is None:
            raise FileNotFoundError(f"no AmericasNLP dev annotations for {lang} under {root}")
        for r in read_jsonl(ann):
            cap = str(r.get("target_caption") or "").strip()
            if not cap:
                continue
            img = _find_image(root, lang, str(r.get("filename", "")))
            sha = hashlib.sha256(img.read_bytes()).hexdigest() if img is not None else None
            uid = f"americasnlp/{r.get('iso_lang')}/{r.get('id')}"
            rows[uid] = _Row(uid, lang, str(r.get("language")), str(r.get("iso_lang")), str(r.get("culture")), cap,
                             str(img) if img is not None else None, sha, "dev")
    pilot = root / "reconstructed_v1" / "upstream" / "data" / "pilot" / "wixarika.jsonl"
    if pilot.is_file():
        for r in read_jsonl(pilot):
            cap = str(r.get("target_caption") or "").strip()
            if cap:
                uid = f"americasnlp/{r.get('iso_lang')}/pilot-{r.get('id')}"
                rows[uid] = _Row(uid, "wixarika", str(r.get("language")), str(r.get("iso_lang")), str(r.get("culture")),
                                 cap, None, None, "pilot")
    return rows


def _thumbnail(path: str) -> np.ndarray:
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        im = im.resize((THUMB, THUMB), Image.Resampling.BICUBIC)
        return np.asarray(im, dtype=np.uint8)


def _load_thumbs(rows: dict[str, _Row]) -> dict[str, np.ndarray]:
    """uid -> (64, 64, 3) uint8 thumbnail; cached as .npz under cache/tasks/FoR45 keyed by image sha256."""
    cache = cache_dir(CODE) / f"thumbs_{_CACHE_VERSION}.npz"
    have: dict[str, np.ndarray] = {}
    if cache.exists():
        with np.load(cache) as z:
            have = {k: z[k] for k in z.files}
    changed = False
    out: dict[str, np.ndarray] = {}
    for uid, r in rows.items():
        if r.image_path is None or r.image_sha is None:
            continue
        if r.image_sha not in have:
            have[r.image_sha] = _thumbnail(r.image_path)
            changed = True
        out[uid] = have[r.image_sha]
    if changed:
        tmp = cache.with_name(f"{cache.stem}.{os.getpid()}.{threading.get_ident()}.tmp.npz")
        np.savez_compressed(tmp, **have)
        tmp.replace(cache)
    return out


@dataclass
class _Data:
    rows: dict[str, _Row]
    thumbs: dict[str, np.ndarray]
    eval_split: dict[str, dict[str, list[str]]]     # split -> lang dir -> uids
    train_images: dict[str, dict[str, list[str]]]   # "iid"/"ood" -> lang -> image rows of the train partition
    train_captions: dict[str, dict[str, list[str]]]  # "iid"/"ood" -> lang -> caption-only rows


def _build(root: Path, partition_seed: int) -> _Data:
    rows = _load_rows(root)
    thumbs = _load_thumbs(rows)
    ev: dict[str, dict[str, list[str]]] = {s: {} for s in ("src", "val", "id", "ood")}
    timg: dict[str, dict[str, list[str]]] = {"iid": {}, "ood": {}}
    tcap: dict[str, dict[str, list[str]]] = {"iid": {}, "ood": {}}
    for lang in LANG_DIRS:
        pool = "iid" if lang in IID_LANGS else "ood"
        img = {u: r.image_sha for u, r in rows.items() if r.lang_dir == lang and r.image_sha is not None}
        parts = partition_groups(img, IID_PARTS if pool == "iid" else OOD_PARTS, f"{CODE}|{partition_seed}|{lang}")
        for s, ids in parts.items():
            if s == "train":
                timg[pool][lang] = sorted(ids)
            else:
                ev[s][lang] = sorted(ids)
        tcap[pool][lang] = sorted(u for u, r in rows.items() if r.lang_dir == lang and r.image_sha is None)
    return _Data(rows, thumbs, ev, timg, tcap)


# ----------------------------------------------------------------------------------------------- image descriptors
def image_descriptors(images: np.ndarray, kind: str = "all") -> tuple[np.ndarray, list[str]]:
    """Simple CPU image features of (n, H, W, 3) uint8 images -> (X (n, d) float32, feature names).

    kind: ``color_hist`` (4x4x4 RGB histogram, L1-normalised), ``gray_thumb`` (8x8 grey thumbnail in [0, 1]),
    ``hsv_stats`` (mean/std of H, S, V + edge density) or ``all`` (concatenation).
    """
    arr = np.asarray(images)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"images must have shape (n, H, W, 3), got {list(arr.shape)}")
    x = arr.astype(np.float32) / 255.0
    n = x.shape[0]
    feats: list[np.ndarray] = []
    names: list[str] = []
    if kind in ("color_hist", "all"):
        q = np.clip((x * 4).astype(int), 0, 3)
        code = q[..., 0] * 16 + q[..., 1] * 4 + q[..., 2]
        h = np.stack([np.bincount(code[i].ravel(), minlength=64) for i in range(n)]).astype(np.float32)
        feats.append(h / np.maximum(h.sum(axis=1, keepdims=True), 1.0))
        names += [f"rgbhist_{j}" for j in range(64)]
    if kind in ("gray_thumb", "all"):
        g = x.mean(axis=-1)
        H, W = g.shape[1:]
        bh, bw = max(H // 8, 1), max(W // 8, 1)
        t = g[:, :bh * 8, :bw * 8].reshape(n, 8, bh, 8, bw).mean(axis=(2, 4)).reshape(n, 64)
        feats.append(t.astype(np.float32))
        names += [f"gray8_{j}" for j in range(64)]
    if kind in ("hsv_stats", "all"):
        mx, mn = x.max(axis=-1), x.min(axis=-1)
        v, s = mx, np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
        r, gg, b = x[..., 0], x[..., 1], x[..., 2]
        hue = (np.arctan2(np.sqrt(3.0) * (gg - b), 2 * r - gg - b) / (2 * np.pi)) % 1.0
        g = x.mean(axis=-1)
        edge = (np.abs(np.diff(g, axis=1)).mean(axis=(1, 2)) + np.abs(np.diff(g, axis=2)).mean(axis=(1, 2)))
        st = np.column_stack([hue.mean(axis=(1, 2)), hue.std(axis=(1, 2)), s.mean(axis=(1, 2)), s.std(axis=(1, 2)),
                              v.mean(axis=(1, 2)), v.std(axis=(1, 2)), edge])
        feats.append(st.astype(np.float32))
        names += ["hue_mean", "hue_std", "sat_mean", "sat_std", "val_mean", "val_std", "edge_density"]
    if not feats:
        raise ValueError(f"unknown kind {kind!r}; use color_hist | gray_thumb | hsv_stats | all")
    return np.concatenate(feats, axis=1), names


def medoid_captions(captions: list[str], langs: list[str]) -> dict[str, str]:
    """Per language, the caption with the highest mean sentence chrF++ against the language's other captions."""
    out: dict[str, str] = {}
    for lang in sorted(set(langs)):
        caps = sorted({c for c, l in zip(captions, langs) if l == lang})
        if len(caps) == 1:
            out[lang] = caps[0]
            continue
        best, best_s = caps[0], -1.0
        for c in caps:
            s = float(np.mean([chrf_pp(c, o) for o in caps if o != c]))
            if s > best_s + 1e-12:
                best, best_s = c, s
        out[lang] = best
    return out


def image_knn_captions(train_rows: list[dict[str, Any]], train_images: np.ndarray,
                       eval_items: list[dict[str, Any]], eval_images: np.ndarray,
                       k: int = 1, kind: str = "all") -> list[str]:
    """Retrieve captions from visible same-language training images.

    This is deliberately a small CPU reference route: it uses the already disclosed ``image_descriptors``
    features and cosine nearest neighbours, never evaluation labels.  Caption-only rows and zero thumbnails are
    excluded from the candidate set.  For ``k > 1`` the medoid of the selected neighbour captions is returned;
    ties are resolved by the stable row order.  If a language has no visible image, its visible-caption medoid is
    used, which makes the fallback explicit rather than silently dropping an item.
    """
    if not isinstance(k, (int, np.integer)) or int(k) < 1:
        raise ValueError("k must be a positive integer")
    if len(train_rows) != len(train_images):
        raise ValueError("train_rows and train_images must have the same length")
    if len(eval_items) != len(eval_images):
        raise ValueError("eval_items and eval_images must have the same length")
    train_images = np.asarray(train_images)
    eval_images = np.asarray(eval_images)
    if train_images.ndim != 4 or eval_images.ndim != 4:
        raise ValueError("train_images and eval_images must be image batches")
    # Keep language fallback captions independent of image availability.  This is the same visible data used by
    # the official medoid reference, but is only used when retrieval cannot find a same-language image.
    fallback: dict[str, str] = {}
    for lang in sorted({str(r.get("iso_lang", "")) for r in train_rows}):
        caps = [str(r.get("caption", "")).strip() for r in train_rows if str(r.get("iso_lang", "")) == lang
                and str(r.get("caption", "")).strip()]
        if caps:
            fallback[lang] = medoid_captions(caps, [lang] * len(caps))[lang]

    candidates = [i for i, r in enumerate(train_rows)
                  if bool(r.get("has_image")) and str(r.get("caption", "")).strip()]
    if candidates:
        X_train, _ = image_descriptors(train_images[candidates], kind)
        X_eval, _ = image_descriptors(eval_images, kind)
        train_norm = np.linalg.norm(X_train, axis=1, keepdims=True)
        eval_norm = np.linalg.norm(X_eval, axis=1, keepdims=True)
        X_train = X_train / np.maximum(train_norm, 1e-12)
        X_eval = X_eval / np.maximum(eval_norm, 1e-12)
    else:
        X_train = X_eval = np.zeros((0, 1), dtype=np.float32)

    out: list[str] = []
    for j, item in enumerate(eval_items):
        lang = str(item.get("iso_lang", ""))
        lang_pos = [q for q, i in enumerate(candidates) if str(train_rows[i].get("iso_lang", "")) == lang]
        if not lang_pos:
            out.append(fallback.get(lang, ""))
            continue
        sims = X_train[lang_pos] @ X_eval[j]
        # lexsort gives descending similarity, then original candidate position for deterministic ties.
        order = np.lexsort((np.asarray(lang_pos), -sims))[:min(int(k), len(lang_pos))]
        chosen = [str(train_rows[candidates[lang_pos[int(q)]]]["caption"]).strip() for q in order]
        out.append(chosen[0] if len(chosen) == 1 else medoid_captions(chosen, [lang] * len(chosen))[lang])
    return out


# ----------------------------------------------------------------------------------------------- adapter
class AmericasNLPAdapter:
    """TaskAdapter for FoR45 (see module docstring)."""

    discipline = CODE
    name = "AmericasNLP-2026-captioning"
    family = FAMILY
    metric = "mean sentence chrF++"
    direction = "max"
    task_type = "indigenous_language_captioning"

    def __init__(self, data_root: str | None = None, partition_seed: int = PARTITION_SEED, max_items: int = 8,
                 n_dev: int = 6, margin: float = 1.0, max_caption_chars: int = 1000, budget: Budget | None = None,
                 **_: Any) -> None:
        self.root = resolve_data_root(data_root) / DATASET_DIR
        self.partition_seed = int(partition_seed)
        self.max_items = int(max_items)
        self.n_dev = int(n_dev)
        self.margin = float(margin)
        self.max_caption_chars = int(max_caption_chars)
        self.budget = budget
        self._data: Lazy[_Data] = Lazy(lambda: _build(self.root, self.partition_seed))

    # ------------------------------------------------------------ protocol
    def available(self) -> tuple[bool, str]:
        missing = [lang for lang in LANG_DIRS if _annotation_file(self.root, lang) is None]
        if missing:
            return False, f"missing AmericasNLP 2026 dev annotation files for {missing} under {self.root}"
        n_img = {lang: sum(1 for r in read_jsonl(_annotation_file(self.root, lang))  # type: ignore[arg-type]
                           if _find_image(self.root, lang, str(r.get("filename", ""))) is not None)
                 for lang in LANG_DIRS}
        need = {lang: (sum(IID_PARTS.values()) - IID_PARTS["train"] if lang in IID_LANGS else 16) for lang in LANG_DIRS}
        short = {lang: n for lang, n in n_img.items() if n < need[lang] + 2}
        if short:
            return False, (f"too few local dev images per language {short} (need about {need}); the data team must "
                           "fetch the remaining data/dev/<lang>/images files of the AmericasNLP 2026 repository")
        try:
            import PIL  # noqa: F401
            import sacrebleu  # noqa: F401
        except ImportError as ex:
            return False, f"missing package: {ex}"
        return True, f"AmericasNLP 2026 dev (commit 7ff73013) at {self.root}; images per language {n_img}"

    def build_episodes(self, split: str, n: int, seed: int, items_per_episode: int = 16) -> list[Episode]:
        check_split(split)
        if items_per_episode < 1:
            raise ValueError("items_per_episode must be >= 1")
        k_items = min(int(items_per_episode), self.max_items)
        data = self._data.get()
        order = OOD_LANGS if split == "ood" else IID_LANGS
        rng = episode_rng(CODE, split, seed, self.partition_seed)
        draws = draw_stratified(data.eval_split[split], int(n), k_items, rng, f"{CODE}/{split}", strata_order=order)
        return [self._episode(data, split, k, int(seed), items, int(items_per_episode)) for k, items in enumerate(draws)]

    def pooled_metric(self, per_episode: list[dict]) -> float | None:
        """Mean sentence chrF++ over all items of all episodes (invalid outputs scored with the reference)."""
        scores: list[float] = []
        for p in per_episode:
            if not p:
                continue
            s = p.get("chrf_pred")
            if s is None:
                s = p.get("chrf_ref")
            if s is None:
                raise ValueError("FoR45 pooled payload needs chrf_pred or chrf_ref")
            scores.extend(float(v) for v in s)
        return float(np.mean(scores)) if scores else None

    def pooled_diagnostics(self, per_episode: list[dict]) -> dict[str, Any]:
        """Aggregate the official primary, medoid reference and image-kNN reference scores.

        The kNN values are trusted-side diagnostics computed from visible training images.  This method never
        changes :meth:`pooled_metric`, the episode evaluator, or the acceptance rule.
        """
        primary: list[float] = []
        reference: list[float] = []
        knn: list[float] = []
        n_episodes = 0
        for raw in per_episode:
            p = raw.get("pooled_payload", raw) if isinstance(raw, dict) else raw
            if not p:
                continue
            n_episodes += 1
            pred = p.get("chrf_pred")
            if pred is None:
                pred = p.get("chrf_ref", [])
            primary.extend(float(v) for v in pred)
            reference.extend(float(v) for v in p.get("chrf_ref", []))
            knn.extend(float(v) for v in p.get("chrf_knn_ref", []))
        out: dict[str, Any] = {
            "n_episodes": n_episodes,
            "n_items": len(primary),
            "pooled_primary": float(np.mean(primary)) if primary else None,
            "pooled_medoid_reference": float(np.mean(reference)) if reference else None,
            "pooled_image_knn_reference": float(np.mean(knn)) if knn else None,
            "diagnostic_only": True,
        }
        if out["pooled_medoid_reference"] is not None and out["pooled_image_knn_reference"] is not None:
            out["image_knn_gain_over_medoid"] = float(out["pooled_image_knn_reference"] -
                                                        out["pooled_medoid_reference"])
        return out

    def full_split_reference(self, split: str) -> dict[str, Any]:
        """Score the visible-caption references over every image in ``split``.

        Episode references use a small, seed-dependent training sample, so their chrF++ values are not a stable
        comparison with a published system.  This trusted-side helper uses the complete visible training partition
        for the requested pool and every image-backed evaluation row in the split.  It reports the existing
        language-medoid reference alongside the image-conditioned nearest-neighbour diagnostic on the same item
        set.  Target captions are read only here for the aggregate; nothing is added to a tool output and the
        official evaluator, acceptance margin, and pooled primary remain unchanged.
        """
        check_split(split)
        data = self._data.get()
        pool = "ood" if split == "ood" else "iid"
        split_by_lang = data.eval_split[split]
        item_ids = [uid for lang in sorted(split_by_lang) for uid in split_by_lang[lang]]
        train_ids = [uid for lang in sorted(data.train_images[pool])
                     for uid in data.train_images[pool].get(lang, [])]
        train_ids.extend(uid for lang in sorted(data.train_captions[pool])
                          for uid in data.train_captions[pool].get(lang, []))
        train_ids = sorted(train_ids)
        if not item_ids:
            return {
                "split": split,
                "pool": pool,
                "n_items": 0,
                "n_languages": 0,
                "n_train": len(train_ids),
                "n_train_images": sum(data.rows[u].image_sha is not None for u in train_ids),
                "n_train_caption_only": sum(data.rows[u].image_sha is None for u in train_ids),
                "medoid_reference_chrf_pp": None,
                "image_knn_reference_chrf_pp": None,
                "image_knn_gain_over_medoid": None,
                "per_language": {},
                "train_ids_sha256": ids_digest(train_ids),
                "split_ids_sha256": ids_digest(item_ids),
                "diagnostic_only": True,
            }

        train_rows = [{"id": data.rows[u].uid.rsplit("/", 1)[1],
                       "language": data.rows[u].language, "iso_lang": data.rows[u].iso,
                       "culture": data.rows[u].culture, "caption": data.rows[u].caption,
                       "has_image": data.rows[u].image_sha is not None}
                      for u in train_ids]
        blank = np.zeros((THUMB, THUMB, 3), dtype=np.uint8)
        train_images = np.stack([data.thumbs.get(u, blank) for u in train_ids])
        eval_rows = [{"index": j, "language": data.rows[u].language,
                      "iso_lang": data.rows[u].iso, "culture": data.rows[u].culture}
                     for j, u in enumerate(item_ids)]
        eval_images = np.stack([data.thumbs[u] for u in item_ids])
        medoids = medoid_captions([r["caption"] for r in train_rows], [data.rows[u].lang_dir for u in train_ids])
        targets = [data.rows[u].caption for u in item_ids]
        languages = [data.rows[u].lang_dir for u in item_ids]
        medoid_pred = [medoids.get(lang, "") for lang in languages]
        knn_pred = image_knn_captions(train_rows, train_images, eval_rows, eval_images, k=1, kind="all")
        medoid_scores = [chrf_pp(p, t) for p, t in zip(medoid_pred, targets)]
        knn_scores = [chrf_pp(p, t) for p, t in zip(knn_pred, targets)]
        per_language = {}
        for lang in sorted(set(languages)):
            sel = [i for i, value in enumerate(languages) if value == lang]
            per_language[lang] = {
                "n_items": len(sel),
                "medoid_reference_chrf_pp": float(np.mean([medoid_scores[i] for i in sel])),
                "image_knn_reference_chrf_pp": float(np.mean([knn_scores[i] for i in sel])),
                "n_visible_image_candidates": int(sum(1 for u in train_ids
                                                       if data.rows[u].lang_dir == lang and
                                                       data.rows[u].image_sha is not None)),
            }
        medoid_score = float(np.mean(medoid_scores))
        knn_score = float(np.mean(knn_scores))
        return {
            "split": split,
            "pool": pool,
            "n_items": len(item_ids),
            "n_languages": len(set(languages)),
            "n_train": len(train_ids),
            "n_train_images": sum(data.rows[u].image_sha is not None for u in train_ids),
            "n_train_caption_only": sum(data.rows[u].image_sha is None for u in train_ids),
            "medoid_reference_chrf_pp": medoid_score,
            "image_knn_reference_chrf_pp": knn_score,
            "image_knn_gain_over_medoid": float(knn_score - medoid_score),
            "per_language": per_language,
            "train_ids_sha256": ids_digest(train_ids),
            "split_ids_sha256": ids_digest(item_ids),
            "diagnostic_only": True,
        }

    # ------------------------------------------------------------ episode
    def _visible(self, data: _Data, pool: str, ep_key: str) -> tuple[list[str], list[str]]:
        rng = np.random.default_rng(stable_int(CODE, "visible", ep_key))
        langs = OOD_LANGS if pool == "ood" else IID_LANGS
        dev: list[str] = []
        train: list[str] = []
        per_lang = [self.n_dev // len(langs) + (1 if j < self.n_dev % len(langs) else 0) for j in range(len(langs))]
        for lang, nd in zip(langs, per_lang):
            imgs = list(data.train_images[pool].get(lang, []))
            perm = [imgs[j] for j in rng.permutation(len(imgs))]
            nd = min(nd, max(len(perm) - 1, 0))
            dev.extend(perm[:nd])
            train.extend(perm[nd:])
            train.extend(data.train_captions[pool].get(lang, []))
        return sorted(train), sorted(dev)

    def _episode(self, data: _Data, split: str, k: int, seed: int, items: list[str], requested: int) -> Episode:
        eid = episode_id(CODE, split, seed, k)
        pool = "ood" if split == "ood" else "iid"
        rows = data.rows
        ev = [rows[i] for i in items]
        n_items = len(ev)
        tr_ids, dev_ids = self._visible(data, pool, eid)
        tr = [rows[i] for i in tr_ids]
        dv = [rows[i] for i in dev_ids]
        n_tr, n_dev = len(tr), len(dv)
        blank = np.zeros((THUMB, THUMB, 3), dtype=np.uint8)

        def meta(r: _Row) -> dict:
            return {"language": r.language, "iso_lang": r.iso, "culture": r.culture}

        train_rows = [{"id": r.uid.rsplit("/", 1)[1], **meta(r), "caption": r.caption,
                       "has_image": r.image_sha is not None} for r in tr]
        train_imgs = np.stack([data.thumbs.get(r.uid, blank) for r in tr]) if tr else np.zeros((0, THUMB, THUMB, 3), np.uint8)
        dev_rows = [{"index": j, **meta(r)} for j, r in enumerate(dv)]
        dev_imgs = np.stack([data.thumbs[r.uid] for r in dv]) if dv else np.zeros((0, THUMB, THUMB, 3), np.uint8)
        eval_rows = [{"index": j, **meta(r)} for j, r in enumerate(ev)]
        eval_imgs = np.stack([data.thumbs[r.uid] for r in ev])

        medoids = Lazy(lambda: medoid_captions([r.caption for r in tr], [r.lang_dir for r in tr]))
        # ``clip_retrieval.retrieve_captions`` indexes its explicit fallback by
        # the policy-visible ``iso_lang`` field. Keep the evaluator's internal
        # language-directory medoids, but expose an ISO-keyed copy to the CLIP
        # route so a language with no visible image still receives its visible
        # medoid instead of an empty caption.
        medoids_by_iso = Lazy(lambda: {r.iso: medoids.get().get(r.lang_dir, "") for r in tr})

        def ref_for(r: _Row) -> str:
            return medoids.get().get(r.lang_dir, "")

        # ---- D_E tools
        def load_train(inputs: dict, config: dict) -> dict:
            return {"train": [dict(r) for r in train_rows], "train_images": train_imgs.copy()}

        def load_dev_inputs(inputs: dict, config: dict) -> dict:
            return {"dev_items": [dict(r) for r in dev_rows], "dev_images": dev_imgs.copy()}

        def score_dev(inputs: dict, config: dict) -> dict:
            pred, why = as_list(inputs.get("dev_captions"))
            if pred is None or len(pred) != n_dev or not all(isinstance(p, str) for p in pred):
                raise ValueError(f"dev_captions must be a list of {n_dev} strings ({why})")
            s = [chrf_pp(p.strip(), r.caption) for p, r in zip(pred, dv)]
            ref = [chrf_pp(ref_for(r), r.caption) for r in dv]
            return {"dev_mean_chrf_pp": float(np.mean(s)) if s else 0.0,
                    "dev_reference_mean_chrf_pp": float(np.mean(ref)) if ref else 0.0,
                    "dev_chrf_pp": np.asarray(s, dtype=float)}

        def load_eval_inputs(inputs: dict, config: dict) -> dict:
            return {"items": [dict(r) for r in eval_rows], "images": eval_imgs.copy()}

        def image_features(inputs: dict, config: dict) -> dict:
            X, names = image_descriptors(np.asarray(inputs.get("images")), str(config.get("kind", "all")))
            return {"X": X, "feature_names": names}

        def image_knn_reference(inputs: dict, config: dict) -> dict:
            """Predict from visible same-language image-caption neighbours only."""
            k = int(config.get("k", 1))
            kind = str(config.get("kind", "all"))
            pred = image_knn_captions(train_rows, train_imgs, eval_rows, eval_imgs, k=k, kind=kind)
            return {"captions": pred}

        def clip_retrieval_status(inputs: dict, config: dict) -> dict:
            """Report the exact staged frozen checkpoint without attempting to load or fetch it."""
            model = str(config.get("model", "open_clip_vit_b32"))
            return {"provenance": clip_retrieval.provenance(model)}

        def clip_knn_reference(inputs: dict, config: dict) -> dict:
            """Retrieve visible same-language captions with a staged frozen CLIP encoder."""
            model = str(config.get("model", "open_clip_vit_b32"))
            k = int(config.get("k", 1))
            selection = str(config.get("selection", "nearest"))
            pred = clip_retrieval.retrieve_captions(
                train_rows, train_imgs, eval_rows, eval_imgs, k=k, model=model,
                fallback=medoids_by_iso.get(), batch_size=int(config.get("batch_size", 32)), selection=selection,
            )
            return {"captions": pred, "provenance": clip_retrieval.provenance(model),
                    "selection": selection, "k": k}

        def clip_knn_reference_medoid(inputs: dict, config: dict) -> dict:
            """Fixed top-three visible-neighbour CLIP route for reproducible comparisons."""
            model = str(config.get("model", "open_clip_vit_b32"))
            k = 3
            selection = "caption_medoid"
            pred = clip_retrieval.retrieve_captions(
                train_rows, train_imgs, eval_rows, eval_imgs, k=k, model=model,
                fallback=medoids_by_iso.get(), batch_size=int(config.get("batch_size", 32)), selection=selection,
            )
            return {"captions": pred, "provenance": clip_retrieval.provenance(model),
                    "selection": selection, "k": k}

        meta_doc = "dict: index, language, iso_lang, culture"
        tools = [
            ToolSpec("load_train", f"{n_tr} visible labelled rows of the episode's languages: id, language, iso_lang, "
                     "culture, caption (target-language caption) and has_image; train_images[i] is the 64x64 RGB "
                     "thumbnail of row i (all zeros when has_image is false).",
                     {}, {"train": PortSchema("list", (n_tr,), dtype="dict", description="labelled caption rows"),
                          "train_images": PortSchema("array", (n_tr, THUMB, THUMB, 3), unit="intensity (0-255)",
                                                     dtype="uint8", description="RGB thumbnails")},
                     load_train),
            ToolSpec("load_dev_inputs", f"{n_dev} further training-partition images whose captions are withheld "
                     "(score captions for them with score_dev).",
                     {}, {"dev_items": PortSchema("list", (n_dev,), dtype="dict", description=meta_doc),
                          "dev_images": PortSchema("array", (n_dev, THUMB, THUMB, 3), unit="intensity (0-255)",
                                                   dtype="uint8")},
                     load_dev_inputs),
            ToolSpec("score_dev", "Mean sentence chrF++ (0-100, higher is better) of dev_captions (one caption per dev "
                     "item, same order) against the withheld dev captions, plus the score of the adapter's reference "
                     f"captions and per-item scores. Only {n_dev} items, so the mean is a noisy estimate.",
                     {"dev_captions": PortSchema("list", (n_dev,), dtype="str")},
                     {"dev_mean_chrf_pp": PortSchema("number", unit="chrF++ points"),
                      "dev_reference_mean_chrf_pp": PortSchema("number", unit="chrF++ points"),
                      "dev_chrf_pp": PortSchema("array", (n_dev,), unit="chrF++ points", dtype="float")},
                     score_dev),
            ToolSpec("load_eval_inputs", f"The {n_items} evaluation images (captions hidden) with their language "
                     "metadata, in the order that y must follow; images[i] is the 64x64 RGB thumbnail of items[i].",
                     {}, {"items": PortSchema("list", (n_items,), dtype="dict", description=meta_doc),
                          "images": PortSchema("array", (n_items, THUMB, THUMB, 3), unit="intensity (0-255)",
                                               dtype="uint8")},
                     load_eval_inputs),
            ToolSpec("image_features", "CPU image descriptors of RGB thumbnails: X[i] = feature vector of images[i].",
                     {"images": PortSchema("array", ("n", THUMB, THUMB, 3), dtype="uint8")},
                     {"X": PortSchema("array", ("n", "d"), dtype="float"),
                      "feature_names": PortSchema("list", ("d",), dtype="str")},
                     image_features,
                     config_doc="{kind: color_hist | gray_thumb | hsv_stats | all (default all)}"),
            ToolSpec("image_knn_reference", "Image-conditioned reference captions from cosine nearest neighbours in the "
                     "visible same-language training images. Caption-only rows are skipped; if no image exists for a "
                     "language, the visible-caption medoid is used. This uses no hidden evaluation labels and is "
                     "a reference/tool route, not a change to the official evaluator.",
                     {}, {"captions": PortSchema("list", (n_items,), dtype="str",
                                                  description="one visible-image neighbour caption per evaluation item")},
                     image_knn_reference,
                     config_doc="{k: positive integer (default 1), kind: color_hist | gray_thumb | hsv_stats | all}"),
            ToolSpec("clip_retrieval_status", "Read-only provenance for the optional frozen open CLIP ViT-B/32 "
                     "encoder. It reports the staged checkpoint SHA-256, package versions and availability; it never "
                     "downloads weights or reads captions/labels.",
                     {}, {"provenance": PortSchema("dict", description="auditable frozen-model status")},
                     clip_retrieval_status,
                     config_doc="{model: open_clip_vit_b32 (default)}"),
            ToolSpec("clip_knn_reference", "Image-conditioned reference captions from a locally staged, frozen open "
                     "CLIP encoder and cosine nearest neighbours among visible same-language image-caption rows. "
                     "Caption-only rows are excluded and a language with no visible image uses the visible medoid. "
                     "The call fails closed when the exact checkpoint or open_clip dependency is unavailable; inspect "
                     "clip_retrieval_status first. It never changes the official evaluator or acceptance rule.",
                     {}, {"captions": PortSchema("list", (n_items,), dtype="str",
                                                  description="one frozen-CLIP neighbour caption per evaluation item"),
                          "provenance": PortSchema("dict", description="checkpoint provenance for this retrieval"),
                          "selection": PortSchema("text", description="nearest or caption_medoid"),
                          "k": PortSchema("number", description="number of top visible neighbours considered")},
                     clip_knn_reference,
                     config_doc="{model: open_clip_vit_b32, k: positive integer (default 1), selection: nearest | caption_medoid (default nearest), batch_size: positive integer (default 32)}"),
            ToolSpec("clip_knn_reference_medoid", "Fixed image-conditioned reference route for controlled comparisons: "
                     "frozen CLIP cosine retrieval over the top three visible same-language image-caption rows, "
                     "followed by a deterministic caption medoid. If a language has fewer than three visible image "
                     "rows, all available rows are used. The route never reads evaluation captions or labels and "
                     "fails closed when the checkpoint or dependency is unavailable. Its k and selection are fixed "
                     "regardless of config, so a formal run can record an unambiguous route identifier.",
                     {}, {"captions": PortSchema("list", (n_items,), dtype="str",
                                                  description="one fixed top-three CLIP medoid caption per evaluation item"),
                          "provenance": PortSchema("dict", description="checkpoint provenance for this retrieval"),
                          "selection": PortSchema("text", description="always caption_medoid"),
                          "k": PortSchema("number", description="always 3; fewer candidates are handled per language")},
                     clip_knn_reference_medoid,
                     config_doc="{model: open_clip_vit_b32, batch_size: positive integer (default 32); k and selection are fixed}"),
        ]

        # ---- D_V constraints
        cap = self.max_caption_chars

        def c_captions(yv: Any, trace: Trace | None) -> tuple[bool, str]:
            lst, why = as_list(yv)
            if lst is None:
                return False, why
            bad = [j for j, t in enumerate(lst) if not isinstance(t, str) or not t.strip() or len(t) > cap]
            return not bad, ("all captions are non-empty strings" if not bad else
                             f"entries that are not non-empty strings of <= {cap} characters: positions {bad[:10]}")

        constraints = [
            c_list_length(n_items, "one caption per evaluation image"),
            ConstraintSpec("captions_valid", f"every entry of y is a non-empty string of at most {cap} characters",
                           c_captions),
        ]

        # ---- D_V evaluator
        def evaluate(yv: Any, trace: Trace | None) -> EvalResult:
            ref_scores = [chrf_pp(ref_for(r), r.caption) for r in ev]
            ref = float(np.mean(ref_scores))
            knn_y = image_knn_captions(train_rows, train_imgs, eval_rows, eval_imgs, k=1, kind="all")
            knn_scores = [chrf_pp(p, r.caption) for p, r in zip(knn_y, ev)]
            payload = {"item_ids": list(items), "languages": [r.iso for r in ev], "chrf_pred": None,
                       "chrf_ref": ref_scores, "chrf_knn_ref": knn_scores}
            base = {"reference": ref, "reference_name": "per-language medoid caption of the visible training rows",
                    "margin": self.margin, "n_items": n_items, "items_requested": requested}
            lst, why = as_list(yv)
            if lst is not None and len(lst) != n_items:
                lst, why = None, f"length {len(lst)} != {n_items}"
            if lst is not None and not all(isinstance(t, str) for t in lst):
                lst, why = None, "non-string entries"
            if lst is None:
                return EvalResult(metrics={"reference_mean_chrf_pp": ref}, primary=None, direction="max",
                                  accepted=False, details={**base, "norm_score": 0.0, "pooled_payload": payload,
                                                           "invalid": why})
            s = [chrf_pp(t.strip(), r.caption) for t, r in zip(lst, ev)]
            score = float(np.mean(s))
            payload["chrf_pred"] = s
            per_lang = {lang: float(np.mean([v for v, r in zip(s, ev) if r.iso == lang]))
                        for lang in sorted({r.iso for r in ev})}
            stripped = [t.strip() for t in lst]
            distinct_per_lang = {lang: len({t for t, r in zip(stripped, ev) if r.iso == lang}) for lang in per_lang}
            diagnostics = {"n_distinct_captions": len(set(stripped)),
                           "distinct_caption_ratio": len(set(stripped)) / n_items,
                           "distinct_captions_per_language": distinct_per_lang,
                           "image_blind": bool(all(v == 1 for v in distinct_per_lang.values())),
                           "mean_caption_words": float(np.mean([len(t.split()) for t in stripped])),
                           "image_knn_reference_mean_chrf_pp": float(np.mean(knn_scores)) if knn_scores else 0.0,
                           "image_knn_reference_per_language": {
                               lang: float(np.mean([v for v, r in zip(knn_scores, ev) if r.iso == lang]))
                               for lang in sorted({r.iso for r in ev})
                           },
                           "image_knn_reference_k": 1,
                           "image_knn_reference_feature": "all"}
            return EvalResult(metrics={"mean_chrf_pp": score, "reference_mean_chrf_pp": ref}, primary=score,
                              direction="max", accepted=bool(score >= ref + self.margin),
                              details={**base, "norm_score": norm_score(score, ref, "max"), "per_language": per_lang,
                                       "diagnostics": diagnostics, "pooled_payload": payload})

        langs = sorted({r.language for r in ev})
        with_img: dict[str, list[int]] = {}
        for r in tr:
            cnt = with_img.setdefault(r.iso, [0, 0])
            cnt[0] += r.image_sha is not None
            cnt[1] += 1
        img_doc = ", ".join(f"{iso} {a}/{b}" for iso, (a, b) in sorted(with_img.items()))
        objective = (
            "Indigenous studies - culturally grounded image captioning in Indigenous languages of the Americas "
            "(AmericasNLP 2026 shared task). Each evaluation item is a photograph depicting the culture of an "
            "Indigenous community, given as a 64x64 RGB thumbnail with its target language, ISO 639-3 code and "
            f"culture (languages in this episode: {', '.join(langs)}). The hidden target is the reference caption "
            "written by speakers in that language (one to a few sentences describing the image and its cultural "
            "context). The image is only available as pixels; no text description of the image is provided.\n"
            f"Visible data: load_train returns {n_tr} labelled rows of the same languages (caption, language "
            "metadata, thumbnail when available; has_image is true for "
            f"{sum(a for a, _ in with_img.values())} of the {n_tr} rows - per language, rows with an image / rows: "
            f"{img_doc} - and the thumbnail of the other rows is all zeros); load_dev_inputs returns {n_dev} further "
            f"images whose captions are withheld and score_dev scores captions for them (it covers only {n_dev} items, "
            "so its mean is a noisy check); load_eval_inputs returns the evaluation images; "
            "image_features computes colour/texture descriptors and image_knn_reference retrieves captions from "
            "same-language visible image neighbours. clip_retrieval_status reports whether an operator-staged, "
            "hashable frozen open CLIP checkpoint is available; when it is available, call clip_retrieval_status "
            "then clip_knn_reference_medoid and wire its captions directly to submit.y. This fixed route uses "
            "the top three visible same-language CLIP neighbours and a deterministic caption medoid; its k and "
            "selection cannot be changed through config. Finish immediately after that frozen visual route: do not "
            "replace its captions with an image-blind string, later code node, or an untracked CPU approximation. "
            "clip_knn_reference_medoid fails closed when the exact checkpoint or dependency is unavailable; only "
            "then use the visible-data fallback routes.\n"
            f"Deliverable y: a list of {n_items} strings; y[i] is the caption, in the language of items[i], for "
            "images[i] from load_eval_inputs (same order).\n"
            "Evaluation: mean over items of sentence-level chrF++ (sacrebleu CHRF with character 6-grams, word "
            "2-grams, beta=2; 0-100 scale) against the hidden reference caption.\n"
            + scilib.describe("captions")
        )
        lineage = {
            "dataset": "AmericasNLP 2026 Cultural Image Captioning", "version": "commit 7ff73013eca06cc54ac1bdaa4e6edbd4154d08f0",
            "source_url": "https://github.com/AmericasNLP/americasnlp2026", "license": "CC BY-NC 4.0",
            "receipt": receipt_info(self.root / "receipt.json"),
            "pool": pool, "languages": [r.iso for r in ev],
            "pool_source": "official dev partition (only partition with public captions)",
            "ood_kind": "proxy_within_dataset" if pool == "ood" else None,
            "ood_shift": ("held-out Uto-Aztecan languages (Central/Orizaba Nahuatl, Wixárika) never used in src/val/id "
                          "episodes") if pool == "ood" else None,
            "split": split, "split_seed": seed, "partition_seed": self.partition_seed, "index": k,
            "item_ids": list(items), "n_items": n_items, "items_requested": requested,
            "items_per_episode_capped": n_items < requested,
            "image_groups": sorted({str(r.image_sha) for r in ev}),
            "train_source": "train partition of the same languages' dev rows + caption-only rows (+ Wixárika pilot)",
            "n_train": n_tr, "n_dev": n_dev, "train_ids_sha256": ids_digest(tr_ids), "dev_item_ids": list(dev_ids),
            "rebuilt_split": True, "historical_ids_recovered": False,
        }
        return Episode(
            id=eid, discipline=CODE, family=FAMILY, split=split, task_type=self.task_type, objective=objective,
            required_output=PortSchema("list", (n_items,), dtype="str",
                                       description="caption per evaluation image, load_eval_inputs order"),
            tools=tools, constraints=constraints,
            budget=self.budget or Budget(max_steps=12, max_policy_tokens=200_000, max_wall_s=1800.0, max_node_s=600.0,
                                         max_llm_items=8 * max(n_items, n_dev)),
            lineage=lineage,
            acceptance=(f"mean sentence chrF++ >= reference + {self.margin:g} points; reference = per-language medoid "
                        "caption of the visible training captions"),
            tolerance={"rtol": 1e-6, "atol": 1e-6},
            tags=[CODE, "indigenous-languages", "image-captioning", "low-resource", "chrF++", "AmericasNLP",
                  "multilingual", *sorted({r.iso for r in ev}), pool],
            metric=self.metric, direction=self.direction, n_items=n_items,
            _evaluate=evaluate, _dev_evaluate=None,
        )


Adapter = AmericasNLPAdapter

__all__ = ["Adapter", "AmericasNLPAdapter", "chrf_pp", "image_descriptors", "image_knn_captions",
           "medoid_captions", "PoolExhausted"]
