# FoR36 Creative arts and writing — MUSDB18 four-stem separation (mean target-median SDR in dB, higher is better)

Adapter: `scienceclaw/bench/tasks/for36_musdb.py` (`Adapter = MusdbAdapter`).
Status: **available** on the data team's `full_v1` (all 150 official tracks) as of 2026-09-28. `build_episodes("ood")`
returns `[]` (there is no OOD collection); the other three splits are complete.
Tests: `tests/test_task_FoR36.py` (skip when `full_v1` or a runnable ffmpeg is missing; ~20 s on a cold excerpt
cache, a few seconds warm).

## Dataset
| field | value |
|---|---|
| name | MUSDB18 v1.0.0 (SiSEC 2018 music separation corpus; 150 tracks: 100 official train / 50 official test; stems mixture, drums, bass, other, vocals) |
| URL | https://zenodo.org/records/1117372 (doi 10.5281/zenodo.1117372); config https://github.com/sigsep/sigsep-mus-db (`mus.yaml`: stem ids mixture 0, drums 1, bass 2, other 3, vocals 4) |
| license | MUSDB18 terms (academic use) |
| local copy (data-team receipt 2026-09-28) | `<DATA_ROOT>/for36-musdb18/full_v1/`: `original/{train,test}/*.stem.mp4` (AAC multi-stream MP4, 44.1 kHz stereo, complete tracks), `roles/{source,validation,ID,reserve,OOD}.jsonl`, `sampling-manifest.json`, `loader-config.json`. The old 16-clip `preview/` folder is no longer used. |
| integrity | role files are checked against the sha256 and counts of `sampling-manifest.json`; decoded PCM against `decoded_pcm_sha256` of the role files (bit-exact, see Decoding) |

## Roles, pools and splits (as implemented)
Lineage unit = track; the data team's roles are track-disjoint (asserted by the loader).

| data-team role (tracks) | adapter pool | excerpts / track | excerpts | use |
|---|---|---|---|---|
| `source` (64 official-train tracks) | `src` | 2 | 128 | source episodes (a round-major stream; recycles only after 8 episodes of 16) |
| `validation` (14 package validation tracks) | `val` | 5 | 70 | val episodes (up to 4 disjoint episodes of 16) |
| `ID` (50 official test tracks) | `id` | 3 | 150 | id episodes (up to 9 disjoint episodes of 16) |
| `OOD` (empty) | `ood` | – | 0 | `build_episodes("ood", ...)` returns `[]` |
| `reserve` (22 train tracks) | visible `train` (16 tracks) and `dev` (6 tracks) | 4 | 62 + 24 | `load_train` / `load_dev_inputs` / `score_dev` |

* **Reserve choice (owner-vetoable).** The data team labelled `reserve` "unused". It is *not* evaluated; it is cut
  by track (sha256, partition seed 20260928) into 16 training and 6 dev tracks that serve as the visible data D_E
  of every episode (8 training and 4 dev excerpts per episode, seeded). Hence D_E is track-disjoint from every
  src / val / id item, and no evaluated split lost a track. The alternative (reserve as extra src / val tracks, per
  the data team's note that they may be used as such) would leave the visible data to be drawn from the src tracks,
  which is then no longer track-disjoint from the src episodes; if the owner prefers it, change `ROLE_POOL`,
  `EXCERPTS_PER_TRACK` and `RESERVE_SPLIT` in the adapter.
* **OOD.** `roles/OOD.jsonl` is empty by design ("no independent cross-dataset collection supplied; no random split
  is mislabeled OOD"). `SplitPlan.build` tolerates it (warning `FoR36/ood: adapter returned 0 of N requested
  episodes`; the runner then has no OOD episodes for this discipline). A genuine OOD set would need another
  multitrack corpus (MoisesDB, MedleyDB 2.0).
* **Item** = one 6-second stereo excerpt; slots are 6-s aligned, non-overlapping and chosen by sha256 rank per track
  (prefix-stable, seed 20260928). Item id = `<track id>@<start second>`, e.g. `musdb18/test/PR - Oh No@54`.
  Very short tracks give fewer slots (`Music Delta - Rock`, 12.9 s, gives 2 of its 4).
* Episode = 16 excerpts, block-wise from a seeded permutation (prefix-stable; src may recycle in later cycles).
  With the default protocol (16 items, R = 7, n_val = 2, n_id = 4): 7 of 8 src, 2 of 4 val, 4 of 9 id episodes.
  Asking for more val/id episodes raises `PoolExhausted` (repo convention).

## Decoding
The stems are AAC (`.stem.mp4`, stream 0 mixture, 1 drums, 2 bass, 3 other, 4 vocals). The main venv has only numpy
and scipy (no soundfile / av / librosa / torch), so the adapter runs an **ffmpeg subprocess**:
* binary: `$SCIENCECLAW_FFMPEG`, else the data team's pinned binary from `full_v1/loader-config.json`
  (`ffmpeg-macos-aarch64-v7.1` under `<DATA_ROOT>/../environments/data/...`, sha256 `6d175a47...`), else `ffmpeg` on
  PATH, else `imageio_ffmpeg`. **On a machine without that binary (e.g. a Linux cluster node) set `SCIENCECLAW_FFMPEG`
  or copy the excerpt cache**; with any other build decoding still works but is not bit-exact-verified
  (`lineage.decode_verified = false`, separate cache tag).
* each stream: `ffmpeg -map 0:a:<i> -c:a pcm_f32le -f f32le -` (native 44.1 kHz float32; the five streams of a track
  run in parallel), sha256 compared with the role file's `decoded_pcm_sha256` (pinned binary only), then the
  6-s slots are cut and **decimated to 22,050 Hz with `scipy.signal.resample_poly(x, 1, 2)`** (4096 samples of
  context around each slot; identical to resampling the whole track). ~2 s per track.
* per-track excerpts are cached in `cache/tasks/FoR36/excerpts_v1_22050_6s/*.npz` (float32; ~2.3 GB if all 158
  tracks are touched; override the location with `SCIENCECLAW_TASK_CACHE`) and in a 16-track in-memory LRU. Episodes
  do not retain audio, so many episodes can be alive at once.
* The AAC mixture is not exactly the sum of the AAC stems (data-team limitation note). Measured over all 356
  excerpts (2026-09-28): SNR of `mixture` vs `sum(stems)` has median 31.8 dB (the tests only assert > 10 dB on a
  few excerpts); some excerpts have a silent stem (minimum -inf), which is where the NaN scores below come from.

## Visible data and tools (D_E)
| tool | returns |
|---|---|
| `load_train` | 8 training excerpts: `mixtures` (m, 132300, 2), `stems` (m, 4, 132300, 2) in order vocals/drums/bass/other, `track_ids` (m opaque labels `track0`, `track1`, ...; equal label = same track, so that a cross-validation can keep a track together), `sample_rate` 22050 |
| `load_dev_inputs` | 4 dev `dev_mixtures` (stems withheld) |
| `score_dev(dev_estimates)` | `dev_sdr`, `dev_reference_sdr`, per-target medians (visible dev signal; `_dev_evaluate` is None) |
| `load_eval_inputs` | the 16 evaluation `mixtures` (16, 132300, 2) float32, in output order |

Sizes for a code node: evaluation inputs 17 MB, training excerpts 42 MB, dev inputs 4 MB (float32).

## Domain library `scilib.audiosep` (documented in the objective)
`scilib/audiosep.py` (tests `tests/test_scilib_audiosep.py`; its module docstring is appended to the objective through
`scilib.describe("audiosep")`, as for FoR50). Pure numpy/scipy/LightGBM, reads no files, deterministic, 2 threads:
* `sdr_scores(references, estimates)`: the adapter's metric, bit-identical to `item_scores` / `aggregate` (tested,
  including the NaN / +inf / silent-window rules).
* `separate(train_mixtures, train_stems, mixtures)`: the single entry point (`mixtures` may be a list, e.g. dev + eval).
  An STFT (2048, hop 1024) ratio-mask separator: 16 per-bin features of the mixture, one LightGBM regressor per target
  trained on the ideal ratio mask of the visible training stems (bins weighted by their magnitude, since SDR is an
  energy-weighted error), masks raised to 1.5, renormalised over the four targets, mixture phase. About 5 s to fit and
  2 s per excerpt (<= 100 s for 4 dev + 16 eval excerpts on a loaded machine).
* `cross_validate(..., groups=track_ids)`: grouped k-fold over the visible training excerpts; scores the separator,
  the constant-gain estimate and the unscaled mixture on the same held-out excerpts.
* `gain_only(...)`, `oracle_mask_sdr(...)`, `stft` / `istft`.

### Level artefact of the metric (integrity finding, unchanged by the library)
SDR compares an estimate with the reference sample by sample, so it is **not scale invariant**, and the reference
(mixture as the estimate of every target, about -5.5 dB) contains the three other sources at full amplitude. As a
consequence a large part of the +3 dB acceptance margin is obtained by amplitude alone, without separating anything:
* an estimate with (near) zero amplitude has SDR 0 dB in every window: the null estimate (the 1e-7 offset that satisfies
  `no_silent_window`) scores ~0 dB, i.e. ~+5.5 dB over the reference, and is `accepted`;
* one constant gain per target fitted on the visible training excerpts (`gain_only`) scores ~+0.8 dB;
* the separator adds a further ~+2 to +2.5 dB on top of the constant gain (measured on the visible reserve data with 8
  fitting excerpts: mixture -5.4, null 0.0, constant gain +0.8, separator +3.2 dB on 24 held-out reserve excerpts; other draws of the
  8 fitting excerpts give +2.0 to +3.2).
`cross_validate` and `gain_only` therefore report `mixture_sdr`, `null_sdr` and `gain_only_sdr` beside the separator's
SDR, so that a result can be split into level effect and separation. The metric, the reference, the acceptance margin
and the constraints are deliberately unchanged (owner decision pending: a scale-invariant score such as SI-SDR, or a
reference/acceptance that a no-signal estimate cannot meet, would remove the artefact).

## Metric, reference, acceptance (D_V)
* **BSSEval v4 SDR** exactly as museval (`museval.evaluate(..., mode="v4")` → `metrics.bss_eval(window=hop=1 s,
  framewise_filters=False, bsseval_sources_version=False)`): in the image decomposition
  `e_spat + e_interf + e_artif = ŝ − s`, so per 1-s window `SDR = 10 log10(Σ s² / Σ (ŝ − s)²)` over both channels;
  windows where any reference source or any estimated source is all-zero are NaN; a perfect estimate is +inf.
  Verified against museval 0.4.1 `bss_eval` on random multichannel signals (max abs difference 2e-15 dB; a fixed
  regression case is in the tests).
* Item score per target = median over windows; **episode primary = mean over the 4 targets of the median over
  items** (SiSEC 2018 "median over frames, median over tracks"). Pooled metric = the same over all items (invalid
  episodes contribute the reference item scores).
* **Reference baseline (cheap):** SiSEC "MIX" anchor — the mixture as the estimate of every target.
* **Measured reference (mixture as estimate, all excerpts of the pool, item scores pooled; 2026-09-28):**
  src -5.87 dB, val -5.51 dB, id -5.81 dB (train -5.97, dev -5.44). Every pool has finite aggregate scores for all
  four targets. Some excerpts have a NaN score for at least one target (no valid 1-s window, because a stem is
  silent in every window: src 15/128, val 12/70, id 33/150 excerpts); NaN item scores are ignored by the median
  over items, as in museval.
* **Acceptance:** `primary >= reference + 3 dB`. `norm_score = 10^((primary − reference)/10)` clipped to [0, 10]
  (power-ratio analogue of primary/reference; +inf → 10).
* **Hard constraints:** `output_shape` (16, 4, 132300, 2); `finite`; `no_silent_window` (no estimated target is
  exactly zero over a whole 1-s window — closes the museval loophole of skipping windows with silent estimates).

## Budget
`max_steps 12, max_policy_tokens 200k, max_wall_s 2400, max_node_s 600, max_llm_items 16` (a signal-processing
task; LLM nodes are not needed per item). SDR evaluation of 16 excerpts takes < 0.1 s once the audio is cached.

## Deviations from the historical protocol
* Not a recovery of the historical sample ids or scores (the data team's partition is a new normative one).
* 22.05 kHz instead of 44.1 kHz; 6-s excerpts instead of full tracks; source tracks contribute 2 excerpts each.
* No OOD split (empty by design); the paper's OOD numbers for FoR36 cannot be reproduced from this data.
* `configs/episode-designs/FoR36.json` does not exist in this checkout; the role files, `README.md` and
  `catalog.json` of `full_v1` served as the design.
