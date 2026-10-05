---
name: scienceclaw-benchmark-for36
description: "Use when a task asks to separate stereo music mixtures into four stems (vocals, drums, bass, other) and deliver the estimated stems, optionally with training mixtures plus reference stems, judged by BSSEval-v4 source-to-distortion ratio (SDR, dB), as in MUSDB18 music source separation (FoR36 of the companion ScienceClaw-Eval benchmark)."
metadata: { "openclaw": { "emoji": "📊" } }
---

# Music source separation: vocals / drums / bass / other

## Task
- Input: stereo mixtures `(k, n, 2)` (linear PCM, sample rate given by the task); optionally training mixtures with stems `(m, 4, n, 2)`.
- Deliverable: float estimates `(k, 4, n, 2)`, axis 1 in the order (vocals, drums, bass, other), same rate and length as the input, on the amplitude scale of the mixture. Finite, and no estimated target exactly zero over a whole 1-s window (the metric skips such windows).
- Quality: mean over the four targets of the median over items of the median over 1-s windows of SDR = 10 log10(sum ref^2 / sum (est - ref)^2), dB, higher is better (`scilib.audiosep.sdr_scores`).

## Routes
- Frozen neural separators: `scilib.audiosep_pretrained.separate_pretrained(mixtures, model, sample_rate)` with `htdemucs`, `htdemucs_ft` or `mdx_extra` (asset `demucs`); `scilib.scnet_pretrained.separate_pretrained(mixtures, model="mimo_scnet_small")` (asset `scnet_mimo_small`). Operators `music_source_separation_htdemucs`, `music_source_separation_htdemucs_ft`, `music_source_separation_scnet`. Check `audiosep_pretrained.available(model)`, `scnet_pretrained.available()` and `scienceclaw_tools(operation=weights)` first.
- Trained CPU baseline: `audiosep.separate(train_mixtures, train_stems, mixtures)` (STFT ratio masks from boosted trees; operator `music_source_separation_softmask`; at least 2 training excerpts). `audiosep.cross_validate(..., groups=track_ids)` gives grouped k-fold SDR. Score with `audiosep.sdr_scores` (operator `music_separation_sdr`).

## Pitfalls
- SDR is not scale invariant: an almost-zero estimate scores about 0 dB and copying the mixture into every stem scores strongly negative, so a level change alone moves the score. `audiosep.gain_only` and `cross_validate` report `mixture_sdr`, `null_sdr` and `gain_only_sdr` beside the separator; separate level effect from separation and never submit near-silent stems.
- Pretrained overlap: `htdemucs` was trained on MUSDB18-HQ plus further songs; assume the other Demucs bags overlap likewise. MUSDB18 training tracks (including visible training or validation excerpts) were seen in training and score higher than unseen songs; disclose this, and check the training data of any other checkpoint (e.g. SCNet) before calling it clean.
- Never use reference stems or oracle masks of the items being separated to build the deliverable; keep all excerpts of a track in one fold.
- The wrappers already return the input length and the order (vocals, drums, bass, other); do not permute again. A silent reference stem gives NaN windows that the median ignores.
