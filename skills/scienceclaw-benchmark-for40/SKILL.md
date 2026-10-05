---
name: scienceclaw-benchmark-for40
description: "Use when a task asks for unsupervised anomalous sound detection for one machine type in the first-shot setting (only normal source-domain training clips, no labelled anomalies, unlabeled test clips) and wants one anomaly score per test clip, judged by the DCASE 2024 Task 2 score (harmonic mean of source AUC, target AUC and pAUC); machine condition monitoring from audio with log-mel or pretrained AST / CLAP embeddings (FoR40 of the companion ScienceClaw-Eval benchmark)."
metadata: { "openclaw": { "emoji": "📊" } }
---

# First-shot anomalous sound detection (DCASE 2024 Task 2)

## Task
- Input: normal training clips of one machine type (`waveforms (n, T)` float, zero-padded, with `lengths` and `sample_rate`, 16 kHz mono in DCASE) and unlabeled test clips. There is no labelled anomaly and no target-domain training clip, so the domain shift must be inferred from the data.
- Deliverable: a 1-D finite float array, one anomaly score per test clip in input order, larger = more anomalous; only the ordering matters. Score each machine type separately with its own training clips.
- Quality: official DCASE score = harmonic mean of AUC over source-domain normals plus anomalies, AUC over target-domain normals plus anomalies, and pAUC (max FPR 0.1) over all clips; higher is better. `scilib.anomsound.official_score(y_true, domain, score)` computes it when labels and domains exist.

## Routes
- Log-mel route: `anomsound.log_mel_list(waveforms, lengths, sample_rate)` (n_fft 1024, hop 512, 128 mels, dB; operator `audio_log_mel_spectrogram`), then `component_scores` (`nn_train`, `nn2_pool`, `nn_pool`, `lof`, `band_max`, `maha`) and `rank_average`. One call: `anomsound.fit_predict(train_mels, eval_mels)` (operator `sound_anomaly_scores_log_mel`) or `anomsound.score_clips(train, evalset)`, using `DEFAULT_MEMBERS`. `nn_train` (1-nearest-neighbour distance to the training clips) is the simple baseline.
- Pretrained-embedding route: `scilib.audioenc.embed(waveforms, lengths, sample_rate, model, kind)` with `ast_audioset` (`pooled` or `logits`) or `clap_htsat` (`pooled` or `proj`) (operators `audio_embedding_ast`, `audio_embedding_clap`), then `anomsound.embedding_scores(train_emb, eval_emb)` and `rank_average(raw, ("nn2_pool",))` (operator `sound_anomaly_scores_embeddings`), or `score_clips(train, evalset, embed=["ast_audioset", "clap_htsat"])`. Assets `ast_audioset`, `clap_htsat_unfused`; `beats.encode` (asset `beats_iter3`, mean-pool its tokens) can also feed `embedding_scores`. Check `audioenc.available()` and `scienceclaw_tools(operation=weights)`. It is a candidate to compare with the log-mel route, not a guaranteed replacement.

## Pitfalls
- `nn2_pool`, `nn_pool` and `lof` place the unlabeled test clips in the point cloud together with the training clips. That uses no labels and is fine; but never use file names, ids or metadata that encode domain or condition, and never fit on test clips with guessed labels.
- A single component can be anti-correlated with the anomalies on a small test set, and AUC from a few dozen clips is noisy. A rank average of complementary clip-level scores is more stable; do not pick components by peeking at test labels.
- Clips differ in length (use `lengths`; `log_mel_list` cuts to the true length, so padding is not scored), and the clip descriptors are z-scored with training statistics only (at least 3 training clips are needed).
- Disclose possible pretraining overlap: AST was fine-tuned on AudioSet and CLAP trained on LAION-Audio-630K; overlap with the task audio cannot be ruled out.
