"""Typed operators for audio: pretrained clip encoders, music source separation and anomalous sound detection."""
from __future__ import annotations

from typing import Any

from scienceclaw.program.specs._common import p

_WAVES = p("array", "float waveforms in [-1, 1], one clip per row (zero-padded to the longest)", shape=("n_clips", "n_samples"), dtype="float")
_RATE = p("number", "sampling rate of the waveforms", unit="Hz")
_LENGTHS = p("array", "true number of samples of every clip (rows of the waveforms may be zero-padded)", shape=("n_clips",), dtype="int")
_MIX = p("array", "stereo mixtures, linear PCM amplitude", shape=("k", "n", 2), dtype="float")
_STEMS = p("array", "reference stems; axis 1 = (vocals, drums, bass, other)", shape=("k", 4, "n", 2), dtype="float")
_EST = p("array", "estimates; axis 1 = (vocals, drums, bass, other), on the amplitude scale of the mixture", shape=("k", 4, "n", 2), dtype="float")

# id, tool, description, inputs, outputs, body of run(), pre, post, applicability tags
SPECS: list[dict[str, Any]] = [
    dict(id="audio_embedding_clap", tool="audioenc.embed",
         description="Clip embeddings of audio waveforms in the joint audio-text space of the pretrained CLAP HTS-AT audio encoder (LAION-Audio-630K), "
                     "512-dimensional; cosine similarity of clips for sound retrieval, clustering and nearest-neighbour or anomaly scoring.",
         inputs={"waveforms": _WAVES, "sample_rate": _RATE},
         outputs={"embeddings": p("array", "one vector per clip (not normalised)", shape=("n_clips", 512), dtype="float")},
         code="from scilib import audioenc\nout = audioenc.embed(inputs['waveforms'], sample_rate=float(inputs['sample_rate']), model='clap_htsat', kind='proj')\nreturn {'embeddings': out}",
         pre=[{"port": "waveforms", "check": "finite"}], post=[{"port": "embeddings", "check": "finite"}],
         tags=["audio", "embedding", "pretrained", "clap", "retrieval"]),
    dict(id="audio_event_logits_ast", tool="audioenc.embed",
         description="AudioSet sound-event logits (527 classes: speech, music, animals, machines, environmental sounds) of audio waveforms from the pretrained "
                     "Audio Spectrogram Transformer; multi-label audio tagging and sound classification scores, higher = more likely present.",
         inputs={"waveforms": _WAVES, "sample_rate": _RATE},
         outputs={"logits": p("array", "one logit per AudioSet class and clip, mean over the analysis windows of the clip", shape=("n_clips", 527), dtype="float")},
         code="from scilib import audioenc\nout = audioenc.embed(inputs['waveforms'], sample_rate=float(inputs['sample_rate']), model='ast_audioset', kind='logits')\nreturn {'logits': out}",
         pre=[{"port": "waveforms", "check": "finite"}], post=[{"port": "logits", "check": "finite"}],
         tags=["audio", "classification", "tagging", "pretrained", "ast", "audioset"]),
    dict(id="audio_embedding_beats", tool="beats.encode",
         description="Frozen BEATs (iter3) self-supervised audio embeddings: 768-dimensional patch tokens of the waveform and their mean over time as one "
                     "clip vector; resamples to 16 kHz first; clips of at least one second.",
         inputs={"waveforms": p("array", "float waveforms in [-1, 1], one equal-length clip per row", shape=("n_clips", "n_samples"), dtype="float"),
                 "sample_rate": _RATE},
         outputs={"embeddings": p("array", "mean over the tokens of a clip", shape=("n_clips", 768), dtype="float"),
                  "frame_embeddings": p("array", "token sequence of a clip (about 49.6 tokens per second)", shape=("n_clips", "n_tokens", 768), dtype="float")},
         code=("import math\nimport numpy as np\nfrom scipy.signal import resample_poly\nfrom scilib import beats\n"
               "x = np.asarray(inputs['waveforms'], dtype=np.float32)\nrate = int(round(float(inputs['sample_rate'])))\n"
               "if rate != beats.SAMPLE_RATE:\n    g = math.gcd(rate, beats.SAMPLE_RATE)\n"
               "    x = resample_poly(x, beats.SAMPLE_RATE // g, rate // g, axis=1).astype(np.float32)\n"
               "frames = beats.encode(x, beats.SAMPLE_RATE)\nreturn {'embeddings': frames.mean(axis=1), 'frame_embeddings': frames}"),
         pre=[{"port": "waveforms", "check": "finite"}], post=[{"port": "embeddings", "check": "finite"}, {"port": "frame_embeddings", "check": "finite"}],
         tags=["audio", "embedding", "pretrained", "beats", "self-supervised"]),
    dict(id="audio_log_mel_spectrogram", tool="anomsound.log_mel_list",
         description="Log-mel power spectrogram in dB (1024-point Hann window, hop 512, 128 mel bands) of every audio clip, cut to its true length; "
                     "input features for anomalous sound detection and audio classifiers.",
         inputs={"waveforms": _WAVES, "lengths": _LENGTHS, "sample_rate": _RATE},
         outputs={"log_mel": p("list", "one float array (frames, 128) in dB per clip, frames = 1 + length // 512; floor -100 dB")},
         code="from scilib import anomsound\nout = anomsound.log_mel_list(inputs['waveforms'], inputs['lengths'], sample_rate=float(inputs['sample_rate']), n_fft=1024, hop=512, n_mels=128)\nreturn {'log_mel': out}",
         pre=[{"port": "waveforms", "check": "finite"}], post=[{"port": "log_mel", "check": "nonempty"}],
         tags=["audio", "spectrogram", "mel", "features", "anomaly detection"]),
    dict(id="sound_anomaly_scores_log_mel", tool="anomsound.fit_predict",
         description="Unsupervised anomalous sound detection for one machine type: anomaly score of every unlabeled test clip from normal-only training clips "
                     "and log-mel spectrograms (nearest-neighbour distances, local outlier factor, per-band and Mahalanobis distances, rank-averaged); "
                     "needs at least 3 training clips; larger = more anomalous.",
         inputs={"train_log_mel": p("list", "log-mel arrays (frames, n_mels) in dB of the normal training clips"),
                 "eval_log_mel": p("list", "log-mel arrays (frames, n_mels) in dB of the test clips, same n_mels")},
         outputs={"scores": p("array", "rank-averaged anomaly score in (0, 1], only the ordering is meaningful", shape=("n_eval",), dtype="float")},
         code="from scilib import anomsound\nreturn {'scores': anomsound.fit_predict(inputs['train_log_mel'], inputs['eval_log_mel'])}",
         pre=[{"port": "train_log_mel", "check": "nonempty"}, {"port": "eval_log_mel", "check": "nonempty"}],
         post=[{"port": "scores", "check": "finite"}, {"port": "scores", "check": "range", "value": [0.0, 1.0]}],
         tags=["audio", "anomaly detection", "outlier", "unsupervised", "dcase", "machine condition monitoring"]),
    dict(id="sound_anomaly_scores_embeddings", tool="anomsound.embedding_scores",
         description="Unsupervised anomalous sound detection from clip embeddings (AST, CLAP, BEATs or any encoder): distance of every unlabeled test clip to the "
                     "second-nearest neighbour among the normal training clips and the other test clips (L2-normalised rows), converted to a rank score; "
                     "needs at least 3 training clips; larger = more anomalous.",
         inputs={"train_embeddings": p("array", "embeddings of the normal training clips", shape=("n_train", "d"), dtype="float"),
                 "eval_embeddings": p("array", "embeddings of the test clips, same dimension", shape=("n_eval", "d"), dtype="float")},
         outputs={"scores": p("array", "rank score in (0, 1], only the ordering is meaningful", shape=("n_eval",), dtype="float")},
         code="from scilib import anomsound\nraw = anomsound.embedding_scores(inputs['train_embeddings'], inputs['eval_embeddings'])\nreturn {'scores': anomsound.rank_average(raw, ('nn2_pool',))}",
         pre=[{"port": "train_embeddings", "check": "finite"}, {"port": "eval_embeddings", "check": "finite"}],
         post=[{"port": "scores", "check": "finite"}, {"port": "scores", "check": "range", "value": [0.0, 1.0]}],
         tags=["audio", "anomaly detection", "outlier", "embedding", "nearest neighbour", "unsupervised", "dcase"]),
    dict(id="sound_anomaly_detection_score", tool="anomsound.official_score",
         description="DCASE anomalous-sound-detection metric of anomaly scores against known labels: AUC on source-domain and target-domain normals plus "
                     "partial AUC (false-positive rate up to 0.1) and their harmonic mean.",
         inputs={"y_true": p("array", "1 = anomalous clip, 0 = normal clip", shape=("n",), dtype="int"),
                 "domain": p("list", "'source' or 'target' domain of every clip"),
                 "scores": p("array", "anomaly score of every clip, larger = more anomalous", shape=("n",), dtype="float")},
         outputs={"official_score": p("number", "harmonic mean of auc_source, auc_target and pauc, in [0, 1]", unit="1"),
                  "metrics": p("dict", "auc_source, auc_target, pauc and official_score")},
         code="from scilib import anomsound\nm = anomsound.official_score(inputs['y_true'], inputs['domain'], inputs['scores'])\nreturn {'official_score': m['official_score'], 'metrics': m}",
         pre=[{"port": "y_true", "check": "nonempty"}, {"port": "scores", "check": "finite"}], post=[{"port": "official_score", "check": "finite"}],
         tags=["audio", "anomaly detection", "metric", "auc", "evaluation", "dcase"]),
    dict(id="music_source_separation_softmask", tool="audiosep.separate",
         description="Trained classical stereo music source separation into vocals, drums, bass and other: gradient-boosted ratio masks on the STFT of the mixture "
                     "are fitted on training mixtures with their stems and applied to new mixtures (CPU, seconds per excerpt; needs at least 2 training excerpts).",
         inputs={"train_mixtures": p("array", "training mixtures, at least 2048 samples each", shape=("m", "n", 2), dtype="float"),
                 "train_stems": p("array", "training stems; axis 1 = (vocals, drums, bass, other)", shape=("m", 4, "n", 2), dtype="float"),
                 "mixtures": p("array", "mixtures to separate", shape=("k", "n_test", 2), dtype="float")},
         outputs={"estimates": p("array", "estimates; axis 1 = (vocals, drums, bass, other)", shape=("k", 4, "n_test", 2), dtype="float")},
         code="from scilib import audiosep\nout = audiosep.separate(inputs['train_mixtures'], inputs['train_stems'], inputs['mixtures'])\nreturn {'estimates': out}",
         pre=[{"port": "train_mixtures", "check": "finite"}, {"port": "train_stems", "check": "finite"}, {"port": "mixtures", "check": "finite"}],
         post=[{"port": "estimates", "check": "finite"}],
         tags=["audio", "music", "source separation", "soft mask", "lightgbm", "stft"]),
    dict(id="music_separation_sdr", tool="audiosep.sdr_scores",
         description="Source-to-distortion ratio of separated vocals, drums, bass and other against the reference stems in 1-second windows (median over windows "
                     "and excerpts, mean over the four targets); scale-sensitive, higher is better.",
         inputs={"references": _STEMS, "estimates": _EST, "sample_rate": _RATE},
         outputs={"sdr": p("number", "mean over the 4 targets of the median over excerpts of the median window SDR", unit="dB"),
                  "target_sdr": p("array", "median SDR per target in the order (vocals, drums, bass, other)", shape=(4,), unit="dB", dtype="float")},
         code="import numpy as np\nfrom scilib import audiosep\nsc = audiosep.sdr_scores(inputs['references'], inputs['estimates'], int(inputs['sample_rate']))\nreturn {'sdr': float(sc['sdr']), 'target_sdr': np.asarray(sc['target_median'], dtype=float)}",
         pre=[{"port": "references", "check": "finite"}, {"port": "estimates", "check": "finite"}], post=[{"port": "sdr", "check": "finite"}],
         tags=["audio", "music", "source separation", "metric", "sdr", "evaluation"]),
    dict(id="music_source_separation_htdemucs_ft", tool="audiosep_pretrained.separate_pretrained",
         description="Separate stereo music excerpts into vocals, drums, bass and other with the pretrained fine-tuned Hybrid Transformer Demucs bag "
                     "(four source-specialised models averaged), the strongest frozen Demucs route.",
         inputs={"mixtures": _MIX, "sample_rate": _RATE},
         outputs={"estimates": _EST},
         code="from scilib import audiosep_pretrained\nout = audiosep_pretrained.separate_pretrained(inputs['mixtures'], model='htdemucs_ft', sample_rate=int(inputs['sample_rate']))\nreturn {'estimates': out}",
         pre=[{"port": "mixtures", "check": "finite"}], post=[{"port": "estimates", "check": "finite"}],
         tags=["audio", "music", "source separation", "pretrained", "demucs"]),
    dict(id="music_source_separation_scnet", tool="scnet_pretrained.separate_pretrained",
         description="Separate stereo music excerpts into vocals, drums, bass and other with the pretrained MIMO-SCNet small model (SCNet backbone with two "
                     "mixture-consistent refinement iterations, 44.1 kHz internally).",
         inputs={"mixtures": _MIX, "sample_rate": _RATE},
         outputs={"estimates": _EST},
         code="from scilib import scnet_pretrained\nout = scnet_pretrained.separate_pretrained(inputs['mixtures'], model='mimo_scnet_small', sample_rate=int(inputs['sample_rate']))\nreturn {'estimates': out}",
         pre=[{"port": "mixtures", "check": "finite"}], post=[{"port": "estimates", "check": "finite"}],
         tags=["audio", "music", "source separation", "pretrained", "scnet"]),
]
