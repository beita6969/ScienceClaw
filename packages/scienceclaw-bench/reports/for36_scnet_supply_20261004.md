# FoR36 frozen SCNet supply and visible-dev comparison — 2026-10-04

## Decision

The public MIMO-SCNet small four-source checkpoint is a viable stronger frozen
route for FoR36. It is now exposed as the optional `separate_scnet` ToolSpec;
the call receives mixtures only and returns the task order
`vocals, drums, bass, other`. The existing formal `separate_htdemucs` route
and all consumed formal items are unchanged. No formal item was rerun.

## Visible-dev comparison

The comparison used one deterministic `FoR36-src-00000005-00` episode and its
four visible dev mixtures. Hidden evaluation stems were accessed only by the
existing `score_dev` callback. Both models used the same four inputs, GPU
inference, and the existing FoR36 BSSEval-v4 implementation.

| frozen route | visible-dev mean SDR (dB) | vocals | drums | bass | other |
| --- | ---: | ---: | ---: | ---: | ---: |
| `htdemucs_ft` | 8.091974 | 2.638195 | 12.554064 | 7.737064 | 9.438573 |
| `mimo_scnet_small` | **8.528417** | -1.031511 | 13.897575 | 10.643995 | 10.603609 |

The SCNet candidate is **+0.436443 dB** on this visible-dev batch. The
engineering comparison is not a formal ID/OOD result and does not rewrite the
existing `8.278244 dB` formal batch. FoR36 has no independent OOD pool.

## Frozen artifact and implementation

- Upstream implementation: [SonyResearch/mimo-audio-separation](https://github.com/SonyResearch/mimo-audio-separation), commit `9169bbbf9e6fdca5b22d5735b4244fbad7ab40b5` (MIT).
- Public checkpoint archive: `mimo-scnet_small.zip`, SHA-256 `f0eaaab7e899ffc548b29d51d76b89727253fb0a529b787b258eabfcf47db6fb`.
- Staged remote files: `$L/models/scnet_mimo_small/backbone_model.pth` (42,491,051 bytes, SHA-256 `1d19305440e8439faeb4163d809b2dcaaffacf67f6f926425be581290d6536ee`) and `config.yaml` (SHA-256 `e751edb8a92a777497dc5458beb94c9a676babd037b5e8bfb3f1391d6e0091e1`).
- Source tree: `$F/p3/scnet/mimo-audio-separation`; model dependencies are installed only in the GPU `sc-harness` environment (`hydra-core`, `beartype`, `rotary-embedding-torch`, `hyper-connections`, `librosa`).
- Repository wrapper: `scilib/scnet_pretrained.py`; worker allow-list and FoR36 ToolSpec are updated. The wrapper resamples 22.05 kHz inputs to 44.1 kHz, runs iteration 2, reorders public `vocals,bass,drums,other` to the task contract, then resamples back.

## Runtime smoke

Leonardo job `59277346` (existing one-A100 Qwen allocation, reused for an
engineering check) loaded the checkpoint with no missing or unexpected keys.
The wrapper returned `(4, 4, 132300, 2)` `float32`, all finite, in 3.94 s;
the independent direct-model check returned the same `8.528417 dB` score.
Focused FoR36 and pretrained tests passed (`56 passed, 2 skipped`), and the
three changed remote Python files compiled successfully with the remote
`sc-harness` interpreter. This smoke did not submit a formal episode.

## Boundary

SCNet, like the existing Demucs bundles, was trained on MUSDB18-HQ training
material; this overlap is disclosed and the route is not presented as a clean
OOD claim. The optional route is selected on visible dev only. If a future
fresh formal capacity is supplied, the run must record `separate_scnet` in its
trace and retain the unchanged scorer, split allocation, and acceptance rule.
