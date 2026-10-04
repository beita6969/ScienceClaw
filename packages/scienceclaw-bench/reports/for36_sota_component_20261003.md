# FoR36 frozen separator component probe — 2026-10-03

## Decision

The primary frozen route is now `htdemucs_ft`, the official four-model fine-tuned
Hybrid Transformer Demucs ensemble. It is exposed through the existing
`separate_htdemucs_ft` tool and through `scilib.audiosep_pretrained.separate_pretrained`
with `model="htdemucs_ft"`. The route receives mixtures only; it never receives
stems, labels, or the visible training pool.

`htdemucs` remains the fallback, and `mdx_extra` remains an optional comparison
route. The MDX-quantized bag was removed from the public route because the
official `diffq` extension cannot be built in the current Leonardo environment
(the image has no Python development headers); no incomplete quantized path is
advertised.

## Direct engineering probe

This is a source-pool engineering probe, not a new formal ID/OOD result. The
same four one-excerpt episodes were decoded once and scored with the existing
FoR36 BSSEval-v4 implementation. All three models saw exactly the same mixture
arrays and were run with `shifts=0`, `overlap=0.25`, on one A100.

| model | item SDR values (dB) | mean (dB) | delta vs `htdemucs` |
| --- | --- | ---: | ---: |
| `htdemucs` | -2.0187, 11.1233, 3.2113, 7.8614 | 5.0443 | — |
| `htdemucs_ft` | 1.1734, 11.5840, 4.9243, 8.6766 | **6.5896** | **+1.5453** |
| `mdx_extra` | -2.2934, 10.9221, 2.3269, 9.7018 | 5.1644 | +0.1200 |

The first excerpt improved by 3.1921 dB (`-2.0187` → `1.1734`), and all four
items improved with `htdemucs_ft`. This supports using the fine-tuned ensemble
as the default candidate for the next formal mutually exclusive capacity; it
does not retroactively change any consumed formal pool item.

## Staged artifacts

The Leonardo model root is `/leonardo_scratch/large/userexternal/rqian000/models/demucs`.
The `htdemucs_ft` YAML manifest and four official checkpoints are staged there.
The MDX-extra YAML manifest and four official checkpoints are also staged. The
downloaded files were checked with SHA-256; the complete manifest is recorded at
`/leonardo_scratch/large/userexternal/rqian000/sc-tools/logs/demucs_mdx_sha256.txt`.

The staging script is
`scripts/leonardo/migration/leo_models.sh`; it uses the official Demucs release
endpoints and stores weights outside the repository. The code path is frozen:
there is no fit step and no network access during tool execution.

## Formal-use boundary

FoR36 full_v1 has no independent OOD collection. The existing formal ID result
and its acceptance rules remain unchanged. A future formal run must use a fresh
mutually exclusive allocation and record the required `separate_htdemucs` call
with `config.model=htdemucs_ft` in the trace. The explicit
`separate_htdemucs_ft` alias remains engineering-only so the formal validator
sees its required tool reference. Until that capacity exists, this report is
engineering evidence for the route selection, not a claim of a new formal score.

## Dependency note

`mdx_extra_q` was briefly downloaded while checking the official Demucs model
catalog, but it was deleted from Leonardo and removed from the public route.
Its loader requires `diffq`; `pip install diffq` failed because the Leonardo
Python environment lacks `Python.h`. We do not spend a queue allocation on this
non-improving optional path.

## Provenance

Official model code and manifests: [facebookresearch/demucs](https://github.com/facebookresearch/demucs).
The checkpoint URLs are the release endpoints listed by that repository under
`demucs/remote/files.txt`.

## Default-route update (2026-10-04)

The required formal `separate_htdemucs` tool now defaults to `model=htdemucs_ft` when the agent omits a model, while retaining explicit `htdemucs` and `mdx_extra` fallbacks. This makes the strongest staged frozen bag the one-step default and preserves the required formal tool name. The change is source-only, does not rewrite consumed scores, and the focused FoR36/audio/remote tests are green.
