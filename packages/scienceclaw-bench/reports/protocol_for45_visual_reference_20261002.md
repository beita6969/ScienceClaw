# FoR45 visual reference audit (2026-10-02)

## Finding

FoR45's scorer and acceptance rule remain unchanged. The existing image_blind field correctly identifies the
high-scoring per-language consensus route, but it did not quantify whether visible image-caption pairs provide a
stronger, image-conditioned reference. A CLIP checkpoint is not part of the repository's declared dependencies or
the current server environment, so it was not downloaded or silently substituted.

The adapter now adds a deterministic image_knn_reference tool. It computes the already exposed CPU image
descriptors (color_hist, gray_thumb, hsv_stats, or all), searches only same-language rows from
load_train() with has_image=true, and returns the nearest visible caption. Caption-only rows are skipped; if a
language has no visible image row, the helper falls back to that language's visible-caption medoid. It consumes no
evaluation IDs, captions, or hidden labels. k>1 uses the medoid of the selected visible captions. The evaluator
records its score in details diagnostics and pooled_payload chrf_knn_ref; pooled_metric, the official
medoid reference, and acceptance are untouched.

## Full local split diagnostic

Using the rebuilt public dev split (four ID and four proxy-OOD episodes, 8 items each), the trusted scorer gives:

| split | items | primary (gold sanity check) | medoid reference | image-kNN reference | kNN − medoid |
|---|---:|---:|---:|---:|---:|
| id | 32 | 100.000 | 18.7939 | 13.1488 | −5.6451 |
| ood | 32 | 100.000 | 22.0052 | 16.3611 | −5.6442 |

The primary column is only an evaluator sanity check using the hidden labels inside the trusted scorer; it is not an
agent result. The kNN baseline is consistently below the existing medoid by about 5.6 chrF++ points, so this
lightweight descriptor route is not a SOTA improvement. It is useful as an image-grounding reference and as a
regression diagnostic: the previous image-blind consensus path remains distinguishable from a visible-image route.

## Validation and next step

tests/test_task_for45.py covers deterministic same-language retrieval, caption-only fallback, invalid k, the new
tool output shape, and the diagnostic fields. The current dependency set does not provide CLIP/OpenCLIP weights.
A future CLIP-kNN experiment may be added only after obtaining an approved, hashed checkpoint on the server and
measuring it as the same trusted-side diagnostic; it must not replace the official reference or retroactively alter
the frozen formal episodes.

## Frozen open CLIP component (implementation, not a formal run)

`scilib/clip_retrieval.py` now supplies an optional frozen OpenAI CLIP ViT-B/32 image encoder through
`open_clip_torch`. It is deliberately separate from `image_knn_reference`: the new `clip_retrieval_status` tool
returns the exact checkpoint path, SHA-256, byte count, package versions, licence note, and availability; the
`clip_knn_reference` tool uses that checkpoint only for cosine nearest-neighbour retrieval over visible
same-language image-caption rows. Both tools are trusted-side diagnostics. The evaluator's primary, medoid
reference, margin, and hidden labels are untouched.

The component is fail-closed. It accepts only an operator-staged file under the model root or the explicit
`SCIENCECLAW_CLIP_WEIGHTS` path, passes the concrete file path to `open_clip`, and never passes a registry name or
opens a hub connection. If torch, `open_clip_torch`, or the checkpoint is absent, `clip_knn_reference` raises with
the status provenance instead of silently reverting to CPU descriptors. The status records
`network_allowed=false`, `labels_used=false`, and `frozen=true` for auditability.

### Server deployment recipe

Run these commands in the approved Leonardo environment after obtaining a checkpoint through the project's normal
licence and checksum review. They do not download weights themselves:

```bash
export F=/leonardo_scratch/fast/AIFAC_F02_774/rqian000
export M=/leonardo_scratch/large/userexternal/rqian000/models
export SCIENCECLAW_MODELS="$M"
mkdir -p "$M/clip/open_clip_vit_b32"
# Copy the approved local checkpoint into the directory, then verify it:
find "$M/clip/open_clip_vit_b32" -maxdepth 1 -type f -print0 | xargs -0 sha256sum
"$F/envs/sc-harness/bin/python" -m pip install --no-deps 'open_clip_torch==<approved-version>'
PYTHONPATH="$F/scienceclaw" SCIENCECLAW_MODELS="$M" "$F/envs/sc-harness/bin/python" - <<'PY'
from scilib import clip_retrieval
print(clip_retrieval.provenance())
assert clip_retrieval.provenance()["available"]
PY
```

The placeholder version and checkpoint must be replaced by an approved, recorded pair; `pip` is not run by the
agent and no checkpoint is fetched from the network. Until that pair is present, the status tool is expected to
report unavailable and no CLIP result may be presented as an FoR45 score. Focused tests cover missing-weight
failure, digest/provenance reporting, same-language visible-row isolation, and explicit fallback behavior; no
formal FoR45 split was launched for this change.

The Leonardo policy sandbox currently runs `sc-run` without torch and sends pretrained calls through the broker.
This FoR45-only change does not alter the shared worker allowlist; an operator who wants the formal agent to call
the component must separately add a reviewed `clip_retrieval` broker route after installing the same pinned package
and checkpoint in `sc-harness`, then verify the returned provenance from that worker. Until that shared route is
reviewed and hashed, direct `sc-harness` smoke is the only supported check and a formal agent run remains out of
scope.

### Leonardo code-only check (2026-10-03)

The module and FoR45 task were copied to the fast checkout and compiled with the torch-free `sc-run` interpreter;
the local and remote SHA-256 values match (`clip_retrieval.py`:
`1723937ea7d59fdf52d7e9455682c3c3ebb4a52ccd21cc873803cb2311e993ff`; FoR45 task:
`30958784e9588d5a8896862dbc4b8ec5be54f1f0aa8f7986b85e5b4efa5a90ae`). A read-only environment probe found
`torch=False`, `open_clip=False` and `transformers=False` in `sc-run`; `sc-harness` has torch and transformers but
no open_clip, and the model root has no CLIP checkpoint. This is deployment evidence only: no score, formal
episode, or hidden-label access was produced.
