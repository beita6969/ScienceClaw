# FoR47 formal capacity and duplicate protection audit (2026-10-04)

## Scope

This audit covers only the historical `0|FoR47|8|H6|id,ood|4` request. It does not change the scorer, acceptance rule, manifest, or any existing result. The one-item/one-evaluation rule is applied before launching a new batch.

## Leonardo evidence

The current remote checkout is `/leonardo_scratch/fast/AIFAC_F02_774/rqian000/scienceclaw`, with run root `/leonardo_scratch/large/userexternal/rqian000/sc-runs`.

- SSH is usable and the four existing Qwen service jobs are RUNNING: `59277346`, `59277347`, `59264793`, and `59264794`.
- `check_avail.py FoR47 --strict --splits id,ood --n 4` returns `strict preflight OK`; this is the frozen plan capacity check and does not mean previously consumed episode IDs are reusable.
- `toolon_H6_id` and `toolon_H6_ood` each contain four complete result/trajectory pairs, with episode suffixes `00`–`03`. All eight results have `z=1`, `completed=true`, `hard_ok=true`, and `reproducible=true`, but none of their trajectories contains `pretrained_parse`. These are historical non-tool or direct-code evidence and are already consumed item IDs.
- `toolon_SOTA47B_id` and `toolon_SOTA47B_ood` each contain four complete result/trajectory pairs, with disjoint suffixes `04`–`07`. Every trajectory contains a successful `pretrained_parse` call connected to the final submission.
- The remote split-scoped validator returns `formal post-run evidence OK` for both `SOTA47B/id` and `SOTA47B/ood`.

## Scores already present

| batch | split | episodes | mean LAS | item suffixes |
|---|---:|---:|---:|---|
| H6 | id | 4/4 | 0.753877 | 00–03 |
| H6 | ood | 4/4 | 0.735833 | 00–03 |
| SOTA47B | id | 4/4 | 0.854528 | 04–07 |
| SOTA47B | ood | 4/4 | 0.835036 | 04–07 |

`SOTA47B` is the current formal tool-on evidence. The old H6 item IDs cannot be rerun merely to add a missing tool node; doing so would violate the fixed one-evaluation rule. No `leo_chain.sh` job was submitted and no GPU allocation was consumed by this audit.

## Decision

Do not launch `0|FoR47|8|H6|id,ood|4`. The requested formal objective is already satisfied on a disjoint item set by `SOTA47B`, and the H6 item set is consumed. Any future FoR47 formal run must use a new manifest tag and a new `skip` range after an explicit item-disjointness check.
