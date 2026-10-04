# FoR51 SevenNet-MF-0 route deployment (2026-10-04)

The public frozen SevenNet-MF-0 checkpoint already staged on Leonardo is now exposed as an explicit FoR51 engineering tool. This does not alter the formal SevenNet-l3i5 route or repeat any ID/OOD item.

## Frozen source

- checkpoint: `$F/sv_pkgs/sevenn/pretrained_potentials/SevenNet_MF_0/checkpoint_sevennet_mf_0.pth`
- size: `10,322,158` bytes
- SHA-256: `81791329b37d445f46b531578c182c41792d98c7814222c9e5dde276402225fd`
- modalities: `PBE` and `R2SCAN`, selected explicitly by `modal`
- provenance: official SevenNet-MF-0 multi-fidelity universal potential; trained on Materials Project PBE(+U)/r2SCAN structural trajectories, not on FoR51 phonon labels

## Code path

`scilib.matphonon_mlip` now accepts the isolated model names `sevennet_mf0_pbe` and `sevennet_mf0_r2scan`. Each gets a separate content-addressed cache namespace and constructs `SevenNetCalculator("7net-mf-0", modal=...)`; the existing `sevennet` and `chgnet` keys remain unchanged.

FoR51 adds `fit_sevennet_mf0_mlip`, which selects `modal=pbe` or `modal=r2scan`, extracts frozen phonon features for visible training/evaluation structures, and fits only the existing visible-target log ensemble. The route never reads evaluation targets or updates universal-potential weights. The cache prewarm helper accepts both MF-0 names so a GPU allocation can resume disjoint ranges safely.

## Verification

- local FoR51/scilib focused tests: `23 passed`
- remote FoR51/scilib focused tests: `23 passed`
- remote Python compilation: passed
- server loader: `SevenNetCalculator("7net-mf-0", modal="PBE")` and `modal="R2SCAN"` both loaded successfully

No MF-0 visible-dev score is claimed yet; the route is ready for a non-formal engineering comparison on a free GPU. Existing FoR51 formal evidence remains SevenNet-l3i5 (`ID 2/2`, `OOD 4/4`) and is unchanged.

## Follow-up deployment check (2026-10-04)

The prewarm CLI had one operational edge case: `_cache_dir()` intentionally
returns `None` when `SCIENCECLAW_MLIP_CACHE` does not yet exist, so a fresh
model-specific path could finish without writing rows. Commit `e498f57` plus
`08bdc69` makes the explicit prewarm command create that directory before
calling `_local`; the normal library behavior (no cache directory means no
writes) is unchanged.

On Leonardo allocation `59296763` (`lrdn3272`), both MF-0 modals loaded and
returned finite `(4, 21)` feature arrays on four visible structures: PBE in
14.0 s and R2SCAN in 4.0 s. After creating the fresh cache path, R2SCAN
prewarm rows `[0,16)` completed `16/16` valid in 17.7 s, and 16 immutable JSON
rows were observed in `$L/sc-tools/mlip-mf0`. This is an engineering/deployment
smoke only; no hidden target, scorer, ID/OOD item, or formal score was read or
changed.
