# FoR51 SevenNet-MF-0 supply audit (2026-10-04)

Leonardo already contains the public SevenNet-MF-0 checkpoint at
`$F/sv_pkgs/sevenn/pretrained_potentials/SevenNet_MF_0/checkpoint_sevennet_mf_0.pth`
(`10,322,158` bytes, SHA-256
`81791329b37d445f46b531578c182c41792d98c7814222c9e5dde276402225fd`). With
`PYTHONPATH=$F/sv_pkgs`, the installed SevenNet package loaded
`SevenNetCalculator("7net-mf-0", modal="PBE")` and `modal="R2SCAN"` on the
server CPU.

This is a frozen universal-potential checkpoint trained on Materials Project
PBE(+U)/r2SCAN crystal relaxation trajectories. It is a structure/force prior,
not a phonon-label model; the Materials Project provenance also overlaps the
usual crystal source universe. It therefore cannot be presented as an
independent phonon SOTA result. The current FoR51 route already has a complete
SevenNet-l3i5 cache and the visible-dev comparison in
`for51_visible_dev_model_selection_20261004.md` favors its ExtraTrees route.

The MF-0 cache was then completed for all 1,265 structures for both
`sevennet_mf0_pbe` and `sevennet_mf0_r2scan` (676 train plus the remaining
589 rows). A full visible-dev comparison over the fixed 110-row `dev` pool,
using the same ExtraTrees log-target regressor, gave:

| route | visible-dev MAE (cm^-1) |
|---|---:|
| existing SevenNet-l3i5 ExtraTrees | 24.2811 |
| SevenNet-MF-0 PBE | 26.2827 |
| SevenNet-MF-0 R2SCAN | 37.3410 |

The two engineering probes were `P51MF0C` (PBE) and `P51MF0R2B` (R2SCAN);
artifact provenance records the corresponding frozen model names and 676
visible training targets. The first probe timeout was traced to an incomplete
cache and an omitted `SCIENCECLAW_MLIP_CACHE` sandbox pass-through, fixed in
`b56dbe4`; the explicit modal override remains engineering-only and leaves the
formal default on PBE. Because neither MF-0 variant beats the existing route,
neither is promoted into the formal tool surface.

A one-card smoke allocation request (`59294443`) remained pending under
`MaxNodePerAccount` and was cancelled without running a forward pass. MACE and
MatterSim were not installed with a local frozen checkpoint, so there is no
safe additional route to register in this round. No hidden ID/OOD target was
read, no formal item was repeated, and no acceptance rule changed.
