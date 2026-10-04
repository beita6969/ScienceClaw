# FoR51 external pretrained supply review (2026-10-04)

## Scope and server inventory

This was a read-only supply and comparability check for FoR51. It did not read
hidden ID/OOD targets, run a scorer, repeat an item, or modify the formal
SevenNet/CHGNet routes.

Leonardo was reachable as `rqian000@login01.leonardo.local`. The only
additional public structural potential found already staged in the account is
SevenNet-MF-0:

- checkpoint: `$F/sv_pkgs/sevenn/pretrained_potentials/SevenNet_MF_0/checkpoint_sevennet_mf_0.pth`
- size: `10,322,158` bytes
- SHA-256: `81791329b37d445f46b531578c182c41792d98c7814222c9e5dde276402225fd`
- package source: the staged SevenNet package (`sevenn`); checkpoint config
  version `0.9.4`, 89 chemical species, two modalities
- read-only loader check: `SevenNetCalculator("7net-mf-0", device="cpu", modal="PBE")`
  and `modal="R2SCAN"` both load successfully. The checkpoint exposes
  `modal_map={"R2SCAN": 0, "PBE": 1}` and default modal `PBE`.

The SHA was computed directly from the staged file. The checkpoint configuration
contains no phonon-target head. It is an interatomic potential that predicts
energies/forces/stresses and could, in principle, be used as a frozen force
source for the existing phonopy feature extractor.

The official upstream description identifies SevenNet-MF-0 as trained on
Materials Project PBE(+U) and r²SCAN crystal-relaxation trajectories:
<https://github.com/MDIL-SNU/SevenNet/blob/main/sevenn/pretrained_potentials/SevenNet_MF_0/README.md>.
This is public and auditable, but it is not an independent phonon-label SOTA
model. Materials Project is also the structural lineage of much of Matbench
phonons, so the same structural prior disclosure used for SevenNet-l3i5 and
CHGNet applies. The checkpoint did not train on the Matbench phonon target as
far as the public training description shows; this is a structural-prior
overlap, not evidence of target-label leakage.

The `sc-harness` environment currently has `ase`, `chgnet`, and `phonopy`, but
no installed `mace`, `matgl`, `mattersim`, `fairchem`, `orb_models`, or `jmp`
module. A read-only filename inventory under the account's model/cache roots
found no MACE, MatterSim, JMP, MegNet, MODNet, or independent phonon checkpoint.
Therefore those alternatives cannot be safely wrapped on this server without a
new download and a separate license/training-lineage review.

## Deployment decision

SevenNet-MF-0 is recorded as a **public frozen structural-potential candidate,
not an FoR51 score**. The current `scilib.matphonon_mlip` route only supports
`sevennet` (SevenNet-l3i5) and `chgnet`; it has no MF modal argument or cache
namespace. Adding it would require an explicit model/modal provenance field,
a cache key that includes the model and modal, and a visible-dev-only
comparison before any route promotion. It must not silently replace the
current formal route.

A one-GPU smoke allocation was submitted as `59294443` only to check the
phonopy adapter with the MF potential, but it remained `PENDING` with
`MaxNodePerAccount` and was canceled before starting. No feature row, score, or
formal item was produced. No further allocation is warranted for this supply
review.

Conclusion: SevenNet-MF-0 is a safe **future engineering option** with clear
public provenance, but there is currently no independent, already-integrated
phonon SOTA weight that can be used to raise or rewrite FoR51. Keep the
SevenNet-l3i5 formal evidence, the explicit CHGNet diagnostic, their MP prior
disclosure, and the existing no-label-leakage boundary unchanged.
