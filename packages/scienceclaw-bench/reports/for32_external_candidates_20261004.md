# FoR32 external candidate audit (2026-10-04)

## Decision

FoR32 remains blocked for a formal tool-ON rerun. No candidate found in this pass satisfies all three required conditions at the same time: a frozen public checkpoint, no overlap with the 260 labelled MSD Task04 volumes, and the exact 3-D output contract (one mask per input volume, labels `0=background, 1=anterior, 2=posterior`). No scorer, hidden target, or formal item was touched.

## BiomedParse

The official Microsoft checkpoint is the closest semantic match found. Its model card lists `anterior hippocampus` and `posterior hippocampus` among recommended MRI prompts. However, the same deployment specification requires 2-D 8-bit grayscale/RGB images (default resolution 1024x1024) and returns a 2-D probability mask. It is therefore not an exact 3-D volume route; a slice-wise adapter would be a new unvalidated method rather than a frozen contract match.

The Hugging Face repository is gated behind contact-information terms, and the Leonardo account has no local Hugging Face token or cached checkpoint. We did not bypass the gate or download an unaudited copy. Source: <https://huggingface.co/microsoft/BiomedParse>.

## HSF and Hippodeep

HSF provides frozen models trained on heterogeneous public/private hippocampal data, but its documented output is whole hippocampus or subfields and explicitly does not assign a head/tail (anterior/posterior) class. Its model hub reports no third-party model. Source: <https://hsf.readthedocs.io/en/latest/model-hub/> and <https://hsf.readthedocs.io/en/latest/>.

Hippodeep is a public frozen whole-hippocampus model (left/right binary masks), not an anterior/posterior subfield model. Its published implementation writes `_mask_L` and `_mask_R` outputs. Source: <https://github.com/schellm/hippodeep>.

## 3D-UCaps and Task04-specific repositories

3D-UCaps publishes hippocampus checkpoints, but its README states that the models are trained and cross-validated on the Medical Segmentation Decathlon hippocampus dataset. It is therefore not an independent no-overlap checkpoint for this benchmark. The inspected Task04-specific nnU-Net repositories likewise train on the same 260 labelled cases; they are excluded even when their label names match.

## Next valid unblock

Only one of these events should reopen integration work: (a) a publicly downloadable external 3-D checkpoint with an auditable non-Task04 training set and anterior/posterior outputs, or (b) an authorized BiomedParse checkpoint plus a documented 3-D slice/volume adapter whose visible-data score is measured before any formal submission. Until then the existing visible-train-only U-Net evidence remains engineering-only, and the `SOTA32=blocked` manifest state is unchanged.
