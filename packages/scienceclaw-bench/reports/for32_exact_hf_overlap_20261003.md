# FoR32 exact-contract public checkpoint boundary — 2026-10-03

## Candidate found

The public Hugging Face repository
[`marcocastellaro/theme4-nnunet-hippocampus`](https://huggingface.co/marcocastellaro/theme4-nnunet-hippocampus)
contains a frozen nnU-Net 3D checkpoint that matches the FoR32 interface:
one-channel 3D MRI crops, `patch_size=[40,56,40]`, and labels
`background=0`, `Anterior=1`, `Posterior=2`.  Its `dataset.json` declares the
same Medical Segmentation Decathlon Task04 Hippocampus contract (`260`
training cases and `130` test cases).  The archive contains a complete
`checkpoint_final.pth` (44,910,303 bytes) and fold-0 split metadata (`208`
training + `52` validation cases).

The downloaded archive was checked locally:

```text
theme4_hippocampus_nnunet_backup.zip
sha256 8046c6e37e5555c6b236a719520706aae27e92e87191e3b0a50e2d988a3b9235
checkpoint_final.pth
sha256 84109e63dd4c7d8218e654735287fd1686215e1d7337e572cb36a43341474763
HF revision 809eae778a581cda540cc333cc9b0a2449a9ff93
```

## Fair-use check

The checkpoint is **not eligible for the FoR32 formal route**.  Its split
contains exactly the same `hippocampus_001` … `hippocampus_394` case universe
as the local Task04 `imagesTr` pool: all `208` fold-training cases and all
`52` fold-validation cases overlap (`208/208` and `52/52`), with no missing or
extra IDs.  This is therefore a same-dataset checkpoint, not an independent
pretrained model.  Running it on FoR32 ID/OOD cases would leak the benchmark
distribution and would violate the no-training/no-visible-target rule.

No score was produced and no formal manifest or ToolSpec was changed.  The
archive remains an audit-only discovery.  It confirms that the low score is
not caused by an absent public implementation, but the exact-contract public
weights currently found cannot be used under the project’s provenance rules.

## Next valid route

Keep `SOTA32=blocked` until a checkpoint trained on an independent hippocampus
corpus (with anterior/posterior outputs) is found, or obtain an explicitly
held-out external evaluation allocation.  Binary whole-hippocampus ADNI/HarP
models and the existing InnerEye model do not satisfy the class contract and
must not be relabeled or split heuristically.
