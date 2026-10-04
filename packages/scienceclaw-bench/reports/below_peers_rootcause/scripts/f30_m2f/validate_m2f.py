"""Compare the converted HF Mask2Former (plants / leaves) with the official PRBonn val predictions on a few images."""
import glob, sys, time, zipfile, io
import numpy as np, torch
from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

M = "/leonardo_scratch/large/userexternal/rqian000/models/phenobench_m2f"
D = "/leonardo_scratch/large/userexternal/rqian000/scienceclaw-data/datasets/for30-phenobench/reconstructed_v1/data/PhenoBench/val/images"
names = sorted(p.split("/")[-1] for p in glob.glob(D + "/05-15_*.png") if not p.split("/")[-1].startswith("._"))[:int(sys.argv[1]) if len(sys.argv) > 1 else 3]
torch.set_num_threads(8)
zp = zipfile.ZipFile(M + "/val_pred_panoptic_segmentation.zip"); zl = zipfile.ZipFile(M + "/val_pred_leaf_instance_segmentation.zip")
print("zip dirs:", sorted({n.split("/")[0] for n in zp.namelist()}), sorted({n.split("/")[0] for n in zl.namelist()}))
proc = Mask2FormerImageProcessor(do_resize=True, size={"shortest_edge": 1024, "longest_edge": 1024}, size_divisor=32, do_rescale=False, do_normalize=True,
                                 image_mean=[103.53, 116.28, 123.675], image_std=[1.0, 1.0, 1.0], ignore_index=255, reduce_labels=False, num_labels=3)
models = {k: Mask2FormerForUniversalSegmentation.from_pretrained(f"{M}/hf_{k}").eval() for k in ("plants", "leaves")}


def run(model, img):
    x = proc(images=img, return_tensors="pt")
    with torch.inference_mode():
        out = model(**x)
    r = proc.post_process_panoptic_segmentation(out, threshold=0.8, mask_threshold=0.5, overlap_mask_area_threshold=0.8,
                                                label_ids_to_fuse={0}, target_sizes=[img.shape[:2]])[0]
    return r["segmentation"].numpy(), r["segments_info"]


def official(zf, folder, name):
    return np.array(Image.open(io.BytesIO(zf.read(f"{folder}/{name}"))))


for n in names:
    img = np.array(Image.open(f"{D}/{n}").convert("RGB"))
    t = time.time()
    seg, info = run(models["plants"], img)
    sem = np.zeros(seg.shape, np.uint8); inst = np.zeros(seg.shape, np.int32)
    for s in info:
        m = seg == s["id"]
        if s["label_id"] in (1, 2):
            sem[m] = s["label_id"]; inst[m] = s["id"]
    o_sem = official(zp, "semantics", n) if "semantics/" + n in zp.namelist() else None
    o_inst = official(zp, "plant_instances", n)
    line = f"{n}: {time.time() - t:.1f}s segs={len(info)}"
    if o_sem is not None:
        line += f" | sem agreement {(sem == o_sem).mean():.4f} (crop px ours {int((sem == 1).sum())} off {int((o_sem == 1).sum())}; weed ours {int((sem == 2).sum())} off {int((o_sem == 2).sum())})"
    line += f" | plant instances ours {len(np.unique(inst)) - 1} official {len(np.unique(o_inst)) - 1}"
    seg_l, info_l = run(models["leaves"], img)
    leaf = np.zeros(seg_l.shape, np.int32)
    for s in info_l:
        if s["label_id"] == 1:
            leaf[seg_l == s["id"]] = s["id"]
    o_leaf = official(zl, "leaf_instances", n)
    line += f" | leaves ours {len(np.unique(leaf)) - 1} official {len(np.unique(o_leaf)) - 1}"
    # instance-level agreement: best IoU of each official plant instance
    ious = []
    for k in np.unique(o_inst)[1:]:
        a = o_inst == k
        b_ids, cnt = np.unique(inst[a], return_counts=True)
        best = 0.0
        for b, c in zip(b_ids, cnt):
            if b == 0: continue
            best = max(best, c / (a.sum() + (inst == b).sum() - c))
        ious.append(best)
    line += f" | mean best-IoU of official plant instances {np.mean(ious) if ious else float('nan'):.3f}"
    print(line, flush=True)
