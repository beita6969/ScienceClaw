import time, numpy as np, glob
from PIL import Image
from scilib import phenoseg_m2f as m
D = "/leonardo_scratch/large/userexternal/rqian000/scienceclaw-data/datasets/for30-phenobench/reconstructed_v1/data/PhenoBench/val/images"
fs = sorted(p for p in glob.glob(D + "/05-15_*.png") if "/._" not in p)[:32]
imgs = np.stack([np.array(Image.open(f).convert("RGB").resize((512, 512), Image.BILINEAR)) for f in fs])
t = time.time(); m.predict_panoptic(imgs[:4]); t0 = time.time() - t
t = time.time(); m.predict_panoptic(imgs, batch_size=4); t1 = time.time() - t
print(f"first call (loads checkpoints, 4 images) {t0:.1f}s; 32 images of 512x512, batch 4: {t1:.1f}s = {t1 / 32:.2f}s/image")
import torch; print(torch.cuda.get_device_name(0), "max mem GB", torch.cuda.max_memory_allocated() / 2**30)
