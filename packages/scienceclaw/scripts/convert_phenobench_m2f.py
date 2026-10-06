"""Convert a detectron2 Mask2Former (ResNet-50) PhenoBench checkpoint to Hugging Face ``Mask2FormerForUniversalSegmentation``.

Usage: ``python convert_phenobench_m2f.py <model.pth> <output dir>`` (the two PRBonn checkpoints are listed in ``weights.json``).

Hugging Face's converter handles Swin backbones only. This script downloads that converter (checked against a pinned hash),
drops its detectron2-only imports and adds the ResNet-50 backbone mapping (detectron2 stem, stride in the 3x3 convolution).
Every source key must map; the unused ones are printed.
"""
import hashlib
import sys
import types
import urllib.request

import torch

CONVERTER_URL = ("https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/mask2former/"
                 "convert_mask2former_original_pytorch_checkpoint_to_pytorch.py")
CONVERTER_SHA256 = "fecb57047226597456e8badd6013dab622cdd3996a70cd442ea7eefe2b6aced4"


def _converter_source() -> str:
    data = urllib.request.urlopen(CONVERTER_URL, timeout=60).read()
    if hashlib.sha256(data).hexdigest() != CONVERTER_SHA256:
        raise SystemExit("the Hugging Face converter changed upstream; review it and update CONVERTER_SHA256")
    return data.decode()


src = _converter_source()
# the HF conversion script needs detectron2 only for reading configs and for the demo; drop those imports
src = src.replace("from detectron2.checkpoint import DetectionCheckpointer\n", "").replace("from detectron2.config import get_cfg\n", "")
src = src.replace("from detectron2.projects.deeplab import add_deeplab_config\n", "").replace("from huggingface_hub.utils import httpx\n", "")
mod = types.ModuleType("hfconv"); exec(compile(src, "hfconv", "exec"), mod.__dict__)
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, ResNetConfig

def config():
    bb = ResNetConfig(num_channels=3, embedding_size=64, hidden_sizes=[256, 512, 1024, 2048], depths=[3, 4, 6, 3], layer_type="bottleneck",
                      hidden_act="relu", downsample_in_first_stage=False, downsample_in_bottleneck=False,
                      out_features=["stage1", "stage2", "stage3", "stage4"])
    return Mask2FormerConfig(
        ignore_value=255, num_labels=3, num_queries=100, no_object_weight=0.1, class_weight=2.0, mask_weight=5.0, dice_weight=5.0,
        train_num_points=12544, oversample_ratio=3.0, importance_sample_ratio=0.75, init_std=0.02, init_xavier_std=1.0,
        use_auxiliary_loss=True, feature_strides=[4, 8, 16, 32], backbone_config=bb, id2label={0: "soil", 1: "crop", 2: "weed"},
        label2id={"soil": 0, "crop": 1, "weed": 2}, feature_size=256, mask_feature_size=256, hidden_dim=256, encoder_layers=6,
        encoder_feedforward_dim=1024, decoder_layers=10, num_attention_heads=8, dropout=0.0, dim_feedforward=2048, pre_norm=False,
        enforce_input_proj=False, common_stride=4)

class Conv(mod.OriginalMask2FormerCheckpointToOursConverter):
    def replace_swin_backbone(self, dst, src, config):          # ResNet-50 (detectron2 'basic' stem, stride in the 3x3)
        d, s = "pixel_level_module.encoder", "backbone"
        pairs = []
        def conv_norm(sp, dp):
            pairs.append((f"{sp}.weight", f"{dp}.convolution.weight"))
            for k in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                pairs.append((f"{sp}.norm.{k}", f"{dp}.normalization.{k}"))
        conv_norm(f"{s}.stem.conv1", f"{d}.embedder.embedder")
        for si, depth in enumerate([3, 4, 6, 3]):
            for b in range(depth):
                sp, dp = f"{s}.res{si + 2}.{b}", f"{d}.encoder.stages.{si}.layers.{b}"
                for j in range(3):
                    conv_norm(f"{sp}.conv{j + 1}", f"{dp}.layer.{j}")
                if b == 0:
                    conv_norm(f"{sp}.shortcut", f"{dp}.shortcut")
        self.pop_all(pairs, dst, src)

def convert(ckpt, out):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    sd = dict(sd)
    cfg = config()
    hf = Mask2FormerForUniversalSegmentation(cfg)
    base = hf.model
    c = Conv(types.SimpleNamespace(state_dict=lambda: sd), cfg)
    c.convert(base)
    c.convert_universal_segmentation(hf) if False else None
    # class predictor
    hf.class_predictor.weight.data.copy_(sd.pop("sem_seg_head.predictor.class_embed.weight"))
    hf.class_predictor.bias.data.copy_(sd.pop("sem_seg_head.predictor.class_embed.bias"))
    left = [k for k in sd if not k.startswith(("criterion.",)) and not k.endswith("num_batches_tracked")]
    print("unused source keys:", left)
    hf.eval().save_pretrained(out, safe_serialization=True)
    return hf

if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
