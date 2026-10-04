"""Convert the PRBonn PhenoBench Mask2Former (ResNet-50, detectron2) checkpoints to HF Mask2FormerForUniversalSegmentation.
usage: convert_phenobench_m2f.py <model.pth> <out_dir> [plants|leaves]"""
import re, sys
from pathlib import Path
import torch
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor, ResNetConfig
sys.path.insert(0, str(Path(__file__).parent))
import m2f_conv_lib as C

ckpt, out, kind = sys.argv[1], Path(sys.argv[2]), (sys.argv[3] if len(sys.argv) > 3 else "plants")
sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
bb = ResNetConfig(embedding_size=64, hidden_sizes=[256, 512, 1024, 2048], depths=[3, 4, 6, 3], layer_type="bottleneck", hidden_act="relu",
                  downsample_in_first_stage=False, downsample_in_bottleneck=False, out_features=["stage1", "stage2", "stage3", "stage4"])
id2label = {0: "soil", 1: "crop", 2: "weed"}
cfg = Mask2FormerConfig(ignore_value=255, num_labels=3, num_queries=100, no_object_weight=0.1, class_weight=2.0, mask_weight=5.0, dice_weight=5.0,
                        train_num_points=12544, oversample_ratio=3.0, importance_sample_ratio=0.75, init_std=0.02, init_xavier_std=1.0,
                        use_auxiliary_loss=True, feature_strides=[4, 8, 16, 32], backbone_config=bb, id2label=id2label,
                        label2id={v: k for k, v in id2label.items()}, feature_size=256, mask_feature_size=256, hidden_dim=256, encoder_layers=6,
                        encoder_feedforward_dim=1024, decoder_layers=10, num_attention_heads=8, dropout=0.0, dim_feedforward=2048,
                        pre_norm=False, enforce_input_proj=False, common_stride=4)
model = Mask2FormerForUniversalSegmentation(cfg).eval()


class Conv(C.OriginalMask2FormerCheckpointToOursConverter):
    def replace_swin_backbone(self, dst, src, config):             # ResNet-50 (detectron2 naming) -> HF ResNetBackbone
        pre = "pixel_level_module.encoder"
        def bn(s, d):
            return [(f"{s}.{k}", f"{d}.{k}") for k in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")]
        keys = [("backbone.stem.conv1.weight", f"{pre}.embedder.embedder.convolution.weight")] + bn("backbone.stem.conv1.norm", f"{pre}.embedder.embedder.normalization")
        for si, stage in enumerate(("res2", "res3", "res4", "res5")):
            for j in range(config.backbone_config.depths[si]):
                s, d = f"backbone.{stage}.{j}", f"{pre}.encoder.stages.{si}.layers.{j}"
                if j == 0:
                    keys += [(f"{s}.shortcut.weight", f"{d}.shortcut.convolution.weight")] + bn(f"{s}.shortcut.norm", f"{d}.shortcut.normalization")
                for k in range(3):
                    keys += [(f"{s}.conv{k + 1}.weight", f"{d}.layer.{k}.convolution.weight")] + bn(f"{s}.conv{k + 1}.norm", f"{d}.layer.{k}.normalization")
        self.pop_all(keys, dst, src)


conv = Conv(type("M", (), {"state_dict": lambda self: dict(sd)})(), cfg)
dst = C.TrackedStateDict(model.model.state_dict())
src = dict(sd)
conv.replace_pixel_module(dst, src)
conv.replace_transformer_module(dst, src)
missed = dst.diff()
missed = sorted(missed); print("missed (HF keys never filled):", len(missed), missed[:12])
print("not copied (checkpoint keys unused):", len(src), [k for k in src][:12])
state = {k: dst[k] for k in dst.to_track}
model.model.load_state_dict(state)
cls = C.TrackedStateDict(model.state_dict())
src2 = dict(sd)
conv.replace_universal_segmentation_module(cls, src2)
model.load_state_dict({k: cls[k] for k in cls.to_track})
model.save_pretrained(out)
Mask2FormerImageProcessor(do_resize=True, size={"shortest_edge": 1024, "longest_edge": 1024}, size_divisor=32, do_rescale=True, do_normalize=True,
                          image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], ignore_index=255, reduce_labels=False,
                          num_labels=3).save_pretrained(out)
print("saved", out, "params", sum(p.numel() for p in model.parameters()))
