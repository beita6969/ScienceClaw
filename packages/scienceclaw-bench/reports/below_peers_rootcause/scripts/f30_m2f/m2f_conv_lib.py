"""Detectron2-free copy of the Mask2Former checkpoint converter of huggingface/transformers (Apache-2.0; pixel decoder + transformer decoder
key maps) plus a ResNet-50 backbone map; used to convert the PRBonn PhenoBench Mask2Former checkpoints to HF format."""
from pprint import pformat
from typing import Any

import torch
from torch import Tensor, nn
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, Mask2FormerModel
from transformers.utils import logging

StateDict = dict[str, Tensor]
logger = logging.get_logger()


class TrackedStateDict:
    def __init__(self, to_track: dict):
        """This class "tracks" a python dictionary by keeping track of which item is accessed.

        Args:
            to_track (Dict): The dictionary we wish to track
        """
        self.to_track = to_track
        self._seen: set[str] = set()

    def __getitem__(self, key: str) -> Any:
        return self.to_track[key]

    def __setitem__(self, key: str, item: Any):
        self._seen.add(key)
        self.to_track[key] = item

    def diff(self) -> list[str]:
        """This method returns a set difference between the keys in the tracked state dict and the one we have access so far.
        This is an effective method to check if we have update all the keys

        Returns:
            list[str]: List of keys not yet updated
        """
        return set(self.to_track.keys()) - self._seen

    def copy(self) -> dict:
        # proxy the call to the internal dictionary
        return self.to_track.copy()


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))
    return image




class OriginalMask2FormerCheckpointToOursConverter:
    def __init__(self, original_model: nn.Module, config: Mask2FormerConfig):
        self.original_model = original_model
        self.config = config

    def pop_all(self, renamed_keys: list[tuple[str, str]], dst_state_dict: StateDict, src_state_dict: StateDict):
        for src_key, dst_key in renamed_keys:
            dst_state_dict[dst_key] = src_state_dict.pop(src_key)

    def replace_maskformer_swin_backbone(
        self, dst_state_dict: StateDict, src_state_dict: StateDict, config: Mask2FormerConfig
    ):
        dst_prefix: str = "pixel_level_module.encoder"
        src_prefix: str = "backbone"

        renamed_keys = [
            (
                f"{src_prefix}.patch_embed.proj.weight",
                f"{dst_prefix}.model.embeddings.patch_embeddings.projection.weight",
            ),
            (f"{src_prefix}.patch_embed.proj.bias", f"{dst_prefix}.model.embeddings.patch_embeddings.projection.bias"),
            (f"{src_prefix}.patch_embed.norm.weight", f"{dst_prefix}.model.embeddings.norm.weight"),
            (f"{src_prefix}.patch_embed.norm.bias", f"{dst_prefix}.model.embeddings.norm.bias"),
        ]
        num_layers = len(config.backbone_config.depths)
        for layer_idx in range(num_layers):
            for block_idx in range(config.backbone_config.depths[layer_idx]):
                renamed_keys.extend(
                    [  # src, dst
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm1.weight",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_before.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm1.bias",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_before.bias",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.relative_position_bias_table",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.relative_position_bias_table",
                        ),
                    ]
                )
                # now we need to handle the attentions
                # read in weights + bias of input projection layer of cross-attention

                src_att_weight = src_state_dict[f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.weight"]
                src_att_bias = src_state_dict[f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.bias"]

                size = src_att_weight.shape[0]
                offset = size // 3
                dst_state_dict[
                    f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.query.weight"
                ] = src_att_weight[:offset, :]
                dst_state_dict[
                    f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.query.bias"
                ] = src_att_bias[:offset]

                dst_state_dict[
                    f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.key.weight"
                ] = src_att_weight[offset : offset * 2, :]
                dst_state_dict[
                    f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.key.bias"
                ] = src_att_bias[offset : offset * 2]

                dst_state_dict[
                    f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.value.weight"
                ] = src_att_weight[-offset:, :]
                dst_state_dict[
                    f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.value.bias"
                ] = src_att_bias[-offset:]

                # let's pop them
                src_state_dict.pop(f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.weight")
                src_state_dict.pop(f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.bias")
                # proj
                renamed_keys.extend(
                    [
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.proj.weight",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.output.dense.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.proj.bias",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.output.dense.bias",
                        ),
                    ]
                )

                # second norm
                renamed_keys.extend(
                    [
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm2.weight",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_after.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm2.bias",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_after.bias",
                        ),
                    ]
                )

                # mlp
                renamed_keys.extend(
                    [
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc1.weight",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.intermediate.dense.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc1.bias",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.intermediate.dense.bias",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc2.weight",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.output.dense.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc2.bias",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.output.dense.bias",
                        ),
                    ]
                )

                renamed_keys.extend(
                    [
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.relative_position_index",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.relative_position_index",
                        )
                    ]
                )

            if layer_idx < num_layers - 1:
                # patch merging
                renamed_keys.extend(
                    [
                        (
                            f"{src_prefix}.layers.{layer_idx}.downsample.reduction.weight",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.downsample.reduction.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.downsample.norm.weight",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.downsample.norm.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.downsample.norm.bias",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.downsample.norm.bias",
                        ),
                    ]
                )

            # hidden states norms
            renamed_keys.extend(
                [
                    (
                        f"{src_prefix}.norm{layer_idx}.weight",
                        f"{dst_prefix}.hidden_states_norms.{layer_idx}.weight",
                    ),
                    (
                        f"{src_prefix}.norm{layer_idx}.bias",
                        f"{dst_prefix}.hidden_states_norms.{layer_idx}.bias",
                    ),
                ]
            )
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def replace_swin_backbone(self, dst_state_dict: StateDict, src_state_dict: StateDict, config: Mask2FormerConfig):
        dst_prefix: str = "pixel_level_module.encoder"
        src_prefix: str = "backbone"

        renamed_keys = [
            (
                f"{src_prefix}.patch_embed.proj.weight",
                f"{dst_prefix}.embeddings.patch_embeddings.projection.weight",
            ),
            (f"{src_prefix}.patch_embed.proj.bias", f"{dst_prefix}.embeddings.patch_embeddings.projection.bias"),
            (f"{src_prefix}.patch_embed.norm.weight", f"{dst_prefix}.embeddings.norm.weight"),
            (f"{src_prefix}.patch_embed.norm.bias", f"{dst_prefix}.embeddings.norm.bias"),
        ]

        for layer_idx in range(len(config.backbone_config.depths)):
            for block_idx in range(config.backbone_config.depths[layer_idx]):
                renamed_keys.extend(
                    [  # src, dst
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm1.weight",
                            f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_before.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm1.bias",
                            f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_before.bias",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.relative_position_bias_table",
                            f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.relative_position_bias_table",
                        ),
                    ]
                )
                # now we need to handle the attentions
                # read in weights + bias of input projection layer of cross-attention

                src_att_weight = src_state_dict[f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.weight"]
                src_att_bias = src_state_dict[f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.bias"]

                size = src_att_weight.shape[0]
                offset = size // 3
                dst_state_dict[
                    f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.query.weight"
                ] = src_att_weight[:offset, :]
                dst_state_dict[
                    f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.query.bias"
                ] = src_att_bias[:offset]

                dst_state_dict[
                    f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.key.weight"
                ] = src_att_weight[offset : offset * 2, :]
                dst_state_dict[
                    f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.key.bias"
                ] = src_att_bias[offset : offset * 2]

                dst_state_dict[
                    f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.value.weight"
                ] = src_att_weight[-offset:, :]
                dst_state_dict[
                    f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.value.bias"
                ] = src_att_bias[-offset:]

                # let's pop them
                src_state_dict.pop(f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.weight")
                src_state_dict.pop(f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.bias")
                # proj
                renamed_keys.extend(
                    [
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.proj.weight",
                            f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.output.dense.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.proj.bias",
                            f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.output.dense.bias",
                        ),
                    ]
                )

                # second norm
                renamed_keys.extend(
                    [
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm2.weight",
                            f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_after.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm2.bias",
                            f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_after.bias",
                        ),
                    ]
                )

                # mlp
                renamed_keys.extend(
                    [
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc1.weight",
                            f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.intermediate.dense.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc1.bias",
                            f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.intermediate.dense.bias",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc2.weight",
                            f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.output.dense.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc2.bias",
                            f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.output.dense.bias",
                        ),
                    ]
                )

                renamed_keys.extend(
                    [
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.relative_position_index",
                            f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.relative_position_index",
                        )
                    ]
                )

            if layer_idx < 3:
                # patch merging
                renamed_keys.extend(
                    [
                        (
                            f"{src_prefix}.layers.{layer_idx}.downsample.reduction.weight",
                            f"{dst_prefix}.encoder.layers.{layer_idx}.downsample.reduction.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.downsample.norm.weight",
                            f"{dst_prefix}.encoder.layers.{layer_idx}.downsample.norm.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.downsample.norm.bias",
                            f"{dst_prefix}.encoder.layers.{layer_idx}.downsample.norm.bias",
                        ),
                    ]
                )

            # hidden states norms
            renamed_keys.extend(
                [
                    (
                        f"{src_prefix}.norm{layer_idx}.weight",
                        f"{dst_prefix}.hidden_states_norms.stage{layer_idx + 1}.weight",
                    ),
                    (
                        f"{src_prefix}.norm{layer_idx}.bias",
                        f"{dst_prefix}.hidden_states_norms.stage{layer_idx + 1}.bias",
                    ),
                ]
            )
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    # Backbone + Pixel Decoder
    def replace_pixel_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = "pixel_level_module.decoder"
        src_prefix: str = "sem_seg_head.pixel_decoder"

        self.replace_swin_backbone(dst_state_dict, src_state_dict, self.config)

        def rename_keys_for_weight_bias(src_prefix: str, dst_prefix: str):
            return [
                (f"{src_prefix}.weight", f"{dst_prefix}.weight"),
                (f"{src_prefix}.bias", f"{dst_prefix}.bias"),
            ]

        def rename_keys_for_self_attn(src_prefix: str, dst_prefix: str):
            self_attn_keys = []
            self_attn_keys.extend(
                rename_keys_for_weight_bias(f"{src_prefix}.attention_weights", f"{dst_prefix}.attention_weights")
            )
            self_attn_keys.extend(
                rename_keys_for_weight_bias(f"{src_prefix}.output_proj", f"{dst_prefix}.output_proj")
            )
            self_attn_keys.extend(
                rename_keys_for_weight_bias(f"{src_prefix}.sampling_offsets", f"{dst_prefix}.sampling_offsets")
            )
            self_attn_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.value_proj", f"{dst_prefix}.value_proj"))

            return self_attn_keys

        def rename_keys_for_encoder_layer(src_prefix: str, dst_prefix: str):
            encoder_keys = []
            encoder_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.linear1", f"{dst_prefix}.fc1"))
            encoder_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.linear2", f"{dst_prefix}.fc2"))
            encoder_keys.extend(
                rename_keys_for_weight_bias(f"{src_prefix}.norm1", f"{dst_prefix}.self_attn_layer_norm")
            )
            encoder_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.norm2", f"{dst_prefix}.final_layer_norm"))
            encoder_keys.extend(rename_keys_for_self_attn(f"{src_prefix}.self_attn", f"{dst_prefix}.self_attn"))

            return encoder_keys

        # convolution layer for final features
        renamed_keys = [
            (f"{src_prefix}.adapter_1.weight", f"{dst_prefix}.adapter_1.0.weight"),
            (f"{src_prefix}.adapter_1.norm.weight", f"{dst_prefix}.adapter_1.1.weight"),
            (f"{src_prefix}.adapter_1.norm.bias", f"{dst_prefix}.adapter_1.1.bias"),
        ]

        renamed_keys.extend(
            [
                (f"{src_prefix}.layer_1.weight", f"{dst_prefix}.layer_1.0.weight"),
                (f"{src_prefix}.layer_1.norm.weight", f"{dst_prefix}.layer_1.1.weight"),
                (f"{src_prefix}.layer_1.norm.bias", f"{dst_prefix}.layer_1.1.bias"),
            ]
        )

        # proj layers
        for i in range(3):
            for j in range(2):
                renamed_keys.extend(
                    [
                        (f"{src_prefix}.input_proj.{i}.{j}.weight", f"{dst_prefix}.input_projections.{i}.{j}.weight"),
                        (f"{src_prefix}.input_proj.{i}.{j}.bias", f"{dst_prefix}.input_projections.{i}.{j}.bias"),
                    ]
                )

        renamed_keys.extend([(f"{src_prefix}.transformer.level_embed", f"{dst_prefix}.level_embed")])

        # layers
        for layer_idx in range(self.config.encoder_layers):
            renamed_keys.extend(
                rename_keys_for_encoder_layer(
                    f"{src_prefix}.transformer.encoder.layers.{layer_idx}", f"{dst_prefix}.encoder.layers.{layer_idx}"
                )
            )

        # proj
        renamed_keys.extend(
            [
                (f"{src_prefix}.mask_features.weight", f"{dst_prefix}.mask_projection.weight"),
                (f"{src_prefix}.mask_features.bias", f"{dst_prefix}.mask_projection.bias"),
            ]
        )
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    # Transformer Decoder
    def rename_keys_in_masked_attention_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = "transformer_module.decoder"
        src_prefix: str = "sem_seg_head.predictor"

        rename_keys = []
        for i in range(self.config.decoder_layers - 1):
            rename_keys.append(
                (
                    f"{src_prefix}.transformer_self_attention_layers.{i}.self_attn.out_proj.weight",
                    f"{dst_prefix}.layers.{i}.self_attn.out_proj.weight",
                )
            )
            rename_keys.append(
                (
                    f"{src_prefix}.transformer_self_attention_layers.{i}.self_attn.out_proj.bias",
                    f"{dst_prefix}.layers.{i}.self_attn.out_proj.bias",
                )
            )

            rename_keys.append(
                (
                    f"{src_prefix}.transformer_self_attention_layers.{i}.norm.weight",
                    f"{dst_prefix}.layers.{i}.self_attn_layer_norm.weight",
                )
            )
            rename_keys.append(
                (
                    f"{src_prefix}.transformer_self_attention_layers.{i}.norm.bias",
                    f"{dst_prefix}.layers.{i}.self_attn_layer_norm.bias",
                )
            )

            rename_keys.append(
                (
                    f"{src_prefix}.transformer_cross_attention_layers.{i}.multihead_attn.in_proj_weight",
                    f"{dst_prefix}.layers.{i}.cross_attn.in_proj_weight",
                )
            )
            rename_keys.append(
                (
                    f"{src_prefix}.transformer_cross_attention_layers.{i}.multihead_attn.in_proj_bias",
                    f"{dst_prefix}.layers.{i}.cross_attn.in_proj_bias",
                )
            )
            rename_keys.append(
                (
                    f"{src_prefix}.transformer_cross_attention_layers.{i}.multihead_attn.out_proj.weight",
                    f"{dst_prefix}.layers.{i}.cross_attn.out_proj.weight",
                )
            )
            rename_keys.append(
                (
                    f"{src_prefix}.transformer_cross_attention_layers.{i}.multihead_attn.out_proj.bias",
                    f"{dst_prefix}.layers.{i}.cross_attn.out_proj.bias",
                )
            )

            rename_keys.append(
                (
                    f"{src_prefix}.transformer_cross_attention_layers.{i}.norm.weight",
                    f"{dst_prefix}.layers.{i}.cross_attn_layer_norm.weight",
                )
            )
            rename_keys.append(
                (
                    f"{src_prefix}.transformer_cross_attention_layers.{i}.norm.bias",
                    f"{dst_prefix}.layers.{i}.cross_attn_layer_norm.bias",
                )
            )

            rename_keys.append(
                (f"{src_prefix}.transformer_ffn_layers.{i}.linear1.weight", f"{dst_prefix}.layers.{i}.fc1.weight")
            )
            rename_keys.append(
                (f"{src_prefix}.transformer_ffn_layers.{i}.linear1.bias", f"{dst_prefix}.layers.{i}.fc1.bias")
            )
            rename_keys.append(
                (f"{src_prefix}.transformer_ffn_layers.{i}.linear2.weight", f"{dst_prefix}.layers.{i}.fc2.weight")
            )
            rename_keys.append(
                (f"{src_prefix}.transformer_ffn_layers.{i}.linear2.bias", f"{dst_prefix}.layers.{i}.fc2.bias")
            )
            rename_keys.append(
                (
                    f"{src_prefix}.transformer_ffn_layers.{i}.norm.weight",
                    f"{dst_prefix}.layers.{i}.final_layer_norm.weight",
                )
            )
            rename_keys.append(
                (
                    f"{src_prefix}.transformer_ffn_layers.{i}.norm.bias",
                    f"{dst_prefix}.layers.{i}.final_layer_norm.bias",
                )
            )

        return rename_keys

    def replace_masked_attention_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = "transformer_module.decoder"
        src_prefix: str = "sem_seg_head.predictor"

        renamed_keys = self.rename_keys_in_masked_attention_decoder(dst_state_dict, src_state_dict)

        # add more
        renamed_keys.extend(
            [
                (f"{src_prefix}.decoder_norm.weight", f"{dst_prefix}.layernorm.weight"),
                (f"{src_prefix}.decoder_norm.bias", f"{dst_prefix}.layernorm.bias"),
            ]
        )

        mlp_len = 3
        for i in range(mlp_len):
            renamed_keys.extend(
                [
                    (
                        f"{src_prefix}.mask_embed.layers.{i}.weight",
                        f"{dst_prefix}.mask_predictor.mask_embedder.{i}.0.weight",
                    ),
                    (
                        f"{src_prefix}.mask_embed.layers.{i}.bias",
                        f"{dst_prefix}.mask_predictor.mask_embedder.{i}.0.bias",
                    ),
                ]
            )

        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def replace_keys_qkv_transformer_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = "transformer_module.decoder.layers"
        src_prefix: str = "sem_seg_head.predictor"
        for i in range(self.config.decoder_layers - 1):
            # read in weights + bias of input projection layer of self-attention
            in_proj_weight = src_state_dict.pop(
                f"{src_prefix}.transformer_self_attention_layers.{i}.self_attn.in_proj_weight"
            )
            in_proj_bias = src_state_dict.pop(
                f"{src_prefix}.transformer_self_attention_layers.{i}.self_attn.in_proj_bias"
            )
            # next, add query, keys and values (in that order) to the state dict
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]

    def replace_transformer_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = "transformer_module"
        src_prefix: str = "sem_seg_head.predictor"

        self.replace_masked_attention_decoder(dst_state_dict, src_state_dict)

        renamed_keys = [
            (f"{src_prefix}.query_embed.weight", f"{dst_prefix}.queries_embedder.weight"),
            (f"{src_prefix}.query_feat.weight", f"{dst_prefix}.queries_features.weight"),
            (f"{src_prefix}.level_embed.weight", f"{dst_prefix}.level_embed.weight"),
        ]

        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)
        self.replace_keys_qkv_transformer_decoder(dst_state_dict, src_state_dict)

    def replace_universal_segmentation_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = ""
        src_prefix: str = "sem_seg_head.predictor"

        renamed_keys = [
            (f"{src_prefix}.class_embed.weight", f"{dst_prefix}class_predictor.weight"),
            (f"{src_prefix}.class_embed.bias", f"{dst_prefix}class_predictor.bias"),
        ]

        logger.info(f"Replacing keys {pformat(renamed_keys)}")
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def convert(self, mask2former: Mask2FormerModel) -> Mask2FormerModel:
        dst_state_dict = TrackedStateDict(mask2former.state_dict())
        src_state_dict = self.original_model.state_dict()

        self.replace_pixel_module(dst_state_dict, src_state_dict)
        self.replace_transformer_module(dst_state_dict, src_state_dict)

        logger.info(f"Missed keys are {pformat(dst_state_dict.diff())}")
        logger.info(f"Not copied keys are {pformat(src_state_dict.keys())}")
        logger.info("🙌 Done")

        state_dict = {key: dst_state_dict[key] for key in dst_state_dict.to_track}
        mask2former.load_state_dict(state_dict)
        return mask2former

    def convert_universal_segmentation(
        self, mask2former: Mask2FormerForUniversalSegmentation
    ) -> Mask2FormerForUniversalSegmentation:
        dst_state_dict = TrackedStateDict(mask2former.state_dict())
        src_state_dict = self.original_model.state_dict()

        self.replace_universal_segmentation_module(dst_state_dict, src_state_dict)

        state_dict = {key: dst_state_dict[key] for key in dst_state_dict.to_track}
        mask2former.load_state_dict(state_dict)

        return mask2former
