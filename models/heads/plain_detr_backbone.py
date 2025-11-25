"""
Adapted from:
# ------------------------------------------------------------------------
# Plain-DETR
# Copyright (c) 2023 Xi'an Jiaotong University & Microsoft Research Asia.
# Licensed under The MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""

import logging
from typing import List, Optional, Union, Tuple

from hydra.utils import instantiate

import torch
from torch import nn
import torch.nn.functional as F

from cell_observatory_finetune.models.layers.layers import LayerNorm3D
from cell_observatory_finetune.models.adapters.vit_adapter import build_adapter


class PlainDETRBackbone(nn.Module):
    def __init__(
        self,
        backbone_args: dict,
        adapter_args: Optional[dict],
        backbone_embed_dims: List[int],
        train_backbone: bool,
        blocks_to_train: Optional[List[str]] = None,
        use_layernorm: bool = True,
    ):
        super().__init__()

        self.backbone = build_backbone(backbone_args)

        self.blocks_to_train = blocks_to_train

        for _, (name, parameter) in enumerate(self.backbone.named_parameters()):
            train_condition = any(f".{b}." in name for b in self.blocks_to_train) if self.blocks_to_train else True
            if (not train_backbone) or "mask_token" in name or (not train_condition):
                parameter.requires_grad_(False)
        
        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            self.layer_norms = nn.ModuleList([LayerNorm3D(embed_dim) for embed_dim in backbone_embed_dims])

        if adapter_args is not None:
            self.with_backbone_adapter = True
            self.adapter = build_adapter(adapter_args)
        else:
            # TODO: implement logic to handle positional encodings without adapter
            raise NotImplementedError("Backbone adapter must be specified for PlainDETRBackbone.")
            # self.with_backbone_adapter = False

    def forward(self, data_sample: dict) -> List[dict]:
        features = self.backbone.forward_features(data_sample["data_tensor"])

        if self.with_backbone_adapter:
            # dict[str, Tensor]: {"1": f1, ...}
            features_dict = self.adapter(data_sample["data_tensor"], features)
            # turn into ordered list [f1, f2, f3, f4] assuming keys "1","2","3","4"
            features_list = [features_dict[k] for k in sorted(features_dict.keys(), key=int)]
        else:
            # we assume backbone.forward_features already returns a list of feature maps
            features_list = features

        if self.use_layernorm:
            features_list = [ln(x).contiguous() for ln, x in zip(self.layer_norms, features_list)]

        out = []
        m = data_sample["metainfo"]["padding_mask"]  # [B, Z, Y, X]
        assert m is not None, "data_sample should include padding mask"
        for lvl, x in enumerate(features_list):
            # x: [B, C, D, H, W]
            mask = F.interpolate(m[None].float(), size=x.shape[-3:], mode="nearest").to(torch.bool)[0]
            out.append({"x": x, "mask": mask})

        return out


def build_backbone(backbone_args: dict) -> PlainDETRBackbone:
    input_channels = backbone_args.get("input_channels")
    input_shape = backbone_args.get("input_shape")
    if input_channels is not None:
        assert backbone_args["input_fmt"][-1] == "C", \
            "Input format must end with 'C' when specifying input_channels."
        backbone_args["input_shape"] = list(input_shape)
        backbone_args["input_shape"][-1] = input_channels
        backbone_args["input_shape"] = tuple(backbone_args["input_shape"])
    
    backbone_args.pop("input_channels")

    if backbone_args.get("model") == "FinetuneMaskedAutoEncoder":
        backbone_args["_target_"] = "cell_observatory_finetune.models.meta_arch.maskedautoencoder.FinetuneMaskedAutoEncoder"
        backbone_args.pop("model")
        model = instantiate(backbone_args)
    else:
        raise NotImplementedError(f"Backbone model {backbone_args.get('model')} not implemented.")

    return model


def build_backbone_wrapper(backbone_wrapper_args: dict, adapter_args: Optional[dict]) -> nn.Module:
    model = PlainDETRBackbone(
        backbone_args=backbone_wrapper_args["backbone_args"],
        adapter_args=adapter_args,
        backbone_embed_dims=backbone_wrapper_args["backbone_embed_dims"],
        train_backbone=backbone_wrapper_args["train_backbone"],
        blocks_to_train=backbone_wrapper_args.get("blocks_to_train"),
        use_layernorm=backbone_wrapper_args.get("use_layernorm", True),
    )
    return model