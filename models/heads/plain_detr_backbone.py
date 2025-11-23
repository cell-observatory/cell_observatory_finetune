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

import torch
from torch import nn
import torch.nn.functional as F

from cell_observatory_finetune.models.layers.layers import LayerNorm3D


class PlainDETRBackbone(nn.Module):
    def __init__(
        self,
        patch_size: Tuple[int, int, int],
        backbone: nn.Module,
        backbone_embed_dims: List[int],
        train_backbone: bool,
        blocks_to_train: Optional[List[str]] = None,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.blocks_to_train = blocks_to_train

        for _, (name, parameter) in enumerate(self.backbone.named_parameters()):
            train_condition = any(f".{b}." in name for b in self.blocks_to_train) if self.blocks_to_train else True
            if (not train_backbone) or "mask_token" in name or (not train_condition):
                parameter.requires_grad_(False)
        
        self.patch_size = patch_size
        self.strides = [patch_size]

        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            self.layer_norms = nn.ModuleList([LayerNorm3D(embed_dim) for embed_dim in backbone_embed_dims])

    def forward(self, data_sample: dict) -> List[dict]:
        xs = self.backbone.forward_features(data_sample["data_tensor"])
        if self.use_layernorm:
            xs = [ln(x).contiguous() for ln, x in zip(self.layer_norms, xs)]

        xs = [torch.cat(xs, axis=1)]

        out = []
        for x in xs:
            m = data_sample["metainfo"]["padding_mask"]
            assert m is not None, "data_sample should include padding mask"
            mask = F.interpolate(m[None].float(), size=x.shape[-3:]).to(torch.bool)[0]
            out.append({"x": x, "mask": mask})
        return out