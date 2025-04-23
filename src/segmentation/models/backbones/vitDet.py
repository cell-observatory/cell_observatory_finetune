"""
https://github.com/facebookresearch/detectron2/blob/9604f5995cc628619f0e4fd913453b4d7d61db3f/detectron2/modeling/backbone/vit.py

Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import math
import logging
from typing import Optional, Union, Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import fvcore.nn.weight_init as weight_init
from timm.models.layers import DropPath, Mlp

from segmentation.models.backbones.backbone_utils import (
    PatchEmbed,
    add_decomposed_rel_pos,
    get_abs_pos,
    window_partition,
    window_unpartition,
    CNNBlockBase,
    _assert_strides_are_log2_contiguous
)
from segmentation.models.utils.model_utils import Conv3d
from segmentation.models.backbones.batch_norm import get_norm


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[int] = None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            self.rel_pos_d = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[2] - 1, head_dim))

            if not rel_pos_zero_init:
                nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
                nn.init.trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x):
        B, D, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, D * H * W, C)
        qkv = self.qkv(x).reshape(B, D * H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, D * H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, D * H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_d, self.rel_pos_h, self.rel_pos_w, (D, H, W), (D, H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, D, H, W, -1).permute(0, 2, 3, 4, 1, 5).reshape(B, D, H, W, -1)
        x = self.proj(x)

        return x


class ResBottleneckBlock(CNNBlockBase):
    """
    The standard bottleneck residual block without the last activation layer.
    It contains 3 conv layers with kernels 1x1, 3x3, 1x1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_channels: int,
        norm: Union[str, Callable] = "LN",
        act_layer: Callable = nn.GELU,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            act_layer (callable): activation for all conv layers.
        """
        super().__init__(in_channels, out_channels, 1)

        self.conv1 = Conv3d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = get_norm(norm, bottleneck_channels)
        self.act1 = act_layer()

        self.conv2 = Conv3d(
            bottleneck_channels,
            bottleneck_channels,
            3,
            padding=1,
            bias=False,
        )
        self.norm2 = get_norm(norm, bottleneck_channels)
        self.act2 = act_layer()

        self.conv3 = Conv3d(bottleneck_channels, out_channels, 1, bias=False)
        self.norm3 = get_norm(norm, out_channels)

        for layer in [self.conv1, self.conv2, self.conv3]:
            # Kaiming initialization (caffe style MSRA)
            weight_init.c2_msra_fill(layer)

        for layer in [self.norm1, self.norm2]:
            layer.weight.data.fill_(1.0)
            layer.bias.data.zero_()

        # zero init last norm layer
        self.norm3.weight.data.zero_()
        self.norm3.bias.data.zero_()

    def forward(self, x):
        out = x
        for layer in self.children():
            out = layer(out)
        out = x + out
        return out


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
        norm_layer: str = "LN",
        act_layer: Callable = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        use_residual_block: bool = False,
        input_size: Optional[int] = None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_residual_block (bool): If True, use a residual block after the MLP block.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.norm1 = get_norm(norm_layer, out_channels = dim, channel_dim=4)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size, window_size),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = get_norm(norm_layer, out_channels = dim, channel_dim=4)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

        self.window_size = window_size

        self.use_residual_block = use_residual_block
        if use_residual_block:
            # use a residual block with bottleneck channel as dim // 2
            self.residual = ResBottleneckBlock(
                in_channels=dim,
                out_channels=dim,
                bottleneck_channels=dim // 2,
                norm="LN",
                act_layer=act_layer,
            )

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)

        # window partition x post patch embedding
        if self.window_size > 0:
            D, H, W = x.shape[1], x.shape[2], x.shape[3]
            # returns: (num_windows, window_size, window_size, window_size, C)
            x, pad_dhw = window_partition(x, self.window_size)

        x = self.attn(x)

        # reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_dhw, (D, H, W))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.use_residual_block:
            x = self.residual(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)

        return x


class ViT(nn.Module):
    """
    This module implements Vision Transformer (ViT) backbone in :paper:`vitdet`.
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
        self,
        weights=None,
        img_size: int = 1024,
        patch_size: int = 16,
        channel_in: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        norm_layer: str = "LN",
        act_layer: nn.Module = nn.GELU, # TODO: Switch to same logic as get_norm
        use_abs_pos: bool = True,
        use_rel_pos: bool =False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        window_block_indexes=(),
        residual_block_indexes=(),
        pretrain_img_size: int = 224,
        pretrain_use_cls_token: bool = True,
        out_feature: str = "last_feat",
    ):
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            channel_in (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (str): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            window_block_indexes (list): Indexes for blocks using window attention.
            residual_block_indexes (list): Indexes for blocks using conv propagation.
            pretrain_img_size (int): input image size for pretraining models.
            pretrain_use_cls_token (bool): If True, pretrainig models use class token.
            out_feature (str): name of the feature from the last block.
        """
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size, patch_size),
            stride=(patch_size, patch_size, patch_size),
            in_chans=channel_in,
            embed_dim=embed_dim,
        )

        if use_abs_pos:
            # initialize absolute positional embedding with pretrain image size
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                use_residual_block=i in residual_block_indexes,
                input_size=(img_size // patch_size, img_size // patch_size, img_size // patch_size),
            )

            # TODO: Support activation checkpointing
            self.blocks.append(block)

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # TODO: unify pretrained weight loading logic across models 
        # if weights is not None:
        #     if isinstance(weights, str):
        #         checkpoint = torch.load(weights, map_location="cpu")
        #         # if "model" in checkpoint:
        #         #     # Some checkpoints save under a "model" key 
        #         #     checkpoint = checkpoint["model"]
        #         missing, unexpected = self.load_state_dict(checkpoint, strict=False)
        #         print(f"[ViT] Loaded weights from {weights}")
        #         if missing:
        #             print(f"[ViT] Missing keys: {missing}")
        #         if unexpected:
        #             print(f"[ViT] Unexpected keys: {unexpected}")
        #     else:
        #         raise ValueError(f"'weights' must be a path string, got {type(weights)}")
        # else:
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def output_shape(self):
        """
        Returns:
            dict: mapping from feature map name to shape.
        """
        # this is a backward-compatible default
        return {
            name: {
                "channels": self._out_feature_channels[name], 
                "stride": self._out_feature_strides[name]
            }
            for name in self._out_features
        }

    def forward(self, x: torch.Tensor):
        # 3D Conv with stride=kernel_size=patch_size
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            # get_abs_pos (see utils.py) retrieves absolute positional embedding
            # it will interpolate the positional embedding to the input size
            # if needed (i.e. if pretrain_img_size != img_size)
            x = x + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2], x.shape[3])
            )

        for blk in self.blocks:
            x = blk(x)

        # x: (B, C, D, H, W)
        # self._out_features: ["last_feat"]
        outputs = {self._out_features[0]: x.permute(0, 4, 1, 2, 3)}
        return outputs


class SimpleFeaturePyramid(nn.Module):
    """
    This module implements SimpleFeaturePyramid in :paper:`vitdet`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        net: nn.Module,
        in_feature: str,
        out_channels: int,
        scale_factors: List[float],
        top_block: Optional[Callable] = None,
        norm: str = "LN",
        square_pad: int = 0,
    ):
        """
        Args:
            net (nn.Module): module representing the subnetwork backbone.
            in_feature (str): names of the input feature maps coming
                from the net.
            out_channels (int): number of channels in the output feature maps.
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                pyramid output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra pyramid levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            norm (str): the normalization to use.
            square_pad (int): If > 0, require input images to be padded to specific square size.
        """
        super(SimpleFeaturePyramid, self).__init__()

        self.scale_factors = scale_factors

        input_shapes = net.output_shape()
        # input_shapes ex: {"last_features": {"channels": 256, "stride": 4}, ...}
        # divide output stride by scale factor to get the stride of the feature map
        # after upsampling/downsampling in the pyramid for VitDet (power of 2 required)
        strides = [int(input_shapes[in_feature]["stride"] / scale) for scale in scale_factors]
        _assert_strides_are_log2_contiguous(strides)

        dim = input_shapes[in_feature]["channels"]
        self.stages = []
        use_bias = norm == ""
        # the scale factors are in descending order 4x -> 2x -> ....
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            # ConvTranspose3d is used for upsampling, maxpool3d for downsampling
            # we decrease the number of channels by 2 for each 2x upsampling
            # scale dictates the amount of upsampling/downsampling in the pyramid 
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose3d(dim, dim // 2, kernel_size=2, stride=2),
                    get_norm(norm, out_channels = dim // 2, channel_dim = 1),
                    nn.GELU(),
                    nn.ConvTranspose3d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose3d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool3d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            # irrespective of the scale factor, we always apply 1x1 -> 3x3 conv layers
            # to set channel_dim to out_channels
            layers.extend(
                [
                    Conv3d(
                        out_dim,
                        out_channels,
                        kernel_size=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels = out_channels, channel_dim = 1),
                    ),
                    Conv3d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels = out_channels, channel_dim = 1),
                    ),
                ]
            )
            layers = nn.Sequential(*layers)

            stage = int(math.log2(strides[idx]))
            self.add_module(f"simfp_{stage}", layers)
            self.stages.append(layers)

        self.net = net
        self.in_feature = in_feature
        self.top_block = top_block
        
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        # in accordance with resnet naming conventions
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        
        # top block output feature maps.
        # if top_block is not None, extend out_feature_strides by
        # the number of levels in the top block
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        # decrease/increase input strides from ViT backbone by scale factors
        # p2, p3, p4, p5, p6, ... are the resulting feature map names (log2 stride denomination)
        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        self._square_pad = square_pad

        self.out_channels = out_channels # make accessible for other modules, e.g., mask head

    # TODO: implement padding constraints logic
    # @property
    # def padding_constraints(self):
    #     return {
    #         "size_divisiblity": self._size_divisibility,
    #         "square_size": self._square_pad,
    #     }

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape (N,C,D,H,W). D,H,W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        bottom_up_features = self.net(x)
        # retrieve specified feature stage from backbone 
        # generally returns "last_feat": feat_map
        features = bottom_up_features[self.in_feature]
        results = []

        # apply the deconv/conv stages to the input feature map 
        # defined in the constructor according to the scale factors
        for stage in self.stages:
            results.append(stage(features))

        # optionally apply the top block to the last feature maps 
        # in in_feature list
        if self.top_block is not None:
            # retrieve the input feature map for the top block
            # if the top block input feature is in the bottom up features
            # use the bottom up feature map, else use feature map from fpn
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        # return dict of feat map names (p2,p3,...) and corresponding feature map
        return {f: res for f, res in zip(self._out_features, results)}


def get_vit_lr_decay_rate(name: str, lr_decay_rate: float = 1.0, num_layers: int = 12):
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.

    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)


# TODO: deduplicate with FPN
class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x: torch.Tensor):
        return [F.max_pool3d(x, kernel_size=1, stride=2, padding=0)]