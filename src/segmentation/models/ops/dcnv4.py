from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from segmentation.models.ops.dcnv4_func import DCNv4Function


class CenterFeatureScaleModule(nn.Module):
    def forward(self,
                query,
                center_feature_scale_proj_weight,
                center_feature_scale_proj_bias):
        center_feature_scale = F.linear(query,
                                        weight=center_feature_scale_proj_weight,
                                        bias=center_feature_scale_proj_bias).sigmoid()
        return center_feature_scale


class DCNv4(nn.Module):
    def __init__(
            self,
            channels=64,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=4,
            offset_scale=1.0,
            dw_kernel_size=None,
            center_feature_scale=False,
            remove_center=False,
            output_bias=True,
            without_pointwise=False,
            **kwargs):
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group

        # _d_per_group should be set to a power of 2 since its more efficient
        # in our CUDA implementation
        assert _d_per_group % 16 == 0

        self.channels = channels
        self.offset_scale = offset_scale
        self.kernel_size = kernel_size
        
        self.stride = stride
        self.dilation = dilation
        
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        
        self.offset_scale = offset_scale
        self.dw_kernel_size = dw_kernel_size
        self.center_feature_scale = center_feature_scale
        
        # flag to not include center voxel in the kernel
        self.remove_center = int(remove_center)
        self.without_pointwise = without_pointwise

        # number of points given by number of groups and kernel size
        self.K =  group * (kernel_size * kernel_size * kernel_size - self.remove_center)
        if dw_kernel_size is not None:
            self.offset_mask_dw = nn.Conv3d(channels, channels, dw_kernel_size, stride=1, 
                                            padding=(dw_kernel_size - 1) // 2, groups=channels)

        self.offset_mask = nn.Linear(channels, int(math.ceil((self.K * 4)/8)*8))

        if not without_pointwise:
            self.value_proj = nn.Linear(channels, channels)
            self.output_proj = nn.Linear(channels, channels, bias=output_bias)

        self._reset_parameters()

        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset_mask.weight.data, 0.)
        constant_(self.offset_mask.bias.data, 0.)
        if not self.without_pointwise:
            xavier_uniform_(self.value_proj.weight.data)
            constant_(self.value_proj.bias.data, 0.)
            xavier_uniform_(self.output_proj.weight.data)
            if self.output_proj.bias is not None:
                constant_(self.output_proj.bias.data, 0.)

    def forward(self, input, shape=None):
        N, D, H, W, C = input.shape
        L = D * H * W  # length of the input

        x = input
        if not self.without_pointwise:
            # mlp projection: channels -> channels
            x = self.value_proj(x)

        # x: (N,L,C) -> (N,D,H,W,C)
        # x = x.reshape(N, D, H, W, -1)

        # offset masks are offsets that we apply to the convolution kernel
        if self.dw_kernel_size is not None:
            # offset_mask_input: conv3d op. => (N,D,H,W,C) -> (N,C,D,H,W) -> (N,C,D,H,W)
            offset_mask_input = self.offset_mask_dw(input.permute(0, 4, 1, 2, 3))
            # offset_mask_input: (N,C,D,H,W) -> (N,D,H,W,C) -> (N, L, C)
            offset_mask_input = offset_mask_input.permute(0, 2, 3, 4, 1).view(N, L, C)
        else:
            offset_mask_input = input.view(N, L, C)
        
        # self.offset_mask: mlp projection: channels -> int(math.ceil((self.K * 4)/8)*8)
        # thus offset_mask: (N,L,C) -> (N,L,K') -> (N,D,H,W,K') where K' = int(math.ceil((self.K * 4)/8)*8)
        # where K is group * (kernel_size * kernel_size * kernel_size - self.remove_center)
        offset_mask = self.offset_mask(offset_mask_input).reshape(N, D, H, W, -1)

        x_proj = x

        x = DCNv4Function.apply(
            x, offset_mask,
            self.kernel_size, self.kernel_size, self.kernel_size,
            self.stride, self.stride, self.stride,
            self.pad, self.pad, self.pad,
            self.dilation, self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale,
            # TODO: make this configurable?
            256, # im2col_step
            self.remove_center
            )

        if self.center_feature_scale:
            # F.project (self.center_feature_scale_proj_weight): group -> channels 
            # center_feature_scale: (N,D,H,W,C) -> (N,D,H,W,group)
            center_feature_scale = self.center_feature_scale_module(
                x, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            # (N,D,H,W,G,1) -> # (N,D,H,W,G,group_channels) -> (N,D,H,W,C)
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            # scale x by center_feature_scale * x_proj
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale

        # x: (N,D,H,W,C) -> (N,L,C)
        # x = x.view(N, L, -1)

        if not self.without_pointwise:
            # mlp projection: channels -> channels
            # x: (N,D,H,W,C) -> (N,D,H,W,C)
            x = self.output_proj(x)

        return x