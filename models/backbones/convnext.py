"""
https://github.com/facebookresearch/ConvNeXt-V2/blob/main/pretrained_models/convnextv2.py

# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

MIT License
=======================================================================

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
=======================================================================

Creative Commons is not a party to its public
licenses. Notwithstanding, Creative Commons may elect to apply one of
its public licenses to material it publishes and in those instances
will be considered the “Licensor.” The text of the Creative Commons
public licenses is dedicated to the public domain under the CC0 Public
Domain Dedication. Except for the limited purpose of indicating that
material is shared under a Creative Commons public license or as
otherwise permitted by the Creative Commons policies published at
creativecommons.org/policies, Creative Commons does not authorize the
use of the trademark "Creative Commons" or any other trademark or logo
of Creative Commons without its prior written consent including,
without limitation, in connection with any unauthorized modifications
to any of its public licenses or any other arrangements,
understandings, or agreements concerning use of licensed material. For
the avoidance of doubt, this paragraph does not form part of the
public licenses.

Creative Commons may be contacted at creativecommons.org.
"""


import sys
import logging
from typing import Literal, Tuple

import torch
import torch.nn as nn

from timm.models.layers import DropPath

from finetune.models.layers.norms import get_norm

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# TODO: add a uniform abstraction for str: config conversion
#       that also allows for easily loading pretrained weights
#       for parts of models
# CONFIGS = {
#     'convnext-tiny': {
#         'depths': (3, 3, 9, 3),
#         'dims': (96, 192, 384, 768),
#     },
#     'convnext-small': {
#         'depths': (3, 3, 27, 3),
#         'dims': (96, 192, 384, 768),
#     },
#     'convnext-base': {
#         'depths': (3, 3, 27, 3),
#         'dims': (128, 256, 512, 1024),
#     },
#     'convnext-large': {
#         'depths': (3, 3, 27, 3),
#         'dims': (192, 384, 768, 1536),
#     },
# }


class GRN(nn.Module):
    """ 
    GRN (Global Response Normalization) layer
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))

    def forward(self, x: torch.Tensor):
        gx = torch.norm(x, p=2, dim=(1, 2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x

class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim: int, drop_path: float = 0.):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=(1, 7, 7), padding='same', groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        inputs = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (B, C, Z, Y, X) -> (B, Z, Y, X, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 4, 1, 2, 3)  # (B, Z, Y, X, C) -> (B, C, Z, Y, X)
        x = inputs + self.drop_path(x)
        return x


class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2

    Args:
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(
        self,
        channel_in: int = 3,
        depths: Tuple = (3, 3, 9, 3),
        dims: Tuple = (96, 192, 384, 768),
        drop_path_rate: float =.1,
        return_intermediates: Literal[True] = True,
        num_stages: int = 4,
    ):
        super().__init__()
        
        self.dims = dims
        self.depths = depths
        self.channel_in = channel_in

        self.num_stages = num_stages
        self.return_intermediates = return_intermediates    

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            # TODO: using (kernel_size)^3 vs (1, kernel_size, kernel_size) 
            #       Conv3d layers
            nn.Conv3d(self.channel_in, self.dims[0], kernel_size=(4, 4, 4), stride=(4, 4, 4)),
            # TODO: add kwargs arg. to get_norm to pass eps=1e-6
            get_norm("LN", out_channels = self.dims[0], channel_dim = 1) 
        )
        self.downsample_layers.append(stem)
        for i in range(self.num_stages - 1):
            downsample_layer = nn.Sequential(
                    get_norm("LN", out_channels = self.dims[i], channel_dim = 1), # , eps=1e-6
                    # TODO: using (kernel_size)^3 vs (1, kernel_size, kernel_size) 
                    #       Conv3d layers
                    nn.Conv3d(self.dims[i], self.dims[i+1], kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            )
            self.downsample_layers.append(downsample_layer)

        # num_stages feature resolution stages, each consisting of multiple residual blocks
        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.num_stages):
            stage = nn.Sequential(
                *[Block(dim=self.dims[i], drop_path=dp_rates[cur + j]) for j in range(self.depths[i])]
            )
            self.stages.append(stage)
            cur += self.depths[i]

    def forward(self, x: torch.Tensor):
        intermediates = []
        for i in range(self.num_stages):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if self.return_intermediates:
                intermediates.append(x)
        
        if self.return_intermediates:
            # NOTE: we downsample 4x in the stem thus, the first intermediate
            #       feature map is at 1/8 = 2^{1/3} input resolution (i.e. p3)
            return x, {f"p{s+3}": feature for s, feature in enumerate(intermediates)}
        
        return x