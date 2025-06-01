"""
https://github.com/OpenGVLab/InternImage/blob/31c962dc6c1ceb23e580772f7daaa6944694fbe6/detection/mmdet_custom/models/backbones/intern_image.py

MIT License

Copyright (c) 2022 OpenGVLab

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
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.models.layers import DropPath, trunc_normal_

from segmentation.models.ops.dcn import DCN
from segmentation.layers.norms import get_norm
from segmentation.models.ops.dcnv4 import DCNv4
from segmentation.layers.activations import get_activation



#----------------------------------------------- NOT USED FOR INSTANCE SEG. ---------------------------------------------


class CrossAttention(nn.Module):
    """ Cross Attention Module.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 8
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: False.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        attn_head_dim (int, optional): Dimension of attention head.
        out_dim (int, optional): Dimension of output.
    """

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 attn_head_dim=None,
                 out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        assert all_head_dim == dim, "all_head_dim must be equal to dim"
        
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, k=None, v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = self.k_bias
            v_bias = self.v_bias

        # q: (B, Nq, C) -> (B, Nq, all_head_dim)
        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        # (B, Nq, 1, num_heads, dim) -> (1, B, num_heads, Nq, dim)
        # -> (B, num_heads, Nq, dim)
        q = q.reshape(B, N, 1, self.num_heads,
                      -1).permute(2, 0, 3, 1,
                                  4).squeeze(0) 

        # k: (B, N_k, C) -> (B, N_k, all_head_dim)
        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        # (B, N_k, 1, num_heads, dim) -> (1, B, num_heads, N_k, dim)
        # -> (B, num_heads, N_k, dim)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1,
                                                             4).squeeze(0)

        # v: (B, N_v, C) -> (B, N_v, all_head_dim)
        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)
        # (B, N_v, 1, num_heads, dim) -> (1, B, num_heads, N_v, dim)
        # -> (B, num_heads, N_v, dim)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1,
                                                             4).squeeze(0)

        q = q * self.scale
        # (B,num_heads, N_q,dim) @ (B,num_heads,N_k,dim)
        # -> (B, num_heads, N_q, N_k)
        attn = (q @ k.transpose(-2, -1)) 

        # (B, num_heads, N_q, N_k)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (B, num_heads, N_q, N_k) @ (B, num_heads, N_v, dim)
        # (B, num_heads, N_q, dim) -> (B, N_q, num_heads, dim)
        # -> (B, N_q, all_head_dim)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        # (B, N_q, all_head_dim) -> (B, N_q, out_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class AttentiveBlock(nn.Module):
    """ Attentive Block of InternImage.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 8.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: False.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop (float, optional): Dropout rate. Default: 0.0.
        attn_drop (float, optional): Attention dropout rate. Default: 0.0.
        drop_path (float | tuple[float], optional): Stochastic depth rate.
            Default: 0.0.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm.
        attn_head_dim (int, optional): Dimension of attention head. Default: None.
        out_dim (int, optional): Dimension of output. Default: None.
    """

    def __init__(self,
                 dim,
                 num_heads,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer='LN',
                 attn_head_dim=None,
                 out_dim=None):
        super().__init__()

        self.norm1_q = get_norm(norm=norm_layer, out_channels=dim)
        self.norm1_k = get_norm(norm=norm_layer, out_channels=dim)
        self.norm1_v = get_norm(norm=norm_layer, out_channels=dim)
        self.cross_dcn = CrossAttention(dim,
                                        num_heads=num_heads,
                                        qkv_bias=qkv_bias,
                                        qk_scale=qk_scale,
                                        attn_drop=attn_drop,
                                        proj_drop=drop,
                                        attn_head_dim=attn_head_dim,
                                        out_dim=out_dim)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self,
                x_q,
                x_kv,
                pos_q,
                pos_k,
                bool_masked_pos,
                rel_pos_bias=None):
        x_q = self.norm1_q(x_q + pos_q)
        x_k = self.norm1_k(x_kv + pos_k)
        x_v = self.norm1_v(x_kv)

        x = self.cross_dcn(x_q, k=x_k, v=x_v)

        return x


class AttentionPoolingBlock(AttentiveBlock):

    def forward(self, x):
        # x: (B, N, C) -> x_q: (B, 1, C)
        x_q = x.mean(1, keepdim=True)
        # x_kv: (B, N, C)
        x_kv = x
        pos_q, pos_k = 0, 0
        # x: (B,1,C) -> (B,1,C)
        x = super().forward(x_q, x_kv, pos_q, pos_k,
                            bool_masked_pos=None,
                            rel_pos_bias=None)
        # x: (B, 1, C) -> (B, C)
        x = x.squeeze(1)
        return x
    

#-------------------------------------------------------------------------------------------- 


class StemLayer(nn.Module):
    """ Stem layer of InternImage.
    
    Args:
        in_chans (int): number of input channels.
        out_chans (int): number of output channels.
        act_layer (str): activation layer.
        norm_layer (str): normalization layer.
    """

    def __init__(self,
                 in_chans=3,
                 out_chans=96,
                 act_layer='GELU',
                 norm_layer='BN'):
        super().__init__()
        self.conv1 = nn.Conv3d(in_chans,
                               out_chans // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm1 = get_norm(norm=norm_layer, out_channels=out_chans // 2, channel_dim=1)
        self.act = get_activation(act_layer)
        self.conv2 = nn.Conv3d(out_chans // 2,
                               out_chans,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm2 = get_norm(out_channels=out_chans, norm=norm_layer, channel_dim=1)

    def forward(self, x):
        # stride=2, padding=1, kernel_size=3
        x = self.conv1(x)
        # BN
        x = self.norm1(x)
        # GELU
        x = self.act(x)
        # stride=2, padding=1, kernel_size=3
        x = self.conv2(x)
        # BN
        x = self.norm2(x)
        return x


class DownsampleLayer(nn.Module):
    """ Downsample layer of InternImage.

    Args:
        channels (int): number of input channels
        norm_layer (str): normalization layer
    """

    def __init__(self, channels, norm_layer='LN'):
        super().__init__()
        self.conv = nn.Conv3d(channels,
                              2 * channels,
                              kernel_size=3,
                              stride=2,
                              padding=1,
                              bias=False)
        self.norm = get_norm(norm=norm_layer, out_channels=2 * channels)

    def forward(self, x):
        # conv: kernel_size=3, stride=2, padding=1
        # x: (B, D, H, W, C) -> (B, C, D, H, W)
        x = self.conv(x.permute(0, 4, 1, 2, 3))
        # LayerNormalization 
        # x: (B, C, D, H, W) -> (B, D, H, W, C)
        x = self.norm(x.permute(0, 2, 3, 4, 1))
        return x


class MLPLayer(nn.Module):
    """ MLP layer of InternImage.

    Args:
        in_features (int): number of input features
        hidden_features (int): number of hidden features
        out_features (int): number of output features
        act_layer (str): activation layer
        drop (float): dropout rate
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer='GELU',
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = get_activation(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # in_feature -> hidden_feature
        x = self.fc1(x)
        # GELU
        x = self.act(x)
        x = self.drop(x)
        # hidden_feature -> out_feature
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InternImageLayer(nn.Module):
    """ Basic layer of InternImage.
    
    Args:
        channels (int): number of input channels.
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels.
        drop (float): dropout rate.
        drop_path (float): drop path rate
        act_layer (str): activation layer
        norm_layer (str): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """

    def __init__(self,
                 channels,
                 groups,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=False,
                 layer_scale=None,
                 offset_scale=1.0,
                 with_cp=False,
                 dw_kernel_size=None, # for InternImage-H/G
                 res_post_norm=False, # for InternImage-H/G
                 center_feature_scale=False,
                 remove_center=False,
                 output_bias=True,
                 without_pointwise=False,
                 use_dcn_v4_op=False): # for InternImage-H/G
        super().__init__()
        
        # define number of groups for DCN operation,
        # mlp_ratio for MLP hidden features, 
        # optional activation checkpointing   
        self.channels = channels
        self.groups = groups
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp

        # define modules:
        # 1. Layer Normalization
        # 2. dcn (deformable convolution attenion operation)
        # 3. drop path
        # 4. Layer Normalization
        # 5. MLP
        # 6. layer scale learable scaling parameter
        # 7. Layer Normalization 

        self.norm1 = get_norm(norm=norm_layer, out_channels=channels)
        self.post_norm = post_norm
        
        self.dcn = DCN(
            channels=channels,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=groups,
            offset_scale=offset_scale,
            act_layer=act_layer,
            norm_layer=norm_layer,
            dw_kernel_size=dw_kernel_size, # for InternImage-H/G
            center_feature_scale=center_feature_scale,
            remove_center=remove_center,
            output_bias=output_bias,
            without_pointwise=without_pointwise,
            use_dcn_v4_op=use_dcn_v4_op) # for InternImage-H/G
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = get_norm(norm="LN", out_channels=channels)
        self.mlp = MLPLayer(in_features=channels,
                            hidden_features=int(channels * mlp_ratio),
                            act_layer=act_layer,
                            drop=drop)
        
        self.layer_scale = layer_scale is not None
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
        
        self.res_post_norm = res_post_norm
        if res_post_norm:
            self.res_post_norm1 = get_norm(norm="LN", out_channels=channels)
            self.res_post_norm2 = get_norm(norm="LN", out_channels=channels)

    def forward(self, x):

        def _inner_forward(x):

            # 1_v1: dcn -> LN -> drop_path + res_path
            # 2_v1: MLP -> LN -> drop_path + res_path
            
            # 1_v2: LN -> dcn -> LN -> drop_path + res_path
            # 2_v2: LN -> MLP -> LN -> drop_path + res_path
            
            # 1_v3: LN -> dcn -> drop_path + res_path
            # 2_v3: LN -> MLP -> drop_path + res_path
            
            # 1_v4: dcn -> LN -> scale gamma -> drop_path + res_path
            # 2_v4: MLP -> LN -> scale gamma -> drop_path + res_path
            
            # 1_v5: LN -> dcn -> scale gamma -> drop_path + res_path
            # 2_v5: LN -> MLP -> scale gamma -> drop_path + res_path
            
            if not self.layer_scale:
                if self.post_norm:
                    x = x + self.drop_path(self.norm1(self.dcn(x)))
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
                elif self.res_post_norm: # for InternImage-H/G
                    x = x + self.drop_path(self.res_post_norm1(self.dcn(self.norm1(x))))
                    x = x + self.drop_path(self.res_post_norm2(self.mlp(self.norm2(x))))
                else:
                    x = x + self.drop_path(self.dcn(self.norm1(x)))
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
                return x
            if self.post_norm:
                x = x + self.drop_path(self.gamma1 * self.norm1(self.dcn(x)))
                x = x + self.drop_path(self.gamma2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.gamma1 * self.dcn(self.norm1(x)))
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            return x

        if self.with_cp and x.requires_grad:
            x = checkpoint.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class InternImageBlock(nn.Module):
    """ InternImage Block.
    
    Args:
        channels (int): number of input channels
        depths (list): Depth of each block.
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_layer (str): activation layer
        norm_layer (str): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """

    def __init__(self,
                 channels,
                 depth,
                 groups,
                 downsample=True,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=False,
                 offset_scale=1.0,
                 layer_scale=None,
                 with_cp=False,
                 dw_kernel_size=None, # for InternImage-H/G
                 post_norm_block_ids=None, # for InternImage-H/G
                 res_post_norm=False, # for InternImage-H/G
                 center_feature_scale=False, # for InternImage-H/G
                 remove_center=False,
                 output_bias=True,
                 without_pointwise=False,
                 use_dcn_v4_op=False):
        super().__init__()

        self.depth = depth
        self.channels = channels
        self.post_norm = post_norm
        self.center_feature_scale = center_feature_scale

        # stack depth number of InternImageLayer
        self.blocks = nn.ModuleList([
            InternImageLayer(
                channels=channels,
                groups=groups,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                layer_scale=layer_scale,
                offset_scale=offset_scale,
                with_cp=with_cp,
                dw_kernel_size=dw_kernel_size, # for InternImage-H/G
                res_post_norm=res_post_norm, # for InternImage-H/G
                center_feature_scale=center_feature_scale, # for InternImage-H/G
                remove_center=remove_center,
                output_bias=output_bias,
                without_pointwise=without_pointwise,
                use_dcn_v4_op=use_dcn_v4_op
            ) for i in range(depth)
        ])
        
        if not self.post_norm or center_feature_scale:
            self.norm = get_norm(norm="LN", out_channels=channels)
        self.post_norm_block_ids = post_norm_block_ids
        if post_norm_block_ids is not None: # for InternImage-H/G
            self.post_norms = nn.ModuleList(
                [get_norm(norm="LN", out_channels=channels) for _ in post_norm_block_ids]
            )
        
        self.downsample = DownsampleLayer(
            channels=channels, norm_layer=norm_layer) if downsample else None

    def forward(self, x, return_wo_downsample=False):
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (self.post_norm_block_ids is not None) and (i in self.post_norm_block_ids):
                index = self.post_norm_block_ids.index(i)
                x = self.post_norms[index](x) # for InternImage-H/G
        
        if not self.post_norm or self.center_feature_scale:
            x = self.norm(x)
        
        if return_wo_downsample:
            x_ = x
        
        # optionally downsample feature map
        # after all blocks
        if self.downsample is not None:
            x = self.downsample(x)

        if return_wo_downsample:
            return x, x_
        
        return x


class InternImage(nn.Module):
    """ Implementation of:
        `InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions`
        https://arxiv.org/pdf/2103.14030
    
    Args:
        channels (int): Number of the first stage. Default: 64
        depths (list): Depth of each block. Default: [3, 4, 18, 5]
        groups (list): Groups of each block. Default: [3, 6, 12, 24]
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        drop_rate (float): Probability of an element to be zeroed. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        act_layer (str): Activation layer. Default: 'GELU'
        norm_layer (str): Normalization layer. Default: 'LN'
        layer_scale (bool): Whether to use layer scale. Default: False
        cls_scale (bool): Whether to use class scale. Default: False
        with_cp (bool): Use checkpoint or not. 
        dw_kernel_size (int): Size of the dwconv. Default: None
        level2_post_norm (bool): Whether to use level2 post norm. Default: False
        level2_post_norm_block_ids (list): Indexes of post norm blocks. Default: None
        res_post_norm (bool): Whether to use res post norm. Default: False
        center_feature_scale (bool): Whether to use center feature scale. Default: False
    """

    def __init__(self,
                 return_intermediates=True,
                 in_channels=3,
                 channels=64,
                 depths=[3, 4, 18, 5],
                 groups=[3, 6, 12, 24],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 drop_path_type='linear',
                 act_layer='GELU',
                 norm_layer='LN',
                 layer_scale=None,
                 offset_scale=1.0,
                 post_norm=False,
                 with_cp=False,
                 dw_kernel_size=None,  # for InternImage-H/G
                 level2_post_norm=False,  # for InternImage-H/G
                 level2_post_norm_block_ids=None,  # for InternImage-H/G
                 res_post_norm=False,  # for InternImage-H/G
                 center_feature_scale=False,  # for InternImage-H/G
                 use_dcn_v4_op=False,
                 remove_center=False,
                 output_bias=True,
                 without_pointwise=False,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 **kwargs):
        super().__init__()

        self.return_intermediates = return_intermediates
        self.num_levels = len(depths)
        
        self.depths = depths
        self.channels = channels
        self.num_features = int(channels * 2**(self.num_levels - 1))
        
        self.post_norm = post_norm
        self.mlp_ratio = mlp_ratio
        
        # indices for when to append intermediate 
        # features to output 
        self.out_indices = out_indices
        # whether to apply post normalization after blocks
        self.level2_post_norm_block_ids = level2_post_norm_block_ids
        # what if any stages to freeze during training
        self.frozen_stages = frozen_stages

        # adjust number of in_channels based on 
        # data modality
        self.in_channels = in_channels
        self.patch_embed = StemLayer(in_chans=self.in_channels,
                                     out_chans=channels,
                                     act_layer=act_layer,
                                     norm_layer=norm_layer)
        self.drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        if drop_path_type == 'uniform':
            for i in range(len(dpr)):
                dpr[i] = drop_path_rate

        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            post_norm_block_ids = level2_post_norm_block_ids if level2_post_norm and (
                i == 2) else None # for InternImage-H/G
            level = InternImageBlock(
                # increase channels by a factor of 2
                # at each level to match downsampling
                channels=int(channels * 2**i),
                # depth and group size of 
                # each level
                depth=depths[i],
                groups=groups[i],
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                # drop path rate adjusted over range of depths
                # of block with increase form 0 to drop_path_rate
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                # downsample after all blocks
                downsample=(i < self.num_levels - 1),
                layer_scale=layer_scale,
                offset_scale=offset_scale,
                with_cp=with_cp,
                dw_kernel_size=dw_kernel_size,  # for InternImage-H/G
                post_norm_block_ids=post_norm_block_ids, # for InternImage-H/G
                res_post_norm=res_post_norm, # for InternImage-H/G
                center_feature_scale=center_feature_scale, # for InternImage-H/G
                remove_center=remove_center,
                output_bias=output_bias,
                without_pointwise=without_pointwise,
                use_dcn_v4_op=use_dcn_v4_op,
            )
            self.levels.append(level)

        self.num_layers = len(depths)
        self.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for level in self.levels[:self.frozen_stages]:
                level.eval()
                for param in level.parameters():
                    param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_deform_weights(self, m):
        # TODO: implement DCNv3 and add support here
        if isinstance(m, DCNv4):
            m._reset_parameters()

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.drop(x)

        x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, C, D, H, W) -> (B, D, H, W, C)

        seq_out = []
        for level_idx, level in enumerate(self.levels):
            x, x_ = level(x, return_wo_downsample=True)
            if level_idx in self.out_indices:
                seq_out.append(x_.permute(0, 4, 1, 2, 3).contiguous())
        
        return x, {f"p{s}": feature for s, feature in enumerate(seq_out)} if self.return_intermediates else x 