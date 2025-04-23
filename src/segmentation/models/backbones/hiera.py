"""
https://github.com/facebookresearch/hiera/blob/main/hiera/hiera.py

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
from functools import partial
from typing import Tuple, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, Mlp

from segmentation.models.utils.hf_hub import PyTorchModelHubMixin
from segmentation.models.backbones.batch_norm import get_norm
from segmentation.models.backbones.backbone_utils import conv_nd, do_pool, do_masked_conv, Unroll, Reroll


class MaskUnitAttention(nn.Module):
    """
    Computes either Mask Unit or Global Attention. Also is able to perform q pooling.

    Note: this assumes the tokens have already been flattened and unrolled into mask units.
    See `Unroll` for more details.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        q_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attn: bool = False,
    ):
        """
        Args:
            dim, dim_out: The input and output feature dimensions.
            heads: The number of attention heads.
            q_stride: If greater than 1, pool q with this stride. The stride should be flattened (e.g., 2x2 = 4).
            window_size: The current (flattened) size of a mask unit *after* pooling (if any).
            use_mask_unit_attn: Use Mask Unit or Global Attention.
        """
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.heads = heads

        # core idea: group image into windows
        # within each window, downsample the query sequence
        # to build a hierarchy of feature levels  
        self.q_stride = q_stride

        self.head_dim = dim_out // heads
        self.scale = (self.head_dim) ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim_out)
        self.proj = nn.Linear(dim_out, dim_out)

        self.window_size = window_size
        self.use_mask_unit_attn = use_mask_unit_attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Input should be of shape [batch, tokens, channels]. """
        B, N, _ = x.shape
        # window_size defines how many tokens per window
        num_windows = (
            (N // (self.q_stride * self.window_size)) if self.use_mask_unit_attn else 1
        )

        qkv = (
            self.qkv(x)
            .reshape(B, -1, num_windows, 3, self.heads, self.head_dim)
            .permute(3, 0, 4, 2, 1, 5)
        ) # (3, B, heads, num_windows, tokens_per_window, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.q_stride > 1:
            # refer to Unroll to see how this performs a maxpool-Nd
            # tokens_per_window -> q_stride * G , i.e. reshaped into shape (q_stride, G)
            # thus max-pooling over the q_stride dimension to get resulting G queries
            # for the window
            q = (
                q.view(B, self.heads, num_windows, self.q_stride, -1, self.head_dim)
                .max(dim=3)
                .values
            )

        if hasattr(F, "scaled_dot_product_attention"):
            # Note: the original paper did *not* use SDPA, it's a free boost!
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = (q * self.scale) @ k.transpose(-1, -2)
            attn = attn.softmax(dim=-1)
            x = (attn @ v)

        x = x.transpose(1, 3).reshape(B, -1, self.dim_out) # ()
        x = self.proj(x)
        return x


class HieraBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        q_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attn: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out

        self.norm1 = norm_layer(dim)
        self.attn = MaskUnitAttention(
            dim, dim_out, heads, q_stride, window_size, use_mask_unit_attn
        )

        self.norm2 = norm_layer(dim_out)
        self.mlp = Mlp(dim_out, int(dim_out * mlp_ratio), act_layer=act_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention + Q Pooling
        x_norm = self.norm1(x)
        if self.dim != self.dim_out:
            # recall: do_pool op => x.view(x.shape[0], stride, -1, x.shape[-1]).max(dim=1).values
            # sequence becomes (B, N//stride, dim_out) since we group into G groups of stride
            # tokens and take the max of each group
            x = do_pool(self.proj(x_norm), stride=self.attn.q_stride)
        x = x + self.drop_path(self.attn(x_norm))

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Head(nn.Module):
    def __init__(
        self,
        dim: int,
        num_classes: int,
        dropout_rate: float = 0.0,
        act_func: Callable[[torch.Tensor], torch.Tensor] = lambda x: x.softmax(dim=-1),
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.projection = nn.Linear(dim, num_classes)
        # act_fun for eval and testing only
        self.act_func = act_func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.projection(x)
        if not self.training:
            x = self.act_func(x)
        return x


class PatchEmbed(nn.Module):
    """Patch embed that supports any number of spatial dimensions (1d, 2d, 3d)."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Tuple[int, ...],
    ):
        super().__init__()

        # support any number of spatial dimensions
        self.spatial_dims = len(kernel)
        self.proj = conv_nd(self.spatial_dims)(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # ndConv with x partially zeroed out by mask, resized to x shape
        # prevents information leakage
        x = do_masked_conv(x, self.proj, mask)
        # (B,C,D,H,W) -> (B, C, D*H*W) -> (B, D*H*W, C)
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(2, 1)
        return x


class Hiera(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        input_size: Tuple[int, ...] = (224, 224, 224),
        in_chans: int = 3,
        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        num_classes: int = 1000,
        stages: Tuple[int, ...] = (2, 3, 16, 3),
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, ...] = (2, 2, 2),
        mask_unit_size: Tuple[int, ...] = (8, 8, 8),  # must divide q_stride ** (#stages-1)
        # mask_unit_attn: which stages use mask unit attention?
        mask_unit_attn: Tuple[bool, ...] = (True, True, False, False),
        dim_mul: float = 2.0,
        head_mul: float = 2.0,
        patch_kernel: Tuple[int, ...] = (7, 7, 7), # for conv patch_embed
        patch_stride: Tuple[int, ...] = (4, 4, 4), 
        patch_padding: Tuple[int, ...] = (3, 3, 3),
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        norm_layer: Union[str, nn.Module] = "LN",
        head_dropout: float = 0.0,
        head_init_scale: float = 0.001,
        sep_pos_embed: bool = False,
        return_intermediates: bool = False,
    ):
        super().__init__()

        self.return_intermediates = return_intermediates

        # do it this way to ensure that the init args are all PoD (for config usage)
        if isinstance(norm_layer, str):
            norm_layer = get_norm(norm_layer, channel_dim=-1, partial_init=True)

        # total number of blocks across all stages 
        depth = sum(stages)
        self.patch_stride = patch_stride
        
        # (D,H,W) after 3D patch‑embedding conv
        self.tokens_spatial_shape = [i // s for i, s in zip(input_size, patch_stride)]
        
        # total number of tokens after 3D patch embedding, total token num. for one 
        # mask, and q_stride
        num_tokens = math.prod(self.tokens_spatial_shape)
        flat_mu_size = math.prod(mask_unit_size)
        flat_q_stride = math.prod(q_stride)

        assert q_pool < len(stages)
        self.q_pool, self.q_stride = q_pool, q_stride
        self.mu_size, self.mask_unit_size = flat_mu_size, mask_unit_size
        # size of grid in masks (mask units)
        self.mask_spatial_shape = [
            i // s for i, s in zip(self.tokens_spatial_shape, self.mask_unit_size)
        ]
        # where each stage ends (ex. 1, 4, 20) for stages [2,3,16]
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]

        # module for embedding pixel space image into patches with 3d conv
        # usually 4x downsample (conv7x7x7, s=4x4x4, p=3x3x3)
        # flattened to sequence: [B, L=64x64x64=262 144,C=96]
        self.patch_embed = PatchEmbed(
            in_chans, embed_dim, patch_kernel, patch_stride, patch_padding
        )

        # separate pos. embeddings by axis flag (NOTE: not currently suppoorted!)
        self.sep_pos_embed = sep_pos_embed
        if sep_pos_embed:
            raise NotImplementedError("Separate positional embeddings not supported yet.")
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(
                    1,
                    self.tokens_spatial_shape[1] * self.tokens_spatial_shape[2],
                    embed_dim,
                )
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.tokens_spatial_shape[0], embed_dim)
            )
        else:
            # table for positional embeddings
            self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))

        # setup roll and reroll modules
        self.unroll = Unroll(
            input_size, patch_stride, [q_stride] * len(self.stage_ends[:-1])
        )
        self.reroll = Reroll(
            input_size,
            patch_stride,
            [q_stride] * len(self.stage_ends[:-1]),
            self.stage_ends,
            q_pool,
        )
        # q_pool locations (take first q_pool ends and add 1 => apply q-pool after stage end)
        q_pool_blocks = [x + 1 for x in self.stage_ends[:q_pool]]
        
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # transformer blocks
        cur_stage = 0
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            # mask unit or global attention
            # lag by 1 block, so that global attention,
            # applied post pooling on lower resolution
            use_mask_unit_attn = mask_unit_attn[cur_stage]

            # stage_end -> stage_end + 1 = inc. dim_out, num_heads
            # and if we do q_pool (may limit q_pool to first q_pool stages)
            # then apply q_stride max pooling and decrease spatial (token) dims
            # (B, C, D, H, W) -> (B, C, D//q_stride, H//q_stride, W//q_stride)
            # we adapt mask unit size accordingly
            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1
                if i in q_pool_blocks:
                    flat_mu_size //= flat_q_stride

            block = HieraBlock(
                dim=embed_dim,
                dim_out=dim_out,
                heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                q_stride=(flat_q_stride if i in q_pool_blocks else 1),
                window_size=flat_mu_size,
                use_mask_unit_attn=use_mask_unit_attn,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.norm = norm_layer(embed_dim)
        self.head = Head(embed_dim, num_classes, dropout_rate=head_dropout)

        # initialize everything
        if self.sep_pos_embed:
            raise NotImplementedError("Separate positional embeddings not supported yet.")
            nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)
        else:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(partial(self._init_weights))
        self.head.projection.weight.data.mul_(head_init_scale)
        self.head.projection.bias.data.mul_(head_init_scale)

    def _init_weights(self, m, init_bias=0.02):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, init_bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, init_bias)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.sep_pos_embed:
            raise NotImplementedError("Separate positional embeddings not supported yet.")
            return ["pos_embed_spatial", "pos_embed_temporal"]
        else:
            return ["pos_embed"]

    def get_random_mask(self, x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """
        Generates a random mask, mask_ratio fraction are dropped.
        1 is *keep*, 0 is *remove*. Useful for MAE, FLIP, etc.
        """
        B = x.shape[0]
        # tokens selected for masking at mask unit level
        num_windows = math.prod(self.mask_spatial_shape)  # num_mask_units
        len_keep = int(num_windows * (1 - mask_ratio))
        noise = torch.rand(B, num_windows, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 1 is *keep*, 0 is *remove*
        # note this is opposite to original MAE
        mask = torch.zeros([B, num_windows], device=x.device)
        mask[:, :len_keep] = 1
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask.bool()

    def get_pos_embed(self) -> torch.Tensor:
        if self.sep_pos_embed:
            raise NotImplementedError("Separate positional embeddings not supported yet.")
            return self.pos_embed_spatial.repeat(
                1, self.tokens_spatial_shape[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.tokens_spatial_shape[1] * self.tokens_spatial_shape[2],
                dim=1,
            )
        else:
            return self.pos_embed

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        mask should be a boolean tensor of shape [B, #MUt*#MUy*#MUx] where #MU are the number of mask units in that dim.
        Note: 1 in mask is *keep*, 0 is *remove*; mask.sum(dim=-1) should be the same across the batch.
        """
        # Slowfast training passes in a list
        if isinstance(x, list):
            x = x[0]
        intermediates = []

        x = self.patch_embed(
            x,
            mask=mask.view(
                x.shape[0], 1, *self.mask_spatial_shape
            )  # B, C, *mask_spatial_shape
            if mask is not None
            else None,
        )
        x = x + self.get_pos_embed()
        x = self.unroll(x)

        # Discard masked tokens 
        # NOTE: mask is per mask_unit, hence tile across mu_size and c
        if mask is not None:
            x = x[mask[..., None].tile(1, self.mu_size, x.shape[2])].view(
                x.shape[0], -1, x.shape[-1]
            )

        for i, blk in enumerate(self.blocks):
            x = blk(x)

            # we return intermediates at stage_ends but q-pool
            # happens at the end of the stage => intermediate spatial dims
            # lag by 1 in stride
            if self.return_intermediates and i in self.stage_ends:
                # NOTE: x may not always be in spatial order here.
                #       e.g. if q_pool = 2, mask_unit_size = (8, 8), and
                #       q_stride = (2, 2), not all unrolls were consumed,
                #       intermediates[-1] is x in spatial order
                #       hence reroll to get the original spatial order
                intermediates.append(self.reroll(x, i, mask=mask))

        if self.return_intermediates:
            # reshape to (B,C,D,H,W)
            intermediates = [intermediate.permute(0,4,1,2,3) for intermediate in intermediates]

        # NOTE: linear probing layer removed
        # if mask is None:
        #     x = x.mean(dim=1)
        #     x = self.norm(x)
        #     x = self.head(x) # classification head

        return x, {f"p{s}": feature for s, feature in enumerate(intermediates)} if self.return_intermediates else x 


# TODO: add support for loading weights from pretrained models 
#       for all backbones 

# Image models


# @pretrained_model({
#     "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_tiny_224.pth",
#     "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_tiny_224.pth",
# }, default="mae_in1k_ft_in1k")
# def hiera_tiny_224(**kwdargs):
#     return Hiera(embed_dim=96, num_heads=1, stages=(1, 2, 7, 2), **kwdargs)


# @pretrained_model({
#     "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_small_224.pth",
#     "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_small_224.pth",
# }, default="mae_in1k_ft_in1k")
# def hiera_small_224(**kwdargs):
#     return Hiera(embed_dim=96, num_heads=1, stages=(1, 2, 11, 2), **kwdargs)


# @pretrained_model({
#     "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_base_224.pth",
#     "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_224.pth",
# }, default="mae_in1k_ft_in1k")
# def hiera_base_224(**kwdargs):
#     return Hiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3), **kwdargs)


# @pretrained_model({
#     "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_base_plus_224.pth",
#     "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_plus_224.pth",
# }, default="mae_in1k_ft_in1k")
# def hiera_base_plus_224(**kwdargs):
#     return Hiera(embed_dim=112, num_heads=2, stages=(2, 3, 16, 3), **kwdargs)


# @pretrained_model({
#     "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_large_224.pth",
#     "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_large_224.pth",
# }, default="mae_in1k_ft_in1k")
# def hiera_large_224(**kwdargs):
#     return Hiera(embed_dim=144, num_heads=2, stages=(2, 6, 36, 4), **kwdargs)


# @pretrained_model({
#     "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_huge_224.pth",
#     "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_huge_224.pth",
# }, default="mae_in1k_ft_in1k")
# def hiera_huge_224(**kwdargs):
#     return Hiera(embed_dim=256, num_heads=4, stages=(2, 6, 36, 4), **kwdargs)


# # Video models


# @pretrained_model({
#     "mae_k400_ft_k400": "https://dl.fbaipublicfiles.com/hiera/hiera_base_16x224.pth",
#     "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_16x224.pth",
# }, default="mae_k400_ft_k400")
# def hiera_base_16x224(num_classes: int = 400, **kwdargs):
#     return Hiera(
#         num_classes=num_classes,  # K400 has 400 classes
#         input_size=(16, 224, 224),
#         q_stride=(1, 2, 2),
#         mask_unit_size=(1, 8, 8),
#         patch_kernel=(3, 7, 7),
#         patch_stride=(2, 4, 4),
#         patch_padding=(1, 3, 3),
#         sep_pos_embed=True,
#         **kwdargs
#     )


# @pretrained_model({
#     "mae_k400_ft_k400": "https://dl.fbaipublicfiles.com/hiera/hiera_base_plus_16x224.pth",
#     "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_plus_16x224.pth",
# }, default="mae_k400_ft_k400")
# def hiera_base_plus_16x224(**kwdargs):
#     return hiera_base_16x224(
#         embed_dim=112, num_heads=2, stages=(2, 3, 16, 3), **kwdargs
#     )


# @pretrained_model({
#     "mae_k400_ft_k400": "https://dl.fbaipublicfiles.com/hiera/hiera_large_16x224.pth",
#     "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_large_16x224.pth",
# }, default="mae_k400_ft_k400")
# def hiera_large_16x224(**kwdargs):
#     return hiera_base_16x224(
#         embed_dim=144, num_heads=2, stages=(2, 6, 36, 4), **kwdargs
#     )


# @pretrained_model({
#     "mae_k400_ft_k400": "https://dl.fbaipublicfiles.com/hiera/hiera_huge_16x224.pth",
#     "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_huge_16x224.pth",
# }, default="mae_k400_ft_k400")
# def hiera_huge_16x224(**kwdargs):
#     return hiera_base_16x224(
#         embed_dim=256, num_heads=4, stages=(2, 6, 36, 4), **kwdargs
#     )