"""
https://github.com/IDEA-Research/MaskDINO/blob/main/maskdino/modeling/pixel_decoder/position_encoding.py

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
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


# ------------------------- ------------------------- MASKDINO ------------------------- -------------------------


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()

        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def _forward_queries(self, x, shape, mask=None):
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        x_embed = x[:, :, 0] * self.scale
        y_embed = x[:, :, 1] * self.scale
        z_embed = x[:, :, 2] * self.scale
        
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_z = z_embed[:, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_z = torch.stack((pos_z[:, :, 0::2].sin(), pos_z[:, :, 1::2].cos()), dim=3).flatten(2)

        if x.size(-1) == 3:
            pos = torch.cat((pos_z, pos_y, pos_x), dim=2)
        elif x.size(-1) == 6:
            w_embed = x[:, :, 3] * self.scale
            pos_w = w_embed[:, :, None] / dim_t
            pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(-2)

            h_embed = x[:, :, 4] * self.scale
            pos_h = h_embed[:, :, None] / dim_t
            pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(-2)

            d_embed = x[:, :, 5] * self.scale
            pos_d = d_embed[:, :, None] / dim_t
            pos_d = torch.stack((pos_d[:, :, 0::2].sin(), pos_d[:, :, 1::2].cos()), dim=3).flatten(-2)

            pos = torch.cat((pos_z, pos_y, pos_x, pos_w, pos_h, pos_d), dim=2)
        else:
            raise ValueError("Unknown x shape(-1):{}".format(x.size(-1)))
        return pos
    
    def _forward_image(self, x, shape, mask=None):
        N, C, D, H, W = shape
        if mask is None:
            mask = torch.zeros((N, D, H, W), device=x.device, dtype=torch.bool)
        not_mask = ~mask

        # cumsum gives a valid position sequence 
        # even with padding 
        z_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            # gives count at last z/y/x position => normalizes to [0,1]
            z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # (N, D, H, W, num_pos_feats)
        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        pos_z = z_embed[:, :, :, :, None] / dim_t

        # (N,D,H,W,num_pos_feats/2,2) [sin, cos] pairs of positional theta values
        # for each (N, D, H, W) position -> (N, D, H, W, num_pos_feats)
        pos_x = torch.stack(
            (pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=4
        ).flatten(4)
        pos_y = torch.stack(
            (pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=4
        ).flatten(4)
        pos_z = torch.stack(
            (pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=4
        ).flatten(4)

        #  (N, D, H, W, 3*num_pos_feats) -> (N, 3*num_pos_feats, D, H, W)
        pos = torch.cat((pos_z, pos_y, pos_x), dim=4).permute(0, 4, 1, 2, 3)
        return pos

    def forward(self, x, mask=None):
        if x.dim() == 5:
            # x is a 5D tensor (N, C, D, H, W)
            shape = x.shape
            return self._forward_image(x, shape, mask)
        elif x.dim() == 3:
            # x is a 3D tensor (N, C, L)
            shape = x.shape
            return self._forward_queries(x, shape, mask)
        else:
            raise ValueError(f"Unsupported input tensor shape: {x.shape}. Expected 3D or 5D tensor.")


# ------------------------- ------------------------- ViTDet ------------------------- -------------------------


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor):
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.

    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # interpolate rel pos if needed
    # (L, C) -> (1, C, L) -> interpolate -> (max_rel_dist, C)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # scale the coords with short length if shapes for q and k are different
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    # broadcast gives (q_size, k_size) grid of relative positions, then shift
    # to ensure positive indexing 
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn: torch.Tensor, 
                           q: torch.Tensor, 
                           rel_pos_d: torch.Tensor, 
                           rel_pos_h: torch.Tensor, 
                           rel_pos_w: torch.Tensor, 
                           q_size: Tuple[int, int, int], 
                           k_size: Tuple[int, int, int]
):
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_d * q_h * q_w, C).
        rel_pos_d (Tensor): relative position embeddings (Ld, C) for depth axis.
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_d, q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_d, k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    # TODO: add tests
    q_d, q_h, q_w = q_size
    k_d, k_h, k_w = k_size
    Rd = get_rel_pos(q_d, k_d, rel_pos_d)
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_d, q_h, q_w, dim)
    rel_d = torch.einsum("bdhwc,dkc->bdhwk", r_q, Rd)
    rel_h = torch.einsum("bdhwc,hkc->bdhwk", r_q, Rh)
    rel_w = torch.einsum("bdhwc,wkc->bdhwk", r_q, Rw)

    attn = (
        attn.view(B, q_d, q_h, q_w, k_d, k_h, k_w) 
        + rel_d[:, :, :, :, :, None, None]
        + rel_h[:, :, :, :, None, :, None]
        + rel_w[:, :, :, :, None, None, :]
    ).view(B, q_d * q_h * q_w, k_d * k_h * k_w)

    return attn


def get_abs_pos(abs_pos: torch.Tensor, has_cls_token: bool, dhw: Tuple[int, int, int]):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.

    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        dhw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, D, H, W, C)
    """
    d, h, w = dhw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]

    num_positions = abs_pos.shape[1]
    size = round(math.pow(num_positions, 1 / 3))
    assert size * size * size == num_positions, f"size is {size}, but xyz_num is {num_positions}."

    # interpolate abs pos if image size is different
    # from pretraining image size
    if size != h or size != w or size != d:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, size, -1).permute(0, 4, 1, 2, 3), # (bs, c, z, y, x)
            size=(d, h, w),
            mode="trilinear",
            align_corners=False,
        )

        return new_abs_pos.permute(0, 2, 3, 4, 1)
    else:
        return abs_pos.reshape(1, d, h, w, -1)


# ------------------------- ------------------------- --- ---- --- ------------------------- -------------------------