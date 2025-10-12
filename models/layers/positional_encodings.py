"""
Adapted from:
https://github.com/IDEA-Research/MaskDINO/main/maskdino/modeling/pixel_decoder/position_encoding.py
"""

import math

import torch
from torch import nn


class PositionalEmbeddingSinCos(nn.Module):
    def __init__(self, num_pos_feats, temperature=10000, normalize=False, scale=None):
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
        F = int(self.num_pos_feats)
        Fe = F - (F % 2)
        
        # dim_t: (num_pos_feats,)
        dim_t = torch.arange(Fe, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / Fe)

        # x: (N, L, 3) or (N, L, 6)
        x_embed = x[:, :, 0] * self.scale
        y_embed = x[:, :, 1] * self.scale
        z_embed = x[:, :, 2] * self.scale
        
        # x: (N, L, num_pos_feats)
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_z = z_embed[:, :, None] / dim_t

        # (N,L,num_pos_feats/2,2) [sin, cos] pairs of positional theta values
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_z = torch.stack((pos_z[:, :, 0::2].sin(), pos_z[:, :, 1::2].cos()), dim=3).flatten(2)

        if x.size(-1) == 3:
            pos = torch.cat((pos_z, pos_y, pos_x), dim=2)
            remainder = 3 * F - 3 * Fe
            if remainder > 0:
                pos = torch.cat([pos, pos.new_zeros(pos.shape[0], pos.shape[1], remainder)], dim=2)
        
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

            remainder = 6 * F - 6 * Fe
            if remainder > 0:
                pos = torch.cat([pos, pos.new_zeros(pos.shape[0], pos.shape[1], remainder)], dim=2)

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

        F = int(self.num_pos_feats)
        Fe = F - (F % 2)

        dim_t = torch.arange(Fe, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / Fe)

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
        remainder = 3 * F - 3 * Fe
        if remainder > 0:
            pos = torch.cat([pos, pos.new_zeros(N, remainder, D, H, W)], dim=1)

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