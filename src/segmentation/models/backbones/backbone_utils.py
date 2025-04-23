"""
https://github.com/facebookresearch/detectron2/blob/9604f5995cc628619f0e4fd913453b4d7d61db3f/detectron2/modeling/backbone/utils.py
https://github.com/facebookresearch/hiera/blob/main/hiera/hiera_utils.py

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
from typing import Dict, List, Tuple, Callable, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation.models.backbones.batch_norm import FrozenBatchNorm3d


# TODO: merge/deduplicate utility functions from different backbones (kept separated for now)

########################################################################### ViTDet ###########################################################################


def window_partition(x: torch.Tensor, window_size: int):
    """
    Partition input tensor into non-overlapping windows, with padding if needed.

    Args:
        x (tensor): input tokens with [B, D, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, window_size, C].
        (Dp, Hp, Wp): padded depth, height, and width before partition
    """
    # TODO: support nD
    B, D, H, W, C = x.shape

    pad_d = (window_size - D % window_size) % window_size
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_d or pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_d, 0, pad_w, 0, pad_h))
    Dp, Hp, Wp = D + pad_d, H + pad_h, W + pad_w

    x = x.view(B, Dp // window_size, window_size, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    return windows, (Dp, Hp, Wp)


def window_unpartition(windows: torch.Tensor, window_size: int, pad_dhw: Tuple[int, int, int], dhw: Tuple[int, int, int]):
    """
    Window unpartition into original sequences and removing padding.

    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, window_size, C].
        window_size (int): window size.
        pad_dhw (Tuple): padded depth, height and width (Dp, Hp, Wp).
        dhw (Tuple): original depth, height and width (D, H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, D, H, W, C].
    """
    # TODO: support nD
    Dp, Hp, Wp = pad_dhw
    D, H, W = dhw
    
    B = windows.shape[0] // (Dp * Hp * Wp // window_size // window_size // window_size)
    x = windows.view(B, Dp // window_size, Hp // window_size, Wp // window_size, window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, Dp, Hp, Wp, -1)

    if Dp > D or Hp > H or Wp > W:
        x = x[:, :D, :H, :W, :].contiguous()
    return x


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


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self, 
        kernel_size: Tuple[int, int, int] = (16, 16, 16), 
        stride: Tuple[int, int, int] = (16, 16, 16), 
        padding: Tuple[int, int, int] = (0, 0, 0), 
        in_chans: int = 3, embed_dim: int = 768
    ):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x):
        x = self.proj(x)
        # B C D H W -> B D H W C
        x = x.permute(0, 2, 3, 4, 1)
        return x
    

class CNNBlockBase(nn.Module):
    """
    A CNN block is assumed to have input channels, output channels and a stride.
    The input and output of `forward()` method must be NCDHW tensors.
    The method can perform arbitrary computation but must match the given
    channels and stride specification.

    Attribute:
        in_channels (int):
        out_channels (int):
        stride (int):
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            in_channels (int):
            out_channels (int):
            stride (int):
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        """
        Make this block not trainable.
        This method sets all parameters to `requires_grad=False`,
        and convert all BatchNorm layers to FrozenBatchNorm

        Returns:
            the block itself
        """
        for p in self.parameters():
            p.requires_grad = False
        # see norm.py for details
        FrozenBatchNorm3d.convert_frozen_batchnorm(self)
        return self

    
def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )


############################################################### HIERA ######################################################################################


# TODO: add support for loading pretrained weights accross all backbones
# def pretrained_model(checkpoints: Dict[str, str], default: str = None) -> Callable:
#     """ Loads a Hiera model from a pretrained source (if pretrained=True). Use "checkpoint" to specify the checkpoint. """

#     def inner(model_func: Callable) -> Callable:
#         def model_def(pretrained: bool = False, checkpoint: str = default, strict: bool = True, **kwdargs) -> nn.Module:
#             if pretrained:
#                 if checkpoints is None:
#                     raise RuntimeError("This model currently doesn't have pretrained weights available.")
#                 elif checkpoint is None:
#                     raise RuntimeError("No checkpoint specified.")
#                 elif checkpoint not in checkpoints:
#                     raise RuntimeError(f"Invalid checkpoint specified ({checkpoint}). Options are: {list(checkpoints.keys())}.")

#                 state_dict = torch.hub.load_state_dict_from_url(checkpoints[checkpoint], map_location="cpu")
            
#                 if "head.projection.weight" in state_dict["model_state"]:
#                     # set the number of classes equal to the state_dict only if the user doesn't want to overwrite it
#                     if "num_classes" not in kwdargs:
#                         kwdargs["num_classes"] = state_dict["model_state"]["head.projection.weight"].shape[0]
#                     # if the user specified a different number of classes, remove the projection weights or else we'll error out
#                     elif kwdargs["num_classes"] != state_dict["model_state"]["head.projection.weight"].shape[0]:
#                         del state_dict["model_state"]["head.projection.weight"]
#                         del state_dict["model_state"]["head.projection.bias"]

#             model = model_func(**kwdargs)
#             if pretrained:
#                 # disable being strict when trying to load a encoder-decoder model into an encoder-only model
#                 if "decoder_pos_embed" in state_dict["model_state"] and not hasattr(model, "decoder_pos_embed"):
#                     strict = False

#                 model.load_state_dict(state_dict["model_state"], strict=strict)
            
#             return model

#         # keep some metadata so we can do things that require looping through all available models
#         model_def.checkpoints = checkpoints
#         model_def.default = default

#         return model_def
    
#     return inner


def conv_nd(n: int) -> Type[nn.Module]:
    """
    Returns a conv with nd (e.g., Conv2d for n=2). Work up to n=3.
    For a 4d Hiera, we could probably just implement this for n=4.
    """
    return [nn.Identity, nn.Conv1d, nn.Conv2d, nn.Conv3d][n]


def do_pool(x: torch.Tensor, stride: int) -> torch.Tensor:
    # refer to `Unroll` to see how this performs a maxpool-Nd
    return x.view(x.shape[0], stride, -1, x.shape[-1]).max(dim=1).values


def get_resized_mask(target_size: torch.Size, mask: torch.Tensor) -> torch.Tensor:
    # target_size: [(D), (H), W]
    # (spatial) mask: [B, C, (d), (h), w]
    if mask is None:
        return mask

    assert len(mask.shape[2:]) == len(target_size)
    if mask.shape[2:] != target_size:
        return F.interpolate(mask.float(), size=target_size)
    return mask


def do_masked_conv(
    x: torch.Tensor, conv: nn.Module, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Zero-out the masked regions of the input before conv.
    Prevents leakage of masked regions when using overlapping kernels.
    """
    if conv is None:
        return x
    if mask is None:
        return conv(x)

    # interpolates mask size s.t. matches x.shape[2:]
    # mask out regions before conv
    mask = get_resized_mask(target_size=x.shape[2:], mask=mask)
    return conv(x * mask.bool())


def undo_windowing(
    x: torch.Tensor, shape: List[int], mu_shape: List[int]
) -> torch.Tensor:
    """
    Restore spatial organization by undoing windowed organization of mask units.

    Args:
        x: organized by mask units windows, e.g. in 2d [B, #MUy*#MUx, MUy, MUx, C]
        shape: current spatial shape, if it were not organized into mask unit
            windows, e.g. in 2d [B, #MUy*MUy, #MUx*MUx, C].
        mu_shape: current mask unit shape, e.g. in 2d [MUy, MUx]
    Returns:
        x: e.g. in 2d, [B, #MUy*MUy, #MUx*MUx, C]
    """
    D = len(shape)
    B, C = x.shape[0], x.shape[-1]
    # [B, #MUy*#MUx, MUy, MUx, C] -> [B, #MUy, #MUx, MUy, MUx, C]
    num_MUs = [s // mu for s, mu in zip(shape, mu_shape)]
    x = x.view(B, *num_MUs, *mu_shape, C)

    # [B, #MUy, #MUx, MUy, MUx, C] -> [B, #MUy*MUy, #MUx*MUx, C]
    permute = (
        [0]
        + sum(
            [list(p) for p in zip(range(1, 1 + D), range(1 + D, 1 + 2 * D))],
            [],
        )
        + [len(x.shape) - 1]
    )
    x = x.permute(permute).reshape(B, *shape, C)

    return x


class Unroll(nn.Module):
    """
    Reorders the tokens such that patches are contiguous in memory.
    E.g., given [B, (H, W), C] and stride of (Sy, Sx), this will re-order the tokens as
                           [B, (Sy, Sx, H // Sy, W // Sx), C]

    This allows operations like Max2d to be computed as x.view(B, Sx*Sy, -1, C).max(dim=1).
    Not only is this faster, but it also makes it easy to support inputs of arbitrary
    dimensions in addition to patch-wise sparsity.

    Performing this operation multiple times in sequence puts entire windows as contiguous
    in memory. For instance, if you applied the stride (2, 2) 3 times, entire windows of
    size 8x8 would be contiguous in memory, allowing operations like mask unit attention
    computed easily and efficiently, while also allowing max to be applied sequentially.

    Note: This means that intermediate values of the model are not in HxW order, so they
    need to be re-rolled if you want to use the intermediate values as a HxW feature map.
    The last block of the network is fine though, since by then the strides are all consumed.
    This is important for all encoder-decoder architectures that leverage FPNs/intermediate
    feature maps.
    """

    def __init__(
        self,
        input_size: Tuple[int, ...],
        patch_stride: Tuple[int, ...],
        unroll_schedule: List[Tuple[int, ...]],
    ):
        super().__init__()
        self.size = [i // s for i, s in zip(input_size, patch_stride)]
        self.schedule = unroll_schedule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: Flattened patch embeddings [B, N, C]
        Output: Patch embeddings [B, N, C] permuted such that [B, 4, N//4, C].max(1) etc. performs MaxPoolNd
        """
        B, _, C = x.shape

        cur_size = self.size
        x = x.view(*([B] + cur_size + [C]))

        for strides in self.schedule:
            # move patches with the given strides to the batch dimension

            # create a view of the tensor with the patch stride as separate dims
            # for example in 2d: [B, H // Sy, Sy, W // Sx, Sx, C]
            cur_size = [i // s for i, s in zip(cur_size, strides)]
            new_shape = [B] + sum([[i, s] for i, s in zip(cur_size, strides)], []) + [C]
            x = x.view(new_shape)

            # move the patch stride into the batch dimension
            # for example in 2d: [B, Sy, Sx, H // Sy, W // Sx, C]
            L = len(new_shape)
            permute = (
                [0] + list(range(2, L - 1, 2)) + list(range(1, L - 1, 2)) + [L - 1]
            )
            x = x.permute(permute)

            # now finally flatten the relevant dims into the batch dimension
            x = x.flatten(0, len(strides))
            B *= math.prod(strides)

        x = x.reshape(-1, math.prod(self.size), C)
        return x


class Reroll(nn.Module):
    """
    Undos the "unroll" operation so that you can use intermediate features.
    """

    def __init__(
        self,
        input_size: Tuple[int, ...],
        patch_stride: Tuple[int, ...],
        unroll_schedule: List[Tuple[int, ...]],
        stage_ends: List[int],
        q_pool: int,
    ):
        super().__init__()
        self.size = [i // s for i, s in zip(input_size, patch_stride)]

        # the first stage has to reverse everything
        # the next stage has to reverse all but the first unroll, etc.
        self.schedule = {}
        size = self.size
        for i in range(stage_ends[-1] + 1):
            self.schedule[i] = unroll_schedule, size
            # schedule unchanged if no pooling at a stage end
            if i in stage_ends[:q_pool]:
                if len(unroll_schedule) > 0:
                    size = [n // s for n, s in zip(size, unroll_schedule[0])]
                unroll_schedule = unroll_schedule[1:]

    def forward(
        self, x: torch.Tensor, block_idx: int, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Roll the given tensor back up to spatial order assuming it's from the given block.

        If no mask is provided:
            - Returns [B, H, W, C] for 2d, [B, D, H, W, C] for 3d, etc.
        If a mask is provided:
            - Returns [B, #MUs, MUy, MUx, C] for 2d, etc.
        """
        schedule, size = self.schedule[block_idx]
        B, N, C = x.shape

        D = len(size)
        cur_mu_shape = [1] * D

        for strides in schedule:
            # extract the current patch from N
            x = x.view(B, *strides, N // math.prod(strides), *cur_mu_shape, C)

            # move that patch into the current MU
            # example in 2d: [B, Sy, Sx, N//(Sy*Sx), MUy, MUx, C] -> [B, N//(Sy*Sx), Sy, MUy, Sx, MUx, C]
            L = len(x.shape)
            permute = (
                [0, 1 + D]
                + sum(
                    [list(p) for p in zip(range(1, 1 + D), range(1 + D + 1, L - 1))],
                    [],
                )
                + [L - 1]
            )
            x = x.permute(permute)

            # reshape to [B, N//(Sy*Sx), *MU, C]
            for i in range(D):
                cur_mu_shape[i] *= strides[i]
            x = x.reshape(B, -1, *cur_mu_shape, C)
            N = x.shape[1]

        # current shape (e.g., 2d: [B, #MUy*#MUx, MUy, MUx, C])
        x = x.view(B, N, *cur_mu_shape, C)

        # if masked, return [B, #MUs, MUy, MUx, C]
        if mask is not None:
            return x

        # if not masked, we can return [B, H, W, C]
        x = undo_windowing(x, size, cur_mu_shape)

        return x
    

############################################################################################################################################################################