"""
https://github.com/IDEA-Research/MaskDINO/blob/3831d8514a3728535ace8d4ecc7d28044c42dd14/maskdino/utils/utils.py#L36
https://github.com/facebookresearch/detectron2/blob/9604f5995cc628619f0e4fd913453b4d7d61db3f/detectron2/layers/wrappers.py#L103

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
from typing import Type, Optional, List, Tuple, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from segmentation.layers.utils import undo_windowing
from segmentation.layers.norms import FrozenBatchNorm3d
from segmentation.structures.data_objects.image_list import Shape

# ------------------------------------------------------------ MLP ------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


# -------------------------------- ------------------------------ CONV ------------------------------------------------------------


def conv_nd(n: int) -> Type[nn.Module]:
    """
    Returns a conv with nd (e.g., Conv2d for n=2). Work up to n=3.
    For a 4d Hiera, we could probably just implement this for n=4.
    """
    return [nn.Identity, nn.Conv1d, nn.Conv2d, nn.Conv3d][n]


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


class Conv3d(torch.nn.Conv3d):
    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv3d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x: torch.Tensor):
        x = F.conv3d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
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


# -------------------------------- ------------------------------ PATCHIFY ------------------------------------------------------------


#  ------ ViTDet ------ 


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


#  ------ Hiera ------


def do_pool(x: torch.Tensor, stride: int) -> torch.Tensor:
    # refer to `Unroll` to see how this performs a maxpool-Nd
    return x.view(x.shape[0], stride, -1, x.shape[-1]).max(dim=1).values


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


# -------------------------------------------------------- MASK GENERATOR ----------------------------------------------------------------


# TODO: currently assume TDHWC layout and not optimal design 
#       will be redesigned soon
class MaskGenerator:
    def __init__(
        self,
        device: str = 'cuda',
        patchify_scheme='downsample_time',
        batch_size=1,
        num_channels: int = 1,
        input_shape=(1, 128, 128, 128, 1),
        lateral_patch_size=16,
        axial_patch_size=16,
        temporal_patch_size=1,
        channels_to_mask: Optional[Sequence[int]] = None,
        time_downsample_pattern: Optional[Sequence[int]] = None,
        lateral_range: Optional[Tuple[float]]=(0.2, 0.8),
        axial_range: Optional[Tuple[float]]=(.5, 1.0),
        temporal_range: Optional[Tuple[float]]=(0.5, 1.0)
    ):
        super(MaskGenerator, self).__init__()

        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device)

        # task to perform (TODO: make patchify a variable?)        
        self.patchify_scheme = patchify_scheme
        patchify = True if patchify_scheme in ['downsample_time'] else False
        
        # for time downsampling task
        self.time_downsample_pattern = time_downsample_pattern
        
        # for channel prediction task (TODO: convert to one list)
        self.axial_range = axial_range
        self.lateral_range = lateral_range
        self.temporal_range = temporal_range
        self.channels_to_mask = channels_to_mask

        # mask/image dimensions (TODO: make all this one list)
        self.batch_size = batch_size
        self.num_channels = num_channels

        # TODO: don't assume layout
        if input_shape[1] > 1:
            self.time = input_shape[1] // temporal_patch_size if patchify else input_shape[1] 
        else:
            self.time = None
        if input_shape[2] > 1:
            self.depth = input_shape[2] // axial_patch_size if patchify else input_shape[2]
        else:
            self.depth = None
        self.height  = input_shape[3] // lateral_patch_size if patchify else input_shape[3]
        self.width = input_shape[4] // lateral_patch_size if patchify else input_shape[4]

        self.axes = [
            ("T", self.time,  temporal_range, self.time  is not None),
            ("Z", self.depth, axial_range, self.depth is not None),
            ("Y", self.height, lateral_range, True),
            ("X", self.width,  lateral_range, True),
        ]

    def _sample_block_size(self, low, high, full):
        """
        Draws a block length L so that L / full in [low , high].
        """
        if not (0.0 <= low <= high <= 1.0):
            raise ValueError("low <= high must both be in [0,1]")

        if low == high:
            frac = torch.tensor(low, device=self.device)
        else:
            frac = torch.distributions.Uniform(
            low, high).sample((1,)).to(self.device)

        length = torch.clamp((frac * full).round(), 1, full).int()
        return length

    # TODO: consider redesigning masking
    def get_random_block(self):
        """
        Returns a *binary* tensor whose rank depends on which axes are active.
        1 = block area (to mask later)
        0 = keep
        """
        block_shape, slices = [], []
        for _, axis_len, axis_sample_range, axis_exists in self.axes:
            if not axis_exists:
                block_shape.append(axis_len)
                slices.append(slice(None))
                continue

            size = self._sample_block_size(*axis_sample_range, axis_len)
            start = torch.randint(0, axis_len - size + 1, 
                                  (1,), generator=self.generator, 
                                  device=self.device).item()
            block_shape.append(axis_len)
            slices.append(slice(start, start + size.item()))

        shape = [s for s in block_shape if s is not None]

        block = torch.zeros(*shape, dtype=torch.bool, device=self.device)
        block[tuple(slices[i] for i, (_, _, _, a) in enumerate(self.axes) if a)] = True
        return block

    def mask_random_patches_per_channel(self):
        """
        1. Draw a block mask on the volume
        2. Expand it to every item in the batch
        3. Broadcast it across channels *only* for the
        indices in `channels_to_mask`
        """
        # (T?, Z?, H, W)
        spatial = self.get_random_block()
        # (B, T?, Z?, H, W)
        spatial = spatial.unsqueeze(0)
        spatial = spatial.expand(self.batch_size, *([-1] * (spatial.dim())))

        # (C,) -> (1,...,C)
        channels = torch.zeros(self.num_channels,
                            dtype=spatial.dtype,
                            device=spatial.device)
        channels[self.channels_to_mask] = 1
        channels = channels.view(*(1,) * spatial.dim(), -1)
        # broadcast multiply: 
        # (1,...,C) * (B, T?, Z?, H, W, 1) -> (B, T?, Z?, H, W, C)
        # now blocks are masked for the given channels
        mask = spatial.unsqueeze(-1) * channels

        return mask, None

    def mask_downsample_time(self, pattern: Sequence[int]):
        """
        Generates masks that downsample the time dimension by a factor. 
        """
        mask_pattern = torch.tensor(pattern, dtype=torch.bool, device=self.device)  
        K = mask_pattern.shape[0]  
        
        # mod all time values by K to extend 
        # the mask pattern across the time dimension
        time_indices = torch.arange(self.time, device=self.device) % K    
        time_mask = mask_pattern[time_indices]                            

        # mask: (time,) -> (time, (depth), height, width) -> (bs, time * (depth) * height * width)
        # later, we repeat across channel dimension (see masking.py in platform)
        # TODO: drop channel dim in logic?
        if self.depth:
            mask = time_mask.view(self.time, 1, 1, 1, 1).expand(-1, self.depth, self.height, self.width, 1)
            mask = mask.unsqueeze(0).expand(self.batch_size, -1, -1, -1, -1, -1)
            mask = mask.contiguous().view(self.batch_size, -1)
        else:
            mask = time_mask.view(self.time, 1, 1, 1).expand(-1, self.height, self.width, 1)
            mask = mask.unsqueeze(0).expand(self.batch_size, -1, -1, -1)
            mask = mask.contiguous().view(self.batch_size, -1)

        # masked patches are 1, unmasked are 0
        # so argsort will give us the original patch indices in (B,L)
        # TODO: double check this logic
        original_patch_indices = mask.int().argsort(dim=1, stable=True)
        
        return mask, original_patch_indices

    def __call__(self):
        if self.patchify_scheme == 'downsample_time':
            if self.time is None:
                raise ValueError("Time downsampling is not applicable for 3D data without a time dimension.")
            return self.mask_downsample_time(pattern=self.time_downsample_pattern)
        elif self.patchify_scheme == 'random_patches_per_channel':
            return self.mask_random_patches_per_channel()
        else:
            raise ValueError(f"Unknown patchify scheme: {self.patchify_scheme}")


# --------------------------------------------------------------------------------------------------------------------------------