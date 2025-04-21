"""
https://github.com/pytorch/vision/blob/309bd7a1512ad9ff0e9729fbdad043cb3472e4cb/torchvision/models/detection/anchor_utils.py#L12

BSD 3-Clause License

Copyright (c) Soumith Chintala 2016, 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


from typing import List, Optional

import torch
from torch import nn, Tensor

from segmentation.utils.image_list import ImageList


class AnchorGenerator(nn.Module):
    """
    Module that generates anchors for a set of feature maps and
    image sizes.

    The module support computing anchors at multiple sizes and aspect ratios
    per feature map. This module assumes aspect ratio = height / width for
    each anchor.

    sizes and aspect_ratios should have the same number of elements, and it should
    correspond to the number of feature maps.

    sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
    and AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] * aspect_ratios_z anchors
    per spatial location for feature map i.

    Args:
        sizes (Tuple[Tuple[int]]): 
        aspect_ratios (Tuple[Tuple[float]]): Aspect ratios for width and height.
        aspect_ratios_z (Tuple[Tuple[float]]): Aspect ratios for depth.
    """

    __annotations__ = {
        "cell_anchors": List[torch.Tensor],
    }

    def __init__(
        self,
        sizes=((128, 256, 512),), 
        aspect_ratios=((0.5, 1.0, 2.0),),
        aspect_ratios_z=((0.5, 1.0, 2.0),),  # Default: same scaling in depth.
    ):
        super().__init__()

        # if not isinstance(sizes[0], (list, tuple)):
        #     # TODO change this
        #     sizes = tuple((s,) for s in sizes)
        # if not isinstance(aspect_ratios[0], (list, tuple)):
        #     aspect_ratios = (aspect_ratios,) * len(sizes)
        # if not isinstance(aspect_ratios_z[0], (list, tuple)):
        #     aspect_ratios_z = (aspect_ratios_z,) * len(sizes)

        # self.sizes = sizes
        # self.aspect_ratios = aspect_ratios
        # self.aspect_ratios_z = aspect_ratios_z

        # Normalize everything to tuple-of-tuples
        self.sizes = tuple(tuple(s) for s in sizes)
        self.aspect_ratios = tuple(tuple(a) for a in aspect_ratios)
        self.aspect_ratios_z = tuple(tuple(a) for a in aspect_ratios_z)

        self.cell_anchors = [
            self.generate_anchors(size, aspect_ratio, aspect_ratio_z) for size, aspect_ratio, aspect_ratio_z in zip(sizes, aspect_ratios, aspect_ratios_z)
        ]

    # TODO: https://github.com/pytorch/pytorch/issues/26792
    # For every (aspect_ratios, aspect_ratios_z, scales) combination, output a zero-centered anchor with those values.
    # (scales, aspect_ratios, aspect_ratios_z) are usually an element of zip(self.scales, self.aspect_ratios, self.aspect_ratios_z).
    # This method assumes aspect ratio = height / width for an anchor.
    def generate_anchors(
        self,
        scales: List[int],
        aspect_ratios: List[float],
        aspect_ratios_z: List[float],
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> Tensor:
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        aspect_ratios_z = torch.as_tensor(aspect_ratios_z, dtype=dtype, device=device)
        
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        w_grid, scale_grid, z_aspect_grid = torch.meshgrid(w_ratios, scales, aspect_ratios_z, indexing="ij")
        h_grid, scale_grid, z_aspect_grid = torch.meshgrid(h_ratios, scales, aspect_ratios_z, indexing="ij")
        
        ws = (w_grid * scale_grid).reshape(-1) # flatten
        hs = (h_grid * scale_grid).reshape(-1) 
        zs = (z_aspect_grid * scale_grid).reshape(-1)
    
        base_anchors = torch.stack([-ws, -hs, -zs, ws, hs, zs], dim=1) / 2 # (x_min, y_min, z_min, x_max, y_max, z_max) for all combinations of scales and aspect ratios
        return base_anchors.round()

    def set_cell_anchors(self, dtype: torch.dtype, device: torch.device):
        self.cell_anchors = [cell_anchor.to(dtype=dtype, device=device) for cell_anchor in self.cell_anchors]

    def num_anchors_per_location(self) -> List[int]:
        return [len(s) * len(a) * len(z) for s, a, z in zip(self.sizes, self.aspect_ratios, self.aspect_ratios_z)]

    # For every combination of (a, (g, s), i) in (self.cell_anchors, zip(grid_sizes, strides), 0:2),
    # output g[i] anchors that are s[i] distance apart in direction i, with the same dimensions as a.
    def grid_anchors(self, grid_sizes: List[List[int]], strides: List[List[Tensor]]) -> List[Tensor]:
        anchors = []
        cell_anchors = self.cell_anchors
        torch._assert(cell_anchors is not None, "cell_anchors should not be None")
        torch._assert(
            len(grid_sizes) == len(strides) == len(cell_anchors),
            "Anchors should be Tuple[Tuple[int]] because each feature "
            "map could potentially have different sizes and aspect ratios. "
            "There needs to be a match between the number of "
            "feature maps passed and the number of sizes / aspect ratios specified. " \
            f"Currently, there are {len(grid_sizes)} grid sizes and {len(cell_anchors)} cell" \
            f"anchors and {len(strides)} strides!"
            ,
        )

        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            # print(f"size: {size}, stride: {stride}, base_anchors: {base_anchors}")
            # raise ValueError("DEBUG anchors")
            grid_depth, grid_height, grid_width = size
            stride_depth, stride_height, stride_width = stride
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, z_center, x_center, y_center, z_center]
            shifts_x = torch.arange(0, grid_width, dtype=torch.int32, device=device) * stride_width # recall: given by image_size / feature_map_size
            shifts_y = torch.arange(0, grid_height, dtype=torch.int32, device=device) * stride_height
            shifts_z = torch.arange(0, grid_depth, dtype=torch.int32, device=device) * stride_depth
            
            shifts_z, shift_y, shift_x = torch.meshgrid(shifts_z, shifts_y, shifts_x, indexing="ij")
            
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shift_z = shifts_z.reshape(-1)
            
            shifts = torch.stack((shift_x, shift_y, shift_z, shift_x, shift_y, shift_z), dim=1) 

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            # I.e. for each base position given by shifts (x,y,z), we add base_anchors 
            # according to aspect_ratios and scales. Anchors is list of all such anchors. 
            anchors.append((shifts.view(-1, 1, 6) + base_anchors.view(1, -1, 6)).reshape(-1, 6))  

        return anchors

    def forward(self, image_list: ImageList, feature_maps: List[Tensor]) -> List[Tensor]:
        grid_sizes = [feature_map.shape[-3:] for feature_map in feature_maps]
        image_size = image_list.tensors.shape[-3:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device

        strides = [
            [
                torch.empty((), dtype=torch.int64, device=device).fill_(image_size[0] // g[0]),
                torch.empty((), dtype=torch.int64, device=device).fill_(image_size[1] // g[1]),
                torch.empty((), dtype=torch.int64, device=device).fill_(image_size[2] // g[2]),
            ]
            for g in grid_sizes
        ]

        self.set_cell_anchors(dtype, device)
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)
        anchors: List[List[torch.Tensor]] = []

        for _ in range(len(image_list.image_sizes)):
            anchors_in_image = [anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]
            anchors.append(anchors_in_image)

        # for anchor in anchors_in_image[0]: 
        #     print(anchor)
        # raise ValueError("DEBUG anchors")

        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors] # over images & feature maps
        return anchors