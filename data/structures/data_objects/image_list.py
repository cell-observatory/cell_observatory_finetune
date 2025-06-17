"""
https://github.com/facebookresearch/detectron2/blob/main/detectron2/structures/image_list.py

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


from itertools import chain
from typing import Any, List, Tuple, Optional, Dict, Sequence

import torch
from torch import device

from cell_observatory_platform.data.io import record_init
from cell_observatory_finetune.data.structures.data_objects.data_shapes import (
    MULTICHANNEL_3D_HYPERCUBE,
    MULTICHANNEL_4D_HYPERCUBE
)


class ImageList:
    """
        Structure that holds a list of images (of possibly varying sizes)
        as a single tensor. This works by padding the images to the same size.
        The original size of each image is stored in `image_sizes`.
    """

    @record_init
    def __init__(
        self,
        tensor: torch.Tensor,
        image_sizes: List[Tuple],
        layout: MULTICHANNEL_3D_HYPERCUBE | MULTICHANNEL_4D_HYPERCUBE = MULTICHANNEL_4D_HYPERCUBE.TZYXC,
        orig_layout: MULTICHANNEL_3D_HYPERCUBE | MULTICHANNEL_4D_HYPERCUBE = None,
    ):
        """
        Arguments:
            tensor (Tensor): of shape (N, [T,D], H, W) or (N, C_1, ..., C_K, [T,D], H, W) where K >= 1
            image_sizes (List[Tuple[int, int, int]] | List[Tuple[int, int, int, int]])
            layout (MULTICHANNEL_HYPERCUBE): Desired tensor layout
            orig_layout (MULTICHANNEL_HYPERCUBE): Current tensor layout
        """
        self.tensor = tensor
        self.image_sizes = image_sizes

        self.layout = layout
        self.orig_layout = orig_layout if orig_layout is not None else layout

        if self.layout != self.orig_layout:
            if self.layout == MULTICHANNEL_3D_HYPERCUBE.TYXC or self.layout == MULTICHANNEL_4D_HYPERCUBE.TZYXC:
                self.tensor = self.layout.to_channel_last(self.tensor)
            elif self.layout == MULTICHANNEL_3D_HYPERCUBE.CTYX or self.layout == MULTICHANNEL_4D_HYPERCUBE.CTZYX:
                self.tensor = self.layout.to_channel_first(self.tensor)
            else:
                raise NotImplementedError(f"Unsupported layout {self.layout}")

        if isinstance(self.layout, MULTICHANNEL_3D_HYPERCUBE):
            if self.tensor.ndim == 4:
                self.tensor = self.tensor.unsqueeze(0) # (C, D, H, W) -> (1, C, D, H, W)

        elif isinstance(self.layout, MULTICHANNEL_4D_HYPERCUBE):
            if self.tensor.ndim == 5:
                self.tensor = self.tensor.unsqueeze(0) # (T, C, D, H, W) -> (1, T, C, D, H, W)

    @property
    def has_time(self) -> bool:
        return self.layout.has_temporal_dim()

    @property
    def num_timepoints(self) -> int | None:
        return self.layout.num_timepoints(self.tensor)

    @property
    def num_channels(self) -> int:
        return self.layout.num_channels(self.tensor)

    @property
    def image_shape(self) -> Tuple[int, int, int] | Tuple[int, int]:
        return self.layout.get_spatial_shape(self.tensor)

    @property
    def shape(self) -> Tuple:
        return self.tensor.shape

    @torch.jit.unused
    def to(self, *args: Any, **kwargs: Any) -> "ImageList":
        cast_tensor = self.tensor.to(*args, **kwargs)
        return ImageList(cast_tensor, image_sizes=self.image_sizes, layout=self.layout, orig_layout=self.orig_layout)

    @property
    def dtype(self) -> torch.dtype:
        return self.tensor.dtype

    @property
    def device(self) -> device:
        return self.tensor.device

    def __len__(self) -> int:
        return len(self.image_sizes)

    def __getitem__(self, idx) -> torch.Tensor:
        """ Access the individual image in its original size. """
        s = self.image_sizes[idx]

        if self.layout == MULTICHANNEL_3D_HYPERCUBE:
            if self.layout.is_channel_first():
                return self.tensor[idx, :, :s[0], :s[1], :s[2]]
            else:
                return self.tensor[idx, :s[0], :s[1], :s[2]]

        elif self.layout == MULTICHANNEL_4D_HYPERCUBE:
            if self.layout.is_channel_first():
                return self.tensor[idx, :, :s[0], :s[1], :s[2], :s[3]]
            else:
                return self.tensor[idx, :s[0], :s[1], :s[2], :s[3]]

        else:
            raise NotImplementedError(f"Unknown layout {self.layout}")

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __repr__(self) -> str:
        shape = self.layout.get_image_shape_dict(self.tensor)

        return (
            f"<ImageList  "
            f"N={self.tensor.shape[0]} | {shape} "
            f"layout={self.layout}  "
            f"orig_layout={self.orig_layout}  "
            f"device={self.tensor.device}>"
        )


    def copy(self, *, deep: bool = False) -> "ImageList":
        return ImageList(
            self.tensor.clone() if deep else self.tensor,
            self.image_sizes.copy(),
            layout=self.layout,
            orig_layout=self.orig_layout,
        )

    def get_image_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.layout, MULTICHANNEL_3D_HYPERCUBE):
            if self.layout.is_channel_first(): # (B, C, D, H, W)
                mean = self.tensor.mean(dim=(2, 3, 4), keepdim=True)
                std = self.tensor.std(dim=(2, 3, 4), keepdim=True)
            else:  # (B, D, H, W, C)
                mean = self.tensor.mean(dim=(1, 2, 3), keepdim=True)
                std = self.tensor.std(dim=(1, 2, 3), keepdim=True)

        elif isinstance(self.layout, MULTICHANNEL_4D_HYPERCUBE):
            if self.layout.is_channel_first(): # (B, C, T, D, H, W)
                mean = self.tensor.mean(dim=(2, 3, 4, 5), keepdim=True)
                std = self.tensor.std(dim=(2, 3, 4, 5), keepdim=True)
            else: # (B, T, D, H, W, C)
                mean = self.tensor.mean(dim=(1, 2, 3, 4), keepdim=True)
                std = self.tensor.std(dim=(1, 2, 3, 4), keepdim=True)

        else:
            raise NotImplementedError(f"Unsupported layout {self.layout}")

        return mean, std


    @staticmethod
    def from_tensors(
        tensors: List[torch.Tensor],
        layout: MULTICHANNEL_3D_HYPERCUBE | MULTICHANNEL_4D_HYPERCUBE = MULTICHANNEL_4D_HYPERCUBE.TZYXC,
        size_divisibility: int = 0,
        pad_value: float = 0.0,
        padding_constraints: Optional[Dict[str, int]] = None,
    ) -> "ImageList":
        """
        Args:
        tensors: a tuple or list of `torch.Tensor`, Each tuple is ([t, d], h, w, c) or (c, [t, d], h, w).
            The Tensors will be padded to the same shape with `pad_value`.
        size_divisibility (int): If `size_divisibility > 0`, add padding to ensure
            the common height and width is divisible by `size_divisibility`.
            This depends on the model and many models need a divisibility of 32.
        pad_value (float): value to pad.
        padding_constraints (optional[Dict]): If given, it would follow the format as
            {"size_divisibility": int, "square_size": int}, where `size_divisibility` will
            overwrite the above one if presented and `square_size` indicates the
            square padding size if `square_size` > 0.
        Returns:
            an `ImageList`.
        """
        assert len(tensors) > 0

        assert isinstance(tensors, (tuple, list))
        for t in tensors:
            assert isinstance(t, torch.Tensor), type(t)

        image_sizes = [layout.get_spatial_shape(im) for im in tensors] # List[Tuple[:]]
        image_sizes_tensor = [torch.as_tensor(x) for x in image_sizes]
        max_size = torch.stack(image_sizes_tensor).max(0).values  # List[Nx:] -> (N, :) -> (:,)

        if padding_constraints is not None:
            sq = padding_constraints.get("square_size", 0)

            if sq > 0: # pad to square.
                max_size = (sq for s in max_size)

                if "size_divisibility" in padding_constraints:
                    size_divisibility = padding_constraints["size_divisibility"]

        if size_divisibility > 1:
            stride = size_divisibility
            max_size = (max_size + (stride - 1)).div(stride, rounding_mode="floor") * stride

        batch_size = len(tensors)
        num_channels = layout.num_channels(tensors[0])

        if layout.has_temporal_dim():
            num_timepoints = layout.num_timepoints(tensors[0])
            if layout.is_channel_last():
                output_shape = [batch_size, num_timepoints, *max_size, num_channels]
            else:
                output_shape = [batch_size, num_channels, num_timepoints, *max_size]
        else:
            if layout.is_channel_last():
                output_shape = [batch_size, *max_size, num_channels]
            else:
                output_shape = [batch_size, num_channels, *max_size]

        padded_tensor = tensors[0].new_full(output_shape, fill_value=pad_value)
        padded_tensor = padded_tensor.to(tensors[0].device)

        for i, img in enumerate(tensors): # fill in the tensor with the images
            s = layout.get_spatial_shape(img)
            if len(s) == 2: # 2D image
                if layout.has_temporal_dim():
                    if layout.is_channel_last(): # (batch_size, T, Y, X, C)
                        padded_tensor[i, :, :s[0], :s[1], :].copy_(img)
                    else:  # (batch_size, C, T, Y, X)
                        padded_tensor[i, :, :, :s[0], :s[1]].copy_(img)
                else:
                    if layout.is_channel_last(): # (batch_size, Y, X, C)
                        padded_tensor[i, :s[0], :s[1], :].copy_(img)
                    else: # (batch_size, C, Y, X)
                        padded_tensor[i, :, :s[0], :s[1]].copy_(img)

            else: # 3D volume
                if layout.has_temporal_dim():
                    if layout.is_channel_last(): # (batch_size, T, Z, Y, X, C)
                        padded_tensor[i, :, :s[0], :s[1], :s[2], :].copy_(img)
                    else:  # (batch_size, C, T, Z, Y, X)
                        padded_tensor[i, :, :, :s[0], :s[1], :s[2]].copy_(img)
                else:
                    if layout.is_channel_last(): # (batch_size, Z, Y, X, C)
                        padded_tensor[i, :s[0], :s[1], :s[2], :].copy_(img)
                    else: # (batch_size, C, Z, Y, X)
                        padded_tensor[i, :, :s[0], :s[1], :s[2]].copy_(img)

        return ImageList(padded_tensor.contiguous(), image_sizes=image_sizes, layout=layout)


def cat_image_lists(
        image_lists: Sequence["ImageList"],
        pad_value: float = 0.0,
) -> "ImageList":
    """Concatenate multiple :class:`ImageList`s along the batch axis.

    All inputs must use the **same** `layout`.

    Args:
        image_lists : Sequence[ImageList]
        pad_value   : float
            Value to pad with when images differ in spatial size.

    Returns:
        One ImageList batched object with ``N1+N2+...`` images.
    """
    layout = image_lists[0].layout
    if any(il.layout != layout for il in image_lists):
        raise ValueError("All ImageList objects must share the same layout")
    
    shapes = {il.tensor.shape for il in image_lists}
    if len(shapes) == 1:                                   
        # e.g. (N_total, (T), C, D, H, W) OR # (N_total, (T), D, H, W, C)
        batched = torch.cat([il.tensor for il in image_lists], dim=0)  
        image_sizes = list(chain.from_iterable(il.image_sizes for il in image_lists))
        return ImageList(batched, image_sizes, layout=layout)

    tensors = [img for il in image_lists for img in il]
    return ImageList.from_tensors(
        tensors,
        pad_value=pad_value,
        layout=layout
    )


# --------------------------------------------------------------------------- OLD VERSION --------------------------------------------------


# from __future__ import division

# from enum import Enum
# from itertools import chain
# from typing import Any, Sequence, List, Optional, Tuple

# import torch
# from torch import device
# from torch.nn import functional as F

# from segmentation.structures.sample_objects.utils import record_init


# class Shape(Enum):
#     """Supported image layouts.

#     * ``CZYX`` - channel-first, no time
#     ``(C, Z, Y, X)``  or ``(N, C, Z, Y, X)``

#     * ``ZYXC`` - channel-last, no time
#     ``(Z, Y, X, C)``  or ``(N, Z, Y, X, C)``

#     * ``TCZYX`` - channel-first with time
#     ``(T, C, Z, Y, X)``  or ``(N, T, C, Z, Y, X)``

#     * ``TZYXC`` - channel-last with time
#     ``(T, Z, Y, X, C)``  or ``(N, T, Z, Y, X, C)``
#     """

#     CZYX = "CZYX"
#     ZYXC = "ZYXC"
#     TCZYX = "TCZYX"  
#     TZYXC = "TZYXC"

#     @property
#     def axes(self) -> Tuple[str, ...]:
#         # e.g. ("C","Z","Y","X")
#         # very useful function for 
#         # locating the spatial dimensions
#         return tuple(self.value)

#     # TODO: not used anymore, will be removed
#     def to_standard(self, tensor: torch.Tensor) -> torch.Tensor:
#         """Return a view with **channels first** (CZYX) or (TCZYX)."""
#         # already correct
#         if self is Shape.CZYX:
#             return tensor
#         if self is Shape.TCZYX:
#             return tensor

#         if self is Shape.ZYXC:
#             if tensor.ndim == 5:
#                 perm = (0, 4, 1, 2, 3) # (N, C, Z, Y, X)
#             else:
#                 perm = (3, 0, 1, 2) # (C, Z, Y, X)
#         elif self is Shape.TZYXC:
#             if tensor.ndim == 6:
#                 perm = (0, 1, 5, 2, 3, 4) # (N, T, C, Z, Y, X)
#             else:
#                 perm = (0, 4, 1, 2, 3) # (T, C, Z, Y, X)
#         else:
#             raise ValueError(f"Unsupported layout: {self}")
        
#         return tensor.permute(*perm)
    
#     def convert_layout(
#         self,
#         tensor: torch.Tensor,
#         image_sizes: List[Tuple[int, ...]],
#         new_layout: "Shape",
#     ) -> Tuple[torch.Tensor, List[Tuple[int, ...]]]:
#         """
#         Convert *tensor* **and** *image_sizes* from layout ``self`` to ``new_layout``.
#         """
#         if self is new_layout:
#             return tensor, image_sizes

#         if len(self.axes) != len(new_layout.axes):
#             raise ValueError(
#                 f"Incompatible layouts: {self.name} <-> {new_layout.name}"
#             )

#         # determine if there is a batch dim (lead == 1) or not (lead == 0)
#         lead = tensor.ndim - len(self.axes)
#         if lead not in (0, 1):
#             raise ValueError(
#                 f"Tensor rank ({tensor.ndim}) does not match layout {self.name}"
#             )
        
#         # this logic is a bit excessive given that we only really
#         # consider an extra batch dimension
#         perm = list(range(lead)) + [
#             lead + self.axes.index(ax) for ax in new_layout.axes
#         ]
#         tensor_out = tensor.permute(*perm)

#         # e.g. ["C","Z","Y","X"]
#         old_axes = list(self.axes)
#         # e.g. ["Z","Y","X","C"]
#         new_axes = list(new_layout.axes)

#         if any(len(sz) != len(old_axes) for sz in image_sizes):
#             raise ValueError(
#                 f"All image_size entries must have {len(old_axes)} dimensions "
#                 f"(layout {self.name}); got {[len(sz) for sz in image_sizes]}"
#             )

#         # where to pick the value from 
#         # in each tuple
#         axis_map = [
#             old_axes.index(ax)
#             for ax in new_axes
#         ]

#         def reorder(sz: Tuple[int, ...]) -> Tuple[int, ...]:
#             return tuple(sz[i] for i in axis_map)

#         image_sizes_out = [reorder(sz) for sz in image_sizes]

#         return tensor_out, image_sizes_out
            


# class ImageList:
#     """
#     Structure that holds a list of images (of possibly varying sizes)
#     as a single tensor. This works by padding the images to the same size.
#     The sizes of each image is stored in `image_sizes`.
#     """
#     @record_init
#     def __init__(self, 
#         tensor: torch.Tensor, 
#         image_sizes: List[Tuple[int, int, int, int]],
#         layout: Shape = Shape.CZYX, 
#         orig_layout: Shape = Shape.CZYX,
#         orig_image_sizes: Optional[List[Tuple[int, int]]] = None):
#         """
#         Arguments:
#             tensor (Tensor): image tensor.
#             image_sizes (list[tuple[int, int]]): Each tuple is ((t), d, h, w). It can
#                 be smaller than (T, D, H, W) due to padding.
#         """
#         self.layout = layout
#         if tensor.ndim < 4:
#             raise ValueError(
#                 f"ImageList expects a 4D or 5D tensor, got {tensor.ndim}D tensor with shape {tensor.shape}"
#             )
        
#         self.tensor = tensor

#         if self.layout in (Shape.TCZYX, Shape.TZYXC):
#             if self.tensor.ndim == 5:
#                 # (T, C, D, H, W) -> (1, T, C, D, H, W)
#                 self.tensor = self.tensor.unsqueeze(0)
#         elif self.layout in (Shape.CZYX, Shape.ZYXC):
#             if self.tensor.ndim == 4:
#                 # (C, D, H, W) -> (1, C, D, H, W)
#                 self.tensor = self.tensor.unsqueeze(0)

#         self.image_sizes = image_sizes
#         self.orig_layout = orig_layout if orig_layout is not None else layout
#         self.orig_image_sizes = orig_image_sizes if orig_image_sizes is not None else image_sizes

#     @property
#     def shape(self) -> Tuple[int, int, int, int]:
#         if self.layout == Shape.CZYX:
#             # get: (D, H, W) from (B, C, D, H, W)
#             return self.tensor.shape[2:]
#         elif self.layout == Shape.TCZYX:
#             # get: (D, H, W) from (B, T, C, D, H, W)
#             return self.tensor.shape[3:]
#         elif self.layout == Shape.ZYXC:
#             # get: (D, H, W) from (B, D, H, W, C)
#             return self.tensor.shape[1:4]
#         elif self.layout == Shape.TZYXC:
#             # get: (D, H, W) from (B, T, D, H, W, C)
#             return self.tensor.shape[2:5]
#         else:
#             raise ValueError(f"Unsupported layout: {self.layout}")
        
#     def convert_layout(self, new_layout: Shape):
#         self.tensor, self.image_sizes = self.layout.convert_layout(self.tensor, self.image_sizes, new_layout)
#         self.layout = new_layout

#     def _has_time_axis(self) -> bool:
#         return self.layout in (Shape.TCZYX, Shape.TZYXC)
    
#     def get_image_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
#         if self.layout == Shape.CZYX:
#             # dimensions: (B, C, D, H, W)
#             mean = self.tensor.mean(dim=(2, 3, 4), keepdim=True) 
#             std = self.tensor.std(dim=(2, 3, 4), keepdim=True)
#         elif self.layout == Shape.TCZYX:
#             # dimensions: (B, T, C, D, H, W)
#             mean = self.tensor.mean(dim=(1, 3, 4, 5), keepdim=True)
#             std = self.tensor.std(dim=(1, 3, 4, 5), keepdim=True)
#         elif self.layout == Shape.ZYXC:
#             # dimensions: (B, D, H, W, C)
#             mean = self.tensor.mean(dim=(1, 2, 3), keepdim=True)
#             std = self.tensor.std(dim=(1, 2, 3), keepdim=True)
#         elif self.layout == Shape.TZYXC:
#             mean = self.tensor.mean(dim=(1, 2, 3, 4), keepdim=True)
#             std = self.tensor.std(dim=(1, 2, 3, 4), keepdim=True)
#         else:
#             raise ValueError(f"Unsupported layout: {self.layout}")
#         return mean, std

#     def resize(self, new_size: tuple[int, int, int], mode: str = "trilinear"):
#         """
#         Resize spatial volume to `new_size = (D, H, W)`.
#         Works for layouts:
#             - CZYX   : [B, C, D, H, W]
#             - TCZYX  : [B, T, C, D, H, W]
#             - ZYXC   : [B, D, H, W, C]
#             - TZYXC  : [B, T, D, H, W, C]
#         """
#         # TODO: does this work as intended if tensors
#         #       are not even sized, i.e. if they are padded?
#         x = self.tensor                                   
#         if self.layout == Shape.CZYX:                    
#             x = F.interpolate(x, size=new_size, mode=mode, align_corners=False)
#         elif self.layout == Shape.TCZYX:                 
#             B, T, C = x.shape[:3]
#             # merge T into batch
#             x_flat = x.reshape(B * T, C, *x.shape[-3:])
#             x_flat = F.interpolate(x_flat, size=new_size, mode=mode, align_corners=False)
#             # restore T
#             x = x_flat.view(B, T, C, *new_size)
#         elif self.layout == Shape.ZYXC:
#             # [B, D, H, W, C]
#             B, C = x.shape[0], x.shape[-1]
#             # [B, C, D, H, W]
#             x_perm = x.permute(0, 4, 1, 2, 3)           
#             x_perm = F.interpolate(
#                 x_perm, size=new_size, mode=mode, align_corners=False
#             )
#             # restore channel-last order
#             # [B, D′, H′, W′, C]
#             x = x_perm.permute(0, 2, 3, 4, 1)           
#         elif self.layout == Shape.TZYXC:
#             # [B, T, D, H, W, C]
#             B, T = x.shape[:2]
#             C = x.shape[-1]
#             # [B, T, D, H, W, C] -> [B, T, C, D, H, W] 
#             x_perm = x.permute(0, 1, 5, 2, 3, 4)
#             # [BxT, C, D, H, W]
#             x_flat = x_perm.reshape(B * T, C, *x_perm.shape[-3:])
#             x_flat = F.interpolate(
#                 x_flat, size=new_size, mode=mode, align_corners=False
#             )
#             # reshape back and restore channel-last
#             x_perm = x_flat.view(B, T, C, *new_size)
#             x = x_perm.permute(0, 1, 3, 4, 5, 2)
#         else:
#             raise NotImplementedError(f"resize not implemented for layout {self.layout}")

#         self.tensor = x
#         self.image_sizes = [tuple(x.shape[1:])] * len(self)          
#         return self

#     def __len__(self) -> int:
#         return len(self.image_sizes)

#     def __getitem__(self, idx) -> torch.Tensor:
#         """
#         Access the individual image in its original size.

#         Args:
#             idx: int or slice

#         Returns:
#             Tensor: an image of shape ((T), C, D, H, W)
#         """
#         size = self.image_sizes[idx]
#         # TODO: this is redundant, can be simplified 
#         if self.layout == Shape.TCZYX:
#             return self.tensor[idx, : size[0], : size[1], : size[2], : size[3], : size[4]]
#         elif self.layout == Shape.TZYXC:
#             return self.tensor[idx, : size[0], : size[1], : size[2], : size[3], : size[4]]
#         elif self.layout == Shape.CZYX:
#             return self.tensor[idx, : size[0], : size[1], : size[2], : size[3]]
#         elif self.layout == Shape.ZYXC:
#             return self.tensor[idx, : size[0], : size[1], : size[2], : size[3]]
#         else:
#             raise ValueError(f"Unsupported layout: {self.layout}")

#     @torch.jit.unused
#     def to(self, *args: Any, **kwargs: Any) -> "ImageList":
#         cast_tensor = self.tensor.to(*args, **kwargs)
#         return ImageList(cast_tensor, self.image_sizes)

#     @property
#     def dtype(self) -> torch.dtype:
#         """Return the dtype of the underlying tensor."""
#         return self.tensor.dtype

#     @property
#     def device(self) -> device:
#         return self.tensor.device

#     @staticmethod
#     def from_tensors(
#         tensors: List[torch.Tensor],
#         pad_value: float = 0.0,
#         layout: Shape = Shape.CZYX,
#     ) -> "ImageList":
#         """
#         Build an ImageList from a list of tensors that all follow *layout*.
#         Each tensor must have exactly ``len(layout.axes)`` dimensions
#         (e.g. 4 for CZYX/ZYXC, 5 for TCZYX/TZYXC).
#         """
#         assert tensors and isinstance(tensors, (list, tuple)), "`tensors` empty!"

#         expected_ndim, image_sizes = len(layout.axes), []
#         for i, t in enumerate(tensors):
#             if t.ndim != expected_ndim:
#                 raise ValueError(
#                     f"Tensor #{i} has {t.ndim} dims, but layout {layout.name} "
#                     f"expects {expected_ndim}"
#                 )
#             image_sizes.append(tuple(t.shape))

#         max_size = torch.stack([torch.tensor(sz) for sz in image_sizes]).max(0).values

#         batch_shape = [len(tensors)] + max_size.tolist()
#         batched_imgs = tensors[0].new_full(batch_shape, pad_value)

#         for i, img in enumerate(tensors):
#             slices = (i,) + tuple(slice(0, s) for s in img.shape)
#             batched_imgs[slices].copy_(img)

#         return ImageList(
#             batched_imgs.contiguous(),
#             image_sizes,
#             layout=layout,
#             orig_layout=layout,
#         )
  
#     def __iter__(self):
#         for idx in range(len(self)):
#             yield self[idx]

#     def copy(self, *, deep: bool = False) -> "ImageList":
#         return ImageList(
#             self.tensor.clone() if deep else self.tensor,
#             self.image_sizes.copy(),
#             layout=self.layout,
#             orig_layout=self.orig_layout,
#             orig_image_sizes=self.orig_image_sizes.copy(),
#         )

#     def __repr__(self) -> str:
#         b = self.tensor.shape[0]  
#         # TODO: logic can probably be simplified
#         if self.layout == Shape.TCZYX:
#             t = self.tensor.shape[1]
#             c = self.tensor.shape[2]
#             d, h, w = self.tensor.shape[-3:]        
#         elif self.layout == Shape.TZYXC:
#             t = self.tensor.shape[1]
#             d, h, w = self.tensor.shape[-4:-1]
#             c = self.tensor.shape[-1]
#         elif self.layout == Shape.CZYX:
#             c = self.tensor.shape[1]
#             d, h, w = self.tensor.shape[-3:]
#             t = 1
#         elif self.layout == Shape.ZYXC:
#             d, h, w = self.tensor.shape[-4:-1]
#             c = self.tensor.shape[-1]
#             t = 1

#         return (
#             f"<ImageList  "
#             f"N={b} | T={t} | C={c} | DxHxW={d}x{h}x{w})  "
#             f"orig={self.orig_layout.name}  "
#             f"device={self.tensor.device}>"
#         )
  
# def cat_image_lists(
#         image_lists: Sequence["ImageList"],
#         pad_value: float = 0.0,
# ) -> "ImageList":
#     """Concatenate multiple :class:`ImageList`s along the batch axis.

#     All inputs must use the **same** `layout`.

#     Args:
#         image_lists : Sequence[ImageList]
#         pad_value   : float
#             Value to pad with when images differ in spatial size.

#     Returns:
#         One ImageList batched object with ``N1+N2+...`` images.
#     """
#     layout = image_lists[0].layout
#     if any(il.layout != layout for il in image_lists):
#         raise ValueError("All ImageList objects must share the same layout")
    
#     shapes = {il.tensor.shape for il in image_lists}
#     if len(shapes) == 1:                                   
#         # e.g. (N_total, (T), C, D, H, W) OR # (N_total, (T), D, H, W, C)
#         batched = torch.cat([il.tensor for il in image_lists], dim=0)  
#         image_sizes = list(chain.from_iterable(il.image_sizes for il in image_lists))
#         return ImageList(batched, image_sizes, layout=layout)

#     tensors = [img for il in image_lists for img in il]
#     return ImageList.from_tensors(
#         tensors,
#         pad_value=pad_value,
#         layout=layout
#     )


# def spatial_dims(layout: Shape, tensor_ndim: int) -> Tuple[int, int, int]:
#     """
#     Return the (Z, Y, X) dimension indices for `tensor` that follows `layout`.
#     """
#     lead = tensor_ndim - len(layout.axes)
#     # map from semantic axis -> absolute dim index
#     ax2dim = {ax: lead + i for i, ax in enumerate(layout.axes)}
#     return (ax2dim["Z"], ax2dim["Y"], ax2dim["X"])