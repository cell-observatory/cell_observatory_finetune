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

from __future__ import division

from enum import Enum
from itertools import chain
from collections.abc import Mapping
from typing import Any, Dict, Sequence, List, Optional, Tuple

import torch
from torch import device
from torch.nn import functional as F

from segmentation.structures.sample_objects.utils import record_init


class Shape(Enum):
  """Supported 3-D image layouts.

  * ``CZYX`` - channel-first (torch / PyTorch convention)  
  ``(..., C, Z, Y, X)``  e.g. ``(N, C, D, H, W)``

  * ``ZYXC`` - channel-last (common in Zarr / image stacks)  
    ``(..., Z, Y, X, C)``  e.g. ``(N, D, H, W, C)``
  """

  CZYX = "CZYX"
  ZYXC = "ZYXC"

  @property
  def axes(self) -> Tuple[str, ...]:
      return tuple(self.value)                # e.g. ("C","Z","Y","X")

  def to_standard(self, tensor: torch.Tensor) -> torch.Tensor:
    """Return a view with **channels first** (CZYX)."""
    if self is Shape.CZYX:
        return tensor                       # already correct
    has_batch = tensor.ndim == 5            # (N, Z, Y, X, C)
    perm = (0, 4, 1, 2, 3) if has_batch else (3, 0, 1, 2)
    return tensor.permute(*perm)

  def from_standard(self, tensor: torch.Tensor) -> torch.Tensor:
    """Convert a *channel-first* tensor back to this layout."""
    if self is Shape.ZYXC:
        return tensor
    has_batch = tensor.ndim == 5            # (N, C, Z, Y, X)
    perm = (0, 2, 3, 4, 1) if has_batch else (1, 2, 3, 0)
    return tensor.permute(*perm)


class ImageList:
    """
    Structure that holds a list of images (of possibly varying sizes)
    as a single tensor. This works by padding the images to the same size.
    The original sizes of each image is stored in `image_sizes`.

    Attributes:
    image_sizes (list[tuple[int, int]]): each tuple is (d, h, w).
    """
    @record_init
    def __init__(self, 
        tensor: torch.Tensor, 
        image_sizes: List[Tuple[int, int]],
        standardize: bool = True,
        layout: Shape = Shape.CZYX, 
        orig_layout: Shape = Shape.CZYX,
        orig_image_sizes: Optional[List[Tuple[int, int]]] = None):
        """
        Arguments:
        tensor (Tensor): of shape (N, D, H, W) or (N, C_1, ..., C_K, D, H, W) where K >= 1
        image_sizes (list[tuple[int, int]]): Each tuple is (d, h, w). It can
            be smaller than (D, H, W) due to padding.
        """
        self.layout = layout
        if tensor.ndim < 4:
            raise ValueError(
                f"ImageList expects a 4D or 5D tensor, got {tensor.ndim}D tensor with shape {tensor.shape}"
            )
        if standardize:
            if tensor.ndim == 4:
                tensor = tensor.unsqueeze(0)
            self.tensor: torch.Tensor = tensor
            self._to_standard()  # convert to CZYX layout
        else:
            self.tensor: torch.Tensor = tensor
        self.image_sizes = image_sizes
        self.orig_layout = orig_layout if orig_layout is not None else layout
        self.orig_image_sizes = orig_image_sizes if orig_image_sizes is not None else image_sizes

    def _to_standard(self):
        """Convert the tensor to standard CZYX layout."""
        if self.layout != Shape.CZYX:
            self.tensor = self.layout.to_standard(self.tensor)
            self.layout = Shape.CZYX
    
    def _from_standard(self):
        """Convert the tensor from standard CZYX layout to original layout."""
        if self.layout != Shape.ZYXC:
            self.tensor = self.layout.from_standard(self.tensor)
            self.layout = Shape.ZYXC

    def resize(self, new_size: Tuple[int,int,int], mode="trilinear"):
        self.tensor = F.interpolate(self.tensor, size=new_size, mode=mode, align_corners=False)
        self.image_sizes = [new_size] * len(self)
        return self

    def __len__(self) -> int:
        return len(self.image_sizes)

    def __getitem__(self, idx) -> torch.Tensor:
        """
        Access the individual image in its original size.

        Args:
        idx: int or slice

        Returns:
        Tensor: an image of shape (D, H, W) or (C_1, ..., C_K, D, H, W) where K >= 1
        """
        size = self.image_sizes[idx]
        return self.tensor[idx, ..., : size[0], : size[1], : size[2]]

    @torch.jit.unused
    def to(self, *args: Any, **kwargs: Any) -> "ImageList":
        cast_tensor = self.tensor.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the underlying tensor."""
        return self.tensor.dtype

    @property
    def device(self) -> device:
        return self.tensor.device

    # TODO: method currently assumes layout CZYX
    @staticmethod
    def from_tensors(
        tensors: List[torch.Tensor],
        size_divisibility: int = 0,
        pad_value: float = 0.0,
        padding_constraints: Optional[Dict[str, int]] = None,
    ) -> "ImageList":
        """
        Args:
        tensors: a tuple or list of `torch.Tensor`, each of shape (Di, Hi, Wi) or
            (C_1, ..., C_K, Di, Hi, Wi) where K >= 1. The Tensors will be padded
            to the same shape with `pad_value`.
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
        # all dimensions should be the same except 
        # perhaps the last 3
        assert t.shape[:-3] == tensors[0].shape[:-3], t.shape

        image_sizes = [(im.shape[-3], im.shape[-2], im.shape[-1]) for im in tensors]
        # List[Tuple[int, int, int]]
        image_sizes_tensor = [torch.as_tensor(x) for x in image_sizes]
        # List[Nx3] -> (N, 3) -> (3,)
        max_size = torch.stack(image_sizes_tensor).max(0).values

        if padding_constraints is not None:
            square_size = padding_constraints.get("square_size", 0)
        if square_size > 0:
            # pad to square.
            max_size[0] = max_size[1] = max_size[2] = square_size
            if "size_divisibility" in padding_constraints:
                size_divisibility = padding_constraints["size_divisibility"]
        
        if size_divisibility > 1:
            stride = size_divisibility
        # the last dims D,H,W, all subject to divisibility requirement
        max_size = (max_size + (stride - 1)).div(stride, rounding_mode="floor") * stride

        if len(tensors) == 1:
            d_pad = max_size[-3] - image_sizes_tensor[0][0]
            h_pad = max_size[-2] - image_sizes_tensor[0][1]
            w_pad = max_size[-1] - image_sizes_tensor[0][2]
            # F.pad expects (W, H, D) order in its pad tuple:
            #   (w_left, w_right, h_left, h_right, d_left, d_right)
            # returns: (1, C, D_max, H_max, W_max)
            batched_imgs = F.pad(
                tensors[0],(0, w_pad, 0, h_pad, 0, d_pad),
                value=pad_value).unsqueeze_(0)  # add batch dim
        else:
            # batch_shape: (N, C, D_max, H_max, W_max)
            batch_shape = [len(tensors)] + list(tensors[0].shape[:-3]) + list(max_size)
            
            # allocates tensor (N, C, D_max, H_max, W_max) with pad_value
            # on device of tensors[0]
            batched_imgs = tensors[0].new_full(batch_shape, fill_value=pad_value)
            batched_imgs = batched_imgs.to(tensors[0].device)
            
        # fill in the tensor with the images
        for i, img in enumerate(tensors):
            batched_imgs[i, ..., : img.shape[-3], : img.shape[-2], : img.shape[-1]].copy_(img)

        return ImageList(batched_imgs.contiguous(), image_sizes)
  
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __repr__(self) -> str:
        b    = self.tensor.shape[0]
        c    = self.tensor.shape[1]
        d, h, w = self.tensor.shape[-3:]
        d_hw   = f"{d}x{h}x{w}"

        return (
            f"<ImageList  "
            f"N={b} | C={c} | DxHxW=({d_hw})  "
            f"orig={self.orig_layout.name}  "
            f"device={self.tensor.device}>"
        )
  
def cat_image_lists(
        image_lists: Sequence["ImageList"],
        pad_value: float = 0.0,
        size_divisibility: int = 0,
        padding_constraints: Optional[Dict[str, int]] = None,
    ) -> "ImageList":
    """Concatenate multiple :class:`ImageList`s along the batch axis.

    All inputs must use the **same** `layout`.

    Args:
        image_lists : Sequence[ImageList]
        pad_value   : float
            Value to pad with when images differ in spatial size.
        size_divisibility : int
            Passed straight to :meth:`from_tensors`.
        padding_constraints : dict | None
            Passed straight to :meth:`from_tensors`.

    Returns:
        ImageList
            One batched object with ``N1+N2+...`` images.
    """
    if not image_lists:
        raise ValueError("image_lists must be non-empty")

    layout = image_lists[0].layout
    if any(il.layout != layout for il in image_lists):
        raise ValueError("all ImageList objects must share the same layout")
    
    shapes = {il.tensor.shape for il in image_lists}
    if len(shapes) == 1:                                   
        batched = torch.cat([il.tensor for il in image_lists], dim=0)  # (N_total, C, D, H, W)
        image_sizes = list(chain.from_iterable(il.image_sizes for il in image_lists))
        return ImageList(batched, image_sizes, layout=layout)

    if layout != Shape.CZYX:
        raise ValueError(f"ImageList concatenation only supports CZYX layout currently, got {layout}")

    tensors = [img for il in image_lists for img in il]     # flat list of (C,D_i,H_i,W_i)
    return ImageList.from_tensors(
        tensors,
        size_divisibility=size_divisibility,
        pad_value=pad_value,
        padding_constraints=padding_constraints,
        layout=layout
    )