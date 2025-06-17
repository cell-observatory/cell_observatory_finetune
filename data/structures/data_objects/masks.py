"""
https://github.com/facebookresearch/detectron2/blob/536dc9d527074e3b15df5f6677ffe1f4e104a4ab/detectron2/structures/masks.py#L88

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


# TODO: extend with time axis


from typing import Any, Iterator, List, Union

import numpy as np

import torch
import torch.nn.functional as F

from cell_observatory_platform.data.io import record_init
from cell_observatory_finetune.models.ops.roi_align_nd import RoIAlign3DFunction
from cell_observatory_finetune.data.structures.data_objects.boxes import Boxes, expand_boxes


class BitMasks:
    """
    This class stores the segmentation masks for all objects in one image, in
    the form of bitmaps.

    Attributes:
        tensor: bool Tensor of (N,D,H,W) representing N instances in the image.
    """
    @record_init
    def __init__(self, tensor: Union[torch.Tensor, np.ndarray]):
        """
        Args:
            tensor: bool Tensor of (N,D,H,W) representing N instances in the image.
        """        
        assert tensor.dim() == 4
        self.tensor = tensor.to(torch.bool)
        self.image_size = tensor.shape[1:]

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the underlying tensor."""
        return self.tensor.dtype
        
    def clone(self) -> "BitMasks":
        """
        Clone the Masks.

        Returns:
            Masks
        """
        return BitMasks(self.tensor.clone())

    def cpu(self) -> "BitMasks":
        """Move Masks to CPU."""
        return BitMasks(self.tensor.cpu())

    def cuda(self, device=None) -> "BitMasks":
        """Move BitMasks to CUDA device."""
        return BitMasks(self.tensor.cuda(device))

    def detach(self) -> "BitMasks":
        """Detach from computation graph."""
        return BitMasks(self.tensor.detach())

    @torch.jit.unused
    def to(self, *args: Any, **kwargs: Any) -> "BitMasks":
        return BitMasks(self.tensor.to(*args, **kwargs))

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @torch.jit.unused
    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "BitMasks":
        """
        Returns:
            BitMasks: Create a new :class:`BitMasks` by indexing.

        The following usage are allowed:

        1. `new_masks = masks[3]`: return a `BitMasks` which contains only one mask.
        2. `new_masks = masks[2:10]`: return a slice of masks.
        3. `new_masks = masks[vector]`, where vector is a torch.BoolTensor
           with `length = len(masks)`. Nonzero elements in the vector will be selected.

        Note that the returned object might share storage with this object,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return BitMasks(self.tensor[item].unsqueeze(0))
        m = self.tensor[item]
        assert m.dim() == 4, "Indexing on BitMasks with {} returns a tensor with shape {}!".format(item, m.shape)
        return BitMasks(m)
        
    @torch.jit.unused
    def __iter__(self):
        yield from self.tensor

    @torch.jit.unused
    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={})".format(len(self.tensor))
        return s

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def nonempty(self) -> torch.Tensor:
        """
        Find masks that are non-empty.

        Returns:
            Tensor: a BoolTensor which represents
                whether each mask is empty (False) or non-empty (True).
        """
        return self.tensor.flatten(1).any(dim=1)

    def get_bounding_boxes(self) -> Boxes:
        """
        Returns:
            Boxes: tight bounding boxes around bitmasks.
            If a mask is empty, it's bounding box will be all zero.
        """
        N, D, H, W = self.tensor.shape
        device     = self.tensor.device
        boxes = torch.zeros(self.tensor.shape[0], 6, dtype=torch.float32)
        
        # occupancy along each principal axis
        x_any = self.tensor.any(dim=(1, 2))   # -> (N, W)
        y_any = self.tensor.any(dim=(1, 3))   # -> (N, H)
        z_any = self.tensor.any(dim=(2, 3))   # -> (N, D)

        for idx in range(N):
            xs = torch.where(x_any[idx])[0]
            ys = torch.where(y_any[idx])[0]
            zs = torch.where(z_any[idx])[0]

            if len(xs) and len(ys) and len(zs):
                boxes[idx] = torch.tensor(
                    [xs[0], ys[0], zs[0],   # min corner
                    xs[-1] + 1, ys[-1] + 1, zs[-1] + 1],     # max corner 
                    dtype=torch.float32,
                    device=device,
                )

        return Boxes(boxes)

    @staticmethod
    def cat(bitmasks_list: List["BitMasks"]) -> "BitMasks":
        """
        Concatenates a list of BitMasks into a single BitMasks

        Arguments:
            bitmasks_list (list[BitMasks])

        Returns:
            BitMasks: the concatenated BitMasks
        """
        assert isinstance(bitmasks_list, (list, tuple))
        assert len(bitmasks_list) > 0
        assert all(isinstance(bitmask, BitMasks) for bitmask in bitmasks_list)

        cat_bitmasks = type(bitmasks_list[0])(torch.cat([bm.tensor for bm in bitmasks_list], dim=0))
        return cat_bitmasks


def project_masks_on_boxes(gt_masks, boxes, matched_idxs, M):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.
    """
    matched_idxs = matched_idxs.to(boxes)
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    gt_masks = gt_masks[:, None].to(rois)
    
    roi_align = RoIAlign3DFunction.apply

    gt_masks_gpu = gt_masks.to("cuda")
    rois_gpu = rois.to("cuda")

    result = roi_align(gt_masks_gpu, rois_gpu, (M, M, M), 1.0)[:, 0] 
    result = result.to(gt_masks.device)
    return result


def expand_masks(mask, padding):
    M = mask.shape[-1]
    scale = float(M + 2 * padding) / M
    padded_mask = F.pad(mask, (padding,) * 6)
    return padded_mask, scale


def paste_mask_in_image(mask, box, im_d, im_h, im_w):
    TO_REMOVE = 1
    w = int(box[3] - box[0] + TO_REMOVE)
    h = int(box[4] - box[1] + TO_REMOVE)
    d = int(box[5] - box[2] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)
    d = max(d, 1)

    # if box is totally outside the image, return empty mask
    if (box[3] < 0 or box[0] >= im_w or  
        box[4] < 0 or box[1] >= im_h or 
        box[5] < 0 or box[2] >= im_d): 
        return torch.zeros((im_d, im_h, im_w), dtype=mask.dtype, device=mask.device)

    # Set shape to [batchxCxDxHxW]
    mask = mask.expand((1, 1, -1, -1, -1))

    # Resize mask 
    mask = F.interpolate(mask, size=(d, h, w), mode="trilinear", align_corners=False)
    mask = mask[0][0]

    im_mask = torch.zeros((im_d, im_h, im_w), dtype=mask.dtype, device=mask.device)
    x_0 = max(box[0], 0)
    x_1 = min(box[3] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[4] + 1, im_h)
    z_0 = max(box[2], 0)
    z_1 = min(box[5] + 1, im_d)

    # if box is invalid, return empty mask (consider box[2] = 2, box[5] = 30, im_d = 25) 
    # TODO: is this fully covered by first check?
    if x_1 <= x_0 or y_1 <= y_0 or z_1 <= z_0:
        return im_mask 

    # mask[0, 0, 0] corresponds to im_mask[z0, y0, x0] hence shift 
    im_mask[z_0: z_1, y_0:y_1, x_0:x_1] = mask[(z_0 - box[2]) : (z_1 - box[2]), (y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])]
    return im_mask


def paste_masks_in_image(masks, boxes, img_shape, padding=1):
    masks, scale = expand_masks(masks, padding=padding)
    boxes = expand_boxes(boxes, scale).to(dtype=torch.int64)
    im_d, im_h, im_w = img_shape

    res = [paste_mask_in_image(m[0], b, im_d, im_h, im_w) for m, b in zip(masks, boxes)]
    if len(res) > 0:
        ret = torch.stack(res, dim=0)[:, None]
    else:
        ret = masks.new_empty((0, 1, im_d, im_h, im_w))
    return ret
