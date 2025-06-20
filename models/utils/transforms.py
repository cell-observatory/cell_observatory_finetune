"""
https://github.com/pytorch/vision/blob/ef4718ad85dab0a3694b0c3f740f46ab891f50cc/torchvision/models/detection/transform.py#L86

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


import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor

from cell_observatory_finetune.data.structures.sample_objects.instances import Instances
from cell_observatory_finetune.data.structures.sample_objects.data_sample import DataSample

from cell_observatory_finetune.data.structures.data_objects.image_list import ImageList
from cell_observatory_finetune.data.structures.data_objects.masks import paste_masks_in_image


def _resize_image_and_masks(
    image: Tensor,
    self_min_size: int,
    self_max_size: int,
    target: Optional[Dict[str, Tensor]] = None,
    fixed_size: Optional[Tuple[int, int, int]] = None,
) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
    im_shape = image.shape[-3:]

    size: Optional[List[int]] = None
    scale_factor: Optional[float] = None
    recompute_scale_factor: Optional[bool] = None
    
    if fixed_size is not None:
        size = [fixed_size[2], fixed_size[1], fixed_size[0]]
    else:
        min_size = min(im_shape)
        max_size = max(im_shape)
        scale_factor = min(self_min_size / min_size, self_max_size / max_size)

        recompute_scale_factor = True

    image = torch.nn.functional.interpolate(
        image[None],
        size=size,
        scale_factor=scale_factor,
        mode="trilinear",
        recompute_scale_factor=recompute_scale_factor,
        align_corners=False,
    )[0]

    if target is None:
        return image, target

    if "masks" in target:
        mask = target["masks"]
        mask = torch.nn.functional.interpolate(
            mask[:, None].float(), 
            size=size, 
            scale_factor=scale_factor, 
            recompute_scale_factor=recompute_scale_factor,
            # mode="nearest",
        )[:, 0].byte()
        target["masks"] = mask
    return image, target


class GeneralizedRCNNTransform(nn.Module):
    """
    Performs input / target transformation before feeding the data to a GeneralizedRCNN
    model.

    The transformations it performs are:
        - (Optional) input normalization (mean subtraction and std division)
        - input / target resizing to match min_size / max_size

    It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
    """

    def __init__(
        self,
        min_size: int,
        max_size: int,
        image_mean: List[float],
        image_std: List[float],
        size_divisible: int = 32,
        fixed_size: Optional[Tuple[int, int, int]] = None,
        **kwargs: Any,
    ):
        super().__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        
        self.min_size = min_size
        self.max_size = max_size
        
        self.image_mean = image_mean
        self.image_std = image_std
        
        self.size_divisible = size_divisible
        self.fixed_size = fixed_size

        # flags to skip resizing and normalization
        # if handled elsewhere
        self._skip_resize = kwargs.pop("skip_resize", False)
        self._skip_normalize = kwargs.pop("skip_normalize", False)

    def forward(self, data_samples: DataSample):
        images, targets = data_samples.data_tensor.tensor, [] if self.training else None
        if self.training:
            for instances in data_samples.gt_instances:
                targets.append(instances.instances_to_dict())
        for i in range(len(images)):
            image = images[i]
            target = targets[i] if targets is not None else None

            if image.dim() != 4:
                raise ValueError(f"images is expected to be a list of 4d tensors of shape [C, D, H, W], got {image.shape}")

            # normalize and resize            
            if not self._skip_normalize:
                image = self.normalize(image)

            image, target = self.resize(image, target)

            images[i] = image
            if targets is not None and target is not None:
                targets[i] = target

        image_sizes = [img.shape[-3:] for img in images]
        # ensure that all images have the same size
        # and are divisible by size_divisible
        images = self.batch_images(images, size_divisible=self.size_divisible)
        image_sizes_list: List[Tuple[int, int, int]] = []
        for image_size in image_sizes:
            torch._assert(
                len(image_size) == 3,
                f"Input tensors expected to have in the last three elements D and H and W, instead got {image_size}",
            )
            image_sizes_list.append((image_size[0], image_size[1], image_size[2]))

        image_list = ImageList(images, 
                               image_sizes_list, 
                               orig_image_sizes=data_samples.data_tensor.orig_image_sizes)
        instances = []
        if targets is not None:
            for target, gt_instance in zip(targets, data_samples.gt_instances):
                instances_transform = Instances(metainfo=gt_instance.metainfo)
                instances_transform.dict_to_instances(target)
                instances.append(instances_transform)

        data_samples.data_tensor = image_list
        data_samples.gt_instances = instances
        
        return data_samples

    def normalize(self, image: Tensor) -> Tensor:
        if not image.is_floating_point():
            raise TypeError(
                f"Expected input images to be of floating type (in range [0, 1]), "
                f"but found type {image.dtype} instead"
            )
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        # normalize across the channel dimension
        return (image - mean[:, None, None, None]) / std[:, None, None, None] 


    def torch_choice(self, k: List[int]) -> int:
        """
        Implements `random.choice` via torch ops, so it can be compiled with
        TorchScript and we use PyTorch's RNG (not native RNG)
        """
        index = int(torch.empty(1).uniform_(0.0, float(len(k))).item())
        return k[index]


    def resize(
        self,
        image: Tensor,
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        d, h, w = image.shape[-3:]
        # in eval mode we always resize to the largest size
        # in training mode, we randomly select a size or don't resize
        # depending on the _skip_resize flag
        if self.training:
            if self._skip_resize:
                return image, target
            size = self.torch_choice(self.min_size)
        else:
            size = self.min_size[-1]
        image, target = _resize_image_and_masks(image, size, self.max_size, target, self.fixed_size)
        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (d, h, w), image.shape[-3:])
        target["boxes"] = bbox
        return image, target


    def max_by_axis(self, the_list: List[List[int]]) -> List[int]:
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes


    def batch_images(self, images: List[Tensor], size_divisible: int = 32) -> Tensor:
        # ensures that all images in the batch have the same size
        # and are divisible by size_divisible
        max_size = self.max_by_axis([list(img.shape) for img in images])
        stride = float(size_divisible)
        max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)
        max_size[3] = int(math.ceil(float(max_size[3]) / stride) * stride)

        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for i in range(batched_imgs.shape[0]):
            img = images[i]
            batched_imgs[i, : img.shape[0], : img.shape[1], : img.shape[2], :img.shape[3]].copy_(img)

        return batched_imgs


    def postprocess(
        self,
        result: List[Instances],
        image_shapes: List[Tuple[int, int, int]],
        original_image_sizes: List[Tuple[int, int, int]],
    ) -> List[Dict[str, Tensor]]:
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = resize_boxes(pred["boxes"], im_s, o_im_s)
            result[i]["boxes"] = boxes
            if "masks" in pred:
                masks = paste_masks_in_image(pred["masks"], boxes, o_im_s)
                result[i]["masks"] = masks
        return result


def resize_boxes(boxes: Tensor, original_size: List[int], new_size: List[int]) -> Tensor:
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device)
        / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_depth, ratio_height, ratio_width = ratios
    xmin, ymin, zmin, xmax, ymax, zmax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    zmin = zmin * ratio_depth
    zmax = zmax * ratio_depth

    return torch.stack((xmin, ymin, zmin, xmax, ymax, zmax), dim=1)