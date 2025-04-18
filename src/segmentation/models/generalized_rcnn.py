"""
https://github.com/pytorch/vision/blob/main/torchvision/models/detection/generalized_rcnn.py

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

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn, Tensor


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Args:
        backbone (nn.Module): backbone module that takes an image and returns a feature map.
        rpn (nn.Module): takes the feature map and returns a set of proposals.
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone: nn.Module, rpn: nn.Module, roi_heads: nn.Module, transform: nn.Module) -> None:
        super().__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False


    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 6,
                            f"Expected target boxes to be a tensor of shape [N, 6], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-3:] # BCZYX
            torch._assert(
                len(val) == 3,
                f"expecting the last three dimensions of the Tensor to be Z and Y and X instead got {img.shape[-3:]}",
            )
            original_image_sizes.append((val[0], val[1], val[2]))

        # normalize (optional), resize (optional), pad and batch images
        images, targets = self.transform(images, targets)

        # Check for degenerate boxes (NOTE: MOVED TO PREPROCESSING TO PREVENT CUDA ERROR)
        # if targets is not None:
        #     for target_idx, target in enumerate(targets):
        #         boxes = target["boxes"]
        #         degenerate_boxes = boxes[:, 3:] <= boxes[:, :3]
        #         if degenerate_boxes.any():
        #             # print the first degenerate box
        #             bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
        #             degen_bb: List[float] = boxes[bb_idx].tolist()
        #             torch._assert(
        #                 False,
        #                 "All bounding boxes should have positive z, y, and x."
        #                 f" Found invalid box {degen_bb} for target at index {target_idx}.",
        #             )

        # get feature maps from backbone
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        
        # generate proposals from the feature maps 
        # anchor_generator uses feature maps for scaling, rpn_head uses
        # feature maps for classification & regression
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        
        # import skimage
        # import numpy as np
        # from segmentation.utils.plot import plot_boxes
        # from ray.train import get_context
        # if get_context().get_world_rank() == 0:
        #     skimage.io.imsave("/clusterfs/nvme/segment_4d/test_5/masks_after_train.tif", targets[0]["masks"][0].cpu().numpy())
        #     box_test = [targets[0]["boxes"][i].cpu().numpy() for i in range(len(targets[0]["boxes"]))]
        #     plot_boxes(box_test, sample_indices=[0], image_shape=images.tensors.shape[-3:], save_path="/clusterfs/nvme/segment_4d/test_5/bx_after_full_pred.tif")
        # # raise ValueError("DEBUG")
        # # skimage.io.imsave("/clusterfs/nvme/segment_4d/test_5/test_input.tif", images.tensors[0,0].cpu().numpy())

        # if testing, resize boxes and masks to original image size
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        
        return losses, detections