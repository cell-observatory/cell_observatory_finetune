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

from typing import List, Optional
from collections import OrderedDict

import torch
from torch import nn

from segmentation.models.meta_arch.base_model import BaseModel
from segmentation.models.meta_arch.preprocessor import PreProcessor

from segmentation.structures.data_objects.boxes import Boxes
from segmentation.structures.sample_objects.instances import Instances
from segmentation.structures.sample_objects.data_sample import DataSample


class GeneralizedRCNN(BaseModel):
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

    def __init__(self, 
                 preprocessor: PreProcessor,
                 backbone: nn.Module, 
                 rpn: nn.Module, 
                 roi_heads: nn.Module, 
                 transform: nn.Module
    ) -> None:
        super().__init__(preprocessor)
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self._has_warned = False


    def _forward(self, data_samples: DataSample):
        """
        Args:
            data_samples (DataSample): the input data sample. It contains the image tensor
                and the ground truth instances (if in training mode).

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training:
            inst = getattr(data_samples, "gt_instances", None)
            torch._assert(
                inst is not None and isinstance(inst, list) and len(inst) > 0,
                "gt_instances must be a non-empty list when in training mode",
            )
            torch._assert(
                all(isinstance(i, Instances) for i in inst),
                f"Every element of gt_instances must be `Instances`, got "
                f"{[type(i).__name__ for i in inst]}",
            )

            for idx, ins in enumerate(inst):
                boxes = ins.boxes
                torch._assert(
                    isinstance(boxes, Boxes),
                    f"gt_instances[{idx}].boxes must be `Boxes`, got {type(boxes)}",
                )

                shape = boxes.tensor.shape        
                torch._assert(
                    len(shape) == 2 and shape[1] == 6,
                    f"gt_instances[{idx}].boxes.tensor expected shape (N, 6), got {shape}",
                )

        # normalize (optional), resize (optional), pad and batch images
        data_samples = self.transform(data_samples)

        # get feature maps from backbone
        features = self.backbone(data_samples.data_tensor.tensor.contiguous())
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        
        # generate proposals from the feature maps 
        # anchor_generator uses feature maps for scaling, rpn_head uses
        # feature maps for classification & regression
        proposals, proposal_losses = self.rpn(data_samples.data_tensor, 
                                              features, 
                                              data_samples.gt_instances)
        detections, detector_losses = self.roi_heads(features, 
                                                     proposals, 
                                                     data_samples.data_tensor.image_sizes, 
                                                     data_samples.gt_instances)

        # if testing, resize boxes and masks to original image size
        detections = self.transform.postprocess(detections, 
                                                data_samples.data_tensor.image_sizes, 
                                                data_samples.data_tensor.orig_image_sizes)  

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses, detections