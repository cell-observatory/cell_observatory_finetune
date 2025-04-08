"""

https://github.com/pytorch/vision/blob/309bd7a1512ad9ff0e9729fbdad043cb3472e4cb/torchvision/models/detection/_utils.py

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
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class BalancedPositiveNegativeSampler:
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image: int, positive_fraction: float) -> None:
        """
        Args:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentage of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Args:
            matched_idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image in matched_idxs:
            positive = torch.where(matched_idxs_per_image >= 1)[0]
            negative = torch.where(matched_idxs_per_image == 0)[0]

            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)

            # randomly select positive and negative examples
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)
            neg_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)

            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx


@torch.jit._script_if_tracing
def encode_boxes(reference_boxes: Tensor, proposals: Tensor, weights: Tensor) -> Tensor:
    """
    Encode a set of proposals with respect to some
    reference boxes

    Args:
        reference_boxes (Tensor): reference boxes
        proposals (Tensor): boxes to be encoded
        weights (Tensor[4]): the weights for ``(x, y, z, w, h, d)``
    """

    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    wz = weights[2]
    ww = weights[3]
    wh = weights[4]
    wd = weights[5]

    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_z1 = proposals[:, 2].unsqueeze(1)

    proposals_x2 = proposals[:, 3].unsqueeze(1)
    proposals_y2 = proposals[:, 4].unsqueeze(1)
    proposals_z2 = proposals[:, 5].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_z1 = reference_boxes[:, 2].unsqueeze(1)

    reference_boxes_x2 = reference_boxes[:, 3].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 4].unsqueeze(1)
    reference_boxes_z2 = reference_boxes[:, 5].unsqueeze(1)

    # implementation starts here
    # encoding scheme: log-scale diff in box widhts & rel diff (arithmetic) in box centers (w. weights)
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    ex_depths = proposals_z2 - proposals_z1

    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights
    ex_ctr_z = proposals_z1 + 0.5 * ex_depths

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_depths = reference_boxes_z2 - reference_boxes_z1

    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights
    gt_ctr_z = reference_boxes_z1 + 0.5 * gt_depths

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dz = wz * (gt_ctr_z - ex_ctr_z) / ex_depths

    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)
    targets_dd = wd * torch.log(gt_depths / ex_depths)

    targets = torch.cat((targets_dx, targets_dy, targets_dz, targets_dw, targets_dh, targets_dd), dim=1)
    return targets


class BoxCoder:
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(
        self, weights: Tuple[float, float, float, float, float, float], bbox_xform_clip: float = math.log(1000.0 / 16)
    ) -> None:
        """
        Args:
            weights (6-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes: List[Tensor], proposals: List[Tensor]) -> List[Tensor]:
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)
        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_image, 0)

    def encode_single(self, reference_boxes: Tensor, proposals: Tensor) -> Tensor:
        """
        Encode a set of proposals with respect to some
        reference boxes

        Args:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes, proposals, weights)

        return targets

    def decode(self, rel_codes: Tensor, boxes: List[Tensor]) -> Tensor:
        torch._assert(
            isinstance(boxes, (list, tuple)),
            "This function expects boxes of type list or tuple.",
        )
        torch._assert(
            isinstance(rel_codes, torch.Tensor),
            "This function expects rel_codes of type torch.Tensor.",
        )
        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val
        if box_sum > 0:
            rel_codes = rel_codes.reshape(box_sum, -1)
        pred_boxes = self.decode_single(rel_codes, concat_boxes)
        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 6)
        return pred_boxes

    def decode_single(self, rel_codes: Tensor, boxes: Tensor) -> Tensor:
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Args:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """

        boxes = boxes.to(rel_codes.dtype)

        widths = boxes[:, 3] - boxes[:, 0]
        heights = boxes[:, 4] - boxes[:, 1]
        depths = boxes[:, 5] - boxes[:, 2]
        
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        ctr_z = boxes[:, 2] + 0.5 * depths

        wx, wy, wz, ww, wh, wd = self.weights
        dx = rel_codes[:, 0::6] / wx
        dy = rel_codes[:, 1::6] / wy
        dz = rel_codes[:, 2::6] / wz

        dw = rel_codes[:, 3::6] / ww
        dh = rel_codes[:, 4::6] / wh
        dd = rel_codes[:, 5::6] / wd

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)
        dd = torch.clamp(dd, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_ctr_z = dz * depths[:, None] + ctr_z[:, None]

        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]
        pred_d = torch.exp(dd) * depths[:, None]

        # Distance from center to box's corner.
        c_to_c_h = torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        c_to_c_w = torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        c_to_c_d = torch.tensor(0.5, dtype=pred_ctr_z.dtype, device=pred_d.device) * pred_d

        pred_boxes1 = pred_ctr_x - c_to_c_w
        pred_boxes2 = pred_ctr_y - c_to_c_h
        pred_boxes3 = pred_ctr_z - c_to_c_d

        pred_boxes4 = pred_ctr_x + c_to_c_w
        pred_boxes5 = pred_ctr_y + c_to_c_h
        pred_boxes6 = pred_ctr_z + c_to_c_d

        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4, pred_boxes5, pred_boxes6), dim=2).flatten(1)
        return pred_boxes


class Matcher:
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.

    Matching is based on the MxN match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.

    The matcher returns a tensor of size N containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    """

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    __annotations__ = {
        "BELOW_LOW_THRESHOLD": int,
        "BETWEEN_THRESHOLDS": int,
    }

    def __init__(self, high_threshold: float, low_threshold: float, allow_low_quality_matches: bool = False) -> None:
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        torch._assert(low_threshold <= high_threshold, "low_threshold should be <= high_threshold")
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix: Tensor) -> Tensor:
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError("No ground-truth boxes available for one of the images during training")
            else:
                raise ValueError("No proposal boxes available for one of the images during training")

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None  # type: ignore[assignment]

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (matched_vals < self.high_threshold)
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            if all_matches is None:
                torch._assert(False, "all_matches should not be None")
            else:
                self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches: Tensor, all_matches: Tensor, match_quality_matrix: Tensor) -> None:
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        # For each gt, find the prediction with which it has the highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find the highest quality match available, even if it is low, including ties
        gt_pred_pairs_of_highest_quality = torch.where(match_quality_matrix == highest_quality_foreach_gt[:, None])
        # Example gt_pred_pairs_of_highest_quality:
        # (tensor([0, 1, 1, 2, 2, 3, 3, 4, 5, 5]),
        #  tensor([39796, 32055, 32070, 39190, 40255, 40390, 41455, 45470, 45325, 46390]))
        # Each element in the first tensor is a gt index, and each element in second tensor is a prediction index
        # Note how gt items 1, 2, 3, and 5 each have two ties

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]


@torch.jit.unused
def _fake_cast_onnx(v: Tensor) -> int:
    return v  # type: ignore[return-value]


def _topk_min(input: Tensor, orig_kval: int, axis: int) -> int:
    """
    ONNX spec requires the k-value to be less than or equal to the number of inputs along
    provided dim. Certain models use the number of elements along a particular axis instead of K
    if K exceeds the number of elements along that axis. Previously, python's min() function was
    used to determine whether to use the provided k-value or the specified dim axis value.

    However, in cases where the model is being exported in tracing mode, python min() is
    static causing the model to be traced incorrectly and eventually fail at the topk node.
    In order to avoid this situation, in tracing mode, torch.min() is used instead.

    Args:
        input (Tensor): The original input tensor.
        orig_kval (int): The provided k-value.
        axis(int): Axis along which we retrieve the input size.

    Returns:
        min_kval (int): Appropriately selected k-value.
    """
    if not torch.jit.is_tracing():
        return min(orig_kval, input.size(axis))
    axis_dim_val = torch._shape_as_tensor(input)[axis].unsqueeze(0)
    min_kval = torch.min(torch.cat((torch.tensor([orig_kval], dtype=axis_dim_val.dtype), axis_dim_val), 0))
    return _fake_cast_onnx(min_kval)


# def _box_loss(
#     type: str,
#     box_coder: BoxCoder,
#     anchors_per_image: Tensor,
#     matched_gt_boxes_per_image: Tensor,
#     bbox_regression_per_image: Tensor,
#     cnf: Optional[Dict[str, float]] = None,
# ) -> Tensor:
#     torch._assert(type in ["l1", "smooth_l1", "ciou", "diou", "giou"], f"Unsupported loss: {type}")

#     if type == "l1":
#         target_regression = box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
#         return F.l1_loss(bbox_regression_per_image, target_regression, reduction="sum")
#     elif type == "smooth_l1":
#         target_regression = box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
#         beta = cnf["beta"] if cnf is not None and "beta" in cnf else 1.0
#         return F.smooth_l1_loss(bbox_regression_per_image, target_regression, reduction="sum", beta=beta)
#     else:
#         bbox_per_image = box_coder.decode_single(bbox_regression_per_image, anchors_per_image)
#         eps = cnf["eps"] if cnf is not None and "eps" in cnf else 1e-7
#         if type == "ciou":
#             return complete_box_iou_loss(bbox_per_image, matched_gt_boxes_per_image, reduction="sum", eps=eps)
#         if type == "diou":
#             return distance_box_iou_loss(bbox_per_image, matched_gt_boxes_per_image, reduction="sum", eps=eps)
#         # otherwise giou
#         return generalized_box_iou_loss(bbox_per_image, matched_gt_boxes_per_image, reduction="sum", eps=eps)