"""
https://github.com/pytorch/vision/blob/309bd7a1512ad9ff0e9729fbdad043cb3472e4cb/torchvision/models/detection/_utils.py#L317
https://github.com/IDEA-Research/MaskDINO/blob/main/maskdino/modeling/matcher.py

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


from scipy.optimize import linear_sum_assignment

import torch
from torch import nn
from torch import Tensor
from torch.amp import autocast

from cell_observatory_finetune.models.layers.utils import point_sample
from cell_observatory_finetune.data.structures.data_objects.boxes import generalized_box_iou, box_cxcyczwhd_to_xyzxyz
from cell_observatory_finetune.models.layers.losses import batch_sigmoid_ce_loss, batch_sigmoid_ce_loss_jit, batch_dice_loss, batch_dice_loss_jit


# ------------------------------ ------------------------------ MASKRCNN ------------------------------ ------------------------------


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


# ------------------------------ ------------------------------ DETR ------------------------------ ------------------------------


class HungarianMatcher(nn.Module):
    """
    Computes an assignment between targets and model predictions.

    For efficiency, the targets do not include the no object class. Because of this, there may be
    more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are left unmatched (and thus treated as non-objects/background).
    """

    def __init__(self, 
                 cost_classification: float = 1, 
                 cost_mask: float = 1, 
                 cost_mask_dice: float = 1, 
                 num_points: int = 0,
                 cost_box: float = 0, 
                 cost_box_giou: float = 0, 
    ):
        """
        Args:
            cost_classification: Relative weight of the classification error in the matching cost
            cost_mask: Relative weight of the focal loss in the matching cost
            cost_mask_dice: Relative weight of the dice loss in matching cost
        """
        super().__init__()
        self.cost_classification = cost_classification

        self.cost_mask = cost_mask
        self.cost_mask_dice = cost_mask_dice
        
        self.cost_box = cost_box
        self.cost_box_giou = cost_box_giou

        assert cost_classification != 0 or cost_mask != 0 or cost_mask_dice != 0, "All costs can not be 0."

        self.num_points = num_points

    @torch.no_grad()
    def forward(self, outputs, targets, costs=["cls", "box", "mask"], alpha = 0.25, gamma = 2.0):
        """
        Args:
            outputs: Dict that contains at least the following entries:
                "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                "pred_masks": Tensor of dim [batch_size, num_queries, D_pred, H_pred, W_pred] with the predicted masks

            targets: List of targets (where len(targets) = batch_size). Each target is a Dict that contains:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, D_gt, H_gt, W_gt] containing target masks

        Returns:
            List of size batch_size, containing tuples (index_i, index_j) where:
                - index_i is the indices of the selected predictions 
                - index_j is the indices of the corresponding selected targets
            For each batch element, we must have that:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["pred_logits"].shape[:2]

        matched_masks = []
        for batch_idx in range(batch_size):
            predicted_bboxes = outputs["pred_boxes"][batch_idx]
            if 'box' in costs:
                target_bboxes = targets[batch_idx].boxes.tensor
                # calculates the p-norm (p=1) distance between each pair of the
                # two collections of tensors
                cost_bbox = torch.cdist(predicted_bboxes, target_bboxes, p=1)
                # we omit constant terms in the cost function, so we can just use the negative of the generalized box iou
                cost_box_giou = -generalized_box_iou(box_cxcyczwhd_to_xyzxyz(predicted_bboxes), box_cxcyczwhd_to_xyzxyz(target_bboxes))
            else:
                cost_bbox = torch.tensor(0).to(predicted_bboxes)
                cost_box_giou = torch.tensor(0).to(predicted_bboxes)

            # predicted_logits: [num_queries, num_classes]
            predicted_logits = outputs["pred_logits"][batch_idx].sigmoid()  
            targets_labels = targets[batch_idx].labels.tensor
            
            # focal loss
            negative_cost_classification = (1 - alpha) * (predicted_logits ** gamma) * (-(1 - predicted_logits + 1e-8).log())
            positive_cost_classification = alpha * ((1 - predicted_logits) ** gamma) * (-(predicted_logits + 1e-8).log())
            cost_classification = positive_cost_classification[:, targets_labels] - negative_cost_classification[:, targets_labels]

            # compute classification cost, contrary to the loss computation, we don't use the NLL
            # but approximate it as: 1 - proba[target class]
            # since constants don't change optimization, we set cost_class = -out_prob[:, target_labels]
            if 'mask' in costs:
                # predicted_masks/target_masks: [num_queries, 1, D_pred, H_pred, W_pred]
                predicted_masks = outputs["pred_masks"][batch_idx].unsqueeze(1) 
                # NOTE: gt masks are already padded when preparing targets
                target_masks = (targets[batch_idx].masks.tensor.unsqueeze(1)).to(predicted_masks)
                
                # all masks share the same set of points for efficient matching
                # point_ccords: (1, num_points, 3)
                point_coords = torch.rand(1, self.num_points, 3, device=predicted_masks.device)
                # target_masks: (num_target_masks, num_points)
                target_masks = point_sample(
                    target_masks,
                    # repeat point_coords for each target mask in the batch
                    point_coords.repeat(target_masks.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)
                predicted_masks = point_sample(
                    predicted_masks,
                    point_coords.repeat(predicted_masks.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                with autocast(enabled=False, device_type="cuda"):
                    predicted_masks, target_masks = predicted_masks.float(), target_masks.float()
                    if predicted_masks.shape[0] == 0 or target_masks.shape[0] == 0:
                        # returns: (N, 0) if we have N predictions or (0, M) if we have M targets
                        cost_mask = batch_sigmoid_ce_loss(predicted_masks, target_masks)
                        cost_mask_dice = batch_dice_loss(predicted_masks, target_masks)
                    else:
                        cost_mask = batch_sigmoid_ce_loss_jit(predicted_masks, target_masks)
                        cost_mask_dice = batch_dice_loss_jit(predicted_masks, target_masks)

            else:
                cost_mask = torch.tensor(0).to(predicted_bboxes)
                cost_mask_dice = torch.tensor(0).to(predicted_bboxes)

            C = (
                self.cost_mask * cost_mask
                + self.cost_classification * cost_classification
                + self.cost_mask_dice * cost_mask_dice
                + self.cost_box * cost_bbox
                + self.cost_box_giou * cost_box_giou
            )
            # C: (num_queries, num_target_boxes) 
            C = C.reshape(num_queries, -1).cpu()
            matched_masks.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in matched_masks
        ]
    
# ------------------------------ ------------------------------ ------------------------------ ------------------------------