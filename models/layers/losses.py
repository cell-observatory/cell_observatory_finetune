"""
https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py
https://github.com/IDEA-Research/MaskDINO/blob/3831d8514a3728535ace8d4ecc7d28044c42dd14/maskdino/modeling/criterion.py#L139

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


import torch
from torch import nn
import torch.nn.functional as F

from finetune.models.layers.utils import batch_tensors
from finetune.data.structures.data_objects import boxes
from finetune.utils.comm import get_world_size, in_torch_dist
from finetune.data.structures.data_objects.masks import project_masks_on_boxes
from finetune.models.layers.utils import point_sample, get_uncertain_point_coords_with_randomness


# --------------------------------------------- GENERAL LOSSES ---------------------------------------------


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # p_t is large for hard examples where the model mispredicts
    # i.e. prob is close to 0 for targets=1 or close to 1 for targets=0
    # the degree of loss modulation is controlled by gamma 
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        # upweight/downweight positive vs negative examples
        # if alpha is close to 1, the loss will be more focused 
        # on positive examples and vice versa
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    # dice loss is 1 - Dice_Coeff 
    # dice_coeff = 2 x (object overlap) / (sum of pixels in both masks)
    # hence loss is smaller for larger overlap / IOU
    inputs = inputs.sigmoid()
    # masks: (N, D, H, W) -> (N, D*H*W)
    inputs = inputs.flatten(1)
    # (N,D*H*W)x(N,D*H*W)->(N,D*H*W)->(N,) 
    numerator = 2 * (inputs * targets).sum(-1)
    # (N,) + (N,) -> (N,)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).

    Returns:
        Loss tensor
    """
    # binary_cross_entropy_with_logits returns: (num_masks, num_pixels)
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # (num_masks, num_pixels) -> (num_masks,) -> loss / num_masks
    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


# --------------------------------------------- DETR TYPE MODEL LOSSES ---------------------------------------------


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
    foreground class.
    
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class DETR_Set_Loss(nn.Module):
    """
    This class computes the loss for DETR.

    The process happens in two steps:
        1) Compute hungarian assignment between ground truth boxes and the outputs of the model
        2) Supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, 
                 num_classes, 
                 matcher, 
                 loss_weight_dict, 
                 no_object_loss_weight, 
                 losses,
                 num_points, 
                 oversample_ratio, 
                 importance_sample_ratio,
                 denoise: bool = False,
                 # TODO: make Enum for denoise_type
                 denoise_type: str = "seg",
                 denoise_losses = [], 
                 semantic_ce_loss = True,
                 focal_alpha: float = 0.25
                 ):
        """
        Args:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module that computes matching between targets and proposals
            loss_weight_dict: dict given by {loss names : relative weights}
            no_object_loss_weight: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied
        """
        super().__init__()

        self.matcher = matcher
        self.num_classes = num_classes
        self.loss_weight_dict = loss_weight_dict

        self.denoise_type = denoise_type
        
        self.losses = losses
        self.no_object_loss_weight = no_object_loss_weight
        
        self.denoise = denoise
        self.denoise_losses = denoise_losses
        
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.no_object_loss_weight
        
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        
        self.focal_alpha = focal_alpha
        self.semantic_ce_loss = semantic_ce_loss

    def loss_labels_ce(self, outputs, targets, indices):
        """
            Classification Loss: Cross Entropy Loss
        """
        # model predictions: (B, num_queries, num_classes)
        source_logits = outputs["pred_logits"].float()

        # idx is a tuple (batch_idx, src_idx), batch_idx is the index of the batch 
        # for a given set of matched source and target indices
        # hence idx = (B, num_queries) where each element is batch idx, source query 
        query_indices = self._get_query_indices(indices)
        
        # get the labels for all targets that were matched to the source indices
        # indices is a list of tuples (src_idx, tgt_idx) where src_idx are the indices of
        # the source boxes that were matched to the target boxes, and similar for tgt_idx 
        target_labels = torch.cat([target.labels.tensor[matched_target_idx] for target, (_, matched_target_idx) in zip(targets, indices)])
        # make tensor (B, num_queries) with values equal to num_classes for all elements
        target_classes = torch.full(
            source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device
        )
        # for each (batch_idx, source_idx) tuple we set the corresponding target label
        # in target_classes to the value of target_labels, thus target_classes
        # is now of the form (batch_idx, num_queries) = matched target label
        target_classes[query_indices] = target_labels

        # compute cross entropy loss between source logits (B, num_queries, num_classes) and target_classes (B, num_queries)
        loss_ce = F.cross_entropy(source_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {"loss_ce": loss_ce}

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """
            Classification loss: Binary Focal Loss
        """
        source_logits = outputs['pred_logits']
        query_indices = self._get_query_indices(indices)

        target_labels = torch.cat([target.labels.tensor[matched_target_idx] for target, (_, matched_target_idx) in zip(targets, indices)])
        target_classes = torch.full(source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device)
        target_classes[query_indices] = target_labels

        # create one-hot encoding of target classes, add 1 extra dimension for the no-object class
        target_classes_onehot = torch.zeros([source_logits.shape[0], source_logits.shape[1], source_logits.shape[2]+1],
                                            dtype=source_logits.dtype, layout=source_logits.layout, device=source_logits.device)
        # scatter_ to write 1s in the correct class slot for each query (we write to channel dim, i.e. dim=2)
        # we need index tensor of shape (B, Q, 1) to tell scatter_ where to put the 1
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        # drop the last channel since focal loss is computed over the 
        # real object classes
        target_classes_onehot = target_classes_onehot[:,:,:-1]

        loss_ce = sigmoid_focal_loss(source_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * source_logits.shape[1]
        return {'loss_ce': loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
            Compute L1 regression loss and the GIoU loss over bounding box coordinates.
            Target boxes are expected in format (center_x, center_y, center_z, w, h, d), normalized by the image size.
        """
        query_indices = self._get_query_indices(indices)
        source_boxes = outputs['pred_boxes'][query_indices]
        target_boxes = torch.cat([target.boxes.tensor[matched_target_idx] for target, (_, matched_target_idx) in zip(targets, indices)], dim=0)

        losses = {}

        # L1 loss over bounding box corordinates, normalized by nr. of boxes
        loss_bbox = F.l1_loss(source_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # generalized IoU loss over bounding box corordinates, normalized by nr. of boxes
        # generalized box IOU (https://giou.stanford.edu/): 
        # 1. get intersection volume for boxes A,B
        # 2. get union volume for boxes A,B
        # 3. find the smallest box that encloses both A and B (convex hull)
        # 4. compute GIOU = IoU−|C∖(A∪B)||C| where C is the convex hull
        # generalized_box_iou call returns an MxM matrix comparing every source against every target
        # we hence take the diagonal to get the IoU for each matched pair
        loss_giou = 1 - torch.diag(boxes.generalized_box_iou(
            boxes.box_cxcyczwhd_to_xyzxyz(source_boxes),
            boxes.box_cxcyczwhd_to_xyzxyz(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """
            Compute mask loss: Focal Loss and Dice Loss.
        """
        query_indices = self._get_query_indices(indices)
        target_class_indices = self._get_target_class_indices(indices)

        source_masks = outputs["pred_masks"][query_indices]
        masks = [target.masks.tensor for target in targets]

        # TODO: use valid to mask invalid areas due to padding in loss
        target_masks, valid = batch_tensors(masks)
        target_masks = target_masks.to(source_masks)
        target_masks = target_masks[target_class_indices]

        # no need to upsample predictions as we are using normalized coordinates
        # source/target masks: (N, 1, D, H, W)
        source_masks = source_masks[:, None]
        target_masks = target_masks[:, None]

        # Motivated by PointRend & Implicit PointRend
        # train with mask loss calculated on K randomly
        # sampled points instead of whole mask
        with torch.no_grad():
            # sample point_coordinates
            point_coords = get_uncertain_point_coords_with_randomness(
                source_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points, # K
                self.oversample_ratio,
                # ratio of points that are sampled via importance sampling
                self.importance_sample_ratio,
            )
            # samples from target mask at point_coords
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)
            
        # samples from source mask at point_coords
        point_logits = point_sample(
            source_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        # compute losses: cross entropy classifcation loss and dice mask loss 
        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del source_masks, target_masks
        return losses

    def preprocess_masks(self, mask_dict):
        predicted_denoise_bboxes, denoise_target_indices = mask_dict['predicted_denoise_bboxes'], mask_dict['denoise_target_indices']
        max_query_pad_size, denoise_queries_per_label = mask_dict['max_query_pad_size'], mask_dict['denoise_queries_per_label']
        query_pad_size_per_label = max_query_pad_size // denoise_queries_per_label
        num_targets = denoise_target_indices.numel()
        return predicted_denoise_bboxes, num_targets, query_pad_size_per_label, denoise_queries_per_label

    def _get_query_indices(self, indices):
        batch_indices = torch.cat([torch.full_like(source_idx, i) for i, (source_idx, target_idx) in enumerate(indices)])
        source_indices = torch.cat([source_idx for (source_idx, target_idx) in indices])
        return batch_indices, source_indices

    def _get_target_class_indices(self, indices):
        batch_indices = torch.cat([torch.full_like(target_idx, i) for i, (source_idx, target_idx) in enumerate(indices)])
        target_indices = torch.cat([target_idx for (source_idx, target_idx) in indices])
        return batch_indices, target_indices

    def compute_loss(self, loss, outputs, targets, indices, num_masks):
        loss_dict = {
            'labels': self.loss_labels_ce if self.semantic_ce_loss else self.loss_labels,
            'masks': self.loss_masks,
            'boxes': self.loss_boxes,
        }
        return loss_dict[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets, denoise_predictions = None):
        outputs_without_aux_data = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}

        # compute loss for denoising and mask predictions
        if self.denoise and denoise_predictions is not None:
            predicted_denoise_bboxes, num_targets, query_pad_size_per_label, denoise_queries_per_label = self.preprocess_masks(denoise_predictions)
            denoise_query_target_indices = []
            for target in targets:
                # we have L target labels and for each label we create denoise_queries_per_label queries
                # hence we have L * denoise_queries_per_label queries in total, here we get their indices
                # however, in decoder each batch element has a different number of target labels, so we need to pad the indices
                # to the max number of denoise queries per label, which is what we do  
                target_labels = target.labels
                if len(target_labels) > 0:
                    # (num_target_labels, ) = [0,1,...,num_target_labels-1]
                    target_label_indices = torch.arange(0, len(target_labels)).long().cuda()
                    # (1, num_target_labels) -> (denoise_queries_per_label, num_target_labels)
                    # hence we get [[0, 1, ..., num_target_labels-1], ....]
                    # NOTE: repeat(A,B,...) repeats 0th dim A times and 1st dim B times etc.  
                    target_label_indices = target_label_indices.unsqueeze(0).repeat(denoise_queries_per_label, 1)
                    # (denoise_queries_per_label, num_target_labels) -> (denoise_queries_per_label * num_target_labels, )
                    denoise_query_target_index = target_label_indices.flatten()
                    # build output indices into the predicted denoise queries:
                    # torch.arange(R) gives [0,1,...,R−1], shape: (denoise_queries_per_label,)
                    # multiply by pad_size P to get [r*P, r*P, ..., r*P], start offset for each row of queries
                    # unsqueeze -> (R,1), then broadcast-add t row r becomes [r*P + 0, r*P + 1, ..., r*P + (num_target_labels-1)]]
                    padded_denoise_query_target_index = (torch.tensor(range(denoise_queries_per_label)) * query_pad_size_per_label).long().cuda().unsqueeze(1)
                    padded_denoise_query_target_index = padded_denoise_query_target_index  + target_label_indices
                    padded_denoise_query_target_index = padded_denoise_query_target_index.flatten()
                else:
                    padded_denoise_query_target_index = denoise_query_target_index = torch.tensor([]).long().cuda()
                denoise_query_target_indices.append((padded_denoise_query_target_index, denoise_query_target_index))

        # use Hungarian matcher to compute the indices of the matched predictions and targets
        matched_target_indices = self.matcher(outputs_without_aux_data, targets)

        # compute number of target boxes accross all nodes for normalization
        total_num_masks = sum(len(target.labels) for target in targets)
        total_num_masks = torch.as_tensor(
            [total_num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        # TODO: perform dist init and available check with Ray instead?
        if in_torch_dist():
            torch.distributed.all_reduce(total_num_masks)
        average_num_masks_per_node = torch.clamp(total_num_masks / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.compute_loss(loss, outputs, targets, matched_target_indices, average_num_masks_per_node))

        # compute denosing losses if denoise is enabled
        if self.denoise and denoise_predictions is not None:
            extra_losses = {}
            for loss in self.denoise_losses:
                extra_losses.update(self.compute_loss(loss, 
                                                      predicted_denoise_bboxes, 
                                                      targets, 
                                                      denoise_query_target_indices, 
                                                      average_num_masks_per_node * denoise_queries_per_label)
                                                      )
            extra_losses = {k + f'_denoise': v for k, v in extra_losses.items()}
            losses.update(extra_losses)
        
        # compute loss for denoising
        elif self.denoise:
            extra_losses = dict()            
            extra_losses['loss_bbox_denoise'] = torch.as_tensor(0.).to('cuda')
            extra_losses['loss_giou_denoise'] = torch.as_tensor(0.).to('cuda')
            extra_losses['loss_ce_denoise'] = torch.as_tensor(0.).to('cuda')
            if self.denoise_type == "seg":
                extra_losses['loss_mask_denoise'] = torch.as_tensor(0.).to('cuda')
                extra_losses['loss_dice_denoise'] = torch.as_tensor(0.).to('cuda')
            
            losses.update(extra_losses)

        # in case of auxiliary losses, we repeat loss computation with the output of intermediate layers
        if "auxiliary_outputs" in outputs:
            first_auxiliary_output_idx = 0 if 'intermediate_outputs' in outputs else 1
            for i, auxiliary_output in enumerate(outputs["auxiliary_outputs"]):
                # hungarian matcher to get indices of the matched auxiliary_outputs and targets
                auxiliary_matched_target_indices = self.matcher(auxiliary_output, targets)
                for loss in self.losses:
                    extra_losses = self.compute_loss(loss, auxiliary_output, targets, auxiliary_matched_target_indices, average_num_masks_per_node)
                    extra_losses = {k + f"_{i}": v for k, v in extra_losses.items()}
                    losses.update(extra_losses)
                                
                if i >= first_auxiliary_output_idx:
                    if self.denoise and denoise_predictions is not None:
                        auxiliary_predicted_denoise_bboxes = predicted_denoise_bboxes['auxiliary_outputs'][i]
                        extra_losses = {}
                        for loss in self.denoise_losses:
                            extra_losses.update(self.compute_loss(loss, 
                                                                  auxiliary_predicted_denoise_bboxes, 
                                                                  targets, 
                                                                  denoise_query_target_indices, 
                                                                  average_num_masks_per_node * denoise_queries_per_label))
                        extra_losses = {k + f'_denoise_{i}': v for k, v in extra_losses.items()}
                        losses.update(extra_losses)
                    
                    elif self.denoise:
                        extra_losses = dict()
                        extra_losses[f'loss_bbox_denoise_{i}'] = torch.as_tensor(0.).to('cuda')
                        extra_losses[f'loss_giou_denoise_{i}'] = torch.as_tensor(0.).to('cuda')
                        extra_losses[f'loss_ce_denoise_{i}'] = torch.as_tensor(0.).to('cuda')
                        if self.denoise_type == "seg":
                            extra_losses[f'loss_mask_denoise_{i}'] = torch.as_tensor(0.).to('cuda')
                            extra_losses[f'loss_dice_denoise_{i}'] = torch.as_tensor(0.).to('cuda')
                        
                        losses.update(extra_losses)
        
        if 'intermediate_outputs' in outputs:
            intermediate_outputs = outputs['intermediate_outputs']
            intermediate_matched_target_indices = self.matcher(intermediate_outputs, targets)
            for loss in self.losses:
                extra_losses = self.compute_loss(loss, intermediate_outputs, targets, intermediate_matched_target_indices, average_num_masks_per_node)
                extra_losses = {k + f'_intermediate': v for k, v in extra_losses.items()}
                losses.update(extra_losses)

        return losses


# ---------------------------------------------  ----------------- RCNN LOSSES ----------------- ---------------------------------------------


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor): 
        box_regression (Tensor): 
        labels (list[BoxList]): 
        regression_targets (Tensor): 

    Returns:
        classification_loss (Tensor): 
        box_loss (Tensor): 
    """
    labels = torch.cat(labels, dim=0) # (N, )
    regression_targets = torch.cat(regression_targets, dim=0) # (N,6)

    classification_loss = F.cross_entropy(class_logits, labels) 

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels
    sampled_pos_inds_subset = torch.where(labels > 0)[0] 
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 6, 6) # (N, Kx6) -> (N, K, 6) for K classes

    # class-aware loss (i.e. predictions supervised for the correct class)
    box_loss = F.smooth_l1_loss(
        # get regression preds for correct class
        box_regression[sampled_pos_inds_subset, labels_pos],   
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9, 
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


def maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
    """
    Args:
        proposals (list[BoxList])
        mask_logits (Tensor)
        targets (list[BoxList])

    Return:
        mask_loss (Tensor): scalar tensor containing the loss
    """
    discretization_size = mask_logits.shape[-1] # mask: (M, M, M)
    # get the indices of the positive proposals (correct class)
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]

    # match gt masks with size of proposal logits (uses ROIAlign under the hood)
    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size) for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]

    labels = torch.cat(labels, dim=0) # (N,)
    mask_targets = torch.cat(mask_targets, dim=0) # (N, M, M, M)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    # class-aware loss (i.e. predictions supervised for the correct class)
    mask_loss = F.binary_cross_entropy_with_logits(
        mask_logits[torch.arange(labels.shape[0], device=labels.device), labels], mask_targets
    )
    return mask_loss


# ---------------------------------------------  ----------------- MAE LOSSES ----------------- ---------------------------------------------


class MAELoss(nn.Module):
    """
    Mean Absolute Error (MAE) loss for masked autoencoders.
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        # TODO: remove probably, since we are 
        #       not using others most likely
        self.reduction = reduction

    def forward(
        self,
        preds: torch.Tensor, # (B, N, C)
        targets: torch.Tensor, # (B, N, C)
        # (B, N) or (B, N, C) or None
        # (B, N, C) for channel prediction
        # since masking is accross 
        # some channels x patches
        # (B, N) for temporal upsampling
        # for all patches for some time steps 
        # None for spatial upsampling, 
        # denoising, and channel splitting
        # since we are predicting full
        # image i.e. all patches
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # (B, N, C) -> (B, N, C)
        diff = (preds.float() - targets.float()).pow(2)

        # no masking
        if mask is None:
            per_patch = diff.mean(dim=-1)
            loss = per_patch.mean()
            return loss

        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        assert mask.shape == diff.shape[:-1] + (diff.shape[-1] if mask.dim() == 3 else 1)

        # broadcasts mask if needed
        diff.mul_(mask)

        # TODO: for channel predict 
        #       do we want to do normalization
        #       accross all elements or rather 
        #       average across all patches
        #       notwithstanding that technically
        #       not all channels for a masked patched
        #       are predicted
        per_patch = diff.mean(dim=-1)
        num = mask.sum()
        loss = per_patch.sum() / num

        return loss


# ---------------------------------------------  ----------------- ----------------- ---------------------------------------------