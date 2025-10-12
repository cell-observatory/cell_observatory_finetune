"""
Adapted from:
https://github.com/IDEA-Research/MaskDINO/blob/3831d8514a3728535ace8d4ecc7d28044c42dd14/maskdino/modeling/meta_arch/maskdino_head.py#L4
"""

import torch
from torch import nn
from torch.nn import functional as F

from cell_observatory_finetune.data.structures.boxes import box_cxcyczwhd_to_xyzxyz


class MaskDINOHead(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        pixel_decoders: nn.Module,
        decoders: nn.Module,
    ):
        """
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            ignore_value: category id to be ignored during training.
            decoder: the transformer decoder that makes prediction
        """
        super().__init__()
        self.num_classes = num_classes
        self.decoder = decoders
        self.pixel_decoder = pixel_decoders

    def forward(self, features, mask = None, targets = None):
        mask_features, transformer_encoder_features, \
            multi_scale_features = self.pixel_decoder.forward_features(features, mask)
        predictions = self.decoder(multi_scale_features, mask_features, mask, targets = targets)
        return predictions 


class MaskDINO(nn.Module):
    def __init__(
        self,
        # modules
        matchers: nn.Module,
        backbones: nn.Module,
        criterion: nn.Module,
        heads: nn.Module,
        # training parameters
        num_queries: int,
        size_divisibility: int,
        # inference
        instance_segmentation_flag: bool,
        topk_per_image: int,
        focus_on_boxes: bool = False,
        use_softmax_loss: bool = False
    ):
        """
        Args:
            backbone: backbone module that extracts features from input images
            segmentation_head: module that performs semantic segmentations using backbone features
            criterion: module that computes the loss
            num_queries: number of queries
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. Used to override such requirements.
            instance_segmentation_flag: bool, whether to output instance segmentation prediction
            topk_per_image: keep topk instances per image for instance segmentation
            use_softmax_loss: transform sigmoid scores into softmax scores to make scores sharper
            semantic_segmentation_ce_loss: whether use cross-entroy loss for classification
        """
        super().__init__()

        self.matcher = matchers
        self.backbone = backbones
        self.criterion = criterion
        self.segmentation_head = heads

        self.num_queries = num_queries
        
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility

        self.topk_per_image = topk_per_image
        self.instance_segmentation_flag = instance_segmentation_flag

        self.focus_on_boxes = focus_on_boxes
        self.use_softmax_loss = use_softmax_loss

    def _forward(self, data_sample: dict):
        """
        Args:
            inputs: Tensor, image in (C, D, H, W) format
            targets: Dict, per-region ground truth instances
        
        Returns:
            list[dict]: Results for one image.        
        """
        last_feature_map, features_dict = self.backbone(data_sample['data_tensor'])

        if self.training:
            outputs, denoise_predictions = self.segmentation_head(features_dict, 
                                                                  targets=data_sample['metainfo']['gt_instances'])

            # bipartite matching-based loss
            losses = self.criterion(outputs, data_sample['metainfo']['gt_instances'], denoise_predictions)

            for loss in list(losses.keys()):
                if loss in self.criterion.loss_weight_dict:
                    losses[loss] *= self.criterion.loss_weight_dict[loss]
                else:
                    # remove this loss if not specified in loss_weight_dict
                    losses.pop(loss)

            return losses, outputs
        
        else:
            outputs, _ = self.segmentation_head(features_dict, targets=None)

            predicted_labels, predicted_boxes, predicted_masks = [
                outputs[key] for key in ("pred_logits", "pred_boxes", "pred_masks")
            ]

            # upsample masks to original image size
            predicted_masks = F.interpolate(
                predicted_masks,
                size=(data_sample['metainfo']['image_sizes'][0]),
                mode="trilinear",
                align_corners=False,
            )

            del outputs

            predictions = []
            for predicted_label, predicted_mask, predicted_box, image_size_pad, orig_image_size in zip(
                predicted_labels, predicted_masks, predicted_boxes, 
                data_sample['metainfo']['image_sizes'], data_sample['metainfo']['orig_image_sizes']
            ):
                # padded size (divisible by 32)
                depth, height, width = [
                    new_dim/image_dim_pad * orig_dim
                    for new_dim, image_dim_pad, orig_dim in zip(
                        predicted_mask.shape[-3:],  # (new_d, new_h, new_w)
                        image_size_pad, # (orig_d, orig_h, orig_w)
                        orig_image_size # (orig_d, orig_h, orig_w)
                    )
                ]
                predicted_box = self.box_postprocess(predicted_box, depth, height, width)
                
                instance_predictions = self.inference(predicted_label, predicted_mask, predicted_box)
                predictions.append(instance_predictions) 
            
            return None, predictions

    def inference(self, predicted_labels, predicted_masks, predicted_boxes):
        # (num_queries, num_classes) -> (num_queries, num_classes)
        predicted_labels = predicted_labels.sigmoid()
        
        # (num_queries, num_classes) -> (num_queries * num_classes,) -> Tuple(topk predicted labels, indices)
        predicted_labels_topk, topk_indices = predicted_labels.flatten(0, 1).topk(self.topk_per_image, sorted=False)
        
        # recover which query (0...Q-1) each top-K came from
        # flattened index is q*C + c => integer-dividing by C retrieves q
        topk_query_indices = topk_indices // self.segmentation_head.num_classes

        predicted_masks = predicted_masks[topk_query_indices]        

        instance_predictions = {}
        # predicted masks pre-sigmoid
        instance_predictions['masks'] = (predicted_masks > 0).float()
        instance_predictions['boxes'] = predicted_boxes[topk_query_indices]

        # average mask confidence inside each mask
        predicted_masks_flattened = instance_predictions['masks'].flatten(1)
        predicted_masks_sigmoid_flattened = predicted_masks.sigmoid().flatten(1)
        mask_confidence_score = (predicted_masks_sigmoid_flattened * predicted_masks_flattened).sum(1) \
            / (predicted_masks_flattened.sum(1) + 1e-6)
        
        if self.focus_on_boxes:
            instance_predictions['predicted_labels'] = predicted_labels_topk
        else:
            instance_predictions['predicted_labels'] = predicted_labels_topk * mask_confidence_score

        return instance_predictions

    def box_postprocess(self, bboxes, depth, height, width):
        # postprocess box height and width
        scale_factor = torch.tensor([width, height, depth, width, height, depth,])
        scale_factor = scale_factor.to(bboxes)
        bboxes = box_cxcyczwhd_to_xyzxyz(bboxes)
        bboxes = bboxes * scale_factor
        return bboxes