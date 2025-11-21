from typing import Dict, List, Optional, Literal

import torch
from torch import nn
from torch.nn import functional as F

from cell_observatory_finetune.training.losses import DETR_Set_Loss
from cell_observatory_finetune.data.structures import box_cxcyczwhd_to_xyzxyz
from cell_observatory_finetune.models.heads.maskdino_head import MaskDINOHead
from cell_observatory_finetune.models.adapters.vit_adapter import EncoderAdapter
from cell_observatory_finetune.models.heads.pixel_decoders import MaskDINOEncoder
from cell_observatory_finetune.models.heads.maskdino_decoder import MaskDINODecoder
from cell_observatory_finetune.models.utils.matchers import HungarianMatcher


class MaskDINO(nn.Module):
    def __init__(
        self,
        # parameters for modules
        # MaskDINOEncoder (pixel decoder)
        input_shape: Dict,
        transformer_in_features: List[str],
        target_min_stride: int,
        total_num_feature_levels: int,
        transformer_encoder_dropout: float,
        transformer_encoder_num_heads: int,
        transformer_encoder_dim_feedforward: int,
        num_transformer_encoder_layers: int,
        conv_dim: int,
        mask_dim: int,
        norm: str,
        # MaskDINODecoder
        in_channels: int,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        feedforward_dim: int,
        decoder_num_layers: int,
        enforce_input_projection: bool,
        two_stage_flag: bool,
        denoise_queries_flag: bool,
        noise_scale: float,
        total_denosing_queries:int,
        initialize_box_type: Optional[Literal["bitmask", "mask2box"]],
        with_initial_prediction: bool,
        learn_query_embeddings: bool,
        dropout: float,
        activation: str,
        num_heads: int,
        decoder_num_points: int,
        return_intermediates_decoder: bool,
        query_dim: int,
        share_decoder_layers: bool,
        # matchers
        cost_classification: float, 
        cost_mask: float, 
        cost_mask_dice: float, 
        num_points: int,
        cost_box: float, 
        cost_box_giou: float, 
        # backbone
        backbone: nn.Module,
        # backbone adapter
        with_backbone_adapter: bool,
        dim: int,
        adapter_in_channels: int,
        backbone_embed_dim: int,
        input_format: str,
        dtype: str,
        patch_shape: tuple[int, int, int],
        interaction_indexes: list[int],
        add_vit_feature: bool,
        conv_inplane: int,
        use_deform_attention: bool,
        n_points: int,
        deform_num_heads: int,
        drop_path_rate: float,
        init_values: float,
        with_cffn: bool,
        cffn_ratio: float,
        deform_ratio: float,
        use_extra_extractor: bool,
        strategy: str,
        spatial_prior_module_strides: dict[str, tuple[int, int, int]],
        # criterion
        loss_weight_dict: dict, 
        no_object_loss_weight: float, 
        losses: list[str],
        oversample_ratio: float,
        importance_sample_ratio: float,
        denoise: bool,
        denoise_type: str,
        denoise_losses: list[str], 
        semantic_ce_loss: bool,
        focal_alpha: float,
        # training parameters
        instance_segmentation_flag: bool,
        topk_per_image: int,
        focus_on_boxes: bool = False,
        use_softmax_loss: bool = False
    ):
        super().__init__()

        self.backbone = backbone

        self.with_adapter = with_backbone_adapter
        if self.with_adapter:
            self.adapter = EncoderAdapter(
                dim=dim,
                in_channels=adapter_in_channels,
                backbone_embed_dim=backbone_embed_dim,
                input_format=input_format,
                dtype=dtype,
                patch_shape=patch_shape,
                interaction_indexes=interaction_indexes,
                add_vit_feature=add_vit_feature,
                conv_inplane=conv_inplane,
                use_deform_attention=use_deform_attention,
                n_points=n_points,
                deform_num_heads=deform_num_heads,
                drop_path_rate=drop_path_rate,
                init_values=init_values,
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio,
                deform_ratio=deform_ratio,
                use_extra_extractor=use_extra_extractor,
                strategy=strategy,
                spatial_prior_module_strides=spatial_prior_module_strides,
            )

        self.matcher = HungarianMatcher(
            cost_classification=cost_classification,
            cost_mask=cost_mask,
            cost_mask_dice=cost_mask_dice,
            cost_box=cost_box,
            cost_box_giou=cost_box_giou,
        )
        
        self.criterion = DETR_Set_Loss(
            num_classes=num_classes,
            matcher=self.matcher,
            weight_dict=loss_weight_dict,
            eos_coef=no_object_loss_weight,
            losses=losses,
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            denoise=denoise,
            denoise_type=denoise_type,
            denoise_losses=denoise_losses,
            semantic_ce_loss=semantic_ce_loss,
            focal_alpha=focal_alpha
        )

        pixel_decoder=MaskDINOEncoder(
            input_shape=input_shape,
            transformer_in_features=transformer_in_features,
            target_min_stride=target_min_stride,
            total_num_feature_levels=total_num_feature_levels,
            transformer_encoder_dropout=transformer_encoder_dropout,
            transformer_encoder_num_heads=transformer_encoder_num_heads,
            transformer_encoder_dim_feedforward=transformer_encoder_dim_feedforward,
            num_transformer_encoder_layers=num_transformer_encoder_layers,
            conv_dim=conv_dim,
            mask_dim=mask_dim,
            norm=norm,
        )

        decoder = MaskDINODecoder(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            feedforward_dim=feedforward_dim,
            decoder_num_layers=decoder_num_layers,
            mask_dim=mask_dim,
            enforce_input_projection=enforce_input_projection,
            two_stage_flag=two_stage_flag,
            denoise_queries_flag=denoise_queries_flag,
            noise_scale=noise_scale,
            total_denosing_queries=total_denosing_queries,
            initialize_box_type=initialize_box_type,
            with_initial_prediction=with_initial_prediction,
            learn_query_embeddings=learn_query_embeddings,
            total_num_feature_levels=total_num_feature_levels,
            dropout=dropout,
            activation=activation,
            num_heads=num_heads,
            decoder_num_points=decoder_num_points,
            return_intermediates_decoder=return_intermediates_decoder,
            query_dim=query_dim,
            share_decoder_layers=share_decoder_layers,
        )

        self.segmentation_head = MaskDINOHead(
            num_classes=num_classes,
            pixel_decoders=pixel_decoder,
            decoders=decoder,
        )

        self.num_queries = num_queries
        self.topk_per_image = topk_per_image
        self.instance_segmentation_flag = instance_segmentation_flag
        self.focus_on_boxes = focus_on_boxes
        self.use_softmax_loss = use_softmax_loss

    def forward(self, data_sample: dict):
        features = self.backbone(data_sample['data_tensor'])
        if self.with_adapter:
            features_dict = self.adapter(data_sample['data_tensor'], features)
        else:
            features_dict = features

        outputs, denoise_predictions = self.segmentation_head(features_dict, targets=data_sample['metainfo']['targets'])

        # bipartite matching-based loss
        losses = self.criterion(outputs, data_sample['metainfo']['targets'], denoise_predictions)

        for loss in list(losses.keys()):
            if loss in self.criterion.loss_weight_dict:
                losses[loss] *= self.criterion.loss_weight_dict[loss]
            else:
                # remove this loss if not specified in loss_weight_dict
                losses.pop(loss)

        return losses, outputs

    def predict(self, data_sample: dict):
        features = self.backbone(data_sample['data_tensor'])
        if self.with_adapter:
            features_dict = self.adapter(data_sample['data_tensor'], features)
        else:
            features_dict = features
            
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
            # scale postprocess boxes to original image size
            predicted_box = self.box_postprocess(predicted_box, depth, height, width)

            instance_predictions = self._predict(predicted_label, predicted_mask, predicted_box)
            predictions.append(instance_predictions)

        return predictions

    def _predict(self, predicted_labels, predicted_masks, predicted_boxes):
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