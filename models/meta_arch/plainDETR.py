"""
Adapted from:
# ------------------------------------------------------------------------
# Plain-DETR
# Copyright (c) 2023 Xi'an Jiaotong University & Microsoft Research Asia.
# Licensed under The MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from cell_observatory_finetune.training.helpers import get_clones
from cell_observatory_finetune.models.layers.layers import inverse_sigmoid

from cell_observatory_finetune.models.heads.plain_detr_backbone import build_backbone
from cell_observatory_finetune.models.heads.plain_detr_transformer import build_transformer
from cell_observatory_finetune.data.structures import box_xyzxyz_to_cxcyczwhd, delta2bbox, box_cxcyczwhd_to_xyzxyz


class PlainDETR(nn.Module):
    def __init__(
        self,
        backbone,
        backbone_embed_dim,
        transformer,
        loss,
        num_classes,
        num_feature_levels,
        aux_loss=True,
        with_box_refine=False,
        two_stage=False,
        num_queries_one2one=300,
        num_queries_one2many=0,
        mixed_selection=False,
    ):
        super().__init__()
        
        num_queries = num_queries_one2one + num_queries_one2many
        
        self.num_queries = num_queries
        self.transformer = transformer
        
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)
        
        self.num_feature_levels = num_feature_levels
        
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 24)
        elif mixed_selection:
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        self.input_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(backbone_embed_dim, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )
            ]
        )
        
        self.backbone = backbone
        self.aux_loss = aux_loss
        
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[3:], -2.0)
            # HACK: iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[3:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        if two_stage:
            # HACK: for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[3:], 0.0)

        self.num_queries_one2one = num_queries_one2one
        self.mixed_selection = mixed_selection

        self.loss = loss

    def _forward(self, samples: List[Dict]):
        """The forward expects a List, which consists of:
           - data_sample: batched images, of shape [batch_size x C x D x H x W]
           - metainfo.padded_mask: a binary mask of shape [batch_size x D x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, center_z, height, width, depth). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionaries containing the two above keys for each decoder layer.
        """
        features, pos = self.backbone(samples)

        srcs, masks = [], []
        for layer, feat in enumerate(features):
            src, mask = feat["x"], feat["mask"]
            srcs.append(self.input_proj[layer](src))
            masks.append(mask)
            assert mask is not None, "backbone must provide padding mask"

        query_embeds = None
        if not self.two_stage or self.mixed_selection:
            query_embeds = self.query_embed.weight[0 : self.num_queries, :]

        # attention mask to prevent attention leakage 
        # between one2one and one2many queries
        self_attn_mask = torch.zeros(
            [
                self.num_queries,
                self.num_queries,
            ],
            dtype=bool,
            device=src.device,
        )
        self_attn_mask[
            self.num_queries_one2one :,
            0 : self.num_queries_one2one,
        ] = True
        self_attn_mask[
            0 : self.num_queries_one2one,
            self.num_queries_one2one :,
        ] = True

        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
            enc_outputs_delta,
            output_proposals,
            max_shape,
        ) = self.transformer(srcs, masks, pos, query_embeds, self_attn_mask)

        outputs_classes_one2one, outputs_coords_one2one = [], []
        outputs_classes_one2many, outputs_coords_one2many = [], []
        for lvl in range(hs.shape[0]):
            reference = inter_references[lvl - 1] if lvl > 0 else init_reference
            
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            
            if reference.shape[-1] == 6:
                tmp += reference
            else:
                assert reference.shape[-1] == 3, "the last dimension of reference points should be 3 or 6"
                tmp[..., :3] += reference
            
            outputs_coord = tmp.sigmoid()

            outputs_classes_one2one.append(outputs_class[:, 0 : self.num_queries_one2one])
            outputs_classes_one2many.append(outputs_class[:, self.num_queries_one2one :])

            outputs_coords_one2one.append(outputs_coord[:, 0 : self.num_queries_one2one])
            outputs_coords_one2many.append(outputs_coord[:, self.num_queries_one2one :])

        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)

        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)

        out = {
            "pred_logits": outputs_classes_one2one[-1],
            "pred_boxes": outputs_coords_one2one[-1],
            "pred_logits_one2many": outputs_classes_one2many[-1],
            "pred_boxes_one2many": outputs_coords_one2many[-1],
        }

        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_classes_one2one, outputs_coords_one2one)
            out["aux_outputs_one2many"] = self._set_aux_loss(outputs_classes_one2many, outputs_coords_one2many)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
            }

        return out

    def forward(self, samples: List[Dict]):
        outputs = self._forward(samples)
        losses = self.loss(outputs, samples["targets"])
        return outputs, losses

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class PlainDETRReParam(PlainDETR):
    def _forward(self, samples: List[Dict]):
        """
        The forward expects a List[Dict], which consists of:
           - data_tensor: batched images, of shape [batch_size x 3 x D x H x W]
           - metainfo.padded_mask: a binary mask of shape [batch_size x D x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, center_z, height, width, depth). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionaries containing the two above keys for each decoder layer.
        """
        features, pos = self.backbone(samples)

        srcs, masks = [], []
        for layer, feat in enumerate(features):
            src, mask = feat["x"], feat["mask"]
            srcs.append(self.input_proj[layer](src))
            masks.append(mask)
            assert mask is not None, "backbone must provide padding mask"

        query_embeds = None
        if not self.two_stage or self.mixed_selection:
            query_embeds = self.query_embed.weight[0 : self.num_queries, :]

        self_attn_mask = torch.zeros(
            [
                self.num_queries,
                self.num_queries,
            ],
            dtype=bool,
            device=src.device,
        )
        self_attn_mask[
            self.num_queries_one2one :,
            0 : self.num_queries_one2one,
        ] = True
        self_attn_mask[
            0 : self.num_queries_one2one,
            self.num_queries_one2one :,
        ] = True

        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
            enc_outputs_delta,
            output_proposals,
            max_shape,
        ) = self.transformer(srcs, masks, pos, query_embeds, self_attn_mask)

        outputs_classes_one2one, outputs_coords_one2one = [], []
        outputs_classes_one2many, outputs_coords_one2many = [], []

        outputs_coords_old_one2one, outputs_deltas_one2one = [], []
        outputs_coords_old_one2many, outputs_deltas_one2many = [], []

        for lvl in range(hs.shape[0]):
            reference = inter_references[lvl - 1] if lvl > 0 else init_reference

            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 6:
                outputs_coord = box_xyzxyz_to_cxcyczwhd(delta2bbox(reference, tmp, max_shape))
            else:
                raise NotImplementedError("We only support 6-dim reference points for re-parameterized bbox prediction")

            outputs_classes_one2one.append(outputs_class[:, 0 : self.num_queries_one2one])
            outputs_classes_one2many.append(outputs_class[:, self.num_queries_one2one :])

            outputs_coords_one2one.append(outputs_coord[:, 0 : self.num_queries_one2one])
            outputs_coords_one2many.append(outputs_coord[:, self.num_queries_one2one :])

            outputs_coords_old_one2one.append(reference[:, : self.num_queries_one2one])
            outputs_coords_old_one2many.append(reference[:, self.num_queries_one2one :])
            outputs_deltas_one2one.append(tmp[:, : self.num_queries_one2one])
            outputs_deltas_one2many.append(tmp[:, self.num_queries_one2one :])

        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)

        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)

        out = {
            "pred_logits": outputs_classes_one2one[-1],
            "pred_boxes": outputs_coords_one2one[-1],
            "pred_logits_one2many": outputs_classes_one2many[-1],
            "pred_boxes_one2many": outputs_coords_one2many[-1],
            "pred_boxes_old": outputs_coords_old_one2one[-1],
            "pred_deltas": outputs_deltas_one2one[-1],
            "pred_boxes_old_one2many": outputs_coords_old_one2many[-1],
            "pred_deltas_one2many": outputs_deltas_one2many[-1],
        }

        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(
                outputs_classes_one2one, outputs_coords_one2one, outputs_coords_old_one2one, outputs_deltas_one2one
            )
            out["aux_outputs_one2many"] = self._set_aux_loss(
                outputs_classes_one2many, outputs_coords_one2many, outputs_coords_old_one2many, outputs_deltas_one2many
            )

        if self.two_stage:
            out["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord_unact,
                "pred_boxes_old": output_proposals,
                "pred_deltas": enc_outputs_delta,
            }
        return out

    def forward(self, samples: List[Dict]):
        outputs = self._forward(samples)
        losses = self.loss(outputs, samples["targets"])
        return outputs, losses

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_coord_old, outputs_deltas):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {
                "pred_logits": a,
                "pred_boxes": b,
                "pred_boxes_old": c,
                "pred_deltas": d,
            }
            for a, b, c, d in zip(outputs_class[:-1], outputs_coord[:-1], outputs_coord_old[:-1], outputs_deltas[:-1])
        ]


# class PostProcess(nn.Module):
#     def __init__(self, topk=100, reparam=False):
#         super().__init__()
#         self.topk = topk
#         self.reparam = reparam

#     @torch.no_grad()
#     def forward(self, outputs, target_sizes, original_target_sizes=None):
#         """
#         Parameters:
#             outputs: raw outputs of the model
#             target_sizes: tensor of dimension [batch_size x 3] containing the size of each images of the batch
#                           For evaluation, this must be the original image size (before any data augmentation).
#                           For visualization, this should be the image size after data augment, but before padding.
#         """
#         out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

#         assert len(out_logits) == len(target_sizes), "the batch size of out_logits and target_sizes must be equal"
#         assert target_sizes.shape[1] == 3, "target_sizes should have shape [batch_size x 3]"
#         assert not self.reparam or original_target_sizes.shape[1] == 3, "original_target_sizes should have shape [batch_size x 3]"

#         prob = out_logits.sigmoid()
#         topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.topk, dim=1)
        
#         scores = topk_values
#         topk_boxes = topk_indexes // out_logits.shape[2]
#         labels = topk_indexes % out_logits.shape[2]
#         boxes = box_cxcyczwhd_to_xyzxyz(out_bbox)
#         boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 6))

#         img_h, img_w, img_d = target_sizes.unbind(1)
#         if self.reparam:
#             # img_i: [BS, 1, 1, 1]
#             img_h, img_w, img_d = img_h[:, None, None], img_w[:, None, None], img_d[:, None, None]
#             boxes[..., 0::3].clamp_(min=torch.zeros_like(img_w), max=img_w)
#             boxes[..., 1::3].clamp_(min=torch.zeros_like(img_h), max=img_h)
#             boxes[..., 2::3].clamp_(min=torch.zeros_like(img_d), max=img_d)
#             scale_h, scale_w, scale_d = (original_target_sizes / target_sizes).unbind(1)
#             scale_fct = torch.stack([scale_w, scale_h, scale_w, scale_h, scale_d, scale_d], dim=1)
#         else:
#             scale_fct = torch.stack([img_w, img_h, img_w, img_h, img_d, img_d], dim=1)
        
#         boxes = boxes * scale_fct[:, None, :]

#         results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]
#         return results