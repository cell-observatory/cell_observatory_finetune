"""
https://github.com/IDEA-Research/MaskDINO/blob/3831d8514a3728535ace8d4ecc7d28044c42dd14/maskdino/modeling/transformer_decoder/dino_decoder.py

(ADD COPYRIGHT HERE)
"""


import copy
from typing import Optional, List, Union

import torch
from torch import nn, Tensor
from torch.amp import autocast
from torch.nn import functional as F

from segmentation.models.utils.model_utils import MLP, gen_sine_embeddings_for_pos, inverse_sigmoid


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, 
                 embed_dim=256, 
                 dim_feedforward=1024,
                 dropout=0.1, 
                 activation=F.relu,
                 n_levels=4, 
                 n_heads=8, 
                 n_points=4,
                 use_deformable_box_attn=False,
                 key_aware_type=None,
                 ):
        super().__init__()

        # cross attention
        if use_deformable_box_attn:
            raise NotImplementedError("Deformable box attention is not implemented yet")
        else:
            self.cross_attn = MSDeformAttn(embed_dim, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # self attention
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        # ffn
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.activation = activation #TODO: Ensure init logic works same as detectron2 registry based method 
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.key_aware_type = key_aware_type
        self.key_aware_proj = None

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embeddings(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, x):
        x = x + self.dropout4(self.linear2(self.dropout3(self.activation(self.linear1(x)))))
        return self.norm3(x)

    # def forward_ffn(self, tgt):
    #     tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
    #     tgt = tgt + self.dropout4(tgt2)
    #     tgt = self.norm3(tgt)
    #     return tgt

    @autocast(enabled=False)
    def forward(self,
                # for tgt
                target: Optional[Tensor],  # num_queries, bs, embed_dim
                target_query_pos: Optional[Tensor] = None,  # pos_embeddings for query. MLP(Sine(pos_embeddings))
                target_query_sine_embed: Optional[Tensor] = None,  # pos_embeddings for query. Sine(pos_embeddings)
                target_key_padding_mask: Optional[Tensor] = None,
                target_reference_points: Optional[Tensor] = None,  # num_queries, bs, 6
                # for memory
                memory: Optional[Tensor] = None,  # hw, bs, embed_dim
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None,  # num_levels
                memory_shapes: Optional[Tensor] = None,  # bs, num_levels, 3
                memory_pos: Optional[Tensor] = None,  # pos_embeddings for memory
                # for attention masking
                self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
                cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
                ):
        # self attention
        if self.self_attn is not None:
            q = k = self.with_pos_embeddings(target, target_query_pos) # optionally add positional encodings
            target = target + self.dropout2(self.self_attn(q, k, target, attn_mask=self_attn_mask)[0]) 
            target = self.norm2(target)

        # cross attention
        if self.key_aware_type is not None: #  inject a global summary of the memory into the queries
            if self.key_aware_type == 'mean':
                target = target + memory.mean(0, keepdim=True)
            # TODO: Clean up this logic
            elif self.key_aware_type == 'proj_mean':
                target = target + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError("Unknown key_aware_type: {}".format(self.key_aware_type))
        
        # deformable cross attention
        target_cross_attn = self.cross_attn(self.with_pos_embeddings(target, target_query_pos).transpose(0, 1), # (bs, num_queries, embed_dim)
                               target_reference_points.transpose(0, 1).contiguous(), # (bs, num_queries, 3/6)
                               memory.transpose(0, 1),  # (bs, num_tokens, embed_dim)
                               memory_shapes, # (bs, num_levels, 3)
                               memory_level_start_index, # (num_levels)
                               memory_key_padding_mask).transpose(0, 1)
        target = target + self.dropout1(target_cross_attn)
        target = self.norm1(target)

        # ffn
        target = self.forward_ffn(target)
        return target # (num_queries, batch_size, embed_dim)


class TransformerDecoder(nn.Module):
    def __init__(self, 
                 decoder_layer, 
                 num_layers, 
                 norm=None,
                 return_intermediates=True,
                 embed_dim=256, 
                 query_dim=4,
                 modulate_hw_attn=True,
                 num_feature_levels=1,
                 deformable_decoder=True,
                 decoder_query_perturber=None,
                 dec_layer_number=None,  # number of queries for each layer in decoder
                 rm_dec_query_scale=True,
                 dec_layer_share=False,
                 dec_layer_dropout_prob=None,
                 ):
        # TODO: Move around initlizations into logical units for better readability
        super().__init__()
        
        # transformer decoder layers
        if num_layers > 0:
            if dec_layer_share:
                # share decoder layers
                self.decoder_layers = nn.ModuleList([decoder_layer for _ in range(num_layers)]) # deformable transformer decoder layer
            else:
                self.decoder_layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        else:
            self.decoder_layers = []

        # normalization layer
        self.norm = norm
        self.num_layers = num_layers
        
        # return decoder outputs from all layers 
        self.return_intermediates = return_intermediates
        assert return_intermediates, "Currently return_intermediate=True is only supported"
        
        # query dimension (3D (x, y, z) or 6D (x, y, z, w, h, d))
        self.query_dim = query_dim
        assert query_dim in [3, 6], "query_dim should be 3 or 6 but got {}".format(query_dim)
        
        self.num_feature_levels = num_feature_levels

        # reference point head: maps initial query positions into embed_dim space
        # used for generating reference points for attention sampling
        # recall: MLP args = input_dim, hidden_dim, output_dim, num_layers
        self.ref_point_head = MLP(query_dim // 3 * embed_dim, embed_dim, embed_dim, 2) # embed_dim-dim positional encodings per axis
        
        #  learn a scale to modulate sine positional encodings per query
        #  NOTE: not used with deformable decoder 
        if not deformable_decoder:
            self.query_pos_sine_scale = MLP(embed_dim, embed_dim, embed_dim, 2)
        else:
            self.query_pos_sine_scale = None

        # removes query scaling logic 
        if rm_dec_query_scale:
            self.query_scale = None
        else:
            raise NotImplementedError("dec_query_scale is not implemented yet")
            self.query_scale = MLP(embed_dim, embed_dim, embed_dim, 2)
        
        # bounding box regression and classification head placeholders
        self.bbox_regressor = None
        self.class_predictor = None

        self.embed_dim = embed_dim # transformer hidden dim
        self.modulate_hw_attn = modulate_hw_attn 
        self.deformable_decoder = deformable_decoder

        # TODO: Is this necessary?
        if not deformable_decoder and modulate_hw_attn:
            self.ref_anchor_head = MLP(embed_dim, embed_dim, 2, 2)
        else:
            self.ref_anchor_head = None

        # perturbs the reference points per decoder layer
        # for denoising 
        self.decoder_query_perturber = decoder_query_perturber
        self.box_pred_damping = None

        # number of layers in decoder
        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            assert isinstance(dec_layer_number, list)
            assert len(dec_layer_number) == num_layers

        # probability of dropout in decoder layers
        self.dec_layer_dropout_prob = dec_layer_dropout_prob
        if dec_layer_dropout_prob is not None:
            assert isinstance(dec_layer_dropout_prob, list)
            assert len(dec_layer_dropout_prob) == num_layers
            for i in dec_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self._reset_parameters()


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()


    def forward(self, 
                target, 
                memory,
                target_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                target_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_embeddings: Optional[Tensor] = None,
                reference_points: Optional[Tensor] = None,  # num_queries, bs, 2
                # for memory
                level_start_index: Optional[Tensor] = None,  # num_levels
                shapes: Optional[Tensor] = None,  # bs, num_levels, 2
                valid_ratios: Optional[Tensor] = None,
                ):
        output = target
        device = target.device

        intermediates = []
        reference_points = reference_points.sigmoid().to(device) # (num_queries, bs, 3/6) scaled to [0, 1]
        reference_points_list = [reference_points]

        # refine object queries and reference points per iteration & save intermediates
        for layer_id, decoder_layer in enumerate(self.decoder_layers):
            # perturb reference points
            if self.training and self.decoder_query_perturber is not None and layer_id != 0:
                reference_points = self.decoder_query_perturber(reference_points) # (num_queries, bs, 3/6)

            # Scale the reference points per feature level using valid_ratios (this way attention doesn’t sample into padded regions) 
            # (num_queries, bs, 1, query_dim) * (1, bs, nlevel, query_dim) ->  (num_queries, bs, n_levels, query_dim)
            # Result: for each query and each level, get scaled reference points based on valid ratio.
            reference_points_per_level = reference_points[:, :, None] \
                                         * torch.cat([valid_ratios]*3, dim=-1)[None, :] 
            query_sine_embeddings = gen_sine_embeddings_for_pos(reference_points_per_level[:, :, 0, :]) # (num_queries, bs, 256*query_dim)
            query_pos_embeddings = self.ref_point_head(query_sine_embeddings) # MLP -> (num_queries, bs, embed_dim)

            # DEPRECATED: 
            # raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256
            # pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            # query_pos = pos_scale * raw_query_pos

            output = decoder_layer(
                # for target
                target=output,
                target_query_pos_embeddings=query_pos_embeddings,
                target_query_sine_embed=query_sine_embeddings,
                target_key_padding_mask=target_key_padding_mask,
                target_reference_points=reference_points_per_level,
                # for memory
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_shapes=shapes,
                memory_pos_embeddings=pos_embeddings,
                # for attention masking
                self_attn_mask=target_mask,
                cross_attn_mask=memory_mask
            )

            # iterative reference point update
            # predict deltas from decoder output and update reference points accordingly 
            if self.bbox_embed is not None:
                reference_points_pre_sigmoid = inverse_sigmoid(reference_points)
                deltas = self.bbox_embed[layer_id](output).to(device)
                reference_points_updated = (deltas + reference_points_pre_sigmoid).sigmoid()

                reference_points = reference_points_updated.detach()
                # if layer_id != self.num_layers - 1:
                reference_points_list.append(reference_points_updated)

            intermediates.append(self.norm(output))

        return [
            [intermediate.transpose(0, 1) for intermediate in intermediates],
            [reference_point_element.transpose(0, 1) for reference_point_element in reference_points_list]
        ]

############################################################### OLD CODE ####################################################################################

# self.layers = _get_clones(decoder_layer, num_layers, layer_share=dec_layer_share)