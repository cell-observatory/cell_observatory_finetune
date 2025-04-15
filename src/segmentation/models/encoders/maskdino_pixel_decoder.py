"""
https://github.com/IDEA-Research/MaskDINO/blob/main/maskdino/modeling/pixel_decoder/maskdino_encoder.py

(ADD COPYRIGHT HERE)

"""


import copy
from typing import Dict, List, Optional, Callable, Union

import numpy as np

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

import fvcore.nn.weight_init as weight_init

from segmentation.models.utils.model_utils import _get_activation_fn, compute_unmasked_ratio 
from segmentation.models.decoders.position_encoding import PositionEmbeddingSine
from segmentation.models.utils.model_utils import Conv3d


class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self, 
                 embed_dim=256, 
                 n_head=8,
                 num_encoder_layers=6, 
                 dim_feedforward=1024, 
                 dropout=0.1,
                 activation="relu",
                 num_feature_levels=4, 
                 enc_n_points=4,
                 ):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_head = n_head

        encoder_layer = MSDeformAttnTransformerEncoderLayer(embed_dim, 
                                                            dim_feedforward,
                                                            dropout, 
                                                            activation,
                                                            num_feature_levels, 
                                                            n_head, 
                                                            enc_n_points
                                                            )
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, embed_dim))

        self._reset_parameters()


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)


    def forward(self, features, masks, pos_embeds):
        # pad if input image doesn't divide evenly into the required feature map sizes
        enable_mask=0
        if masks is not None:
            for feature in features:
                if feature.size(2) % 32 or feature.size(3) % 32:
                    enable_mask = 1

        # feature dims: (batch_size, channels, depth, height, width)
        if enable_mask == 0:
            masks = [torch.zeros((feature.size(0), feature.size(2), feature.size(3), feature.size(4)), device=feature.device, dtype=torch.bool) for feature in features]

        shapes = []
        feature_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        for lvl, (feature, mask, pos_embed) in enumerate(zip(features, masks, pos_embeds)):
            bs, c, d, h, w = feature.shape
            shape = (d, h, w)
            shapes.append(shape)
            feature = feature.flatten(2).transpose(1, 2) # [bs, c, d, h, w] -> [bs, c, d*h*w] -> [bs, d*h*w, c]
            mask = mask.flatten(1) # [bs, d, h, w] -> [bs, d*h*w]
            pos_embed = pos_embed.flatten(2).transpose(1, 2) # [bs, c, d, h, w] -> [bs, c, d*h*w] -> [bs, d*h*w, c]
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1) # level_embed: [embed_dim] -> [1, 1, embed_dim] (for given level)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feature_flatten.append(feature)
            mask_flatten.append(mask)

        feature_flatten = torch.cat(feature_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)

        shapes = torch.as_tensor(shapes, dtype=torch.long, device=feature_flatten.device) # [num_levels, 3], with each row = (D, H, W)
        level_start_index = torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1])) # [D1*H1*W1, ..., Dn*Hn*Wn] -> [0, D1*H1*W1, D1*H1*W1 + D2*H2*W2, ...]
        valid_ratios = torch.stack([self.compute_unmasked_ratio(m) for m in masks], 1) # [bs, num_levels, 3] (valid ratio for each level)

        memory = self.encoder(feature_flatten, shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        return memory, shapes, level_start_index


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 embed_dim=256, 
                 d_ffn=1024,
                 dropout=0.1, 
                 activation="relu",
                 n_levels=4, 
                 n_heads=8, 
                 n_points=4
                 ):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(embed_dim, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # ffn
        self.linear1 = nn.Linear(embed_dim, d_ffn)
        self.activation = _get_activation_fn(activation) # TODO: Keep this logic?
        self.dropout2 = nn.Dropout(dropout)

        self.linear2 = nn.Linear(d_ffn, embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    # def forward_ffn(self, src):
    #     src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
    #     src = src + self.dropout3(src2)
    #     src = self.norm2(src)
    #     return src

    # def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
    #     # self attention
    #     src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
    #     src = src + self.dropout1(src2)
    #     src = self.norm1(src)

    #     # ffn
    #     src = self.forward_ffn(src)

    #     return src
    
    def forward_ffn(self, x):
        x = x + self.dropout3(self.linear2(self.dropout2(self.activation(self.linear1(x)))))
        return self.norm2(x)
    
    def forward(self, x, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        x = x + self.dropout1(self.self_attn(self.with_pos_embed(x, pos), reference_points, x, spatial_shapes, level_start_index, padding_mask))
        x = self.norm1(x)

        # ffn
        x = self.forward_ffn(x)
        return x


class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)]) # _get_clones(encoder_layer, num_layers)

    @staticmethod
    def get_reference_points(shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (D_, H_, W_) in enumerate(shapes):
            # create grid [0.5, 1.5, ..., size_dim - 0.5]
            ref_z, ref_y, ref_x = torch.meshgrid(
                                                torch.linspace(0.5, D_ - 0.5, D_, dtype=torch.float32, device=device),
                                                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                                                indexing='ij'
                                                ) 
            
            # Scaling by valid_ratios adjusts the normalized reference
            # grid so that it only spans the unpadded region, i.e. 
            # [1, D*H*W] / (valid_ratio_d * D), i.e. scale grid to [0, 1] adjusted by valid ratio
            ref_z = ref_z.reshape(-1)[None] / (valid_ratios[:, None, lvl, 2] * D_) 
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            
            ref = torch.stack((ref_x, ref_y, ref_z), -1) # [B, D*H*W, 3]
            reference_points_list.append(ref)
        
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, x, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=x.device)
        for _, layer in enumerate(self.layers):
            x = layer(x, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return x


class MaskDINOEncoder(nn.Module):
    def __init__(
        self,
        input_shape: Dict,
        *,
        transformer_dropout: float,
        transformer_nheads: int,
        transformer_dim_feedforward: int,
        transformer_enc_layers: int,
        conv_dim: int,
        mask_dim: int,
        norm: Callable = None,
        # deformable transformer encoder args
        transformer_in_features: List[str],
        common_stride: int,
        num_feature_levels: int,
        total_num_feature_levels: int,
        feature_order: str,
    ):
        """
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (callable): normalization for all conv layers
            num_feature_levels: feature scales used
            total_num_feature_levels: total feature scales used (including downsampled features)
            feature_order: 'low2high' or 'high2low', i.e., 'low2high' means low-resolution features are put in first.
        """
        # TODO: Move around initlizations into logical units for better readability
        super().__init__()
        transformer_input_shape = {
            k: v for k, v in input_shape.items() if k in transformer_in_features
        }
        
        # input shape of pixel decoder (sorted)
        input_shape = sorted(input_shape.items(), key = lambda x: x[1].stride)
        
        # starting from "res2" to "res5"
        self.feature_order = feature_order
        self.in_features, self.feature_strides, self.feature_channels = zip(*[(k, v.stride, v.channels) for k, v in input_shape])

        if feature_order == "low2high":
            transformer_input_shape = sorted(transformer_input_shape.items(), key = lambda x: -x[1].stride)
        else:
            transformer_input_shape = sorted(transformer_input_shape.items(), key = lambda x: x[1].stride)
        
        # starting from "res2" to "res5"
        self.transformer_in_features, self.transformer_feature_strides, transformer_in_channels = zip(*[(k, v.stride, v.channels) for k, v in transformer_input_shape])
        self.transformer_num_feature_levels = len(self.transformer_in_features)
        self.lowest_res_index = transformer_in_channels.index(max(transformer_in_channels))
        self.high_resolution_index = 0 if self.feature_order == 'low2high' else -1

        self.maskdino_num_feature_levels = num_feature_levels  # always use 3 scales
        self.total_num_feature_levels = total_num_feature_levels
        self.common_stride = common_stride
        
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            # Align all feature maps to have the same channel dim
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(nn.Sequential(
                    nn.Conv3d(in_channels, conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                ))
            
            # input projection for downsampling (IDEA: pass a few scales to Transformer, but mask/decoder heads may benefit from more)
            extra_in_channels = [max(transformer_in_channels)] + [conv_dim] * (self.total_num_feature_levels - self.transformer_num_feature_levels - 1)
            extra_downsample_layers = [
                nn.Sequential(
                    nn.Conv3d(in_channels, conv_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, conv_dim),
                )
                for in_channels in extra_in_channels
            ]
            input_proj_list.extend(extra_downsample_layers)
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv3d(transformer_in_channels[-1], conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                )])

        # TODO: Move weight initialization to a separate function
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.transformer = MSDeformAttnTransformerEncoderOnly(
            embed_dim=conv_dim,
            dropout=transformer_dropout,
            n_head=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.total_num_feature_levels,
        )

        self.pos_embedding = PositionEmbeddingSine(conv_dim // 2, normalize=True)

        self.mask_dim = mask_dim
        
        # use 1x1 conv instead (#TODO: is Conv3d wrapper necessary?) 
        self.mask_features = Conv3d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        weight_init.c2_xavier_fill(self.mask_features)
        
        # extra fpn levels (i.e. how many FPN levels (2x upsampling) are needed to top-down-fuse 
        # from higher-res to desired pred low-res above and beyond transformer feature map strides
        # EX: If the lowest-resolution transformer output is 1/8, and you want to predict at 1/4, 
        # then you need one more upsampling level to reach that resolution.
        stride = min(self.transformer_feature_strides) # highest res. feature stride
        self.num_fpn_levels = max(int(np.log2(stride) - np.log2(self.common_stride)), 1)  

        lateral_convs = []
        output_convs = []
        use_bias = True if norm else False # TODO: check if this is correct (updated from string based init in detectron2)

        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
            # TODO: Make sure Hydra setup allows for norm init. 
            lateral_norm = norm(conv_dim)
            output_norm = norm(conv_dim)

            lateral_conv = Conv3d(
                in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv3d(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]


    @autocast(enabled=False)
    def forward_features(self, features, masks):
        # backbone features
        features_list = []
        pos_embeddings_list = []

        # additional downsampled features
        extra_features_list = []
        extra_pos_embeddings_list = []

        # if add extra feature maps, apply input projection to coarsest feature map iteratively 
        if self.total_num_feature_levels > self.transformer_num_feature_levels:
            feature_smallest = features[self.transformer_in_features[self.lowest_res_index]].float() # str index lowest res
            for l in range(self.transformer_num_feature_levels, self.total_num_feature_levels):
                if l == self.transformer_num_feature_levels:
                    feature = self.input_proj[l](feature_smallest)
                else:
                    feature = self.input_proj[l](extra_features_list[-1])
                extra_features_list.append(feature)
                extra_pos_embeddings_list.append(self.pos_embedding(feature))
        # reverse order of extra feature maps
        extra_features_list = extra_features_list[::-1]

        # reverse feature maps
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()  # deformable detr does not support half precision
            features_list.append(self.input_proj[idx](x)) # 3D Conv -> Groupnorm (32 groups) to match conv_dim 
            pos_embeddings_list.append(self.pos_embedding(x))

        # low2high => from high spatial resolution (small channel dim) → low spatial resolution,
        # like res2 → res5 else flipped
        if self.feature_order == 'low2high':
            features_list.extend(extra_features_list)
            pos_embeddings_list.extend(extra_pos_embeddings_list)        
        else:
            extra_features_list.extend(features_list)
            extra_pos_embeddings_list.extend(pos_embeddings_list)
            features_list = extra_features_list
            pos_embeddings_list = extra_pos_embeddings_list
        
        transformer_features, shapes, level_start_index = self.transformer(features_list, masks, pos_embeddings_list)
        batch_size = transformer_features.shape[0]
        
        tokens_per_level = [None] * self.total_num_feature_levels # output from transformer is flattened, so we need to split per level
        for i in range(self.total_num_feature_levels):
            if i < self.total_num_feature_levels - 1:
                tokens_per_level[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                tokens_per_level[i] = transformer_features.shape[1] - level_start_index[i]
        
        # [bs, num_levels * tokens_per_level, embed_dim] -> List[Tensor[bs, tokens_in_level_i, C]]
        transformer_features = torch.split(transformer_features, tokens_per_level, dim=1) 

        output_features = []
        multi_scale_features = []
        for idx, transformer_feature in enumerate(transformer_features):
            # unflatten sequence: [bs, tokens_in_level_i, C] -> [bs, C, tokens_in_level_i] -> [bs, C, D, H, W]
            output_features.append(transformer_feature.transpose(1, 2).view(batch_size, -1, shapes[idx][0], shapes[idx][1], shapes[idx][2]))

        # append `out` with extra FPN levels, reverse feature maps into top-down order (from low to high resolution)
        # recall: num_fpn_levels = extra fpn levels needed to fuse coarsest transformer output with target res
        # and in_features = sorted input features from backbone
        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]): 
            x = features[f].float()
            
            lateral_conv = self.lateral_convs[idx] # 3D conv with 1x1 kernel size with input_channels -> conv_dim channels
            output_conv = self.output_convs[idx] # 3D conv with 3x3 kernel size with conv_dim channels
            
            feature_lateral = lateral_conv(x)

            # Take the lateral feature from the backbone, and add the upsampled feature
            # from the coarser level of the transformer output.
            # Following FPN implementation, we use nearest upsampling here
            # recall: high_res_index = 0 if self.feature_order == 'low2high' else -1
            transformer_features_fpn = feature_lateral + F.interpolate(output_features[self.high_resolution_index], 
                                                                       size=feature_lateral.shape[-3:], 
                                                                       mode="trilinear", 
                                                                       align_corners=False)
            transformer_features_fpn = output_conv(transformer_features_fpn)
            output_features.append(transformer_features_fpn)
        
        # keep self.total_num_feature_levels (total nr of feature scales used) levels 
        assert len(output_features) >= self.total_num_feature_levels # TODO: is this necessary?
        multi_scale_features = output_features[:self.total_num_feature_levels]

        # returns output of 1x1 conv3D op. with mask_dim channels on last fpn feature level, first fpn feature level, and all fpn feature levels  
        return self.mask_features(output_features[-1]), output_features[0], multi_scale_features
    

###################################################### OLD CODE ####################################################################################

# num_cur_levels = 0
# for feature in output_features:
#     if num_cur_levels < self.total_num_feature_levels:
#         multi_scale_features.append(feature)
#         num_cur_levels += 1

# in_channels = max(transformer_in_channels)
# for _ in range(self.total_num_feature_levels - self.transformer_num_feature_levels):  # exclude the res2
#     # 2x downsample
#     input_proj_list.append(nn.Sequential(
#         nn.Conv3d(in_channels, conv_dim, kernel_size=3, stride=2, padding=1),
#         nn.GroupNorm(32, conv_dim),
#     ))
#     in_channels = conv_dim
# self.input_proj = nn.ModuleList(input_proj_list)

# @classmethod
# def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
#     ret = {}
#     ret["input_shape"] = {
#         k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
#     }
#     ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
#     ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
#     ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
#     ret["transformer_dropout"] = cfg.MODEL.MaskDINO.DROPOUT
#     ret["transformer_nheads"] = cfg.MODEL.MaskDINO.NHEADS
#     ret["transformer_dim_feedforward"] = cfg.MODEL.SEM_SEG_HEAD.DIM_FEEDFORWARD  # deformable transformer encoder
#     ret[
#         "transformer_enc_layers"
#     ] = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS  # a separate config
#     ret["transformer_in_features"] = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES  # ['res3', 'res4', 'res5']
#     ret["common_stride"] = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
#     ret["total_num_feature_levels"] = cfg.MODEL.SEM_SEG_HEAD.TOTAL_NUM_FEATURE_LEVELS
#     ret["num_feature_levels"] = cfg.MODEL.SEM_SEG_HEAD.NUM_FEATURE_LEVELS
#     ret["feature_order"] = cfg.MODEL.SEM_SEG_HEAD.FEATURE_ORDER
#     return ret

# features_list.extend(extra_features_list) if self.feature_order == 'low2high' else extra_features_list.extend(features_list)
# pos_embeddings_list.extend(extra_pos_embeddings_list) if self.feature_order == 'low2high' else extra_pos_embeddings_list.extend(pos_embeddings_list)

# if self.feature_order != 'low2high':
#     features_list = extra_features_list
#     pos_embeddings_list = extra_pos_embeddings_list