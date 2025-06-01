"""
https://github.com/IDEA-Research/MaskDINO/blob/main/maskdino/modeling/pixel_decoder/maskdino_encoder.py

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


import copy
from typing import Dict, List, Callable

import torch
import torch.nn as nn

from torch.nn.init import normal_

import fvcore.nn.weight_init as weight_init

from segmentation.layers.layers import Conv3d
from segmentation.layers.activations import get_activation
from segmentation.layers.utils import compute_unmasked_ratio
from segmentation.layers.positional_encodings import PositionEmbeddingSine

from segmentation.models.ops.flash_deform_attn import FlashDeformAttn3D


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 embed_dim=256, 
                 feedforward_dim=1024,
                 dropout=0.1, 
                 activation="relu",
                 n_levels=4, 
                 n_heads=8, 
                 n_points=4
                 ):
        super().__init__()

        # self attention
        self.self_attn = FlashDeformAttn3D(embed_dim, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # ffn
        self.linear1 = nn.Linear(embed_dim, feedforward_dim)
        self.activation = get_activation(activation) 
        self.dropout2 = nn.Dropout(dropout)

        self.linear2 = nn.Linear(feedforward_dim, embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward_ffn(self, x):
        x = x + self.dropout3(self.linear2(self.dropout2(self.activation(self.linear1(x)))))
        return self.norm2(x)
    
    def forward(self, 
                x, 
                pos, 
                reference_points, 
                spatial_shapes, 
                level_start_index, 
                padding_mask=None
    ):
        # self attention
        x_flattened = x.flatten(2)  # [B, D*H*W, C]
        x = x + self.dropout1(self.self_attn(self.with_pos_embed(x, pos), reference_points, x_flattened, spatial_shapes, level_start_index, padding_mask))
        x = self.norm1(x)

        # ffn
        x = self.forward_ffn(x)
        return x


class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, 
                 embed_dim=256, 
                 feedforward_dim=1024, 
                 num_heads=8,
                 num_encoder_layers=6, 
                 dropout=0.1,
                 activation="relu",
                 num_feature_levels=4, 
                 enc_num_points=4,
                 ):
        super().__init__()

        self.n_head = num_heads
        self.embed_dim = embed_dim
        self.num_layers = num_encoder_layers

        encoder_layer = MSDeformAttnTransformerEncoderLayer(embed_dim, 
                                                            feedforward_dim,
                                                            dropout, 
                                                            activation,
                                                            num_feature_levels, 
                                                            num_heads, 
                                                            enc_num_points
                                                            )
        self.encoder_layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_encoder_layers)])

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, embed_dim))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, FlashDeformAttn3D):
                m._reset_parameters()
        normal_(self.level_embed)

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
            
            # scaling by valid_ratios adjusts the normalized reference grid so that it
            # only spans the unpadded region, i.e. [1, D*H*W] / (valid_ratio_d * D), 
            # i.e. scale grid to [0, 1] adjusted by valid ratio
            ref_z = ref_z.reshape(-1)[None] / (valid_ratios[:, None, lvl, 2] * D_) 
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            
            ref = torch.stack((ref_x, ref_y, ref_z), -1) # [B, D*H*W, 3]
            reference_points_list.append(ref)
        
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
    
    def forward_features(self, 
                         x, 
                         spatial_shapes, 
                         level_start_index, 
                         valid_ratios, 
                         pos=None, 
                         padding_mask=None
    ):
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=x.device)
        for layer in self.encoder_layers:
            x = layer(x, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return x
    
    def get_padding_mask(self, masks, features):
        # if any feature needs padding (dim not divisible by 32) we keep the user-passed masks,
        # otherwise we return all-false masks of the right shape
        enable_mask = any(
            (f.size(2) % 32 != 0) or (f.size(3) % 32 != 0)
            for f in features
        )
        if masks is None or not enable_mask:
            return [
                torch.zeros((f.size(0), f.size(2), f.size(3), f.size(4)),
                            device=f.device, dtype=torch.bool)
                for f in features
            ]
        return masks

    def forward(self, features, masks, pos_embeddings):
        # we pad if input features don't divide the required map size
        masks = self.get_padding_mask(masks, features)

        # feature: [bs, c, d, h, w]
        feature_shapes = [feature.shape[2:] for feature in features]
        # feature_shapes: [num_levels, 3], with each row = (D, H, W)
        feature_shapes = torch.as_tensor(feature_shapes, dtype=torch.long, device=features[0].device) 
        # [D1*H1*W1, ..., Dn*Hn*Wn] -> [0, D1*H1*W1, D1*H1*W1 + D2*H2*W2, ...]
        level_start_index = torch.cat((feature_shapes.new_zeros((1, )), feature_shapes.prod(1).cumsum(0)[:-1]))
        
        # [bs, c, d, h, w] -> [bs, c, d*h*w] -> [bs, d*h*w, c] 
        features_flattened = torch.cat([feature.flatten(2).transpose(1, 2) for feature in features], dim = 1)
        
        # [bs, c, d, h, w] -> [bs, c, d*h*w] -> [bs, d*h*w, c]
        positional_embeddings = [pos_embed.flatten(2).transpose(1, 2) for pos_embed in pos_embeddings]
        #  level_embed: [embed_dim] -> [1, 1, embed_dim] (for given level add a level-specific embedding broadcasted to all positions)
        positional_embeddings = [pos_embed + self.level_embed[lvl].view(1, 1, -1) for lvl, pos_embed in enumerate(positional_embeddings)]
        positional_embeddings = torch.cat(positional_embeddings, dim=1) # [bs, d*h*w, embed_dim]
        
        # [bs, l, d, h, w] -> [bs, l, d*h*w]
        masks_flattened = [mask.flatten(1) for mask in masks] 
        # [bs, num_levels, d*h*w]
        masks_flattened = torch.cat(masks_flattened, dim=1)
        
        # [bs, num_levels, 3] (valid ratio for each level)
        valid_ratios = torch.stack([compute_unmasked_ratio(m) for m in masks], 1)

        # call deformable attention layer on features with masks
        # to ensure only working over valid pixels
        memory = self.forward_features(features_flattened, 
                                       feature_shapes, 
                                       level_start_index, 
                                       valid_ratios, 
                                       positional_embeddings, 
                                       masks_flattened
                                       )

        return memory, feature_shapes, level_start_index


class MaskDINOEncoder(nn.Module):
    def __init__(
        self,
        input_shape: Dict,
        *,
        transformer_encoder_dropout: float,
        transformer_encoder_num_heads: int,
        transformer_encoder_dim_feedforward: int,
        num_transformer_encoder_layers: int,
        conv_dim: int,
        mask_dim: int,
        norm: Callable = None,
        # deformable transformer encoder args
        transformer_in_features: List[str],
        target_min_stride: int,
        total_num_feature_levels: int,
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
            total_num_feature_levels: total feature scales used (including downsampled features).
        """
        super().__init__()

        # Computation graph of MaskDINOEncoder:
        # 1. Backbone inputs : len(transformer_in_features)
        # 2. Add Extra encoder levels (downsampled 2x using Conv3D and Groupnorm) N1 = total_num_feature_levels - N0
        # 3. Pass maps (all at conv_dim channels) to Transformer encoder -> outputs same dim as inputs
        # 4. FPN lateral adapters (from backbone) for M0 = num_fpn_levels we do:
        # 5. FPN outputs (top‐down fusion) 2x upsamples transformer output until target_min_stride
        # 6. Final multi‐scale outputs M0 + 1 total maps from coarsest transformer down to target_min_stride
        # 7. Mask head : single 1×1 conv on coarsest FPN map

        # determine shapes of input features
        input_shapes = {k: v for k, v in input_shape.items() if k in transformer_in_features}
        # sort feature shapes from high to low resolution
        input_shapes_sorted = sorted(input_shapes.items(), key=lambda x: -x[1]["stride"])
        
        # define feature maps and determine number of feature levels 
        data_items = [(feature, map["stride"], map["channels"]) for feature, map in input_shapes_sorted]
        self.feature_maps, self.feature_maps_strides, feature_maps_in_channels = zip(*data_items)        
        self.num_feature_levels = len(self.feature_maps)

        # note that this is not sorted in high resolution -> low resolution order
        # this will be important for order in which we iterate for FPN lateral fusion   
        input_shape = sorted(input_shape.items(), key = lambda x: x[1]["stride"])
        self.full_feature_map_set, _, self.full_feature_set_channels = zip(*[(k, v["stride"], v["channels"]) for k, v in input_shape])
        self.total_num_feature_levels = total_num_feature_levels

        # define modules:
        # 1. channel alignment projection blocks to align all feature maps to have the same channel dim 
        #    also includes downsampling layers for extra feature levels if needed
        # 2. transformer encoder (uses deformable attention)
        # 3. position embedding (sine positional encoding)
        # 4. mask feature conv layer (1x1 conv to reduce channels for mask prediction)
        # 5. FPN layers (lateral and output convs for top-down fusion)
        
        if self.num_feature_levels > 1:
            channel_align_blocks = []
            # align all feature maps to have the same channel dim
            for in_channels in feature_maps_in_channels[::-1]:
                channel_align_blocks.append(nn.Sequential(
                    nn.Conv3d(in_channels, conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                ))
            
            # we optionally add extra feature levels
            extra_in_channels = [max(feature_maps_in_channels)] + \
                                [conv_dim] * (self.total_num_feature_levels - self.num_feature_levels - 1)
            extra_downsample_layers = [
                nn.Sequential(
                    nn.Conv3d(in_channels, conv_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, conv_dim),
                )
                for in_channels in extra_in_channels
            ]
            channel_align_blocks.extend(extra_downsample_layers)
            self.channel_align_projection = nn.ModuleList(channel_align_blocks)
        else:
            self.channel_align_projection = nn.ModuleList([
                nn.Sequential(
                    nn.Conv3d(feature_maps_in_channels[-1], conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                )])

        self.transformer_encoder = MSDeformAttnTransformerEncoder(
            embed_dim=conv_dim,
            feedforward_dim=transformer_encoder_dim_feedforward,
            dropout=transformer_encoder_dropout,
            num_heads=transformer_encoder_num_heads,
            num_encoder_layers=num_transformer_encoder_layers,
            num_feature_levels=self.total_num_feature_levels,
        )
        # conv_dim // 2 since 
        self.pos_embedding = PositionEmbeddingSine(conv_dim // 3, normalize=True)         
        self.mask_features = Conv3d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.weight_init()

    def weight_init(self):
        weight_init.c2_xavier_fill(self.mask_features)
        for proj in self.channel_align_projection:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # NOTE: the following FPN logic is in practice not used in MaskDINO, but is left here for reference

        # for lateral_conv, output_conv in zip(self.lateral_convs, self.output_convs):
        #     weight_init.c2_xavier_fill(lateral_conv)
        #     weight_init.c2_xavier_fill(output_conv)

    def forward_features(self, features, masks):
        extra_features_list = []
        extra_pos_embeddings_list = []
        # if add extra feature maps, apply input projection to coarsest feature map iteratively 
        if self.total_num_feature_levels > self.num_feature_levels:
            # we must have at least one extra feature level
            feature_smallest = self.channel_align_projection[l](features[self.feature_maps[-1]])
            extra_features_list.append(feature_smallest)
            extra_pos_embeddings_list.append(self.pos_embedding(feature_smallest))
            for l in range(self.num_feature_levels+1, self.total_num_feature_levels):
                feature = self.channel_align_projection[l](extra_features_list[-1])
                extra_features_list.append(feature)
                extra_pos_embeddings_list.append(self.pos_embedding(feature))
        
        extra_features_list = extra_features_list[::-1]

        features_list = []
        pos_embeddings_list = []
        for idx, feature_map in enumerate(self.feature_maps[::-1]):
            x = features[feature_map]
            features_list.append(self.channel_align_projection[idx](x)) 
            pos_embeddings_list.append(self.pos_embedding(x))

        features_list.extend(extra_features_list)
        pos_embeddings_list.extend(extra_pos_embeddings_list)        
    
        # encode feature maps with deformable attention transformer encoder
        transformer_features, shapes, level_start_index = self.transformer_encoder(features_list, masks, pos_embeddings_list)
        
        # output from transformer is flattened, so we need to split per level
        tokens_per_level = [None] * self.total_num_feature_levels 
        for i in range(self.total_num_feature_levels-1):
            tokens_per_level[i] = level_start_index[i + 1] - level_start_index[i]
        tokens_per_level[self.total_num_feature_levels-1] = transformer_features.shape[1] - level_start_index[self.total_num_feature_levels-1]
        
        # [bs, num_levels * tokens_per_level, embed_dim] -> List[Tensor[bs, tokens_in_level_i, C]]
        transformer_features = torch.split(transformer_features, tokens_per_level, dim=1) 

        output_feature_maps = []
        for transformer_feature, shape in zip(transformer_features, shapes):
            # unflatten sequence: [bs, tokens_in_level_i, C] -> [bs, C, tokens_in_level_i] -> [bs, C, D, H, W]
            output_map = transformer_feature.transpose(1, 2).view(transformer_feature.shape[0], -1, shape[0], shape[1], shape[2])
            output_feature_maps.append(output_map)

        return self.mask_features(output_feature_maps[-1]), output_feature_maps[0], output_feature_maps