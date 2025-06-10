from typing import Union, Literal

import torch
from torch import nn

from platform.models.patch_embeddings import PosEmbedding
from platform.models.maskedpredictor import MaskedPredictor


class MaskedPredictorWrapper(MaskedPredictor):
    """
    Wrapper for the Masked Predictor model from Platform.
    """
    def __init__(self,
        channel_predict: bool,
        num_channels: int,
        model_template: Literal[
            'mp', # custom use `embed_dim`, `depth`, `num_heads` and `mlp_ratio` to config model
            'mp-tiny',
            'mp-small',
            'mp-base',
            'mp-large',
            'mp-huge',
            'mp-giant',
            'mp-gigantic'
        ] = 'mp',
        input_fmt='BZYXC',
        input_shape=(1, 6, 64, 64, 1),
        lateral_patch_size=16,
        axial_patch_size=1,
        temporal_patch_size=1,
        input_embed_dim=768,
        output_embed_dim=768,
        embed_dim=384,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        proj_drop_rate=0.0,
        att_drop_rate=0.0,
        drop_path_rate=0.1,
        init_std=0.02,
        fixed_dropout_depth=False,
        norm_layer: Union[nn.Module, Literal['RmsNorm', 'LayerNorm', 'SyncBatchNorm', 'GroupNorm']] = 'RmsNorm',
        act_layer: Union[nn.Module, Literal['GELU', 'SiLU', 'LeakyReLU', 'GLU', 'Sigmoid', 'Tanh']] = 'SiLU',
        mlp_layer: Union[nn.Module, Literal['Mlp', 'SwiGLU']] = 'SwiGLU',
        **kwargs,
    ):
        super(MaskedPredictorWrapper, self).__init__(
            model_template=model_template,
            input_fmt=input_fmt,
            input_shape=input_shape,
            lateral_patch_size=lateral_patch_size,
            axial_patch_size=axial_patch_size,
            # TODO: support temporal patch size in platform
            temporal_patch_size=temporal_patch_size, 
            input_embed_dim=input_embed_dim,
            output_embed_dim=output_embed_dim,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            proj_drop_rate=proj_drop_rate,
            att_drop_rate=att_drop_rate,
            drop_path_rate=drop_path_rate,
            init_std=init_std,
            fixed_dropout_depth=fixed_dropout_depth,
            norm_layer=norm_layer,
            act_layer=act_layer,
            mlp_layer=mlp_layer,
            **kwargs
        )

        self.num_channels = num_channels
        self.channel_predict = channel_predict
        if self.channel_predict:
            self.token_param = nn.Parameter(torch.zeros([1]* (len(input_shape) - 2) + [self.num_channels]))

    def forward(self, inputs, original_patch_indices=None, target_masks=None):
        """
        Forward pass of the masked encoder.
        """
        if self.channel_predict:
            return self._forward_channel_predict(inputs)
        else:
            return self._forward(inputs, original_patch_indices, target_masks)

    def _forward_channel_predict(self, inputs):
        patches = self.patch_projection(inputs)
        x = patches + self.pos_embedding(patches)
        x = self.encoder(x)
        x = self.norm(x)
        x = self.output_projection(x)
        return x

    def _forward(self, inputs, original_patch_indices=None, target_masks=None):
        tokens = self.patch_projection(inputs)
        
        if target_masks is not None:
            mask_tokens = self.token_param.repeat(inputs.shape[0], target_masks.shape[1], 1)
            patches = torch.cat([tokens, mask_tokens], dim=1)
            patches = torch.gather(
                patches,
                dim=1,
                index=original_patch_indices.unsqueeze(-1).repeat(1, 1, self.embed_dim).to(torch.int64)
            ) # reorder patches to original order
        else:
            patches = tokens

        x = patches + self.pos_embedding(patches)

        x = self.encoder(x)
        x = self.norm(x)
        x = self.output_projection(x)
        return x