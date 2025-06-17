from typing import Union, Literal

import torch
from torch import nn

from train_platform.training.masking import apply_masks
from train_platform.models.maskedencoder import MaskedEncoder


class MaskedEncoderWrapper(MaskedEncoder):
    """
    Wrapper for the MaskedEncoder model from Platform.
    """
    def __init__(self,
        channel_predict: bool,
        num_channels: int,
        model_template: Literal[
            'me', # custom use `embed_dim`, `depth`, `num_heads` and `mlp_ratio` to config model
            'me-tiny',
            'me-small',
            'me-base',
            'me-large',
            'me-huge',
            'me-giant',
            'me-gigantic'
        ] = 'me',
        input_fmt='BZYXC',
        input_shape=(1, 6, 64, 64, 1),
        lateral_patch_size=16,
        axial_patch_size=1,
        temporal_patch_size=1,
        embed_dim=768,
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
        use_conv_proj=False,
        **kwargs
    ):
        super(MaskedEncoderWrapper, self).__init__(
            model_template=model_template,
            input_fmt=input_fmt,
            input_shape=input_shape,
            lateral_patch_size=lateral_patch_size,
            axial_patch_size=axial_patch_size,
            # TODO: support temporal patch size in platform
            temporal_patch_size=temporal_patch_size,
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
            use_conv_proj=use_conv_proj,
            **kwargs
        )

        self.num_channels = num_channels
        self.channel_predict = channel_predict
        if self.channel_predict:
            self.token_param = nn.Parameter(torch.zeros([1] + [1]* (len(input_shape) - 2) + [self.num_channels]))

    def forward(self, inputs, masks=None, concat_masks=True):
        """
        Forward pass of the masked encoder.
        """
        if self.channel_predict:
            return self._forward_channel_predict(inputs, masks)
        else:
            return self._forward(inputs, masks, concat_masks)

    def _forward(self, inputs, masks=None, concat_masks=True):
        """
        Forward pass of the masked encoder. Patch masking or no masking.
        """
        x, patches = self.patch_embedding(inputs, return_patches=True)
        x += self.pos_embedding(inputs)

        if masks is not None:
            x = apply_masks(x, masks.to(torch.int64), concat=concat_masks)

        x = self.encoder(x)
        x = self.norm(x)
        return x, patches
    
    def _forward_channel_predict(self, inputs, masks):
        """
        Forward pass for channel prediction task.
        """
        # masks: (1, (T), D, H, W, C) where mask == 1 for (masked
        # channel, patch combinations) and 0 for unmasked
        # token_param: (1, ..., C) -> broadcasts along dims
        masked_inputs = torch.where(masks, self.token_param.to(masks.device), inputs)

        x, patches = self.patch_embedding(masked_inputs, return_patches=True)
        x += self.pos_embedding(inputs)

        x = self.encoder(x)
        x = self.norm(x)
        return x, patches