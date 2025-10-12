from typing import Union, Literal

import torch.nn as nn

from cell_observatory_finetune.cell_observatory_platform.models.mlp import get_mlp
from cell_observatory_finetune.cell_observatory_platform.models.norm import get_norm
from cell_observatory_finetune.cell_observatory_platform.training.helpers import init_weights
from cell_observatory_finetune.cell_observatory_platform.models.activation import get_activation
from cell_observatory_finetune.cell_observatory_platform.models.maskedencoder import MaskedEncoder
from cell_observatory_finetune.cell_observatory_platform.models.maskedpredictor import MaskedPredictor

from cell_observatory_finetune.training.losses import get_loss_fn
from cell_observatory_platform.data.masking.mask_generator import apply_masks
from cell_observatory_platform.models.maskedautoencoder import MaskedAutoEncoder

from cell_observatory_finetune.models.heads.linear_head import LinearHead
from cell_observatory_finetune.models.heads.dense_predictor_head import DPTHead


CONFIGS = {
    'mae-tiny': {
        'embed_dim': 192,
        'decoder_embed_dim': 96,
        'depth': 12,
        'decoder_depth': 3,
        'num_heads': 3,
        'decoder_num_heads': 3,
        'mlp_ratio': 4,
    },
    'mae-small': {
        'embed_dim': 384,
        'decoder_embed_dim': 192,
        'depth': 12,
        'decoder_depth': 6,
        'num_heads': 6,
        'decoder_num_heads': 6,
        'mlp_ratio': 4,
    },
    'mae-base': {
        'embed_dim': 768,
        'decoder_embed_dim': 256,
        'depth': 12,
        'decoder_depth': 8,
        'num_heads': 12,
        'decoder_num_heads': 8,
        'mlp_ratio': 4,
    },
    'mae-large': {
        'embed_dim': 1024,
        'decoder_embed_dim': 512,
        'depth': 24,
        'decoder_depth': 8,
        'num_heads': 16,
        'decoder_num_heads': 8,
        'mlp_ratio': 4,
    },
    'mae-huge': {
        'embed_dim': 1280,
        'decoder_embed_dim': 512,
        'depth': 32,
        'decoder_depth': 8,
        'num_heads': 16,
        'decoder_num_heads': 8,
        'mlp_ratio': 4,
    },
    'mae-2billion': {
        'embed_dim': 2560,
        'decoder_embed_dim': 512,
        'depth': 24,
        'decoder_depth': 8,
        'num_heads': 32,
        'decoder_num_heads': 8,
        'mlp_ratio': 4,
    },
    'mae-6billion': {
        'embed_dim': 4096,
        'decoder_embed_dim': 512,
        'depth': 32,
        'decoder_depth': 8,
        'num_heads': 32,
        'decoder_num_heads': 8,
        'mlp_ratio': 4,
    },
    'mae-giant': {
        'embed_dim': 1408,
        'decoder_embed_dim': 512,
        'depth': 40,
        'decoder_depth': 8,
        'num_heads': 16,
        'decoder_num_heads': 8,
        'mlp_ratio': 48/11,
    },
    'mae-gigantic': {
        'embed_dim': 1664,
        'decoder_embed_dim': 1024,
        'depth': 48,
        'decoder_depth': 16,
        'num_heads': 16,
        'decoder_num_heads': 16,
        'mlp_ratio': 64/13,
    },
    'mae-enormous': {
        'embed_dim': 1792,
        'decoder_embed_dim': 1024,
        'depth': 56,
        'decoder_depth': 16,
        'num_heads': 16,
        'decoder_num_heads': 16,
        'mlp_ratio': 8.5714285714,
    }
}


class FinetuneMaskedAutoEncoder(MaskedAutoEncoder):
    def __init__(
        self,
        decoder: Literal['vit', 'linear', 'dense_predictor'],
        task: Literal['channel_split', 
                      'upsample_time', 
                      'upsample_space', 
                      'upsample_spacetime'],
        model_template: Literal[
            'mae', # custom use `embed_dim`, `decoder_embed_dim`, `depth`, `num_heads` and `mlp_ratio` to config model
            'mae-tiny',
            'mae-small',
            'mae-base',
            'mae-large',
            'mae-huge',
            'mae-giant',
            'mae-gigantic'
        ] = 'mae',
        input_fmt='TZYXC',
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
        abs_sincos_enc: bool = False,
        rope_pos_enc: bool = True,
        rope_random_rotation_per_head: bool = True,
        rope_mixed: bool = True,
        rope_theta: float = 10.0,
        weight_init_type: str = 'mae',
        mlp_wide_silu: bool = False,
        loss_fn: str = 'l2_masked',
        # vit decoder specific
        decoder_num_heads=8,
        decoder_depth=8,
        decoder_embed_dim=256,
        # linear head specific
        decoder_with_bn=False,
        decoder_num_layers=3,
        decoder_hidden_dim=2048,
        decoder_bottleneck_dim=256,
        decoder_mlp_bias=True,
        # dense predictor specific
        decoder_num_layers_dense=3,
        decoder_hidden_dim_dense=512,
        decoder_mlp_bias_dense=True,
    ):
        if model_template in CONFIGS.keys():
            config = CONFIGS[model_template]
            self.depth = config['depth']
            self.embed_dim = config['embed_dim']
            self.num_heads = config['num_heads']
            self.mlp_ratio = config['mlp_ratio']
        else:
            self.depth = depth
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.mlp_ratio = mlp_ratio

        self.input_fmt = input_fmt
        self.input_shape = input_shape
        
        axis_to_value = dict(zip(input_fmt, input_shape[1:]))
        self.in_chans = axis_to_value['C']
        self.num_frames = axis_to_value['T']

        self.axial_patch_size = axial_patch_size
        self.lateral_patch_size = lateral_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.proj_drop_rate = proj_drop_rate
        self.att_drop_rate = att_drop_rate
        self.drop_path_rate = drop_path_rate
        self.fixed_dropout_depth = fixed_dropout_depth
        
        self.init_std = init_std

        self.norm_layer = get_norm(norm_layer)
        self.act_layer = get_activation(act_layer)
        self.mlp_layer = get_mlp(mlp_layer)

        # positional encoding parameters
        self.abs_sincos_enc = abs_sincos_enc
        self.rope_pos_enc = rope_pos_enc
        self.rope_mixed = rope_mixed
        self.rope_theta = rope_theta
        self.wide_silu = mlp_wide_silu
        self.rope_random_rotation_per_head = rope_random_rotation_per_head

        self.masked_encoder = MaskedEncoder(
            input_fmt=self.input_fmt,
            input_shape=self.input_shape,
            lateral_patch_size=self.lateral_patch_size,
            axial_patch_size=self.axial_patch_size,
            temporal_patch_size=self.temporal_patch_size,
            channels=self.in_chans,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            proj_drop_rate=self.proj_drop_rate,
            att_drop_rate=self.att_drop_rate,
            drop_path_rate=self.drop_path_rate,
            fixed_dropout_depth=self.fixed_dropout_depth,
            norm_layer=self.norm_layer,
            act_layer=self.act_layer,
            mlp_layer=self.mlp_layer,
            init_std=self.init_std,
            abs_sincos_enc=self.abs_sincos_enc,
            rope_pos_enc=self.rope_pos_enc,
            rope_random_rotation_per_head=self.rope_random_rotation_per_head,
            rope_mixed=self.rope_mixed,
            rope_theta=self.rope_theta,
            mlp_wide_silu=mlp_wide_silu
        )

        self.task = task
        self.decoder = decoder
        self.loss_fn = get_loss_fn(loss_fn)

        if self.decoder == "vit":
            if model_template in CONFIGS.keys():
                self.decoder_embed_dim = CONFIGS[model_template].get("decoder_embed_dim", self.decoder_embed_dim)
                self.decoder_depth = CONFIGS[model_template].get("decoder_depth", self.decoder_depth)
                self.decoder_num_heads = CONFIGS[model_template].get("decoder_num_heads", self.decoder_num_heads)
            else:
                self.decoder_embed_dim = decoder_embed_dim
                self.decoder_depth = decoder_depth
                self.decoder_num_heads = decoder_num_heads

            self.masked_decoder = MaskedPredictor(
                input_fmt=self.input_fmt,
                input_shape=self.input_shape,
                lateral_patch_size=self.lateral_patch_size,
                axial_patch_size=self.axial_patch_size,
                temporal_patch_size=self.temporal_patch_size,
                channels=self.in_chans,
                input_embed_dim=self.embed_dim,
                output_embed_dim=self.masked_encoder.patch_embedding.pixels_per_patch * self.output_channels \
                    if self.task == "channel_split" else self.masked_encoder.patch_embedding.pixels_per_patch,
                embed_dim=self.decoder_embed_dim,
                depth=self.decoder_depth,
                num_heads=self.decoder_num_heads,
                mlp_ratio=self.mlp_ratio,
                proj_drop_rate=self.proj_drop_rate,
                att_drop_rate=self.att_drop_rate,
                drop_path_rate=self.drop_path_rate,
                fixed_dropout_depth=self.fixed_dropout_depth,
                norm_layer=self.norm_layer,
                act_layer=self.act_layer,
                mlp_layer=self.mlp_layer,
                init_std=self.init_std,
                abs_sincos_enc=self.abs_sincos_enc,
                rope_pos_enc=self.rope_pos_enc,
                rope_random_rotation_per_head=self.rope_random_rotation_per_head,
                rope_mixed=self.rope_mixed,
                rope_theta=self.rope_theta,
                mlp_wide_silu=mlp_wide_silu
            )
        
        elif self.decoder == "linear":
            self.decoder_with_bn = decoder_with_bn
            self.decoder_num_layers = decoder_num_layers
            self.decoder_hidden_dim = decoder_hidden_dim
            self.decoder_bottleneck_dim = decoder_bottleneck_dim
            self.decoder_mlp_bias = decoder_mlp_bias

            self.masked_decoder = LinearHead(
                in_dim=self.embed_dim,
                output_dim=self.masked_encoder.patch_embedding.pixels_per_patch * self.output_channels \
                    if self.task == "channel_split" else self.masked_encoder.patch_embedding.pixels_per_patch,
                use_bn=self.decoder_with_bn,
                nlayers=self.decoder_num_layers,
                hidden_dim=self.decoder_hidden_dim,
                bottleneck_dim=self.decoder_bottleneck_dim,
                mlp_bias=self.decoder_mlp_bias,
            )
        
        elif self.decoder == "dense_predictor":
            self.decoder_num_layers = decoder_num_layers_dense
            self.decoder_hidden_dim = decoder_hidden_dim_dense
            self.decoder_mlp_bias = decoder_mlp_bias_dense

            self.masked_decoder = DPTHead(
                input_fmt=self.input_fmt,
                input_shape=self.input_shape,
                lateral_patch_size=self.lateral_patch_size,
                axial_patch_size=self.axial_patch_size,
                temporal_patch_size=self.temporal_patch_size,
                channels=self.in_chans,
                input_embed_dim=self.embed_dim,
                output_embed_dim=self.masked_encoder.patch_embedding.pixels_per_patch * self.output_channels \
                    if self.task == "channel_split" else self.masked_encoder.patch_embedding.pixels_per_patch,
                hidden_dim=self.decoder_hidden_dim,
                num_layers=self.decoder_num_layers,
                mlp_bias=self.decoder_mlp_bias,
            )
        
        else:
            raise ValueError(f"Unknown decoder type: {self.decoder}")

        self.weight_init_type = weight_init_type
        init_weights(self, weight_init_type=weight_init_type)

    def forward(self, data_sample: dict):
        inputs, meta= data_sample['data_tensor'], data_sample['metainfo']
        masks, context_masks, targets = meta.get("masks", [None])[0], \
            meta.get('context_masks', [None])[0], meta.get('targets', [None])[0]
        target_masks, original_patch_indices = meta.get('target_masks', [None])[0], \
            meta.get('original_patch_indices', [None])[0]

        # x: List[B, N, C] or [B, N, C]
        if self.task == "upsample_time" or self.task == "upsample_spacetime":
            x, patches = self.encoder(inputs, masks=context_masks)
        else:
            x, patches = self.encoder(inputs)

        if self.task == "upsample_time" or self.task == "upsample_spacetime":
            x = self.decoder(x, original_patch_indices=original_patch_indices, target_masks=target_masks)
        else:
            x = self.decoder(x)

        if self.task == "channel_split":
            # predictions = self.prediction_head(x)
            loss = self.loss_fn(predictions, targets, num_patches=self.get_num_patches())
        elif self.task == "upsample_time":
            # only supervise the masked timepoints
            targets = apply_masks(patches, masks=target_masks)
            predictions = apply_masks(x, masks=target_masks)
            loss = self.loss_fn(predictions, targets, num_patches=masks.sum())
        elif self.task == "upsample_space" or self.task == "upsample_spacetime":
            predictions = x
            loss = self.loss_fn(x, targets, num_patches=self.get_num_patches())
        else:
            raise ValueError(f"Unknown task: {self.task}")

        loss_dict = {
            "step_loss": loss,
        }
        return loss_dict, predictions