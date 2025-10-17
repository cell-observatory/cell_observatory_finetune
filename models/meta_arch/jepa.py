import sys
import logging
from copy import deepcopy
from typing import Literal, Union, Optional

import torch
import torch.nn as nn

from deepspeed.runtime.zero import GatheredParameters
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from cell_observatory_finetune.cell_observatory_platform.models.mlp import get_mlp
from cell_observatory_finetune.cell_observatory_platform.models.norm import get_norm
from cell_observatory_finetune.cell_observatory_platform.training.helpers import init_weights
from cell_observatory_finetune.cell_observatory_platform.models.activation import get_activation
from cell_observatory_finetune.cell_observatory_platform.models.patch_embeddings import calc_num_patches

from cell_observatory_finetune.cell_observatory_platform.models.maskedencoder import MaskedEncoder
from cell_observatory_finetune.cell_observatory_platform.models.maskedpredictor import MaskedPredictor

from cell_observatory_finetune.models.heads.linear_head import LinearHead
from cell_observatory_finetune.models.heads.dense_predictor_head import DPTHead

logging.basicConfig(
	stream=sys.stdout,
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from cell_observatory_finetune.training.losses import get_loss_fn

from cell_observatory_platform.models.jepa import JEPA
from cell_observatory_platform.data.masking.mask_generator import apply_masks

CONFIGS = {
    'jepa-tiny': {
        'embed_dim': 192,
        'predictor_embed_dim': 96,
        'depth': 12,
        'predictor_depth': 3,
        'num_heads': 3,
        'predictor_num_heads': 3,
        'mlp_ratio': 4,
    },
    'jepa-small': {
        'embed_dim': 384,
        'predictor_embed_dim': 192,
        'depth': 12,
        'predictor_depth': 6,
        'num_heads': 6,
        'predictor_num_heads': 6,
        'mlp_ratio': 4,
    },
    'jepa-base': {
        'embed_dim': 768,
        'predictor_embed_dim': 384,
        'depth': 12,
        'predictor_depth': 12,
        'num_heads': 12,
        'predictor_num_heads': 12,
        'mlp_ratio': 4,
    },
    'jepa-large': {
        'embed_dim': 1024,
        'predictor_embed_dim': 384,
        'depth': 24,
        'predictor_depth': 12,
        'num_heads': 16,
        'predictor_num_heads': 12,
        'mlp_ratio': 4,
    },
    'jepa-huge': {
        'embed_dim': 1280,
        'predictor_embed_dim': 384,
        'depth': 32,
        'predictor_depth': 12,
        'num_heads': 16,
        'predictor_num_heads': 12,
        'mlp_ratio': 4,
    },
    'jepa-2billion': {
        'embed_dim': 2560,
        'predictor_embed_dim': 512,
        'depth': 24,
        'predictor_depth': 8,
        'num_heads': 32,
        'predictor_num_heads': 8,
        'mlp_ratio': 4,
    },
    'jepa-6billion': {
        'embed_dim': 4096,
        'predictor_embed_dim': 512,
        'depth': 32,
        'predictor_depth': 8,
        'num_heads': 32,
        'predictor_num_heads': 8,
        'mlp_ratio': 4,
    },
    'jepa-giant': {
        'embed_dim': 1408,
        'predictor_embed_dim': 512,
        'depth': 40,
        'predictor_depth': 12,
        'num_heads': 16,
        'predictor_num_heads': 12,
        'mlp_ratio': 48/11,
    },
    'jepa-gigantic': {
        'embed_dim': 1664,
        'predictor_embed_dim': 1024,
        'depth': 48,
        'predictor_depth': 16,
        'num_heads': 16,
        'predictor_num_heads': 16,
        'mlp_ratio': 64/13,
    },
    'jepa-enormous': {
        'embed_dim': 1792,
        'predictor_embed_dim': 1024,
        'depth': 56,
        'predictor_depth': 16,
        'num_heads': 16,
        'predictor_num_heads': 16,
        'mlp_ratio': 8.5714285714,
    }
}


class FinetuneJEPA(nn.Module):
    def __init__(
        self,
        decoder_args: dict,
        decoder: Literal['vit', 'linear', 'dense_predictor'],
        task: Literal['channel_split', 
                      'upsample_time', 
                      'upsample_space', 
                      'upsample_spacetime'],
        output_channels: Optional[int],
        model_template: Literal[
            'jepa', # custom use `embed_dim`, `predictor_embed_dim`, `depth`, `num_heads` and `mlp_ratio` to config model
            'jepa-tiny',
            'jepa-small',
            'jepa-base',
            'jepa-large',
            'jepa-huge',
            'jepa-giant',
            'jepa-gigantic'
        ] = 'jepa',
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
        weight_init_type: str = 'vjepa2',
        mlp_wide_silu: bool = False,
        loss_fn: str = 'l1_masked'
    ):
        super().__init__()
        
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
        self.num_frames = axis_to_value.get("T", None)

        self.output_channels = output_channels

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
        self.mlp_wide_silu = mlp_wide_silu
        self.rope_random_rotation_per_head = rope_random_rotation_per_head

        self.input_encoder = MaskedEncoder(
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
        if self.task == "upsample_time" or self.task == "upsample_spacetime":
            assert self.decoder == "vit", "For upsample_time and upsample_spacetime tasks " \
                "only 'vit' decoder is supported currently."
    
        if self.decoder == "vit":
            if model_template in CONFIGS.keys():
                self.decoder_embed_dim = CONFIGS[model_template]["decoder_embed_dim"]
                self.decoder_depth = CONFIGS[model_template]["decoder_depth"]
                self.decoder_num_heads = CONFIGS[model_template]["decoder_num_heads"]
            else:
                self.decoder_embed_dim = decoder_args["decoder_embed_dim"]
                self.decoder_depth = decoder_args["decoder_depth"]
                self.decoder_num_heads = decoder_args["decoder_num_heads"]

            self.target_predictor = MaskedPredictor(
                input_fmt=self.input_fmt,
                input_shape=self.input_shape,
                lateral_patch_size=self.lateral_patch_size,
                axial_patch_size=self.axial_patch_size,
                temporal_patch_size=self.temporal_patch_size,
                channels=self.in_chans,
                input_embed_dim=self.embed_dim,
                output_embed_dim=self.input_encoder.patch_embedding.pixels_per_patch * self.output_channels \
                    if self.task == "channel_split" else self.input_encoder.patch_embedding.pixels_per_patch,
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
            self.decoder_with_bn = decoder_args["decoder_with_bn"]
            self.decoder_num_layers = decoder_args["decoder_num_layers"]
            self.decoder_hidden_dim = decoder_args["decoder_hidden_dim"]
            self.decoder_bottleneck_dim = decoder_args["decoder_bottleneck_dim"]
            self.decoder_mlp_bias = decoder_args["decoder_mlp_bias"]

            self.target_predictor = LinearHead(
                in_dim=self.embed_dim,
                output_dim=self.input_encoder.patch_embedding.pixels_per_patch * self.output_channels \
                    if self.task == "channel_split" else self.input_encoder.patch_embedding.pixels_per_patch,
                use_bn=self.decoder_with_bn,
                nlayers=self.decoder_num_layers,
                hidden_dim=self.decoder_hidden_dim,
                bottleneck_dim=self.decoder_bottleneck_dim,
                mlp_bias=self.decoder_mlp_bias,
            )
        
        elif self.decoder == "dense_predictor":
            assert decoder_args["encoder_out_layers"] is not None, \
                "For dense_predictor decoder, please specify the encoder out_layers to use."
            self.decoder_use_bn = decoder_args["decoder_use_bn"]
            self.decoder_feature_map_channels = decoder_args["decoder_feature_map_channels"]
            self.decoder_strategy = decoder_args["decoder_strategy"]
            self.decoder_embed_dim = decoder_args["decoder_embed_dim"]

            self.target_predictor = DPTHead(
                input_format=self.input_fmt,
                input_shape=self.input_shape,
                lateral_patch_size=self.lateral_patch_size,
                axial_patch_size=self.axial_patch_size,
                temporal_patch_size=self.temporal_patch_size,
                input_channels=self.embed_dim,
                output_channels=self.output_channels,
                features=self.decoder_embed_dim,
                use_bn=self.decoder_use_bn,
                feature_map_channels=self.decoder_feature_map_channels,
                strategy=self.decoder_strategy
            )
        
        # TODO: add support for segmentation decoders
        else:
            raise ValueError(f"Unknown decoder type: {self.decoder}")
        
        self.weight_init_type = weight_init_type
        init_weights(self, weight_init_type=weight_init_type)

        # NOTE: do deepcopy after weight init
        self.target_encoder = deepcopy(self.input_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.loss_fn = get_loss_fn(loss_fn)
        
    # see training/hooks.py for usage
    def ema_update(self, beta=0.99):
        def collect_params(params):
            return [
                p for p in params
                if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
            ]

        with torch.no_grad():
            for iparam, tparam in zip(self.input_encoder.parameters(), self.target_encoder.parameters()):
                fetch = collect_params([iparam, tparam])
                # fetches parameters from other ranks if needed
                with GatheredParameters(fetch, enabled=len(fetch) > 0):
                    # input_encoder*B + (target_encoder - input_encoder)*(1-B) = target_encoder*B + input_encoder*(1-B)
                    tparam.data.copy_(torch.lerp(iparam.data, tparam.data, beta))

    @torch.jit.ignore
    def get_input_encoder(self):
        return self.input_encoder

    @torch.jit.ignore
    def get_target_encoder(self):
        return self.target_encoder

    @torch.jit.ignore
    def get_predictor(self):
        return self.target_predictor

    @torch.jit.ignore
    def get_num_patches(self):
        if self.abs_sincos_enc:
            return self.input_encoder.pos_embedding.num_patches
        else:
            num_patches, _ = calc_num_patches(
                input_fmt=self.input_fmt,
                input_shape=self.input_shape,
                lateral_patch_size=self.lateral_patch_size,
                axial_patch_size=self.axial_patch_size,
                temporal_patch_size=self.temporal_patch_size,
            )
            return num_patches

    def forward(self, data_sample: dict):
        inputs, meta= data_sample['data_tensor'], data_sample['metainfo']
        masks, context_masks, targets = meta.get("masks", [None])[0], \
            meta.get('context_masks', [None])[0], meta.get('targets', [None])[0]
        target_masks, original_patch_indices = meta.get('target_masks', [None])[0], \
            meta.get('original_patch_indices', [None])[0]
        
        # x: List[B, N, C] or [B, N, C]
        if self.task == "upsample_time" or self.task == "upsample_spacetime":
            x, patches = self.input_encoder(inputs, masks=context_masks)
        else:
            x, patches = self.input_encoder(inputs)
        
        if self.task == "upsample_time" or self.task == "upsample_spacetime":
            x = self.target_predictor(x, original_patch_indices=original_patch_indices, target_masks=target_masks)
        else:
            x = self.target_predictor(x)

        if self.task == "channel_split":
            predictions = x
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
    

    def predict(self, data_sample: dict):
        inputs, meta= data_sample['data_tensor'], data_sample['metainfo']
        masks, context_masks, targets = meta.get("masks", [None])[0], \
            meta.get('context_masks', [None])[0], meta.get('targets', [None])[0]
        target_masks, original_patch_indices = meta.get('target_masks', [None])[0], \
            meta.get('original_patch_indices', [None])[0]
        
        # x: List[B, N, C] or [B, N, C]
        if self.task == "upsample_time" or self.task == "upsample_spacetime":
            x, patches = self.input_encoder(inputs, masks=context_masks)
        else:
            x, patches = self.input_encoder(inputs)
        
        if self.task == "upsample_time" or self.task == "upsample_spacetime":
            x = self.target_predictor(x, original_patch_indices=original_patch_indices, target_masks=target_masks)
        else:
            x = self.target_predictor(x)

        x = self.input_encoder.patch_embedding.unpatchify(x, out_channels=self.output_channels if self.task == "channel_split" else None)
        return x