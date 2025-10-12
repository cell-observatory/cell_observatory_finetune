import sys
import logging
from typing import Literal, Union

import torch.nn as nn

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


class FinetuneJEPA(JEPA):
    def __init__(
        self,
        task: Literal['channel_split', 'upsample'],
        output_channels: int = None,
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
        predictor_embed_dim=256,
        depth=12,
        predictor_depth=8,
        num_heads=12,
        predictor_num_heads=8,
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
        loss_fn: str = 'l1_masked',
        **kwargs,
    ):
        super().__init__(model_template=model_template,
                         input_fmt=input_fmt,
                         input_shape=input_shape,
                         lateral_patch_size=lateral_patch_size,
                         axial_patch_size=axial_patch_size,
                        temporal_patch_size=temporal_patch_size,
                        embed_dim=embed_dim,
                        predictor_embed_dim=predictor_embed_dim,
                        depth=depth,
                        predictor_depth=predictor_depth,
                        num_heads=num_heads,
                        predictor_num_heads=predictor_num_heads,
                        mlp_ratio=mlp_ratio,
                        proj_drop_rate=proj_drop_rate,
                        att_drop_rate=att_drop_rate,
                        drop_path_rate=drop_path_rate,
                        init_std=init_std,
                        fixed_dropout_depth=fixed_dropout_depth,
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                        mlp_layer=mlp_layer,
                        abs_sincos_enc=abs_sincos_enc,
                        rope_pos_enc=rope_pos_enc,
                        rope_random_rotation_per_head=rope_random_rotation_per_head,
                        rope_mixed=rope_mixed,
                        rope_theta=rope_theta,
                        weight_init_type=weight_init_type,
                        mlp_wide_silu=mlp_wide_silu,
                            **kwargs)
        
        self.loss_fn = get_loss_fn(loss_fn)
        
        self.task = task
        if self.task == "channel_split":
            self.output_channels = output_channels
            self.output_dim = self.input_encoder.patch_embedding.pixels_per_patch * output_channels
            self.prediction_head = nn.Linear(self.target_predictor.output_embed_dim, self.output_dim)
        elif self.task == "upsample_space" \
            or self.task == "upsample_spacetime" \
                or self.task == "upsample_time":
            self.output_channels = output_channels
            self.output_dim = self.input_encoder.patch_embedding.pixels_per_patch
            self.prediction_head = nn.Linear(self.target_predictor.output_embed_dim, self.output_dim)
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def forward(self, data_sample: dict):
        inputs, meta= data_sample['data_tensor'], data_sample['metainfo']
        masks, context_masks, targets = meta.get("masks", [None])[0], \
            meta.get('context_masks', [None])[0], meta.get('targets', [None])[0]
        target_masks, original_patch_indices = meta.get('target_masks', [None])[0], \
            meta.get('original_patch_indices', [None])[0]
        
        embedding, patches = self.input_encoder(inputs, masks=context_masks)
        predictions = self.target_predictor(
            embedding,
            original_patch_indices=original_patch_indices,
            target_masks=target_masks
        )
        
        if self.task == "channel_split":
            predictions = self.prediction_head(predictions)
            loss = self.loss_fn(predictions, targets, num_patches=self.get_num_patches())
        elif self.task == "upsample_time":
                # only supervise the masked timepoints
                predictions = self.prediction_head(predictions)
                targets = apply_masks(patches, masks=target_masks)
                predictions = apply_masks(predictions, masks=target_masks)
                loss = self.loss_fn(predictions, targets, num_patches=masks.sum())
        elif self.task == "upsample_space" or self.task == "upsample_spacetime":
                predictions = self.prediction_head(predictions)
                loss = self.loss_fn(predictions, targets, num_patches=self.get_num_patches())
        else:
            raise ValueError(f"Unknown task: {self.task}")

        loss_dict = {
            "step_loss": loss,
        }
        return loss_dict, predictions