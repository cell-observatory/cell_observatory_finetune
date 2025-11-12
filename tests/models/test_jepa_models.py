import logging
logger = logging.getLogger(__name__)

import shutil
import pytest
pytestmark = pytest.mark.skip(reason="This module is temporarily disabled till we add ops3d to the docker image")

import torch

import numpy as np

from cell_observatory_finetune.tests.conftest import models_kargs, config
from cell_observatory_finetune.models.meta_arch.jepa import FinetuneJEPA
from cell_observatory_finetune.models.meta_arch.preprocessor import FinetunePreprocessor

from cell_observatory_platform.training.helpers import summarize_model
from cell_observatory_platform.data.masking.mask_generator import MaskGenerator, MaskModes


def _delta_psf_3d(d,h,w):
    psf = np.zeros((d,h,w), dtype=np.float32)
    psf[d//2, h//2, w//2] = 1.0
    return psf


@pytest.mark.parametrize("task", ["channel_split", "upsample_time", "upsample_space", "upsample_spacetime"])
def test_jepa_custom(models_kargs, config, task, monkeypatch):
    def _read_file(_ignored):
        return _delta_psf_3d(64, 64, 64)
    monkeypatch.setattr(
        "cell_observatory_finetune.models.meta_arch.preprocessor.read_file",
        _read_file,
        raising=True,
    )

    outdir = models_kargs['outdir']/"tests/baseline/custom"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)

    fmt_no_batch = "TZYXC"
    B, T, Z, Y, X, C = 1, 8, 64, 64, 64, 2
    inputs = (B, T, Z, Y, X, C)

    if task == "channel_split":
        # input has C=2; preprocessor will mean over C -> model sees C=1
        model = FinetuneJEPA(
            task="channel_split",
            decoder_args=dict(
                name="vit",
                decoder_embed_dim=1024,
                decoder_depth=8,
                decoder_num_heads=8
            ),
            decoder="vit",
            output_channels=2,
            model_template="mae",
            input_fmt=fmt_no_batch,
            input_shape=(T, Z, Y, X, 1),  # model expects C=1 after split
            embed_dim=models_kargs["hidden_size"],
            patch_shape=(1, 1, models_kargs["patches"], models_kargs["patches"], None),
            num_heads=models_kargs["heads"],
            depth=models_kargs["repeats"],
            proj_drop_rate=models_kargs["dropout"],
            fixed_dropout_depth=models_kargs["fixed_dropout_depth"],
            rope_pos_enc=False,
            abs_sincos_enc=True,

        ).to("cuda")

        preprocessor = FinetunePreprocessor(
            with_masking=False,
            mask_generator=None,
            task="channel_split",
            dtype=torch.float32,
            input_shape=(T, Z, Y, X, C),
            patch_shape=(1, 1, models_kargs["patches"], models_kargs["patches"], None),
            input_format=fmt_no_batch,
            transforms_list=[]
        )

    elif task == "upsample_time":
        # Model ingests the same C=2 layout; masking operates in time only via BLOCKED_PATTERNED
        model = FinetuneJEPA(
            task="upsample_time",
            output_channels=2,
            decoder_args=dict(
                name="vit",
                decoder_embed_dim=1024,
                decoder_depth=8,
                decoder_num_heads=8
            ),
            decoder="vit",
            model_template="mae",
            input_fmt=fmt_no_batch,
            input_shape=(T, Z, Y, X, C),
            embed_dim=models_kargs["hidden_size"],
            patch_shape=(1, 1, models_kargs["patches"], models_kargs["patches"], None),
            num_heads=models_kargs["heads"],
            depth=models_kargs["repeats"],
            proj_drop_rate=models_kargs["dropout"],
            fixed_dropout_depth=models_kargs["fixed_dropout_depth"],
            rope_pos_enc=False,
            abs_sincos_enc=True,
        ).to("cuda")

        mask_generator = MaskGenerator(
            layout=fmt_no_batch,
            input_shape=(T, Z, Y, X, C),
            patch_shape=(1, 1, models_kargs["patches"], models_kargs["patches"], None),
            lateral_mask_scale=(1.0, 1.0),
            axial_mask_scale=(1.0, 1.0),
            temporal_mask_scale=(1.0, 1.0),
            aspect_ratio_scale_hw=(1.0, 1.0),
            num_blocks=1,
            random_masking_ratio=0.0,
            channels_to_mask=None,
            time_downsample_pattern=[0, 1],
            mask_mode=MaskModes.BLOCKED_PATTERNED,
            device="cuda",
        )

        preprocessor = FinetunePreprocessor(
            with_masking=True,
            mask_generator=mask_generator,
            task="upsample_time",
            dtype=torch.float32,
            input_format=fmt_no_batch,
            input_shape=(T, Z, Y, X, C),
            patch_shape=(1, 1, models_kargs["patches"], models_kargs["patches"], None),
            ideal_psf_path="ignored",
            na_mask_thresholds=[0.0],
            seed=123,
            transforms_list=[]
        )

    elif task == "upsample_space":
        # Spatial NA masking only; no temporal masking generator
        model = FinetuneJEPA(
            task="upsample_space",
            output_channels=2,
            decoder_args=dict(
                name="vit",
                decoder_embed_dim=1024,
                decoder_depth=8,
                decoder_num_heads=8
            ),
            decoder="vit",
            model_template="mae",
            input_fmt=fmt_no_batch,
            input_shape=(T, Z, Y, X, C),
            embed_dim=models_kargs["hidden_size"],
            patch_shape=(1, 1, models_kargs["patches"], models_kargs["patches"], None),
            num_heads=models_kargs["heads"],
            depth=models_kargs["repeats"],
            proj_drop_rate=models_kargs["dropout"],
            fixed_dropout_depth=models_kargs["fixed_dropout_depth"],
            rope_pos_enc=False,
            abs_sincos_enc=True,
        ).to("cuda")

        preprocessor = FinetunePreprocessor(
            with_masking=False,
            mask_generator=None,
            task="upsample_space",
            dtype=torch.float32,
            input_shape=(T, Z, Y, X, C),
            input_format=fmt_no_batch,
            patch_shape=(1, 1, models_kargs["patches"], models_kargs["patches"], None),
            ideal_psf_path="ignored",
            na_mask_thresholds=[0.0],
            seed=123,
            transforms_list=[]
        )

    elif task == "upsample_spacetime":
        model = FinetuneJEPA(
            task="upsample_spacetime",
            output_channels=2,
            decoder_args=dict(
                name="vit",
                decoder_embed_dim=1024,
                decoder_depth=8,
                decoder_num_heads=8
            ),
            decoder="vit",
            model_template="mae",
            input_fmt=fmt_no_batch,
            input_shape=(T, Z, Y, X, C),
            embed_dim=models_kargs["hidden_size"],
            patch_shape=(1, 1, models_kargs["patches"], models_kargs["patches"], None),
            num_heads=models_kargs["heads"],
            depth=models_kargs["repeats"],
            proj_drop_rate=models_kargs["dropout"],
            fixed_dropout_depth=models_kargs["fixed_dropout_depth"],
            rope_pos_enc=False,
            abs_sincos_enc=True,
        ).to("cuda")

        mask_generator = MaskGenerator(
            layout=fmt_no_batch,
            input_shape=(T, Z, Y, X, C),                       
            patch_shape=(1, 1, models_kargs["patches"], models_kargs["patches"]),                       
            lateral_mask_scale=(1.0, 1.0),                     
            axial_mask_scale=(1.0, 1.0),
            temporal_mask_scale=(1.0, 1.0),
            aspect_ratio_scale_hw=(1.0, 1.0),
            num_blocks=1,
            random_masking_ratio=0.0,
            channels_to_mask=None,
            time_downsample_pattern=[0, 1],
            mask_mode=MaskModes.BLOCKED_PATTERNED,
            device="cuda",
        )

        preprocessor = FinetunePreprocessor(
            with_masking=True,
            mask_generator=mask_generator,
            task="upsample_spacetime",
            dtype=torch.float32,
            input_format=fmt_no_batch,
            input_shape=(T, Z, Y, X, C),
            patch_shape=(1, 1, models_kargs["patches"], models_kargs["patches"], None),
            ideal_psf_path="ignored",
            na_mask_thresholds=[0.0],
            seed=123,
            transforms_list=[]
        )

    else:
        raise ValueError(f"Unknown task: {task}")

    batch = {
        "data_tensor": torch.randn(inputs, device="cuda", dtype=torch.float32),
        "metainfo": {}
    }
    data_sample = preprocessor.forward(batch, data_time=0.0)

    assert data_sample["data_tensor"].shape[0] == B
    assert data_sample["data_tensor"].device.type == "cuda"

    # to prevent arg errors in summarize_model
    if task == "upsample_time":
        data_sample["metainfo"]["targets"] = torch.zeros_like(data_sample["data_tensor"])

    summarize_model(
        model=model,
        input_data=(data_sample,),
        batch_size=models_kargs["batch_size"],
        logdir=logdir,
    )