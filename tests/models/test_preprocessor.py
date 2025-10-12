import torch
import pytest

from cell_observatory_finetune.data.utils import resize_mask
from cell_observatory_finetune.models.meta_arch.preprocessor import FinetunePreprocessor

BATCH = 32
TIME = 16
DEPTH = 128
HEIGHT = 128
WIDTH = 128
CHANNELS = 2

FMT_2D = "TYXC"
FMT_3D = "TZYXC"

SHAPE_2D = (BATCH, TIME, HEIGHT, WIDTH, CHANNELS)
SHAPE_3D = (BATCH, TIME, DEPTH, HEIGHT, WIDTH, CHANNELS)


# ---- helpers ----

def _delta_psf_2d(h: int, w: int) -> torch.Tensor:
    psf = torch.zeros((h, w), dtype=torch.float32)
    psf[h // 2, w // 2] = 1.0
    return psf


def _delta_psf_3d(d: int, h: int, w: int) -> torch.Tensor:
    psf = torch.zeros((d, h, w), dtype=torch.float32)
    psf[d // 2, h // 2, w // 2] = 1.0
    return psf


# ---- ---- ----

def test_requires_c_last_in_input_format():
    with pytest.raises(AssertionError):
        FinetunePreprocessor(
            task="channel_split",
            axial_patch_size=2,
            lateral_patch_size=4,
            temporal_patch_size=1,
            transforms_list=[],
            with_masking=False,
            mask_generator=None,
            dtype=torch.float32,
            input_format="TYXZ",
            input_shape=(BATCH, TIME, HEIGHT, WIDTH, DEPTH),
        )


def test_requires_xy_axes_present():
    with pytest.raises(ValueError):
        FinetunePreprocessor(
            task="channel_split",
            dtype=torch.float32,
            axial_patch_size=2,
            lateral_patch_size=4,
            temporal_patch_size=1,
            transforms_list=[],
            with_masking=False,
            mask_generator=None,
            input_format="TZYC",
            input_shape=(BATCH, TIME, DEPTH, HEIGHT, CHANNELS),
        )


@pytest.mark.parametrize(
    "fmt, full_shape, axial_patch_size",
    [
        (FMT_2D, SHAPE_2D, None),
        (FMT_3D, SHAPE_3D, 4),
    ],
)
def test_spatial_dims_and_indices(fmt, full_shape, axial_patch_size):
    expected_axis_index = {ax: idx+1 for idx, ax in enumerate(fmt)}
    axis_to_size = dict(zip(fmt, full_shape[1:]))

    expected_spatial_dims = tuple(
        expected_axis_index[ax]
        for ax in fmt
        if ax in ("Z", "Y", "X")
    )
    expected_axial_shape = axis_to_size.get("Z")
    expected_lateral_shape = (axis_to_size["Y"], axis_to_size["X"])
    expected_spatial_shape = (
        (expected_axial_shape, *expected_lateral_shape)
        if expected_axial_shape is not None
        else expected_lateral_shape
    )

    pp = FinetunePreprocessor(
        task="channel_split",
        dtype=torch.float32,
        axial_patch_size=axial_patch_size,
        lateral_patch_size=4,
        temporal_patch_size=1,
        transforms_list=[],
        with_masking=False,
        mask_generator=None,
        input_format=fmt,
        input_shape=full_shape,
    )

    assert pp.input_format == fmt
    assert pp.input_shape == full_shape
    assert pp.axis_index == expected_axis_index
    assert pp.spatial_dims == expected_spatial_dims

    assert pp.channel_idx == expected_axis_index.get("C")
    assert pp.time_idx == expected_axis_index.get("T")
    assert pp.z_idx == expected_axis_index.get("Z")
    assert pp.y_idx == expected_axis_index.get("Y")
    assert pp.x_idx == expected_axis_index.get("X")

    assert pp.axial_shape == expected_axial_shape
    assert pp.lateral_shape == expected_lateral_shape
    assert pp.spatial_shape == expected_spatial_shape
    assert pp.channels == axis_to_size.get("C")
    assert pp.timepoints == axis_to_size.get("T")


def test_channel_split_keeps_dim_and_means():
    B, T, Y, X, C = SHAPE_2D
    x = torch.zeros(SHAPE_2D, dtype=torch.float32)
    for c in range(C):
        x[..., c] = float(c)
    pp = FinetunePreprocessor(
        task="channel_split",
        dtype=torch.float32,
        axial_patch_size=None,
        lateral_patch_size=4,
        temporal_patch_size=1,
        transforms_list=[],
        with_masking=False,
        mask_generator=None,
        input_format=FMT_2D,
        input_shape=SHAPE_2D,
    )
    out = pp.forward({"data_tensor": x, "metainfo": {}}, data_time=0.0)
    y = out["data_tensor"]
    assert y.shape == (B, T, Y, X, 1)
    assert torch.allclose(y.mean(), torch.tensor((C - 1) / 2, dtype=torch.float32))


def test_resize_mask_broadcast_3d_with_time_and_channel():
    spatial = _delta_psf_3d(DEPTH, HEIGHT, WIDTH)
    mask = resize_mask(
        spatial,
        input_format=FMT_3D,
        channels=CHANNELS,
        timepoints=TIME,
        axial_shape=DEPTH,
        lateral_shape=(HEIGHT, WIDTH),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert mask.ndim == len(FMT_3D)
    assert mask.shape[FMT_3D.index("Z")] == DEPTH
    assert mask.shape[FMT_3D.index("Y")] == HEIGHT
    assert mask.shape[FMT_3D.index("X")] == WIDTH


def test_upsample_identity_with_delta_psf_2d(monkeypatch):
    def _read_file(_):
        return _delta_psf_2d(HEIGHT, WIDTH).numpy()

    monkeypatch.setattr(
        "cell_observatory_finetune.models.meta_arch.preprocessor.read_file",
        _read_file,
        raising=True,
    )

    fmt = FMT_2D
    shape = SHAPE_2D
    x = torch.randn(shape)

    pp = FinetunePreprocessor(
        task="upsample",
        with_masking=False,
        mask_generator=None,
        axial_patch_size=None,
        lateral_patch_size=4,
        temporal_patch_size=1,
        transforms_list=[],
        dtype=torch.float32,
        input_format=fmt,
        input_shape=shape,
        ideal_psf_path="ignored",
        na_mask_thresholds=[0.0],
        seed=123,
    )
    out = pp.forward({"data_tensor": x.clone(), "metainfo": {}}, data_time=0.0)
    y = out["data_tensor"]
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-5)