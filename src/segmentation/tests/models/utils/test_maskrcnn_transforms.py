import torch
import pytest

from omegaconf import OmegaConf
from hydra import initialize, compose

from segmentation.models.utils.transforms import resize_boxes
from segmentation.models.heads.roi_heads.roi_heads import paste_masks_in_image
from segmentation.training.registry import build_dependency_graph_and_instantiate


def test_resize_boxes_scales_correctly():
    # a single box in 128x128x128 image
    test_box = torch.tensor([[0.0, 0.0, 0.0, 60.0, 40.0, 128.0]])
    original_size = (128, 128, 128)
    new_size = (64, 64, 64)

    out = resize_boxes(test_box, original_size, new_size)

    expected = torch.tensor([[0.0, 0.0, 0.0, 30.0, 20.0, 64.0]])
    assert out.shape == expected.shape
    assert torch.allclose(out, expected)


def test_paste_masks_in_image_shape_and_content():
    mask = torch.ones((1, 1, 28, 28, 28), dtype=torch.float32)
    box = torch.tensor([[10, 10, 10, 100, 100, 100]], dtype=torch.int64)
    image_shape = (128, 128, 128)

    pasted = paste_masks_in_image(mask, box, image_shape)

    assert isinstance(pasted, torch.Tensor)
    assert pasted.shape == (1, 1, *image_shape)

    # voxels inside the box should be > 0 (mask got pasted)
    inside = pasted[0, 0, 10:100, 10:100, 10:100]
    assert inside.numel() > 0
    assert torch.all(inside > 0)

    # voxels outside the box should be zero
    assert pasted[0, 0, 0, 0, 0] == 0
    assert pasted[0, 0, 127, 127, 127] == 0


@pytest.fixture(scope="module")
def cfg():
    with initialize(config_path="../../../configs"):
        cfg = compose(
            config_name="config_mrcnn_resnet",
        )
    return cfg


@pytest.fixture(scope="module")
def transform(cfg):
    model = build_dependency_graph_and_instantiate(cfg.models)
    return model.transform


def test_transform_returns_tensors_and_targets(transform):
    inputs = torch.randn(1, 3, 128, 128, 128)
    example_box = torch.tensor([[0.0, 0.0, 0.0, 60.0, 40.0, 128.0]])
    targets = [{
        "masks": torch.randn(1, 128, 128, 128),
        "boxes": example_box.clone(),
        "labels": [1],
    }]

    images, new_targets = transform(inputs, targets)

    assert hasattr(images, "tensors")
    assert isinstance(images.tensors, torch.Tensor)

    assert isinstance(new_targets, list) and len(new_targets) == 1
    nt = new_targets[0]
    for key in ("masks", "boxes", "labels"):
        assert key in nt, f"missing '{key}' in transformed targets"

    assert torch.allclose(nt["boxes"], example_box, atol=1e-6)

    assert isinstance(nt["masks"], torch.Tensor)
    assert nt["masks"].shape[1:] == inputs.shape[2:]  # (D,H,W)

    assert isinstance(nt["labels"], list) and nt["labels"] == [1]