import pytest

import torch

from hydra import initialize, compose

from segmentation.training.registry import build_dependency_graph_and_instantiate


@pytest.fixture(scope="module")
def cfg():
    with initialize(config_path="../../../configs"):
        cfg = compose(
            config_name="config_mrcnn_resnet",
        )
    return cfg


@pytest.fixture(scope="module")
def model(cfg):
    model = build_dependency_graph_and_instantiate(cfg.models)
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        pytest.skip("GPU required for this test")
    return model


def test_resnet_forward_shape(model):
    backbone = model.backbone
    x = torch.randn(1, 3, 128, 128, 128)
    x = x.cuda()

    feat = backbone(x)
    assert isinstance(feat, torch.Tensor)
    assert feat.shape[0] == 1 and  feat.shape[1] == 512 and feat.shape[2] == 16 and feat.shape[3] == 16 and feat.shape[4] == 16