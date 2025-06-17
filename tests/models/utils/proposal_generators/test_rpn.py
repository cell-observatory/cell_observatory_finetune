import pytest
from collections import OrderedDict

from hydra import initialize, compose

import torch

from cell_observatory_finetune.models.utils.proposal_generators.anchor_generator import concat_box_prediction_layers
from cell_observatory_finetune.train.registry import build_dependency_graph_and_instantiate


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


def test_rpn(model):
    rpn = model.rpn
    backbone = model.backbone
    transform = model.transform

    image = torch.randn(1, 3, 128, 128, 128).cuda()
    targets = [{
        "boxes": torch.tensor([[0, 0, 0, 60, 40, 128]], dtype=torch.float32).cuda(),
        "labels": torch.tensor([1], dtype=torch.int64).cuda(),
        "masks": torch.ones((1, 128, 128, 128), dtype=torch.uint8).cuda(),
    }]
    image_shapes = image.shape[-3:]

    # transform and backbone
    image, targets = transform(image, targets)
    features = backbone(image.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    features_list = list(features.values())

    # RPN proposal generation
    objectness, pred_bbox_deltas = rpn.head(features_list)
    anchors = rpn.anchor_generator(image, features_list)
    num_images = len(anchors)

    # check prediction list shapes
    assert isinstance(objectness, list) and isinstance(pred_bbox_deltas, list)
    assert len(objectness) == len(pred_bbox_deltas) > 0
    for obj in objectness:
        assert obj.dim() == 5 
        assert obj.shape[0] == num_images

    # concatenate all feature levels
    num_anchors_per_level = [o[0].numel() for o in objectness]
    objectness_flat, pred_bbox_deltas_flat = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    assert isinstance(objectness_flat, torch.Tensor)
    assert isinstance(pred_bbox_deltas_flat, torch.Tensor)

    total_anchors = sum(num_anchors_per_level) * num_images
    assert objectness_flat.numel() == total_anchors
    assert pred_bbox_deltas_flat.shape == (total_anchors, 6)

    # decode and filter proposals
    proposals = rpn.box_coder.decode(pred_bbox_deltas_flat.detach(), anchors)
    proposals = proposals.view(num_images, -1, 6)
    boxes, scores = rpn.filter_proposals(proposals, objectness_flat, image.image_sizes, num_anchors_per_level)
    assert isinstance(boxes, list) and isinstance(scores, list)
    assert len(boxes) == num_images
    assert boxes[0].dim() == 2 and boxes[0].shape[1] == 6
    assert scores[0].dim() == 1 and scores[0].shape[0] == boxes[0].shape[0]

    # test target assignment
    box_gt = [{"boxes": torch.tensor([[0, 0, 0, 60, 40, 128]], dtype=torch.float32)}]
    anchors_test = torch.tensor([[0, 0, 0, 60, 40, 128], [0, 0, 0, 20, 30, 100]], dtype=torch.float32)
    labels, matched_idx = rpn.assign_targets_to_anchors([anchors_test], box_gt)
    assert isinstance(labels, list) and isinstance(matched_idx, list)
    assert len(labels) == 1 and len(matched_idx) == 1
    # expected: first anchor matched (positive), second background
    expected_labels = torch.tensor([1, 0], dtype=labels[0].dtype)
    assert torch.equal(labels[0], expected_labels)
    expected_matched = torch.tensor([[0, 0, 0, 60, 40, 128], [0, 0, 0, 60, 40, 128]], dtype=matched_idx[0].dtype)
    assert torch.equal(matched_idx[0], expected_matched)