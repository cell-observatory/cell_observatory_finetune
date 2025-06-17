import pytest
from collections import OrderedDict

from hydra import initialize, compose

import torch

from finetune.models.heads.roi_heads.roi_heads import maskrcnn_inference
from finetune.train.registry import build_dependency_graph_and_instantiate


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


def test_roi_heads(model):
    roi_heads = model.roi_heads
    rpn = model.rpn
    backbone = model.backbone
    transform = model.transform

    # test roi_heads class method select_training_samples

    # setup: 3 proposals, 3 targets
    proposals_test = [
        torch.tensor([
            [0,   0,   0, 100, 100, 100],
            [50, 50, 50,  75,  75,  75],
            [0,   0,   0,  10,  10,  10],
        ], dtype=torch.float32).cuda(),
    ]
    targets_test = [{
        "masks": [],
        "boxes": torch.tensor([
            [0,   0,   0, 100, 100, 100],
            [0,   0,   0,  10,  10,  10],
            [50, 50,  50,  75,  75,  75],
        ], dtype=torch.float32).cuda(),
        "labels": torch.tensor([1, 1, 0], dtype=torch.int64).cuda(),
    }]

    proposals_out, matched_idxs, labels, regression_targets = roi_heads.select_training_samples(
        proposals_test, targets_test
    )

    # check proposals concatenation:
    expected_props = torch.tensor([
        [  0,   0,   0, 100, 100, 100],
        [ 50,  50,  50,  75,  75,  75],
        [  0,   0,   0,  10,  10,  10],
        [  0,   0,   0, 100, 100, 100],
        [  0,   0,   0,  10,  10,  10],
        [ 50,  50,  50,  75,  75,  75],
    ], dtype=torch.float32).cuda()
    assert isinstance(proposals_out, list) and len(proposals_out) == 1
    assert torch.equal(proposals_out[0], expected_props)

    # check matched indices:
    expected_matched = torch.tensor([0, 2, 1, 0, 1, 2], dtype=torch.int64).cuda()
    assert isinstance(matched_idxs, list) and len(matched_idxs) == 1
    assert torch.equal(matched_idxs[0], expected_matched)

    # check labels:
    expected_labels = torch.tensor([1, 0, 1, 1, 1, 0], dtype=torch.int64).cuda()
    assert isinstance(labels, list) and len(labels) == 1
    assert torch.equal(labels[0], expected_labels)

    # check regression targets:
    assert isinstance(regression_targets, tuple) and len(regression_targets) == 1
    reg = regression_targets[0]
    assert isinstance(reg, torch.Tensor)
    assert reg.shape == (6, 6)
    assert torch.allclose(reg, torch.zeros_like(reg))

    # TODO: Implement unit test for postprocess_detections in roi_heads
    # TODO: Implement unit test for maskrcnn_inference in roi_heads

    # test roi_heads forward pass
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

    # RPN proposals
    proposals, _ = rpn(image, features, targets)

    # box roi pooling
    box_features = roi_heads.box_roi_pool(features, proposals, [image_shapes])
    assert box_features.dim() == 5 

    # box head and predictor
    box_features = roi_heads.box_head(box_features)
    assert box_features.dim() == 2  

    class_logits, box_regression = roi_heads.box_predictor(box_features)
    assert class_logits.dim() == 2 and box_regression.dim() == 2
    assert class_logits.shape[0] == box_regression.shape[0]

    # post-process detections
    boxes, scores, det_labels = \
        roi_heads.postprocess_detections(class_logits, box_regression, proposals, [image_shapes])
    assert isinstance(boxes, list) and isinstance(scores, list) and isinstance(det_labels, list)
    assert len(boxes) == len(scores) == len(det_labels) == 1
    assert boxes[0].dim() == 2 and boxes[0].shape[1] == 6
    assert scores[0].dim() == 1 and det_labels[0].dim() == 1

    # mask pathway
    result = []
    num_images = len(boxes)
    for i in range(num_images):
        result.append({"boxes": boxes[i], "labels": det_labels[i], "scores": scores[i]})

    if roi_heads.has_mask():
        mask_proposals = [r["boxes"] for r in result]
        assert roi_heads.mask_roi_pool is not None
        mask_features = roi_heads.mask_roi_pool(features, mask_proposals, [image_shapes])
        assert mask_features.dim() == 5  

        mask_features = roi_heads.mask_head(mask_features)
        assert mask_features.dim() == 5

        mask_logits = roi_heads.mask_predictor(mask_features)
        assert mask_logits.dim() == 5

        masks_probs = maskrcnn_inference(mask_logits, det_labels)
        assert len(masks_probs) == num_images

        for mask_prob, r in zip(masks_probs, result):
            assert isinstance(mask_prob, torch.Tensor)
            assert mask_prob.dim() == 5
            assert mask_prob.shape[0] == r["labels"].shape[0]