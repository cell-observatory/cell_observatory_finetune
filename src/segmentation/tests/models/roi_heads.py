import os
from collections import OrderedDict

import hydra
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf, open_dict

import torch

from segmentation.data.gather_dataset import gather_dataset
from segmentation.models.rpn.rpn import concat_box_prediction_layers
from segmentation.training.registry import build_dependency_graph_and_instantiate

from segmentation.utils.plot import plot_boxes
from segmentation.models.heads.roi_heads.roi_heads import maskrcnn_inference

import skimage.io as io

@hydra.main(config_path="../../configs", config_name="skittlez_evaluation", version_base="1.2")
def main(cfg: DictConfig):
    # Print full configuration (for debugging)
    print("\n" + OmegaConf.to_yaml(cfg))

    with open_dict(cfg):
        cfg.datasets.batch_size = 1
        cfg.worker_batch_size = 1
        # cfg.distributed_sampler = False
        # cfg.datasets.database.distributed = False
        # cfg.datasets.split = None

    model = build_dependency_graph_and_instantiate(cfg.models)
    model = model.cuda()

    roi_heads = model.roi_heads
    rpn = model.rpn
    backbone = model.backbone
    transform = model.transform

    datloader = gather_dataset(cfg)
    image, targets = next(iter(datloader))

    image_shapes = image.shape[-3:]

    image = image.cuda().float()  
    targets = [{k: v.cuda().float() for k, v in t.items()} for t in targets] 
    
    image, targets = transform(image, targets)
    features = backbone(image.tensors)
    if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

    proposals, _ = rpn(image, features, targets)

    labels = None
    result = []

    # Roi-Align
    # ROI in image space (B,C,D,H,W) -> (B,C_out,D_out,H_out,W_out) in feature space
    # normally: C_out = 256 or 512 or 1024 or 2048, D_out = 7, H_out = 7, W_out = 7
    box_features = roi_heads.box_roi_pool(features, proposals, [image_shapes])
    print("box_features shape: ", box_features.shape)

    # TwoMLPHead: 
    # flatten (B,C*D*H*W) -> nn.Linear(in_channels, representation_size) -> nn.Linear(representation_size, representation_size)
    box_features = roi_heads.box_head(box_features)
    print("box_features shape after box_head: ", box_features.shape)

    # FastRCNNPredictor: 
    # nn.Linear(in_channels, num_classes) and nn.Linear(in_channels, 6 * num_classes)
    class_logits, box_regression = roi_heads.box_predictor(box_features)
    print("class_logits shape: ", class_logits.shape)
    print("box_regression shape: ", box_regression.shape)

    # remove small and empty boxes, NMS, keep top k boxes (generally k = low thousands)
    boxes, scores, labels = roi_heads.postprocess_detections(class_logits, box_regression, proposals, [image_shapes])

    print("boxes shape: ", boxes[0].shape)
    print("scores shape: ", scores[0].shape)
    print("labels shape: ", labels[0].shape)

    num_images = len(boxes)
    for i in range(num_images):
        result.append(
            {
                "boxes": boxes[i],
                "labels": labels[i],
                "scores": scores[i],
            }
        )

    if roi_heads.has_mask():
        mask_proposals = [p["boxes"] for p in result]

        if roi_heads.mask_roi_pool is not None:
            # Roi-Align
            # ROI in image space (B,C,D,H,W) -> (B,C_out,D_out,H_out,W_out) in feature space
            # normally: C_out = 256 or 512 or 1024 or 2048, D_out = 14, H_out = 14, W_out = 14
            mask_features = roi_heads.mask_roi_pool(features, mask_proposals, [image_shapes])
            print("mask_features shape: ", mask_features.shape)

            # [B,C_in,14,14,14] → [B,256,14,14,14] (x4) with Conv3dNormActivation
            mask_features = roi_heads.mask_head(mask_features)
            print("mask_features shape after mask_head: ", mask_features.shape)

            # upsample to [B,256,28,28,28] -> [B,class_num,28,28,28]
            mask_logits = roi_heads.mask_predictor(mask_features)
            print("mask_logits shape: ", mask_logits.shape)
        else:
            raise Exception("Expected mask_roi_pool to be not None")

        labels = [r["labels"] for r in result]
        masks_probs = maskrcnn_inference(mask_logits, labels)

        # expected output: one BoxList for each image
        print("masks_probs shape: ", masks_probs[0].shape)

        for mask_prob, r in zip(masks_probs, result):
            r["masks"] = mask_prob

    ################################ ################################ Separate Tests ################################ ################################

    # separate test of select_training_samples in roi_heads
    proposals_test = [torch.tensor(
        [[0, 0, 0, 100, 100, 100],      
         [50, 50, 50, 75, 75, 75],      
         [0,0,0,10,10,10]])]
    targets_test = [{"masks": [], 
                    "boxes": torch.tensor([[0, 0, 0, 100, 100, 100], [0,0,0,10,10,10], [50, 50, 50, 75, 75, 75]]),
                    "labels":  torch.tensor([1,1,0])}]
    proposals, matched_idxs, labels, regression_targets = roi_heads.select_training_samples(proposals_test, targets_test)

    print("proposals: ", proposals)
    print("matched_idxs: ", matched_idxs)
    print("labels: ", labels)
    print("regression_targets: ", regression_targets)

    # expected output:
    # proposals (concat. gt and preds):  
    # [tensor([[  0,   0,   0, 100, 100, 100],
    #         [ 50,  50,  50,  75,  75,  75],
    #         [  0,   0,   0,  10,  10,  10],
    #         [  0,   0,   0, 100, 100, 100],
    #         [  0,   0,   0,  10,  10,  10],
    #         [ 50,  50,  50,  75,  75,  75]])]
    # matched_idxs:  [tensor([0, 2, 1, 0, 1, 2])]
    # labels:  [tensor([1, 0, 1, 1, 1, 0])]
    # regression_targets:  
    # (tensor([[0., 0., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0.]]),)

    # TODO: Implement test
    # separate test of postprocess_detections in roi_heads

    # TODO: Implement test
    # separate test of maskrcnn_inference in roi_heads


if __name__ == "__main__":
    main()