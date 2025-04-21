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

    rpn = model.rpn
    backbone = model.backbone
    transform = model.transform

    datloader = gather_dataset(cfg)
    image, targets = next(iter(datloader))

    image = image.cuda().float()  
    targets = [{k: v.cuda().float() for k, v in t.items()} for t in targets] 
    
    image, targets = transform(image, targets)
    features = backbone(image.tensors)
    if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

    features = list(features.values())
    objectness, pred_bbox_deltas = rpn.head(features) 
    anchors = rpn.anchor_generator(image, features) 

    num_images = len(anchors)

    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness] 
    num_anchors_per_level = [s[0] * s[1] * s[2] * s[3] for s in num_anchors_per_level_shape_tensors]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)

    proposals = rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 6)
    
    boxes, scores = rpn.filter_proposals(proposals, objectness, image.image_sizes, num_anchors_per_level)
   
    print(f"Boxes shape: {boxes[0].shape}")
    print(f"Scores shape: {scores[0].shape}")

    # plot top 5 boxes by confidence score to test rpn outputs
    # together with the image
    sorted_indices = torch.argsort(scores[0], descending=True)
    boxes = boxes[0][sorted_indices]
    plot_boxes(boxes=boxes.cpu().numpy(), 
               sample_indices=[0,1,2,3,4,5], 
               image_shape=image.image_sizes[0],
               save_path="/clusterfs/nvme/segment_4d/test_5/test_rpn_boxes.tif")  

    io.imsave("/clusterfs/nvme/segment_4d/test_5/test_rpn_im.tif", image.tensors[0][0].cpu().numpy())

    # TODO:
    # test filter proposals function separately from rpn
    # filter proposals does: remove small boxes, remove low scoring boxes, apply NMS, keep topk scoring predictions
    
    # check: 
    # set small boxes to lower 5% of boxes 
    # set low scoring boxes thresh to lowest 5% of scores
    # set topk to 100 less than resulting boxes

    # test assign_targets_to_anchors (used for training RPN) separately from rpn
    box_gt = [{"boxes" : torch.tensor([[0, 0, 0, 60, 40, 128]])}]
    anchors = torch.tensor([[0, 0, 0, 60, 40, 128], [0, 0, 0, 20, 30, 100]])

    labels, matched_gt_index = rpn.assign_targets_to_anchors([anchors], box_gt)

    # should return:
    # labels : [1, 0] 
    # matched_gt_index : [tensor([[  0,   0,   0,  60,  40, 128], [  0,   0,   0,  60,  40, 128]])]
    print(f"Labels: {labels}")
    print(f"Matched GT index: {matched_gt_index}")


if __name__ == "__main__":
    main()