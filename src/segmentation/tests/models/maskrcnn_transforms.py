import os

import hydra
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf, open_dict

import torch
import torch.distributed as dist
from torch.distributed import init_process_group

from segmentation.transforms.transforms import resize_boxes 
from segmentation.data.gather_dataset import gather_dataset
from segmentation.models.heads.roi_heads.roi_heads import paste_masks_in_image
from segmentation.training.registry import build_dependency_graph_and_instantiate

import skimage.io as io


#NOTE: to test different backbones, change the config file in the hydra decorator
#      to your desired config file
@hydra.main(config_path="../../configs", config_name="config_mrcnn_resnet", version_base="1.2")
def main(cfg: DictConfig):
    # Print full configuration (for debugging)
    print("\n" + OmegaConf.to_yaml(cfg))

    with open_dict(cfg):
        cfg.datasets.batch_size = 1
        cfg.worker_batch_size = 1
        
    model = build_dependency_graph_and_instantiate(cfg.models)
    model = model.cuda()

    transform = model.transform

    inputs = torch.randn(1, 3, 128, 128, 128).cuda() 
    example_box = torch.tensor([0, 0, 0, 60, 40, 128]) 
    targets = [{"masks" : torch.randn(1, 128, 128, 128), "boxes" : example_box.unsqueeze(0), "labels" : [1]}]

    images, targets = transform(inputs, targets)

    print(f"Input shape: {images.tensors.shape}")
    for target in targets:
        for key, value in target.items():
            if isinstance(value, torch.Tensor):
                print(f"Feature map shape for {key}: {value.shape}")
            if isinstance(value, list):
                for i, v in enumerate(value):
                    if isinstance(v, torch.Tensor):
                        print(f"Feature map shape for {key}[{i}]: {v.shape}")

    # transform post-process tests
    
    # test resize_boxes 
    test_box = torch.tensor([[0, 0, 0, 60, 40, 128]])
    original_image_size = (128, 128, 128)
    resized_image_size = (64, 64, 64)

    resized_test_box = resize_boxes(test_box, original_image_size, resized_image_size)

    # expected bbox dims:
    # [0, 0, 0, 30, 20, 64]
    print(f"Original box: {test_box}")
    print(f"Resized box: {resized_test_box}")

    # test paste_masks_in_image
    mask   = torch.ones((28, 28, 28), dtype=torch.float32)
    mask   = mask.unsqueeze(0).unsqueeze(0)
    box    = torch.tensor([[10, 10, 10, 100, 100, 100]], dtype=torch.int64) 
    im_d, im_h, im_w = 128, 128, 128

    # mask will be interpolated to the size of the box
    # mask inside box will be placed inside the image based
    # on the box coordinates
    pasted = paste_masks_in_image(mask, box, (im_d, im_h, im_w))

    io.imsave("/clusterfs/nvme/segment_4d/test_5/mask_paste_test.tif", pasted.cpu().numpy())
    # print(f"Original mask: {mask}")
    # print(f"Original box: {box}")
    # print(f"Pasted mask: {pasted}")


if __name__ == "__main__":
    main()