import os

import hydra
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf, open_dict

import torch
import torch.distributed as dist
from torch.distributed import init_process_group

from segmentation.data.gather_dataset import gather_dataset
from segmentation.training.registry import build_dependency_graph_and_instantiate


# def setup_singleton_pg(backend: str = None):
#     """
#     Create a dummy 1‑process process group so that
#     torch.distributed calls (get_rank(), barrier(), etc.)
#     all work.
#     """

#     if dist.is_initialized():
#         return

#     if backend is None:
#         backend = "nccl" if torch.cuda.is_available() else "gloo"

#     dist.init_process_group(
#         backend=backend,
#         world_size=1,
#         rank=0,
#     )


#NOTE: to test different backbones, change the config file in the hydra decorator
#      to your desired config file
@hydra.main(config_path="../../configs", config_name="config_mrcnn_resnet", version_base="1.2")
def main(cfg: DictConfig):
    # Print full configuration (for debugging)
    print("\n" + OmegaConf.to_yaml(cfg))

    # setup_singleton_pg()

    with open_dict(cfg):
        cfg.datasets.batch_size = 1
        cfg.worker_batch_size = 1
        
    # get dataset
    # test_dataloader = gather_dataset(cfg)

    model = build_dependency_graph_and_instantiate(cfg.models)
    model = model.cuda()

    backbone = model.backbone

    inputs = torch.randn(1, 3, 128, 128, 128).cuda() # Example input tensor
    feature_map = backbone(inputs)

    print(f"Input shape: {inputs.shape}")
    print(f"Feature map shape: {feature_map.shape}")

    # for step, (inputs, targets) in enumerate(test_dataloader):
    #     inputs = inputs.cuda()

    #     feature_map = backbone(inputs)

    #     print(f"Step {step}: Input shape: {inputs.shape}")
    #     print(f"Step {step}: Feature map shape: {feature_map.shape}")

if __name__ == "__main__":
    main()