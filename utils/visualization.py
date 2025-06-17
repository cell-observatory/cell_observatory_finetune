import warnings
from enum import Enum
from typing import Literal, Union, List, Dict, Any

import torch

from cell_observatory_finetune.data.utils import save_file
from cell_observatory_finetune.data.structures.data_objects.boxes import Boxes
from cell_observatory_finetune.data.structures.data_objects.masks import BitMasks


class COLOR_MAP(Enum):
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    BLACK = (0, 0, 0)
    YELLOW = (255, 255, 0)
    CYAN = (0, 255, 255)
    WHITE = (255, 255, 255)


class Visualizer:
    def __init__(self, 
                 save_format: Literal["zarr", "tiff"], 
                 save_metadata: Dict[str, Any]
    ):
        self.save_format = save_format
        self.save_metadata = save_metadata

    # TODO: support new boxes format: (B, (T), 6)
    def visualize_boxes(self, 
                        image_size: tuple[int], 
                        boxes: list[Boxes],
                        edge_color: Union[COLOR_MAP, str, tuple[int]] = COLOR_MAP.RED,
                        line_width: int = 2,
    ):
        # inside_box expects: (x1,y1,z1,x2,y2,z3) box format
        valid = boxes.inside_box([0,0,0] + list(image_size[::-1]))
        if not all(valid):
            warnings.warn("Some boxes are invalid. Skipping visualization for those boxes.")
        bboxes = boxes.tensor[valid]

        device = bboxes.device
        D, H, W = image_size
        vol = torch.zeros((3, D, H, W), dtype=torch.uint8, device=device)

        c = torch.tensor(self._get_color(edge_color),
                         dtype=torch.uint8,
                         device=bboxes.device).view(3, 1, 1, 1) 
        w = line_width
        for (x1, y1, z1, x2, y2, z2) in bboxes.round().int().tolist():
            # redundant check for valid coordinates, keeping 
            # for now out of caution
            x1, y1, z1 = max(x1, 0), max(y1, 0), max(z1, 0)
            x2, y2, z2 = min(x2, W), min(y2, H), min(z2, D)

            vol[:, z1:z1+w, y1:y2, x1:x2] = c
            vol[:, z2-w:z2, y1:y2, x1:x2] = c

            vol[:, z1:z2, y1:y1+w, x1:x2] = c
            vol[:, z1:z2, y2-w:y2, x1:x2] = c

            vol[:, z1:z2, y1:y2, x1:x1+w] = c
            vol[:, z1:z2, y1:y2, x2-w:x2] = c
        
        return vol

    def _get_color(self, color: Union[str, tuple]) -> torch.Tensor:
        if isinstance(color, COLOR_MAP):
            c = color.value
        elif isinstance(color, str):
            c = COLOR_MAP[color.upper()].value
        elif isinstance(color, tuple):
            if len(color) == 1:
                c = (color[0],) * 3
        else:
            raise ValueError("edge_color must be str or tuple")
        return c

    def visualize_masks(self, masks: list[BitMasks], task: str):
        if task == "semantic_segmentation":
            return self._visualize_semantic_masks(masks)
        elif task == "instance_segmentation":
            return self._visualize_instance_masks(masks)
        else:
            raise ValueError(f"Unsupported task: {task}. Supported tasks: \
                            ['semantic_segmentation', 'instance_segmentation']")

    def _visualize_instance_masks(
        self,
        masks: List[torch.Tensor],
    ) -> torch.Tensor:
        return self._merge_instance_masks(masks)

    def _visualize_semantic_masks(
        self,
        masks: List[torch.Tensor],
    ) -> torch.Tensor:
        return self._merge_semantic_masks(masks)

    def _merge_instance_masks(self, masks: List[BitMasks]) -> torch.Tensor:
        if len(masks) == 0:
            warnings.warn("No instance masks passed - returning zeros.")
            return torch.zeros(0)
        merged = torch.zeros_like(masks[0].tensor, dtype=torch.int32)
        for idx, m in enumerate(masks):
            merged[m.tensor.bool()] = idx + 1
        return merged

    def _merge_semantic_masks(self, masks: List[BitMasks]) -> torch.Tensor:
        if len(masks) == 0:
            warnings.warn("No semantic masks passed - returning zeros.")
            return torch.zeros(0)
        merged = torch.zeros_like(masks[0].tensor, dtype=torch.int32)
        for cls_idx, m in enumerate(masks):
            merged[m.tensor.bool()] = cls_idx
        return merged

    def visualize_masked_tensor(
        self,
        tensor: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if tensor.shape != mask.shape:
            raise ValueError("`tensor` and `mask` must have identical shapes.")
        masked = tensor.clone()
        masked[mask.bool()] = 0
        return masked

    def visualize(self, task: str, tensor: torch.Tensor, metadata: dict, save_path: str):
        if task == "detection":
            boxes_viz_tensor = self.visualize_boxes(metadata["image_size"], 
                                                    tensor, 
                                                    **metadata["visualization_params"])
            boxes_viz_tensor = boxes_viz_tensor.detach().cpu().numpy()
            save_file(save_path, boxes_viz_tensor, self.save_metadata)
        elif task == "instance_segmentation" or task == "semantic_segmentation":
            masks_viz_tensor = self.visualize_masks(masks=tensor, task=task)
            masks_viz_tensor = masks_viz_tensor.detach().cpu().numpy()
            save_file(save_path, masks_viz_tensor, self.save_metadata)
        elif task == "masked_prediction":
            masked_im_viz_tensor = self.visualize_masked_tensor(tensor, metadata["mask"])
            masked_im_viz_tensor = masked_im_viz_tensor.detach().cpu().numpy()
            save_file(save_path, masked_im_viz_tensor, self.save_metadata)
        else:
            raise ValueError(f"Unsupported task: {task}. \
                            Supported tasks: ['detection', \
                            'instance_segmentation', \
                            'semantic_segmentation', \
                            'masked_prediction']")