import torch
import torch.nn.functional as F

from segmentation.structures.data_objects.masks import BitMasks
from segmentation.structures.data_objects.image_list import ImageList


class Normalize:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, data_sample):
        image = data_sample.data_tensor.tensor
        mean = image.mean(dim=(1, 2, 3), keepdim=True) if self.mean is None else self.mean
        std = image.std(dim=(1, 2, 3), keepdim=True) if self.std is None else self.std
        image = (image - mean) / std
        data_sample.data_tensor.tensor = image
        return data_sample


class Resize:
    def __init__(self, size, resize_mode='trilinear'):
        self.size = tuple(size)
        self.resize_mode = resize_mode

    def __call__(self, data_sample):
        orig_d, orig_h, orig_w = data_sample.data_tensor.image_sizes[0]
        new_d, new_h, new_w = self.size
        scale_d, scale_h, scale_w = (
            new_d / orig_d,
            new_h / orig_h,
            new_w / orig_w,
        )

        # Resize image
        data_sample.data_tensor.resize(new_size=self.size)

        # Resize masks
        if 'masks' in data_sample.gt_instances:
            data_sample.gt_instances.masks.tensor = F.interpolate(
                data_sample.gt_instances.masks.tensor.unsqueeze(1).float(), 
                size=self.size[-3:],
                mode='nearest'
            ).squeeze(1)

        # Scale boxes
        if 'boxes' in data_sample.gt_instances:
            # TODO: double check all data object formats for consistency 
            boxes = data_sample.gt_instances.boxes.tensor
            scale = torch.tensor([scale_w, scale_h, scale_d, scale_w, scale_h, scale_d], device=boxes.device)
            data_sample.gt_instances.boxes.tensor = boxes * scale
        return data_sample            