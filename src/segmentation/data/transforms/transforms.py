import torch
import torch.nn.functional as F


class Normalize:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, image, targets):
        mean = image.mean(dim=(1, 2, 3), keepdim=True) if self.mean is None else self.mean
        std = image.std(dim=(1, 2, 3), keepdim=True) if self.std is None else self.std
        image = (image - mean) / std
        return image, targets


class Resize:
    def __init__(self, size, resize_mode='trilinear'):
        self.size = tuple(size)
        self.resize_mode = resize_mode

    def __call__(self, image, targets):
        orig_d, orig_h, orig_w = image.shape[-3:]
        new_d, new_h, new_w = self.size
        scale_d, scale_h, scale_w = (
            new_d / orig_d,
            new_h / orig_h,
            new_w / orig_w,
        )

        # Resize image
        image = F.interpolate(image.unsqueeze(0), size=self.size, mode=self.resize_mode, align_corners=False).squeeze(0)

        # Resize masks
        if 'masks' in targets:
            targets['masks'] = F.interpolate(
                targets['masks'].unsqueeze(1).float(),  # (N, 1, D, H, W)
                size=self.size,
                mode='nearest'
            ).squeeze(1)

        # Scale boxes
        if 'boxes' in targets:
            boxes = targets['boxes']
            # boxes are in (x0, y0, z0, x1, y1, z1) format
            scale = torch.tensor([scale_w, scale_h, scale_d, scale_w, scale_h, scale_d], device=boxes.device)
            targets['boxes'] = boxes * scale

        return image, targets


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, targets):
        for t in self.transforms:
            img, targets = t(img, targets)
        return img, targets