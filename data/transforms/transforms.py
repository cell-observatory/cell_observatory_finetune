import torch
import torch.nn.functional as F


class Normalize:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, data_sample):
        image = data_sample.data_tensor.tensor
        
        if self.mean is None and self.std is None:
            mean, std = data_sample.data_tensor.get_image_stats() 
            image = (image - mean) / std
        else:
            mean = torch.tensor(self.mean, dtype=image.dtype, device=image.device)
            std = torch.tensor(self.std, dtype=image.dtype, device=image.device)
            image = (image - mean) / std
        
        data_sample.data_tensor.tensor = image
        return data_sample


class Resize:
    def __init__(self, size, resize_mode='trilinear'):
        self.size = tuple(size)
        self.resize_mode = resize_mode

    def __call__(self, data_sample):
        orig_d, orig_h, orig_w = data_sample.data_tensor.shape
        new_d, new_h, new_w = self.size
        scale_d, scale_h, scale_w = (
            new_d / orig_d,
            new_h / orig_h,
            new_w / orig_w,
        )

        # resize image
        data_sample.data_tensor.resize(new_size=self.size)

        # TODO: still need to allow for more flexibility 
        #       in boxes/labels/masks layout
        if "masks" in data_sample.gt_instances:
            # (N, D, H, W) or (N, T, D, H, W)
            m = data_sample.gt_instances.masks.tensor.float()

            if m.ndim == 4:
                # (N,1,D,H,W) -> # (N, D',H',W')
                m = F.interpolate(m.unsqueeze(1),
                                   size=self.size,
                                   mode="nearest").squeeze(1)         
            elif m.ndim == 5:
                N, T, D, H, W = m.shape
                # merge (N,T) into batch
                m = m.view(N * T, 1, D, H, W)
                m = F.interpolate(m, size=self.size, mode="nearest")
                # split (N,T) back into batch, time
                m = m.view(N, T, *self.size)
            else:
                raise ValueError("Mask tensor must be 4- or 5-D")

            data_sample.gt_instances.masks.tensor = m

        if "boxes" in data_sample.gt_instances:
            # boxes: (N, 6) or (N, T, 6)  -> (x1, y1, z1, x2, y2, z2)
            boxes = data_sample.gt_instances.boxes.tensor
            # broadcast to last dim
            scale_vec = boxes.new_tensor(
                [scale_w, scale_h, scale_d, scale_w, scale_h, scale_d]
            ) 

            data_sample.gt_instances.boxes.tensor = boxes * scale_vec
            
        return data_sample