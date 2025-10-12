"""
https://github.com/facebookresearch/detectron2/blob/536dc9d527074e3b15df5f6677ffe1f4e104a4ab/projects/PointRend/point_rend/point_features.py#L63
https://github.com/facebookresearch/detectron2/blob/536dc9d527074e3b15df5f6677ffe1f4e104a4ab/detectron2/layers/wrappers.py#L65
https://github.com/IDEA-Research/MaskDINO/blob/3831d8514a3728535ace8d4ecc7d28044c42dd14/maskdino/utils/misc.py#L49

Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F


def pack_time(x: torch.Tensor, input_format: str, output_format: str = "TCZYX"):
    if input_format == "TZYXC":
        B, T, Z, Y, X, C = x.shape
        if output_format == "TCZYX":
            x = x.permute(0, 1, 5, 2, 3, 4).reshape(B*T, C, Z, Y, X)
            return x, B, T
        else:
            raise ValueError(f"Unsupported output_format {output_format}")
    elif input_format == "TCZYX":
        # x: [B, T, C, Z, Y, X] -> [B*T, C, Z, Y, X]
        B, T, C, Z, Y, X = x.shape
        if output_format == "TCZYX":
            x = x.reshape(B*T, C, Z, Y, X)
            return x, B, T
        else:
            raise ValueError(f"Unsupported output_format {output_format}")
    else:
        raise ValueError(f"Unsupported input_format {input_format}")


def unpack_time(x: torch.Tensor, B: int, T: int, input_format: str, output_format: str):
    if input_format == "TZYXC":
        _, Z, Y, X, C = x.shape
        if output_format == "TZYXC":
            return x.reshape(B, T, Z, Y, X, C)
        elif output_format == "CT":
            return x.reshape((B, T, Z, Y, X, C)).permute(0, 2, 3, 4, 5, 1).reshape(B*Z*Y*X, C, T)
        elif output_format == "TCZYX":
            return x.reshape((B, T, Z, Y, X, C)).permute(0, 1, 5, 2, 3, 4)
        elif output_format == "ZYXCT":
            return x.reshape((B, T, Z, Y, X, C)).permute(0, 2, 3, 4, 5, 1)
        else:
            raise ValueError(f"Unsupported output_format {output_format}")
    elif input_format == "TCZYX":
        _, C, Z, Y, X = x.shape
        if output_format == "TZYXC":
            return x.reshape(B, T, C, Z, Y, X).permute(0, 1, 3, 4, 5, 2)
        elif output_format == "CT":
            return x.reshape((B, T, C, Z, Y, X)).permute(0, 3, 4, 5, 2, 1).reshape(B*Z*Y*X, C, T)
        elif output_format == "TCZYX":
            return x.reshape((B, T, C, Z, Y, X))
        elif output_format == "ZYXCT":
            return x.reshape((B, T, C, Z, Y, X)).permute(0, 3, 4, 5, 2, 1)
        else:
            raise ValueError(f"Unsupported output_format {output_format}")
    else:
        raise ValueError(f"Unsupported input_format {input_format}")


def pack_spatial(x: torch.Tensor, input_format: str, output_format: str = "CT"):
    if input_format == "TZYXC":
        B, T, Z, Y, X, C = x.shape
        if output_format == "CT":
            return x.permute(0, 2, 3, 4, 5, 1).reshape(B*Z*Y*X, C, T)
        else:
            raise ValueError(f"Unsupported output_format {output_format}")
    elif input_format == "TCZYX":
        B, T, C, Z, Y, X = x.shape
        if output_format == "CT":
            return x.permute(0, 3, 4, 5, 2, 1).reshape(B*Z*Y*X, C, T)
        else:
            raise ValueError(f"Unsupported output_format {output_format}")
    else:
        raise ValueError(f"Unsupported input_format {input_format}")


def unpack_spatial(x: torch.Tensor, B: int, input_format: str, input_shape: List[int], output_format: str):
    _, C, T = x.shape
    if input_format == "TZYXC":
        T, Z, Y, X, C = input_shape
    elif input_format == "TCZYX":
        T, C, Z, Y, X = input_shape
    else:
        raise ValueError(f"Unsupported input_format {input_format}")
    
    if output_format == "TZYXC":
        if input_format == "TZYXC" or input_format == "TCZYX":
            return x.reshape(B, Z, Y, X, C, T).permute(0, 5, 1, 2, 3, 4)
        # in case we want to support 2D+T in future
        else:
            raise ValueError(f"Unsupported input_format {input_format}")
    elif output_format == "TCZYX":
        if input_format == "TZYXC" or input_format == "TCZYX":
            return x.reshape(B, Z, Y, X, C, T).permute(0, 5, 4, 1, 2, 3)
    elif output_format == "CTZYX":
        if input_format == "TZYXC" or input_format == "TCZYX":
            return x.reshape(B, Z, Y, X, C, T).permute(0, 4, 5, 1, 2, 3)
    else:
        raise ValueError(f"Unsupported output_format {output_format}")


def get_reference_points(shapes, valid_ratios, device):
    reference_points_list = []
    for lvl, (D_, H_, W_) in enumerate(shapes):
        # create grid [0.5, 1.5, ..., size_dim - 0.5]
        ref_z, ref_y, ref_x = torch.meshgrid(
                                            torch.linspace(0.5, D_ - 0.5, D_, dtype=torch.float32, device=device),
                                            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                                            indexing='ij'
                                            ) 
        
        # scaling by valid_ratios adjusts the normalized reference grid so that it
        # only spans the unpadded region, i.e. [1, D*H*W] / (valid_ratio_d * D), 
        # i.e. scale grid to [0, 1] adjusted by valid ratio
        ref_z = ref_z.reshape(-1)[None] / (valid_ratios[:, None, lvl, 2] * D_) 
        ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
        ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
        
        ref = torch.stack((ref_x, ref_y, ref_z), -1) # [B, D*H*W, 3]
        reference_points_list.append(ref)
    
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None] * valid_ratios[:, None]
    return reference_points


def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, D, H, W) that contains features map on a D x H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 3) or (N, Dgrid, Wgrid, Hgrid, 3) that contains
        [0, 1] x [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Dgrid, Wgrid, Hgrid) that contains
            features for points in `point_coords`. The features are obtained via trilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        # (N, P, 3) -> (N, P, 1, 1, 3)
        point_coords = point_coords.unsqueeze(2).unsqueeze(3)
    # sample points in [-1, 1] x [-1, 1] x [-1, 1] coordinate space
    # returns: 
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        # (N, C, P, 1, 1) -> (N, C, P)
        output = output.squeeze(4).squeeze(3)
    return output


def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points in [0, 1] x [0, 1] x [0,1] coordinate space based on their uncertainty. The unceratinties
    are calculated for each point using 'uncertainty_func' function that takes point's logit
    prediction as input.

    See PointRend paper for details.

    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Dmask, Hmask, Wmask) or (N, 1, Dmask, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importance sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 3) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1, "oversample_ratio must be >= 1"
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0, "importance_sample_ratio must be in [0, 1]"
    
    num_boxes = coarse_logits.shape[0]
    # NOTE: we oversample points first 
    num_sampled = int(num_points * oversample_ratio)
    
    point_coords = torch.rand(num_boxes, num_sampled, 3, device=coarse_logits.device)
    # NOTE: align_corners passed to grid_sample
    # returns: (N, C, D, H, W)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    
    # It is crucial to calculate uncertainty based on the sampled prediction value for points.
    # Calculating uncertainties of coarse predictions first and sampling them for points leads
    # to incorrect results. To illustrate this:
    # assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value
    # however, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty
    point_uncertainties = uncertainty_func(point_logits) # returns: (N, 1, num_sampled) per point score
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    
    # returns: topk indices of the most uncertain points
    # idx: (N, num_uncertain_points)
    uncertain_points_indices = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    # shift: [0, num_sampled, 2*num_sampled, ..., (num_boxes-1)*num_sampled]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    # uncertain_points_indices: (num_boxes, num_uncertain_points) -> broadcast-add (num_boxes, num_uncertain_points) 
    # i.e. for every box, we add a constant offset given by num_sampled * box_index
    # this allows for global indexing into point_coords
    uncertain_points_indices += shift[:, None]
    
    # point_coords: (num_boxes, num_sampled, 3) -> (num_boxes * num_sampled, 3) 
    # -> (num_boxes*num_uncertain_points, 3) -> (num_boxes, num_uncertain_points, 3) 
    point_coords = point_coords.view(-1, 3)[uncertain_points_indices.view(-1), :].view(
        num_boxes, num_uncertain_points, 3
    )
    
    if num_random_points > 0:
        point_coords = torch.cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 3, device=coarse_logits.device),
            ],
            dim=1,
        )
    
    return point_coords


def _max_by_axis(img_list):
    maxes = img_list[0]
    for sublist in img_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def batch_tensors(tensor_list: List[torch.Tensor]) -> torch.Tensor:
    max_size = _max_by_axis([list(img.shape) for img in tensor_list])
    batch_shape = [len(tensor_list)] + max_size
    b, n, d, h, w = batch_shape
    dtype, device = tensor_list[0].dtype, tensor_list[0].device
    tensors = torch.zeros(batch_shape, dtype=dtype, device=device)
    masks = torch.ones((b, d, h, w), dtype=torch.bool, device=device)
    for img, img_pad, mask in zip(tensor_list, tensors, masks):
        img_pad[: img.shape[0], : img.shape[1], : img.shape[2], : img.shape[3]].copy_(img)
        mask[: img.shape[1], : img.shape[2], : img.shape[3]] = False
    return tensors, masks


def compute_unmasked_ratio(mask):
        _, D, H, W = mask.shape

        valid_D = (~mask).any(dim=(2,3)).sum(dim=1) # [B] — any valid pixel in each D-slice
        valid_H = (~mask).any(dim=(1,3)).sum(dim=1) # [B] — any valid pixel in each H-slice
        valid_W = (~mask).any(dim=(1,2)).sum(dim=1) # [B] — any valid pixel in each W-slice

        valid_ratio_d = valid_D.float() / D
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        
        valid_ratio = torch.stack([valid_ratio_d, valid_ratio_w, valid_ratio_h], -1) # [B, 3]
        return valid_ratio


def c2_xavier_fill(module: nn.Module) -> None:
        # Caffe2 implementation of XavierFill in fact
        # corresponds to kaiming_uniform_ in PyTorch
        nn.init.kaiming_uniform_(module.weight, a=1)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)