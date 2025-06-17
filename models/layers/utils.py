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


from typing import List, Tuple

import torch
from torch.nn import functional as F


# ------------------------------------------------------------ WINDOW PARTITIONING ------------------------------------------------------------


def undo_windowing(
    x: torch.Tensor, shape: List[int], mu_shape: List[int]
) -> torch.Tensor:
    """
    Restore spatial organization by undoing windowed organization of mask units.

    Args:
        x: organized by mask units windows, e.g. in 2d [B, #MUy*#MUx, MUy, MUx, C]
        shape: current spatial shape, if it were not organized into mask unit
            windows, e.g. in 2d [B, #MUy*MUy, #MUx*MUx, C].
        mu_shape: current mask unit shape, e.g. in 2d [MUy, MUx]
    Returns:
        x: e.g. in 2d, [B, #MUy*MUy, #MUx*MUx, C]
    """
    D = len(shape)
    B, C = x.shape[0], x.shape[-1]
    # [B, #MUy*#MUx, MUy, MUx, C] -> [B, #MUy, #MUx, MUy, MUx, C]
    num_MUs = [s // mu for s, mu in zip(shape, mu_shape)]
    x = x.view(B, *num_MUs, *mu_shape, C)

    # [B, #MUy, #MUx, MUy, MUx, C] -> [B, #MUy*MUy, #MUx*MUx, C]
    permute = (
        [0]
        + sum(
            [list(p) for p in zip(range(1, 1 + D), range(1 + D, 1 + 2 * D))],
            [],
        )
        + [len(x.shape) - 1]
    )
    x = x.permute(permute).reshape(B, *shape, C)

    return x


def window_partition(x: torch.Tensor, window_size: int):
    """
    Partition input tensor into non-overlapping windows, with padding if needed.

    Args:
        x (tensor): input tokens with [B, D, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, window_size, C].
        (Dp, Hp, Wp): padded depth, height, and width before partition
    """
    # TODO: support nD
    B, D, H, W, C = x.shape

    pad_d = (window_size - D % window_size) % window_size
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_d or pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_d, 0, pad_w, 0, pad_h))
    Dp, Hp, Wp = D + pad_d, H + pad_h, W + pad_w

    x = x.view(B, Dp // window_size, window_size, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    return windows, (Dp, Hp, Wp)


def window_unpartition(windows: torch.Tensor, window_size: int, pad_dhw: Tuple[int, int, int], dhw: Tuple[int, int, int]):
    """
    Window unpartition into original sequences and removing padding.

    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, window_size, C].
        window_size (int): window size.
        pad_dhw (Tuple): padded depth, height and width (Dp, Hp, Wp).
        dhw (Tuple): original depth, height and width (D, H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, D, H, W, C].
    """
    # TODO: support nD
    Dp, Hp, Wp = pad_dhw
    D, H, W = dhw
    
    B = windows.shape[0] // (Dp * Hp * Wp // window_size // window_size // window_size)
    x = windows.view(B, Dp // window_size, Hp // window_size, Wp // window_size, window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, Dp, Hp, Wp, -1)

    if Dp > D or Hp > H or Wp > W:
        x = x[:, :D, :H, :W, :].contiguous()
    return x


# ------------------------------------------------------------ GRID SAMPLE ------------------------------------------------------------


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
        point_coords = cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 3, device=coarse_logits.device),
            ],
            dim=1,
        )
    
    return point_coords


# ------------------------------------------------------------ HELPERS ------------------------------------------------------------


def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )

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


# ------------------------------------------------------------  ------------------------------------------------------------