"""
https://github.com/pytorch/vision/blob/ef4718ad85dab0a3694b0c3f740f46ab891f50cc/torchvision/ops/boxes.py
https://github.com/pytorch/vision/blob/ef4718ad85dab0a3694b0c3f740f46ab891f50cc/torchvision/ops/_box_convert.py#L2
https://github.com/pytorch/vision/blob/ef4718ad85dab0a3694b0c3f740f46ab891f50cc/torchvision/ops/_utils.py#L73

BSD 3-Clause License

Copyright (c) Soumith Chintala 2016, 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from typing import Tuple

import torch
import torchvision
from torch import Tensor

from ops3d._C import nms_3d


def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


def _box_cxcyczwhd_to_xyzxyz(boxes: Tensor) -> Tensor:
    """
    Converts bounding boxes from (cx, cy, cz, w, h, d) format to (x1, y1, z1, x2, y2, z2) format.
    (cx, cy, cz) refers to center of bounding box
    (w, h, d) are width and height of bounding box
    Args:
        boxes (Tensor[N, 6]): boxes in (cx, cy, cz, w, h, d) format which will be converted.

    Returns:
        boxes (Tensor(N, 6)): boxes in (x1, y1, z1, x2, y2, z2) format.
    """
    # We need to change all 6 of them so some temporary variable is needed.
    cx, cy, cz, w, h, d = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    z1 = cz - 0.5 * d
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    z2 = cz + 0.5 * d

    boxes = torch.stack((x1, y1, z1, x2, y2, z2), dim=-1)

    return boxes


def _box_xyzxyz_to_cxcyczwhd(boxes: Tensor) -> Tensor:
    """
    Converts bounding boxes from (x1, y1, z1, x2, y2, z2) format to (cx, cy, cz, w, h, d) format.
    (x1, y1, z1) refer to top left of bounding box
    (x2, y2, z2) refer to bottom right of bounding box
    Args:
        boxes (Tensor[N, 6]): boxes in (x1, y1, z1, x2, y2, z2) format which will be converted.

    Returns:
        boxes (Tensor(N, 6)): boxes in (cx, cy, cz w, h, d) format.
    """
    x1, y1, z1, x2, y2, z2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    cz = (z1 + z2) / 2
    w = x2 - x1
    h = y2 - y1
    d = z2 - z1

    boxes = torch.stack((cx, cy, cz, w, h, d), dim=-1)

    return boxes


def _box_xyzwhd_to_xyzxyz(boxes: Tensor) -> Tensor:
    """
    Converts bounding boxes from (x, y, z, w, h, d) format to (x1, y1, z1, x2, y2, z2) format.
    (x, y, z) refers to top left of bounding box.
    (w, h, z) refers to width and height of box.
    Args:
        boxes (Tensor[N, 6]): boxes in (x, y, z, w, h, d) which will be converted.

    Returns:
        boxes (Tensor[N, 6]): boxes in (x1, y1, z1, x2, y2, z2) format.
    """
    x, y, z, w, h, d = boxes.unbind(-1)
    boxes = torch.stack([x, y, z, x + w, y + h, z +d], dim=-1)
    return boxes


def _box_xyzxyz_to_xyzwhd(boxes: Tensor) -> Tensor:
    """
    Converts bounding boxes from (x1, y1, z1, x2, y2, z2) format to (x, y, z, w, h, d) format.
    (x1, y1, z1) refer to top left of bounding box
    (x2, y2, z2) refer to bottom right of bounding box
    Args:
        boxes (Tensor[N, 6]): boxes in (x1, y1, z1, x2, y2, z2) which will be converted.

    Returns:
        boxes (Tensor[N, 6]): boxes in (x, y, z, w, h, d) format.
    """
    x1, y1, z1, x2, y2, z2 = boxes.unbind(-1)
    w = x2 - x1  # x2 - x1
    h = y2 - y1  # y2 - y1
    d = z2 - z1  # z2 - z1
    boxes = torch.stack((x1, y1, z1, w, h, d), dim=-1)
    return boxes


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """
    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have an
    IoU greater than ``iou_threshold`` with another (higher scoring)
    box.

    If multiple boxes have the exact same score and satisfy the IoU
    criterion with respect to a reference box, the selected box is
    not guaranteed to be the same between CPU and GPU. This is similar
    to the behavior of argsort in PyTorch when repeated values are present.

    Args:
        boxes (Tensor[N, 6])): boxes to perform NMS on. They
            are expected to be in ``(x1, y1, z1, x2, y2, z2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2`` and ``0 <= z1 < z2.
        scores (Tensor[N]): scores for each one of the boxes
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """    
    device = boxes.device
    boxes = boxes.to("cuda")
    scores = scores.to("cuda")

    # need to recast boxes to float32
    # for IOU calculation in 3D NMS
    boxes = boxes.to(torch.float32) 
    keep = nms_3d(boxes, scores, iou_threshold)
    return keep.to(device)


def batched_nms(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 4]): boxes where NMS will be performed. They
            are expected to be in ``(x1, y1, z1, x2, y2, z2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2`` and ``0 <= z1 < z2.
        scores (Tensor[N]): scores for each one of the boxes
        idxs (Tensor[N]): indices of the categories for each one of the boxes.
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    # Benchmarks that drove the following thresholds are at
    # https://github.com/pytorch/vision/issues/1311#issuecomment-781329339
    # and https://github.com/pytorch/vision/pull/8925
    if boxes.numel() > (4000 if boxes.device.type == "cpu" else 100_000) and not torchvision._is_tracing():
        return _batched_nms_vanilla(boxes, scores, idxs, iou_threshold)
    else:
        return _batched_nms_coordinate_trick(boxes, scores, idxs, iou_threshold)


@torch.jit._script_if_tracing
def _batched_nms_coordinate_trick(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    # strategy: in order to perform NMS independently per class,
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


@torch.jit._script_if_tracing
def _batched_nms_vanilla(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    # Based on Detectron2 implementation, just manually call nms() on each class independently
    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    for class_id in torch.unique(idxs):
        curr_indices = torch.where(idxs == class_id)[0]
        curr_keep_indices = nms(boxes[curr_indices], scores[curr_indices], iou_threshold)
        keep_mask[curr_indices[curr_keep_indices]] = True
    keep_indices = torch.where(keep_mask)[0]
    return keep_indices[scores[keep_indices].sort(descending=True)[1]]


def remove_small_boxes(boxes: Tensor, min_size: float) -> Tensor:
    """
    Remove every box from ``boxes`` which contains at least one side length
    that is smaller than ``min_size``.

    .. note::
        For sanitizing a :class:`~torchvision.tv_tensors.BoundingBoxes` object, consider using
        the transform :func:`~torchvision.transforms.v2.SanitizeBoundingBoxes` instead.

    Args:
        boxes (Tensor[N, 6]): boxes in ``(x1, y1, z1, x2, y2, z2)`` format
            with ``0 <= x1 < x2`` and ``0 <= y1 < y2`` and ``0 <= z1 < z2.
        min_size (float): minimum size

    Returns:
        Tensor[K]: indices of the boxes that have both sides
        larger than ``min_size``
    """
    ws, hs, ds = boxes[:, 3] - boxes[:, 0], boxes[:, 4] - boxes[:, 1], boxes[:, 5] - boxes[:, 2]
    keep = (ws >= min_size) & (hs >= min_size) & (ds >= min_size)
    keep = torch.where(keep)[0]
    return keep


def clip_boxes_to_image(boxes: Tensor, size: Tuple[int, int]) -> Tensor:
    """
    Clip boxes so that they lie inside an image of size ``size``.

    .. note::
        For clipping a :class:`~torchvision.tv_tensors.BoundingBoxes` object, consider using
        the transform :func:`~torchvision.transforms.v2.ClampBoundingBoxes` instead.

    Args:
        boxes (Tensor[N, 6]): boxes in ``(x1, y1, z1, x2, y2, z2)`` format
            with ``0 <= x1 < x2`` and ``0 <= y1 < y2`` and ``0 <= z1 < z2.
        size (Tuple[height, width]): size of the image

    Returns:
        Tensor[N, 6]: clipped boxes
    """
    dim = boxes.dim()
    boxes_x = boxes[..., 0::3]
    boxes_y = boxes[..., 1::3]
    boxes_z = boxes[..., 2::3]
    height, width, depth = size
    
    boxes_x = boxes_x.clamp(min=0, max=width)
    boxes_y = boxes_y.clamp(min=0, max=height)
    boxes_z = boxes_z.clamp(min=0, max=depth)

    clipped_boxes = torch.stack((boxes_x, boxes_y, boxes_z), dim=dim)
    return clipped_boxes.reshape(boxes.shape)


def box_convert(boxes: Tensor, in_fmt: str, out_fmt: str) -> Tensor:
    """
    Converts :class:`torch.Tensor` boxes from a given ``in_fmt`` to ``out_fmt``.

    .. note::
        For converting a :class:`torch.Tensor` or a :class:`~torchvision.tv_tensors.BoundingBoxes` object
        between different formats,
        consider using :func:`~torchvision.transforms.v2.functional.convert_bounding_box_format` instead.
        Or see the corresponding transform :func:`~torchvision.transforms.v2.ConvertBoundingBoxFormat`.

    Supported ``in_fmt`` and ``out_fmt`` strings are:

    ``'xyzxyz'``: boxes are represented via corners, x1, y1, z1 being top left and x2, y2, z2 being bottom right.
    This is the format that torchvision utilities expect.

    ``'xyzwhd'``: boxes are represented via corner, width and height, x1, y1, z1 being top left, w, h, d being width, height, and depth.

    ``'cxcywh'``: boxes are represented via centre, width and height, cx, cy being center of box, w, h
    being width and height.

    ``'xywhr'``: boxes are represented via corner, width and height, x1, y2 being top left, w, h being width and height.
    r is rotation angle w.r.t to the box center by :math:`|r|` degrees counter clock wise in the image plan

    ``'cxcywhr'``: boxes are represented via centre, width and height, cx, cy being center of box, w, h
    being width and height.
    r is rotation angle w.r.t to the box center by :math:`|r|` degrees counter clock wise in the image plan

    ``'xyxyxyxy'``: boxes are represented via corners, x1, y1 being top left, x2, y2 bottom right,
    x3, y3 bottom left, and x4, y4 top right.

    Args:
        boxes (Tensor[N, K]): boxes which will be converted. K is the number of coordinates (4 for unrotated bounding boxes, 5 or 8 for rotated bounding boxes)
        in_fmt (str): Input format of given boxes. Supported formats are ['xyxy', 'xywh', 'cxcywh', 'xywhr', 'cxcywhr', 'xyxyxyxy'].
        out_fmt (str): Output format of given boxes. Supported formats are ['xyxy', 'xywh', 'cxcywh', 'xywhr', 'cxcywhr', 'xyxyxyxy']

    Returns:
        Tensor[N, K]: Boxes into converted format.
    """
    allowed_fmts = (
        "xyxy",
        "xywh",
        "cxcywh",
        # "xywhr",
        # "cxcywhr",
        "xyxyxyxy",
    )
    if in_fmt not in allowed_fmts or out_fmt not in allowed_fmts:
        raise ValueError(f"Unsupported Bounding Box Conversions for given in_fmt {in_fmt} and out_fmt {out_fmt}")

    if in_fmt == out_fmt:
        return boxes.clone()
    e = (in_fmt, out_fmt)
    if e == ("xywh", "xyxy"):
        boxes = _box_xyzwhd_to_xyzxyz(boxes)
    elif e == ("cxcywh", "xyxy"):
        boxes = _box_cxcyczwhd_to_xyzxyz(boxes)
    elif e == ("xyxy", "xywh"):
        boxes = _box_xyzxyz_to_xyzwhd(boxes)
    elif e == ("xyxy", "cxcywh"):
        boxes = _box_xyzxyz_to_cxcyczwhd(boxes)
    elif e == ("xywh", "cxcywh"):
        boxes = _box_xyzwhd_to_xyzxyz(boxes)
        boxes = _box_xyzxyz_to_cxcyczwhd(boxes)
    elif e == ("cxcywh", "xywh"):
        boxes = _box_cxcyczwhd_to_xyzxyz(boxes)
        boxes = _box_xyzxyz_to_xyzwhd(boxes)
    else:
        raise NotImplementedError(f"Unsupported Bounding Box Conversions for given in_fmt {e[0]} and out_fmt {e[1]}")

    return boxes


def box_volume(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by their
    (x1, y1, z1, x2, y2, z2) coordinates.

    Args:
        boxes (Tensor[N, 6]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, z1, x2, y2, z2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2`` and ``0 <= z1 < z2``.

    Returns:
        Tensor[N]: the area for each box
    """
    boxes = _upcast(boxes)
    return (boxes[:, 3] - boxes[:, 0]) * (boxes[:, 4] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 2])


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def _box_inter_union(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    vol1 = box_volume(boxes1)
    vol2 = box_volume(boxes2)

    lt = torch.max(boxes1[:, None, :3], boxes2[:, :3])  # [N,M,3]
    rb = torch.min(boxes1[:, None, 3:], boxes2[:, 3:])  # [N,M,3]

    dims = _upcast(rb - lt).clamp(min=0)  # [N,M,3]
    inter = dims[:, :, 0] * dims[:, :, 1] * dims[:, :, 2]  # [N,M]

    union = vol1[:, None] + vol2 - inter

    return inter, union


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) between two sets of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, z1, x2, y2, z2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2`` and ``0 <= z1 < z2``.

    Args:
        boxes1 (Tensor[N, 6]): first set of boxes
        boxes2 (Tensor[M, 6]): second set of boxes

    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union
    return iou


# Implementation adapted from https://github.com/facebookresearch/detr/blob/master/util/box_ops.py
def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Return generalized intersection-over-union (Jaccard index) between two sets of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, z1, x2, y2, z2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2`` and ``0 <= z1 < z2``.

    Args:
        boxes1 (Tensor[N, 6]): first set of boxes
        boxes2 (Tensor[M, 6]): second set of boxes

    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise generalized IoU values
        for every element in boxes1 and boxes2
    """
    # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #     _log_api_usage_once(generalized_box_iou)

    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union

    lti = torch.min(boxes1[:, None, :3], boxes2[:, :3])
    rbi = torch.max(boxes1[:, None, 3:], boxes2[:, 3:])

    whi = _upcast(rbi - lti).clamp(min=0)  # [N,M,3]
    vol_i = whi[:, :, 0] * whi[:, :, 1] * whi[:, :, 2]

    return iou - (vol_i - union) / vol_i


def complete_box_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Return complete intersection-over-union (Jaccard index) between two sets of boxes.
    Both sets of boxes are expected to be in ``(x1, y1, z1, x2, y2, z2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2`` and ``0 <= z1 < z2``.
    Args:
        boxes1 (Tensor[N, 6]): first set of boxes
        boxes2 (Tensor[M, 6]): second set of boxes
        eps (float, optional): small number to prevent division by zero. Default: 1e-7
    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise complete IoU values
        for every element in boxes1 and boxes2
    """
    boxes1 = _upcast(boxes1)
    boxes2 = _upcast(boxes2)

    diou, iou = _box_diou_iou(boxes1, boxes2, eps)

    w_pred = boxes1[:, None, 3] - boxes1[:, None, 0]
    h_pred = boxes1[:, None, 4] - boxes1[:, None, 1]
    d_pred = boxes1[:, None, 5] - boxes1[:, None, 2]

    w_gt = boxes2[:, 3] - boxes2[:, 0]
    h_gt = boxes2[:, 4] - boxes2[:, 1]
    d_gt = boxes2[:, 5] - boxes2[:, 2]

    # v = (4 / (torch.pi**2)) * torch.pow(torch.atan(w_pred / h_pred) - torch.atan(w_gt / h_gt), 2) # 2D Case

    # Compute arctan of the aspect ratios.
    ar_wh_pred = torch.atan(w_pred / h_pred)
    ar_wh_gt   = torch.atan(w_gt / h_gt)
    
    ar_wd_pred = torch.atan(w_pred / d_pred)
    ar_wd_gt   = torch.atan(w_gt / d_gt)
    
    ar_hd_pred = torch.atan(h_pred / d_pred)
    ar_hd_gt   = torch.atan(h_gt / d_gt)

    # Combine the squared differences
    v = (4 / (torch.pi ** 2)) * (
            torch.pow(ar_wh_pred - ar_wh_gt, 2) +
            torch.pow(ar_wd_pred - ar_wd_gt, 2) +
            torch.pow(ar_hd_pred - ar_hd_gt, 2)
        ) / 3

    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return diou - alpha * v


def distance_box_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Return distance intersection-over-union (Jaccard index) between two sets of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2`` and ``0 <= z1 < z2``.

    Args:
        boxes1 (Tensor[N, 6]): first set of boxes
        boxes2 (Tensor[M, 6]): second set of boxes
        eps (float, optional): small number to prevent division by zero. Default: 1e-7

    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise distance IoU values
        for every element in boxes1 and boxes2
    """
    boxes1 = _upcast(boxes1)
    boxes2 = _upcast(boxes2)
    diou, _ = _box_diou_iou(boxes1, boxes2, eps=eps)
    return diou


def _box_diou_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tuple[Tensor, Tensor]:

    iou = box_iou(boxes1, boxes2)
    lti = torch.min(boxes1[:, None, :3], boxes2[:, :3])
    rbi = torch.max(boxes1[:, None, 3:], boxes2[:, 3:])
    whdi = _upcast(rbi - lti).clamp(min=0)  # [N,M,3]
    diagonal_distance_squared = (whdi[:, :, 0] ** 2) + (whdi[:, :, 1] ** 2) + (whdi[:, :, 2] ** 2) + eps
    
    # centers of boxes
    x_p = (boxes1[:, 0] + boxes1[:, 3]) / 2
    y_p = (boxes1[:, 1] + boxes1[:, 4]) / 2
    z_p = (boxes1[:, 2] + boxes1[:, 5]) / 2

    x_g = (boxes2[:, 0] + boxes2[:, 3]) / 2
    y_g = (boxes2[:, 1] + boxes2[:, 4]) / 2
    z_g = (boxes2[:, 2] + boxes2[:, 5]) / 2

    # The distance between boxes' centers squared.
    centers_distance_squared = (_upcast((x_p[:, None] - x_g[None, :])) ** 2) + \
                                (_upcast((y_p[:, None] - y_g[None, :])) ** 2) + \
                                (_upcast((z_p[:, None] - z_g[None, :])) **2)
    
    # The distance IoU is the IoU penalized by a normalized
    # distance between boxes' centers squared.
    return iou - (centers_distance_squared / diagonal_distance_squared), iou


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Compute the bounding boxes around the provided masks.

    Returns a [N, 6] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 <= x2`` and ``0 <= y1 <= y2`` and ``0 <= z1 < z2``.

    .. warning::

        In most cases the output will guarantee ``x1 < x2`` and ``y1 < y2`` and ``0 <= z1 < z2``. But
        if the input is degenerate, e.g. if a mask is a single row or a single
        column, then the output may have x1 = x2 or y1 = y2 or z1=z2.

    Args:
        masks (Tensor[N, D, H, W]): masks to transform where N is the number of masks
            and (D, H, W) are the spatial dimensions.

    Returns:
        Tensor[N, 6]: bounding boxes
    """
    if masks.numel() == 0:
        return torch.zeros((0, 6), device=masks.device, dtype=torch.float)

    n = masks.shape[0]

    bounding_boxes = torch.zeros((n, 6), device=masks.device, dtype=torch.float)

    for index, mask in enumerate(masks):
        z, y, x = torch.where(mask != 0)

        bounding_boxes[index, 0] = torch.min(x)
        bounding_boxes[index, 1] = torch.min(y)
        bounding_boxes[index, 2] = torch.min(z)
        bounding_boxes[index, 3] = torch.max(x)
        bounding_boxes[index, 4] = torch.max(y)
        bounding_boxes[index, 5] = torch.max(z)

    return bounding_boxes