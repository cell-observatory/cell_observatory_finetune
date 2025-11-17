"""
https://github.com/facebookresearch/detectron2/blob/main/detectron2/structures/boxes.py

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


import math
from enum import IntEnum, unique
from typing import List, Tuple, Union

import torch
import torchvision
from torch import Tensor
from torch import device

try:
    from ops3d._C import nms_3d
except ImportError:
    print("3D NMS op failed to load. Please compile ops3d if needed.")

from cell_observatory_platform.data.io import record_init


# TODO: extend with time axis


# ---------------------------------------- HELPER ---------------------------------------------


def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications
    # by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()
    

# ---------------------------------------- BOXES ENUM ---------------------------------------------


# conversion helpers 


def box_cxcyczwhd_to_xyzxyz(boxes: Tensor) -> Tensor:
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


def box_xyzxyz_to_cxcyczwhd(boxes: Tensor) -> Tensor:
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


def box_xyzwhd_to_xyzxyz(boxes: Tensor) -> Tensor:
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


def box_xyzxyz_to_xyzwhd(boxes: Tensor) -> Tensor:
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


@unique
class BoxMode3D(IntEnum):
    """
    Enum of different ways to represent a box.
    """

    """
    Enumeration of 3-D bounding-box layouts (all coordinates expressed in absolute units).
    """

    XYZXYZ_ABS = 0

    XYZWHD_ABS = 1
    
    CXCYCZWHD_ABS = 2

    @staticmethod
    def convert(boxes: Tensor,
                from_mode: "BoxMode3D",
                to_mode:   "BoxMode3D") -> Tensor:
        """
        Convert boxes from from_mode to to_mode.

        Args:
            boxes : Tensor[N, 6]
                Bounding-boxes whose last dimension matches *from_mode*.
            from_mode / to_mode : BoxMode3D
                Desired conversion end-points.

        Returns:
        Tensor[N, 6]
            A clone of boxes in the requested layout.
        """
        if from_mode == to_mode:
            return boxes.clone()
        
        if from_mode == BoxMode3D.XYZWHD_ABS:           # (x,y,z,w,h,d) -> (x1,y1,z1,x2,y2,z2)
            boxes = box_xyzwhd_to_xyzxyz(boxes)
        elif from_mode == BoxMode3D.CXCYCZWHD_ABS:      # (cx,cy,cz,w,h,d) -> (x1,y1,z1,x2,y2,z2)
            boxes = box_cxcyczwhd_to_xyzxyz(boxes)

        if to_mode == BoxMode3D.XYZXYZ_ABS:
            return boxes
        elif to_mode == BoxMode3D.XYZWHD_ABS:           # (x1,y1,z1,x2,y2,z2) -> (x,y,z,w,h,d)
            return box_xyzxyz_to_xyzwhd(boxes)
        elif to_mode == BoxMode3D.CXCYCZWHD_ABS:        # (x1,y1,z1,x2,y2,z2) -> (cx,cy,cz,w,h,d)
            return box_xyzxyz_to_cxcyczwhd(boxes)

        raise NotImplementedError(
            f"Conversion from {from_mode.name} to {to_mode.name} is not implemented."
        )


# ---------------------------------------- BOXES CLASS ---------------------------------------------


class Boxes:
    """
    This structure stores a list of boxes as a Nx6 torch.Tensor.
    It supports some common methods about boxes
    (`volume`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)

    Attributes:
        tensor (torch.Tensor): float matrix of Nx6. Each row is (x1, y1, z1, x2, y2, z2).
    """
    @record_init
    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx6 matrix.  Each row is (x1, y1, z1, x2, y2, z2).
        """        
        if tensor.numel() == 0:
            tensor = tensor.reshape((-1, 6))

        assert tensor.dim() == 2 and tensor.size(-1) == 6, tensor.size()

        self.tensor = tensor

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the underlying tensor."""
        return self.tensor.dtype

    def clone(self) -> "Boxes":
        """
        Clone the Boxes.

        Returns:
            Boxes
        """
        return Boxes(self.tensor.clone())

    def to(self, *args, **kwargs) -> "Boxes":
         return Boxes(self.tensor.to(*args, **kwargs))

    def volume(self) -> Tensor:
        """
        Computes the volume of a set of bounding boxes, which are specified by their
        (x1, y1, z1, x2, y2, z2) coordinates.

        Args:
            boxes (Tensor[N, 6]): boxes for which the volume will be computed. They
                are expected to be in (x1, y1, z1, x2, y2, z2) format with
                ``0 <= x1 < x2`` and ``0 <= y1 < y2`` and ``0 <= z1 < z2``.

        Returns:
            Tensor[N]: the volume for each box
        """
        boxes = self.tensor
        boxes = _upcast(boxes)
        return (boxes[:, 3] - boxes[:, 0]) * (boxes[:, 4] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 2])

    def clip(self, box_size: Tuple[int, int, int]) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height] and z coordinates to [0, depth].

        Args:
            box_size (height, width, depth): The clipping box's size.
        """
        assert torch.isfinite(self.tensor).all(), "Box tensor contains infinite or NaN!"
        d, h, w = box_size
        x1 = self.tensor[:, 0].clamp(min=0, max=w)
        y1 = self.tensor[:, 1].clamp(min=0, max=h)
        z1 = self.tensor[:, 2].clamp(min=0, max=d)
        x2 = self.tensor[:, 3].clamp(min=0, max=w)
        y2 = self.tensor[:, 4].clamp(min=0, max=h)
        z2 = self.tensor[:, 5].clamp(min=0, max=d)
        self.tensor = torch.stack((x1, y1, z1, x2, y2, z2), dim=-1)

    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if any of its sides are no larger than threshold.

        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        box = self.tensor
        widths = box[:, 3] - box[:, 0]
        heights = box[:, 4] - box[:, 1]
        depths = box[:, 5] - box[:, 2]
        keep = (widths > threshold) & (heights > threshold) & (depths > threshold)
        return keep

    def __getitem__(self, item) -> "Boxes":
        """
        Args:
            item: int, slice, or a BoolTensor

        Returns:
            Boxes: Create a new :class:`Boxes` by indexing.

        The following usage are allowed:

        1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned Boxes might share storage with this Boxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Boxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return Boxes(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "Boxes(" + str(self.tensor) + ")"

    def inside_box(self, box_size: Tuple[int, int, int, int, int, int], boundary_threshold: int = 0) -> torch.Tensor:
        """
        Args:
            box_size (depth, height, width): Size of the reference box.
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".

        Returns:
            a binary vector, indicating whether each box is inside the reference box.
        """
        x1,y1,z1,x2,y2,z3 = box_size
        inds_inside = (
            (self.tensor[..., 0] >= x1 - boundary_threshold)
            & (self.tensor[..., 1] >= y1 - boundary_threshold)
            & (self.tensor[..., 2] >= -z1 - boundary_threshold)
            & (self.tensor[..., 3] < x2 + boundary_threshold)
            & (self.tensor[..., 4] < y2 + boundary_threshold)
            & (self.tensor[..., 5] < z3 + boundary_threshold)
        )
        return inds_inside

    def get_centers(self) -> torch.Tensor:
        """
        Returns:
            The box centers in a Nx2 array of (x, y, z).
        """
        return (self.tensor[:, :3] + self.tensor[:, 3:]) / 2

    def scale(self, scale_x: float, scale_y: float, scale_z: float) -> None:
        """
        Scale the box with horizontal and vertical scaling factors
        """
        self.tensor[:, 0::3] *= scale_x
        self.tensor[:, 1::3] *= scale_y
        self.tensor[:, 2::3] *= scale_z

    @classmethod
    def cat(cls, boxes_list: List["Boxes"]) -> "Boxes":
        """
        Concatenates a list of Boxes into a single Boxes

        Arguments:
            boxes_list (list[Boxes])

        Returns:
            Boxes: the concatenated Boxes
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all([isinstance(box, Boxes) for box in boxes_list])

        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    @property
    def device(self) -> device:
        return self.tensor.device
    
    @property
    def dtype(self) -> torch.dtype:
        return self.tensor.dtype

    # type "Iterator[torch.Tensor]", yield, and iter() not supported by torchscript
    # https://github.com/pytorch/pytorch/issues/18627
    @torch.jit.unused
    def __iter__(self):
        """
        Yield a box as a Tensor of shape (6,) at a time.
        """
        yield from self.tensor


# -------------------------------------------------- NMS --------------------------------------------------


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
    # to protect against numerical overflows
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


# ---------------------------------------- IOU --------------------------------------------------


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def _box_inter_union(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    lt = torch.max(boxes1[:, None, :3], boxes2[:, :3])  # [N,M,3]
    rb = torch.min(boxes1[:, None, 3:], boxes2[:, 3:])  # [N,M,3]

    dims = _upcast(rb - lt).clamp(min=0)  # [N,M,3]
    inter = dims[:, :, 0] * dims[:, :, 1] * dims[:, :, 2]  # [N,M]
    return inter


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M, compute the IoU
    (intersection over union) between **all** N x M pairs of boxes.
    The box order must be (xmin, ymin, zmin, xmax, ymax, zmax).

    Args:
        boxes1, boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    volume1 = boxes1.volume()  # [N]
    volume2 = boxes2.volume()  # [M]
    inter = _box_inter_union(boxes1.tensor, boxes2.tensor)

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (volume1[:, None] + volume2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


def pairwise_ioa(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Similar to :func:`pariwise_iou` but compute the IoA (intersection over boxes2 volume).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoA, sized [N,M].
    """
    volume2 = boxes2.volume()  # [M]
    inter = _box_inter_union(boxes1.tensor, boxes2.tensor)

    # handle empty boxes
    ioa = torch.where(
        inter > 0, inter / volume2, torch.zeros(1, dtype=inter.dtype, device=inter.device)
    )
    return ioa


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


def pairwise_point_box_distance(points: torch.Tensor, boxes: Boxes):
    """
    Pairwise distance between N points and M boxes. The distance between a
    point and a box is represented by the distance from the point to 6 edges
    of the box. Distances are all positive when the point is inside the box.

    Args:
        points: Nx2 coordinates. Each row is (x, y)
        boxes: M boxes

    Returns:
        Tensor: distances of size (N, M, 6). The 6 values are distances from
            the point to the left, top, right, bottom of the box.
    """
    x, y, z = points.unsqueeze(dim=2).unbind(dim=1)  # (N, 1)
    x0, y0, z0, x1, y1, z1 = boxes.tensor.unsqueeze(dim=0).unbind(dim=2)  # (1, M)
    return torch.stack([x - x0, y - y0, z- z0, x1 - x, y1 - y, z1 - z], dim=2)


def matched_pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Compute pairwise intersection over union (IOU) of two sets of matched
    boxes that have the same number of boxes.
    
    Similar to :func:`pairwise_iou`, but computes only diagonal elements of the matrix.

    Args:
        boxes1 (Boxes): bounding boxes, sized [N, 6].
        boxes2 (Boxes): same length as boxes1
    Returns:
        Tensor: iou, sized [N].
    """
    assert len(boxes1) == len(
        boxes2
    ), "boxlists should have the same" "number of entries, got {}, {}".format(
        len(boxes1), len(boxes2)
    )
    volume1 = boxes1.volume()  # [N]
    volume2 = boxes2.volume()  # [N]
    box1, box2 = boxes1.tensor, boxes2.tensor
    
    lt = torch.max(box1[:, :3], box2[:, :3])  # [N,3]
    rb = torch.min(box1[:, 3:], box2[:, 3:])  # [N,3]
    
    wh = (rb - lt).clamp(min=0)  # [N,3]
    inter = wh.prod(dim=1)  # [N]
    iou = inter / (volume1 + volume2 - inter)  # [N]
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
    iou = pairwise_iou(boxes1, boxes2)
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

# ---------------------------------------- AUXILIARY FUNCTIONS --------------------------------------------------


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


def masks_to_boxes(masks: torch.Tensor) -> Tensor:
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


# TODO: reconcile masks_to_boxes and masks_to_boxes_v2
def masks_to_boxes_v2(masks, eps: float = 1e-1) -> Tensor:
    """
    Compute the bounding boxes around the provided masks.
    The masks should be in format [N, D, H, W] where N is 
    the number of masks, (D, H, W) are the spatial dimensions.
    Returns a [N, 6] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 6), device=masks.device)

    d, h, w = masks.shape[-3:]

    z = torch.arange(0, d, dtype=torch.float, device=masks.device)
    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)
    z, y, x = torch.meshgrid(z, y, x, indexing='ij')

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    z_mask = (masks * z.unsqueeze(0))
    z_max = z_mask.flatten(1).max(-1)[0]
    z_min = z_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    mask = torch.stack([x_min, y_min, z_min, x_max, y_max, z_max], 1).to(masks.device, torch.float)
    invalid_mask = (torch.isinf(x_min)) | (torch.isinf(y_min)) | (torch.isinf(z_min))
    mask[invalid_mask] = 0
    return mask

#---------------------------------------------------------------- BOX ENCODERS ----------------------------------------------------------------


@torch.jit._script_if_tracing
def encode_boxes(reference_boxes: Tensor, proposals: Tensor, weights: Tensor) -> Tensor:
    """
    Encode a set of proposals with respect to some
    reference boxes

    Args:
        reference_boxes (Tensor): reference boxes
        proposals (Tensor): boxes to be encoded
        weights (Tensor[4]): the weights for ``(x, y, z, w, h, d)``
    """

    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    wz = weights[2]
    ww = weights[3]
    wh = weights[4]
    wd = weights[5]

    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_z1 = proposals[:, 2].unsqueeze(1)

    proposals_x2 = proposals[:, 3].unsqueeze(1)
    proposals_y2 = proposals[:, 4].unsqueeze(1)
    proposals_z2 = proposals[:, 5].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_z1 = reference_boxes[:, 2].unsqueeze(1)

    reference_boxes_x2 = reference_boxes[:, 3].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 4].unsqueeze(1)
    reference_boxes_z2 = reference_boxes[:, 5].unsqueeze(1)

    # encoding scheme: log-scale diff in box widhts & rel diff (arithmetic)
    # in box centers (w. weights)
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    ex_depths = proposals_z2 - proposals_z1

    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights
    ex_ctr_z = proposals_z1 + 0.5 * ex_depths

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_depths = reference_boxes_z2 - reference_boxes_z1

    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights
    gt_ctr_z = reference_boxes_z1 + 0.5 * gt_depths

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dz = wz * (gt_ctr_z - ex_ctr_z) / ex_depths

    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)
    targets_dd = wd * torch.log(gt_depths / ex_depths)

    targets = torch.cat((targets_dx, targets_dy, targets_dz, targets_dw, targets_dh, targets_dd), dim=1)
    return targets


class BoxCoder:
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(
        self, weights: Tuple[float, float, float, float, float, float], bbox_xform_clip: float = math.log(1000.0 / 16)
    ) -> None:
        """
        Args:
            weights (6-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes: List[Tensor], proposals: List[Tensor]) -> List[Tensor]:
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)
        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_image, 0)

    def encode_single(self, reference_boxes: Tensor, proposals: Tensor) -> Tensor:
        """
        Encode a set of proposals with respect to some
        reference boxes

        Args:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes, proposals, weights)
        return targets

    def decode(self, rel_codes: Tensor, boxes: List[Tensor]) -> Tensor:
        torch._assert(
            isinstance(boxes, (list, tuple)),
            "This function expects boxes of type list or tuple.",
        )
        torch._assert(
            isinstance(rel_codes, torch.Tensor),
            "This function expects rel_codes of type torch.Tensor.",
        )
        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val
        if box_sum > 0:
            rel_codes = rel_codes.reshape(box_sum, -1)
        pred_boxes = self.decode_single(rel_codes, concat_boxes)
        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 6)
        return pred_boxes

    def decode_single(self, rel_codes: Tensor, boxes: Tensor) -> Tensor:
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Args:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """

        boxes = boxes.to(rel_codes.dtype)

        widths = boxes[:, 3] - boxes[:, 0]
        heights = boxes[:, 4] - boxes[:, 1]
        depths = boxes[:, 5] - boxes[:, 2]
        
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        ctr_z = boxes[:, 2] + 0.5 * depths

        wx, wy, wz, ww, wh, wd = self.weights
        dx = rel_codes[:, 0::6] / wx
        dy = rel_codes[:, 1::6] / wy
        dz = rel_codes[:, 2::6] / wz

        dw = rel_codes[:, 3::6] / ww
        dh = rel_codes[:, 4::6] / wh
        dd = rel_codes[:, 5::6] / wd

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)
        dd = torch.clamp(dd, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_ctr_z = dz * depths[:, None] + ctr_z[:, None]

        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]
        pred_d = torch.exp(dd) * depths[:, None]

        # Distance from center to box's corner.
        c_to_c_h = torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        c_to_c_w = torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        c_to_c_d = torch.tensor(0.5, dtype=pred_ctr_z.dtype, device=pred_d.device) * pred_d

        pred_boxes1 = pred_ctr_x - c_to_c_w
        pred_boxes2 = pred_ctr_y - c_to_c_h
        pred_boxes3 = pred_ctr_z - c_to_c_d

        pred_boxes4 = pred_ctr_x + c_to_c_w
        pred_boxes5 = pred_ctr_y + c_to_c_h
        pred_boxes6 = pred_ctr_z + c_to_c_d

        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4, pred_boxes5, pred_boxes6), dim=2).flatten(1)
        return pred_boxes


def expand_boxes(boxes, scale):
    # type: (Tensor, float) -> Tensor

    w_half = (boxes[:, 3] - boxes[:, 0]) * 0.5
    h_half = (boxes[:, 4] - boxes[:, 1]) * 0.5
    d_half = (boxes[:, 5] - boxes[:, 2]) * 0.5

    x_c = (boxes[:, 3] + boxes[:, 0]) * 0.5
    y_c = (boxes[:, 4] + boxes[:, 1]) * 0.5
    z_c = (boxes[:, 5] + boxes[:, 2]) * 0.5

    w_half *= scale
    h_half *= scale
    d_half *= scale

    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 3] = x_c + w_half

    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 4] = y_c + h_half
    
    boxes_exp[:, 2] = z_c - d_half
    boxes_exp[:, 5] = z_c + d_half
    return boxes_exp