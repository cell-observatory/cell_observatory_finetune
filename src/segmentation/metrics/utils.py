"""
https://github.com/MouseLand/cellpose/blob/6398d13700a95bbc9b2bec528d267447984358cf/cellpose/metrics.py#L4

Copyright © 2020 Howard Hughes Medical Institute

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
Neither the name of HHMI nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


from typing import Dict, Callable

import numpy as np
from numba import jit
from scipy.optimize import linear_sum_assignment

import torch


@jit(nopython=True)
def _label_overlap(x, y):
    """
    Compute the pixel-wise overlap matrix between two labeled mask arrays.

    This function calculates how many pixels (or voxels) each ground truth label in `x` overlaps with 
    each predicted label in `y`. Both arrays must be of the same shape, and label `0` is treated as background. 
    The result is a 2D matrix where entry (i, j) gives the number of pixels where `x == i` and `y == j`.

    Args:
        x (np.ndarray): Integer-labeled mask array. Background is 0.
        y (np.ndarray): Integer-labeled mask array. Background is 0.

    Returns:
        overlap (np.ndarray): 2D array of shape [x.max() + 1, y.max() + 1] containing pixel overlap counts.
    """
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap

def _intersection_over_union(masks_true, masks_pred):
    """
    Calculate the intersection over union (IoU) between all pairs of ground truth and predicted masks.

    This function builds an overlap matrix by counting intersecting voxels between ground truth and predicted mask 
    labels. Label 0 is treated as background and included. The matrix is shaped as [n_true + 1, n_pred + 1], where 
    entry (i, j) holds the intersection area between true mask `i` and predicted mask `j`. To compute the union, the 
    sum of each true and predicted mask area is calculated and the overlap is subtracted (since it is counted twice). 
    The result is a dense lookup table of IoU values for each pair of labels.

    Args:
        masks_true (np.ndarray): Integer-labeled 3D or 2D array of ground truth masks. Background is 0.
        masks_pred (np.ndarray): Integer-labeled 3D or 2D array of predicted masks. Background is 0.

    Returns:
        iou (np.ndarray): A 2D array of shape [n_true + 1, n_pred + 1] containing pairwise IoU values.
    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou


def _true_positive(iou, th):
    """
    Calculate the number of true positives given a pairwise IoU matrix and a threshold.

    This function computes the optimal one-to-one matching between ground truth and predicted masks using the 
    Hungarian algorithm (linear sum assignment). A cost matrix is defined based on IoU values and the given threshold, 
    where lower costs represent better matches. The optimal assignment minimizes this cost. After matching, the 
    true positives are counted as pairs whose IoU exceeds the threshold.

    Args:
        iou (np.ndarray): A 2D array of shape [n_true, n_pred] representing IoU values between each pair of ground truth 
                        and predicted masks.
        th (float): IoU threshold for determining a true positive match.

    Returns:
        tp (int): Number of true positives at the specified threshold.
    """
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2 * n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    tp = match_ok.sum()
    return tp


def average_precision(masks_true, masks_pred, threshold=[0.5, 0.6, 0.7, 0.8, 0.9]):
    """
    Average precision estimation: AP = TP / (TP + FP + FN)

    Args:
        masks_true (Union[List[np.ndarray], np.ndarray]): 
            Ground truth masks. Each mask is an integer-labeled array where 0 indicates background 
            and positive integers denote instance labels. Either a list of 3D arrays (one per image) 
            or a single 3D array.
        
        masks_pred (Union[List[np.ndarray], np.ndarray]): 
            Predicted masks in the same format as `masks_true`.

        threshold (Union[List[float], float]): 
            One or more IoU thresholds at which to compute average precision (e.g. [0.5, 0.75, 0.9]).

    Returns:
        ap (np.ndarray): 
            Array of shape [len(masks_true), len(threshold)] giving average precision per image per threshold.

        tp (np.ndarray): 
            Array of true positives per image per threshold.

        fp (np.ndarray): 
            Array of false positives per image per threshold.

        fn (np.ndarray): 
            Array of false negatives per image per threshold.
    """
    not_list = False
    if not isinstance(masks_true, list):
        masks_true = [masks_true]
        masks_pred = [masks_pred]
        not_list = True
    if not isinstance(threshold, list) and not isinstance(threshold, np.ndarray):
        threshold = [threshold]

    if len(masks_true) != len(masks_pred):
        raise ValueError(
            "metrics.average_precision requires len(masks_true)==len(masks_pred)")

    ap = np.zeros((len(masks_true), len(threshold)), np.float32)
    tp = np.zeros((len(masks_true), len(threshold)), np.float32)
    fp = np.zeros((len(masks_true), len(threshold)), np.float32)
    fn = np.zeros((len(masks_true), len(threshold)), np.float32)
    n_true = np.array(list(map(np.max, masks_true)))
    n_pred = np.array(list(map(np.max, masks_pred)))

    for n in range(len(masks_true)):
        #_,mt = np.reshape(np.unique(masks_true[n], return_index=True), masks_pred[n].shape)
        if n_pred[n] > 0:
            iou = _intersection_over_union(masks_true[n], masks_pred[n])[1:, 1:]
            for k, th in enumerate(threshold):
                tp[n, k] = _true_positive(iou, th)
        fp[n] = n_pred[n] - tp[n]
        fn[n] = n_true[n] - tp[n]
        ap[n] = tp[n] / (tp[n] + fp[n] + fn[n])

    if not_list:
        ap, tp, fp, fn = ap[0], tp[0], fp[0], fn[0]
    return ap, tp, fp, fn


def merge_instance_masks_binary(masks: torch.Tensor) -> torch.Tensor:
    merged = torch.zeros_like(masks[0], dtype=torch.int32)
    for idx, m in enumerate(masks):
        merged[m.bool()] = idx + 1
    return merged


def merge_instance_masks_logits(mask_probs: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    masked_probs = torch.where(mask_probs > threshold, mask_probs, torch.tensor(0.0, device=mask_probs.device, dtype=mask_probs.dtype))
    max_vals, max_ids = masked_probs.max(dim=0) # get instance id with max prob for each voxel
    return torch.where(max_vals > 0, max_ids + 1, torch.tensor(0, device=mask_probs.device, dtype=max_ids.dtype))

def compute_mean(x):
    return sum(x) / len(x) if len(x) > 0 else 0.0