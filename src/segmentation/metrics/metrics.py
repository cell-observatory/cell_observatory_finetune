import numpy as np
from typing import Dict, Callable, Optional

from segmentation.metrics.utils import (
    average_precision,
    compute_mean
)


class Metric:
    def __call__(self, outputs, targets):
        pass

    def aggregate(self):
        pass

    def reset(self):
        pass


class AveragePrecision(Metric):
    def __init__(self, aggregate_method: Callable = compute_mean, iou_threshold: float = 0.5):
        self.aggregate_method = aggregate_method
        self.iou_threshold = iou_threshold
        self.ap_values = []

    def __call__(self, outputs, targets):
        # TODO: Relabel masks to be contiguous for safety
        ap, *_ = average_precision(targets, outputs, self.iou_threshold)
        self.ap_values.extend(ap)
        return ap

    def aggregate(self):
        return self.aggregate_method(self.ap_values)

    def reset(self):
        self.ap_values.clear()