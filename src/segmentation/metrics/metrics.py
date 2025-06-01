import abc
from typing import Callable

from segmentation.metrics.utils import (
    average_precision,
    compute_mean
)


class Metric(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, outputs, targets):
        pass

    @abc.abstractmethod
    def aggregate(self):
        pass
    
    @abc.abstractmethod
    def reset(self):
        pass


class AveragePrecision(Metric):
    def __init__(self, aggregate_method: Callable = compute_mean, iou_threshold: float = 0.5):
        self.aggregate_method = aggregate_method
        self.iou_threshold = iou_threshold
        self.ap_values = []

    def __call__(self, outputs, targets):
        # TODO: consider relabelling masks to be contiguous for safety
        ap, *_ = average_precision(targets, outputs, self.iou_threshold)
        self.ap_values.extend(ap)
        return ap

    def aggregate(self):
        return self.aggregate_method(self.ap_values)

    def reset(self):
        self.ap_values.clear()