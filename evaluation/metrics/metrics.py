import abc

from cell_observatory_finetune.evaluation.metrics.utils import (
    average_precision
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


class TrainLosses(Metric):
    def __init__(self, reduce_method: str = "mean"):
        self.reduce_method = reduce_method
        self.loss_values = []

    def __call__(self, outputs, targets, loss):
        self.loss_values.append(loss.item())

    def aggregate(self):
        if self.reduce_method == "mean":
            return sum(self.loss_values) / len(self.loss_values) \
                if self.loss_values else 0.0
        elif self.reduce_method == "min":
            return min(self.loss_values) if self.loss_values \
                else 0.0
        elif self.reduce_method == "max":
            return max(self.loss_values) if self.loss_values \
                else 0.0
        else:
            raise ValueError(f"Unknown reduce method: {self.reduce_method}")

    def reset(self):
        self.loss_values.clear()


class AveragePrecision(Metric):
    def __init__(self, 
                 reduce_method: str = "mean", 
                 iou_threshold: float = 0.5
    ):
        self.reduce_method = reduce_method
        self.iou_threshold = iou_threshold
        self.ap_values = []

    def __call__(self, outputs, targets):
        # TODO: consider relabelling masks to be contiguous for safety
        ap, *_ = average_precision(targets, outputs, self.iou_threshold)
        self.ap_values.extend(ap)
        return ap

    def aggregate(self):
        return self.reduce_method(self.ap_values)

    def reset(self):
        self.ap_values.clear()