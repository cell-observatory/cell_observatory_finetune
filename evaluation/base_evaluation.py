from typing import Dict, List

from cell_observatory_finetune.evaluation.metrics.metrics import TrainLosses
from cell_observatory_finetune.evaluation.evaluator import DatasetEvaluator 


class BaseEvaluator(DatasetEvaluator):
    """
    Evaluate model loss on validation dataset.
    """
    # TODO: consider move initialization and reset
    #       to base class definition
    def __init__(self, training_metrics: List[Dict[str, str]]):
        self._results = {
            metric: None
            for training_metric in training_metrics
            for metric in training_metric.keys()
        }

        self.metrics = {
            metric: TrainLosses(reduce_method=red_method)
            for training_metric in training_metrics
            for metric, red_method in training_metric.items()
        }

    def reset(self):
        self._results = {m: None for m in self._results.keys()}

    def process(self, data_sample, outputs, loss_dict): 
        for metric, metric_impl in self.metrics.items():
            metric_impl(outputs, data_sample, loss_dict)

    def aggregate(self):
        for metric, metric_impl in self.metrics.items():
            self._results[metric] = float(metric_impl.aggregate())

    def evaluate(self):
        self.aggregate()
        return self._results