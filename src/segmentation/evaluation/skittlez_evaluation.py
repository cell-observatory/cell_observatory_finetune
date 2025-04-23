from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from segmentation.metrics.metrics import Metric
from segmentation.evaluation.evaluator import DatasetEvaluator 

from segmentation.metrics.utils import (
    merge_instance_masks_binary,
    merge_instance_masks_logits,
)
    

class SkittlezInstanceEvaluator(DatasetEvaluator):
    """
    Evaluate Instance Segmentation metrics on Skittlez Dataset.
    It saves instance segmentation results in output_dir.
    """
    def __init__(self, 
                 metrics: Dict[str, Metric],
                 lower_is_better: bool=True,
                 ckpt_loss_key: str=None,
                 dataset_name: Optional[Union[str, Path]]=None,
                 output_dir: Optional[Union[str, Path]]=None,
                 detection_mode: bool=True
                 ):
        """
        Args: 
            dataset_name (str): the name of the dataset.
            output_dir (str): the directory to save the predictions.
        """
        super().__init__()
        self.metrics = metrics
        assert all(hasattr(metric, "aggregate") for metric in metrics.values()), \
            "All metrics should have a aggregate method"
        assert all(hasattr(metric, "reset") for metric in metrics.values()), \
            "All metrics should have a reset method"
        assert all(hasattr(metric, "__call__") for metric in metrics.values()), \
            "All metrics should be callable"
        
        self.lower_is_better = lower_is_better
        self.ckpt_loss_key = ckpt_loss_key

        self._dataset_name = dataset_name
        self._output_dir = output_dir

        self._predictions = {key: [] for key in self.metrics.keys()}
        self.detection_mode = detection_mode

    def reset(self):
        self._predictions = {key: [] for key in self.metrics.keys()}
        for metric in self.metrics.values():
            metric.reset()

    def process(self, targets, outputs):
        # TODO: move metrics computations to GPU & move part of this logic into a separate function
        #       in general, better abstractions are needed for metrics 
        pred_masks = []
        gt_masks = []
        for output, target in zip(outputs, targets):
            pred_mask = output["masks"]
            target_mask = target["masks"]
            target_mask = merge_instance_masks_binary(target_mask).cpu().numpy()
            if len(pred_mask) > 0:
                if self.detection_mode:
                    pred_masks.append(merge_instance_masks_logits(pred_mask).cpu().numpy())
                else:
                    pred_masks.append(pred_mask.cpu().numpy())
            else:
                pred_masks.append(np.zeros_like(target_mask))
            gt_masks.append(target_mask)

        for metric_name, metric in self.metrics.items():
            result = metric(pred_masks, gt_masks)
            # TODO: rewrite to better handle different metric result
            #       types, consider requiring all metrics to return a value
            self._predictions[metric_name].extend(result) if isinstance(result, list) else self._predictions[metric_name].append(result)

    def aggregate(self):
        for name, metric in self.metrics.items():
            self._predictions[name] = [float(metric.aggregate())]

    def evaluate(self):
        # TODO: incorporate more advanced evaluation/plotting logic
        self.aggregate()
        self._predictions = {k: v[0] for k, v in self._predictions.items()}
        if self.ckpt_loss_key is not None:
            ckpt_loss = self._predictions[self.ckpt_loss_key] if self.lower_is_better else -self._predictions[self.ckpt_loss_key]
            return self._predictions, ckpt_loss 
        return self._predictions