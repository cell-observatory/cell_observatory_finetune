from typing import Dict, List

import torch

from cell_observatory_finetune.evaluation.metrics.metrics import Metric
from cell_observatory_finetune.evaluation.evaluator import DatasetEvaluator 
from cell_observatory_finetune.data.structures.sample_objects.instances import Instances

from cell_observatory_finetune.evaluation.metrics.utils import (
    merge_instance_masks_binary,
    merge_instance_masks_logits,
)


class SkittlezInstanceEvaluator(DatasetEvaluator):
    """
    Evaluate Instance Segmentation metrics on Skittlez Dataset.
    """
    def __init__(self, metrics: Dict[str, Metric]):
        super().__init__()
        self.metrics = metrics
        assert all(hasattr(metric, "aggregate") for metric in metrics.values()), \
            "All metrics should have a aggregate method"
        assert all(hasattr(metric, "reset") for metric in metrics.values()), \
            "All metrics should have a reset method"
        assert all(hasattr(metric, "__call__") for metric in metrics.values()), \
            "All metrics should be callable"

        self._results = {metric: None for metric in self.metrics.keys()}

    def reset(self):
        self._results = {m: None for m in self.metrics.keys()}
        for m in self.metrics.values():
            m.reset()

    def process(self, targets, outputs): 
        assert isinstance(targets, list) and isinstance(outputs, list), \
            "Targets and outputs must be lists."
        assert all(isinstance(t, Instances) for t in targets), \
            "Every element in `targets` must be an Instances object."
        assert all(isinstance(o, Instances) for o in outputs), \
            "Every element in `outputs` must be an Instances object."
        output_masks = [output.masks.tensor for output in outputs]
        target_masks = [target.masks.tensor for target in targets]

        # for error checks in evaluation, we are chiefly concerned
        # with making sure that good evaluation scores only occur
        # when the model is performing well, if we were to 
        # simply ignore empty predictions, we could end up
        # with scenarios where the model sometimes predicts
        # nothing, and sometimes predicts something, and the
        # evaluation scores would be high, even though the model
        # is not performing well. Saving these model states
        # is not desirable, so for empty predictions, we 
        # simply create a zero mask of the same shape as the target masks.
        # this should give a poor evaluation score
        for idx, output_mask in enumerate(output_masks):
            if output_mask.numel() == 0:
                output_masks[idx] = torch.zeros_like(target_masks[idx])

        pred_masks, gt_masks = [], []
        for pred_mask, target_mask in zip(output_masks, target_masks):
            target_mask = merge_instance_masks_binary(target_mask).cpu().numpy()
            pred_mask = merge_instance_masks_logits(pred_mask).cpu().numpy()
            pred_masks.append(pred_mask)
            gt_masks.append(target_mask)

        for metric, metric_impl in self.metrics.items():
            metric_impl(pred_masks, gt_masks)
    
    def aggregate(self):
        for metric, metric_impl in self.metrics.items():
            self._results[metric] = float(metric_impl.aggregate())

    def evaluate(self):
        self.aggregate()
        return self._results