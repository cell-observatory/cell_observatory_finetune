from typing import List, Dict, Any, Callable

import numpy as np

from segmentation.metrics.metrics import Metric
from segmentation.evaluation.evaluator import DatasetEvaluator 

from segmentation.metrics.utils import (
    merge_instance_masks_binary,
    merge_instance_masks_logits,
)
    

class SkittlezInstanceEvaluator(DatasetEvaluator):
    """
    Evaluate Instance Quality metrics on Skittlez.
    It saves instance segmentation predictions in `output_dir`.
    """
    def __init__(self, 
                 metrics: Dict[str, Metric],
                 lower_is_better: bool=True,
                 ckpt_loss_key: str=None,
                 dataset_name=None,
                 output_dir=None,
                 detection_mode=True
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
        # TODO: Move metrics computation to GPU & move this logic into separate function
        pred_masks = []
        gt_masks = []
        for output, target in zip(outputs, targets):
            pred_mask = output["masks"]
            target_mask = target["masks"]
            if len(pred_mask) > 0:
                if self.detection_mode:
                    pred_masks.append(merge_instance_masks_logits(pred_mask).cpu().numpy())
                else:
                    pred_masks.append(pred_mask.cpu().numpy())
            else:
                pred_masks.append(np.zeros_like(target_mask))
            gt_masks.append(merge_instance_masks_binary(target_mask).cpu().numpy())

            # import skimage
            # from segmentation.utils.plot import plot_boxes
            # gt_box = [targets[0]["boxes"][i].cpu().numpy() for i in range(len(targets[0]["boxes"]))]
            # t_box = [outputs[0]["boxes"][i].cpu().numpy() for i in range(len(outputs[0]["boxes"]))]
            # print("DEBUG GT BOXES:", gt_box[:1])
            # print("DEBUG PRED BOXES:", t_box[:1])
            # plot_boxes(gt_box, image_shape= pred_masks[0].shape, save_path="/clusterfs/nvme/segment_4d/test_5/all_gt_test_box.tif")
            # plot_boxes(gt_box, sample_indices=[0], image_shape= pred_masks[0].shape, sample_num=5, save_path="/clusterfs/nvme/segment_4d/test_5/gt_test_box.tif")
            # plot_boxes(t_box, sample_indices=[0], image_shape=pred_masks[0].shape, sample_num=5, save_path="/clusterfs/nvme/segment_4d/test_5/pred_test_box.tif")
            # skimage.io.imsave("/clusterfs/nvme/segment_4d/test_5/pred_test.tif", pred_masks[0].astype(np.uint16))
            # skimage.io.imsave("/clusterfs/nvme/segment_4d/test_5/gt_test.tif", gt_masks[0])
            # raise ValueError("Debugging maskrcnn_inference") 

        for metric_name, metric in self.metrics.items():
            result = metric(pred_masks, gt_masks)
            # TODO: more generic way to handle different metric results
            #       consider requiring all metrics to return a value
            self._predictions[metric_name].extend(result) if isinstance(result, list) else self._predictions[metric_name].append(result)

    def aggregate(self):
        for name, metric in self.metrics.items():
            self._predictions[name] = [float(metric.aggregate())]

    def evaluate(self):
        # TODO: Consider incorporating more advanced
        #       evaluation logic
        self.aggregate()
        if self.ckpt_loss_key is not None:
            ckpt_loss = self._predictions[self.ckpt_loss_key][0] if self.lower_is_better else -self._predictions[self.ckpt_loss_key][0]
        return self._predictions, ckpt_loss if self.ckpt_loss_key is not None else self._predictions