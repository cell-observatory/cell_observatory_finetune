"""
https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/evaluator.py

(ADD COPYRIGHT HERE)
"""


import os
import time
import json 
import logging
import datetime
from typing import List, Union

from collections import abc
from contextlib import ExitStack

import torch
from torch import nn
from torch.amp import autocast

from segmentation.utils.comm import inference_context
from segmentation.data.gather_dataset import gather_dataset
from segmentation.checkpoint.checkpoint import load_checkpoint
from segmentation.training.registry import build_dependency_graph_and_instantiate
from segmentation.evaluation.skittlez_evaluation import SkittlezInstanceEvaluator
from segmentation.evaluation.evaluator import DatasetEvaluator, DatasetEvaluators

import hydra
from hydra.utils import get_method, instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

logger = logging.getLogger("evaluation")
logger.setLevel(logging.DEBUG)


def inference_on_dataset(
    model,
    data_loader,
    evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None],
    warmup_iters=5,
    callbacks=None,
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.
        callbacks (dict of callables): a dictionary of callback functions which can be
            called at each stage of inference.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = torch.cuda.device_count()
    logger.info("Start inference on {} batches".format(len(data_loader)))
    inference_logbook = {}

    total = len(data_loader)  # inference data loader must have a fixed length
    
    if evaluator is None:
        raise ValueError("Evaluator is None. Please provide an evaluator.")        
    # TODO: test case where evaluator is list of evaluators
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(warmup_iters, total - 1)
    start_time = time.perf_counter()
    
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    # TODO: Rewrite context manager for clarity
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        dict.get(callbacks or {}, "on_start", lambda: None)()
        for idx, (inputs, targets) in enumerate(data_loader):
            # # DEBUG
            # if idx > 5:
            #     break
            step_utilization, step_vram = [], []
            total_data_time += time.perf_counter() - start_data_time

            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0
                step_utilization, step_vram = [], []

            start_compute_time = time.perf_counter()

            # invoke before_inference callback if exists
            dict.get(callbacks or {}, "before_inference", lambda: None)()
            
            inputs = inputs.cuda()   
            losses_dict, outputs = model(inputs, None)

            # DEBUG
            # print(f"MODEL MDE: {model.training}")
            # print(f"OUTPUTS: {outputs}")
            # idx = 0
            # boxes_test = [outputs[idx]['boxes'][i].cpu().numpy() for i in range(len(outputs[idx]['boxes']))]
            # import segmentation.utils.plot as plot_boxes
            # plot_boxes.plot_boxes(boxes_test, inputs.shape[-3:], save_path='/clusterfs/nvme/segment_4d/test_5/bx_test.tif')
            # import skimage
            # import numpy as np
            # from segmentation.metrics.utils import merge_instance_masks_logits, merge_instance_masks_binary         
            # test_gt_mask = merge_instance_masks_binary(targets[idx]['masks']).cpu().numpy()
            # skimage.io.imsave("/clusterfs/nvme/segment_4d/test_5/gt_mask_test.tif", test_gt_mask.astype(np.uint16))  
            # skimage.io.imsave("/clusterfs/nvme/segment_4d/test_5/mask_test.tif", outputs[idx]['masks'][0].cpu().numpy())
            # inputs = inputs[idx][0].cpu().numpy()
            # zmin, zmax = inputs.min(), inputs.max()
            # inputs = (inputs - zmin) / (zmax - zmin) 
            # u16 = np.rint(inputs * 65535.0).astype(np.int32) 
            # s16 = u16 - 32768
            # final = np.clip(s16, -32768, 32767).astype(np.int16)
            # print(f"INPUTS MAX: {inputs.max()}")
            # skimage.io.imsave('/clusterfs/nvme/segment_4d/test_5/gt_im.tif', final)
            # test_full_mask = merge_instance_masks_logits(outputs[idx]['masks'], threshold=0.2).cpu().numpy()
            # skimage.io.imsave("/clusterfs/nvme/segment_4d/test_5/full_mask_test.tif", test_full_mask.astype(np.uint16))
            # print(f"MAX OUTPUTS: {outputs[idx]['masks'][0].max()}")
            # print(f"MAX OUTPUTS: {outputs[idx]['masks'][1].max()}")
            # print(f"MAX OUTPUTS: {outputs[idx]['masks'][2].max()}")
            # raise ValueError("DEBUG")

            # invoke after_inference callback if exists
            dict.get(callbacks or {}, "after_inference", lambda: None)()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            cuda_util = torch.cuda.utilization()
            cuda_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
            step_utilization.append(cuda_util)
            step_vram.append(cuda_vram)

            start_eval_time = time.perf_counter()
            evaluator.process(targets, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start

            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                logger.info(
                    f"Inference done {idx + 1}/{total}. "
                    f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                    f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                    f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                    f"Total: {total_seconds_per_iter:.4f} s/iter. "
                    f"Cuda Memory Allocated: {cuda_vram:.4g} GB. "
                    f"Cuda Utilization: {cuda_util:.4g} GB. "
                    f"ETA={eta}"
                )
            
            inference_logbook[idx] = {
                "iteration": idx,
                # timings
                "data_time": data_seconds_per_iter,
                "compute_time": compute_seconds_per_iter,
                "eval_time": eval_seconds_per_iter,
                "total_time": total_seconds_per_iter,
                # cuda / vram utilization
                "cuda_utilization": cuda_util,
                "cuda_memory_allocated": cuda_vram,
            }

            start_data_time = time.perf_counter()

        dict.get(callbacks or {}, "on_end", lambda: None)()

    # Measure the time only for this worker
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))

    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str,
            total_compute_time / (total - num_warmup),
            num_devices,
        )
    )

    results = evaluator.evaluate()

    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results, inference_logbook


@hydra.main(config_path="../configs", config_name="skittlez_evaluation", version_base="1.2") 
def main(cfg: DictConfig):
    # Print full configuration (for debugging)
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    with open_dict(cfg):
        if cfg.gpu_workers == -1:
            cfg.gpu_workers = torch.cuda.device_count()
        cfg.worker_batch_size = cfg.datasets.batch_size // (
            cfg.workers * cfg.gpu_workers
        )
    
    # get dataloader
    test_dataloader = gather_dataset(cfg) 
    
    # instantiate model
    model = build_dependency_graph_and_instantiate(cfg.models)
    load_checkpoint(
            model_engine=model,
            opt=None,
            config=cfg,
            logger=logger,
            checkpointdir=cfg.checkpointdir,
            ckpt_suffix=cfg.ckpt_suffix,
            dtype=cfg.amp
        )
    model = model.cuda()
    
    # instantiate evaluator
    evaluator = build_dependency_graph_and_instantiate(cfg.metrics)

    # run inference
    results, inference_logbook = inference_on_dataset(
        model=model,
        data_loader=test_dataloader,
        evaluator=evaluator,
        warmup_iters=cfg.warmup_iters,
        callbacks=None,
    )

    # save results logs
    os.makedirs(os.path.join(cfg.results_dir, cfg.eval_type), exist_ok=True)
    with open(os.path.join(cfg.results_dir, cfg.eval_type, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
    with open(os.path.join(cfg.results_dir, cfg.eval_type, "logbook.json"), "w") as f:
        json.dump(inference_logbook, f, indent=4)
    
    logger.info(f"Results saved to {cfg.results_dir}")
    logger.info(f"Logbook saved to {cfg.results_dir}")
    logger.info(f"Results: {results}")
    logger.info(f"Logbook: {inference_logbook}")


if __name__ == "__main__":
    main()