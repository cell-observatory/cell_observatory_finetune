"""
https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/evaluator.py

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


import os
import time
import json 
import logging
import datetime
from pathlib import Path
from collections import abc
from typing import List, Union, Dict, Any, Callable, Optional, Sequence, Tuple

import torch

from cell_observatory_finetune.data.utils import move_to_device
from cell_observatory_finetune.utils.comm import inference_context
from cell_observatory_finetune.data.dataloaders import get_dataloader
from cell_observatory_finetune.utils.checkpoint import load_checkpoint
from cell_observatory_finetune.data.structures.sample_objects.data_sample import DataSample
from cell_observatory_finetune.train.registry import build_dependency_graph_and_instantiate
from cell_observatory_finetune.evaluation.evaluator import DatasetEvaluator, DatasetEvaluators

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

logger = logging.getLogger("evaluation")
logger.setLevel(logging.DEBUG)


def inference_on_dataset(
    model: Callable[..., Any],
    data_loader: Sequence[Any],
    evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None],
    warmup_iters: int = 5,
    callbacks: Optional[Dict[str, Callable[..., Any]]] = None,
) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
    """
    Run model on data_loader output and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.
            If it's an nn.Module, it will be temporarily set to `eval` mode.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.
        callbacks (dict of callables): a dictionary of callback functions which can be
            called at each stage of inference.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    logger.info("Start inference on {} batches".format(len(data_loader)))
    
    inference_logbook = {}
    # inference dataloader must have a fixed length
    total = len(data_loader)  
    num_devices = torch.cuda.device_count()
    
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
    with inference_context(model):
        with torch.no_grad():
            start_data_time = time.perf_counter()
            dict.get(callbacks or {}, "on_start", lambda: None)()
            
            for idx, data_sample in enumerate(data_loader):
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
                
                data_sample = move_to_device(data_sample, auto_transfer=True)  
                losses_dict, outputs = model(data_sample)

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
                evaluator.process(DataSample.from_dict(data_sample).gt_instances, outputs)
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
                        f"Cuda Utilization: {cuda_util:.4g} %. "
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

    # an evaluator may return None when not in main process
    # replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results, inference_logbook


# modify Hydra config on cmd line to evaluate different models
@hydra.main(config_path="../configs", config_name="skittlez_evaluation", version_base="1.2") 
def main(cfg: DictConfig):
    # print full configuration (for debugging)
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    Path(cfg.outdir).mkdir(exist_ok=True, parents=True)

    with open_dict(cfg):
        if cfg.clusters.total_gpus is None or cfg.clusters.batch_size is None:
            raise ValueError("total_gpus and batch_size must be specified in the Hydra configuration.")
        cfg.clusters.worker_batch_size = cfg.clusters.batch_size // cfg.clusters.total_gpus
    
    # get dataloader
    test_dataloader = get_dataloader(cfg) 
    
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