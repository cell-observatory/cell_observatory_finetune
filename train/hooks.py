"""
https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/hooks.py
https://github.com/open-mmlab/mmengine/tree/main/mmengine/hooks

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
import sys
import time
import math
import ujson
import logging
import operator
import datetime
from enum import Enum
from pathlib import Path
from collections import Counter
from typing import Optional, Union, Sequence, List, Literal

import torch
import torch.nn as nn
from torchinfo import summary
from torch.profiler import ProfilerActivity
from fvcore.common.timer import Timer

from cell_observatory_finetune.utils.logging import EventWriter
from cell_observatory_finetune.utils.comm import is_main_process, process_rank


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HOOK_PRIORITY(Enum):
    VERY_LOW = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERY_HIGH = 4


class HookBase:
    """
    Base class for hooks that can be registered with :class:`BaseTrainer`.
    """

    # A weak reference to the trainer object. 
    # Set by the trainer when the hook is registered.
    trainer = None
    
    # The priority of the hook.
    # Hooks with higher priority will be executed earlier.
    PRIORITY = HOOK_PRIORITY.MEDIUM

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def before_epoch(self):
        """
        Called before train epoch.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_backward(self):
        """
        Called after the backward pass of each iteration.
        """
        pass

    def after_step(self, data_sample, outputs, loss_dict):
        """
        Called after each iteration.
        """
        pass

    def after_epoch(self):
        """
        Called after train epoch.
        """
        pass

    def before_val_step(self):
        """
        Called before each validation step.
        """
        pass

    def after_val_step(self, data_sample, outputs, loss_dict):
        """
        Called after each validation step.
        """
        pass

    def before_validation(self):
        """
        Called before the validation loop starts.
        """
        pass

    def after_validation(self):
        """
        Called after the validation loop ends.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_test(self):
        """
        Called before the test loop starts.
        """
        pass

    def after_test(self):
        """
        Called after the test loop ends.
        """
        pass

    def before_test_step(self):
        """
        Called before each test step.
        """
        pass

    def after_test_step(self, data_sample, outputs, loss_dict):
        """
        Called after each test step.
        """
        pass

    def state_dict(self):
        """
        Hooks are stateless by default, but can be made checkpointable by
        implementing `state_dict` and `load_state_dict`.
        """
        return {}


class AnomalyDetector(HookBase):
    """Wrap each epoch in torch.autograd.detect_anomaly."""
    def __init__(self):
        self._anom_ctx = torch.autograd.set_detect_anomaly(True, check_nan=False)
        self.loss_nans = 0

    def before_epoch(self):
        self._anom_ctx.__enter__()

    def after_epoch(self):
        self._anom_ctx.__exit__(None, None, None)

    def after_step(self, data_sample, outputs, loss_dict):
        if torch.isnan(loss_dict["step_loss"]):
            self.loss_nans += 1
            logger.warning(f"Step loss is {loss_dict['step_loss']} \
                           for step {self.trainer._iter} in epoch {self.trainer._epoch}")
            if self.loss_nans > 5:
                raise Exception(f"Step loss is {loss_dict['step_loss']} \
                                for step {self.trainer._iter} in epoch {self.trainer._epoch}.")


class SamplerSetter(HookBase):
    """
    A hook that sets the sampler for the trainer.
    """
    def before_epoch(self):
        if self.trainer.ray_context.get_world_size() > 1:
            self.trainer.train_dataloader.sampler.set_epoch(self.trainer._epoch)


class LRScheduler(HookBase):
    """
    A hook which executes a scheduler step and summarizes the LR
    for each parameter group in the optimizer.
    """
    def before_train(self):
        self.optimizer = self.trainer.opt
        self.scheduler = self.trainer.scheduler

        self._group_labels = [
            g.get("name", f"group{i}") for i, g in enumerate(self.optimizer.param_groups)
        ]
        self._best_param_group_id = self.get_best_param_group_id(self.optimizer)

    # NOTE: is it more desirable to summarize LR or use current approach?
    @staticmethod
    def get_best_param_group_id(optimizer):
        # NOTE: some heuristics on what LR to summarize
        # summarize the param group with most parameters
        largest_group = max(len(g["params"]) for g in optimizer.param_groups)

        if largest_group == 1:
            # if all groups have one parameter,
            # then find the most common initial LR, and use it for summary
            lr_count = Counter([g["lr"] for g in optimizer.param_groups])
            lr = lr_count.most_common()[0][0]
            for i, g in enumerate(optimizer.param_groups):
                if g["lr"] == lr:
                    return i
        else:
            for i, g in enumerate(optimizer.param_groups):
                if len(g["params"]) == largest_group:
                    return i

    def after_step(self, data_sample, outputs, loss_dict):
        lr = self.optimizer.param_groups[self._best_param_group_id]["lr"]
        self.trainer.event_recorder.put_scalar("lr", lr)
        # for label, group in zip(self._group_labels, self.optimizer.param_groups):
        #     self.trainer.event_recorder.put_scalar(f"lr/{label}", group["lr"])
        self.scheduler.step(epoch=self.trainer._epoch)


class IterationTimer(HookBase):
    """
    Track the time spent for each iteration (each run_step call in the trainer).
    Print a summary in the end of training.

    Under the convention that :meth:`before_step` of all hooks should only
    take negligible amount of time, the :class:`IterationTimer` hook should be
    placed at the beginning of the list of hooks to obtain accurate timing.
    """

    # setting priority to high to ensure that 
    # this hook runs early in the hook chain
    PRIORITY = HOOK_PRIORITY.HIGH

    def __init__(self, warmup_iter=3):
        """
        Args:
            warmup_iter (int): the number of iterations at the beginning to exclude
                from timing.
        """
        self._warmup_iter = warmup_iter

        # train step timer and
        # train epoch timer
        self._step_timer = Timer()
        self._epoch_timer = Timer()
        
        # validation step timer and
        # validation epoch timer
        self._val_step_timer = Timer()
        self._val_timer = Timer()
        
        # test step timer
        # total time spent in test
        # given by the difference
        # between _start_time and current time
        self._test_timer = Timer()
        
        # for the total time spent not in hooks
        # different from time between
        # _start_time and current time
        # which includes time spent in hooks
        # and anywhere not in train step
        self._total_timer = Timer()

        # for the overall training time
        self._start_time = time.perf_counter()

    def before_train(self):
        self._start_time = time.perf_counter()
        self._total_timer.reset()
        self._total_timer.pause()

    def after_train(self):
        total_time = time.perf_counter() - self._start_time
        total_time_minus_hooks = self._total_timer.seconds()
        hook_time = total_time - total_time_minus_hooks

        num_iter = self.trainer._iter + 1 - self.trainer.start_iter - self._warmup_iter

        if num_iter > 0 and total_time_minus_hooks > 0:
            # speed is meaningful only after warmup
            logger.info(
                "Overall training speed: {} iterations in {} ({:.4f} s / it)".format(
                    num_iter,
                    str(datetime.timedelta(seconds=int(total_time_minus_hooks))),
                    total_time_minus_hooks / num_iter,
                )
            )

        logger.info(
            "Total training time: {} ({} on hooks/not train step)".format(
                str(datetime.timedelta(seconds=int(total_time))),
                str(datetime.timedelta(seconds=int(hook_time))),
            )
        )

    def before_step(self):
        self._step_timer.reset()
        self._total_timer.resume()

    def after_step(self, data_sample, outputs, loss_dict):
        # +1 because we're in after_step, the current step is done
        # but not yet counted
        iter_done = self.trainer._iter - self.trainer.start_iter + 1
        if iter_done >= self._warmup_iter:
            sec = self._step_timer.seconds()
            self.trainer.event_recorder.put_scalars(step_time=sec)
        else:
            # reset _total_timer and _start_time
            # to avoid counting the warmup iterations
            self._start_time = time.perf_counter()
            self._total_timer.reset()

        # _total_timer only counts
        # total time in step excluding hooks
        self._total_timer.pause()

    def before_epoch(self):
        """
        Reset the timer at the beginning of each epoch.
        """
        self._epoch_timer.reset()

    def after_epoch(self):
        sec = self._epoch_timer.seconds()
        self.trainer.event_recorder.put_scalars(epoch_time=sec, scope="epoch")

        remaining_epochs = self.trainer._max_epochs - (self.trainer._epoch + 1)
        eta = sec * remaining_epochs / 3600
        self.trainer.event_recorder.put_scalars(eta=eta)

    def before_validation(self):
        # stop epoch timer
        # to omit counting
        # validation time
        self._val_timer.reset()
        self._epoch_timer.pause()

    def after_validation(self):
        # resume the epoch timer
        # after the validation loop
        sec = self._val_timer.seconds()
        self.trainer.event_recorder.put_scalars(val_time=sec, \
                                                scope="epoch")
        self._epoch_timer.resume()

    def before_val_step(self):
        """
        Reset the timer at the beginning of each validation step.
        """
        self._val_step_timer.reset()

    def after_val_step(self, data_sample, outputs, loss_dict):
        """
        Record the time spent on the validation step.
        """
        sec = self._val_step_timer.seconds()
        self.trainer.event_recorder.put_scalars(val_step_time=sec)

        # Reset the timer for the next validation step
        self._val_step_timer.reset()
    
    def before_test(self):
        """
        Reset the timer at the beginning of each test step.
        """
        self._start_time = time.perf_counter()
        self._test_timer.reset()
        self._total_timer.reset()
        self._total_timer.pause()

    def after_test(self):
        total_time = time.perf_counter() - self._start_time
        total_time_minus_hooks = self._total_timer.seconds()
        hook_time = total_time - total_time_minus_hooks

        num_iter = self.trainer._iter + 1 - self.trainer.start_iter - self._warmup_iter

        if num_iter > 0 and total_time_minus_hooks > 0:
            # speed is meaningful only after warmup
            logger.info(
                "Overall test speed: {} iterations in {} ({:.4f} s / it)".format(
                    num_iter,
                    str(datetime.timedelta(seconds=int(total_time_minus_hooks))),
                    total_time_minus_hooks / num_iter,
                )
            )

        logger.info(
            "Total test time: {} ({} on hooks/not train step)".format(
                str(datetime.timedelta(seconds=int(total_time))),
                str(datetime.timedelta(seconds=int(hook_time))),
            )
        )

    def before_test_step(self):
        self._test_timer.reset()
        self._total_timer.resume()

    def after_test_step(self, data_sample, outputs, loss_dict):
        # +1 because we're in after_step, the current step is done
        # but not yet counted
        iter_done = self.trainer._iter - self.trainer.start_iter + 1
        if iter_done >= self._warmup_iter:
            sec = self._test_timer.seconds()
            self.trainer.event_recorder.put_scalars(test_step_time=sec)
        else:
            # reset _total_timer and _start_time
            # to avoid counting the warmup iterations
            self._start_time = time.perf_counter()
            self._total_timer.reset()

        # _total_timer only counts
        # total time in step excluding hooks
        self._total_timer.pause()


class PeriodicWriter(HookBase):
    """
    Write events with EventWriters periodically.
    It is executed every ``period`` iterations/epochs
    and after the last epoch.
    """

    def __init__(self, writers, period):
        """
        Args:
            writers (list[EventWriter]): a list of EventWriter instances
                to write events to.
            period (int): the period of writing events, in epochs.
        """
        self._period = period
        self._writers = writers
        
        for w in self._writers.writers:
            assert isinstance(w, EventWriter), "All writers must be EventWriter instances."

    def after_epoch(self):
        if (self.trainer._epoch + 1) % self._period == 0:
            self._writers.write()
            # for writer in self._writers:
            #     writer.write()

            self.trainer.event_recorder.clear()

    def after_train(self):
        # for writer in self._writers:
            # write last epoch's data before closing
        self._writers.write()
        self._writers.close()

    def after_test(self):
        """
        Write events after the test loop ends.
        """
        self._writers.write()
        self._writers.close()


class PeriodicCheckpointer(HookBase):
    """
    Checkpointing, executed every ``period`` epoch and after the last epoch.
    """
    def __init__(self, period=1, file_prefix="latest_model"):
        self.period = period
        self.file_prefix = file_prefix

    def after_epoch(self):
        """
        Checkpointing is done after each epoch.
        """
        if (self.trainer._epoch + 1) % self.period == 0:
            self.trainer.checkpoint_manager.save(prefix=self.file_prefix, 
                                                 epoch=self.trainer._epoch + 1,
                                                best_loss=self.trainer.best_metric,
                                                iter=self.trainer._iter
                                                #  best_loss=self.trainer._best_loss
                                                 )
        
    def after_train(self):
        """
        Checkpointing is done after the last epoch.
        """
        if self.trainer._epoch + 1 >= self.trainer._max_epochs:
            self.trainer.checkpoint_manager.save(prefix=self.file_prefix, 
                                                 epoch=self.trainer._epoch + 1,
                                                 best_loss=self.trainer.best_metric,
                                                 iter=self.trainer._iter
                                                #  best_loss=self.trainer._best_loss
                                                )


class TorchMemoryStats(HookBase):
    """
    Writes pytorch's cuda memory statistics periodically.
    """

    def __init__(self, 
                 step_period=20, 
                 epoch_period=1, 
                 max_runs=10, 
                 logdir=None
    ):
        """
        Args:
            period (int): Output stats each 'period' iterations
            max_runs (int): Stop the logging after 'max_runs'
        """
        self._step_period = step_period
        self._epoch_period = epoch_period

        self._max_runs = max_runs
        self._runs = 0

        self._logdir = Path(logdir) / 'memory' 
        self._logdir.mkdir(parents=True, exist_ok=True)

    def after_step(self, data_sample, outputs, loss_dict):
        if self._runs > self._max_runs:
            return

        if (self.trainer._iter + 1) % self._step_period == 0:
            if torch.cuda.is_available():
                max_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
                reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
                max_allocated_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)

                self.trainer.event_recorder.put_scalars(
                    max_reserved_mem=max_reserved_gb,
                    reserved_mem=reserved_gb,
                    max_allocated_mem=max_allocated_gb,
                    allocated_mem=allocated_gb,
                )

                self._runs += 1
                torch.cuda.reset_peak_memory_stats()

    def after_epoch(self):
        if (self.trainer._epoch + 1) % self._epoch_period == 0:
            mem_log = torch.cuda.memory_summary()

            # TODO: support for saving table to
            #       wandb/tensorboard
            if is_main_process():
                with (self._logdir / f'{self.trainer._epoch}.log').open('w') as f:
                    f.write(str(mem_log))

    def after_test_step(self, data_sample, outputs, loss_dict):
        """
        Write memory stats after each test step.
        """
        if self._runs > self._max_runs:
            return

        if (self.trainer._iter + 1) % self._step_period == 0:
            if torch.cuda.is_available():
                max_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
                reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
                max_allocated_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)

                self.trainer.event_recorder.put_scalars(
                    max_reserved_mem=max_reserved_gb,
                    reserved_mem=reserved_gb,
                    max_allocated_mem=max_allocated_gb,
                    allocated_mem=allocated_gb,
                )

                self._runs += 1
                torch.cuda.reset_peak_memory_stats()

    def after_test(self):
        """
        Write memory stats after the test loop ends.
        """
        mem_log = torch.cuda.memory_summary()
        if is_main_process():
            os.makedirs(self._logdir / 'test', exist_ok=True)
            with (self._logdir / 'test' / 'memory_test.log').open('w') as f:
                f.write(str(mem_log))


# TODO: support for saving table to wandb/tensorboard
class ModelSummaryHook(HookBase):
    """
    A hook that summarizes the model architecture and parameters.
    It is executed once at the beginning of training.
    """

    def __init__(self, 
                 logdir: Union[str, Path], 
                 input_shape: tuple[int], 
                 batch_size: int
    ):
        self._logdir = Path(logdir)
        assert self._logdir.is_dir(), f"Log directory does \
            not exist: {self._logdir}"

        self.batch_size = batch_size
        self.input_shape = input_shape

    def before_train(self):
        if is_main_process():
            self.summarize_model(
                model=self.trainer.model,
                inputs=self.input_shape,
                batch_size=self.batch_size,
                logdir=self._logdir
            )
    
    def summarize_model(self, 
                        model: nn.Module, 
                        inputs: tuple, 
                        batch_size: int, 
                        logdir: Path
    ):
        model_logbook = {}
        model_stats = summary(
            model=model,
            input_size=inputs,
            depth=5,
            col_width=25,
            col_names=["kernel_size", "output_size", "num_params"],
            row_settings=["var_names"],
            verbose=0,
            mode='eval'
        )
        train_stats = summary(
            model=model,
            input_size=inputs,
            depth=5,
            col_width=25,
            col_names=["kernel_size", "output_size", "num_params"],
            row_settings=["var_names"],
            verbose=1,
            mode='train'
        )

        with (logdir / 'model.log').open('w') as f:
            f.write(str(model_stats))

        model_logbook['training_batch_size'] = batch_size
        model_logbook['input_bytes'] = model_stats.total_input
        model_logbook['total_params'] = model_stats.total_params
        model_logbook['trainable_params'] = model_stats.trainable_params
        model_logbook['param_bytes'] = model_stats.total_param_bytes

        model_logbook['eval_macs'] = model_stats.total_mult_adds
        model_logbook['training_macs'] = train_stats.total_mult_adds

        model_logbook['forward_pass_bytes'] = model_stats.total_output_bytes
        model_logbook['forward_backward_pass_bytes'] = train_stats.total_output_bytes

        model_logbook['eval_model_bytes'] = model_logbook['param_bytes'] \
            + model_logbook['forward_pass_bytes']
        model_logbook['training_model_bytes'] = model_logbook['param_bytes'] \
            + model_logbook['forward_backward_pass_bytes']

        model_logbook['eval_bytes'] = model_logbook['input_bytes'] + \
            model_logbook['eval_model_bytes']
        model_logbook['training_bytes'] = model_logbook['input_bytes'] + \
            model_logbook['training_model_bytes']

        model_logbook['layers'] = {}
        for layer in train_stats.summary_list:
            if layer.is_leaf_layer:
                model_logbook['layers'][f'{layer.class_name}_{layer.var_name}'] = {
                    'macs': layer.macs,
                    'params': max(layer.num_params, 0),
                    'param_bytes': layer.param_bytes,
                    'forward_pass_bytes': layer.output_bytes,
                    'forward_backward_pass_bytes': layer.output_bytes * 2, # x2 for gradients
                    'output_shape': layer.output_size,
                }

        with (logdir / 'model_logbook.json').open('w') as f:
            ujson.dump(
                model_logbook,
                f,
                indent=4,
                sort_keys=False,
                ensure_ascii=False,
                escape_forward_slashes=False
            )


class BestMetricSaver(HookBase):
    def __init__(self, 
                 metric_name: str, 
                 compare_fn: Literal["gt", "lt"] = "lt",
                 eval_after_validation: bool = True, 
                 period: int = 1
    ):
        self.metric_name = metric_name
        self.compare_fn = operator.gt if compare_fn == "gt" else operator.lt
        
        self.eval_after_validation = eval_after_validation
        self.period = period

    def _update_best_metrics(self, val):
        if math.isnan(val) or math.isinf(val):
            return False
        self.trainer.best_metric = val
        return True

    def update_best_metrics(self, latest_metric_val):
        if self.compare_fn(latest_metric_val, self.trainer.best_metric):
            self._update_best_metrics(latest_metric_val)

    def after_validation(self):
        if self.eval_after_validation:
            epoch_scalars = self.trainer.event_recorder.get_epoch_scalars()
            if self.metric_name not in epoch_scalars:
                raise ValueError(
                    f"Metric {self.metric_name} not found in epoch logs. "
                    "Make sure to set `val_metric` in the trainer config."
                )
            latest_metric_val, *_ = epoch_scalars[self.metric_name].pop()
            self.update_best_metrics(latest_metric_val)
    
    def after_epoch(self):
        """
        Check if the latest metric is the best so far.
        """
        # should match period of validation loop
        if (self.trainer._epoch + 1) % self.period == 0:
            if not self.eval_after_validation:
                epoch_scalars = self.trainer.event_recorder.get_epoch_scalars()
                if self.metric_name not in epoch_scalars:
                    raise ValueError(
                        f"Metric {self.metric_name} not found in epoch logs. "
                        "Make sure to set `val_metric` in the trainer config."
                    )
                latest_metric_val, *_ = epoch_scalars[self.metric_name].pop()
                self.update_best_metrics(latest_metric_val)

    def after_test(self):
        test_scalars = self.trainer.event_recorder.get_epoch_scalars()
        if self.metric_name not in test_scalars:
            raise ValueError(
                f"Metric {self.metric_name} not found in test logs. "
            )
        test_metric_val, *_ = test_scalars[self.metric_name].pop()
        self._update_best_metrics(test_metric_val)

# TODO: (-1) Torch Profiler Hook (almost done)
#       (0) Early Stop Hook
#       (1) save inference result hook (viz. segmentations, detections, etc.)
#       (2) activations, weights, gradients histogram hook (for train debug)