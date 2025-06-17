"""
https://github.com/facebookresearch/detectron2/blob/65184fc057d4fab080a98564f6b60fae0b94edc4/detectron2/utils/events.py#L28

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
import math
import json
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Literal, Optional, Tuple, Dict, List

import atexit
import concurrent.futures
from typing import Protocol

import wandb
import pandas as pd

import torch
from torch.utils.tensorboard import SummaryWriter

from ray.train import Checkpoint, report

from cell_observatory_finetune.utils.comm import (is_main_process, 
                                 in_torch_dist, 
                                 process_rank, 
                                 get_world_size,
                                 barrier)
from cell_observatory_finetune.utils.visualization import Visualizer


class EventRecorder:
    def __init__(self):
        self._iter, self._epoch, self._val_iter = 0, 0, 0
        
        self._step_scalars : dict[str, list[tuple[float, int, int]]] = defaultdict(list)
        self._epoch_scalars : dict[str, list[tuple[float, int, int]]] = defaultdict(list)

        self._tensors, self._histograms, self._traces = [], [], []
        
        self._reduce_methods_rank: dict[str, str | None] = {}
        self._reduce_methods_step: dict[str, str | None] = {}

    def put_tensor(self, tensor_name, tensor, tensor_metadata):
        self._tensors.append((tensor_name, tensor, tensor_metadata, self._iter, self._epoch))

    def put_scalar(self, 
                    name, 
                    value, 
                    scope: Literal["step", "epoch"] = "step", 
                    reduce_rank: str | None = "mean",
                    reduce_step: str | None = "mean"
    ):
        # we need to reduce per rank and per step to get epoch averages
        # either we set this dynamically or we have a config with 
        # the reduce methods for each scalar 
        if name not in self._reduce_methods_step or \
            name not in self._reduce_methods_rank:
            self._reduce_methods_step[name] = reduce_step
            self._reduce_methods_rank[name] = reduce_rank
        if scope == "step":
            self._step_scalars[name].append((value, self._iter, self._epoch))
        elif scope == "epoch":
            self._epoch_scalars[name].append((value, self._iter, self._epoch))

    def put_scalars(self, 
                    scope="step", 
                    reduce_rank="mean", 
                    reduce_step="mean", 
                    prefix=None, 
                    **kwargs
    ):
        for k, v in kwargs.items():
            assert isinstance(v, (int, float)), \
                f"Scalar value must be an int or float, got {type(v)} for key '{k}'"
            if not math.isfinite(v):
                raise ValueError(f"Scalar value for key '{k}' is not finite: {v}")
            k = f"{prefix}{k}" if prefix else k
            self.put_scalar(k, v, scope=scope, 
                    reduce_rank=reduce_rank, reduce_step=reduce_step)

    def put_histogram(self, hist_name, hist_tensor):
        self._histograms.append((hist_name, hist_tensor, self._iter, self._epoch))

    def put_trace(self, trace_name: str, trace_path: str):
        """
        Record a trace for later retrieval.
        
        Args:
            trace_name (str): The name of the trace.
            trace_path (str): The path to the trace file.
        """
        self._traces.append((trace_name, trace_path))

    def get_tensors(self):
        """
        Get the list of tensors recorded so far.
        Returns:
            List[Tuple[str, torch.Tensor, int]]: A list of tuples containing tensor name,
            tensor value, and the iteration number.
        """
        return self._tensors

    def get_step_scalars(self):
        """
        Get the dictionary of scalars recorded so far.
        Returns:
            Dict[str, List[Tuple[float, int]]]: A dictionary where keys are scalar names
            and values are lists of tuples containing scalar value and iteration number.
        """
        return self._step_scalars

    def get_epoch_scalars(self):
        """
        Get the dictionary of epoch scalars recorded so far.
        Returns:
            Dict[str, List[Tuple[float, int]]]: A dictionary where keys are scalar names
            and values are lists of tuples containing scalar value and epoch number.
        """
        return self._epoch_scalars
    
    def get_histograms(self):
        """
        Get the list of histograms recorded so far.
        Returns:
            List[Dict]: A list of dictionaries containing histogram parameters.
        """
        return self._histograms
    
    def get_traces(self) -> Tuple[str, str]:
        """
        Get the trace name and path recorded so far.
        Returns:
            Tuple[str, str]: A tuple containing the trace name and path.
        """
        if not self._traces:
            return None, None
        return self._traces[-1]

    def clear_tensors(self):
        self._tensors = []

    def clear_histograms(self):
        self._histograms = []

    def clear_scalars(self):
        for k, v in self._step_scalars.items():
            v.clear()
        
        for k, v in self._epoch_scalars.items():
            v.clear()

    def clear_traces(self):
        """
        Clear the recorded traces.
        """
        self._traces = []
    
    def clear(self):
        """
        Clear all recorded events.
        This method is typically called after writing events to a writer.
        """
        self.clear_tensors()
        self.clear_histograms()
        self.clear_scalars()
        self.clear_traces()

    def get_reduce_op(self, name, scope: Literal["step", "rank"]):
        if scope == "step":
            return self._reduce_methods_step.get(name)
        elif scope == "rank":
            return self._reduce_methods_rank.get(name)
        else:
            raise ValueError(f"Unknown scope: {scope!r}. "
                             f"Supported scopes: 'step', 'rank'.")
    
    def resume(self, iter: int, epoch: int):
        """
        Resume the recorder with the given iteration and epoch.
        This is useful for resuming training from a checkpoint.
        
        Args:
            iter (int): The iteration number to resume from.
            epoch (int): The epoch number to resume from.
        """
        self._iter = iter
        self._epoch = epoch
        self._val_iter = 0


class EventWriter:
    """
    Base class for writers that obtain events from :class:`EventRecorder` and process them.
    """

    @abstractmethod
    def write_tensor(self):
        pass
    
    # writer_scalars handles the writing of scalars to 
    # the desired backend (e.g., TensorBoard, W&B, etc.)
    # since each worker process has its own EventRecorder,
    # with its own sclars, we need to gather all scalars
    # from all workers and then write them in a single place
    # otherwise the workers end up overwriting each other's 
    # data, if we want to record data from all workers
    # we rename the scalars to include the rank
    # e.g. "rank0_loss", "rank1_loss", etc.
    # NOTE: Ray.Report reports metrics from the rank 0
    #       worker so for multi worker training
    #       we use distributed primitives to gather
    #       scalars from all workers and then write them
    #       to the desired backend. See:
    # https://docs.ray.io/en/latest/train/user-guides/monitoring-logging.html
    def reduce_scalars(self):
        distributed = in_torch_dist()
        world = get_world_size()
        rank = process_rank()

        step_scalars_gathered = self._gather_scalars(
            scalars=self.event_recorder.get_step_scalars(),
            rank=rank, 
            world=world, 
            distributed=distributed
        )
        epoch_scalars_gathered = self._gather_scalars(
            scalars=self.event_recorder.get_epoch_scalars(),
            rank=rank, 
            world=world, 
            distributed=distributed
        )

        if rank == 0:
            # reduce step scalars and add to epoch scalars
            epoch_scalars_gathered = self._reduce_step_scalars(
                epoch_scalars = epoch_scalars_gathered,
                step_scalars = step_scalars_gathered
            )

        return step_scalars_gathered, epoch_scalars_gathered
        #     # hand off to concrete writer
        #     self._write_scalar_impl(step_scalars_gathered, scope="step")
        #     self._write_scalar_impl(epoch_scalars_gathered, scope="epoch")
        
        # barrier()
    
    def _reduce_step_scalars(self, 
                             epoch_scalars: Dict[str, List[Tuple[float, int, int]]],
                             step_scalars: Dict[str, List[Tuple[float, int, int]]]
    ) -> Dict[str, List[Tuple[float, int, int]]]:
        for metric, records in step_scalars.items():
            if metric not in epoch_scalars:
                epoch_scalars[metric] = []
                reduce_method = self.event_recorder.get_reduce_op(metric, scope="step")
                vals   = [v for v, *_ in records]
                # all records have the same epoch
                # since this function is called
                # after each epoch to reduce all
                # step scalars for this epoch
                epoch = records[0][2]
                if reduce_method == "mean":
                    val = sum(vals) / len(vals)
                elif reduce_method == "sum":
                    val = sum(vals)
                elif reduce_method == "max":
                    val = max(vals)
                elif reduce_method == "min":
                    val = min(vals)
                else:
                    raise ValueError(f"Unknown reduce method: {reduce_method!r} \
                            for metric {metric!r}")
                epoch_scalars[metric].append((val, -1, epoch))
        return epoch_scalars

    def _gather_scalars(self, 
                        scalars: Dict[str, List[Tuple[float, int, int]]], 
                        rank: int, world: int, 
                        distributed: bool = True
    ):
        if distributed and world > 1:
            gathered = [None] * world
            torch.distributed.all_gather_object(gathered, scalars)
        else:
            gathered = [scalars]

        if rank == 0:
            # {metric: {(it,ep): [vals,...]}}
            buckets = defaultdict(lambda: defaultdict(list))
            
            # {metric: {(it, ep): [val_rank0, ...]}}
            for rank, dict in enumerate(gathered):
                for name, records in dict.items():
                    reduce_op = self.event_recorder.get_reduce_op(name, scope="rank")
                    for val, it, ep in records:
                        if reduce_op is None:
                            step_reduce_op = self.event_recorder.get_reduce_op(name, scope="step")
                            self.event_recorder._reduce_methods_step.setdefault(f"rank{rank}_{name}", step_reduce_op)
                            buckets[f"rank{rank}_{name}"][(it, ep)].append(val)
                        else:
                            buckets[name][(it, ep)].append(val)

            # apply reductions
            merged = defaultdict(list)
            for metric, rows in buckets.items():
                for (it, ep), vals in rows.items():
                    if metric.startswith("rank"):
                        v = vals[0]
                    else:
                        red = self.event_recorder.get_reduce_op(metric, scope="rank")
                        if red == "sum":
                            v = sum(vals)
                        elif red == "mean":
                            v = sum(vals) / len(vals)
                        elif red == "max":
                            v = max(vals)
                        elif red == "min":
                            v = min(vals)
                        else:
                            raise ValueError(f"Unknown reduce {red!r}")

                    if math.isfinite(v) and not math.isnan(v):
                        merged[metric].append((v, it, ep))
                    else:
                        raise ValueError(f"Invalid {metric}: {v} @ iter={it}, epoch={ep}")
            return merged
        else:
            # other ranks return empty dict
            return {}

    def _write_scalar_impl(self, 
                            scalar_dict: Dict[str, List[Tuple[float, int, int]]], 
                            scope: Literal["step", "epoch"] = "step"
    ):
        """
        Override in concrete writer (TB, W&B, Local, ...).
        """
        raise NotImplementedError

    @abstractmethod
    def write_histograms(self):
        pass

    @abstractmethod
    def write_traces(self):
        pass

    def write(self):
        """
        Write all events to the writer.
        This method should be called at the end of each iteration or epoch.
        """
        self.write_tensor()
        self.write_scalars()
        self.write_histograms()
        self.write_traces()
    
    @abstractmethod
    def close(self):
        pass

    def _make_step_table(self, scalar_dict):
        rows = {}
        for metric, data in scalar_dict.items():
            for val, itr, ep in data:
                row = rows.setdefault((itr, ep), {"iter": itr, "epoch": ep})
                row[metric] = val

        if not rows:
            return 

        df = (
            pd.DataFrame.from_records(list(rows.values()))
            .sort_values(["epoch", "iter"])
            .reset_index(drop=True)
        )
        return df

    def _make_epoch_table(self, scalar_dict):
        rows = {}
        for metric, data in scalar_dict.items():
            for val, _, ep in data:
                row = rows.setdefault((ep), {"epoch": ep})
                row[metric] = val

        if not rows:
            return 

        df = (
            pd.DataFrame.from_records(list(rows.values()))
            .sort_values(["epoch"])
            .reset_index(drop=True)
        )
        return df


class LocalEventWriter(EventWriter):
    """
    A local event writer that writes events to disk.
    """
    def __init__(self, 
                 event_recorder: EventRecorder, 
                 visualizer: Optional[Visualizer],
                 save_dir: str | Path, 
                 step_scalars_prefix: str,
                 epoch_scalars_prefix: str,
                 tensors_prefix: str,
                 scalars_save_format: Literal["csv"] = "csv",
                 tensors_save_format: Literal["zarr", "tiff"] = "tiff"
    ):
        self.max_iter = 0
        self.visualizer = visualizer
        self.event_recorder = event_recorder

        self.step_scalars_prefix = step_scalars_prefix
        self.epoch_scalars_prefix = epoch_scalars_prefix
        self.tensors_prefix = tensors_prefix
        
        self.scalars_save_format = scalars_save_format
        self.tensors_save_format = tensors_save_format

        # tensors save dir: 
        # <save_dir>/tensors/{self.prefix}/{self.tensor_name}/{self.event_recorder._epoch}.{self.tensors_save_format}
        # scalars save dir: 
        # <save_dir>/scalars/{self.scalars_prefix}.json
        self.tensors_save_dir = Path(save_dir) / "tensors"
        self.step_scalars_savepath = Path(save_dir) / "scalars"/ \
                        f"{self.step_scalars_prefix}.{self.scalars_save_format}"
        self.epoch_scalars_savepath = Path(save_dir) / "scalars" / \
            f"{self.epoch_scalars_prefix}.{self.scalars_save_format}"
        
        os.makedirs(self.tensors_save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "scalars"), exist_ok=True)

    def write_tensor(self):
        assert self.visualizer is not None, \
            "Visualizer is not set. Cannot write tensors without a visualizer."
        if process_rank() == 0:
            for name, tensor, metadata, itr, epoch in self.event_recorder.get_tensors():
                dir_path = os.path.join(self.tensors_save_dir, self.tensors_prefix, name)
                os.makedirs(dir_path, exist_ok=True)
                file_path = os.path.join(dir_path, f"epoch_{epoch}.{self.tensors_save_format}")
                self.visualizer.visualize(task=name,
                                        tensor=tensor,
                                        metadata=metadata,
                                        save_path=file_path)
        
        barrier()

    def _write_scalar_impl(self, scalar_dict, scope: Literal["step", "epoch"] = "step"):
        if process_rank() == 0:
            if not scalar_dict:
                raise ValueError("No scalars to write. "
                                "Please ensure scalars are recorded before writing.")
            if self.scalars_save_format == "csv":
                if scope == "step":
                    df = self._make_step_table(scalar_dict)
                    savepath = self.step_scalars_savepath
                elif scope == "epoch":
                    df = self._make_epoch_table(scalar_dict)
                    savepath = self.epoch_scalars_savepath
                df.to_csv(savepath,
                    mode="a",
                    header=not savepath.exists(),
                    index=False
                )

            else:
                raise NotImplementedError(f"Unsupported scalars_save_format: "
                                        f"{self.scalars_save_format}. "
                                        f"Supported formats: 'csv'.")

        barrier()

    def close(self):
        """
        Close the event writer, if necessary.
        """
        pass
    
    def write_histograms(self):
        pass
    
    def write_traces(self):
        pass


class RayEventWriter(EventWriter):
    """
    A Ray event writer that writes events using Ray.train.report.
    """
    def __init__(self, 
                 checkpointdir: str | Path, 
                 event_recorder: EventRecorder,
    ):
        self.checkpoint_dir = Path(checkpointdir)
        self.event_recorder = event_recorder
    
    # TODO: investigate if Ray can handle logging to W&B and TensorBoard
    #       meaning we don't have to implement these writers ourselves
    #       there is some utility to the local writes since it allows
    #       us to log multiple times for different scenarios without
    #       having to figure out how to fit it all into a single
    #       Ray.train.report call
    def _write_scalar_impl(self, 
                           scalar_dict, 
                           scope: Literal["step", "epoch"] = "step"
    ):
        # NOTE: from Ray docs:
        # in order to ensure consistency, train.report() 
        # acts as a barrier and must be called on each worker
        # however only rank 0 report is used 
        # thus we call report on each worker but only the 
        # gathered scalars from rank 0 are used and only 
        # for rank 0 do we pass in the real path to the checkpoint
        # for other ranks we pass in None
        checkpoint = Checkpoint.from_directory(self.checkpoint_dir)  \
            if is_main_process() else None
        if scope == "epoch":
            report(metrics=scalar_dict, checkpoint=checkpoint)

    def write_tensor(self):
        pass

    def write_histograms(self):
        pass

    def write_traces(self):
        pass

    def close(self):
        """
        Close the Ray event writer.
        """
        # No specific close operation for RayEventWriter
        pass


class EventWriterList(EventWriter):
    def __init__(self, 
                 writers: List[EventWriter]
    ):
        self.writers = writers
        self.event_recorder = writers[0].event_recorder
        assert all(writer.event_recorder is self.event_recorder for writer in writers), \
            "All writers must share the same EventRecorder instance."

    def write(self):
        # write scalars
        step_scalars_gathered, epoch_scalars_gathered = self.reduce_scalars()
        for writer in self.writers:
            writer._write_scalar_impl(step_scalars_gathered, scope="step")
            writer._write_scalar_impl(epoch_scalars_gathered, scope="epoch")

        # TODO: write tensors, histograms, traces...

    def write_tensor(self):
        pass

    def write_scalars(self):
        pass

    def write_histograms(self):
        pass

    def write_traces(self):
        pass

    def close(self):
        for writer in self.writers:
            writer.close()