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
import itertools
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Literal, Optional, Tuple, Dict, List

import wandb
import pandas as pd

import torch

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
        
        self._reduce_methods: dict[str, str | None] = {}

    def put_tensor(self, tensor_name, tensor, tensor_metadata):
        self._tensors.append((tensor_name, tensor, tensor_metadata, self._iter, self._epoch))

    def put_scalar(self, 
                    name, 
                    value, 
                    scope: Literal["step", "epoch"] = "step", 
                    reduce_method: str | None = "mean"
    ):
        # we need to reduce per rank and per step to get epoch averages
        # either we set this dynamically or we have a config with 
        # the reduce methods for each scalar 
        if name not in self._reduce_methods:
            self._reduce_methods[name] = reduce_method
        if scope == "step":
            self._step_scalars[name].append((value, self._iter, self._epoch))
        elif scope == "epoch":
            self._epoch_scalars[name].append((value, self._iter, self._epoch))

    def put_scalars(self, 
                    scope="step", 
                    reduce_method="mean", 
                    prefix=None, 
                    **kwargs
    ):
        for k, v in kwargs.items():
            assert isinstance(v, (int, float)), \
                f"Scalar value must be an int or float, got {type(v)} for key '{k}'"
            if not math.isfinite(v):
                raise ValueError(f"Scalar value for key '{k}' is not finite: {v}")
            k = f"{prefix}{k}" if prefix else k
            self.put_scalar(k, v, scope=scope, reduce_method=reduce_method)

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

    def get_reduce_op(self, name):
            return self._reduce_methods.get(name)
    
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
    
    # helper methods for writing scalars

    # writer_scalars handles the writing of scalars to 
    # the desired backend (e.g., TensorBoard, W&B, etc.)
    # since each worker process has its own EventRecorder,
    # with its own sclars, we need to gather all scalars
    # from all workers and then write them in a single place
    def reduce_scalars(self):
        distributed = in_torch_dist()
        world = get_world_size()
        rank = process_rank()

        step_scalars_per_epoch, step_scalars = self._gather_scalars(
            scalars=self.event_recorder.get_step_scalars(),
            rank=rank, 
            world=world, 
            distributed=distributed,
            keep_steps_data=True
        )
        epoch_scalars, _ = self._gather_scalars(
            scalars=self.event_recorder.get_epoch_scalars(),
            rank=rank, 
            world=world, 
            distributed=distributed,
            keep_steps_data=False
        )

        if rank == 0:
            # reduce step scalars and add to epoch scalars
            epoch_scalars.update(step_scalars_per_epoch)  

        return step_scalars, epoch_scalars

    def _gather_scalars(self, 
                        scalars: Dict[str, List[Tuple[float, int, int]]], 
                        rank: int, 
                        world: int, 
                        distributed: bool = True,
                        keep_steps_data: bool = False
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
                    for val, it, ep in records:
                        buckets[name][(it, ep)].append(val)

            # apply reductions
            merged, merged_per_step = defaultdict(list), defaultdict(list)
            for metric, rows in buckets.items():
                reduce_op = self.event_recorder.get_reduce_op(metric)
                # vals_per_rank = [[val_rank0_iter0, ...], [val_rank0_iter1, ...] ...]
                vals_per_rank = [v for _, v in rows.items()]
                vals = list(itertools.chain.from_iterable(vals_per_rank))
                v = self._reduce(reduce_op, vals)
                merged[metric].append((v, it, ep))
                
                if keep_steps_data:
                    vals_per_step = [self._reduce(reduce_op, vals_rank) for vals_rank in vals_per_rank]
                    merged_per_step[metric] = [
                        (val, it, ep) for val, (it, ep) in zip(vals_per_step, rows.keys())
                    ]

            return merged, merged_per_step
        else:
            # other ranks return empty dict
            return {}, {}
        
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
        
    def _reduce(self, reduce_method: str, values: List[float]) -> float:
        """
        Reduce values based on the specified method.
        """
        if reduce_method == "sum":
            return sum(values)
        elif reduce_method == "mean":
            return sum(values) / len(values)
        elif reduce_method == "max":
            return max(values)
        elif reduce_method == "min":
            return min(values)
        else:
            raise ValueError(f"Unknown reduce method: {reduce_method!r}")

    @abstractmethod
    def _write_scalar_impl(self, 
                            scalar_dict: Dict[str, List[Tuple[float, int, int]]], 
                            scope: Literal["step", "epoch"] = "step"
    ):
        pass

    @abstractmethod
    def _write_tensor_impl(self):
        pass

    @abstractmethod
    def _write_histograms_impl(self):
        pass

    @abstractmethod
    def _write_traces_impl(self):
        pass

    @abstractmethod
    def close(self):
        pass


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

    def _write_tensor_impl(self):
        pass

    def _write_histograms_impl(self):
        pass
    
    def _write_traces_impl(self):
        pass

    def close(self):
        pass


class WandBEventWriter(EventWriter):
    def __init__(self, 
                 event_recorder: EventRecorder, 
                 run_config: dict,
                 project: str,
                 dir: str | Path,
                 scalar_keys: List[str],
                 entity: str | None = None,
                 name: str | None = None,
                 tags: List[str] | None = None,
                 resume_from: str | None = None,
                 id: str | None = None,  
                 notes: str | None = None, 
                 force: bool = True
    ):
        wandb.login()

        self.event_recorder = event_recorder
        self.run = wandb.init(project=project,
                                entity=entity,
                                # config=run_config,
                                dir=dir,
                                name=name,
                                tags=tags,
                                resume=resume_from,
                                id=id,
                                notes=notes,
                                force=force)
        
        self.scalar_keys = scalar_keys
        self.step_table = wandb.Table(columns=["iter", "epoch", *scalar_keys],  log_mode="INCREMENTAL")
        self.epoch_table = wandb.Table(columns=["iter", *scalar_keys], log_mode="INCREMENTAL")
        self.run.log({"step_logbook":  self.step_table,
                      "epoch_logbook": self.epoch_table})
        
    def _write_scalar_impl(self, 
                           scalar_dict, 
                           scope: Literal["step", "epoch"] = "step"
    ):
        if process_rank() == 0:
            if not scalar_dict:
                raise ValueError("No scalars to write. "
                                "Please ensure scalars are recorded before writing.")
            if scope == "step":
                df = self._make_step_table(scalar_dict)                    
                
                for rec in df.to_dict(orient="records"):
                    vals = [rec["iter"], rec["epoch"]] + [rec[k] for k in self.scalar_keys]
                    self.step_table.add_data(*vals)
                    self.run.log(rec, step=rec["iter"], commit=False)
                
                self.run.log({}, commit=True)  # commits the batch of logs
                self.run.log({"step_logbook": self.step_table})

            elif scope == "epoch":
                df = self._make_epoch_table(scalar_dict)            
                for rec in df.to_dict(orient="records"):
                    vals = [rec["epoch"]] + [rec[k] for k in self.scalar_keys]
                    self.epoch_table.add_data(*vals)
                
                self.run.log({"epoch_logbook": self.epoch_table})

    def _write_histograms_impl(self):
        pass

    def write_traces_impl(self):
        pass

    def _write_tensor_impl(self):
        pass

    def close(self):
        if self.run is not None:
            self.run.finish()


class EventWriterList(EventWriter):
    def __init__(self, 
                 writers: List[EventWriter]
    ):
        self.writers = writers
        self.event_recorder = writers[0].event_recorder
        assert all(writer.event_recorder is self.event_recorder for writer in writers), \
            "All writers must share the same EventRecorder instance."

    def write(self):
        self.write_scalars()
        self.write_tensor()
        self.write_histograms()
        self.write_traces()

    def write_scalars(self):
        step_scalars_gathered, epoch_scalars_gathered = self.reduce_scalars()
        for writer in self.writers:
            writer._write_scalar_impl(step_scalars_gathered, scope="step")
            writer._write_scalar_impl(epoch_scalars_gathered, scope="epoch")

    def write_tensor(self):
        pass

    def write_histograms(self):
        pass

    def write_traces(self):
        pass

    def close(self):
        for writer in self.writers:
            writer.close()