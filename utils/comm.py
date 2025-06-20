"""
https://github.com/facebookresearch/detectron2/blob/400a49c1ec11a18dd25aea3910507bc3bcd15794/detectron2/evaluation/evaluator.py#L224

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


from enum import Enum
from typing import Literal, Optional

from ray.train import get_context
from contextlib import contextmanager

import torch
from torch import distributed as dist 


class OpMap(Enum):
    """
    Map of supported reduce operations.
    """
    SUM = dist.ReduceOp.SUM
    MAX = dist.ReduceOp.MAX
    MIN = dist.ReduceOp.MIN
    MEAN = dist.ReduceOp.SUM  # Use SUM for mean, divide by world size later


def in_torch_dist() -> bool:
    """True if torch.distributed is initialised."""
    return dist.is_available() and dist.is_initialized()


def process_rank() -> int:
    """
    Return the global rank, falling back to 0 for single-process runs.
    Works for:
      - Ray Train via `ray.train.get_context()`
      - Plain torchrun/DDP - via `torch.distributed`
      - Local debugging - rank 0
    """
    try:
        return get_context().get_world_rank()
    except RuntimeError:
        # can happen if Ray Train is not initialised
        # or if the context is not available
        pass
    if in_torch_dist():
        return dist.get_rank()
    else:
        return 0
    

def get_world_size() -> int:
    """
    Return the global world size, falling back to 1 for single-process runs.
    Works for:
      - Ray Train via `ray.train.get_context()`
      - Plain torchrun/DDP - via `torch.distributed`
      - Local debugging - world size 1
    """
    try:
        return get_context().get_world_size()
    except RuntimeError:
        pass
    if in_torch_dist():
        return dist.get_world_size()
    else:
        return 1


def is_main_process() -> bool:
    """True for rank-0 worker (or the only process)."""
    return process_rank() == 0


def barrier(device_ids: Optional[int] = None) -> None:
    """
    Global synchronisation:
      - Ray Train barrier when available
      - torch.distributed.barrier() NCCL backend if available
      - fallback to cpu/Gloo otherwise 
      - No-op in single-process mode
    """
    try:
        ctx = get_context()
        if hasattr(ctx, "barrier"):
            ctx.barrier()
            return
    except RuntimeError:
        pass
    if in_torch_dist():
        dist.barrier(device_ids=[device_ids]) if device_ids is not None else dist.barrier()
        return
    return


def gather_and_reduce(scalars: torch.Tensor, reduce_op: str = 'mean'):
    if not in_torch_dist():
        return scalars.clone()

    if reduce_op.upper() not in OpMap.__members__:
        raise ValueError(f"Unsupported op: {reduce_op}")

    dist.all_reduce(scalars, op=OpMap[reduce_op.upper()].value)
    if reduce_op == "mean":
        scalars /= dist.get_world_size()
    return scalars

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)