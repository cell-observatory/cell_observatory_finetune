import os
import re
import sys
import shlex
import subprocess
import logging
import warnings
from pathlib import Path 

from collections import OrderedDict
from typing import Dict, Optional, Union, Literal

import torch 

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

from cell_observatory_platform.data.data_types import TORCH_DTYPES
from cell_observatory_finetune.utils.comm import is_main_process, get_world_size, barrier


class CheckpointManager:
    def __init__(self,
                 model: torch.nn.Module, 
                 checkpointdir: Union[str, Path],
                 engine: Literal["deepspeed"] = "deepspeed", 
                 checkpoint_tag: str = "best_model",
                 load_dtype: Optional[Literal["fp16", "bf16"]] = None,
                 max_keep: Optional[int] = None
    ):
        self.model = model
        self.engine = engine
        self.checkpoint_tag = checkpoint_tag
        self.load_dtype = load_dtype

        self.checkpointdir = Path(checkpointdir)
        assert self.checkpointdir.is_dir(), f"Checkpoint \
            directory does not exist: {self.checkpointdir}"

        self.max_keep = max_keep

    def save(self, 
             prefix: str, 
             epoch: int = None,
             iter: int = None, 
             best_loss: Optional[float] = None,
    ):
        if self.engine == "deepspeed":
            client_state = {
                "epoch": epoch,
                "iter": iter,
                "best_loss": best_loss
            }
            self.model.save_checkpoint(self.checkpointdir, client_state=client_state, tag=prefix)
        else:
            raise NotImplementedError("Saving sharded checkpoints for " \
                "other engines not implemented yet.")

    def load(self):  
        world_size = get_world_size()
        num_ckpt_shards = self._infer_ckpt_shards_num(
            ckpt_dir=os.path.join(self.checkpointdir, self.checkpoint_tag),
            pattern=r"mp_rank_.+_model_states\.pt$$"
        )
        
        if num_ckpt_shards > 1 and num_ckpt_shards != world_size:
            if self.engine == "deepspeed":
                if is_main_process():
                    warnings.warn(
                        "Loading a checkpoint with DeepSpeed where number of processes. " \
                        "does not match the number of checkpoint shards. " \
                        "Converting to a standard checkpoint and saving to disk."
                    )
                    self._convert_zero_checkpoint_to_universal(
                        input_folder=os.path.join(self.checkpointdir, 
                        self.checkpoint_tag),
                        output_folder=os.path.join(self.checkpointdir, 
                                                   f"{self.checkpoint_tag}_universal"),
                    )
                
                barrier()
                
                ckpt_path, client_state = self.model.load_checkpoint(
                    load_dir=self.checkpointdir,
                    tag=f"{self.checkpoint_tag}_universal",
                    # custom_load_fn=state_dict_filter_fn
                )
            else:
                raise NotImplementedError("Loading checkpoints for " \
                    "other engines not implemented yet.")
        else:
            if self.engine == "deepspeed":
                ckpt_path, client_state = self.model.load_checkpoint(
                    load_dir=self.checkpointdir,
                    tag=self.checkpoint_tag,
                    # custom_load_fn=state_dict_filter_fn,
                )
            else:
                raise NotImplementedError("Loading checkpoints for " \
                    "other engines not implemented yet.")

        # get target dtype if specified
        if self.load_dtype is not None:
            module = getattr(self.model, "module", self.model)
            module.to(TORCH_DTYPES[self.load_dtype].value)

        return ckpt_path, client_state
    
    def _infer_ckpt_shards_num(self, 
                               ckpt_dir: Union[str, Path], 
                               pattern: str = r"mp_rank_.+_model_states\.pt$$",
    ) -> int:
        """
        Infer the number of checkpoint shards in a directory.
        """
        ckpt_dir = Path(ckpt_dir)
        if not ckpt_dir.is_dir():
            raise ValueError(f"Checkpoint directory does not exist: {ckpt_dir}")

        # find all files that match the pattern
        regex = re.compile(pattern)
        ckpt_files = [f for f in ckpt_dir.iterdir() if regex.match(f.name)]
        return len(ckpt_files)

    def _prefix_aware_load_state_dict(self, 
                                      state_dict: Dict[str, torch.Tensor], 
                                      model: torch.nn.Module
    ):
        ckpt_has_module   = any(k.startswith("module.") for k in state_dict)
        model_expects_mod = any(k.startswith("module.") for k in model.state_dict())

        if ckpt_has_module and not model_expects_mod:
            # Loading a DDP checkpoint into a non‑DDP model
            state_dict = self._strip_prefix(state_dict, prefix="module.")
        elif not ckpt_has_module and model_expects_mod:
            # Loading a non‑DDP checkpoint into a DDP‑wrapped model
            state_dict = self._add_prefix(state_dict, prefix="module.")
        
        model.load_state_dict(state_dict)

    @staticmethod
    def _add_prefix(state_dict: Dict[str, torch.Tensor],
                    prefix: str = "module."
    ) -> Dict[str, torch.Tensor]:
        """
        If none of the keys start with `prefix`, add it to every key.
        Otherwise return the dict unchanged.
        """
        return OrderedDict(
            (k if k.startswith(prefix) else f"{prefix}{k}", v)
            for k, v in state_dict.items()
        )

    @classmethod
    def _strip_prefix(state_dict: dict, prefix: str = "module.") -> dict:
        """
        If the keys in `state_dict` are all prefixed with `prefix`, remove it.
        Otherwise return the dict unchanged.
        """
        # do all keys start with the prefix?
        if all(key.startswith(prefix) for key in state_dict.keys()):
            new_state = OrderedDict()
            for key, value in state_dict.items():
                new_key = key[len(prefix):]
                new_state[new_key] = value
            return new_state
        else:
            return state_dict
        
    def _convert_zero_checkpoint_to_universal(self, input_folder, output_folder):
        """
        Convert a DeepSpeed Zero Stage 3 checkpoint to a standard checkpoint.
        This is useful for loading the checkpoint with training configs.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        cmd = [
            sys.executable, "-m", "deepspeed.checkpoint.ds_to_universal",
            "--input_folder", str(input_folder),
            "--output_folder", str(output_folder)
        ]
        subprocess.check_call(cmd)
        

# useful utility function to convert a deepspeed zero stage 3 checkpoint to a standard checkpoint
def convert_zero_checkpoint(checkpoint_path: str, output_dir: str, tag: str = "best_model", ckpt_prefix: str = "best"):
    state = get_fp32_state_dict_from_zero_checkpoint(checkpoint_path, tag)
    output_path = os.path.join(output_dir, f"{ckpt_prefix}_model.bin")
    torch.save(state, output_path)