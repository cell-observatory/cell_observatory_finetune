import os
import warnings
from pathlib import Path 
from collections import OrderedDict

import torch 

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

_DTYPES = {
    "float32": torch.float32,
    "fp32":    torch.float32,
    "float16": torch.float16,
    "fp16":    torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16":     torch.bfloat16,
}


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

def load_checkpoint(model_engine, opt, config, logger, checkpointdir, ckpt_suffix="best", dtype=None):
    logger.info(f"Loading pretrained model @ resumed checkpoint -> {checkpointdir}")

    if dtype is not None:
        try:
            target_dtype = _DTYPES[dtype]
        except KeyError:
            raise ValueError(f"Unsupported dtype '{dtype}'. Valid: {list(_DTYPES)}")

    if getattr(config.deepspeed_config.zero_optimization, "stage", None) == 3:
        # load_checkpointreturns a tuple (load_dir, client_states) with load_dir
        # not being None if the checkpoint was loaded successfully
        load_path, _ = model_engine.load_checkpoint(checkpointdir)
        if load_path is None:
            raise RuntimeError(f"Failed to load DeepSpeed checkpoint from {checkpointdir}")
        if target_dtype is not None:
            model_engine.module.to(target_dtype)
        return 
    
    model_path = Path(os.path.join(checkpointdir, f"{ckpt_suffix}_model.bin"))
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model_state = torch.load(Path(checkpointdir) / f"{ckpt_suffix}_model.bin", map_location="cpu")
    model_state = _strip_prefix(model_state, prefix="module.")
    model_engine.load_state_dict(model_state)

    if target_dtype is not None:
        module = getattr(model_engine, "module", model_engine)
        module.to(target_dtype)
    
    if opt is not None:
        optimizer_path = Path(os.path.join(checkpointdir, f"{ckpt_suffix}_optimizer.bin"))
        if optimizer_path.exists():
            optimizer_state = torch.load(optimizer_path, map_location="cpu")
            opt.load_state_dict(optimizer_state)
        else:
            warnings.warn(FileNotFoundError(f"Optimizer file not found: {optimizer_path}"))

def convert_zero_checkpoint(checkpoint_path: str, output_dir: str, tag: str = "best_model", ckpt_prefix: str = "best"):
    state = get_fp32_state_dict_from_zero_checkpoint(checkpoint_path, tag)
    output_path = os.path.join(output_dir, f"{ckpt_prefix}_model.bin")
    torch.save(state, output_path)