from typing import Optional, Union, Callable    

import torch.nn as nn
import torch.nn.functional as F


def get_activation(activation: Optional[Union[str, Callable]], partial_init: bool = False):
    """
    Returns an activation layer (instantiated) or a constructor for later use.

    Args:
        activation (str or callable or None):
            - None or ""          -> no activation (returns None)  
            - String (case-insensitive):
                "RELU", "GELU", "SILU"/"SWISH", "LEAKYRELU", "TANH", "SIGMOID"
            - Callable             -> custom activation class / factory

        partial_init (bool):  
            - False (default) -> return an instance (nn.Module)  
            - True            -> return the constructor (callable) only

    Returns:
        nn.Module or callable or None
    """
    if activation in (None, ""):
        return None

    if isinstance(activation, str):
        act = activation.upper()
        mapping = {
            "RELU": nn.ReLU,
            "GELU": nn.GELU,
            "SILU": nn.SiLU,        
            "SWISH": nn.SiLU,
            "LEAKYRELU": nn.LeakyReLU,
            "TANH": nn.Tanh,
            "SIGMOID": nn.Sigmoid,
        }
        try:
            constructor = mapping[act]
        except KeyError as err:
            raise ValueError(f"Unknown activation string: {activation}") from err

    elif callable(activation):
        if isinstance(activation, nn.Module):
            return activation
        constructor = activation

    else:  
        raise TypeError(
            "`activation` must be None, str, nn.Module, or callable returning nn.Module"
        )

    return constructor if partial_init else constructor()