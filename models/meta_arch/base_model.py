import abc
from functools import wraps
from operator import attrgetter
from typing import Optional, Dict, Union, List

import torch
from torch.utils.checkpoint import checkpoint

from cell_observatory_finetune.models.meta_arch.preprocessor import PreProcessor
from cell_observatory_finetune.data.structures.sample_objects.base_data_element import BaseDataElement


class BaseModel(torch.nn.Module):
    """
    Base class for finetune models.
    All finetune models should subclass this class.
    """
    def __init__(self, preprocessor: PreProcessor):
        super(BaseModel, self).__init__()
        self.preprocessor = preprocessor 

    @abc.abstractmethod
    def _forward(self, data_sample: BaseDataElement):
        """
        Forward pass of the model.
        This method should be implemented by subclasses.
        """
        pass    

    def forward(self, data_sample: Dict = None):
        """
        Forward pass of the model.
        This method should be implemented by subclasses.
        """
        data_sample = self.preprocessor(data_sample)
        return self._forward(data_sample)

    def wrap_forward(self, forward):
        @wraps(forward)
        def wrapper(*args):
            return checkpoint(forward, *args)
        return wrapper

    # from: mmengine/runner/activation_checkpointing.py
    def activation_checkpoint(self, model: torch.nn.Module,
                                        modules: Union[List[str], str]):
        """""
        Wrap the forward method of the specified modules
        with activation checkpointing.
        """

        if isinstance(modules, str):
            modules = [modules]
        for module_name in modules:
            module = attrgetter(module_name)(model)
            module.forward = self.wrap_forward(module.forward)

    def freeze(self, modules: Union[str, List[str]]):
        """
        Freeze the parameters of the model.
        """
        if isinstance(modules, str):
            modules = [modules]
        for module_name in modules:
            module = attrgetter(module_name)(self)
            for param in module.parameters():
                param.requires_grad = False