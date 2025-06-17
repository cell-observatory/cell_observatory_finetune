import abc
from typing import Optional, Dict

import torch
from finetune.models.meta_arch.preprocessor import PreProcessor
from finetune.data.structures.sample_objects.base_data_element import BaseDataElement


class BaseModel(torch.nn.Module):
    """
    Base class for segmentation models.
    All segmentation models should subclass this class.
    """
    def __init__(self, preprocessor: Optional[PreProcessor] = None):
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

    # TODO: include more methods that should be implemented
    #       by subclasses (freeze, unfreeze, load weights, etc.)
    # @abc.abstractmethod
    # def init_weights(self):
    #     """Initialize the weights of the model."""
    #     pass