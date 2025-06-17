from enum import Enum
from typing import Dict, Union

import torch

from cell_observatory_finetune.data.structures.sample_objects.data_sample import DataSample
from cell_observatory_finetune.data.structures.sample_objects.base_data_element import BaseDataElement

class SampleObjectType(Enum):
    DATA_SAMPLE = DataSample
    
class PreProcessor(torch.nn.Module):
    def __init__(self, sample_object: Union[BaseDataElement, str] = DataSample):
        super().__init__()
        
        if isinstance(sample_object, str):
            sample_object = SampleObjectType[sample_object.upper()].value
        elif not issubclass(sample_object, BaseDataElement):
            raise TypeError(f"sample_object must be a subclass of BaseDataElement, got {type(sample_object)}")
        self.sample_object = sample_object

    def forward(self, data_sample: Dict=None):
        return self.sample_object.from_dict(data_sample or {})