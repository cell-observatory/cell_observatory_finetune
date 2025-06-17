import pytest

import torch

from finetune.data.structures.data_objects.boxes import Boxes
from finetune.data.structures.data_objects.masks import BitMasks

from finetune.data.structures.sample_objects.instances import Instances
from finetune.data.structures.sample_objects.data_sample_old import DataSample


def make_instances(N=3):
    inst = Instances()
    inst.boxes  = Boxes(torch.zeros(N, 6))
    inst.masks  = BitMasks(torch.zeros(N, 4, 5, 6))
    inst.labels = torch.arange(N)
    return inst


def test_set_wrong_type_raises(prop, bad_val):
    ds = DataSample()
    with pytest.raises(AssertionError):
        setattr(ds, prop, bad_val)