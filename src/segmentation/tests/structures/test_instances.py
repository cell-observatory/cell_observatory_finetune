import pytest

import torch
import numpy as np

from segmentation.structures.data_objects.boxes import Boxes
from segmentation.structures.data_objects.masks import BitMasks
from segmentation.structures.sample_objects.instances import Instances


@pytest.fixture
def inst() -> Instances:
    N = 4
    
    boxes   = torch.arange(N * 6, dtype=torch.float32).view(N, 6)
    masks   = torch.rand((N, 6, 6, 6), dtype=torch.float32) < 0.5
    labels  = torch.tensor([1, 0, 2, 2], dtype=torch.int64)
    names   = ["a", "b", "c", "d"]

    boxes   = Boxes(boxes)
    masks   = BitMasks(masks)

    meta = dict(ids=[42,45, 43, 44], img_id=1)
    ins  = Instances(metainfo=meta)
    
    ins.boxes  = boxes
    ins.labels = labels
    ins.masks  = masks
    ins.names  = names

    return ins


def test_len_and_fields(inst):
    assert len(inst) == 4
    assert set(inst.keys()) == {"boxes", "labels", "names", "masks"}
    assert set(inst.metainfo_keys()) == {"ids", "img_id"}


def test_length_mismatch_raises():
    ins = Instances()
    ins.vec = torch.zeros(3)
    with pytest.raises(AssertionError):
        ins.vec2 = torch.zeros(2)        # mismatching length


def test_index_int(inst):
    single = inst[1]
    assert isinstance(single, Instances)
    assert len(single) == 1
    assert torch.allclose(single.boxes.tensor, inst.boxes[1].tensor)
    assert single.names[0] == "b"


def test_index_slice(inst):
    sl = inst[1:3]
    assert len(sl) == 2
    assert torch.equal(sl.labels, inst.labels[1:3])


def test_index_str(inst):
    # returns raw field if key is str
    names = inst["names"]
    assert names == ["a", "b", "c", "d"]


def test_index_bool_tensor(inst):
    sel = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
    filt = inst[sel]
    assert len(filt) == 2
    assert torch.equal(filt.labels, inst.labels[[0, 2]])


def test_index_numpy_array(inst):
    filt = inst[np.array([3, 1])]
    assert filt.names == ["d", "b"]


def test_cat(inst):
    a, b = inst[:2], inst[2:]
    merged = Instances.cat([a, b])
    assert len(merged) == len(inst)
    # all fields concatenated in the right order
    assert torch.equal(merged.boxes.tensor, inst.boxes.tensor)
    assert merged.names == inst.names


def test_cat_key_mismatch():
    x = Instances()
    x.vec = torch.zeros(1)
    y = Instances()
    y.vec = torch.zeros(1)
    y.extra = torch.zeros(1)
    with pytest.raises(AssertionError):
        Instances.cat([x, y])