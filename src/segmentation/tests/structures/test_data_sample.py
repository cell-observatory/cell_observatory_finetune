import pytest

import torch

from segmentation.structures.data_objects.boxes import Boxes
from segmentation.structures.data_objects.masks import BitMasks

from segmentation.structures.sample_objects.instances import Instances
from segmentation.structures.sample_objects.pixel_data import PixelData
from segmentation.structures.sample_objects.data_sample_old import DataSample


def make_pixel(shape=(1, 3, 4, 5, 6)):
    pd = PixelData()
    pd.img = torch.zeros(shape)
    return pd


def make_instances(N=3):
    inst = Instances()
    inst.boxes  = Boxes(torch.zeros(N, 6))
    inst.masks  = BitMasks(torch.zeros(N, 4, 5, 6))
    inst.labels = torch.arange(N)
    return inst


def test_set_and_get_properties():
    ds = DataSample()

    in_px   = make_pixel()
    seg_px  = make_pixel((4, 5, 6))          # 3-D
    instances = make_instances()

    # setters should accept correct types
    ds.inputs       = in_px
    ds.gt_seg       = seg_px
    ds.gt_instances = instances

    # getters return identical objects
    assert ds.inputs is in_px
    assert ds.gt_seg is seg_px
    assert ds.gt_instances is instances


@pytest.mark.parametrize(
    "prop, bad_val",
    [
        ("inputs", torch.zeros(1, 3, 3)),        # not a PixelData
        ("gt_seg",  {"a": 1}),                   # arbitrary dict
        ("gt_instances", PixelData()),           # wrong class
    ],
)
def test_set_wrong_type_raises(prop, bad_val):
    ds = DataSample()
    with pytest.raises(AssertionError):
        setattr(ds, prop, bad_val)


def test_deletion_removes_field():
    ds = DataSample()
    ds.inputs = make_pixel()

    # property exists
    _ = ds.inputs
    del ds.inputs

    with pytest.raises(AttributeError):
        _ = ds.inputs

    # underlying private key removed from _data_fields
    assert "_inputs" not in ds._data_fields


def test_keys_after_setting_properties():
    ds = DataSample()
    ds.inputs       = make_pixel()
    ds.gt_seg       = make_pixel((4, 5, 6))
    ds.gt_instances = make_instances()

    # metainfo can still be added normally
    ds.set_metainfo({"fname": "img.tif"})
    assert ds.fname == "img.tif"
    assert "fname" in ds.metainfo_keys()