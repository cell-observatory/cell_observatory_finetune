import pytest
import numpy as np

import torch

from finetune.data.structures.sample_objects.base_data_element_old import BaseDataElement


@pytest.fixture
def sample() -> BaseDataElement:
    meta = dict(fname="img.tif", shape=(4, 5, 6))
    data = dict(mask=torch.arange(6).reshape(1, 2, 3), score=torch.tensor([0.3]))
    return BaseDataElement(metainfo=meta, **data)


def test_basic_fields(sample):
    # attribute & dict-like access
    assert sample.fname == "img.tif"
    assert "mask" in sample and "shape" in sample

    # get / pop
    assert sample.get("shape") == (4, 5, 6)
    popped = sample.pop("score")
    assert popped.item() - 0.3 < 1e-5
    assert "score" not in sample


def test_clone_and_new(sample):
    clone = sample.clone()
    assert clone is not sample
    assert clone.metainfo == sample.metainfo
    assert torch.equal(clone.mask, sample.mask)

    new = sample.new(metainfo=dict(fname="img.tif", shape=(1, 1, 1), extra="foo"))
    assert new.fname == "img.tif"          
    assert new.shape == (1, 1, 1)          
    assert new.extra == "foo"

    # update()
    other = BaseDataElement(metainfo=dict(id=9), label="cat")
    sample.update(other)
    assert sample.id == 9 and sample.label == "cat"


def test_tensor_helpers(sample):
    # prepare a tensor field for conversions
    sample.feat = torch.ones(2, 2, requires_grad=True)

    cpu = sample.cpu()
    assert cpu.feat.device.type == "cpu"

    if torch.cuda.is_available():
        gpu = sample.cuda()
        assert gpu.feat.device.type == "cuda"
        # round-trip
        back = gpu.cpu()
        assert torch.equal(back.feat, sample.feat)

    detached = sample.detach()
    assert detached.feat.requires_grad is False

    nped = sample.numpy()
    assert isinstance(nped.feat, np.ndarray) and nped.feat.shape == (2, 2)

    recon = nped.to_tensor()
    assert isinstance(recon.feat, torch.Tensor)
    assert torch.equal(torch.from_numpy(nped.feat), recon.feat)

    # generic .to()
    half = sample.to(dtype=torch.float16)
    assert half.feat.dtype == torch.float16


def test_delete_and_contains(sample):
    assert "mask" in sample
    del sample.mask
    assert "mask" not in sample
    with pytest.raises(AttributeError):
        _ = sample.mask