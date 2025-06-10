import torch
import pytest

from scrap.for_later.pixel_data import PixelData


def make_tensor(shape):
    """Return a Tensor filled with its linear indices."""
    return torch.arange(int(torch.prod(torch.tensor(shape))), dtype=torch.float32).reshape(shape)


def test_basic_shape():
    data = make_tensor((2, 3, 4, 5, 6))          # (T,C,D,H,W)
    pd   = PixelData()
    pd.raw = data

    assert pd.shape == (4, 5, 6)                 # last three dims


def test_multiple_fields_with_same_spatial_dims():
    pd = PixelData()
    pd.raw   = make_tensor((4, 5, 6))            # (D,H,W)
    pd.label = torch.zeros((1, 4, 5, 6))         # (C,D,H,W) – OK

    keys = ["raw", "label"]
    assert pd.keys()[0] in keys and pd.keys()[1] in keys


def test_spatial_mismatch_raises():
    pd = PixelData()
    pd.raw = torch.ones((4, 5, 6))

    with pytest.raises(AssertionError):
        pd.bad = torch.zeros((4, 5, 7))          # wrong width


@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4, 5, 6, 7)])
def test_invalid_ndim(shape):
    pd = PixelData()
    with pytest.raises(AssertionError):
        pd.img = torch.zeros(shape)


def test_tuple_slice_returns_new_pixeldata():
    data = make_tensor((2, 3, 4, 5, 6))
    pd   = PixelData()
    pd.img = data
    pd.set_metainfo(dict(name="foo"))                               # metainfo

    sub = pd[:, :, 1:3]                           # slice depth

    # new object
    assert isinstance(sub, PixelData) and sub is not pd
    # metainfo preserved
    assert sub.name == "foo"
    # shape updated: depth reduced from 4 -> 2
    assert sub.shape == (2, 5, 6)
    # pixel values match original slice
    assert torch.equal(sub.img, data[:, :, 1:3])


def test_invalid_index_type():
    pd = PixelData()
    pd.img = torch.zeros((4, 5, 6))
    with pytest.raises(TypeError):
        _ = pd[0]                                 # not a tuple
