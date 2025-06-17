import torch
import pytest

from cell_observatory_finetune.data.structures.data_objects.image_list import ImageList


def rand_img(shape, device="cpu", val=1.0):
    """Return a tensor filled with a constant so we can spot padding."""
    return torch.full(shape, val, device=device)


def test_basic_single():
    img = rand_img((4, 5, 6))                # (D,H,W)
    lst = ImageList.from_tensors([img])

    assert lst.tensor.shape == (1, 4, 5, 6)
    assert lst.image_sizes == [(4, 5, 6)]
    assert len(lst) == 1

    out = lst[0]
    assert torch.equal(out, img)             # no padding on single image

    assert lst.device.type == "cpu"


def test_multi_padding():
    a = rand_img((2, 3, 4), val=5)           # (D=2,H=3,W=4)
    b = rand_img((2, 5, 2), val=7)           # different H/W
    pad_val = -1.0

    lst = ImageList.from_tensors([a, b], pad_value=pad_val)
    # Expected unified shape: D=2, H=5, W=4
    assert lst.tensor.shape == (2, 2, 5, 4)

    # check padded region
    # for img 0: rows 3-4 padded in H
    assert torch.all(lst.tensor[0, :, 3:, :].eq(pad_val))
    # for img 1: cols 2-3 padded in W
    assert torch.all(lst.tensor[1, :, :, 2:].eq(pad_val))

    # __getitem__ returns crop of original size
    assert lst[0].shape == (2, 3, 4)
    assert lst[1].shape == (2, 5, 2)

    # original pixels preserved
    assert torch.equal(lst[0], a)
    assert torch.equal(lst[1], b)


@pytest.mark.parametrize("div", [2, 8])
def test_size_divisibility(div):
    img = rand_img((5, 6, 7))
    lst = ImageList.from_tensors([img], size_divisibility=div)
    # shape must be largest multiple of div >= original
    D, H, W = lst.tensor.shape[-3:]
    assert D % div == 0 and H % div == 0 and W % div == 0
    assert D >= 5 and H >= 6 and W >= 7


def test_square_padding():
    a = rand_img((4, 3, 6))
    lst = ImageList.from_tensors(
        [a],
        padding_constraints={"square_size": 8, "size_divisibility": 4},
    )
    assert lst.tensor.shape[-3:] == (8, 8, 8)   # forced square
    # crop returns original
    assert torch.equal(lst[0], a)


def test_to_device_roundtrip():
    img = rand_img((2, 2, 2))
    lst = ImageList.from_tensors([img])
    if torch.cuda.is_available():
        cuda_lst = lst.to("cuda")
        assert cuda_lst.device.type == "cuda"
        assert torch.equal(cuda_lst[0].cpu(), img)   # data unchanged