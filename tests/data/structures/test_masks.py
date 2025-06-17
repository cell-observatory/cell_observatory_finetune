import numpy as np

import torch
import pytest

from finetune.data.structures.data_objects.boxes import Boxes
from finetune.data.structures.data_objects.masks import BitMasks


def make_mask(shape, coords):
    m = torch.zeros(shape, dtype=torch.bool)
    if coords:
        zs, ys, xs = zip(*coords)
        m[zs, ys, xs] = True
    return m


def test_from_numpy_and_tensor():
    npmask = np.random.rand(2, 4, 5, 6) > 0.5
    bm1 = BitMasks(npmask)                   
    bm2 = BitMasks(torch.from_numpy(npmask)) 
    assert bm1.tensor.dtype == torch.bool
    assert torch.equal(bm1.tensor, bm2.tensor)
    assert bm1.image_size == (4, 5, 6)
    assert len(bm1) == 2


def test_indexing_variants():
    masks = BitMasks(torch.rand(5, 2, 3, 3) > 0.5)

    single = masks[2]
    assert isinstance(single, BitMasks) and len(single) == 1
    assert torch.equal(single.tensor.squeeze(0), masks.tensor[2])

    sliced = masks[1:4]
    assert len(sliced) == 3
    assert torch.equal(sliced.tensor, masks.tensor[1:4])

    selector = torch.tensor([0, 1, 0, 1, 0], dtype=torch.bool)
    selected = masks[selector]
    assert len(selected) == 2
    assert torch.equal(selected.tensor[0], masks.tensor[1])
    assert torch.equal(selected.tensor[1], masks.tensor[3])


def test_nonempty():
    empty  = torch.zeros((1, 4, 4, 4), dtype=torch.bool)
    filled = make_mask((1, 4, 4, 4), [(0, 1, 1)])
    bm = BitMasks(torch.cat([empty, filled], dim=0))
    flags = bm.nonempty()
    assert torch.equal(flags, torch.tensor([False, True]))


@pytest.mark.parametrize("voxels, expected",
    [
        ([(0, 0, 0)],  [0, 0, 0, 1, 1, 1]),       
        ([(0, 1, 2), (2, 3, 4)], [2, 1, 0, 5, 4, 3]) 
    ]
)
def test_get_bounding_boxes(voxels, expected):
    m = make_mask((4, 5, 6), voxels).unsqueeze(0)     
    bm = BitMasks(m)
    box = bm.get_bounding_boxes()
    assert isinstance(box, Boxes)
    assert torch.equal(box.tensor[0], torch.tensor(expected, dtype=torch.float32))


def test_cat():
    a = BitMasks(torch.rand(2, 1, 3, 3) > 0.5)
    b = BitMasks(torch.rand(3, 1, 3, 3) > 0.5)
    cat = BitMasks.cat([a, b])
    assert len(cat) == 5
    assert torch.equal(cat.tensor[:2], a.tensor)
    assert torch.equal(cat.tensor[2:], b.tensor)


def test_to_device():
    masks = BitMasks(torch.rand(1, 2, 2, 2) > 0.5)
    if torch.cuda.is_available():
        cuda = masks.to("cuda")
        assert cuda.device.type == "cuda"
        assert torch.equal(cuda.tensor.cpu(), masks.tensor)
    cpu = masks.to("cpu")
    assert cpu.device.type == "cpu"