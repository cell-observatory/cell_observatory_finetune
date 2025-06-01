import pytest

import torch

from segmentation.structures.data_objects.boxes import (
    Boxes,
    BoxMode3D,
)


@pytest.fixture
def boxes_xyzxyz() -> Boxes:
    # two 3-D boxes, format (x1,y1,z1,x2,y2,z2)
    data = torch.tensor(
        [
            [0.0,  0.0,  0.0, 10.0, 20.0, 30.0],      # vol = 10*20*30 = 6000
            [5.0, 10.0, 15.0,  9.0, 12.0, 18.0],      # vol = 4*2*3   =   24
        ]
    )
    return Boxes(data.clone())


def test_failed_ctor_shape():
    with pytest.raises(AssertionError):
        Boxes(torch.randn(5, 4))     # must be …x6


def test_volume(boxes_xyzxyz):
    vol = boxes_xyzxyz.volume()
    assert torch.allclose(vol, torch.tensor([6000.0, 24.0]))


def test_clip_in_place(boxes_xyzxyz):
    b = boxes_xyzxyz.clone()
    b.clip((16, 12, 8))                  # depth, H, W
    exp = torch.tensor(
        [[0, 0, 0, 8, 12, 16],           # clipped at depth=16, height=12, width=8
         [5,10,15, 8,12,16]]
    ).float()
    assert torch.allclose(b.tensor, exp)


def test_nonempty(boxes_xyzxyz):
    keep = boxes_xyzxyz.nonempty()
    assert keep.tolist() == [True, True]

    # shrink one side to zero width
    b = boxes_xyzxyz.clone()
    b.tensor[1, 3] = b.tensor[1, 0]      # x2 == x1
    keep = b.nonempty()
    assert keep.tolist() == [True, False]


def test_inside_box(boxes_xyzxyz):
    ref = (0, 0, 0, 40, 40, 40)                   # D,H,W
    inside = boxes_xyzxyz.inside_box(ref)
    assert inside.tolist() == [True, True]

    ref = (5, 5, 10, 10, 20, 30)
    inside = boxes_xyzxyz.inside_box(ref)
    assert inside.tolist() == [False, True]


def test_centers(boxes_xyzxyz):
    ctr = boxes_xyzxyz.get_centers()
    expected = torch.tensor([[5,10,15], [7,11,16.5]])
    assert torch.allclose(ctr, expected)


def test_scale(boxes_xyzxyz):
    b = boxes_xyzxyz.clone()
    b.scale(0.5, 0.5, 2.0)               # x&y shrink, z stretch
    assert torch.allclose(
        b.tensor[0],
        torch.tensor([0,0,0, 5,10,60], dtype=torch.float)
    )


def test_indexing(boxes_xyzxyz):
    first = boxes_xyzxyz[0]
    assert isinstance(first, Boxes)
    assert first.tensor.shape == (1, 6)
    sli = boxes_xyzxyz[:1]
    assert isinstance(sli, Boxes)
    mask = torch.tensor([True, False])
    masked = boxes_xyzxyz[mask]
    assert len(masked) == 1
    assert torch.equal(masked.tensor.squeeze(0), boxes_xyzxyz.tensor[0])


def test_cat(boxes_xyzxyz):
    merged = Boxes.cat([boxes_xyzxyz, boxes_xyzxyz])
    assert len(merged) == 4
    assert torch.equal(merged.tensor[:2], merged.tensor[2:])


@pytest.mark.parametrize("start,end", [
    (BoxMode3D.XYZWHD_ABS,  BoxMode3D.XYZXYZ_ABS),
    (BoxMode3D.CXCYCZWHD_ABS, BoxMode3D.XYZWHD_ABS),
    (BoxMode3D.XYZXYZ_ABS,  BoxMode3D.CXCYCZWHD_ABS),
])
def test_boxmode_roundtrip(start, end):
    # synth box
    src = torch.tensor([[2.,3.,4., 8.,9.,10.]])    # xyzxyz
    if start != BoxMode3D.XYZXYZ_ABS:              # convert once so src matches
        src = BoxMode3D.convert(src, BoxMode3D.XYZXYZ_ABS, start)
    back = BoxMode3D.convert(
        BoxMode3D.convert(src, start, end), end, start
    )
    assert torch.allclose(src, back, atol=1e-4, rtol=0)