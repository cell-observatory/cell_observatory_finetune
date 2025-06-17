import pytest
import torch

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from cell_observatory_finetune.data.structures.data_objects.boxes import Boxes
from cell_observatory_finetune.utils.visualization import Visualizer, COLOR_MAP


@pytest.fixture
def viz():
    return Visualizer(save_format="zarr", 
                      save_metadata={
                          "shard_cube_shape" : (1, 128, 128, 128, 1),
                          "chunk_shape" : (1, 64, 64, 64, 1)
                          })


def test_get_color_enum_and_tuple(viz):
    assert viz._get_color(COLOR_MAP.GREEN) == (0, 255, 0)
    # single-value tuple should broadcast to R=G=B
    assert viz._get_color((17,)) == (17, 17, 17)

def test_merge_instance_masks(viz):
    m1 = torch.tensor([[1, 0], [0, 0]], dtype=torch.bool)
    m2 = torch.tensor([[0, 1], [1, 0]], dtype=torch.bool)

    merged = viz._merge_instance_masks([m1, m2])
    expected = torch.tensor([[1, 2],
                             [2, 0]], dtype=torch.int32)
    assert torch.equal(merged, expected)

def test_merge_semantic_masks(viz):
    m_bg = torch.tensor([[0, 1], [1, 0]], dtype=torch.bool)  # class-0
    m_fg = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)  # class-1

    merged = viz._merge_semantic_masks([m_bg, m_fg])
    expected = torch.tensor([[1, 0],
                             [0, 1]], dtype=torch.int32)
    assert torch.equal(merged, expected)

def test_visualize_masked_tensor(viz):
    t = torch.arange(6.).reshape(2, 3)
    m = torch.tensor([[1, 0, 0],
                      [0, 1, 0]], dtype=torch.bool)

    out = viz.visualize_masked_tensor(t, m)
    expected = t.clone()
    expected[m] = 0
    assert torch.equal(out, expected)

@pytest.mark.parametrize("shape_bad", [(3, 2), (2, 3, 1)])
def test_visualize_masked_tensor_shape_mismatch(viz, shape_bad):
    t = torch.zeros(2, 3)
    bad_mask = torch.zeros(shape_bad, dtype=torch.bool)
    with pytest.raises(ValueError):
        viz.visualize_masked_tensor(t, bad_mask)

# draw boxes
def test_visualize_boxes_draws_edges(viz):
    D, H, W = (64, 64, 64)
    img_size = (D, H, W)

    box = torch.tensor([[1, 1, 1, 50, 50, 50]], dtype=torch.float32)
    boxes = Boxes(box)

    vol = viz.visualize_boxes(img_size, boxes, edge_color=(1,), line_width=1)
    np_vol = vol.detach().cpu().numpy()

    z1, z2 = 1, 49
    y1, y2 = 1, 49
    x1, x2 = 1, 49

    # z-faces (top & bottom)
    assert (np_vol[:, z1, y1:y2, x1:x2] == 1).all()
    assert (np_vol[:, z2, y1:y2, x1:x2] == 1).all()

    # y-faces (front & back)
    assert (np_vol[:, z1:z2, y1, x1:x2] == 1).all()
    assert (np_vol[:, z1:z2, y2, x1:x2] == 1).all()

    # x-faces (left & right)
    assert (np_vol[:, z1:z2, y1:y2, x1] == 1).all()
    assert (np_vol[:, z1:z2, y1:y2, x2] == 1).all()

    assert (np_vol[:, z1+1 : z2, y1+1 : y2, x1+1 : x2] == 0).all()