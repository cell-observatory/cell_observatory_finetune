import pytest 

import torch

from finetune.models.proposal_generators.anchor_generator import AnchorGenerator


@pytest.fixture
def gather_anchor_generator():
    sizes = ((2,),)
    aspect_ratios = ((1.0,),)
    aspect_ratios_z = ((1.0,),)
    return AnchorGenerator(
        sizes=sizes,
        aspect_ratios=aspect_ratios,
        aspect_ratios_z=aspect_ratios_z,
    )


def test_cell_anchors(gather_anchor_generator):
    # cell_anchors should be a list of length 1,
    # each entry is a (num_anchors_per_location, 6) tensor
    cell_list = gather_anchor_generator.cell_anchors
    assert isinstance(cell_list, list)
    assert len(cell_list) == 1

    cell_anchors = cell_list[0]
    # only one size x one aspect_ratio_z x one aspect_ratio => 1 anchor per location
    assert cell_anchors.shape == (1, 6)

    expected = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    assert torch.allclose(cell_anchors[0], expected)


def test_grid_anchors(gather_anchor_generator):
    grid_sizes = [(2, 2, 2)]
    strides = [(1, 1, 1)]
    all_anchors = gather_anchor_generator.grid_anchors(grid_sizes, strides)

    # should again be a list of length 1
    assert isinstance(all_anchors, list)
    assert len(all_anchors) == 1

    grid = all_anchors[0]
    # 2x2x2 grid x 1 anchor per location => 8 boxes, each with 6 coords
    assert grid.shape == (8, 6)

    # we check the first and last anchor
    first = grid[0]
    exp_first = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    assert torch.allclose(first, exp_first)

    last = grid[-1]
    exp_last = torch.tensor([0.0, 0.0, 0.0, 2.0, 2.0, 2.0])
    assert torch.allclose(last, exp_last)