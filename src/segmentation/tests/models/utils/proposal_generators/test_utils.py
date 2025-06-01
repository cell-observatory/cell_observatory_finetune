import pytest

import torch

from segmentation.models.utils import samplers


@pytest.fixture
def box_coder():
    return samplers.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))


def test_decode_returns_correct_boxes(box_coder):
    # deltas = [dx, dy, dz, dw, dh, dl] where dx = delta centre x, dy = delta centre y, dz = delta centre z
    # thus: [0.0, 0.0, 0.0, 0.1, 0.1, 0.1] designates move centre 0 units and 
    # move width 0.1 units in x, height 0.1 units in y, depth 0.1 units in z (encoded space)
    # equivalent to approx. 5 units in x, 5 units in y, 5 units in z (in original space) since
    # exp(0.1) = 1.1 and 1.1 * 50 = 55 - 50 = 5 for anchor box [0.0, 0.0, 0.0, 100.0, 100.0, 100.0]
    deltas = torch.tensor([[0.0, 0.0, 0.0, 0.1, 0.1, 0.1]])
    anchor = torch.tensor([[0.0, 0.0, 0.0, 100.0, 100.0, 100.0]])
    decoded = box_coder.decode(deltas, [anchor])

    assert decoded.shape ==  torch.Size([1, 1, 6])
    expected = torch.tensor([[ -5.2585,  -5.2585,  -5.2585, 105.2585, 105.2585, 105.2585]])
    assert torch.allclose(decoded, expected, atol=1e-2), f"got {decoded}, want {expected}"


def test_encode_roundtrips(box_coder):
    # decode some deltas and then encode back
    # for x decoder will do (same for y, z):
    # 1. dx * x_width + ctr_x (new pred centre x after apply dx)
    # 2. exp(dw) * width_x * 0.5 (new pred width from centre from sigmoid space delta w applied to half width)
    # 3. pred_ctr_x +- c_to_c_w => predicted new box coords for x
    original_deltas = torch.tensor([[0.0, 0.0, 0.0, 0.1, 0.1, 0.1]])
    anchor = torch.tensor([[0.0, 0.0, 0.0, 100.0, 100.0, 100.0]])
    decoded = box_coder.decode(original_deltas, [anchor])

    reencoded = box_coder.encode([decoded[0]], [anchor])
    assert reencoded[0].shape == torch.Size([1, 6])
    assert torch.allclose(reencoded[0], original_deltas, atol=1e-6), f"got {reencoded}, want {original_deltas}"