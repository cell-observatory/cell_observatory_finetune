import pytest
import torch

from cell_observatory_finetune.models.heads.linear_head import LinearHead


@pytest.mark.parametrize(
    "input_format,input_shape,in_dim,bottleneck_dim",
    [
        # TZYXC: (B, T, Z, Y, X, C)
        ("TZYXC", (2, 4, 8, 8, 8, 16), 128, 64),
        # ZYXC:  (B, Z, Y, X, C)
        ("ZYXC",  (3, 16, 16, 16, 32), 256, 128),
    ],
)
def test_linear_head_default_shapes(input_format, input_shape, in_dim, bottleneck_dim):
    head = LinearHead(
        in_dim=in_dim,
        input_shape=input_shape,
        input_format=input_format,
        use_bn=False,
        nlayers=3,
        hidden_dim=max(2 * bottleneck_dim, bottleneck_dim),
        bottleneck_dim=bottleneck_dim,
        mlp_bias=True,
    )

    B = input_shape[0]
    L = 17
    C = input_shape[-1]

    x = torch.randn(B, L, in_dim)
    with torch.no_grad():
        y = head(x)

    assert y.shape == (B, L, C)


@pytest.mark.parametrize(
    "input_format,input_shape,in_dim,bottleneck_dim",
    [
        ("TZYXC", (1, 2, 4, 4, 4, 8), 64, 32),
        ("ZYXC",  (2, 8, 8, 8, 16), 128, 64),
    ],
)
def test_linear_head_no_last_layer_shapes(input_format, input_shape, in_dim, bottleneck_dim):
    head = LinearHead(
        in_dim=in_dim,
        input_shape=input_shape,
        input_format=input_format,
        use_bn=False,
        nlayers=2,
        hidden_dim=max(2 * bottleneck_dim, bottleneck_dim),
        bottleneck_dim=bottleneck_dim,
        mlp_bias=True,
    )

    B = input_shape[0]
    L = 5

    x = torch.randn(B, L, in_dim)
    with torch.no_grad():
        y = head(x, no_last_layer=True)

    assert y.shape == (B, L, bottleneck_dim)


@pytest.mark.parametrize(
    "input_format,input_shape,bottleneck_dim",
    [
        ("TZYXC", (2, 3, 6, 6, 6, 10), 64),
        ("ZYXC",  (1, 10, 10, 10, 20), 128),
    ],
)
def test_linear_head_only_last_layer_shapes(input_format, input_shape, bottleneck_dim):
    head = LinearHead(
        in_dim=999,
        input_shape=input_shape,
        input_format=input_format,
        use_bn=False,
        nlayers=3,
        hidden_dim=bottleneck_dim * 2,
        bottleneck_dim=bottleneck_dim,
        mlp_bias=True,
    )

    B = input_shape[0]
    L = 11
    C = input_shape[-1]

    x = torch.randn(B, L, bottleneck_dim)
    with torch.no_grad():
        y = head(x, only_last_layer=True)

    assert y.shape == (B, L, C)
