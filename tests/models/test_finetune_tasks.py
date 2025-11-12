import pytest
pytestmark = pytest.mark.skip(reason="This module is temporarily disabled till we add ops3d to the docker image")

import torch

from cell_observatory_finetune.models.meta_arch.maskedautoencoder import FinetuneMaskedAutoEncoder


def _get_model(
    decoder,
    decoder_args,
    out_channels,
    B=1,
    T=16,
    Z=64,
    Y=64,
    X=64,
    C=2,
    lateral_patch_size=16,
    axial_patch_size=16,
    temporal_patch_size=4,
    model_template="mae-small",
    task="channel_split",
):

    model = FinetuneMaskedAutoEncoder(
        decoder_args=decoder_args,
        decoder=decoder,
        task=task,
        model_template=model_template,
        output_channels=out_channels,
        input_fmt="TZYXC",
        input_shape=(T, Z, Y, X, C),
        patch_shape=(temporal_patch_size, axial_patch_size, lateral_patch_size, lateral_patch_size, None),
        proj_drop_rate=0.0,
        att_drop_rate=0.0,
        drop_path_rate=0.0,
        fixed_dropout_depth=False,
        abs_sincos_enc=False,
        rope_pos_enc=True,
        rope_random_rotation_per_head=True,
        rope_mixed=True,
        rope_theta=10.0,
        weight_init_type="mae",
        mlp_wide_silu=False,
        loss_fn="l2_masked",
    ).to('cuda')
    return model

# -----------------------
# channel_split: linear
# -----------------------

@pytest.mark.parametrize("B,T,Z,Y,X,C", [
    (2, 8, 64, 64, 64, 2),
    (1, 8, 64, 64, 64, 2),
])
def test_channel_split_linear(B, T, Z, Y, X, C):
    torch.manual_seed(0)

    in_channels = 1

    decoder = "linear"
    decoder_args = dict(
        decoder_with_bn=False,
        decoder_num_layers=3,
        decoder_hidden_dim=256,
        decoder_bottleneck_dim=128,
        decoder_mlp_bias=True,
    )

    model = _get_model(B=B, T=T, Z=Z, Y=Y, X=X, C=in_channels, decoder=decoder, decoder_args=decoder_args, out_channels=C)
    model.eval()

    inputs = torch.randn(B, T, Z, Y, X, in_channels, dtype=torch.float32, device='cuda')

    num_patches = model.get_num_patches()
    pixels_per_patch = model.masked_encoder.patch_embedding.pixels_per_patch
    out_dim = pixels_per_patch * model.output_channels
    targets = torch.randn(B, num_patches, out_dim, dtype=torch.float32, device='cuda')

    data_sample = {
        "data_tensor": inputs,
        "metainfo": {
            "targets": [targets],
            "masks": [None],
            "context_masks": [None],
            "target_masks": [None],
            "original_patch_indices": [None],
        },
    }

    loss_dict, preds = model.forward(data_sample)

    assert "step_loss" in loss_dict
    assert torch.is_tensor(loss_dict["step_loss"])
    assert loss_dict["step_loss"].ndim == 0

    assert preds.shape == (B, num_patches, out_dim)

    assert torch.isfinite(preds).all(), "Predictions contain NaN/Inf"
    assert torch.isfinite(loss_dict["step_loss"]), "Loss is NaN/Inf"


# -----------------------
# channel_split: vit
# -----------------------

@pytest.mark.parametrize("B,T,Z,Y,X,C", [
    (2, 8, 64, 64, 64, 2),
    (1, 8, 64, 64, 64, 2),
])
@pytest.mark.parametrize("decoder,decoder_args", [
    ("vit",  dict(
        decoder_embed_dim=192, decoder_depth=3, decoder_num_heads=3
    )),
    # ("dense_predictor", dict(
    #     decoder_use_bn=True,
    #     decoder_feature_map_channels=[32, 32, 32, 32],
    #     decoder_strategy="axial",
    #     decoder_embed_dim=32,
    #     encoder_out_layers=[5, 6, 7, 8]
    # )),
])
def test_channel_split_decoders(B, T, Z, Y, X, C, decoder, decoder_args):
    torch.manual_seed(0)

    in_channels = 1

    model = _get_model(
        decoder=decoder,
        out_channels=C,
        decoder_args=decoder_args,
        B=B, T=T, Z=Z, Y=Y, X=X, C=in_channels,
        task="channel_split",
    )
    model.eval()

    # (B, T, Z, Y, X, C)
    inputs = torch.randn(B, T, Z, Y, X, in_channels, dtype=torch.float32, device="cuda")

    # Targets shape: [B, num_patches, pixels_per_patch * C]
    num_patches = model.get_num_patches()
    pixels_per_patch = model.masked_encoder.patch_embedding.pixels_per_patch
    out_dim = pixels_per_patch * model.output_channels
    targets = torch.randn(B, num_patches, out_dim, dtype=torch.float32, device="cuda")

    data_sample = {
        "data_tensor": inputs,
        "metainfo": {
            "targets": [targets],
            "masks": [None],
            "context_masks": [None],
            "target_masks": [None],
            "original_patch_indices": [None],
        },
    }

    loss_dict, preds = model.forward(data_sample)

    assert "step_loss" in loss_dict
    assert torch.is_tensor(loss_dict["step_loss"])
    assert loss_dict["step_loss"].ndim == 0

    assert preds.shape == (B, num_patches, out_dim)
    assert torch.isfinite(preds).all()
    assert torch.isfinite(loss_dict["step_loss"])


# -----------------------
# upsample_space: linear, vit
# -----------------------

@pytest.mark.parametrize("B,T,Z,Y,X,C", [
    (2, 8, 64, 64, 64, 2),
    (1, 8, 64, 64, 64, 2),
])
@pytest.mark.parametrize("decoder,decoder_args", [
    ("vit",  dict(
        decoder_embed_dim=192, decoder_depth=3, decoder_num_heads=3
    )),
    # ("dense_predictor", dict(
    #     decoder_use_bn=True,
    #     decoder_feature_map_channels=[32, 32, 32, 32],
    #     decoder_strategy="axial",
    #     decoder_embed_dim=32,
    #     encoder_out_layers=[5, 6, 7, 8]
    # )),
    ("linear", dict(
        decoder_with_bn=False,
        decoder_num_layers=3,
        decoder_hidden_dim=256,
        decoder_bottleneck_dim=128,
        decoder_mlp_bias=True,
    )),
])
def test_upsample_space_all_decoders(B, T, Z, Y, X, C, decoder, decoder_args):
    torch.manual_seed(0)

    model = _get_model(
        decoder=decoder,
        out_channels=C,
        decoder_args=decoder_args,
        B=B, T=T, Z=Z, Y=Y, X=X, C=C,
        task="upsample_space",
    )
    model.eval()

    # Inputs (B, T, Z, Y, X, C)
    inputs = torch.randn(B, T, Z, Y, X, C, dtype=torch.float32, device="cuda")

    # upsample_space/spacetime, decoder predicts pixels_per_patch (not * C)
    num_patches = model.get_num_patches()
    pixels_per_patch = model.masked_encoder.patch_embedding.pixels_per_patch
    out_dim = pixels_per_patch

    # Targets: [B, N, out_dim]
    targets = torch.randn(B, num_patches, out_dim, dtype=torch.float32, device="cuda")

    data_sample = {
        "data_tensor": inputs,
        "metainfo": {
            "targets": [targets],
            "masks": [None],
            "context_masks": [None],
            "target_masks": [None],
            "original_patch_indices": [None],
        },
    }

    loss_dict, preds = model.forward(data_sample)

    assert "step_loss" in loss_dict
    assert torch.is_tensor(loss_dict["step_loss"])
    assert loss_dict["step_loss"].ndim == 0

    assert preds.shape == (B, num_patches, out_dim)
    assert torch.isfinite(preds).all()
    assert torch.isfinite(loss_dict["step_loss"])


# -----------------------
# upsample_time: vit only
# -----------------------

@pytest.mark.parametrize("B,T,Z,Y,X,C", [
    (2, 8, 64, 64, 64, 2),
    (1, 8, 64, 64, 64, 2),
])
def test_upsample_time_vit(B, T, Z, Y, X, C):
    torch.manual_seed(0)

    decoder = "vit"
    decoder_args = dict(
        decoder_embed_dim=192, decoder_depth=3, decoder_num_heads=3
    )

    model = _get_model(
        B=B, T=T, Z=Z, Y=Y, X=X, C=C,
        out_channels=C,
        decoder=decoder, decoder_args=decoder_args,
        task="upsample_time",
    )
    model.eval()

    # Inputs (B, T, Z, Y, X, C)
    inputs = torch.randn(B, T, Z, Y, X, C, dtype=torch.float32, device="cuda")
    num_patches = model.get_num_patches()
    out_dim = model.masked_encoder.patch_embedding.pixels_per_patch

    B = inputs.shape[0]
    k = max(1, num_patches // 4)

    target_idx_list = []
    context_idx_list = []
    binary_masks_list = []

    for _ in range(B):
        perm = torch.randperm(num_patches, device="cuda")
        tgt = perm[:k].to(torch.long)
        ctx = perm[k:].to(torch.long)
        m = torch.zeros(num_patches, dtype=torch.int32, device="cuda")
        m[tgt] = 1
        target_idx_list.append(tgt)
        context_idx_list.append(ctx)
        binary_masks_list.append(m)

    target_masks = torch.stack(target_idx_list, dim=0)
    context_masks = torch.stack(context_idx_list, dim=0)

    original_patch_indices = torch.arange(num_patches, device="cuda").expand(B, -1).to(torch.long)

    targets = torch.randn(B, num_patches // 4, out_dim, dtype=torch.float32, device="cuda")

    data_sample = {
        "data_tensor": inputs,
        "metainfo": {
            "targets": [targets],
            "masks": [target_masks],                # used for num_patches=masks.sum()
            "context_masks": [context_masks],       # passed to encoder()
            "target_masks": [target_masks],         # supervision positions
            "original_patch_indices": [original_patch_indices],
        },
    }

    loss_dict, preds = model.forward(data_sample)

    assert "step_loss" in loss_dict
    assert torch.is_tensor(loss_dict["step_loss"]) and loss_dict["step_loss"].ndim == 0
    assert preds.shape == (B, num_patches//4, out_dim)
    assert torch.isfinite(preds).all()
    assert torch.isfinite(loss_dict["step_loss"])


# ----------------------------
# upsample_spacetime: vit only
# ----------------------------

@pytest.mark.parametrize("B,T,Z,Y,X,C", [
    (2, 8, 64, 64, 64, 2),
    (1, 8, 64, 64, 64, 2),
])
def test_upsample_spacetime_vit(B, T, Z, Y, X, C):
    torch.manual_seed(0)

    decoder = "vit"
    decoder_args = dict(
        decoder_embed_dim=192, decoder_depth=3, decoder_num_heads=3
    )

    model = _get_model(
        B=B, T=T, Z=Z, Y=Y, X=X, C=C,
        out_channels=C,
        decoder=decoder, decoder_args=decoder_args,
        task="upsample_spacetime",
    )
    model.eval()

    inputs = torch.randn(B, T, Z, Y, X, C, dtype=torch.float32, device="cuda")
    num_patches = model.get_num_patches()
    out_dim = model.masked_encoder.patch_embedding.pixels_per_patch

    B = inputs.shape[0]
    k = max(1, num_patches // 4)

    target_idx_list = []
    context_idx_list = []
    binary_masks_list = []

    for _ in range(B):
        perm = torch.randperm(num_patches, device="cuda")
        tgt = perm[:k].to(torch.long)
        ctx = perm[k:].to(torch.long)
        m = torch.zeros(num_patches, dtype=torch.int32, device="cuda")
        m[tgt] = 1
        target_idx_list.append(tgt)
        context_idx_list.append(ctx)
        binary_masks_list.append(m)

    target_masks = torch.stack(target_idx_list, dim=0)
    context_masks = torch.stack(context_idx_list, dim=0)

    original_patch_indices = torch.arange(num_patches, device="cuda").expand(B, -1).to(torch.long)

    targets = torch.randn(B, num_patches, out_dim, dtype=torch.float32, device="cuda")

    data_sample = {
        "data_tensor": inputs,
        "metainfo": {
            "targets": [targets],
            "masks": [target_masks],
            "context_masks": [context_masks],
            "target_masks": [target_masks],
            "original_patch_indices": [original_patch_indices],
        },
    }

    loss_dict, preds = model.forward(data_sample)

    assert "step_loss" in loss_dict
    assert torch.is_tensor(loss_dict["step_loss"]) and loss_dict["step_loss"].ndim == 0

    assert preds.shape == (B, num_patches, out_dim)
    assert torch.isfinite(preds).all()
    assert torch.isfinite(loss_dict["step_loss"])