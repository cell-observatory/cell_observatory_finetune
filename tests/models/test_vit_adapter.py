import pytest

import math
import torch

from cell_observatory_finetune.models.adapters.vit_adapter import (
    ConvFFN,
    DWConv,
    CrossAttention,
    Extractor,
    SpatialPriorModule,
    EncoderAdapter
)


# --- helpers ----


def _prod(tup):
    p = 1
    for v in tup:
        p *= int(v)
    return p


# --- ---- ----


@pytest.mark.parametrize("B,Nq,Nk,C,H", [
    (2, 13, 29, 32, 4),
    (1, 7,  11, 64, 8),
])
def test_cross_attention_shapes(B, Nq, Nk, C, H):
    m = CrossAttention(dim=C, num_heads=H)
    q = torch.randn(B, Nq, C)
    k = torch.randn(B, Nk, C)
    out = m(q, k)
    assert out.shape == (B, Nq, C)


# DWConv — 3D and 4D paths
@pytest.mark.parametrize("B,C,grids", [
    (2, 16, [(6, 8, 10), (3, 4, 5), (2, 2, 2)]),
    (1,  8, [(4, 4, 4),  (2, 2, 2), (1, 1, 1)]),
])
def test_dwconv_3d_shape(B, C, grids):
    offsets = [int(z * y * x) for (z, y, x) in grids]
    N = sum(offsets)
    x = torch.randn(B, N, C)
    m = DWConv(dim=3, embed_dim=C, strategy="axial")
    y = m(x, grids=grids, offsets=offsets)
    assert y.shape == (B, N, C)


def test_dwconv_4d_shape():
    B, C = 1, 8
    grids = [(2, 2, 2, 2), (2, 1, 1, 1), (1, 1, 1, 1)]  # (T,Z,Y,X)
    offsets = [g[0] * g[1] * g[2] * g[3] for g in grids]
    N = sum(offsets)
    x = torch.randn(B, N, C)
    m = DWConv(dim=4, embed_dim=C, strategy="axial")
    y = m(x, grids=grids, offsets=offsets)
    assert y.shape == (B, N, C)


# ConvFFN — shape preservation over tokens
@pytest.mark.parametrize("dim,offset_grids", [
    (3, [(4,4,4), (2,2,2), (1,1,1)]),
    (3, [(3,5,7), (2,2,2), (1,1,1)]),
])
def test_convffn_tokens_shape(dim, offset_grids):
    B, C = 2, 32
    offsets = [_prod(g) for g in offset_grids]
    N = sum(offsets)
    x = torch.randn(B, N, C)

    grids = offset_grids

    m = ConvFFN(in_features=C, 
                dim=dim,
                hidden_features=C, 
                out_features=C, 
                drop=0.0, 
                strategy="axial"
    )
    y = m(x, grids=grids, offsets=offsets)
    assert y.shape == (B, N, C)


def test_convffn_4d():
    B, C = 1, 16
    grids = [(2, 2, 2, 2), (1, 2, 2, 2), (1, 1, 1, 1)]
    offsets = [_prod(g) for g in grids]
    N = sum(offsets)
    x = torch.randn(B, N, C)
    m = ConvFFN(in_features=C, dim=4, hidden_features=C, out_features=C, drop=0.0, strategy="axial")
    y = m(x, grids=grids, offsets=offsets)
    assert y.shape == (B, N, C)


# Extractor — CrossAttention branch (no deformable)
@pytest.mark.parametrize("B, Nk, C, num_heads, grids", [
    (2, 256, 48, 6, [(4,4,4), (4,4,2), (4,4,2)]),  # 64+32+32=128
    (1,  64,  32, 4, [(4,4,2), (4,2,2), (4,2,2)]), # 32+16+16=64
])
def test_extractor_cross_attention_shapes(B, Nk, C, num_heads, grids):
    m = Extractor(embed_dim=C, dim=3, use_deform_attention=False,
                  num_heads=num_heads, with_cffn=True, cffn_ratio=0.5,
                  strategy="axial")

    offsets = [_prod(g) for g in grids]
    Nq = sum(offsets)

    query = torch.randn(B, Nq, C)
    feat  = torch.randn(B, Nk, C)

    out = m(query, feat,
            reference_points=None, spatial_shapes=None, level_start_index=None,
            grids=grids, offsets=offsets)
    assert out.shape == (B, Nq, C)


# --- 4D Extractor —---
@pytest.mark.parametrize("B,C,num_heads,grids", [
    (2, 48, 6, [(2, 2, 2, 2), (1, 2, 2, 1), (1, 1, 1, 1)]),  # 16 + 4 + 1 = 21
    (1, 32, 4, [(3, 2, 2, 1), (1, 2, 1, 1), (1, 1, 1, 1)]),  # 12 + 2 + 1 = 15
])
def test_extractor_cross_attention_only_4d(B, C, num_heads, grids):
    dim = 4
    m = Extractor(
        embed_dim=C,
        dim=dim,
        use_deform_attention=False,
        num_heads=num_heads,
        with_cffn=False,
        cffn_ratio=0.5,
        strategy="axial",
    )

    def _prod(t):
        p = 1
        for v in t: p *= int(v)
        return p

    offsets = [_prod(g) for g in grids]
    Nq = sum(offsets)
    Nk = max(Nq // 2, 1)

    query = torch.randn(B, Nq, C)
    feat  = torch.randn(B, Nk, C)

    out = m(
        query=query,
        features=feat,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        grids=grids,
        offsets=offsets,
    )
    assert out.shape == (B, Nq, C)


@pytest.mark.parametrize("B,C,num_heads,grids", [
    (2, 48, 6, [(2, 4, 4, 2), (1, 2, 2, 2), (1, 1, 1, 1)]),  # 64 + 8 + 1 = 73
    (1, 32, 4, [(3, 2, 2, 2), (1, 2, 1, 1), (1, 1, 1, 1)]),  # 24 + 2 + 1 = 27
])
def test_extractor_cross_attention_shapes_4d_with_ffn(B, C, num_heads, grids):
    dim = 4
    m = Extractor(
        embed_dim=C,
        dim=dim,
        use_deform_attention=False,
        num_heads=num_heads,
        with_cffn=True,
        cffn_ratio=0.5,
        strategy="axial",
    )

    def _prod(t):
        p = 1
        for v in t: p *= int(v)
        return p

    offsets = [_prod(g) for g in grids]
    Nq = sum(offsets)
    Nk = max(Nq // 2, 1)

    query = torch.randn(B, Nq, C)
    feat  = torch.randn(B, Nk, C)

    out = m(
        query=query,
        features=feat,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        grids=grids,
        offsets=offsets,
    )
    assert out.shape == (B, Nq, C)


# SpatialPriorModule — 3D and 4D exact shapes
@pytest.mark.parametrize(
    "B,in_ch,inplanes,embed_dim,ZYX",
    [
        (2, 2, 16, 32, (32, 48, 64)),  # divisible by 16
        (1, 3,  8, 16, (48, 64, 80)),  # divisible by 16
    ],
)
def test_spatial_prior_module_3d_simple_shapes(B, in_ch, inplanes, embed_dim, ZYX):
    Z, Y, X = ZYX
    x = torch.randn(B, in_ch, Z, Y, X)
    m = SpatialPriorModule(
        in_ch=in_ch, inplanes=inplanes, embed_dim=embed_dim, dim=3, strategy="axial"
    )
    out = m(x)
    c1, c2, c3, c4 = out

    # Known downsamples from strides: /2, /4, /8, /16
    z1, y1, x1 = Z // 2,  Y // 2,  X // 2
    z2, y2, x2 = Z // 4,  Y // 4,  X // 4
    z3, y3, x3 = Z // 8,  Y // 8,  X // 8
    z4, y4, x4 = Z // 16, Y // 16, X // 16

    assert c1.shape == (B, embed_dim, z1, y1, x1)
    assert c2.shape == (B, embed_dim, z2, y2, x2)
    assert c3.shape == (B, embed_dim, z3, y3, x3)
    assert c4.shape == (B, embed_dim, z4, y4, x4)


@pytest.mark.parametrize(
    "B,in_ch,inplanes,embed_dim,TZYX",
    [
        (1, 2, 16, 32, (4, 32, 48, 64)),  # Z/Y/X divisible by 16
        (2, 3,  8, 24, (3, 48, 64, 80)),  # Z/Y/X divisible by 16
    ],
)
def test_spatial_prior_module_4d_simple_shapes(B, in_ch, inplanes, embed_dim, TZYX):
    T, Z, Y, X = TZYX
    x = torch.randn(B, T, Z, Y, X, in_ch)
    m = SpatialPriorModule(
        in_ch=in_ch, inplanes=inplanes, embed_dim=embed_dim, dim=4, strategy="axial"
    )
    out = m(x)
    c1, c2, c3, c4 = out

    # Temporal length preserved; spatial downsamples /2, /4, /8, /16
    z1, y1, x1 = Z // 2,  Y // 2,  X // 2
    z2, y2, x2 = Z // 4,  Y // 4,  X // 4
    z3, y3, x3 = Z // 8,  Y // 8,  X // 8
    z4, y4, x4 = Z // 16, Y // 16, X // 16
    # Shapes are (B, T, C, Z, Y, X) after the projection/reshape path
    assert c1.shape == (B, T, embed_dim, z1, y1, x1)
    assert c2.shape == (B, T, embed_dim, z2, y2, x2)
    assert c3.shape == (B, T, embed_dim, z3, y3, x3)
    assert c4.shape == (B, T, embed_dim, z4, y4, x4)


# EncoderAdapter (ViT Adapter) — metadata helpers & forward


@pytest.mark.parametrize("dim,input_format,input_shape,patch", [
    (3, "ZYX",   (2, 32, 32, 32, 2), (None, 8, 8, )),   # temporal_patch ignored
    (4, "TZYX",  (1, 4, 16, 16, 16, 2), (2, 8, 8)),     # Tp=2, Zp=8, YXp=8
])
def test_encoder_adapter_metadata_shapes(dim, input_format, input_shape, patch):
    in_ch = input_shape[-1]
    adapter = EncoderAdapter(
        dim=dim,
        in_channels=in_ch,
        backbone_embed_dim=48,
        input_shape=input_shape,
        input_format=input_format,
        dtype="float32",
        axial_patch_size=patch[-2],
        lateral_patch_size=patch[-1],
        temporal_patch_size=patch[0] if dim == 4 else 1,
        use_deform_attention=False,
        deform_num_heads=6,
        strategy="axial",
    )

    def _prod(tup):
        p = 1
        for v in tup:
            p *= int(v)
        return p

    dummy_x = torch.randn(*input_shape)
    md = adapter._get_deformable_and_ffn_metadata(dummy_x)
    if dim == 3:
        (ref, spatial_shapes, lsi, vr, grids, offsets) = md
        assert ref.shape[-1] == 3  # reference_points (..., 3)
        assert len(grids) == 3 and len(offsets) == 3
    else:
        (ref, spatial_shapes, lsi, vr, grids, offsets) = md
        assert ref is None and spatial_shapes is None
        assert len(grids) == 3 and len(offsets) == 3


def test_encoder_adapter_forward():
    B, T, Z, Y, X, Cin = 1, 16, 128, 128, 128, 2
    embed_dim = 48
    Tp, Zp, YXp = 2, 16, 16
    adapter = EncoderAdapter(
        dim=4,
        in_channels=Cin,
        backbone_embed_dim=embed_dim,
        input_shape=(B, T, Z, Y, X, Cin),
        input_format="TZYX",
        dtype="float32",
        axial_patch_size=Zp,
        lateral_patch_size=YXp,
        temporal_patch_size=Tp,
        use_deform_attention=False,
        deform_num_heads=6,
        strategy="axial",
    )
    Tp_, Zp_, Yp_, Xp_ = (T // Tp), (Z // Zp), (Y // YXp), (X // YXp)
    base = (Tp_, Zp_, Yp_, Xp_)  # (2, 2, 2, 2)
    pyr = [
        (base[0], base[1] * 2, base[2] * 2, base[3] * 2), 
        base, 
        (base[0], max(1, base[1] // 2), max(1, base[2] // 2), 
         max(1, base[3] // 2))
    ]
    N = _prod(base)
    features = [torch.randn(B, N, embed_dim) for _ in range(4)]
    x = torch.randn(B, T, Z, Y, X, Cin)
    y = adapter(x, features)
    f1, f2, f3, f4 = y
    assert f1.shape == (B,embed_dim, T, Z // 2, Y // 2, X // 2)
    assert f2.shape == (B,embed_dim, T // 2, Z // 8, Y // 8, X // 8)
    assert f3.shape == (B,embed_dim, T // 2, Z // 16, Y // 16, X // 16)
    assert f4.shape == (B,embed_dim, T // 2, Z // 32, Y // 32, X // 32)