from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pytest

import torch

from segmentation.models.ops.dcnv4_func import DCNv4Function, dcn_core_pytorch


cuda_is_available = torch.cuda.is_available()
@pytest.mark.skipif(not cuda_is_available, reason="CUDA required for DCNv4 kernel")


@torch.no_grad()
def test_dcnv4_forward():
    torch.manual_seed(42)
    device = torch.device("cuda")

    # batch, groups, channels per group
    N, M, D = 1, 4, 32
    # kernel size
    Kd = Kh = Kw = 3 
    remove_center = False
    P = Kd * Kh * Kw - remove_center

    offset_scale = 2.0

    pad = 1
    dilation = 1
    stride = 1

    D_in = H_in = W_in = 32

    D_out = (D_in + 2 * pad - (dilation * (Kd - 1) + 1)) // stride + 1
    H_out = (H_in + 2 * pad - (dilation * (Kh - 1) + 1)) // stride + 1
    W_out = (W_in + 2 * pad - (dilation * (Kw - 1) + 1)) // stride + 1

    im2col_step = 8             

    dtype = torch.float32    
    
    inp = torch.rand(N, D_in, H_in, W_in, M * D, device=device, dtype=dtype)
    off = (torch.rand(N, D_out, H_out, W_out, M * P * 3,
                        device=device, dtype=dtype)) * 10

    mask = torch.rand(N, D_out, H_out, W_out, M, P,
                       device=device, dtype=dtype) + 1e-5
    mask /= mask.sum(-1, keepdim=True)
    mask = mask.reshape(N, D_out, H_out, W_out, M*P)

    ref_out = dcn_core_pytorch(
        inp, off, mask,
        Kd, Kh, Kw,
        stride, stride, stride,
        pad, pad, pad,
        dilation, dilation, dilation,
        M, D,
        offset_scale,
    )

    # (N, D_out, H_out, W_out, M, P, 4) -> (N, D_out, H_out, W_out, M * P * 4)
    offset_flat = off.view(N, D_out, H_out, W_out, M, P * 3)      
    mask_flat   = mask.view(N, D_out, H_out, W_out, M, P)         
    offset_mask = torch.cat([offset_flat, mask_flat], dim=-1)     
    offset_mask = offset_mask.flatten(-2)
    
    print(f"Input shape: {inp.shape}, "
          f"Offset/Mask shape: {offset_mask.shape}, "
          f"Output shape: {ref_out.shape}")

    cuda_out = DCNv4Function.apply(
        inp.to(dtype), 
        offset_mask.to(dtype),
        Kd, Kh, Kw,
        stride, stride, stride,
        pad, pad, pad,
        dilation, dilation, dilation,
        M, D, offset_scale,
        im2col_step, remove_center,
    )

    assert torch.allclose(cuda_out, ref_out, rtol=1e-2, atol=1e-3), (
        f"max abs err: {(cuda_out-ref_out).abs().max().item():.4e}, "
        f"max rel err: "
        f"{((cuda_out-ref_out).abs()/(ref_out.abs()+1e-6)).max().item():.4e}"
    )