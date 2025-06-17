from __future__ import absolute_import, division, print_function

import pytest
import torch

from cell_observatory_finetune.models.ops.dcnv4_func import DCNv4Function, dcn_core_pytorch

CUDA_AVAILABLE = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="CUDA required for DCNv4 kernel"
)


def _split_offset_mask(grad_offset_mask, N, Dz, Hy, Wx, G, P):
    grad_offset_mask = grad_offset_mask.view(N, Dz, Hy, Wx, G, P * 4)

    grad_offset = grad_offset_mask[..., : P * 3]              
    grad_mask   = grad_offset_mask[..., P * 3 :]              

    grad_offset = grad_offset.reshape(N, Dz, Hy, Wx, G * P * 3)
    grad_mask   = grad_mask.reshape(N, Dz, Hy, Wx, G, P)

    return grad_offset, grad_mask


@torch.no_grad()
def _make_offsets_and_masks(N, Dz, Hy, Wx, G, P, device, dtype):
    off = (torch.rand(N, Dz, Hy, Wx, G * P * 3, device=device, dtype=dtype) * 20) - 10
    mask = torch.rand(N, Dz, Hy, Wx, G, P, device=device, dtype=dtype) + 1e-4
    mask /= mask.sum(-1, keepdim=True)
    return off, mask


def test_dcnv4_forward_and_backward():
    torch.manual_seed(42)
    dev   = torch.device("cuda")
    dtype = torch.float32          

    N, G, Cg = 1, 4, 32          
    Kd = Kh = Kw = 3               
    remove_center = False
    P  = Kd * Kh * Kw - int(remove_center)   
    offset_scale  = 2.0

    pad = stride = dilation = 1

    Dz = Hy = Wx = 32              
    Dz_out = (Dz + 2 * pad - (dilation * (Kd - 1) + 1)) // stride + 1
    Hy_out = (Hy + 2 * pad - (dilation * (Kh - 1) + 1)) // stride + 1
    Wx_out = (Wx + 2 * pad - (dilation * (Kw - 1) + 1)) // stride + 1

    inp = torch.rand(N, Dz, Hy, Wx, G * Cg, device=dev, dtype=dtype, requires_grad=True)
    off_ref, mask_ref = _make_offsets_and_masks(
        N, Dz_out, Hy_out, Wx_out, G, P, dev, dtype
    )
    
    off_ref.requires_grad = True
    mask_ref.requires_grad = True

    off_flat  = off_ref.view(N, Dz_out, Hy_out, Wx_out, G, P * 3)
    mask_flat = mask_ref.view(N, Dz_out, Hy_out, Wx_out, G, P)
    offset_mask = torch.cat([off_flat, mask_flat], dim=-1).flatten(-2).detach()
    offset_mask.requires_grad = True

    ref_out = dcn_core_pytorch(
        inp, off_ref, mask_ref,
        Kd, Kh, Kw,
        stride, stride, stride,
        pad, pad, pad,
        dilation, dilation, dilation,
        G, Cg,
        offset_scale,
    )

    cuda_out = DCNv4Function.apply(
        inp, offset_mask,
        Kd, Kh, Kw,
        stride, stride, stride,
        pad, pad, pad,
        dilation, dilation, dilation,
        G, Cg, offset_scale,
        128, # im2col_step
        remove_center,
    )

    assert torch.allclose(cuda_out, ref_out, rtol=1e-2, atol=1e-3), (
        f"forward mismatch - max abs { (cuda_out - ref_out).abs().max():.4e}"
    )

    loss_ref  = ref_out.sum()
    loss_ref.backward(retain_graph=True)

    # gradients from reference path
    g_inp_ref   = inp.grad.detach().clone()
    g_off_ref   = off_ref.grad.detach().clone()
    g_mask_ref  = mask_ref.grad.detach().clone()

    inp.grad.zero_()

    loss_cuda = cuda_out.sum()
    loss_cuda.backward()

    # gradients from CUDA path
    g_inp_cuda  = inp.grad
    g_off_cuda, g_mask_cuda = _split_offset_mask(
        offset_mask.grad, N, Dz_out, Hy_out, Wx_out, G, P
    )

    assert torch.allclose(
        g_inp_cuda, g_inp_ref, rtol=1e-2, atol=1e-3
    ), "grad wrt input mismatch"

    assert torch.allclose(
        g_off_cuda, g_off_ref, rtol=1e-2, atol=1e-3
    ), "grad wrt offset mismatch"

    assert torch.allclose(
        g_mask_cuda, g_mask_ref, rtol=1e-2, atol=1e-3
    ), "grad wrt mask mismatch"
