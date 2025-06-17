from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ops3d import _C


def factors(N):
    """Returns the integer divisors of N as a list."""
    res = []
    for i in range(1, N+1):
        if N % i == 0:
            res.append(i)
    return res

def findspec(B, D, H, W, G, C):
    # key = f"{B}x{D}x{H}x{W}x{G}x{C}"
    d_stride = 8
    ms = factors(B*D*H*W)
    
    multiplier = 1
    for m in ms:
        if m <= 64 and (m * G * C // d_stride) <= 512:
            multiplier = m
    
    n_thread = multiplier * G * C // d_stride
    # key = f"{B}x{D}x{H}x{W}x{G}x{C}"
    return d_stride, n_thread


def find_spec_bwd(B, D, H, W, G, C, max_tpb=512, max_mult=64):
    # 1) pick d_stride so channels_per_thread = C//d_stride fits
    for d in (1, 2, 4, 8, 16, 32):
        if C % d == 0 and G * (C // d) <= max_tpb:
            d_stride = d
            break
    else:
        raise RuntimeError(f"Cannot fit G={G}, C={C} into {max_tpb} threads")

    # 2) pick multiplier so we can split B*Q / multiplier blocks 
    #    without exceeding max_tpb
    best_mult = 1
    for m in factors(B*D*H*W):
        thr = m * G * (C // d_stride)
        if m <= max_mult and thr <= max_tpb:
            best_mult = max(best_mult, m)

    blockthread = best_mult * G * (C // d_stride)
    return d_stride, blockthread


class DCNv4Function(Function):
    @staticmethod
    def forward(
            ctx, 
            input, offset_mask,
            kernel_d, kernel_h, kernel_w, 
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w, 
            dilation_d, dilation_h, dilation_w,
            group, group_channels, offset_scale,
            im2col_step, remove_center):

        # a given block is assigned block_multiplier * group * group_channels threads
        # and each thread does d_stride number of channels of work hence number of threads
        # needed for a given block is block_multiplier * group * group_channels / d_stride
        # where we choose block_multiplier to be the largest factor of B*D*H*W such that 
        # total number of theads per block stays below 512 (max is 1024 but generally 512 is used)
        # similar idea for the backward pass but we use a different d_stride and max thead count
        # this is all calibrated to take into account register and cache pressure and occupancy 
        forward_d_stride, forward_block_thread = findspec(input.shape[0], input.shape[1], 
                                                        input.shape[2], input.shape[3], group, group_channels)
        backward_d_stride, backward_block_thread = find_spec_bwd(input.shape[0], input.shape[1], input.shape[2],
                                                                input.shape[3], group, group_channels)

        ctx.kernel_d = kernel_d
        ctx.kernel_h = kernel_h
        ctx.kernel_w = kernel_w
        
        ctx.stride_d = stride_d
        ctx.stride_h = stride_h
        ctx.stride_w = stride_w

        ctx.pad_d = pad_d
        ctx.pad_h = pad_h
        ctx.pad_w = pad_w

        ctx.dilation_d = dilation_d
        ctx.dilation_h = dilation_h
        ctx.dilation_w = dilation_w

        ctx.group = group
        ctx.group_channels = group_channels
        ctx.offset_scale = offset_scale
        ctx.im2col_step = im2col_step
        ctx.remove_center = remove_center
        ctx.backward_d_stride = backward_d_stride
        ctx.backward_block_thread = backward_block_thread

        args = [
            input, offset_mask, 
            kernel_d, kernel_h, kernel_w, 
            stride_d, stride_h, stride_w, 
            pad_d, pad_h, pad_w, 
            dilation_d, dilation_h, dilation_w, 
            group, group_channels, offset_scale,
            ctx.im2col_step,
            remove_center,
            forward_d_stride,
            forward_block_thread,
            False,
        ]

        output = _C.dcnv4_forward(*args)
        ctx.save_for_backward(input, offset_mask)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset_mask = ctx.saved_tensors

        args = [
            input, offset_mask, 
            ctx.kernel_d, ctx.kernel_h, ctx.kernel_w, 
            ctx.stride_d, ctx.stride_h, ctx.stride_w, 
            ctx.pad_d, ctx.pad_h, ctx.pad_w, 
            ctx.dilation_d, ctx.dilation_h, ctx.dilation_w, 
            ctx.group, ctx.group_channels, ctx.offset_scale, ctx.im2col_step,
            grad_output.contiguous(), ctx.remove_center,
            ctx.backward_d_stride, ctx.backward_block_thread,
            False
        ]

        grad_input, grad_offset_mask = \
            _C.dcnv4_backward(*args)

        return grad_input, grad_offset_mask, \
            None, None, None, None, None, None, None,\
            None, None, None, None, None, None, None, \
            None, None, None, None
    

# -------------------------------------------- Torch Version for Debugging/Tests --------------------------------------------


def _get_reference_points(spatial_shapes, device, 
                          kernel_d, kernel_h, kernel_w, 
                          dilation_d, dilation_h, dilation_w, 
                          pad_d=0, pad_h=0, pad_w=0, 
                          stride_d=1, stride_h=1, stride_w=1):
    _, D_, H_, W_, _ = spatial_shapes
    D_out = (D_ - (dilation_d * (kernel_d - 1) + 1)) // stride_d + 1
    H_out = (H_ - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    W_out = (W_ - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1

    # returns: (3,D,H,W) meshgrid
    ref_z, ref_y, ref_x = torch.meshgrid(
        # linspace of kernel centers [half-span + 0.5, stride_h + half-span + 0.5 , ...]
        # and similar for x and z
        torch.linspace(
            # pad_d + 0.5,
            # D_ - pad_d - 0.5,
            (dilation_d * (kernel_d - 1)) // 2 + 0.5,
            (dilation_d * (kernel_d - 1)) // 2 + 0.5 + (D_out - 1) * stride_d,
            D_out,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            # pad_h + 0.5,
            # H_ - pad_h - 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5 + (H_out - 1) * stride_h,
            H_out,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            # pad_w + 0.5,
            # W_ - pad_w - 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5 + (W_out - 1) * stride_w,
            W_out,
            dtype=torch.float32,
            device=device))
    
    # ref_z: (D_out, H_out, W_out) -> (D_out*H_out*W_out,1) -> scale to [0,1]
    ref_z = ref_z.reshape(-1)[None] / D_
    ref_y = ref_y.reshape(-1)[None] / H_
    ref_x = ref_x.reshape(-1)[None] / W_

    # ref: (D_out*H_out*W_out,3) -> (1,D_out,H_out,W_out,1,3)
    ref = torch.stack((ref_x, ref_y, ref_z), -1).reshape(
        1, D_out, H_out, W_out, 1, 3)
    return ref


def _generate_dilation_grids(spatial_shapes, 
                             kernel_d, kernel_h, kernel_w, 
                             dilation_d, dilation_h, dilation_w, 
                             group, device):
    _, D_, H_, W_, _ = spatial_shapes
    
    points_list = []
    # (3, dil_d, dil_h, dil_w)
    x, y, z = torch.meshgrid(
        # linspace of dilation grid: [-half-span, +half-span] 
        # in kernel_w steps 
        torch.linspace(
            -((dilation_w * (kernel_w - 1)) // 2),
            -((dilation_w * (kernel_w - 1)) // 2) +
            (kernel_w - 1) * dilation_w, kernel_w,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            -((dilation_h * (kernel_h - 1)) // 2),
            -((dilation_h * (kernel_h - 1)) // 2) +
            (kernel_h - 1) * dilation_h, kernel_h,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            -((dilation_d * (kernel_d - 1)) // 2),
            -((dilation_d * (kernel_d - 1)) // 2) +
            (kernel_d - 1) * dilation_d, kernel_d,
            dtype=torch.float32,
            device=device))

    # scale to [0,1]
    points_list.extend([x / W_, y / H_, z / D_])

    # grid: (ker_h, ker_w, ker_d, 3) -> (ker_d*ker_h*ker_h, 1, 3)
    # -> (ker_d*ker_h*ker_h, group, 3) -> (ker_d*ker_h*ker_h*group, 1, 3)
    grid = torch.stack(points_list, -1).reshape(-1, 1, 3).\
        repeat(1, group, 1).permute(1, 0, 2)
    # grid: (1, 1, 1, group*ker_d*ker_h*ker_h, 3)
    # x,y,z dilated grid offsets for each group  
    grid = grid.reshape(1, 1, 1, group * kernel_d * kernel_h * kernel_w, 3)
    return grid


def dcn_core_pytorch(
        input, offset, mask, 
        kernel_d, kernel_h, kernel_w, 
        stride_d, stride_h, stride_w, 
        pad_d, pad_h, pad_w, 
        dilation_d, dilation_h, dilation_w, 
        group, group_channels, 
        offset_scale):
    input = F.pad(
        input,
        [0, 0, pad_d, pad_d, pad_h, pad_h, pad_w, pad_w])
    N_, D_in, H_in, W_in, _ = input.shape
    _, D_out, H_out, W_out, _ = offset.shape

    # ref: (1, W_out, H_out, D_out,1,3)
    ref = _get_reference_points(
        input.shape, input.device, 
        kernel_d, kernel_h, kernel_w, 
        dilation_d, dilation_h, dilation_w, 
        pad_d, pad_h, pad_w, 
        stride_d, stride_h, stride_w
    ).to(input.dtype)

    # grid: (1, 1, 1, group*kernel_d*kernel_h*kernel_w, 3)    
    grid = _generate_dilation_grids(
        input.shape, 
        kernel_d, kernel_h, kernel_w, 
        dilation_d, dilation_h, dilation_w, 
        group, input.device
    ).to(input.dtype)
    
    # (3,) -> (1,1,1,3) -> (1,1,1,3*group*kernel_d*kernel_h*kernel_w)
    spatial_norm = torch.tensor([W_in, H_in, D_in]).reshape(1, 1, 1, 3).\
        repeat(1, 1, 1, group*kernel_d*kernel_h*kernel_w).to(input.device, dtype=input.dtype)

    # sampling_locations: (N_, D_out, H_out, W_out, group*kernel_d*kernel_h*kernel_w, 3)
    # -> (N_, D_out, H_out, W_out, group*kernel_d*kernel_h*kernel_w*3) 
    # -> add offset: (N_, D_out, H_out, W_out, group*kernel_d*kernel_h*kernel_w*3) 
    sampling_locations = (ref + grid * offset_scale).repeat(N_, 1, 1, 1, 1, 1).flatten(4, 5) + \
        offset * offset_scale / spatial_norm

    # number of sampling points
    P_ = kernel_d * kernel_h * kernel_w
    # sampling_grids: 
    sampling_grids = 2 * sampling_locations - 1
    
    # (N_, D_in, H_in, W_in, group*group_channels) -> (N_, D_in*H_in*W_in, group*group_channels)
    # -> (N_, group*group_channels, D_in*H_in*W_in) -> (N_*group, group_channels, D_in, H_in, W_in)
    input_ = input.view(N_, D_in*H_in*W_in, group*group_channels).transpose(1, 2).\
        reshape(N_*group, group_channels, D_in, H_in, W_in)
    
    # (N_, D_out, H_out, W_out, group*P_*3) -> (N_, D_out*H_out*W_out, group, P_, 3) 
    # -> (N_, group, D_out*H_out*W_out, P_, 3) -> (N_*group, 1, D_out*H_out*W_out, P_, 3)
    sampling_grid_ = sampling_grids.view(N_, D_out*H_out*W_out, group, P_, 3).transpose(1, 2).flatten(0, 1).unsqueeze(1)
    
    # sampling_input_: (N_*group, group_channels, D_out*H_out*W_out, P_)
    sampling_input_ = F.grid_sample(
        input_, sampling_grid_, 
        mode='bilinear', padding_mode='zeros', 
        align_corners=False)
    
    sampling_input_ = sampling_input_.squeeze(2)

    # (N_, D_out, H_out, W_out, group*P_) -> (N_, D_out*H_out*W_out, group, P_) 
    # -> (N_, group, D_out*H_out*W_out, P_) -> (N_*group, 1, D_out*H_out*W_out, P_)
    mask = mask.view(N_, D_out*H_out*W_out, group, P_).transpose(1, 2).\
        reshape(N_*group, 1, D_out*H_out*W_out, P_)
    # (N_*group, group_channels, D_out*H_out*W_out, P_) * (N_*group, 1, D_out*H_out*W_out, P_)
    # -> (N_*group, group_channels, D_out*H_out*W_out, P_) -> (N_*group, group_channels, D_out*H_out*W_out)
    # -> (N_, group*group_channels, D_out*H_out*W_out)
    output = (sampling_input_ * mask).sum(-1).view(N_,
                                                   group*group_channels, D_out*H_out*W_out)
    # returns: (N_, D_out*H_out*W_out, group*group_channels) -> (N_, D_out, H_out, W_out, group*group_channels)
    return output.transpose(1, 2).reshape(N_, D_out, H_out, W_out, -1).contiguous()