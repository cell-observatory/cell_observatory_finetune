"""
https://github.com/facebookresearch/detectron2/blob/9604f5995cc628619f0e4fd913453b4d7d61db3f/detectron2/layers/wrappers.py#L103
https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/layers/drop_path.py#L17
(ADD IDEA MASKDINO LINK)

(ADD COPYRIGHT HERE)

"""


import math

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


################################################################ ViTDet ############################################################

class Conv3d(torch.nn.Conv3d):
    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv3d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        # if not torch.jit.is_scripting():
        #     # Dynamo doesn't support context managers yet
        #     is_dynamo_compiling = check_if_dynamo_compiling()
        #     if not is_dynamo_compiling:
        #         with warnings.catch_warnings(record=True):
        #             if x.numel() == 0 and self.training:
        #                 # https://github.com/pytorch/pytorch/issues/12013
        #                 assert not isinstance(
        #                     self.norm, torch.nn.SyncBatchNorm
        #                 ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv3d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


################################################################ MaskDINO ############################################################


# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
#         return x
    

# def gen_sine_embeddings_for_pos(pos_tensor):
#     # n_query, bs, _ = pos_tensor.size()
#     # sineembed_tensor = torch.zeros(n_query, bs, 256)
#     scale = 2 * math.pi
#     dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
#     dim_t = 10000 ** (2 * (dim_t // 2) / 128)
#     x_embed = pos_tensor[:, :, 0] * scale
#     y_embed = pos_tensor[:, :, 1] * scale
#     pos_x = x_embed[:, :, None] / dim_t
#     pos_y = y_embed[:, :, None] / dim_t
#     pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
#     pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
#     if pos_tensor.size(-1) == 2:
#         pos = torch.cat((pos_y, pos_x), dim=2)
#     elif pos_tensor.size(-1) == 4:
#         w_embed = pos_tensor[:, :, 2] * scale
#         pos_w = w_embed[:, :, None] / dim_t
#         pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

#         h_embed = pos_tensor[:, :, 3] * scale
#         pos_h = h_embed[:, :, None] / dim_t
#         pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

#         pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
#     else:
#         raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
#     return pos


# def inverse_sigmoid(x, eps=1e-5):
#     x = x.clamp(min=0, max=1)
#     x1 = x.clamp(min=eps)
#     x2 = (1 - x).clamp(min=eps)
#     return torch.log(x1/x2)


# def gen_encoder_output_proposals(memory:Tensor, memory_padding_mask:Tensor, spatial_shapes:Tensor):
#     """
#     Input:
#         - memory: bs, \sum{hw}, d_model
#         - memory_padding_mask: bs, \sum{hw}
#         - spatial_shapes: nlevel, 2
#     Output:
#         - output_memory: bs, \sum{hw}, d_model
#         - output_proposals: bs, \sum{hw}, 4
#     """
#     N_, S_, C_ = memory.shape
#     base_scale = 4.0
#     proposals = []
#     _cur = 0
#     for lvl, (H_, W_) in enumerate(spatial_shapes):
#         mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
#         valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
#         valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

#         grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
#                                         torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
#         grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

#         scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
#         grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
#         wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
#         proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
#         proposals.append(proposal)
#         _cur += (H_ * W_)
#     output_proposals = torch.cat(proposals, 1)
#     output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
#     output_proposals = torch.log(output_proposals / (1 - output_proposals))
#     output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
#     output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

#     output_memory = memory
#     output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
#     output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
#     return output_memory, output_proposals


# def compute_unmasked_ratio(self, mask):
#     _, D, H, W = mask.shape
    
#     # valid_H = torch.sum(~mask[:, :, 0], 1)
#     # valid_W = torch.sum(~mask[:, 0, :], 1)

#     valid_D = (~mask).any(dim=(2,3)).sum(dim=1) # [B] — any valid pixel in each D-slice
#     valid_H = (~mask).any(dim=(1,3)).sum(dim=1) # [B] — any valid pixel in each H-slice
#     valid_W = (~mask).any(dim=(1,2)).sum(dim=1) # [B] — any valid pixel in each W-slice

#     valid_ratio_d = valid_D.float() / D
#     valid_ratio_h = valid_H.float() / H
#     valid_ratio_w = valid_W.float() / W
    
#     valid_ratio = torch.stack([valid_ratio_d, valid_ratio_w, valid_ratio_h], -1) # [B, 3]
#     return valid_ratio


############################################################ Swin Transformer ############################################################


# def _no_grad_trunc_normal_(tensor, mean, std, a, b):
#     """Tensor initialization with truncated normal distribution.
#     Based on:
#     https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
#     https://github.com/rwightman/pytorch-image-models

#     Args:
#        tensor: an n-dimensional `torch.Tensor`.
#        mean: the mean of the normal distribution.
#        std: the standard deviation of the normal distribution.
#        a: the minimum cutoff value.
#        b: the maximum cutoff value.
#     """

#     def norm_cdf(x):
#         return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

#     with torch.no_grad():
#         l = norm_cdf((a - mean) / std)
#         u = norm_cdf((b - mean) / std)
#         tensor.uniform_(2 * l - 1, 2 * u - 1)
#         tensor.erfinv_()
#         tensor.mul_(std * math.sqrt(2.0))
#         tensor.add_(mean)
#         tensor.clamp_(min=a, max=b)
#         return tensor

# def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
#     """Tensor initialization with truncated normal distribution.
#     Based on:
#     https://github.com/rwightman/pytorch-image-models

#     Args:
#        tensor: an n-dimensional `torch.Tensor`
#        mean: the mean of the normal distribution
#        std: the standard deviation of the normal distribution
#        a: the minimum cutoff value
#        b: the maximum cutoff value
#     """

#     if std <= 0:
#         raise ValueError("the standard deviation should be greater than zero.")

#     if a >= b:
#         raise ValueError("minimum cutoff value (a) should be smaller than maximum cutoff value (b).")

#     return _no_grad_trunc_normal_(tensor, mean, std, a, b)

# class DropPath(nn.Module):
#     """Stochastic drop paths per sample for residual blocks.
#     Based on:
#     https://github.com/rwightman/pytorch-image-models
#     """

#     def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
#         """
#         Args:
#             drop_prob: drop path probability.
#             scale_by_keep: scaling by non-dropped probability.
#         """
#         super().__init__()
#         self.drop_prob = drop_prob
#         self.scale_by_keep = scale_by_keep

#         if not (0 <= drop_prob <= 1):
#             raise ValueError("Drop path prob should be between 0 and 1.")

#     def drop_path(self, x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
#         if drop_prob == 0.0 or not training:
#             return x
#         keep_prob = 1 - drop_prob
#         shape = (x.shape[0],) + (1,) * (x.ndim - 1)
#         random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
#         if keep_prob > 0.0 and scale_by_keep:
#             random_tensor.div_(keep_prob)
#         return x * random_tensor

#     def forward(self, x):
#         return self.drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

# # TODO: Is this functionally same as other MLP?
# class MLPBlock(nn.Module):
#     """
#     A multi-layer perceptron block, based on: "Dosovitskiy et al.,
#     An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
#     """

#     def __init__(
#         self, hidden_size: int, mlp_dim: int, dropout_rate: float = 0.0, act: tuple | str = "GELU", dropout_mode="vit"
#     ) -> None:
#         """
#         Args:
#             hidden_size: dimension of hidden layer.
#             mlp_dim: dimension of feedforward layer. If 0, `hidden_size` will be used.
#             dropout_rate: fraction of the input units to drop.
#             act: activation type and arguments. Defaults to GELU. Also supports "GEGLU" and others.
#             dropout_mode: dropout mode, can be "vit" or "swin".
#                 "vit" mode uses two dropout instances as implemented in
#                 https://github.com/google-research/vision_transformer/blob/main/vit_jax/models.py#L87
#                 "swin" corresponds to one instance as implemented in
#                 https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_mlp.py#L23
#                 "vista3d" mode does not use dropout.

#         """

#         super().__init__()

#         if not (0 <= dropout_rate <= 1):
#             raise ValueError("dropout_rate should be between 0 and 1.")
#         mlp_dim = mlp_dim or hidden_size
#         act_name, _ = split_args(act)
#         self.linear1 = nn.Linear(hidden_size, mlp_dim) if act_name != "GEGLU" else nn.Linear(hidden_size, mlp_dim * 2)
#         self.linear2 = nn.Linear(mlp_dim, hidden_size)
#         self.fn = get_act_layer(act)
#         # Use Union[nn.Dropout, nn.Identity] for type annotations
#         self.drop1: Union[nn.Dropout, nn.Identity]
#         self.drop2: Union[nn.Dropout, nn.Identity]

#         dropout_opt = look_up_option(dropout_mode, SUPPORTED_DROPOUT_MODE)
#         if dropout_opt == "vit":
#             self.drop1 = nn.Dropout(dropout_rate)
#             self.drop2 = nn.Dropout(dropout_rate)
#         elif dropout_opt == "swin":
#             self.drop1 = nn.Dropout(dropout_rate)
#             self.drop2 = self.drop1
#         elif dropout_opt == "vista3d":
#             self.drop1 = nn.Identity()
#             self.drop2 = nn.Identity()
#         else:
#             raise ValueError(f"dropout_mode should be one of {SUPPORTED_DROPOUT_MODE}")

#     def forward(self, x):
#         x = self.fn(self.linear1(x))
#         x = self.drop1(x)
#         x = self.linear2(x)
#         x = self.drop2(x)
#         return x


# class PatchEmbed(nn.Module):
#     """
#     Patch embedding block based on: "Liu et al.,
#     Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
#     <https://arxiv.org/abs/2103.14030>"
#     https://github.com/microsoft/Swin-Transformer

#     Unlike ViT patch embedding block: (1) input is padded to satisfy window size requirements (2) normalized if
#     specified (3) position embedding is not used.

#     Example::

#         >>> from monai.networks.blocks import PatchEmbed
#         >>> PatchEmbed(patch_size=2, in_chans=1, embed_dim=48, norm_layer=nn.LayerNorm, spatial_dims=3)
#     """

#     def __init__(
#         self,
#         patch_size: Sequence[int] | int = 2,
#         in_chans: int = 1,
#         embed_dim: int = 48,
#         norm_layer: type[LayerNorm] = nn.LayerNorm,
#         spatial_dims: int = 3,
#     ) -> None:
#         """
#         Args:
#             patch_size: dimension of patch size.
#             in_chans: dimension of input channels.
#             embed_dim: number of linear projection output channels.
#             norm_layer: normalization layer.
#             spatial_dims: spatial dimension.
#         """

#         super().__init__()

#         if spatial_dims not in (2, 3):
#             raise ValueError("spatial dimension should be 2 or 3.")

#         patch_size = ensure_tuple_rep(patch_size, spatial_dims)
#         self.patch_size = patch_size
#         self.embed_dim = embed_dim
#         self.proj = Conv[Conv.CONV, spatial_dims]( # should be able to get away w conv3D here...
#             in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size
#         )
#         if norm_layer is not None:
#             self.norm = norm_layer(embed_dim)
#         else:
#             self.norm = None

#     def forward(self, x):
#         x_shape = x.size()
#         if len(x_shape) == 5:
#             _, _, d, h, w = x_shape
#             if w % self.patch_size[2] != 0:
#                 x = F.pad(x, (0, self.patch_size[2] - w % self.patch_size[2]))
#             if h % self.patch_size[1] != 0:
#                 x = F.pad(x, (0, 0, 0, self.patch_size[1] - h % self.patch_size[1]))
#             if d % self.patch_size[0] != 0:
#                 x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - d % self.patch_size[0]))

#         elif len(x_shape) == 4:
#             _, _, h, w = x_shape
#             if w % self.patch_size[1] != 0:
#                 x = F.pad(x, (0, self.patch_size[1] - w % self.patch_size[1]))
#             if h % self.patch_size[0] != 0:
#                 x = F.pad(x, (0, 0, 0, self.patch_size[0] - h % self.patch_size[0]))

#         x = self.proj(x)
#         if self.norm is not None:
#             x_shape = x.size()
#             x = x.flatten(2).transpose(1, 2)
#             x = self.norm(x)
#             if len(x_shape) == 5:
#                 d, wh, ww = x_shape[2], x_shape[3], x_shape[4]
#                 x = x.transpose(1, 2).view(-1, self.embed_dim, d, wh, ww)
#             elif len(x_shape) == 4:
#                 wh, ww = x_shape[2], x_shape[3]
#                 x = x.transpose(1, 2).view(-1, self.embed_dim, wh, ww)
#         return x

####################################################################################################################################