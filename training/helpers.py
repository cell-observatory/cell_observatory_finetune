import copy
from typing import List, Tuple, Dict, Any, Optional

import torch
from torch import Tensor, nn


def get_supervised_input_data(model, inputs, mask_generator, device: Optional[torch.device] = 'cuda'):
    B = inputs[0]
    masks, context_masks, target_masks, \
                original_patch_indices, channels_to_mask = mask_generator(B)

    meta = {
        'masks': [masks.to(device)],
        'context_masks': [context_masks.to(device)],
        'target_masks': [target_masks.to(device)],
        'original_patch_indices': [original_patch_indices.to(device)],
        'channels_to_mask': [channels_to_mask.to(device)] if channels_to_mask is not None else None,
    }

    # summary() will unpack the input data but the fwd function in
    # JEPA and MAE models expects a dict hence we wrap the input data
    # in a tuple with a single dict element
    input_data = ({"data_tensor": torch.randn(*inputs, device=device), "metainfo": meta},)
    return input_data


def mask_ids_to_masks(batch_size, spatial_shape, mask_ids_batch, masks, device):
    """
    Convert per-sample mask IDs to per-sample binary masks.

    Args:
        batch_size (int): Number of samples in the batch.
        spatial_shape (tuple): Shape of the spatial dimensions.
        mask_ids_batch (list[list[int]]): For each sample in the batch, a list of instance IDs.
        masks (torch.Tensor): Tensor containing instance-ID maps.
                              Shape: [B, *spatial] or [*spatial] (then B assumed 1).
        input_format (str): Input format string (e.g. "TZYXC"). Used for sanity checks.
        input_shape (tuple): Shape of the input (no batch), matching input_format.
        device (torch.device): Device for output tensors.

    Returns:
        list[torch.Tensor]: For each sample b, a tensor of shape
                            [NUM_INST_b, *spatial], dtype=bool.
    """
    masks = masks.to(device)

    B = batch_size
    if len(mask_ids_batch) != B:
        raise ValueError(
            f"mask_ids_batch length ({len(mask_ids_batch)}) "
            f"does not match batch size ({B})."
        )

    binary_masks_batch = []
    for b in range(B):
        instance_ids = list(mask_ids_batch[b])
        m = masks[b]

        if len(instance_ids) == 0:
            # No instances: return empty [0, *spatial]
            empty = torch.zeros(
                (0,) + spatial_shape,
                dtype=torch.bool,
                device=device,
            )
            binary_masks_batch.append(empty)
            continue

        ids_tensor = torch.as_tensor(instance_ids, device=device, dtype=m.dtype)
        view_shape = (len(instance_ids),) + (1,) * m.dim()  # [N_inst, 1, 1, ...]
        binary_masks = (m.unsqueeze(0) == ids_tensor.view(view_shape))  # [N_inst, *spatial]
        binary_masks_batch.append(binary_masks.to(torch.bool))

    return binary_masks_batch


def get_image_sizes(
    input_format: str,
    input_shape: Tuple[int, ...],
    batch_size: int,
    metadata: Dict[str, Any],
    device: Optional[torch.device] = None,
):
    """
    Get image sizes and a 3D padding mask for each sample in the batch.

    Args:
        input_format (str): Input format string (e.g. "TZYXC", "ZYXC").
        input_shape (tuple): Shape of the input (no batch), matching input_format.
        batch_size (int): Number of samples in the batch.
        metadata (dict): Batch metadata; each key maps to a 1D array of
                         length `batch_size` (e.g. "y_size", "x_size", ...).
        device (torch.device, optional): Device on which to allocate the padding
                         mask. If None, uses CPU.

    Returns:
        image_sizes:        list[tuple], per-sample "current" sizes
        orig_image_sizes:   list[tuple], per-sample original sizes (or image_sizes)
        padding_mask:       torch.BoolTensor of shape [B, Z, Y, X] or [B, Y, X]
                            True = padded voxel, False = valid voxel.
    """
    if input_format == "TZYXC":
        ax_names = ("time", "z", "y", "x")
    elif input_format == "ZYXC":
        ax_names = ("z", "y", "x")
    elif input_format == "TCZYX":
        ax_names = ("time", "channel", "z", "y", "x")
    elif input_format == "CZYX":
        ax_names = ("channel", "z", "y", "x")
    else:
        raise ValueError(f"Unsupported input_format: {input_format}")

    image_sizes: List[Tuple[int, ...]] = []
    for i in range(batch_size):
        spatial_dims = [int(metadata[f"{ax}_size"][i]) for ax in ax_names]
        image_sizes.append(tuple(spatial_dims))

    # use orig_* sizes only if *all* are present
    if all(f"orig_{ax}_size" in metadata for ax in ax_names):
        orig_image_sizes: List[Tuple[int, ...]] = []
        for i in range(batch_size):
            spatial_dims = [int(metadata[f"orig_{ax}_size"][i]) for ax in ax_names]
            orig_image_sizes.append(tuple(spatial_dims))
    else:
        orig_image_sizes = image_sizes

    # Build a 3D padding mask [B, Z, Y, X] or [B, Y, X]
    # We only care about spatial volume axes for DETR-style masks.
    spatial_axes = [ax for ax in ("Z", "Y", "X") if ax in input_format]

    # map axis -> full size from input_shape
    axis_to_size = dict(zip(input_format, input_shape))
    full_sizes = {ax: int(axis_to_size[ax]) for ax in spatial_axes}

    # spatial mask shape (Z, Y, X) or (Y, X)
    spatial_shape = tuple(full_sizes[ax] for ax in spatial_axes)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    padding_mask = torch.zeros(
        (batch_size, *spatial_shape),
        dtype=torch.bool,
        device=device,
    )

    # metadata keys for sizes: z_size, y_size, x_size
    size_keys = {ax: f"{ax.lower()}_size" for ax in spatial_axes}

    for b in range(batch_size):
        # actual sizes along each spatial axis (default: full size if missing)
        actual = {}
        for ax in spatial_axes:
            key = size_keys[ax]
            if key in metadata:
                actual[ax] = int(metadata[key][b])
            else:
                actual[ax] = full_sizes[ax]

        # Mark padded voxels as True
        # We want: padded if index >= actual[ax] along ANY spatial axis.
        if spatial_axes == ["Z", "Y", "X"]:
            Z_full, Y_full, X_full = full_sizes["Z"], full_sizes["Y"], full_sizes["X"]
            z_lim, y_lim, x_lim = actual["Z"], actual["Y"], actual["X"]

            if z_lim < Z_full:
                padding_mask[b, z_lim:, :, :] = True
            if y_lim < Y_full:
                padding_mask[b, :, y_lim:, :] = True
            if x_lim < X_full:
                padding_mask[b, :, :, x_lim:] = True

        elif spatial_axes == ["Y", "X"]:
            Y_full, X_full = full_sizes["Y"], full_sizes["X"]
            y_lim, x_lim = actual["Y"], actual["X"]

            if y_lim < Y_full:
                padding_mask[b, y_lim:, :] = True
            if x_lim < X_full:
                padding_mask[b, :, x_lim:] = True

        else:
            raise ValueError(f"Unsupported spatial_axes combination: {spatial_axes}")

    return image_sizes, orig_image_sizes, padding_mask


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])