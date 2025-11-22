from typing import Optional

import torch


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


def mask_ids_to_masks(batch_size, spatial_shape, mask_ids_batch, masks, input_format, input_shape, device):
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


def get_image_sizes(input_format, input_shape, batch_size, metadata):
    """
    Get image sizes for each sample in the batch.

    Args:
        input_format (str): Input format string (e.g. "TZYXC").
        input_shape (tuple): Shape of the input (no batch), matching input_format.
        batch_size (int): Number of samples in the batch.
        metadata (dict): Batch metadata; each key maps to a 1D array of
                         length `batch_size` (e.g. "y_size", "x_size", ...).

    Returns:
        list[tuple], list[tuple]: (image_sizes, orig_image_sizes) for each sample.
    """
    if input_format == "TZYXC":
        ax_names = ('time', 'z', 'y', 'x')
    elif input_format == "ZYXC":
        ax_names = ('z', 'y', 'x')
    elif input_format == "TCZYX":
        ax_names = ('time', 'channel', 'z', 'y', 'x')
    elif input_format == "CZYX":
        ax_names = ('channel', 'z', 'y', 'x')
    else:
        raise ValueError(f"Unsupported input_format: {input_format}")

    image_sizes = []
    for i in range(batch_size):
        spatial_dims = [metadata[f"{ax}_size"][i] for ax in ax_names]
        image_sizes.append(tuple(spatial_dims))

    # use orig_* sizes only if all of them are present
    if all(f"orig_{ax}_size" in metadata for ax in ax_names):
        orig_image_sizes = []
        for i in range(batch_size):
            spatial_dims = [metadata[f"orig_{ax}_size"][i] for ax in ax_names]
            orig_image_sizes.append(tuple(spatial_dims))
    else:
        orig_image_sizes = image_sizes

    return image_sizes, orig_image_sizes