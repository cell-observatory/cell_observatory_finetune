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