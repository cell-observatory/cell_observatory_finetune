import time

import torch

from cell_observatory_finetune.training.helpers import mask_ids_to_masks, get_image_sizes
from cell_observatory_finetune.data.utils import downsample, create_na_masks, resize_mask

from cell_observatory_platform.data.io import read_file
from cell_observatory_platform.data.data_types import TORCH_DTYPES, NUMPY_DTYPES
from cell_observatory_platform.models.preprocessor import RayPreprocessor
from cell_observatory_platform.models.patch_embeddings import PatchEmbedding


class FinetunePreprocessor(RayPreprocessor):
    def __init__(self,
                 transforms_list: list | None,
                 with_masking: bool,
                 mask_generator,
                 task: str,
                 patch_shape: tuple[int, int, int],
                 dtype: torch.dtype,
                 input_format: str,
                 input_shape: tuple[int, ...],
                 seed: int | None = None,
                 ideal_psf_path: str | None = None,
                 na_mask_thresholds: list[float] | None = None,
                 resize_na_masks: bool = True,
    ):
        super().__init__(dtype=dtype,
                         transforms_list=transforms_list,
                         with_masking=with_masking,
                         mask_generator=mask_generator)

        self.input_format = input_format
        # TODO: consider supporting different channel axis positions
        assert input_format[-1] == 'C', "Input format must end with 'C' (channels)"
        self.input_shape = input_shape

        # increment for batch dim
        self.axis_index = {ax: i+1 for i, ax in enumerate(input_format)}
        self.channel_idx = self.axis_index.get('C', None)
        self.time_idx = self.axis_index.get('T', None)
        self.z_idx = self.axis_index.get('Z', None)
        self.y_idx = self.axis_index.get('Y', None)
        self.x_idx = self.axis_index.get('X', None)

        # spatial dim indices for FFT (must be dims, not sizes)
        # increment by 1 to account for batch dim
        self.spatial_dims = tuple(
            i for ax, i in self.axis_index.items() if ax in ('Z', 'Y', 'X')
        )
        axis_to_size = dict(zip(input_format, input_shape)) # skip batch dim
        self.axial_shape = axis_to_size.get('Z', None)
        self.timepoints  = axis_to_size.get('T', None)
        if 'Y' not in axis_to_size or 'X' not in axis_to_size:
            raise ValueError("Input must include Y and X axes.")
        self.lateral_shape = (axis_to_size['Y'], axis_to_size['X'])
        self.channels = axis_to_size.get('C', None)
        self.spatial_shape = ((self.axial_shape,) + self.lateral_shape) \
            if self.axial_shape is not None else self.lateral_shape

        # dtype
        self.dtype = TORCH_DTYPES[dtype].value if isinstance(dtype, str) else dtype

        # set rng state
        self.rng = torch.Generator()
        if seed is None:
            self.rng.manual_seed(torch.initial_seed())
        else:
            self.rng.manual_seed(int(seed))

        self.task = task
        if self.task == "upsample_space" or self.task == "upsample_spacetime":
            if ideal_psf_path is None:
                raise ValueError("ideal_psf_path must be provided for upsample task")
            if na_mask_thresholds is None:
                raise ValueError("na_mask_thresholds must be provided for upsample task")

            self.resize_na_masks = resize_na_masks
            self.ideal_psf = torch.from_numpy(read_file(ideal_psf_path))
            self.na_masks = create_na_masks(
                self.ideal_psf,
                thresholds=na_mask_thresholds,
                target_shape=self.spatial_shape,
                resize=self.resize_na_masks,
            )

        self.patch_shape = patch_shape
        self.patch_embedding = PatchEmbedding(
            input_fmt=self.input_format,
            input_shape=self.input_shape,
            patch_shape=self.patch_shape,
            # dummy value to satisfy init; not used in preprocessor
            embed_dim=1,
            channels=self.channels,
        )

    def _split_inputs_and_masks(self, inputs: torch.Tensor):
        """
        Split `inputs` into:
        - inputs_wo_mask: all channels except `self.mask_idx`
        - masks: the mask channel (instance-id label map), with channel dim removed
        """
        C = inputs.shape[self.channel_idx]
        device = inputs.device

        # 1. Extract mask channel -> shape [B, *spatial]
        masks = inputs.select(dim=self.channel_idx, index=self.mask_idx)

        # 2. Build indices for all channels except the mask channel
        all_idx = torch.arange(C, device=device)
        keep_idx = torch.cat(
            [all_idx[: self.mask_idx], all_idx[self.mask_idx + 1 :]]
        )  # (C-1,)

        # 3. Select remaining channels along channel dim
        inputs_wo_mask = inputs.index_select(self.channel_idx, keep_idx)

        return inputs_wo_mask, masks

    def forward(self, data_sample: dict, data_time: float) -> dict:
        preprocess_time = time.time()

        inputs = data_sample['data_tensor']
        meta = data_sample.get('metainfo', {})

        if inputs.dtype != self.dtype:
            inputs = inputs.to(self.dtype)

        if self.transforms is not None:
            t0 = time.time()
            for transform in self.transforms:
                inputs = transform(inputs)
            transform_time = time.time() - t0
        else:
            transform_time = -1.0

        assert inputs.dtype == self.dtype, f"{inputs.dtype} != {self.dtype}"

        if self.task == "channel_split":
            if self.channel_idx is None:
                raise ValueError("Channel axis 'C' not present in input_format; cannot channel_split.")
            targets = self.patch_embedding.patchify(inputs)
            inputs = inputs.mean(dim=self.channel_idx, keepdim=True)

        elif self.task == "upsample_space" or self.task == "upsample_spacetime":
            targets = self.patch_embedding.patchify(inputs)
            idx = torch.randint(low=0,
                                high=self.na_masks.shape[0], 
                                size=(1,), 
                                generator=self.rng).item()
            na_mask = resize_mask(
                self.na_masks[idx],
                input_format=self.input_format,
                channels=self.channels,
                timepoints=self.timepoints,
                axial_shape=self.axial_shape,
                lateral_shape=self.lateral_shape,
                dtype=inputs.real.dtype if inputs.is_complex() else inputs.dtype,
                device=inputs.device,
            )
            inputs = downsample(
                na_mask=na_mask,
                inputs=inputs,
                spatial_dims=self.spatial_dims,
            )
        elif self.task == "upsample_time":
            targets = None
        
        elif self.task == "instance_segmentation":
            if self.channel_idx is None:
                raise ValueError(
                    "Channel axis 'C' not present in input_format; "
                    "cannot perform instance_segmentation."
                )

            inputs, masks = self._split_inputs_and_masks(inputs)

            # List of length B; each entry is a dict {mask_id: bbox}
            instances_list = meta["metadata_json"]["mask_bbox_dict"]

            mask_ids_batch: list[list[int]] = []
            bboxes_batch: list[torch.Tensor] = []

            for instances in instances_list:
                ids = list(instances.keys())
                mask_ids_batch.append(ids)

                if len(ids) == 0:
                    bboxes_batch.append(
                        torch.zeros((0, 4), device=inputs.device, dtype=torch.float32)
                    )
                else:
                    # Collect bboxes in same order as ids
                    bboxes_batch.append(
                        torch.as_tensor(
                            [instances[i] for i in ids],
                            device=inputs.device,
                            dtype=torch.float32,
                        )
                    )

            # Convert label maps -> per-sample binary masks (list of tensors)
            binary_masks_batch = mask_ids_to_masks(
                mask_ids_batch=mask_ids_batch,
                masks=masks,
                input_format=self.input_format,
                input_shape=self.input_shape,
                device=inputs.device,
            )

            # Build per-sample target dicts
            targets = []
            for mask_ids, bm, boxes in zip(mask_ids_batch, binary_masks_batch, bboxes_batch):
                targets.append(
                    {
                        "masks": bm,  # [NUM_INST, *spatial]
                        "bboxes": boxes,  # [NUM_INST, box_dim]
                        "mask_ids": torch.as_tensor(
                            mask_ids, device=inputs.device, dtype=torch.long
                        ),
                    }
                )

            # TODO: generalize to arbitrary spatial dims per sample across entire
            #       data pipeline 
            image_sizes, orig_image_sizes = get_image_sizes(
                input_format=self.input_format,
                input_shape=self.input_shape,
                batch_size=inputs.shape[0],
                metadata=meta
            )
            meta['image_sizes'] = image_sizes
            meta['orig_image_sizes'] = orig_image_sizes
        
        else:
            raise ValueError(f"Unknown task: {self.task}")

        if self.with_masking:
            mt0 = time.time()
            B = inputs.shape[0]
            masks, context_masks, target_masks, \
                original_patch_indices, channels_to_mask, patches_used = self.mask_generator(B)
            masking_time = time.time() - mt0

            mask_lists={}
            for name, mask in zip(
                ['masks', 'context_masks', 'target_masks', 'original_patch_indices', 'channels_to_mask'],
                [masks, context_masks, target_masks, original_patch_indices, channels_to_mask]
            ):
                if mask is not None:
                    mask_lists[name] = [mask]

            return {
                'data_tensor': inputs,
                'metainfo': {
                    **meta,
                    **mask_lists,                    
                    'targets': [targets],
                    'preprocess_time': time.time() - preprocess_time,
                    'data_time': data_time,
                    'masking_time': masking_time,
                    'transform_time': transform_time,
                }
            }
        else:
            return {
                'data_tensor': inputs,
                'metainfo': {
                    'preprocess_time': time.time() - preprocess_time,
                    'data_time': data_time,
                    'transform_time': transform_time,
                    'masking_time': -1.0,
                    'targets': [targets],
                    **meta
                }
            }