import os
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import get_worker_info

from cell_observatory_finetune.data.utils import read_zarr, read_file, create_na_masks

from cell_observatory_finetune.data.structures.data_objects.image_list import ImageList, spatial_dims, Shape
from cell_observatory_finetune.data.structures.sample_objects.instances import Instances
from cell_observatory_finetune.data.structures.sample_objects.data_sample import DataSample

from cell_observatory_finetune.data.datasets.base_dataset import BaseDataset


class UpsampleDataset(BaseDataset):
    """
    Dataset for upsampling in space and time task.
    """
    def __init__(self, 
                 ideal_psf_path: str | Path,
                 na_mask_thresholds: List[float] = [0.1, 0.9],
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._zarr_handles_data = {}

        ideal_psf = read_file(ideal_psf_path)
        self.na_masks = create_na_masks(ideal_psf, na_mask_thresholds)

    def worker_init_fn(self, worker_id):
        worker_info = get_worker_info()
        worker_seed = torch.initial_seed() 
        # TODO: might be better ways to sample from
        #       first dim of tensor than this
        self._rng = random.Random(worker_seed)

        # re-open handles in this worker only 
        # important to pass to dataloader
        paths = {
            os.path.join(sf, of, tn)
            for sf, of, tn in 
            zip(self.db.data_table["server_folder"], 
                self.db.data_table["output_folder"], 
                self.db.data_table["tile_name"]
            )
        }
        self._zarr_handles_data = {
            p: read_zarr(p, return_handle=True)
            for p in paths
        }

    def _process_tables(self) -> None:
        # no-op for now, consider potentially filtering
        # data_table based on some criteria here
        return self.db.data_table

    def _build_index(self) -> None:
        # convert df into a list of Python dicts
        self._index = self.db.data_table.to_dict(orient="records")

    def _load_sample(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Read raw image crop into memory."""
        data_tensor = self._zarr_handles_data[os.path.join(meta["server_folder"], meta["output_folder"], meta["tile_name"])]
        
        t, c = slice(meta["t0"], meta["t1"]), slice(meta["c0"], meta["c1"])  
        z, y, x = slice(meta["z0"], meta["z1"]), slice(meta["y0"], meta["y1"]), slice(meta["x0"], meta["x1"])

        img = data_tensor[t, z, y, x, c].read().result()  
        return dict(meta=meta, image=img)

    def _collate(self, _data: Dict[str, Any]) -> DataSample:
        meta  = _data["meta"]

        img_tensor = torch.tensor(_data["image"], dtype=torch.float32).clone() 

        # TODO: add logic here to only downsample spatially if desired
        na_mask = self._random_na_mask(img_tensor)
        img_tensor_downsample = self.downsample(img_tensor, 
                                            na_mask=na_mask,
                                            spatial_dims=spatial_dims(self.layout, img_tensor.ndim))
        
        img_sample = ImageList(img_tensor,
                                layout=self.layout,
                                image_sizes=[img_tensor.shape])
        img_sample_downsample = ImageList(img_tensor_downsample, 
                                          layout=self.layout, 
                                          image_sizes=[img_tensor_downsample.shape])        

        instances = Instances()
        instances.image = img_sample  

        # default_collate (see utils.py) will convert
        # the image list tensor to a 5D tensor (B, C, D, H, W)
        # gt_instances will be a list of Instances
        sample = DataSample(metainfo=meta)
        sample.data_tensor, sample.gt_instances = img_sample_downsample, instances         
        return sample

    def downsample(self, 
                   img_tensor: torch.Tensor, 
                   na_mask: torch.Tensor, 
                   spatial_dims: tuple[int, ...]
    ) -> torch.Tensor:
        """Downsample the image tensor by a given factor."""
        # TODO: does doing no_grad make op. faster?
        with torch.no_grad():
            # FFT, shift to centre
            k = torch.fft.fftn(img_tensor, dim=spatial_dims)
            k = torch.fft.fftshift(k,  dim=spatial_dims)

            # clip: element-wise multiply 
            k.mul_(na_mask)

            # shift back and inverse FFT
            k  = torch.fft.ifftshift(k, dim=spatial_dims)
            out = torch.fft.ifftn(k, dim=spatial_dims).real
        return out      
    
    def _random_na_mask(self, target: torch.Tensor) -> torch.Tensor:
        """Pick one NA-mask, resize to (D,H,W), then broadcast for layout."""
        base = self._rng.choice(self.na_masks)
        z_idx, y_idx, x_idx = spatial_dims(self.layout, target.ndim)
        tgt_zyx = (target.shape[z_idx], target.shape[y_idx], target.shape[x_idx])

        mask_zyx = self._resize_mask(base, tgt_zyx)
        return self._broadcast_mask_to_layout(mask_zyx, target, self.layout)

    @staticmethod
    def _broadcast_mask_to_layout(
        mask_zyx: torch.Tensor,
        target: torch.Tensor,
        layout: Shape,
    ) -> torch.Tensor:
        """
        Return `mask_zyx` reshaped so that it broadcasts against `target`
        which follows `layout`.
        """
        assert mask_zyx.ndim == 3, "mask must be (D,H,W)"

        D, H, W = mask_zyx.shape
        tgt_ndim = target.ndim
        # TODO: is this needed, I think
        #       can be skipped, double check
        lead = tgt_ndim - len(layout.axes)

        shape = [1] * tgt_ndim
        axis_map = {"Z": D, "Y": H, "X": W}

        for ax, size in axis_map.items():
            dim = lead + layout.axes.index(ax)
            shape[dim] = size

        # adds extra dimensions to mask 
        # for broadcasting
        return mask_zyx.reshape(shape).to(
            dtype=target.dtype, device=target.device
        )
    
    @staticmethod
    def _resize_mask(
        mask_zyx: torch.Tensor,
        target_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Ensure the NA mask matches `target_shape` exactly.
        """
        if mask_zyx.shape == target_shape:
            return mask_zyx

        resized = F.interpolate(
            # (1,1,D,H,W)
            mask_zyx.unsqueeze(0).unsqueeze(0),      
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        return (resized > 0.5).float()