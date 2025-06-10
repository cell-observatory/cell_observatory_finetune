import os
from typing import Dict, Any

import torch
from torch.utils.data import get_worker_info

from segmentation.data.utils import read_zarr

from segmentation.structures.data_objects.image_list import ImageList
from segmentation.structures.sample_objects.instances import Instances
from segmentation.structures.sample_objects.data_sample import DataSample

from segmentation.data.datasets.base_dataset import BaseDataset


class ChannelSplitDataset(BaseDataset):
    """
    Dataset for channel splitting task.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._zarr_handles_data = {}

    def worker_init_fn(self, worker_id):
        worker_info = get_worker_info()
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
        img_tensor_merge = img_tensor.mean(dim=-1, keepdim=True)  # mean across channels
        img_sample = ImageList(img_tensor,
                                layout=self.layout,
                                image_sizes=[img_tensor.shape])
        img_sample_merge = ImageList(img_tensor_merge, 
                               layout=self.layout, 
                               image_sizes=[img_tensor_merge.shape])        

        inst = Instances()
        inst.image = img_sample  

        # default_collate (see utils.py) will convert
        # the image list tensor to a 5D tensor (B, C, D, H, W)
        # gt_instances will be a list of Instances
        sample = DataSample(metainfo=meta)
        sample.data_tensor, sample.gt_instances = img_sample_merge, inst         
        return sample