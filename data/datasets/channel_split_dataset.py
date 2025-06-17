import os
from typing import Dict, Any

import torch
from torch.utils.data import get_worker_info

from cell_observatory_finetune.data.utils import read_zarr

from cell_observatory_finetune.data.structures.data_objects.image_list import ImageList
from cell_observatory_finetune.data.structures.sample_objects.instances import Instances
from cell_observatory_finetune.data.structures.sample_objects.data_sample import DataSample

from cell_observatory_finetune.data.datasets.base_dataset import BaseDataset


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
            zip(self.db.metadata_table["server_folder"], 
                self.db.metadata_table["output_folder"], 
                self.db.metadata_table["tile_name"]
            )
        }
        self._zarr_handles_data = {
            p: read_zarr(p, return_handle=True)
            for p in paths
        }

    def _process_tables(self) -> None:
        # no-op for now, consider potentially filtering
        # data_table based on some criteria here
        return self.db.metadata_table

    def _build_index(self) -> None:
        # convert df into a list of Python dicts
        self._index = self.db.metadata_table.to_dict(orient="records")

    def _load_sample(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Read raw image crop into memory."""
        data_tensor = self._zarr_handles_data[os.path.join(meta["server_folder"], meta["output_folder"], meta["tile_name"])]
        # FIXME: for now doing this but we have to rework db structure
        #        channel and time will not work if we have >1 strides that we want 
        #        to use in these cases will need a t_start and c_start and a size variable
        #        also patched so will use + meta["cube_size"] soon instead of hardcoded 128
        #        also cubes dataset should have all information needed to load
        t, c = slice(meta["time"], int(meta["time"])+1), slice(meta["channel"], int(meta["channel"])+1)  
        z = slice(meta["z_start"] - 28, int(meta["z_start"] + 128 - 28)) 
        y = slice(meta["y_start"] - 0, int(meta["y_start"] + 128 - 0))
        x = slice(meta["x_start"] - 14, int(meta["x_start"] + 128 - 14))

        # z = slice(meta["z_start"] - meta["global_z_start"], int(meta["z_start"] + meta["cube_size"])) 
        # y = slice(meta["y_start"] - meta["global_y_start"], int(meta["y_start"] + meta["cube_size"]))
        # x = slice(meta["x_start"] - meta["global_x_start"], int(meta["x_start"] + meta["cube_size"]))
        
        img = data_tensor[t, z, y, x, c].read().result()  
        return dict(meta=meta, image=img)

    def _collate(self, _data: Dict[str, Any]) -> DataSample:
        meta  = _data["meta"]

        img_tensor = torch.tensor(_data["image"], dtype=torch.float32).clone() 
        # mean across channels
        # FIXME: assumes channel last
        img_tensor_merge = img_tensor.mean(dim=-1, keepdim=True)
        img_sample = ImageList(img_tensor,
                                layout=self.input_format,
                                image_sizes=[img_tensor.shape])
        img_sample_merge = ImageList(img_tensor_merge, 
                               layout=self.input_format, 
                               image_sizes=[img_tensor_merge.shape])        

        inst = Instances()
        inst.image = img_sample  

        # default_collate (see utils.py) will convert
        # the image list tensor to a 5D tensor (B, C, D, H, W)
        # gt_instances will be a list of Instances
        sample = DataSample(metainfo=meta)
        sample.data_tensor, sample.gt_instances = img_sample_merge, inst         
        return sample