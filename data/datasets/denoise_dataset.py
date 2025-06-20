import os
from typing import Dict, Any

import torch
from torch.utils.data import get_worker_info

from cell_observatory_finetune.data.utils import read_zarr

from cell_observatory_finetune.data.structures.data_objects.image_list import ImageList
from cell_observatory_finetune.data.structures.sample_objects.instances import Instances
from cell_observatory_finetune.data.structures.sample_objects.data_sample import DataSample

from cell_observatory_finetune.data.datasets.base_dataset import BaseDataset


class DenoiseDataset(BaseDataset):
    """
    Dataset for denoise task.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._zarr_handles_data = {}

    def worker_init_fn(self, worker_id):
        worker_info = get_worker_info()
        # re-open handles in this worker only 
        # important to pass to dataloader
        data_paths = {
            os.path.join(sf, of, tn)
            for sf, of, tn in 
            zip(self.db.data_table["server_folder"], 
                self.db.data_table["output_folder"], 
                self.db.data_table["tile_name"]
            )
        }
        label_paths = {
            os.path.join(sf, of, tn)
            for sf, of, tn in 
            zip(self.db.label_table["server_folder"], 
                self.db.label_table["output_folder"], 
                self.db.label_table["tile_name"]
            )
        }

        self._zarr_handles_data = {
            p: read_zarr(p, return_handle=True)
            for p in data_paths
        }
        self._zarr_handles_labels = {
            p: read_zarr(p, return_handle=True)
            for p in label_paths
        }

    def _process_tables(self) -> None:
        # no-op for now, consider potentially filtering
        # data_table/label_table based on some criteria here
        return self.db.data_table, self.db.label_table

    def _build_index(self) -> None:
        # TODO: add a _label_index here that yields
        #       unique key : row pairs for labels
        #       this will be used in _load_sample to load
        #       the label crop for a given data crop
        #       we  don't currently have metadata/data
        #       for denoising so just using data for now

        # convert df into a list of Python dicts
        self._index = self.db.data_table.to_dict(orient="records")

    def _load_sample(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Read raw image crop into memory."""
        data_tensor = self._zarr_handles_data[os.path.join(meta["server_folder"], meta["output_folder"], meta["tile_name"])]
        
        # TODO: add support here for loading labels based on _label_index
        #       for now we just load the data crop, see instance_seg_dataset
        #       for ideas on how this would work

        t, c = slice(meta["t0"], meta["t1"]), slice(meta["c0"], meta["c1"])  
        z, y, x = slice(meta["z0"], meta["z1"]), slice(meta["y0"], meta["y1"]), slice(meta["x0"], meta["x1"])
        
        img = data_tensor[t, z, y, x, c].read().result()

        # hack for now until we have metadata for denoising
        label_tensor = data_tensor[t, z, y, x, c].read().result()  # using data as label for now

        return dict(meta=meta, image=img, labels=label_tensor)

    def _collate(self, _data: Dict[str, Any]) -> DataSample:
        meta  = _data["meta"]

        img_tensor = torch.tensor(_data["image"], dtype=torch.float32).clone() 
        label_tensor = torch.tensor(_data["labels"], dtype=torch.float32).clone()

        img_sample = ImageList(img_tensor,
                                layout=self.layout,
                                image_sizes=[img_tensor.shape])
        labels = ImageList(label_tensor, 
                               layout=self.layout, 
                               image_sizes=[label_tensor.shape])        

        inst = Instances()
        inst.image = img_sample  

        # default_collate (see utils.py) will convert
        # the image list tensor to a 5D tensor (B, C, D, H, W)
        # gt_instances will be a list of Instances
        sample = DataSample(metainfo=meta)
        sample.data_tensor, sample.gt_instances = labels, inst         
        return sample