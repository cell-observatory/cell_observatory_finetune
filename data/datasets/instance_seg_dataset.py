import json
import random
from typing import Dict, Any

import numpy as np

import torch
from torch.utils.data import get_worker_info

from finetune.data.utils import read_zarr

from finetune.data.structures.data_objects.boxes import Boxes
from finetune.data.structures.data_objects.labels import Labels
from finetune.data.structures.data_objects.masks import BitMasks
from finetune.data.structures.data_objects.image_list import Shape
from finetune.data.structures.data_objects.image_list import ImageList

from finetune.data.structures.sample_objects.instances import Instances
from finetune.data.structures.sample_objects.data_sample import DataSample

from finetune.data.datasets.base_dataset import BaseDataset


class InstanceSegDataset(BaseDataset):
    """
    Dataset for 3D Instance Segmentation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._zarr_handles_data = {}
        self._zarr_handles_labels = {}

    def worker_init_fn(self, 
                       worker_id
    ):
        worker_info = get_worker_info()
        # re-open handles in this worker only 
        # important to pass to dataloader
        self._zarr_handles_data = {
            p: read_zarr(p, return_handle=True)
            for p in self.db.data_table["img_path"].unique()
        }
        self._zarr_handles_labels = {
            p: read_zarr(p, return_handle=True)
            for p in self.db.label_table["label_path"].unique()
        }

    #TODO: add support for removing crops with < N instances
    def _process_tables(self) -> None:
        # remove empty data tiles (with no instances)
        labels_keys = (
            self.db.label_table[self.key_cols]     
            .drop_duplicates()                
        )
        self.db.data_table = (
            self.db.data_table
                .merge(labels_keys, on=self.key_cols, how="inner")
                .reset_index(drop=True)
        )

    def _build_index(self) -> None:
        # build multiindex for label table
        # img_id & crop coordinates is the minimal
        # unique identifier for a label crop
        self._label_index = (
            self.db.label_table
                .set_index(self.key_cols)
                .sort_index()
        )
        # convert df into a list of Python dicts
        self._index = self.db.data_table.to_dict(orient="records")

    def _load_sample(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Read raw image crop & its label crop into memory."""
        key = tuple(meta[k] for k in self.key_cols)
        lrow  = self._label_index.loc[key]

        data_tensor = self._zarr_handles_data[meta["img_path"]]
        label_tensor = self._zarr_handles_labels[lrow["label_path"]]
        
        t, c = slice(meta["t0"], meta["t1"]), slice(meta["c0"], meta["c1"])  
        z, y, x = slice(meta["z0"], meta["z1"]), slice(meta["y0"], meta["y1"]), slice(meta["x0"], meta["x1"])

        img = data_tensor[t, z, y, x, c].read().result()  
        labels = label_tensor[t, z, y, x].read().result()

        instance_ids = np.asarray(json.loads(lrow["instance_ids"]), dtype=bool).astype(int)
        bboxes = np.asarray(json.loads(lrow["bboxes"]), dtype=np.float32)

        masks = self._ids_to_masks(instance_ids=instance_ids, labels=labels.squeeze(0))
        return dict(meta=meta, image=img, masks=masks, bboxes=bboxes, labels=instance_ids)

    def _collate(self, _data: Dict[str, Any]) -> DataSample:
        meta  = _data["meta"]

        img_tensor = torch.from_numpy(_data["image"]).float() 
        img_sample = ImageList(img_tensor, 
                               layout=self.layout, 
                               # TODO: is this the best way to handle image sizes?
                               image_sizes=[tuple(self.db.data_tile[-3:])],
                               standardize=True)        
        boxes = Boxes(torch.tensor(_data["bboxes"], dtype=torch.float32).clone())      
        masks = BitMasks(torch.tensor(_data["masks"], dtype=torch.bool).clone())
        labels = Labels(torch.tensor(_data["labels"], dtype=torch.int64).clone(), num_classes=2)

        inst = Instances()
        inst.boxes, inst.masks, inst.labels = boxes, masks, labels

        # default_collate (see utils.py) will convert
        # the image list tensor to a 5D tensor (B, C, D, H, W)
        # gt_instances will be a list of Instances
        sample = DataSample(metainfo=meta)
        sample.data_tensor, sample.gt_instances = img_sample, inst         
        return sample

    def _ids_to_masks(self, instance_ids: list, labels: np.ndarray, out_dtype = np.uint8) -> np.ndarray:
        instance_ids = np.asarray(instance_ids)        
        # fast broadcast compare
        masks = (labels[..., None] == instance_ids).astype(out_dtype)
        return np.moveaxis(masks, -1, 0)   