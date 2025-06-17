"""
https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/structures/seg_data_sample.py

Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


from typing import List, Optional, Union, Dict, Any

from finetune.data.structures.data_objects.boxes import Boxes
from finetune.data.structures.data_objects.labels import Labels
from finetune.data.structures.data_objects.masks import BitMasks

from finetune.data.structures.data_objects.image_list import ImageList
from finetune.data.structures.sample_objects.instances import Instances
from finetune.data.structures.sample_objects.base_data_element import BaseDataElement


INSTANCE_FIELDS = {
    "masks": BitMasks,
    "boxes": Boxes,
    "labels": Labels,
    "image": ImageList,
}


class DataSample(BaseDataElement):
    """
    Unified container for training / inference data.
    Attributes:
        data_tensor : ImageList
            The (possibly-padded) image batch for this sample.
        gt_instances : Instances | list[Instances]
            Instance annotations.  During training with a batch size > 1 we keep a
            ``list[Instances]`` so every element still refers to its own image.
    """
    def __init__(self, *, metainfo: Optional[dict] = None, **kwargs):
        super().__init__(metainfo=metainfo, **kwargs)

    @property
    def gt_instances(self) -> Union[Instances, List[Instances]]:
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value: Union[Instances, List[Instances]]):
        if isinstance(value, Instances):
            pass
        elif isinstance(value, list) and all(isinstance(v, Instances) for v in value):
            pass
        else:
            raise TypeError(
                "`gt_instances` must be an `Instances` or list of `Instances`, "
                f"got {type(value)}"
            )
        self.set_field(value, "_gt_instances", dtype=(Instances, list))

    @gt_instances.deleter
    def gt_instances(self):
        del self._gt_instances

    @property
    def data_tensor(self) -> ImageList:
        return self._data_tensor

    @data_tensor.setter
    def data_tensor(self, value: ImageList):
        self.set_field(value=value, name="_data_tensor", dtype=ImageList, field_type='data')

    @data_tensor.deleter
    def data_tensor(self):
        del self._data_tensor

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        d["metainfo"] = self.metainfo

        # TODO: _init_args already captures tensor
        #       can be made more efficient
        img_list: ImageList = self.data_tensor
        d["data_tensor"] = img_list.tensor
        d["data_tensor_meta"] = img_list._init_args

        instance_list = (
            self.gt_instances
            if isinstance(self.gt_instances, list)
            else [self.gt_instances]
        )
        # for each data object container, grab:
        #   a) the raw tensor for each field
        #   b) the init_args for that container type
        per_image_instances = []
        for instance in instance_list:
            instance_dict: Dict[str,Any] = {}
            for field_name in INSTANCE_FIELDS:
                field_obj = getattr(instance, field_name, None)
                if field_obj is None:
                    continue
                instance_dict[field_name + "_meta"] = field_obj._init_args
            per_image_instances.append(instance_dict)

        d["gt_instances"] = per_image_instances
        return d
    
    @classmethod
    def from_dict(cls, d):
        inst = cls(metainfo=d["metainfo"])
        # rebuild ImageList
        args = d["data_tensor_meta"]
        img_list = ImageList(**{k: args[k] for k in args})
        inst.data_tensor = img_list

        # rebuild instances
        instances = []
        for img_meta in d["gt_instances"]:
            kw = {}
            for field_name, object in INSTANCE_FIELDS.items():
                if field_name + "_meta" not in img_meta:
                    continue
                init_args = img_meta[field_name + "_meta"]
                kw[field_name] = object(**init_args)
            instances.append(Instances(**kw))

        # TODO: double check if this is correct
        #       approach
        if len(instances) == 1:
            inst.gt_instances = instances[0]
        else:
            inst.gt_instances = instances
        return inst