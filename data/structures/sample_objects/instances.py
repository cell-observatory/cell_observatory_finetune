"""
https://github.com/open-mmlab/mmengine/blob/main/mmengine/structures/instance_data.py

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


import itertools
from collections.abc import Sized
from typing import Any, List, Union, Dict

import numpy as np
import torch

from finetune.data.structures.data_objects.boxes import Boxes
from finetune.data.structures.data_objects.labels import Labels
from finetune.data.structures.data_objects.masks import BitMasks
from finetune.data.structures.sample_objects.base_data_element import BaseDataElement

BoolTypeTensor = Union[torch.BoolTensor, torch.cuda.BoolTensor]
LongTypeTensor = Union[torch.LongTensor, torch.cuda.LongTensor]
IndexType = Union[str, slice, int, list, LongTypeTensor,
                            BoolTypeTensor, np.ndarray]


class Instances(BaseDataElement):
    """
    Data structure for instance-level annotations or predictions.

    Subclass of :class:`BaseDataElement`. All value in `data_fields`should have the same length. 
    InstanceData also support extra functions: ``index``, ``slice`` and ``cat`` for data field. 
    The type of value in data field can be base data structure such as `torch.Tensor`, `numpy.ndarray`,
    `list`, `str`, `tuple`, and can be customized data structure that has
    ``__len__``, ``__getitem__`` and ``cat`` attributes.
    """

    def instances_to_dict(self) -> Dict[str, torch.Tensor]:
        target: Dict[str, torch.Tensor] = {}
        if hasattr(self, "boxes"):
            target["boxes"] = self.boxes.tensor
        if hasattr(self, "masks"):
            target["masks"]  = self.masks.tensor
        if hasattr(self, "labels"):
            target["labels"] = {"tensor" : self.labels.tensor, "num_classes" : self.labels.num_classes}
        return target
    
    def dict_to_instances(self, t: dict):
        if "boxes" in t:
            self.boxes = Boxes(t["boxes"])
        if "masks" in t:
            self.masks  = BitMasks(t["masks"])
        if "labels" in t:
            self.labels = Labels(t["labels"]["tensor"], t["labels"]["num_classes"])

    def __setattr__(self, name: str, value: Sized):
        """Setattr is only used to set data.

        The value must have the attribute of `__len__` and have the same length
        of `Instances`.
        """
        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(f'{name} has been used as a '
                                     'private attribute, which is immutable.')

        else:
            assert isinstance(value,
                              Sized), 'value must contain `__len__` attribute'

            if len(self) > 0:
                assert len(value) == len(self), 'The length of ' \
                                                f'values {len(value)} is ' \
                                                'not consistent with ' \
                                                'the length of this ' \
                                                ':obj:`Instances` ' \
                                                f'{len(self)}'
            super().__setattr__(name, value)

    __setitem__ = __setattr__

    def __getitem__(self, item: IndexType) -> 'Instances':
        """
        Args:
            item (str, int, list, :obj:`slice`, :obj:`numpy.ndarray`,
                :obj:`torch.LongTensor`, :obj:`torch.BoolTensor`):
                Get the corresponding values according to item.

        Returns:
            :obj:`Instances`: Corresponding values.
        """
        assert isinstance(item, IndexType.__args__)
        if isinstance(item, list):
            item = np.array(item)
        if isinstance(item, np.ndarray):
            # The default int type of numpy is platform dependent, int32 for
            # windows and int64 for linux. `torch.Tensor` requires the index
            # should be int64, therefore we simply convert it to int64 here.
            # More details in https://github.com/numpy/numpy/issues/9464
            item = item.astype(np.int64) if item.dtype == np.int32 else item
            item = torch.from_numpy(item)

        if isinstance(item, str):
            return getattr(self, item)

        if isinstance(item, int):
            if item >= len(self) or item < -len(self):  # type:ignore
                raise IndexError(f'Index {item} out of range!')
            else:
                # keep the dimension
                item = slice(item, None, len(self))

        new_data = self.__class__(metainfo=self.metainfo)
        if isinstance(item, torch.Tensor):
            assert item.dim() == 1, 'Only support to get the' \
                                    ' values along the first dimension.'
            if isinstance(item, BoolTypeTensor.__args__):
                assert len(item) == len(self), 'The shape of the ' \
                                               'input(BoolTensor) ' \
                                               f'{len(item)} ' \
                                               'does not match the shape ' \
                                               'of the indexed tensor ' \
                                               'in results_field ' \
                                               f'{len(self)} at ' \
                                               'first dimension.'

            for k, v in self.items():
                if isinstance(v, torch.Tensor):
                    new_data[k] = v[item]
                elif isinstance(v, np.ndarray):
                    # convert PyTorch tensor index to a NumPy array of ints
                    new_data[k] = v[item.cpu().numpy()]
                elif isinstance(
                        v, (str, list, tuple)) or (hasattr(v, '__getitem__')
                                                   and hasattr(v, 'cat')):
                    # convert to indexes from BoolTensor
                    # need to transform the Boolean mask into a list 
                    # of slice objects for sequence or custom containers
                    if isinstance(item, BoolTypeTensor.__args__):
                        # if idx is a BoolTensor, first extract
                        # the True positions
                        indexes = torch.nonzero(item).view(
                            -1).cpu().numpy().tolist()
                    else:
                        indexes = item.cpu().numpy().tolist()
                    
                    # build list of slice objects so that
                    # each element can be selected in turn
                    slice_list = []
                    if indexes:
                        for index in indexes:
                            # take v[idx:] in a way
                            # that preserves container __getitem__ semantics
                            slice_list.append(slice(index, None, len(v)))
                    else:
                        # if no index is selected, generate empty
                        # selection
                        slice_list.append(slice(None, 0, None))
                    r_list = [v[s] for s in slice_list]

                    # concatenate strings or Python-sequences
                    if isinstance(v, (str, list, tuple)):
                        new_value = r_list[0]
                        for r in r_list[1:]:
                            new_value = new_value + r
                    else:
                        # assuming v has a .cat method,
                        # so we can call it to merge pieces
                        new_value = v.cat(r_list)
                    new_data[k] = new_value
                else:
                    raise ValueError(
                        f'The type of `{k}` is `{type(v)}`, which has no '
                        'attribute of `cat`, so it does not '
                        'support slice with `bool`')

        else:
            # item is a slice
            for k, v in self.items():
                new_data[k] = v[item]
        return new_data  # type:ignore

    @staticmethod
    def cat(instances_list: List['Instances']) -> 'Instances':
        """Concat the instances of all :obj:`Instances` in the list.

        Note: To ensure that cat returns as expected, make sure that
        all elements in the list must have exactly the same keys.

        Args:
            instances_list (list[:obj:`Instances`]): A list
                of :obj:`Instances`.

        Returns:
            :obj:`Instances`
        """
        assert all(
            isinstance(results, Instances) for results in instances_list)
        assert len(instances_list) > 0
        if len(instances_list) == 1:
            return instances_list[0]

        # metainfo and data_fields must be exactly the
        # same for each element to avoid exceptions.
        field_keys_list = [
            instances.all_keys() for instances in instances_list
        ]
        assert len({len(field_keys) for field_keys in field_keys_list}) \
               == 1 and len(set(itertools.chain(*field_keys_list))) \
               == len(field_keys_list[0]), 'There are different keys in ' \
                                           '`instances_list`, which may ' \
                                           'cause the cat operation ' \
                                           'to fail. Please make sure all ' \
                                           'elements in `instances_list` ' \
                                           'have the exact same key.'

        new_data = instances_list[0].__class__(
            metainfo=instances_list[0].metainfo)
        for k in instances_list[0].keys():
            values = [results[k] for results in instances_list]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                new_values = torch.cat(values, dim=0)
            elif isinstance(v0, np.ndarray):
                new_values = np.concatenate(values, axis=0)
            elif isinstance(v0, (str, list, tuple)):
                new_values = v0[:]
                for v in values[1:]:
                    new_values += v
            elif hasattr(v0, 'cat'):
                new_values = v0.cat(values)
            else:
                raise ValueError(
                    f'The type of `{k}` is `{type(v0)}` which has no '
                    'attribute of `cat`')
            new_data[k] = new_values
        return new_data  # type:ignore

    def __len__(self) -> int:
        """int: The length of InstanceData."""
        if len(self._data_fields) > 0:
            return len(self.values()[0])
        else:
            return 0