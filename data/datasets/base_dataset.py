from __future__ import annotations

import abc
from typing import (Any, Callable, Optional, 
                    Sequence, List, Union, Dict, Mapping
)

from omegaconf import ListConfig

from torch.utils.data import Dataset

from cell_observatory_finetune.data.databases.database import Database
from cell_observatory_finetune.data.structures.data_objects.data_shapes import (
    MULTICHANNEL_3D_HYPERCUBE,
    MULTICHANNEL_4D_HYPERCUBE
)
from cell_observatory_finetune.data.structures.sample_objects.data_sample import DataSample


class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict, callable], optional): Sequence of transform
            object to be composed.
    """

    def __init__(self, transforms: Optional[Sequence[Callable]] = None):
        self.transforms = list(transforms) if transforms is not None else []

    def __call__(self, data_sample: DataSample) -> Optional[DataSample]:
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        for t in self.transforms:
            data_sample = t(data_sample)
        return data_sample


class BaseDataset(Dataset, metaclass=abc.ABCMeta):
    """Base class for all datasets."""

    def __init__(
        self,
        db: Database,            
        key_cols: List[str],
        input_format: Union[MULTICHANNEL_3D_HYPERCUBE,
                      MULTICHANNEL_4D_HYPERCUBE
                      ],      
        transforms: Optional[Sequence] = None,
    ):
        super().__init__()
        self.input_format = input_format
        
        self.db = db
        self.key_cols = list(key_cols) if key_cols is not None else []

        self._process_tables()
        
        self._index = None
        self._build_index()
        
        self.transforms = Compose(transforms)

    @abc.abstractmethod
    def _process_tables(self) -> None:
        """Process tables in the database."""
        pass

    @abc.abstractmethod
    def _build_index(self) -> None:
        pass

    @abc.abstractmethod
    def _load_sample(self, idx_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Given one entry from the pre-built index, fetch & return raw data."""
        pass

    @abc.abstractmethod
    def _collate(self, raw: Dict[str, Any]) -> Mapping[str, Any]:
        """Turn raw dict into sample and data objects."""
        pass

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        sample_metadata = self._index[idx]
        _data = self._load_sample(sample_metadata)
        data = self._collate(_data)
        return self.transforms(data)