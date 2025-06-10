import pytest

import torch

from hydra.utils import instantiate
from hydra import initialize, compose

from segmentation.data.datasets.instance_seg_dataset import InstanceSegDataset
from segmentation.data.databases.database import SQLiteDatabase
from segmentation.structures.sample_objects.data_sample_old import DataSample


KEY_COLS=["img_id", "t0", "t1", "z0", "y0", "x0"]

@pytest.fixture(scope="module")
def db():
    with initialize(config_path="../../../configs/datasets/databases"):
        cfg = compose(config_name="sqlite_database")

    label_spec = instantiate(cfg.label_spec)
    data_specs = [instantiate(s) for s in cfg.data_specs]

    db = SQLiteDatabase(
        db_path      = cfg.db_path,
        data_tile    = cfg.data_tile,
        label_tile   = cfg.label_tile,
        label_spec   = label_spec,
        data_specs   = data_specs,
        db_readpath  = cfg.db_readpath,
        db_savepath  = cfg.db_savepath,
        db_read_method = getattr(cfg, "db_read_method", "feather"),
        db_save_method = getattr(cfg, "db_save_method", "feather"),
        force_create_db = False
    )
    return db


@pytest.fixture(scope="module")
def dataset(db):
    """Return the dataset under test."""
    return InstanceSegDataset(
        key_cols=KEY_COLS,
        db=db,
        transforms=None,
    )


def test_next_iter(dataset):
    assert len(dataset) > 0

    sample = next(iter(dataset))

    assert isinstance(sample, DataSample)

    assert hasattr(sample, "data_tensor")
    assert torch.is_tensor(sample.data_tensor.data)

    inst = sample.gt_instances
    assert torch.is_tensor(inst.bboxes.tensor)
    assert torch.is_tensor(inst.masks.tensor)

    assert sample.data_tensor.data.ndim in (4, 5)
    assert sample.data_tensor.data.dtype == torch.float32

    print("Sample: ", sample)