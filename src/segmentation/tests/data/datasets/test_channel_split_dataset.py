import pytest
import torch

from hydra.utils import instantiate
from hydra import initialize, compose

from segmentation.data.databases.supabase_database import SupabaseDatabase
from segmentation.structures.sample_objects.data_sample import DataSample
from segmentation.data.datasets.channel_split_dataset import ChannelSplitDataset


@pytest.fixture(scope="module")
def db():
    with initialize(config_path="../../../configs/datasets/databases"):
        cfg = compose(config_name="supabase_database")

    label_spec = instantiate(cfg.label_spec)
    data_specs = [instantiate(s) for s in cfg.data_specs]

    return SupabaseDatabase(
        "channel_split",
        cfg.dotenv_path,
        cfg.data_tile,
        cfg.label_tile,
        label_spec,
        data_specs,
        cfg.tile,
        cfg.with_labels,
        cfg.db_readpath,
        cfg.db_savepath,
        cfg.db_read_method,
        cfg.db_save_method,
        cfg.force_create_db,
        )


@pytest.fixture(scope="module")
def dataset(db):
    return ChannelSplitDataset(
        key_cols = None,
        db = db,
        transforms = None,
        layout = "TZYXC", 
    )


def test_next_iter(dataset):
    assert len(dataset) > 0

    sample = next(iter(dataset))

    assert isinstance(sample, DataSample)

    data_tensor = sample.data_tensor.tensor
    assert torch.is_tensor(data_tensor)
    assert data_tensor.dtype == torch.float32

    # check that gt_instances is an Instances object
    # and that its gt_image also has a tensor
    inst = sample.gt_instances
    # ChannelSplitDataset sets inst.gt_image = ImageList(original_channels)
    assert hasattr(inst, "gt_image")
    gt_img = inst.gt_image.tensor
    assert torch.is_tensor(gt_img)
    assert gt_img.dtype == torch.float32

    # the merged‐channel tensor’s last three dims should match data_tile[-3:] 
    
    # TCZYX
    _, _, z_size, y_size, x_size = dataset.db.data_tile
    # BTCZYX 
    _, _, _, d, h, w = data_tensor.shape
    assert (d, h, w) == (z_size, y_size, x_size)

    print("Sample.meta:", sample.metainfo)
    print("data_tensor.shape:", data_tensor.shape)
    print("gt_image.shape:", gt_img.shape)