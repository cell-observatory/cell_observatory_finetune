import pytest
from hydra.utils import instantiate
from hydra import initialize, compose                               

from cell_observatory_finetune.data.utils import print_db
from cell_observatory_finetune.data.databases.database import SQLiteDatabase


@pytest.fixture(scope="module")
def cfg():
    with initialize(config_path="../../../configs/datasets/databases"):
        cfg = compose(
            config_name="sqlite_database",
        )
    return cfg

def test_full_constructor(cfg):
    label_spec  = instantiate(cfg.label_spec)
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

    print("\n======== data_table (head) ========")
    print(db.data_table.head())

    print("\n======== label_table (head) =======")
    print(db.label_table.head())

    print("\n======== SQL TABLE =======")
    print_db(db.db_path)