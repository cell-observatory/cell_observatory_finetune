import pytest

from segmentation.data.databases.supabase_database import SupabaseDatabase


def test_supabase_fetch_and_init():
    """
    Instantiate SupabaseDatabase with the specified init args,
    and print out the resulting data_table and label_table. 
    """

    dotenv_path = "/clusterfs/nvme/hph/git_managed/env/credentials.env"  
    data_tile = [1, 2, 128, 128, 128]
    label_tile = [128, 128, 128]

    label_specs = {}

    data_specs = {
        "table": "prepared",
        "data_attributes": {
            "id": None,
            "server_folder": None,
            "output_folder": None,
            # "img_path": None,
            # "shape": None,
            # "dtype": None,
            # "time_size": None,
            # "channel_size": None,
            "z_start": None,
            "z_end": None,
            "y_start": None,
            "y_end": None,
            "x_start": None,
            "x_end": None,
        },
    }

    tile = True
    with_labels = False

    db_readpath = "/clusterfs/nvme/segment_4d/testing2/db_tables"
    db_savepath = "/clusterfs/nvme/segment_4d/testing2/db_tables"

    db_read_method = "feather"
    db_save_method = "feather"

    force_create_db = False

    db = SupabaseDatabase(
        dotenv_path,
        data_tile,
        label_tile,
        label_specs,
        [data_specs],
        tile,
        with_labels,
        db_readpath,
        db_savepath,
        db_read_method,
        db_save_method,
        force_create_db,
    )

    print("=== SupabaseDatabase.data_table ===")
    print(db.data_table)

    print("\n=== SupabaseDatabase.label_table ===")
    print(db.label_table)