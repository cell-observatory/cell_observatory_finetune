import sqlite3
from enum import Enum
from itertools import product

import torch


class Dtypes(Enum):
    fp16 = torch.float16
    bf16 = torch.bfloat16
    fp32 = torch.float32


class Dimension(Enum):
    DIM_3D_BZYX   = "BZYX"
    DIM_4D_BZYXC  = "BZYXC"
    DIM_4D_BTZYX  = "BTZYX"
    DIM_5D_BTZYXC = "BTZYXC"


class ColorMode(Enum):
    AVG = "Average"
    MATCH = "MATCH"
    # TODO: add different color modes: index, target protein etc


def get_base_dataset(dataset):
    while isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    return dataset


def worker_init_fn_db(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = get_base_dataset(worker_info.dataset)
    # Close the existing connection (if any) and open a new one.
    if hasattr(dataset, "con") and dataset.con is not None:
        dataset.con.close()
    dataset.con = sqlite3.connect(dataset.local_db_name)
    dataset.cur = dataset.con.cursor()


def collate_fn_segmentation(batch):
    data_items, label_items = zip(*batch)
    return torch.stack(data_items, dim=0), label_items


class DataConfig:
    def __init__(self, t = None, z = 128, y = 128, x = 128, c = None, color_mode = ColorMode.AVG):
        self.t = t
        self.z = z
        self.y = y
        self.x = x
        self.c = c

        if isinstance(color_mode, str):
            color_mode = ColorMode[color_mode]  # Convert string to enum member

        self.color_mode = color_mode
        if self.color_mode == ColorMode.MATCH:
            if c is None:
                raise ValueError("Color mode MATCH requires number of channels (c) to be set.")
            self.c = c
        elif self.color_mode == ColorMode.AVG:
            self.c = 1
        else:
            raise ValueError(f"Unknown color mode: {color_mode}")

        self.dim = self._determine_data_dimension()

        if t is None:
            self.t = 1


    def _determine_data_dimension(self):
        has_time = self.t is not None
        has_color = self.color_mode == ColorMode.MATCH

        if has_time and has_color:
            return Dimension.DIM_5D_BTZYXC
        elif has_time and not has_color:
            return Dimension.DIM_4D_BTZYX
        elif has_color and not has_time:
            return Dimension.DIM_4D_BZYXC
        else:
            return Dimension.DIM_3D_BZYX

    def __repr__(self):
        t = 1 if self.t is None else 1
        c = 1 if self.c is None else 1
        return f"data_config_t_{t}_z_{self.z}_y_{self.y}_x_{self.x}_dim_{self.dim.name}_color_mode_{self.color_mode.name}"
    

def index_mapper(shape: tuple[int, int, int, int ,int],
                 batch_config : DataConfig) -> list[tuple[int, int, int, int, int]]:
    """
    Given a object shape and data config which contains batch shape information, returns a list of tuples
    that map an index to
    
    Args:
        shape: Object shape
        batch_config: DataConfig object which contains batch shape information and how to handle color channels

    Returns:
        indices: list of indices that map a batch index to the tile index and {time,z,y,x,c} slices
    """
    # Tensorstore object dimensions assumed to be in (N,Z,Y,X,C) format
    n_z, n_c, n_y, n_x = shape

    # Calculate the number of batches in each store
    if batch_config.color_mode == ColorMode.MATCH and n_c != batch_config.c:
        return None

    if batch_config.color_mode == ColorMode.AVG or  batch_config.color_mode == ColorMode.MATCH:
        # AVG: output will be averaged so will have a single color channel
        # MATCH: output channel size must match input channel size, therefore color channel won't be s
        n_c = 1
    else:
        n_c = batch_config.c

    n_z = n_z // batch_config.z
    n_y = n_y // batch_config.y
    n_x = n_x // batch_config.x

    if n_z == 0 or n_y == 0 or n_x == 0:
        raise ValueError(f"Cropping with batch size {batch_config.z, batch_config.y, batch_config.x} is too large for object of shape {shape}")

    indices = list(product(range(n_z), range(n_y), range(n_x), range(n_c)))

    return indices


def middle_out_crop_start_index(shape: tuple[int, int, int, int ,int], batch_config : DataConfig) -> tuple[int, int]:
    """
    When cropping batches out of the full Object, we want the crops to be centered about the middle of the volume.
    Since data will be worse when going deeper (or longer in time), for Z and T we align their offsets to the beginning (offset=zero).
    This function returns the offset where the first crop begins. 
    
    Args:
        shape: Object shape
        batch_config: DataConfig object which contains batch shape information and how to handle color channels

    Returns:
        (y0, x0): Pixel offset to achieve middle out crop
    """
    # Object dimensions assumed to be in (N,Z,C,Y,X) format
    n_z, n_c, n_y, n_x = shape

    y0 = (n_y % batch_config.y) // 2
    x0 = (n_x % batch_config.x) // 2

    return y0, x0
