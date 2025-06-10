import re
import os
import json
import random
import collections.abc
from pathlib import Path
from contextlib import nullcontext
from typing import Sequence, Any, Mapping, Tuple, List

import sqlite3
import tifffile
import numpy as np
import tensorstore as ts

import pandas as pd
from skimage.io import imread, imsave

import torch
import torch.fft as fft
from torch.utils.data._utils.collate import \
    default_collate as torch_default_collate

from segmentation.structures.sample_objects.instances import Instances
from segmentation.structures.sample_objects.data_sample import DataSample
from segmentation.structures.data_objects.image_list import ImageList
from segmentation.structures.data_objects.image_list import cat_image_lists


# ---------------------------------------- READ/SAVE FILES ---------------------------------------- #


def read_file(image_path: str | Path, **kwargs) -> str:
    """
    Infer the file format of the image based on its extension.
    """
    image_path = str(image_path)
    if image_path.endswith(".zarr"):
        return read_zarr(image_path, **kwargs)
    elif image_path.endswith(".tiff") or image_path.endswith(".tif"):
        return read_tiff(image_path)
    else:
        raise ValueError(f"Unsupported file format for {image_path}")

def read_zarr(image_path : str | Path, zarr_driver: str = "zarr3", return_handle: bool = False) -> np.ndarray:
    """
    Read a Zarr file and return the data as a NumPy array.
    """
    if isinstance(image_path, Path):
        image_path = str(image_path)
        
    spec = {
        "driver": zarr_driver,
        "kvstore": {"driver": "file", "path": image_path},
    }
    ds = ts.open(spec, read=True).result()
    return ds

def read_tiff(image_path : str) -> np.ndarray:
    """
    Read a TIFF file and return the data as a NumPy array.
    """
    return imread(image_path)

def save_file(image_path: str, data: np.ndarray, **kwargs) -> None:
    """
    Save a NumPy array to a file based on its extension.
    """
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    if image_path.endswith(".zarr"):
        save_zarr(image_path, data, **kwargs)
    elif image_path.endswith(".tiff") or image_path.endswith(".tif"):
        save_tiff(image_path, data)
    else:
        raise ValueError(f"Unsupported file format for {image_path}")

# NOTE: taken from ml-data-platform
def create_zarr_spec(zarr_version, path, data_shape, shard_cube_shape, chunk_shape, num_timepoints_per_image):
    if zarr_version == 'zarr3':
        if len(data_shape) == 5:
            shard_shape = [num_timepoints_per_image, shard_cube_shape[0], shard_cube_shape[1], shard_cube_shape[2], 1]
            chunk_shape = [num_timepoints_per_image, chunk_shape[0], chunk_shape[1], chunk_shape[2], 1]
        elif len(data_shape) == 4:
            shard_shape = [num_timepoints_per_image, shard_cube_shape[0], shard_cube_shape[1], shard_cube_shape[2]]
            chunk_shape = [num_timepoints_per_image, chunk_shape[0], chunk_shape[1], chunk_shape[2]]
        else:
            raise ValueError(f"Unsupported data shape length: {len(data_shape)}")

        zarr_spec = {
            'driver': zarr_version,
            'kvstore': {
                'driver': 'file',
                'path': path
            },
            'metadata': {
                'data_type': 'uint16',
                'shape': data_shape,
                'chunk_grid': {'name': 'regular', 'configuration': {'chunk_shape': shard_shape}},
                'codecs': [{
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": chunk_shape,
                        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}},
                                   {"name": "blosc", "configuration": {
                                       "cname": "zstd", "clevel": 1, "blocksize": 0, "shuffle": "shuffle"}}],
                        "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}],
                        "index_location": "end"
                    }
                }],
                'fill_value': 0,
            },
            'create': True,
            'delete_existing': True
        }
    else:
        zarr_spec = {
            'driver': zarr_version,
            'kvstore': {
                'driver': 'file',
                'path': path
            },
            'metadata': {
                'dtype': '<u2',
                'shape': data_shape,
                'chunks': chunk_shape,
                'compressor': {'blocksize': 0, 'clevel': 1, 'cname': 'zstd', 'id': 'blosc', 'shuffle': 1},
                'fill_value': 0,
                'order': 'C'
            },
            'create': True,
            'delete_existing': True
        }
    return zarr_spec

def save_zarr(image_path: str, 
              data: np.ndarray, 
              shard_cube_shape: Tuple[int, int, int],
              chunk_shape: Tuple[int, int, int],
              zarr_driver: str = "zarr3"
) -> None:
    """
    Save a NumPy array as a Zarr file.
    """
    data_shape, num_timepoints_per_image = data.shape, data.shape[0]
    zarr_spec = create_zarr_spec(zarr_driver, image_path, data_shape,
                                shard_cube_shape, chunk_shape, 
                                num_timepoints_per_image)
    
    ds = ts.open(zarr_spec).result()
    ds[:] = data

def save_tiff(image_path: str, data: np.ndarray) -> None:
    """
    Save a NumPy array as a TIFF file.
    """
    imsave(image_path, data)

def get_shape_from_file_tiff(image_path: str) -> tuple:
    """
    Get the shape of a TIFF file.
    """
    path = Path(image_path)
    with tifffile.TiffFile(str(path)) as tif:
        # series[0] is the first image series (e.g. the main image)
        # .shape might be (Z,Y,X), (C,Z,Y,X), (T,Z,Y,X) or (T,C,Z,Y,X), etc.
        return tif.series[0].shape


# ---------------------------------------- DATABASE QUERIES FOR LABEL GENERATION ---------------------------------------- #


# TODO: migrate utils to label_generation repo.


_COLUMN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def sql_quote_ident(identifier: str) -> str:
    if not _COLUMN_RE.match(identifier):
        raise ValueError(f"Illegal SQL identifier: {identifier!r}")
    return f'"{identifier}"'

def get_db_query_pair(mapping: Mapping[str, Any]):
    cols, params = [], []
    for col, val in mapping.items():
        cols.append(sql_quote_ident(col))
        if isinstance(val, (list, dict)):
            params.append(json.dumps(val))
        elif isinstance(val, Path):
            params.append(str(val))
        else:
            params.append(val)
    return cols, params

def read_database_query(db_path: str | Path, db_type: str, 
                        table: str, selectors: Mapping[str, Any] = {}
) -> pd.DataFrame:
    """
    Read from database and return a DataFrame.
    """
    selectors = selectors or {}

    if db_type == "sqlite":
        cols, vals = get_db_query_pair(selectors)
        where_sql = ""
        if cols:
            where_sql = " WHERE " + " AND ".join(f"{col}=?" for col in cols)

        table_sql = sql_quote_ident(table)
        query = f"SELECT * FROM {table_sql}{where_sql}"

        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(query, conn, params=vals)
        conn.close()
    elif db_type == "csv":
        df = pd.read_csv(db_path)
        # apply selectors in-memory
        for key, val in selectors.items():
            df = df[df[key] == val]
    elif db_type == "json":
        df = pd.read_json(db_path)
        # apply selectors in-memory
        for key, val in selectors.items():
            df = df[df[key] == val]
    else:
        raise ValueError(f"Unsupported acquisition database type: {db_type!r}")
    return df

def strip_key(s : str) -> str:
    """
    Strip quotes from a string.
    """
    return s.replace("'", "").replace('"', '').replace(" ", "_").replace(":", "_")

def print_db(db_path: str | Path, db_type: str = "sqlite") -> None:
    """
    Print the contents of a database.
    """
    if db_type == "sqlite":
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        for (table_name,) in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
            print(f"\n── {table_name} ────────────────────────────────────────────")
            rows = conn.execute(f"SELECT * FROM {table_name}").fetchall()
            if not rows:
                print("(no rows)")
                continue
            # print column headers
            headers = rows[0].keys()
            print("\t".join(headers))
            for r in rows:
                print("\t".join(str(r[h]) for h in headers))

        conn.close()
    else:
        raise ValueError(f"Unsupported database type: {db_type}")
    

# ---------------------------------------- DATA LOADING ---------------------------------------- #


# from: https://github.com/ray-project/ray/python/ray/train/torch/train_loop_utils.py
def move_to_device(
    item,
    device: torch.device | str = "cuda",
    *,
    auto_transfer: bool = False,
    stream=None,
    stream_context=nullcontext,
):
    """
    Recursively move `item` to `device`.
    """
    if item is None:
        return None

    def _to(t: torch.Tensor):
        return t.to(device, non_blocking=auto_transfer)

    with stream_context(stream):
        if isinstance(item, collections.abc.Mapping):
            return {k: move_to_device(v, device,
                                      auto_transfer=auto_transfer,
                                      stream=stream,
                                      stream_context=stream_context)
                    for k, v in item.items()}

        if isinstance(item, tuple):
            return tuple(
                move_to_device(v, device,
                               auto_transfer=auto_transfer,
                               stream=stream,
                               stream_context=stream_context)
                for v in item
            )

        if isinstance(item, list):
            return [
                move_to_device(v, device,
                               auto_transfer=auto_transfer,
                               stream=stream,
                               stream_context=stream_context)
                for v in item
            ]

        if isinstance(item, torch.Tensor):
            return _to(item)

        # anything else is left untouched
        return item

def worker_init_fn(worker_id: int,
                   num_workers: int,
                   rank: int,
                   seed: int
) -> None:
    """
    This function will be called on each worker subprocess after seeding and
    before data loading.

    Args:
        worker_id (int): Worker id in [0, num_workers - 1].
        num_workers (int): How many subprocesses to use for data loading.
        rank (int): Rank of process in distributed environment. If in
            non-distributed environment, it is a constant number `0`.
        seed (int): Random seed.
    """
    # The seed of each worker equals:
    # num_worker * rank + worker_id + user_seed
    # we set numpy seed since any custom ops. using
    # numpy random functions should be seeded with
    # a unique seed which is not set by default
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def collate_instance_segmentation(samples: list["DataSample"]) -> "DataSample":
    """Stack images & gather instances into a batched DataSample."""
    metainfo = default_collate([s.metainfo for s in samples])
    batched_img = cat_image_lists(image_lists=[s.data_tensor for s in samples])
    inst_list = [s.gt_instances for s in samples]                        

    batch = DataSample(metainfo=metainfo)  
    batch.data_tensor = batched_img
    batch.gt_instances = inst_list
    return batch.to_dict()

# TODO: merge collate_fns that are identical
def collate_channel_split(samples: list["DataSample"]) -> "DataSample":
    metainfo = default_collate([s.metainfo for s in samples])
    batch = DataSample(metainfo=metainfo)
    
    batched_img = cat_image_lists(image_lists=[s.data_tensor for s in samples])
    batch.data_tensor = batched_img

    instance = Instances()
    instance.image = cat_image_lists(image_lists=[s.gt_instances.image for s in samples])
    batch.gt_instances = instance

    return batch.to_dict()

def collate_upsample(samples: list["DataSample"]) -> "DataSample":
    metainfo = default_collate([s.metainfo for s in samples])
    batch = DataSample(metainfo=metainfo)
    
    batched_img = cat_image_lists(image_lists=[s.data_tensor for s in samples])
    batch.data_tensor = batched_img

    instance = Instances()
    instance.image = cat_image_lists(image_lists=[s.gt_instances.image for s in samples])
    batch.gt_instances = instance

    return batch.to_dict()

def collate_denoise(samples: list["DataSample"]) -> "DataSample":
    metainfo = default_collate([s.metainfo for s in samples])
    batch = DataSample(metainfo=metainfo)
    
    batched_img = cat_image_lists(image_lists=[s.data_tensor for s in samples])
    batch.data_tensor = batched_img

    instance = Instances()
    instance.image = cat_image_lists(image_lists=[s.gt_instances.image for s in samples])
    batch.gt_instances = instance

    return batch.to_dict()

def collate_channel_predict(samples: list["DataSample"]) -> "DataSample":
    metainfo = default_collate([s.metainfo for s in samples])
    batch = DataSample(metainfo=metainfo)
    
    batched_img = cat_image_lists(image_lists=[s.data_tensor for s in samples])
    batch.data_tensor = batched_img

    instance = Instances()
    instance.image = cat_image_lists(image_lists=[s.gt_instances.image for s in samples])
    batch.gt_instances = instance
    
    return batch.to_dict()

# from: https://github.com/open-mmlab/mmengine/main/mmengine/dataset/utils.py
def default_collate(data_batch: Sequence) -> Any:
    """
    Convert list of data sampled from dataset into a batch of data, of which
    type consistent with the type of each element in ``data_batch``.

    Args:
        data_batch (Sequence): Data sampled from dataset.

    Returns:
        Any: Data in the same format as the data element of ``data_batch``, of which
        tensors have been stacked, and ndarray, int, float have been
        converted to tensors, etc.
    """  
    # NOTE: we assume the each data element in data_batch
    #       is of the same type
    data_item = data_batch[0]
    data_item_type = type(data_item)

    # recursive collate
    if isinstance(data_item, (str, bytes)):
        return data_batch
    elif isinstance(data_item, tuple) and hasattr(data_item, '_fields'):
        # named tuple
        # we transpose the batch to get a tuple of lists
        # recursively collate each list, then rebuild same
        # named tuple type
        return data_item_type(*(default_collate(samples)
                                for samples in zip(*data_batch)))
    elif isinstance(data_item, Sequence):
        # check to make sure that the elements in batch have
        # consistent size
        it = iter(data_batch)
        data_item_size = len(next(it))
        if not all(len(data_item) == data_item_size for data_item in it):
            raise RuntimeError(
                'each data_itement in list of batch should be of equal size')
        
        # from [(a, b), (c, d)] to [(a, c), (b, d)]
        transposed = list(zip(*data_batch))

        # from [(a, c), (b, d)] to [collated[a, c], collated[b, d]]
        if isinstance(data_item, tuple):
            return [default_collate(samples)
                    for samples in transposed]  # Compat with Pytorch
        else:
            try:
                return data_item_type(
                    [default_collate(samples) for samples in transposed])
            except TypeError:
                # sequence type may not support `__init__(iterable)`
                # (e.g., `range`). Fall back to list.
                return [default_collate(samples) for samples in transposed]
    elif isinstance(data_item, Mapping):
        # [{"img": img1, "label": label1},
        #  {"img": img2, "label": label2}]
        # to {"img": collate[img1, img2], "label": collate[label1, label2]}
        return data_item_type({
            key: default_collate([d[key] for d in data_batch])
            for key in data_item
        })
    else:
        return torch_default_collate(data_batch)


# ---------------------------------------- FINETUNING TASK HELPERS  ---------------------------------------- #


# from https://github.com/cell-observatory/aovift/src/synthetic.py
@torch.no_grad()
def create_na_masks(ipsf: torch.Tensor, thresholds: List[float]) -> torch.Tensor:
    """
    OTF Mask by binary thresholding ideal theoretical OTF.

    Args:
        thresholds: where to threshold after normalizing to the OTF max

    Returns:
        3D array where == 1 inside NA_Mask, == 0 outside NA mask

    """
    ipsf = torch.as_tensor(ipsf, dtype=torch.float32)
    if ipsf.ndim != 3:
        raise ValueError(f"ipsf must be 3-D, got shape {ipsf.shape}")
    
    otf = torch.abs(fft(ipsf))

    # NaN safe max operator
    max_val = torch.where(torch.isnan(otf), otf.new_full((), float("-inf")), otf).max()
    if not torch.isfinite(max_val):
        raise ValueError("OTF is all-NaN — cannot build NA mask")

    # max normalize
    mask = otf / max_val
    
    masks = []
    for thr in thresholds:
        if not (0.0 <= thr <= 1.0):
            raise ValueError(f"Threshold {thr} outside [0,1]")
        # keep magnitudes >= threshold
        binary_mask = (mask >= thr).float()
        masks.append(binary_mask)
        
    return torch.stack(masks)

def fft(image):
    fft_image = torch.fft.ifftshift(image)
    fft_image = torch.fft.fftn(fft_image)
    fft_image = torch.fft.fftshift(fft_image)
    return fft_image

def ifft(fft_image):
    image = torch.fft.fftshift(fft_image)
    image = torch.fft.ifftn(image)
    image = torch.abs(torch.fft.ifftshift(image))
    return image


# ---------------------------------------- --- ---- ---- ---- ----  ---------------------------------------- #