import os
import logging
import sqlite3
import warnings

from sqlite3 import Connection, Cursor
from typing import Optional, Callable, Dict

from tqdm import tqdm
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from skimage.measure import regionprops

import tifffile

from tensorstore import TensorStore

from segmentation.data.data_utils import index_mapper, middle_out_crop_start_index, ColorMode, DataConfig

import torch.distributed as dist


class Skittlez_Database:
    """
    Access the preprocessed dataset and metadata.
    """
    def __init__(self,
                 db_path = None,
                 train_db_savedir = None,
                 batch_config: DataConfig = None,
                 transforms: Optional[Callable] = None,
                 selection_criteria: Dict = None,
                 force_create_db = False,   
                 clean_up_db = False,
                 metadata = None,
                 dtype = np.uint16,
                 with_zarr = False,
                 with_tiff = True,
                 training: bool = True,
                 distributed: bool = True,
                 ):
        """
        Skittlez Dataset.
        
        Args:
            batch_config: DataConfig object contains the shape information for a single batch. If None, default will be used.
            force_create_db: Flag that determines whether the cached local sqlite3 database should be created (True) or reused (False). Default is False.
            clean_up_db: Flag that determines whether the cached local sqlite3 database should be deleted in the FishDatabase destructor. Default is False.
            metadata: dict that contains metadata. Required keys are ["created_at", "output_folder", "exists"]. Default behavior is to query this data from the remote database.
            dtype: Numpy dtype that determines the output format. Default is np.uint16.
            with_zarr: Flag that determines whether database uses zarr files. Default is False.
            with_tiff: Flag that determines whether database uses tiff files. Default is True.
            db_path: Path to the database. Default is None.
            transforms: Optional transforms to be applied to the data. Default is None.
        """
        # Construct default DataConfig if batch_config is not provided (defaults to color_mode = "AVG")
        if batch_config is None:
            warnings.warn("Batch config is not provided. Defaulting to color_mode = 'AVG'")
            batch_config = DataConfig()

        self.batch_config = batch_config
        self.selection_criteria = selection_criteria

        self.force_create_db = force_create_db
        self.clean_up_db = clean_up_db

        self.with_zarr = with_zarr
        self.with_tiff = with_tiff
        self.db_path = db_path

        self.train_db_savedir = train_db_savedir
        os.makedirs(self.train_db_savedir, exist_ok=True)

        self.transforms = transforms
        self.training = "train" if training else "test"

        # instantiate fields that will be populated later for bookkeeping
        self.con: Connection = None
        self.cur: Cursor = None
        self.local_db_name: str = None
        self.stores:list[TensorStore] = []    # each store is roughly an experiment
        self.file_paths: list[Path] = []  # each file path is a tiff file
        self.label_files: list[Path] = []  # each file path is a tiff file
        self.length: int = 0
        self.dtype: np.dtype = dtype

        # Query metadata if not provided (Pandas provides a uniform metadata interface)
        if metadata is None:
            metadata = self._query_db(self.db_path)
        else:
            metadata = pd.DataFrame(metadata)
        self.metadata = metadata

        # return if no metadata
        if len(self.metadata) == 0:
            return

        # TODO: unify current skittles database logic with pre-training
        # database logic & move to zarr
        #############################################################################
        # # Query metadata if not provided
        # # Pandas provides a uniform metadata interface
        # if metadata is None:
        #     metadata = self._query_remote_db()
        # else:
        #     metadata = pd.DataFrame(metadata)
        # self.metadata = metadata

        # # return if no metadata
        # if len(self.metadata) == 0:
        #     return

        # # check required metadata fields exist
        # required_fields = ["created_at", "output_folder", "exists"]
        # for field in required_fields:
        #     if field not in self.metadata:
        #         raise ValueError(f"Metadata required fields are missing: {required_fields}")

        # # Sorted using record creation time. When indexing into each store or slice within a store, time
        # # should always be increasing. Doing this consistently will minimize temporal data leakage.
        # self.metadata = pd.DataFrame(self.metadata)
        # self.metadata = self.metadata.sort_values(by='created_at')
        #############################################################################

        # Open all the zarr/tiff files provided in the metadata. If a file does not exist, it is skipped.
        if self.with_zarr:
            self._open_zarr_files()
        elif self.with_tiff:
            self._open_tiff_files()
        else:
            raise ValueError("Invalid file format. Only zarr and tiff are supported.")

        # A local sqlite3 database is created with tables to hold index mapping and slicing information.
        if distributed:
            self._init_local_db_distributed()
        else:
            self._init_local_db()

        if self.con:
            self.con.close()
            self.con = None
            self.cur = None


    def _query_db(self, db_path: Path) -> pd.DataFrame:
        # Connect to SQLite database
        conn = sqlite3.connect(str(db_path))

        query = """
            SELECT file_path, filename, mask_type_synthetic_data, experiment_name, fish_name, roi, tile
            FROM annotations 
            WHERE train_val_test = ? AND cp_grade = '1' AND testing = '1' 
        """
        metadata = pd.read_sql_query(query, conn, params=[self.training])

        conn.close()
        return metadata

    # TODO: This is hacky, need to unify DB logic with pre-training database logic (but using this for now)
    def _open_tiff_files(self):        
        image_files = []
        label_files = []
        for _, row in tqdm(self.metadata.iterrows(), desc="Loading tiff files", total=len(self.metadata)):
            file_path = Path(row['file_path']).parent
            filename = row['filename']
            mask_type = row['mask_type_synthetic_data']

            image_path = file_path / 'syn_rgb' / filename

            if mask_type == 'CP':
                label_path = file_path / 'masks_cp' / filename
            elif mask_type == 'PL':
                label_path = file_path / 'masks_plantseg' / filename
            else:
                raise ValueError(f"Unknown mask type: {mask_type}")

            if image_path.exists() and label_path.exists():
                image_files.append(image_path)
                label_files.append(label_path)
        
        self.file_paths = image_files
        self.label_files = label_files

    
    def _create_local_db_tables(self):
        # Create store_index_map table
        cmd = """
            CREATE TABLE store_index_map (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path STRING,
                label_path STRING,
                z INTEGER,
                y INTEGER,
                x INTEGER,
                c INTEGER
            );
        """
        self.cur.execute(cmd)

        # Create bbox table
        cmd =  """
            CREATE TABLE bboxes (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                crop_id INTEGER,
                x1 INTEGER,
                y1 INTEGER,
                z1 INTEGER,
                x2 INTEGER,
                y2 INTEGER,
                z2 INTEGER,
                FOREIGN KEY (crop_id) REFERENCES store_index_map(rowid)
            );
        """
        self.cur.execute(cmd)

        # Create masks table
        cmd = """
            CREATE TABLE masks (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                crop_id INTEGER,
                mask_id INTEGER,
                FOREIGN KEY (crop_id) REFERENCES store_index_map(rowid)
            );
        """
        self.cur.execute(cmd)

        # Create middle_out_table with one-to-many support
        cmd = """
            CREATE TABLE middle_out_table (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                crop_id INTEGER UNIQUE,
                y0 INTEGER,
                x0 INTEGER,
                FOREIGN KEY (crop_id) REFERENCES store_index_map(rowid)
            );
        """
        self.cur.execute(cmd)


    # TODO: join distributed and non-distributed code to prevent
    #       code duplication
    def _init_local_db(self):
        local_db_name = os.path.join(self.train_db_savedir, self.training + "_" + repr(self.batch_config) + ".db")
        self.local_db_name = local_db_name

        # if force creating db, we need to delete the existing one first
        if self.force_create_db and os.path.isfile(local_db_name):
            os.remove(local_db_name)

        # check db exists before .connect since it would create the db if it didn't
        create_db = not os.path.isfile(local_db_name)

        logging.info(f"[SkittlesDatabase] Local db file {local_db_name} exists: {not create_db}")

        self.con = sqlite3.connect(local_db_name)
        self.cur = self.con.cursor()

        if create_db or self.force_create_db:
            logging.info(f"[SkittlesDatabase] Creating local db file {local_db_name}...")

            self._create_local_db_tables()

            for file_path, label_path in tqdm(zip(self.file_paths, self.label_files), desc="Updating DB with target info...", total=len(self.file_paths)):
                label_item = tifffile.memmap(label_path)
                data_item_shape = tifffile.memmap(file_path).shape
                indices = index_mapper(data_item_shape, self.batch_config)

                if indices is None:
                    continue

                y0, x0 = middle_out_crop_start_index(data_item_shape, self.batch_config)

                valid_indices, bboxes, mask_ids = self.indices_to_instances(indices, y0, x0, label_item)

                for idx, bbox_list, mask_id_list in zip(valid_indices, bboxes, mask_ids):
                    # store_index_map db update
                    z, y, x, c = idx
                    cmd = "INSERT INTO store_index_map(file_path, label_path, z, y, x, c) VALUES (?, ?, ?, ?, ?, ?)"
                    self.cur.execute(cmd,(str(file_path), str(label_path), int(z), int(y), int(x), int(c)))
                    crop_id = self.cur.lastrowid  # get rowid of inserted crop

                    # bboxes db update
                    for bbox in bbox_list:
                        x1, y1, z1, x2, y2, z2 = bbox
                        cmd = "INSERT INTO bboxes(crop_id, x1, y1, z1, x2, y2, z2) VALUES (?, ?, ?, ?, ?, ?, ?)"
                        self.cur.execute(cmd, (int(crop_id), int(x1), int(y1), int(z1), int(x2), int(y2), int(z2)))

                    # masks db update
                    for mask_id in mask_id_list:
                        cmd = "INSERT INTO masks(crop_id, mask_id) VALUES (?, ?)"
                        self.cur.execute(cmd, (int(crop_id), int(mask_id)))

                    # middle_out_table db update
                    self.cur.execute(
                        "INSERT INTO middle_out_table(crop_id, y0, x0) VALUES (?, ?, ?)",
                        (int(crop_id), int(y0), int(x0))
                    )
            # need to commit the changes to the database for workers in dataloader to see them        
            self.con.commit()

        # update dataset length
        res = self.cur.execute("SELECT COUNT(*) FROM store_index_map")
        self.length = res.fetchone()[0]

        logging.info(f"[SkittlesDatabase] Found {self.length} crops in the database.")


    def _init_local_db_distributed(self):
        local_db_name = os.path.join(self.train_db_savedir, self.training + "_" + repr(self.batch_config) + ".db")
        self.local_db_name = local_db_name

        # if force creating db, we need to delete the existing one first
        if self.force_create_db and os.path.isfile(local_db_name):
            os.remove(local_db_name)

        # check db exists before .connect since it would create the db if it didn't
        create_db = not os.path.isfile(local_db_name)

        logging.info(f"[SkittlesDatabase] Local db file {local_db_name} exists: {not create_db}")

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if rank == 0:
            self.con = sqlite3.connect(local_db_name)
            self.cur = self.con.cursor()

            if create_db or self.force_create_db:
                logging.info(f"[SkittlesDatabase] Creating local db file {local_db_name}...")

                self._create_local_db_tables()

                for file_path, label_path in tqdm(zip(self.file_paths, self.label_files), desc="Updating DB with target info...", total=len(self.file_paths)):
                    label_item = tifffile.memmap(label_path)
                    data_item_shape = tifffile.memmap(file_path).shape
                    indices = index_mapper(data_item_shape, self.batch_config)

                    if indices is None:
                        continue

                    y0, x0 = middle_out_crop_start_index(data_item_shape, self.batch_config)

                    valid_indices, bboxes, mask_ids = self.indices_to_instances(indices, y0, x0, label_item)

                    for idx, bbox_list, mask_id_list in zip(valid_indices, bboxes, mask_ids):
                        # store_index_map db update
                        z, y, x, c = idx
                        cmd = "INSERT INTO store_index_map(file_path, label_path, z, y, x, c) VALUES (?, ?, ?, ?, ?, ?)"
                        self.cur.execute(cmd,(str(file_path), str(label_path), int(z), int(y), int(x), int(c)))
                        crop_id = self.cur.lastrowid  # get rowid of inserted crop

                        # bboxes db update
                        for bbox in bbox_list:
                            x1, y1, z1, x2, y2, z2 = bbox
                            cmd = "INSERT INTO bboxes(crop_id, x1, y1, z1, x2, y2, z2) VALUES (?, ?, ?, ?, ?, ?, ?)"
                            self.cur.execute(cmd, (int(crop_id), int(x1), int(y1), int(z1), int(x2), int(y2), int(z2)))

                        # masks db update
                        for mask_id in mask_id_list:
                            cmd = "INSERT INTO masks(crop_id, mask_id) VALUES (?, ?)"
                            self.cur.execute(cmd, (int(crop_id), int(mask_id)))

                        # middle_out_table db update
                        self.cur.execute(
                            "INSERT INTO middle_out_table(crop_id, y0, x0) VALUES (?, ?, ?)",
                            (int(crop_id), int(y0), int(x0))
                        )
                # need to commit the changes to the database for workers in dataloader to see them        
                self.con.commit()

            # update dataset length
            res = self.cur.execute("SELECT COUNT(*) FROM store_index_map")
            self.length = res.fetchone()[0]

            logging.info(f"PROCESS {rank}: [SkittlesDatabase] Found {self.length} crops in the database.")

        # all processes wait here until db is fully written
        # only process 0 will create the db to prevent race conditions
        if world_size > 1:
            dist.barrier()
        
        if rank != 0:
            self.con = sqlite3.connect(local_db_name)
            self.cur = self.con.cursor()
            res = self.cur.execute("SELECT COUNT(*) FROM store_index_map")
            self.length = res.fetchone()[0]
            logging.info(f"PROCESS {rank}: [SkittlesDatabase] Found {self.length} crops in the database.")

    # TODO: A band-aid solution. Redo once dataset is updated. Trainig is bottlenecked by dataloading 
    #       for larger batch size currently.
    def indices_to_instances(self, indices, y0, x0, label_item):
        valid_indices = []
        bboxes = []
        mask_ids = []
        for (z, y, x, c) in indices:
            bboxes_crop = []
            mask_ids_crop = []

            # compute index slices
            z1, z2 = z * self.batch_config.z, (z + 1) * self.batch_config.z
            y1, y2 = y * self.batch_config.y, (y + 1) * self.batch_config.y
            x1, x2 = x * self.batch_config.x, (x + 1) * self.batch_config.x

            label_crop = label_item[slice(z1,z2), slice(y1+y0, y2+y0), slice(x1+x0, x2+x0)]
            if not np.any(label_crop):
                continue

            # Regionprops to get bounding boxes (min_z, min_y, min_x, max_z, max_y, max_x)
            props = regionprops(label_crop)
            for p in props:
                min_z, min_y, min_x, max_z, max_y, max_x = p.bbox
                volume = p.area
                if (max_z > min_z) and (max_y > min_y) and (max_x > min_x) and volume > self.selection_criteria["volume"]:
                    bboxes_crop.append([min_x, min_y, min_z, max_x, max_y, max_z])
                    mask_ids_crop.append(p.label)

            if len(bboxes_crop) > 0:
                valid_indices.append((z, y, x, c))
                bboxes.append(bboxes_crop) # list of list of lists (image -> crop -> bboxes)
                mask_ids.append(mask_ids_crop) # list of list of ints (image -> crop -> mask ids)
        
        return valid_indices, bboxes, mask_ids

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index >= self.length:
            raise IndexError
        
        # look up corresponding indices (note SQL uses 1-based indexing for autoincremented row id)
        cmd = "SELECT * FROM store_index_map WHERE rowid = ?"
        res = self.cur.execute(cmd, (index + 1,))
        resp = res.fetchone()
        if resp is None:
            raise IndexError(f"No entry found in store_index_map for index {index}")
        
        rowid, file_path, label_path, z, y, x, c = resp

        # get pixel offset values for y and x
        cmd = "SELECT y0, x0 FROM middle_out_table WHERE crop_id = ?"
        res = self.cur.execute(cmd, (rowid,))
        resp = res.fetchone()
        if resp is None:
            raise IndexError(f"No entry found in middle_out_table for crop_id {rowid}")
        
        y0, x0 = resp

        # compute index slices
        z1, z2 = z * self.batch_config.z, (z + 1) * self.batch_config.z
        y1, y2 = y * self.batch_config.y, (y + 1) * self.batch_config.y
        x1, x2 = x * self.batch_config.x, (x + 1) * self.batch_config.x

        # slice data based on color mode (TODO: move to zarr)
        if self.batch_config.color_mode == ColorMode.MATCH:
            # TODO: add support for other operations, pass c1, c2 in batch_config
            c1, c2 = 0, self.batch_config.c
            data_item = tifffile.imread(file_path)
            data_item = data_item[slice(z1, z2), slice(c1, c2), slice(y1 + y0, y2 + y0), slice(x1 + x0, x2 + x0)]
            label_item = tifffile.imread(label_path)
            label_item = label_item[slice(z1, z2), slice(y1 + y0, y2 + y0), slice(x1 + x0, x2 + x0)]
            # item = store[tile, t1:t2, z1:z2, y1+y0:y2+y0, x1+x0:x2+x0, c1:c2].read().result()
        elif self.batch_config.color_mode == ColorMode.AVG:
            #NOTE: This is not actually supported in yet
            data_item = tifffile.imread(file_path)
            data_item = data_item[slice(z1, z2), :, slice(y1 + y0, y2 + y0), slice(x1 + x0, x2 + x0)]
            label_item = tifffile.imread(label_path)
            label_item = label_item[slice(z1, z2), slice(y1 + y0, y2 + y0), slice(x1 + x0, x2 + x0)]
            # item = store[tile, t1:t2, z1:z2, y1+y0:y2+y0, x1+x0:x2+x0, :].read().result()
            if data_item.shape[1] > 1:
                # cast to double (implicit) before averaging
                data_item = data_item.mean(1)
                # cast and reshape to original
                data_item = data_item[:, np.newaxis, ...]
        else:
            raise NotImplementedError(f"Color mode {self.batch_config.color_mode} not implemented")

        # load labels, boxes, masks
        bbox_cmd = "SELECT x1, y1, z1, x2, y2, z2 FROM bboxes WHERE crop_id = ?"
        res_bboxes = self.cur.execute(bbox_cmd, (rowid,))
        bboxes = res_bboxes.fetchall() # list of (x1, y1, z1, x2, y2, z2)

        mask_id_cmd = "SELECT mask_id FROM masks WHERE crop_id = ?"
        res_mask_ids = self.cur.execute(mask_id_cmd, (rowid,))
        mask_ids = [row[0] for row in res_mask_ids.fetchall()]  # flatten to list of ints

        masks, labels = self.mask_ids_to_masks(mask_ids, label_item)

        data_item = torch.tensor(data_item, dtype=self.dtype).permute(1, 0, 2, 3) # data_item: (Z, C, Y, X) -> (C, Z, Y, X)
        label_item = {"boxes": torch.tensor(bboxes, dtype=torch.uint16), 
                      "labels": torch.tensor(labels, dtype=torch.int64),
                      "masks": torch.tensor(masks, dtype=torch.uint8)}
        
        # import skimage
        # from segmentation.utils.plot import plot_boxes
        # from ray.train import get_context
        # if get_context().get_world_rank() == 0:
        #     print(f"SIZE OF IMAGE: {data_item.shape}")
        #     box_dl = [label_item["boxes"][i].cpu().numpy() for i in range(len(label_item["boxes"]))]
        #     plot_boxes(box_dl, sample_indices=[0], image_shape=data_item.shape[1:], save_path="/clusterfs/nvme/segment_4d/test_5/box_b_transform.tif")        
        #     skimage.io.imsave("/clusterfs/nvme/segment_4d/test_5/masks_b_transform.tif", label_item["masks"][0].cpu().numpy())
        #     skimage.io.imsave("/clusterfs/nvme/segment_4d/test_5/im_b_transform.tif", data_item[0].cpu().numpy())
        
        if self.transforms:
            item = self.transforms(data_item, label_item)

        # if get_context().get_world_rank() == 0:
        #     box_dl = [label_item["boxes"][i].cpu().numpy() for i in range(len(label_item["boxes"]))]
        #     plot_boxes(box_dl, sample_indices=[0], image_shape=data_item.shape[1:], save_path="/clusterfs/nvme/segment_4d/test_5/box_after_transform.tif")        
        #     skimage.io.imsave("/clusterfs/nvme/segment_4d/test_5/masks_after_transform.tif", label_item["masks"][0].cpu().numpy())
        #     skimage.io.imsave("/clusterfs/nvme/segment_4d/test_5/im_after_transform.tif", data_item[0].cpu().numpy())

        return item 

    def mask_ids_to_masks(self, mask_ids, label_item):
        binary_masks = [(label_item == mid).astype(np.uint8) for mid in mask_ids]
        return np.stack(binary_masks, axis=0), np.ones(len(mask_ids), dtype=np.int64)

    def __del__(self):
        if self.con:
            self.con.close()

        if self.clean_up_db and self.local_db_name is not None:
            try:
                os.remove(self.local_db_name)
            except Exception as e:
                logging.info(f'Failed to delete local db file: {self.local_db_name}.')
