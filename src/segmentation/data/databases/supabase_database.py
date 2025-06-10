import os
import json
from tqdm import tqdm
from pathlib import Path
from functools import reduce
from itertools import product
from typing import Optional, Any, List, Tuple, Hashable, Sequence,  Dict, Union

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

from segmentation.data.utils import sql_quote_ident
from segmentation.data.databases.database import Database
from segmentation.data.databases.schema import DB_DataSchema, DB_LabelSchema, Comparators


class SupabaseDatabase(Database):
    """
    Supabase database class.
    """
    
    def __init__(self, 
                train_task: str,
                dotenv_path: str,
                data_tile: List[int],
                label_tile: List[int],
                label_spec: Union[DB_LabelSchema, Dict],
                data_specs: Union[List[DB_LabelSchema], List[Dict]],
                tile: bool = False,
                with_labels: bool = True,
                db_readpath: Optional[str] = None,
                db_savepath: Optional[str] = None,
                db_read_method: Optional[str] = "feather",
                db_save_method: Optional[str] = "feather",
                force_create_db: Optional[bool] = False
    ):  
        self.train_task = train_task
        self.db_client = self._get_db_client(dotenv_path)

        self.tile = tile
        self.with_labels = with_labels

        self.data_tile = data_tile
        self.label_tile = label_tile

        self.data_specs = []
        for data_spec in data_specs:
            if not isinstance(data_spec, DB_DataSchema):
                data_spec = DB_DataSchema(**data_spec)
            self.data_specs.append(data_spec)
        
        if self.with_labels:
            if not isinstance(label_spec, DB_LabelSchema):
                label_spec = DB_LabelSchema(**label_spec)
            self.label_spec = label_spec
        else:
            self.label_spec = None

        self.db_savepath = Path(db_savepath) if db_savepath else None
        self.db_readpath = Path(db_readpath) if db_readpath else None
        self.db_save_method = db_save_method

        if self.db_readpath and Path(self.db_readpath, f"data_table_{self.train_task}.feather").exists():
            if db_read_method == "feather":
                self.data_table = pd.read_feather(self.db_readpath / f"data_table_{self.train_task}.feather")
                if self.with_labels:
                    self.label_table = pd.read_feather(self.db_readpath / f"label_table_{self.train_task}.feather")
            else:
                raise NotImplementedError(f"Read method {db_read_method} not implemented yet")
        else:
            if self.db_savepath and self.db_savepath.exists():
                if force_create_db:
                    print(f"Removing existing db at {self.db_savepath}")
                    self._clean_db_dir()
                else:
                    raise ValueError(f"Database path {db_savepath} already exists and force_create_db is False")

            self.data_table, self.label_table = None, None

            data_tables = []
            for data_spec in self.data_specs:
                data_tables.append(self._fetch(database=self.db_client, table=data_spec.table, data_schema=data_spec))        
            
            if self.with_labels:
                label_table = self._fetch(database=self.db_client, table=self.label_spec.table, data_schema=self.label_spec)
            else:
                label_table = None

            self._init_db(data_tile = data_tile,
                          label_tile = label_tile,
                          data_tables = data_tables,
                          label_table = label_table,
                          label_task = self.train_task if self.with_labels else None)

    def _get_db_client(self, dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        url: str = os.environ.get("SUPABASE_URL")
        key: str = os.environ.get("SUPABASE_KEY")
        assert url, f"Environment variable 'SUPABASE_URL' is unset or is empty. A local .env file could contain 'SUPABASE_URL=https://XXXXXXXXXXXXXXXXXXXX.supabase.co' and 'SUPABASE_KEY='"
        assert key, f"Environment variable 'SUPABASE_KEY' is not set or is empty. This could be a public key that you find from the 'connect' page on supabase."
        # connect to the database
        db_client: Client = create_client(url, key)
        return db_client

    def _clean_db_dir(self):
        for pattern in ("*.feather", "*.db"):
            for file in self.db_savepath.glob(pattern):
                os.remove(file)

    def _normalize_spec(self, spec: Any) -> Tuple[Optional[Comparators], Any]:
        if spec is None or spec == (None, None):
            return None, None

        # explicit (comp, val) tuple
        if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], Comparators):
            return spec # already (comp, val)

        return Comparators.EQUAL, spec

    def _build_comp_condition(self, col: str, comp: Comparators | None, val: Any):
        ident = sql_quote_ident(col)
        if comp is None:
            return None, []

        match comp:
            case (
                Comparators.EQUAL
                | Comparators.NOT_EQUAL
                | Comparators.GREATER_THAN
                | Comparators.GREATER_THAN_OR_EQUAL
                | Comparators.LESS_THAN
                | Comparators.LESS_THAN_OR_EQUAL
                | Comparators.LIKE
            ):
                return f"{ident} {comp.value} ?", [val]

            case Comparators.IN | Comparators.NOT_IN:
                if not isinstance(val, (list, tuple, set)):
                    raise ValueError(f"{col}: value for {comp.value} must be sequence")
                ph = ", ".join("?" * len(val))
                return f"{ident} {comp.value} ({ph})", list(val)

            case Comparators.BETWEEN:
                if not (isinstance(val, (list, tuple)) and len(val) == 2):
                    raise ValueError(f"{col}: BETWEEN needs (low, high)")
                return f"{ident} BETWEEN ? AND ?", list(val)

            case Comparators.IS_NULL | Comparators.IS_NOT_NULL:
                # no param
                return f"{ident} {comp.value}", []

            case _:
                raise ValueError(f"Unhandled comparator: {comp}")

    def expand_rows_with_zarr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Temporary helper. Given a DataFrame `df` with columns "server_folder" and "output_folder",
        produce a new DataFrame where each original row is repeated once for every
        *.zarr file found under server_folder/output_folder.  A new column "tile_name"
        holds the full path to each .zarr.
        """
        records = []
        for _, row in df.iterrows():
            base_dir = Path(row["server_folder"]) / Path(row["output_folder"])
            if not base_dir.is_dir():
                raise ValueError(f"Expected {base_dir} to be a directory, but it does not exist or is not a directory.")

            zarr_paths = list(base_dir.glob("*.zarr"))
            for zp in zarr_paths:
                new_row = row.to_dict()
                new_row["tile_name"] = zp.name
                records.append(new_row)

        return pd.DataFrame.from_records(records)

    def _fetch(
        self,
        database: Client, 
        table: str,
        data_schema: DB_DataSchema
    ) -> pd.DataFrame:
        select_cols, where_clauses = [], []
        
        for col, raw_spec in data_schema.iter_specs():
            comp, value = self._normalize_spec(raw_spec)
            select_cols.append(col)

            if comp is None:
                continue

            # collect tuple (col, comp, value) for later
            where_clauses.append((col, comp, value))

        select_string = ", ".join(select_cols) if select_cols else "*"
        query = database.table(table).select(select_string)

        for col, comp, value in where_clauses:
            if comp == Comparators.EQUAL:
                query = query.eq(col, value)
            elif comp == Comparators.NOT_EQUAL:
                query = query.neq(col, value)
            elif comp == Comparators.GREATER:
                query = query.gt(col, value)
            elif comp == Comparators.GREATER_EQUAL:
                query = query.gte(col, value)
            elif comp == Comparators.LESS:
                query = query.lt(col, value)
            elif comp == Comparators.LESS_EQUAL:
                query = query.lte(col, value)
            elif comp == Comparators.IN:
                # value must be a list or tuple
                query = query.in_(col, value)
            else:
                raise ValueError(f"Unsupported comparator: {comp}")
            
        response = query.execute()
        data = getattr(response, "data", None)
        if data is None:
            raise RuntimeError(f"Supabase query failed or returned no data for table {table} with conditions: {where_clauses}")
            
        df = pd.DataFrame(data)

        # introducing this hack until we add time and channel size to db
        # just for testing
        df["time_size"] = 20
        df["channel_size"] = 2
        
        # used for now since database stores information per ROI
        # not per tile
        return self.expand_rows_with_zarr(df)

    def _volume_data_tiles(self, shape: tuple[int, int, int, int, int], tile: tuple[int, int, int, int, int]):
        t, c, z_start, z_end, y_start, y_end, x_start, x_end = shape
        tt, tc, tz, ty, tx = tile
        if tt <= 0 or tc <=0 or tz <= 0 or ty <= 0 or tx <= 0:
            raise ValueError(f"Tile size must be positive: {tile}")
        for it, ic, iz, iy, ix in product(range(0, t, tt), range(0, c, tc), range(0, z_end-z_start, tz),
                                           range(0, y_end-y_start, ty), range(0, x_end-x_start, tx)):
            yield (
                it, min(it + tt, t),
                ic, min(ic + tc, c),
                iz, min(iz + tz, z_end),
                iy, min(iy + ty, y_end),
                ix, min(ix + tx, x_end),
            )

    def _volume_label_tiles(self, shape: tuple[int, int, int, int], tile: tuple[int, int, int, int]):
        t, z_start, z_end, y_start, y_end, x_start, x_end = shape
        tt, tz, ty, tx = tile
        if tt <= 0 or tz <= 0 or ty <= 0 or tx <= 0:
            raise ValueError(f"Tile size must be positive: {tile}")
        for it, iz, iy, ix in product(range(0, t, tt), range(0, z_end-z_start, tz),
                                      range(0, y_end-y_start, ty), range(0, x_end-x_start, tx)):
            yield (
                it, min(it + tt, t),
                iz, min(iz + tz, z_end),
                iy, min(iy + ty, y_end),
                ix, min(ix + tx, x_end),
            )

    def _bbox_intersects_tile(self, bbox, tile_bounds):
        bz0, by0, bx0, bz1, by1, bx1 = bbox
        _, tz0, ty0, tx0, _, tz1, ty1, tx1 = tile_bounds
        return not (
            bz1 <= tz0 or bz0 >= tz1 or
            by1 <= ty0 or by0 >= ty1 or
            bx1 <= tx0 or bx0 >= tx1
        )

    def _crop_bbox_to_tile(self, bbox, tile_bounds):
        bz0, by0, bx0, bz1, by1, bx1 = bbox
        _, tz0, ty0, tx0, _, tz1, ty1, tx1 = tile_bounds
        if not self._bbox_intersects_tile(bbox, tile_bounds):
            return None
        return [
            max(0, bz0 - tz0), max(0, by0 - ty0), max(0, bx0 - tx0),
            min(tz1 - tz0, bz1 - tz0), min(ty1 - ty0, by1 - ty0), min(tx1 - tx0, bx1 - tx0)
        ]
    
    def _merge_tables(
        self,
        data_tables: List[pd.DataFrame],
        on: Sequence[Hashable] | Hashable = "id",
    ) -> pd.DataFrame:
        """
        Horizontally combine the data tables so that every column
        from every table ends up in the result.
        """
        missing = [
            i for i, df in enumerate(data_tables)
            if any(c not in df.columns for c in (on if isinstance(on, (list, tuple)) else [on]))
        ]
        if missing:
            raise KeyError(f"Tables at positions {missing} do not contain merge key(s) {on!r}")
        merged = reduce(lambda l, r: pd.merge(l, r, on=on, how="outer"), data_tables)
        return merged.reset_index(drop=True)
    
    def _process_label_row(self, label_rows, label_row, label_task):
        # TODO: update processing instance segmentation task 
        #       once database is populated
        if label_task == "instance_segmentations":
            instance_ids = json.loads(label_row["instance_ids"])
            bboxes = json.loads(label_row["bboxes"])
            tile_indices = [label_row[k] for k in ("t0", "z0","y0","x0","t1","z1","y1","x1")]

            keep_ids, keep_bboxes = [], []
            for lid, box in zip(instance_ids, bboxes):
                new_bbox = self._crop_bbox_to_tile(box, tile_indices)
                if new_bbox is not None:
                    keep_ids.append(lid)
                    keep_bboxes.append(new_bbox)

            if keep_ids:
                label_row["instance_ids"] = json.dumps(keep_ids)
                label_row["bboxes"]    = json.dumps(keep_bboxes)
                label_rows.append(label_row)
        elif label_task == "denoise":
            # for denoise task, we just keep the row as is
            label_rows.append(label_row)
        else:
            raise NotImplementedError(f"Label task {label_task} not implemented yet")
    
    def _init_db(self,
                 data_tile: List[int],
                 label_tile: List[int],
                 label_table: Optional[DB_LabelSchema],
                 data_tables: List[DB_DataSchema],  
                 label_task: str
    ):
        data_df = self._merge_tables(data_tables)

        if self.tile:
            tiled_rows = []
            for _, row in tqdm(data_df.iterrows(), desc="Tiling data table", total=len(data_df)):
                for (t0, t1, c0, c1, z0, z1, y0, y1, x0, x1) in self._volume_data_tiles((row["time_size"], 
                                                                                         row["channel_size"], 
                                                                                         row["z_start"], row["z_end"], 
                                                                                         row["y_start"], row["y_end"],
                                                                                         row["x_start"], row["x_end"]), 
                                                                                         data_tile):
                    new_row = row.to_dict()  
                    new_row.update({
                        "t0": t0, "t1": t1,
                        "c0": c0, "c1": c1,
                        "z0": z0, "z1": z1,
                        "y0": y0, "y1": y1,
                        "x0": x0, "x1": x1,
                    })
                    del new_row["time_size"], new_row["channel_size"], new_row["z_start"], \
                        new_row["z_end"], new_row["y_start"], new_row["y_end"], new_row["x_start"], \
                        new_row["x_end"], 
                    tiled_rows.append(new_row)

            tiled_df = pd.DataFrame(tiled_rows)

            if self.with_labels:
                tiled_label_rows = []
                for _, lrow in tqdm(label_table.iterrows(), desc="Tiling label table", total=len(label_table)):
                    for (t0, t1, z0, z1, y0, y1, x0, x1) in self._volume_label_tiles((lrow["time_size"],
                                                                              lrow["z_start"], lrow["z_end"], 
                                                                              lrow["y_start"], lrow["y_end"],
                                                                              lrow["x_start"], lrow["x_end"]), label_tile):
                        new_lrow = lrow.to_dict()  
                        new_lrow.update({
                            "t0": t0, "t1": t1,
                            "z0": z0, "z1": z1,
                            "y0": y0, "y1": y1,
                            "x0": x0, "x1": x1,
                        })
                        del new_lrow["time_size"], new_lrow["z_start"], new_lrow["z_end"], \
                            new_lrow["y_start"], new_lrow["y_end"], \
                            new_lrow["x_start"], new_lrow["x_end"]
                        self._process_label_row(tiled_label_rows, new_lrow, label_task)

                tiled_label_df = pd.DataFrame(tiled_label_rows)
        else:
            tiled_df = data_df

        # save the dataframes to disk for reuse for future training runs
        if self.db_savepath:
            self.db_savepath.mkdir(parents=True, exist_ok=True)

            if self.db_save_method == "feather":
                tiled_df.to_feather(self.db_savepath / f"data_table_{self.train_task}.feather")
                if self.with_labels:
                    # save label table only if it has data
                    tiled_label_df.to_feather(self.db_savepath / f"label_table_{self.train_task}.feather")
            else:
                raise NotImplementedError(f"Save method {self.db_save_method} not implemented yet")

        self.data_table  = tiled_df.reset_index(drop=True)
        self.label_table = tiled_label_df.reset_index(drop=True) if self.with_labels else pd.DataFrame()