import os
import json
from tqdm import tqdm
from pathlib import Path
from functools import reduce
from itertools import product
from abc import ABC, abstractmethod
from typing import Optional, Any, List, Tuple, Hashable, Sequence,  Dict, Union

import sqlite3
import pandas as pd

from segmentation.data.utils import sql_quote_ident
from segmentation.data.databases.schema import DB_DataSchema, DB_LabelSchema, Comparators


class Database(ABC):

    @abstractmethod
    def _fetch(self, database : str | Path, table: str, data_schema: DB_LabelSchema):
        """
        Return rows whose columns satisfy the query.
        """
        pass

    @abstractmethod
    def _init_db(self, tile_size: List[int], 
                 data_tables: List[DB_DataSchema], 
                 label_table: DB_LabelSchema, 
                 label_task: str):
        """
        Initialize database.
        """
        pass


class SQLiteDatabase(Database):
    """
    SQLite database class.
    """
    
    def __init__(self, 
                db_path: str,
                data_tile: List[int],
                label_tile: List[int],
                label_spec: Union[DB_LabelSchema, Dict],
                data_specs: Union[List[DB_LabelSchema], List[Dict]],
                db_readpath: Optional[str] = None,
                db_savepath: Optional[str] = None,
                db_read_method: Optional[str] = "feather",
                db_save_method: Optional[str] = "feather",
                force_create_db: Optional[bool] = False
    ):  
        self.data_tile = data_tile

        self.data_specs = []
        for data_spec in data_specs:
            if not isinstance(data_spec, DB_DataSchema):
                data_spec = DB_DataSchema(**data_spec)
            self.data_specs.append(data_spec)
        
        if not isinstance(label_spec, DB_LabelSchema):
            label_spec = DB_LabelSchema(**label_spec)
        self.label_spec = label_spec

        self.db_path = db_path
        self.db_savepath = Path(db_savepath) if db_savepath else None
        self.db_readpath = Path(db_readpath) if db_readpath else None
        self.db_save_method = db_save_method

        if self.db_readpath and self.db_readpath.exists():
            if db_read_method == "feather":
                self.data_table = pd.read_feather(self.db_readpath / f"data_table_{label_spec.table}.feather")
                self.label_table = pd.read_feather(self.db_readpath / f"label_table_{label_spec.table}.feather")
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
                data_tables.append(self._fetch(database=db_path, table=data_spec.table, data_schema=data_spec))        
            label_table = self._fetch(database=db_path, table=self.label_spec.table, data_schema=self.label_spec)
            
            self._init_db(data_tile = data_tile,
                          label_tile = label_tile,
                          data_tables = data_tables,
                          label_table = label_table,
                          label_task = self.label_spec.table)

    def _clean_db_dir(self):
        for pattern in ("*.feather", "*.db"):
            for file in self.db_savepath.glob(pattern):
                os.remove(file)

    def _normalize_spec(self, spec: Any) -> Tuple[Optional[Comparators], Any]:
        if spec is None or spec == (None, None):
            return None, None

        # explicit (comp, val) tuple
        if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], Comparators):
            return spec            # already (comp, val)

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

    def _fetch(
        self,
        database:   str | Path,
        table:      str,
        data_schema:  DB_DataSchema
    ) -> pd.DataFrame:
        select_cols: List[str]   = []          
        where_parts: List[str]   = []
        params:      List[Any]   = []

        # data_schema: {col: (comp, value)} where comp is a Comparator
        # and value is the value to compare against
        # e.g. {col: (Comparators.EQUAL, value)} will require col == value
        # in the final SQL query
        for col, raw_spec in data_schema.iter_specs():
            comp, value = self._normalize_spec(raw_spec)
            select_cols.append(sql_quote_ident(col))

            cond, cond_params = self._build_comp_condition(col, comp, value)
            if cond:                                 
                where_parts.append(cond)
                params.extend(cond_params)

        table_sql = sql_quote_ident(table)
        select_sql = ", ".join(select_cols) if select_cols else "*"
        where_sql  = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""

        query = f"SELECT {select_sql} FROM {table_sql}{where_sql}"

        conn = sqlite3.connect(str(database))
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
    
    def _volume_data_tiles(self, shape: tuple[int, int, int, int, int], tile: tuple[int, int, int, int, int]):
        t, c, z, y, x = shape
        tt, tc, tz, ty, tx = tile
        if tt <= 0 or tc <=0 or tz <= 0 or ty <= 0 or tx <= 0:
            raise ValueError(f"Tile size must be positive: {tile}")
        for it, ic, iz, iy, ix in product(range(0, t, tt), range(0, c, tc), range(0, z, tz), range(0, y, ty), range(0, x, tx)):
            yield (
                it, min(it + tt, t),
                ic, min(ic + tc, c),
                iz, min(iz + tz, z),
                iy, min(iy + ty, y),
                ix, min(ix + tx, x),
            )

    def _volume_label_tiles(self, shape: tuple[int, int, int, int], tile: tuple[int, int, int, int]):
        z, y, x = shape
        tz, ty, tx = tile
        if tz <= 0 or ty <= 0 or tx <= 0:
            raise ValueError(f"Tile size must be positive: {tile}")
        for iz, iy, ix in product(range(0, z, tz), range(0, y, ty), range(0, x, tx)):
            yield (
                iz, min(iz + tz, z),
                iy, min(iy + ty, y),
                ix, min(ix + tx, x),
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
        on: Sequence[Hashable] | Hashable = "img_id",
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
        else:
            raise NotImplementedError(f"Label task {label_task} not implemented yet")
    
    def _init_db(self, 
                 data_tile: List[int],
                 label_tile: List[int],
                 label_table: DB_LabelSchema,
                 data_tables: List[DB_DataSchema],  
                 label_task: str
    ):
        data_df = self._merge_tables(data_tables)

        tiled_rows = []
        for _, row in tqdm(data_df.iterrows(), desc="Tiling data table", total=len(data_df)):
            for (t0, t1, c0, c1, z0, z1, y0, y1, x0, x1) in self._volume_data_tiles((row["t"], row["c"], row["z"], row["y"], row["x"]), data_tile):
                new_row = row.to_dict()  
                new_row.update({
                    "t0": t0, "t1": t1,
                    "c0": c0, "c1": c1,
                    "z0": z0, "z1": z1,
                    "y0": y0, "y1": y1,
                    "x0": x0, "x1": x1,
                })
                del new_row["t"], new_row["c"], new_row["z"], new_row["y"], new_row["x"] 
                tiled_rows.append(new_row)

        tiled_df = pd.DataFrame(tiled_rows)

        tiled_label_rows = []
        for _, lrow in tqdm(label_table.iterrows(), desc="Tiling label table", total=len(label_table)):
            for (z0, z1, y0, y1, x0, x1) in self._volume_label_tiles((lrow["z"], lrow["y"], lrow["x"]), label_tile):
                new_lrow = lrow.to_dict()  
                new_lrow.update({
                    "z0": z0, "z1": z1,
                    "y0": y0, "y1": y1,
                    "x0": x0, "x1": x1,
                })
                del new_lrow["z"], new_lrow["y"], new_lrow["x"]
                self._process_label_row(tiled_label_rows, new_lrow, label_task)

        tiled_label_df = pd.DataFrame(tiled_label_rows)

        # save the dataframes to disk for reuse for future training runs
        if self.db_savepath:
            self.db_savepath.mkdir(parents=True, exist_ok=True)

            if self.db_save_method == "feather":
                tiled_df.to_feather(self.db_savepath / f"data_table_{label_task}.feather")
                tiled_label_df.to_feather(self.db_savepath / f"label_table_{label_task}.feather")
            else:
                raise NotImplementedError(f"Save method {self.db_save_method} not implemented yet")

        self.data_table  = tiled_df.reset_index(drop=True)
        self.label_table = tiled_label_df.reset_index(drop=True)