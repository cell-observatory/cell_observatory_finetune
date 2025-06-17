import os
import json
from tqdm import tqdm
from pathlib import Path
from functools import reduce
from typing import Optional, Any, List, Tuple, Hashable, Sequence

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

from finetune.data.utils import sql_quote_ident
from finetune.data.databases.database import Database
from finetune.data.databases.schema import TableSchemaList, Comparators, DB_TableSchema


class SupabaseDatabase(Database):
    """
    Supabase database class.
    """
    
    def __init__(self, 
                dotenv_path: str,
                table_specs: TableSchemaList,
                db_readpath: Optional[str] = None,
                db_savepath: Optional[str] = None,
                db_read_method: Optional[str] = "feather",
                db_save_method: Optional[str] = "feather",
                load_cached_db: Optional[bool] = False,
                force_create_db: Optional[bool] = False,
                data_cubes_table: str = "prepared_cubes_1x128x128x128x1",
                label_cubes_table: Optional[str] = None, 
    ):  
        self.db_client = self._get_db_client(dotenv_path)
        
        self.table_specs = table_specs

        self.data_cubes_table = data_cubes_table
        self.label_cubes_table = label_cubes_table

        self.db_savepath = Path(db_savepath) if db_savepath else None
        self.db_readpath = Path(db_readpath) if db_readpath else None
        
        self.db_save_method = db_save_method

        if load_cached_db:
            if db_read_method == "feather":
                assert Path(self.db_readpath).exists(), \
                    f"Database read path {self.db_readpath} does not exist. " \
                    "Please set `load_cached_db` to False to create a new database."
                self.metadata_table = pd.read_feather(self.db_readpath)
            else:
                raise NotImplementedError(f"Read method {db_read_method} not implemented yet")
        else:
            if self.db_savepath is not None and self.db_savepath.exists():
                if force_create_db:
                    print(f"Removing existing db at {self.db_savepath}")
                    self._clean_db_dir()
                else:
                    raise ValueError(f"Database path {db_savepath} already exists and force_create_db is False")

            self._fetch(database=self.db_client, table_list=self.table_specs)

    def _get_db_client(self, dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        url: str = os.environ.get("SUPABASE_URL")
        key: str = os.environ.get("SUPABASE_KEY")
        assert url, f"Environment variable 'SUPABASE_URL' is unset or \
        is empty. A local .env file could contain \
        'SUPABASE_URL=https://XXXXXXXXXXXXXXXXXXXX.supabase.co' \
        and 'SUPABASE_KEY='"
        assert key, f"Environment variable 'SUPABASE_KEY' is not \
            set or is empty. This could be a public key that you find \
            from the 'connect' page on supabase."
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

    def _fetch(
        self,
        database: Client, 
        table_list: Sequence[DB_TableSchema],
    ) -> pd.DataFrame:
        # fetch data from tables from Supabase database
        # for each table we optionally treshold the columns
        # with where clauses, all tables share a common 
        # prepared_id that will be used to merge them
        # this all works on the roi level
        df_tables = []
        for table_schema in table_list:
            select_cols, where_clauses = [], []
            
            for col, raw_spec in table_schema.iter_specs():
                comp, value = self._normalize_spec(raw_spec)
                select_cols.append(col)

                if comp is None:
                    continue

                # collect tuple (col, comp, value) for later
                where_clauses.append((col, comp, value))

            select_string = ", ".join(select_cols) if select_cols else "*"
            query = database.table(table_schema.table).select(select_string)

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
                raise RuntimeError(f"Supabase query failed or returned no data \
                                   for table {table_schema.table} with conditions: {where_clauses}")
            
            df_tables.append(pd.DataFrame(data))

        # merge metadata from all tables, making sure to 
        # drop any rows where the merge key is not existent
        # in all tables
        merged_df = self._merge_tables(df_tables, on=["prepared_id", "tile_name"])

        # select data cubes for the selected tiles using the 
        # prepared_id and tile_name columns (forms unique key) 
        # TODO: here we should only select the subset of tiles
        #       in (merged_df["prepared_id"], merged_df["tile_name"])
        #       working on an efficient way to do this
        query = database.table(self.data_cubes_table).select(
            "prepared_id, tile_name, time, channel, " \
            "z_start, y_start, x_start"
        )

        response = query.execute()
        data = getattr(response, "data", None)

        if data is None:
            raise RuntimeError(f"Supabase query failed or returned no data for table {self.data_cubes_table}")
        
        data_cubes_df = pd.DataFrame(data)

        # if with labels we need two queries: first one gets
        # the data cubes columns, second one gets the label
        # cubes bbox and label_id columns and appends to 
        # dataframe from first query to create a single 
        # dataframe with all data cubes and labels
        if self.label_cubes_table is not None:
            # TODO: add a where clause to only select the tiles
            #       that are in the merged_df["prepared_id"] and
            #       merged_df["tile_name"] columns
            query = database.table(self.label_cubes_table).select(
                "prepared_id, tile_name, bbox, label_id, " \
                "time, z_start, y_start, x_start"
            )

            response = query.execute()
            data = getattr(response, "data", None)
            if data is None:
                raise RuntimeError(f"Supabase query failed or returned no data for table {self.label_cubes_table}")
            
            label_cubes_df = pd.DataFrame(data)

            # merge the label cubes with the data cubes
            metadata_table = pd.merge(
                data_cubes_df, 
                label_cubes_df, 
                on=["prepared_id", "tile_name", "time", "z_start", "y_start", "x_start"], 
                how="left"
            )
        
        # broadcast merged_df to all rows in the data cubes
        metadata_table = pd.merge(
            data_cubes_df, 
            merged_df, 
            on=["prepared_id", "tile_name"], 
            how="left"
        )

        # save the metadata table to disk for reuse for future training runs
        if self.db_savepath:
            if not self.db_savepath.parent.exists():
                self.db_savepath.parent.mkdir(parents=True, exist_ok=True)
            if self.db_save_method == "feather":
                metadata_table.to_feather(self.db_savepath)
            else:
                raise NotImplementedError(f"Save method {self.db_save_method} not implemented yet")

        self.metadata_table = metadata_table.reset_index(drop=True)    
    
    def _merge_tables(
        self,
        data_tables: List[pd.DataFrame],
        on: Sequence[Hashable] | Hashable,
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
        merged = reduce(
            lambda left, right: pd.merge(left, right, on=on, how="inner"),
            data_tables[1:],
            data_tables[0],
        )
        return merged.reset_index(drop=True)